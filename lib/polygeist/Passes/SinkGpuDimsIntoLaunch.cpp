//===- SinkGpuDimsIntoLaunch.cpp - Sink GPU dims into launch regions ------===//
//
// This pass sinks gpu.block_dim and gpu.grid_dim operations (and their
// dependent pure computations) into gpu.launch regions before kernel
// outlining. This eliminates synthetic kernel arguments for dimension values.
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace {

/// Check if an operation is pure and can be safely cloned into a launch body.
/// We only sink operations that:
/// 1. Have no side effects
/// 2. Are GPU dimension operations or arithmetic on such values
static bool isPureSinkable(Operation *op) {
  if (!op)
    return false;

  // GPU dimension operations are the primary targets
  if (isa<gpu::BlockDimOp, gpu::GridDimOp>(op))
    return true;

  // Constants can always be sunk
  if (isa<arith::ConstantOp>(op))
    return true;

  // Cast operations are pure
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp,
          arith::FPToSIOp, arith::FPToUIOp>(op))
    return true;

  // Binary arithmetic operations are pure
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
          arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
          arith::ShRUIOp>(op))
    return true;

  // Comparison operations are pure
  if (isa<arith::CmpIOp, arith::CmpFOp>(op))
    return true;

  // Select is pure
  if (isa<arith::SelectOp>(op))
    return true;

  return false;
}

/// Check if an operation's def-chain contains GPU dimension operations.
/// We only want to sink computations that are derived from GPU dimensions,
/// not arbitrary pure computations.
static bool dependsOnGpuDims(Operation *op,
                             llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (!op || visited.count(op))
    return false;
  visited.insert(op);

  // Direct GPU dimension operation
  if (isa<gpu::BlockDimOp, gpu::GridDimOp>(op))
    return true;

  // Check if any operand depends on GPU dims
  for (Value operand : op->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (dependsOnGpuDims(defOp, visited))
        return true;
    }
  }

  return false;
}

/// Collect all operations in the def-chain that need to be sunk.
/// Only collects operations that are pure AND depend on GPU dimensions.
static void collectDefChain(Value v, llvm::SetVector<Operation *> &toSink,
                            Region &launchBody,
                            llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return;

  // Already processed
  if (visited.count(defOp))
    return;

  // Already inside launch - nothing to sink
  if (launchBody.isAncestor(defOp->getParentRegion()))
    return;

  // Not sinkable - this value will become a kernel arg
  if (!isPureSinkable(defOp))
    return;

  // Recursively collect operand def chains first (topological order)
  for (Value operand : defOp->getOperands()) {
    collectDefChain(operand, toSink, launchBody, visited);
  }

  visited.insert(defOp);

  // Only add if it depends on GPU dimensions
  llvm::SmallPtrSet<Operation *, 16> dimVisited;
  if (dependsOnGpuDims(defOp, dimVisited)) {
    toSink.insert(defOp);
  }
}

struct SinkGpuDimsIntoLaunchPass
    : public PassWrapper<SinkGpuDimsIntoLaunchPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkGpuDimsIntoLaunchPass)

  StringRef getArgument() const override { return "sink-gpu-dims-into-launch"; }

  StringRef getDescription() const override {
    return "Sink GPU dimension operations into launch regions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool changed = true;
    int iterations = 0;
    const int maxIterations = 10;

    // Iterate until no more changes (with safety limit)
    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;

      op->walk([&](gpu::LaunchOp launchOp) {
        if (processLaunch(launchOp))
          changed = true;
      });
    }

    if (iterations > 1) {
      llvm::outs() << "[SinkGpuDims] Converged after " << iterations
                   << " iterations\n";
    }
  }

  bool processLaunch(gpu::LaunchOp launchOp) {
    Region &launchBody = launchOp.getBody();
    bool changed = false;

    // Collect values used inside launch but defined outside
    llvm::SetVector<Value> capturedValues;
    mlir::getUsedValuesDefinedAbove(launchBody, capturedValues);

    // For each external use, collect the full def chain that can be sunk
    llvm::SetVector<Operation *> toSink;
    llvm::SmallPtrSet<Operation *, 32> visited;

    for (Value v : capturedValues) {
      collectDefChain(v, toSink, launchBody, visited);
    }

    if (toSink.empty())
      return false;

    llvm::outs() << "[SinkGpuDims] Sinking " << toSink.size()
                 << " operations into gpu.launch\n";

    // Clone operations into launch body entry block
    Block &entryBlock = launchBody.front();
    OpBuilder builder(&entryBlock, entryBlock.begin());

    // Map from original values to cloned values
    IRMapping mapping;

    // Clone in topological order (SetVector preserves insertion order)
    for (Operation *op : toSink) {
      Operation *cloned = builder.clone(*op, mapping);
      mapping.map(op->getResults(), cloned->getResults());

      llvm::outs() << "[SinkGpuDims]   Sunk: " << op->getName() << "\n";
      changed = true;
    }

    // Replace uses of original values with cloned values inside launch
    for (const auto &pair : mapping.getValueMap()) {
      Value original = pair.first;
      Value cloned = pair.second;
      original.replaceUsesWithIf(cloned, [&](OpOperand &use) {
        return launchBody.isAncestor(use.getOwner()->getParentRegion());
      });
    }

    return changed;
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createSinkGpuDimsIntoLaunchPass() {
  return std::make_unique<SinkGpuDimsIntoLaunchPass>();
}

} // namespace polygeist
} // namespace mlir
