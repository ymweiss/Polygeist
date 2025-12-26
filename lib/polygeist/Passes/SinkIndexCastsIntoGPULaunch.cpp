//===- SinkIndexCastsIntoGPULaunch.cpp - Sink index casts into GPU launches =//
//
// This pass sinks arith.index_cast operations into gpu.launch regions to
// reduce the number of kernel parameters after outlining.
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

/// This pass sinks arith.index_cast operations into gpu.launch regions.
/// When a value is used both directly in a launch and via an index_cast,
/// both values become kernel parameters. By sinking the cast, we ensure
/// only the original value is captured.
struct SinkIndexCastsIntoGPULaunchPass
    : public PassWrapper<SinkIndexCastsIntoGPULaunchPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkIndexCastsIntoGPULaunchPass)

  StringRef getArgument() const override { return "sink-index-casts-into-gpu-launch"; }

  StringRef getDescription() const override {
    return "Sink arith.index_cast operations into gpu.launch regions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    bool changed = true;

    llvm::outs() << "[SinkIndexCasts] Starting pass\n";

    // Iterate until no more changes
    while (changed) {
      changed = false;

      // Walk all gpu.launch operations
      op->walk([&](gpu::LaunchOp launchOp) {
        llvm::outs() << "[SinkIndexCasts] Found gpu.launch\n";
        // Collect index_cast operations that can be sunk
        llvm::SmallVector<arith::IndexCastOp, 8> castsToSink;
        Region &launchBody = launchOp.getBody();

        // Debug: show what values would be captured
        llvm::SetVector<Value> capturedValues;
        mlir::getUsedValuesDefinedAbove(launchBody, capturedValues);
        llvm::outs() << "[SinkIndexCasts] Captured values: " << capturedValues.size() << "\n";
        for (Value v : capturedValues) {
          if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
            llvm::outs() << "[SinkIndexCasts]   - index_cast result, source type: "
                         << castOp.getIn().getType() << "\n";
            // This is a candidate for sinking!
            Value source = castOp.getIn();
            bool sourceAlsoCaptured = capturedValues.contains(source);
            llvm::outs() << "[SinkIndexCasts]     source also captured: "
                         << (sourceAlsoCaptured ? "yes" : "no") << "\n";
            if (sourceAlsoCaptured) {
              // Both the cast result AND its source are captured - we should sink the cast
              castsToSink.push_back(castOp);
            }
          } else if (v.getType().isIndex()) {
            llvm::outs() << "[SinkIndexCasts]   - index value (not from cast)\n";
          }
        }

        // Also check inside the launch body for casts that could be sunk
        for (Block &block : launchBody) {
          for (Operation &innerOp : block) {
            for (Value operand : innerOp.getOperands()) {
              if (auto castOp = operand.getDefiningOp<arith::IndexCastOp>()) {
                // The cast is outside the launch
                if (!launchOp->isAncestor(castOp)) {
                  // Check if the source of the cast is also used in the launch
                  Value source = castOp.getIn();
                  bool sourceUsedInLaunch = false;
                  for (Operation *user : source.getUsers()) {
                    if (launchOp->isAncestor(user) || user == launchOp) {
                      sourceUsedInLaunch = true;
                      break;
                    }
                  }

                  // We sink the cast if it's only used in this launch
                  // and has no other uses outside
                  bool onlyUsedInThisLaunch = true;
                  for (Operation *user : castOp.getResult().getUsers()) {
                    if (!launchOp->isAncestor(user)) {
                      onlyUsedInThisLaunch = false;
                      break;
                    }
                  }

                  llvm::outs() << "[SinkIndexCasts] Found cast candidate: "
                               << "sourceUsedInLaunch=" << sourceUsedInLaunch
                               << ", onlyUsedInThisLaunch=" << onlyUsedInThisLaunch << "\n";

                  if (onlyUsedInThisLaunch && !llvm::is_contained(castsToSink, castOp)) {
                    castsToSink.push_back(castOp);
                  }
                }
              }
            }
          }
        }

        // Sink each identified cast into the launch
        for (arith::IndexCastOp castOp : castsToSink) {
          // Create a clone of the cast at the beginning of the launch body
          Block &entryBlock = launchBody.front();
          OpBuilder builder(&entryBlock, entryBlock.begin());

          // Skip past the block arguments (grid/block sizes, etc.)
          // Find first non-argument operation
          Operation *insertPoint = nullptr;
          for (Operation &op : entryBlock) {
            insertPoint = &op;
            break;
          }

          if (insertPoint) {
            builder.setInsertionPoint(insertPoint);
          }

          // Create new cast operation inside the launch
          Value newCast = builder.create<arith::IndexCastOp>(
              castOp.getLoc(), castOp.getType(), castOp.getIn());

          // Replace uses inside the launch with the new cast
          castOp.getResult().replaceUsesWithIf(newCast, [&](OpOperand &use) {
            return launchOp->isAncestor(use.getOwner());
          });

          // If the original cast has no more uses, erase it
          if (castOp.getResult().use_empty()) {
            castOp.erase();
          }

          changed = true;
          llvm::outs() << "[SinkIndexCasts] Sunk index_cast into gpu.launch\n";
        }
      });
    }
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createSinkIndexCastsIntoGPULaunchPass() {
  return std::make_unique<SinkIndexCastsIntoGPULaunchPass>();
}

} // namespace polygeist
} // namespace mlir
