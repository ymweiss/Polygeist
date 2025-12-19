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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

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

    // Iterate until no more changes
    while (changed) {
      changed = false;

      // Walk all gpu.launch operations
      op->walk([&](gpu::LaunchOp launchOp) {
        // Collect index_cast operations that can be sunk
        llvm::SmallVector<arith::IndexCastOp, 8> castsToSink;
        Region &launchBody = launchOp.getBody();

        // Find all arith.index_cast ops outside the launch whose results are used inside
        for (Block &block : launchBody) {
          for (Operation &innerOp : block) {
            for (Value operand : innerOp.getOperands()) {
              if (auto castOp = operand.getDefiningOp<arith::IndexCastOp>()) {
                // The cast is outside the launch
                if (!launchOp->isAncestor(castOp)) {
                  // Check if the source of the cast is also used in the launch
                  // (This would mean both the cast result and source are captured)
                  Value source = castOp.getIn();
                  bool sourceUsedInLaunch = false;
                  for (Operation *user : source.getUsers()) {
                    if (launchOp->isAncestor(user) || user == launchOp) {
                      sourceUsedInLaunch = true;
                      break;
                    }
                  }

                  // Also check if source is passed as a block argument
                  // (it would be captured for the launch)

                  // We sink the cast if it's only used in this launch
                  // and has no other uses outside
                  bool onlyUsedInThisLaunch = true;
                  for (Operation *user : castOp.getResult().getUsers()) {
                    if (!launchOp->isAncestor(user)) {
                      onlyUsedInThisLaunch = false;
                      break;
                    }
                  }

                  if (onlyUsedInThisLaunch) {
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
