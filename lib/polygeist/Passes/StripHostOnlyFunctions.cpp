//===- StripHostOnlyFunctions.cpp - Remove host-only functions from device code -===//
//
// This pass removes functions marked with polygeist.host_only_func attribute.
// It's used to clean up host-side code before lowering to device (Vortex) code.
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

/// Pass to remove functions marked with polygeist.host_only_func attribute
struct StripHostOnlyFunctionsPass
    : public PassWrapper<StripHostOnlyFunctionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripHostOnlyFunctionsPass)

  StringRef getArgument() const override { return "strip-host-only-functions"; }

  StringRef getDescription() const override {
    return "Remove functions marked with polygeist.host_only_func attribute";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect functions to erase (can't erase while iterating)
    SmallVector<Operation *, 16> toErase;

    // Walk all func.func operations
    module.walk([&](func::FuncOp funcOp) {
      // Check for polygeist.host_only_func attribute
      if (funcOp->hasAttr("polygeist.host_only_func")) {
        toErase.push_back(funcOp);
      }
    });

    // Also check LLVM::LLVMFuncOp in case functions were already lowered
    module.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp->hasAttr("polygeist.host_only_func")) {
        toErase.push_back(funcOp);
      }
    });

    // Erase collected functions
    for (auto *op : toErase) {
      // Drop all uses first to avoid dangling references
      // This handles cases where host functions call each other
      for (auto result : op->getResults()) {
        result.dropAllUses();
      }
      
      // For func.func with symbol uses, we need to drop symbol uses too
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        SymbolTable::symbolKnownUseEmpty(funcOp, module);
      }
      
      op->erase();
    }

    // Report statistics
    if (!toErase.empty()) {
      llvm::outs() << "strip-host-only-functions: removed " << toErase.size()
                   << " host-only function(s)\n";
    }
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createStripHostOnlyFunctionsPass() {
  return std::make_unique<StripHostOnlyFunctionsPass>();
}

} // namespace polygeist
} // namespace mlir
