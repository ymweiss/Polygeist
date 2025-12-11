//===- InsertVortexDivergence.cpp - Insert vx_split/vx_join for Vortex ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts Vortex divergence handling intrinsics (vx_split/vx_join)
// around divergent branches in kernel code. Vortex requires explicit split/join
// calls to handle SIMT divergence, unlike NVIDIA GPUs which handle it in hardware.
//
// The pass:
// 1. Performs divergence analysis to find values dependent on thread ID
// 2. Identifies conditional branches with divergent conditions
// 3. Inserts vx_split_abi() before divergent branches
// 4. Inserts vx_join_abi() at convergence points (post-dominators)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Divergence Analysis
//===----------------------------------------------------------------------===//

/// Simple divergence analysis that tracks which values depend on thread ID.
/// A value is divergent if:
/// - It is the result of a call to vx_get_threadIdx (or blockIdx in certain cases)
/// - It is computed from a divergent value
class DivergenceAnalysis {
public:
  DivergenceAnalysis(func::FuncOp funcOp) {
    // Initialize: find sources of divergence (threadIdx accesses)
    funcOp.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCalleeAttr();
      if (!callee)
        return;

      StringRef calleeName = callee.getValue();
      // vx_get_threadIdx returns divergent values (different per thread)
      // vx_get_blockIdx is uniform within a block but divergent across blocks
      if (calleeName.startswith("vx_get_threadIdx")) {
        for (auto result : callOp.getResults()) {
          markDivergent(result);
        }
      }
    });

    // Propagate divergence through the dataflow graph
    propagateDivergence(funcOp);
  }

  bool isDivergent(Value value) const {
    return divergentValues.contains(value);
  }

private:
  void markDivergent(Value value) {
    if (divergentValues.insert(value).second) {
      worklist.push_back(value);
    }
  }

  void propagateDivergence(func::FuncOp funcOp) {
    // Fixed-point iteration: propagate divergence to users
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();

      for (Operation *user : value.getUsers()) {
        // All results of an operation with a divergent operand become divergent
        for (auto result : user->getResults()) {
          markDivergent(result);
        }

        // For branch operations, the branch itself is divergent but we handle
        // that separately in the main pass
      }
    }
  }

  llvm::DenseSet<Value> divergentValues;
  llvm::SmallVector<Value> worklist;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_INSERTVORTEXDIVERGENCE
#define GEN_PASS_DEF_INSERTVORTEXDIVERGENCE
#include "polygeist/Passes/Passes.h.inc"

struct InsertVortexDivergencePass
    : public impl::InsertVortexDivergenceBase<InsertVortexDivergencePass> {

  InsertVortexDivergencePass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each function in the module
    module.walk([&](func::FuncOp funcOp) {
      processFunction(funcOp);
    });
  }

private:
  /// Declare vx_split_abi function if not already present
  LLVM::LLVMFuncOp getOrCreateSplitFunc(ModuleOp module) {
    MLIRContext *context = module.getContext();
    const char *funcName = "vx_split_abi";

    if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
      return existing;
    }

    OpBuilder builder(module.getBody()->getTerminator());
    auto i32Type = IntegerType::get(context, 32);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {i32Type}, false);

    return builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), funcName, funcType, LLVM::Linkage::External);
  }

  /// Declare vx_join_abi function if not already present
  LLVM::LLVMFuncOp getOrCreateJoinFunc(ModuleOp module) {
    MLIRContext *context = module.getContext();
    const char *funcName = "vx_join_abi";

    if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
      return existing;
    }

    OpBuilder builder(module.getBody()->getTerminator());
    auto i32Type = IntegerType::get(context, 32);
    auto voidType = LLVM::LLVMVoidType::get(context);
    auto funcType = LLVM::LLVMFunctionType::get(voidType, {i32Type}, false);

    return builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), funcName, funcType, LLVM::Linkage::External);
  }

  void processFunction(func::FuncOp funcOp) {
    if (funcOp.isExternal())
      return;

    // Run divergence analysis
    DivergenceAnalysis divergence(funcOp);

    // Compute post-dominance for finding convergence points
    PostDominanceInfo postDomInfo(funcOp);

    // Find divergent conditional branches
    SmallVector<cf::CondBranchOp> divergentBranches;
    funcOp.walk([&](cf::CondBranchOp brOp) {
      if (divergence.isDivergent(brOp.getCondition())) {
        divergentBranches.push_back(brOp);
      }
    });

    if (divergentBranches.empty())
      return;

    // Get module and declare split/join functions
    auto module = funcOp->getParentOfType<ModuleOp>();
    auto splitFunc = getOrCreateSplitFunc(module);
    auto joinFunc = getOrCreateJoinFunc(module);

    llvm::outs() << "Found " << divergentBranches.size()
                 << " divergent branches in " << funcOp.getName() << "\n";

    // Process each divergent branch
    for (auto brOp : divergentBranches) {
      insertSplitJoin(brOp, postDomInfo, splitFunc, joinFunc);
    }
  }

  void insertSplitJoin(cf::CondBranchOp brOp,
                       PostDominanceInfo &postDomInfo,
                       LLVM::LLVMFuncOp splitFunc,
                       LLVM::LLVMFuncOp joinFunc) {
    Location loc = brOp.getLoc();
    Block *branchBlock = brOp->getBlock();
    MLIRContext *context = brOp.getContext();

    // Find the convergence point (immediate post-dominator)
    // This is where all paths from the branch reconverge
    Block *convergenceBlock = findConvergenceBlock(brOp, postDomInfo);

    if (!convergenceBlock) {
      llvm::errs() << "Warning: Could not find convergence point for divergent "
                      "branch at "
                   << loc << "\n";
      return;
    }

    // Insert vx_split_abi(condition) before the branch
    OpBuilder builder(brOp);
    auto i32Type = IntegerType::get(context, 32);

    // Convert condition to i32 if needed (condition is typically i1)
    Value condition = brOp.getCondition();
    if (condition.getType() != i32Type) {
      condition = builder.create<LLVM::ZExtOp>(loc, i32Type, condition);
    }

    // Call vx_split_abi(condition) -> returns stack pointer
    auto splitCall = builder.create<LLVM::CallOp>(
        loc, splitFunc, ValueRange{condition});
    Value stackPtr = splitCall.getResult();

    // Insert vx_join_abi(stack_ptr) at the beginning of the convergence block
    // We need to insert at the very start, before any phi operations
    builder.setInsertionPointToStart(convergenceBlock);

    // Skip any block arguments (which act like phi nodes in MLIR)
    // Insert the join call at the first non-argument position
    builder.create<LLVM::CallOp>(loc, joinFunc, ValueRange{stackPtr});

    llvm::outs() << "  Inserted split/join for branch at " << loc << "\n";
  }

  /// Find the convergence block for a conditional branch.
  /// This is the immediate post-dominator - the first block that all paths
  /// from the branch must pass through.
  Block *findConvergenceBlock(cf::CondBranchOp brOp,
                              PostDominanceInfo &postDomInfo) {
    Block *branchBlock = brOp->getBlock();
    Block *trueBlock = brOp.getTrueDest();
    Block *falseBlock = brOp.getFalseDest();

    // If true and false destinations are the same, that's our convergence
    if (trueBlock == falseBlock) {
      return trueBlock;
    }

    // Find the nearest common post-dominator of true and false branches
    // This is where control flow reconverges
    return postDomInfo.findNearestCommonDominator(trueBlock, falseBlock);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createInsertVortexDivergencePass() {
  return std::make_unique<InsertVortexDivergencePass>();
}

} // namespace polygeist
} // namespace mlir
