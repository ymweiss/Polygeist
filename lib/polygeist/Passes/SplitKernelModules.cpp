//===- SplitKernelModules.cpp - Split multi-kernel modules ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass supports multi-kernel HIP programs by extracting individual kernels
// into separate modules. Each kernel gets its own .vxbin with its own main().
//
// The pass can operate in two modes:
// 1. Extract a single kernel: --split-kernel-modules="kernel=kernel_iadd"
// 2. List all kernels: --split-kernel-modules (outputs kernel names to stderr)
//
// Usage in compile script:
//   # First, list all kernels in the module
//   KERNELS=$(polygeist-opt input.mlir --split-kernel-modules 2>&1 | grep "^KERNEL:")
//
//   # Then extract each kernel into a separate file
//   for K in $KERNELS; do
//     polygeist-opt input.mlir --split-kernel-modules="kernel=$K" -o kernel_$K.mlir
//   done
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Find all kernel functions in the module
/// Kernels are identified by having the kernel_arg_mapping attribute
/// (set by ConvertGPUToVortex/ReorderGPUKernelArgs passes)
static SmallVector<LLVMFuncOp> findAllKernels(ModuleOp module) {
  SmallVector<LLVMFuncOp> kernels;

  module.walk([&](LLVMFuncOp func) {
    // Primary check: has kernel_arg_mapping attribute (most reliable)
    if (func->hasAttr("kernel_arg_mapping")) {
      kernels.push_back(func);
      return;
    }

    // Fallback check: name contains "kernel" but not "kernel_body"
    StringRef name = func.getName();
    if ((name.contains("_kernel") || name.startswith("kernel_")) &&
        !name.startswith("kernel_body")) {
      kernels.push_back(func);
    }
  });

  return kernels;
}

/// Check if a function is used by the given kernel (simple check)
/// This is a conservative check - we keep all non-kernel functions
static bool isUsedByKernel(LLVMFuncOp func, LLVMFuncOp kernel) {
  // For now, keep all helper functions - they might be needed
  // A more sophisticated analysis could track actual usage
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_SPLITKERNELMODULES
#define GEN_PASS_DEF_SPLITKERNELMODULES
#include "polygeist/Passes/Passes.h.inc"

struct SplitKernelModulesPass
    : public impl::SplitKernelModulesBase<SplitKernelModulesPass> {

  SplitKernelModulesPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find all kernel functions
    auto kernels = findAllKernels(module);

    if (kernels.empty()) {
      // No kernels found - nothing to do
      return;
    }

    // If kernel option is empty, list all kernels and return
    if (kernel.empty()) {
      llvm::errs() << "Found " << kernels.size() << " kernel(s):\n";
      for (auto &k : kernels) {
        llvm::errs() << "KERNEL:" << k.getName() << "\n";
      }
      return;
    }

    // Find the kernel to extract
    LLVMFuncOp targetKernel = nullptr;
    for (auto &k : kernels) {
      if (k.getName() == kernel) {
        targetKernel = k;
        break;
      }
    }

    if (!targetKernel) {
      emitError(module.getLoc()) << "Kernel not found: " << kernel;
      signalPassFailure();
      return;
    }

    // Remove all OTHER kernels from the module
    // Keep helper functions (they might be shared)
    for (auto &k : kernels) {
      if (k != targetKernel) {
        // Delete this kernel
        k.erase();
      }
    }

    // Also remove kernel_body wrappers for other kernels if they exist
    // (from previous GenerateVortexMain runs - shouldn't happen but be safe)
    SmallVector<LLVMFuncOp> toRemove;
    module.walk([&](LLVMFuncOp func) {
      StringRef name = func.getName();
      if (name.startswith("kernel_body") && name != "kernel_body") {
        // Check if it's for a different kernel
        if (!name.endswith(targetKernel.getName())) {
          toRemove.push_back(func);
        }
      }
    });
    for (auto &func : toRemove) {
      func.erase();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createSplitKernelModulesPass() {
  return std::make_unique<SplitKernelModulesPass>();
}

} // namespace polygeist
} // namespace mlir
