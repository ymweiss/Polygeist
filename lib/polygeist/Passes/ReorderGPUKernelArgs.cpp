//===- ReorderGPUKernelArgs.cpp - Reorder GPU kernel arguments ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass reorders GPU kernel arguments to match the original HIP kernel
// signature. Polygeist reorders arguments (scalars before pointers), and this
// pass undoes that transformation for Vortex compatibility.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Extract base kernel name from Polygeist-generated kernel name.
/// E.g., "__polygeist_launch_vecadd_kernel_kernel94..." -> "__polygeist_launch_vecadd_kernel"
static StringRef extractBaseKernelName(StringRef fullName) {
  // Look for the pattern "_kernel<digits>" at the end and remove it
  // Need to find the LAST occurrence since the kernel name itself might contain "_kernel"

  size_t lastKernelPos = StringRef::npos;
  size_t searchPos = 0;

  while (true) {
    size_t pos = fullName.find("_kernel", searchPos);
    if (pos == StringRef::npos)
      break;

    // Check if this is followed by digits (Polygeist suffix)
    size_t afterKernel = pos + 7; // length of "_kernel"
    if (afterKernel < fullName.size() && std::isdigit(fullName[afterKernel])) {
      // Verify the rest are all digits
      bool allDigits = true;
      for (size_t i = afterKernel; i < fullName.size(); ++i) {
        if (!std::isdigit(fullName[i])) {
          allDigits = false;
          break;
        }
      }
      if (allDigits) {
        lastKernelPos = pos;
      }
    }
    searchPos = pos + 1;
  }

  if (lastKernelPos != StringRef::npos) {
    return fullName.substr(0, lastKernelPos);
  }

  return fullName;
}

/// Compute the permutation from device (Polygeist-reordered) to original order.
/// Returns deviceToOriginal[device_idx] = original_idx
static std::vector<unsigned> computeArgPermutation(
    const std::vector<bool> &originalIsPointer) {

  unsigned numUserArgs = originalIsPointer.size();
  std::vector<unsigned> deviceToOriginal(numUserArgs);

  // Polygeist puts scalars first, then pointers, preserving relative order
  unsigned numOriginalScalars = 0;
  for (bool isPtr : originalIsPointer) {
    if (!isPtr) numOriginalScalars++;
  }

  // Map scalars: they're at the front in device order
  unsigned currentScalarIdx = 0;
  for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
    if (!originalIsPointer[origIdx]) {
      deviceToOriginal[currentScalarIdx++] = origIdx;
    }
  }
  // Map pointers: they're after scalars in device order
  unsigned currentPtrIdx = 0;
  for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
    if (originalIsPointer[origIdx]) {
      deviceToOriginal[numOriginalScalars + currentPtrIdx++] = origIdx;
    }
  }

  return deviceToOriginal;
}

/// Reorder GPU kernel arguments to match the original host wrapper order.
/// Creates a new GPU function with arguments in original order and updates call sites.
/// Returns true if reordering was performed.
static bool reorderGPUKernelArguments(
    gpu::GPUModuleOp gpuModule,
    gpu::GPUFuncOp gpuFunc,
    const std::vector<bool> &originalIsPointer,
    ModuleOp module) {

  auto argTypes = gpuFunc.getArgumentTypes();

  // GPU kernel has internal args (index, i32) before user args
  unsigned numLeadingScalars = 0;
  for (auto argType : argTypes) {
    if (argType.isa<MemRefType>() || argType.isa<LLVM::LLVMPointerType>())
      break;
    ++numLeadingScalars;
  }

  // Skip first 2 internal args (derived from block_dim)
  unsigned argsToSkip = (numLeadingScalars >= 2) ? 2 : 0;
  unsigned numUserArgs = argTypes.size() - argsToSkip;

  if (numUserArgs != originalIsPointer.size()) {
    llvm::errs() << "Warning: Argument count mismatch for reordering "
                 << gpuFunc.getName() << " - expected " << originalIsPointer.size()
                 << " user args, got " << numUserArgs << "\n";
    return false;
  }

  // Compute permutation: deviceToOriginal[device_idx] = original_idx
  auto deviceToOriginal = computeArgPermutation(originalIsPointer);

  // Check if reordering is needed
  bool needsReorder = false;
  for (unsigned i = 0; i < numUserArgs; ++i) {
    if (deviceToOriginal[i] != i) {
      needsReorder = true;
      break;
    }
  }

  if (!needsReorder)
    return false;

  // Compute inverse: originalToDevice[original_idx] = device_idx
  std::vector<unsigned> originalToDevice(numUserArgs);
  for (unsigned devIdx = 0; devIdx < numUserArgs; ++devIdx) {
    originalToDevice[deviceToOriginal[devIdx]] = devIdx;
  }

  // Build new argument types in original order
  SmallVector<Type> newArgTypes;
  for (unsigned i = 0; i < argsToSkip; ++i) {
    newArgTypes.push_back(argTypes[i]);  // Keep internal args as-is
  }
  for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
    unsigned devIdx = originalToDevice[origIdx];
    newArgTypes.push_back(argTypes[argsToSkip + devIdx]);
  }

  // Save the function name before any modifications
  std::string funcNameStr = gpuFunc.getName().str();

  // Clone the entire function first
  OpBuilder builder(gpuFunc);
  IRMapping emptyMapping;
  auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*gpuFunc, emptyMapping));

  // Rename the clone temporarily
  std::string tempName = funcNameStr + "_reordered";
  clonedFunc.setName(tempName);

  // Get the cloned function's entry block
  Block &clonedEntry = clonedFunc.getBody().front();

  // The cloned function has arguments in device order.
  // We need to permute them to original order.
  //
  // Current: [internal0, internal1, dev_user0, dev_user1, dev_user2, dev_user3]
  // Desired: [internal0, internal1, orig_user0, orig_user1, orig_user2, orig_user3]
  //
  // where dev_user[i] has type for device position i
  // and orig_user[i] should have type for original position i
  //
  // The body uses block args. We need to:
  // 1. Create new block args in the right order
  // 2. RAUW old block args with corresponding new ones
  // 3. Erase old block args

  // Step 1: Collect current block args and their types
  SmallVector<BlockArgument> currentArgs;
  for (auto arg : clonedEntry.getArguments()) {
    currentArgs.push_back(arg);
  }

  // Step 2: For each position in the NEW signature, add a new block argument
  // We'll add them at the end first, then reorder
  SmallVector<BlockArgument> newInternalArgs;
  SmallVector<BlockArgument> newUserArgs(numUserArgs);

  // Add new arguments for internal args (these stay in same position)
  for (unsigned i = 0; i < argsToSkip; ++i) {
    // Internal args are unchanged, just use existing
    newInternalArgs.push_back(currentArgs[i]);
  }

  // For user args, we need to add them in the NEW order (original order)
  // and map old uses to new args
  for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
    unsigned devIdx = originalToDevice[origIdx];
    // The type at original position origIdx should be the type from device position devIdx
    Type argType = currentArgs[argsToSkip + devIdx].getType();
    // Add a new argument at the end
    auto newArg = clonedEntry.addArgument(argType, clonedFunc.getLoc());
    newUserArgs[origIdx] = newArg;
  }

  // Step 3: RAUW - redirect uses of old args to new args
  // Old arg at device position devIdx should be replaced with new arg at original position
  // deviceToOriginal[devIdx] = origIdx means device pos devIdx came from original pos origIdx
  // So uses of oldArg[argsToSkip + devIdx] should use newUserArgs[origIdx]
  for (unsigned devIdx = 0; devIdx < numUserArgs; ++devIdx) {
    unsigned origIdx = deviceToOriginal[devIdx];
    currentArgs[argsToSkip + devIdx].replaceAllUsesWith(newUserArgs[origIdx]);
  }

  // Step 4: Remove old user arguments (in reverse order to maintain indices)
  for (int i = numUserArgs - 1; i >= 0; --i) {
    clonedEntry.eraseArgument(argsToSkip + i);
  }

  // Step 5: Update function type to match new signature
  clonedFunc.setFunctionType(FunctionType::get(gpuFunc.getContext(), newArgTypes, {}));

  // Update all gpu.launch_func call sites to pass args in new order
  // Collect launches to modify (can't modify while walking)
  SmallVector<gpu::LaunchFuncOp> launchesToUpdate;
  module.walk([&](gpu::LaunchFuncOp launchOp) {
    auto callee = launchOp.getKernelAttr();
    if (callee.getLeafReference() == funcNameStr)
      launchesToUpdate.push_back(launchOp);
  });

  for (auto launchOp : launchesToUpdate) {
    auto oldOperands = launchOp.getKernelOperands();
    if (oldOperands.size() != argTypes.size()) {
      llvm::errs() << "Warning: Launch operand count mismatch\n";
      continue;
    }

    // Build new operands in original order
    SmallVector<Value> newOperands;
    for (unsigned i = 0; i < argsToSkip; ++i) {
      newOperands.push_back(oldOperands[i]);  // Keep internal args as-is
    }
    for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
      unsigned devIdx = originalToDevice[origIdx];
      newOperands.push_back(oldOperands[argsToSkip + devIdx]);
    }

    // Create new launch with temp function name
    OpBuilder launchBuilder(launchOp);
    auto newKernelAttr = SymbolRefAttr::get(
        launchOp.getContext(),
        launchOp.getKernelAttr().getRootReference(),
        {FlatSymbolRefAttr::get(launchOp.getContext(), tempName)});

    launchBuilder.create<gpu::LaunchFuncOp>(
        launchOp.getLoc(),
        newKernelAttr,
        launchOp.getGridSizeOperandValues(),
        launchOp.getBlockSizeOperandValues(),
        launchOp.getDynamicSharedMemorySize(),
        newOperands);

    launchOp.erase();
  }

  // Erase old function
  gpuFunc.erase();

  // Rename cloned function to original name
  clonedFunc.setName(funcNameStr);

  // Update all launch_func to use the final name
  module.walk([&](gpu::LaunchFuncOp launchOp) {
    auto callee = launchOp.getKernelAttr();
    if (callee.getLeafReference() == tempName) {
      auto newKernelAttr = SymbolRefAttr::get(
          launchOp.getContext(),
          callee.getRootReference(),
          {FlatSymbolRefAttr::get(launchOp.getContext(), funcNameStr)});
      launchOp.setKernelAttr(newKernelAttr);
    }
  });

  llvm::outs() << "Reordered kernel arguments: " << funcNameStr << "\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_REORDERGPUKERNELARGS
#define GEN_PASS_DEF_REORDERGPUKERNELARGS
#include "polygeist/Passes/Passes.h.inc"

struct ReorderGPUKernelArgsPass
    : public impl::ReorderGPUKernelArgsBase<ReorderGPUKernelArgsPass> {

  ReorderGPUKernelArgsPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build argument order map from host wrapper functions
    // This maps kernel base name -> list of (isPointer) for original args
    llvm::StringMap<std::vector<bool>> originalArgIsPointer;

    // Find host wrapper functions (func.func @__polygeist_launch_<name>)
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      StringRef funcName = funcOp.getName();
      if (!funcName.startswith("__polygeist_launch_"))
        continue;

      // Host wrapper args: user args... + blocks + threads (last 2 are launch params)
      auto hostArgTypes = funcOp.getArgumentTypes();
      unsigned numHostUserArgs = hostArgTypes.size() > 2 ? hostArgTypes.size() - 2 : 0;

      std::vector<bool> isPointerVec;
      for (unsigned i = 0; i < numHostUserArgs; ++i) {
        isPointerVec.push_back(hostArgTypes[i].isa<MemRefType>() ||
                               hostArgTypes[i].isa<LLVM::LLVMPointerType>());
      }
      originalArgIsPointer[funcName] = std::move(isPointerVec);
    }

    // Reorder GPU kernel arguments
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      // Collect kernels to reorder (can't modify while walking)
      SmallVector<gpu::GPUFuncOp> kernelsToReorder;
      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (gpuFunc.isKernel())
          kernelsToReorder.push_back(gpuFunc);
      }

      for (auto gpuFunc : kernelsToReorder) {
        // Find base kernel name to look up original arg info
        std::string baseName = extractBaseKernelName(gpuFunc.getName().str()).str();

        // Look up original argument order
        auto it = originalArgIsPointer.find(baseName);
        if (it != originalArgIsPointer.end()) {
          reorderGPUKernelArguments(gpuModule, gpuFunc, it->second, module);
        }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createReorderGPUKernelArgsPass() {
  return std::make_unique<ReorderGPUKernelArgsPass>();
}

} // namespace polygeist
} // namespace mlir
