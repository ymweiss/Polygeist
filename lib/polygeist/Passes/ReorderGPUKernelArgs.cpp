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

/// Analyze gpu.launch_func to find which kernel args come from host wrapper args.
/// Returns a vector where userArgMapping[kernel_arg_idx] = host_user_arg_idx if it's a user arg,
/// or -1 if it's a synthetic/derived arg.
///
/// This works by tracing each kernel operand back to see if it's a BlockArgument
/// from the host wrapper function.
static std::vector<int> analyzeKernelArgsFromLaunch(
    gpu::LaunchFuncOp launchOp,
    func::FuncOp hostWrapper,
    unsigned numHostUserArgs) {

  auto kernelOperands = launchOp.getKernelOperands();
  std::vector<int> userArgMapping(kernelOperands.size(), -1);

  for (unsigned i = 0; i < kernelOperands.size(); ++i) {
    Value operand = kernelOperands[i];

    // Check if this operand is a BlockArgument from the host wrapper
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      if (blockArg.getOwner() == &hostWrapper.getBody().front()) {
        unsigned argIdx = blockArg.getArgNumber();
        // Only count if it's within user args (not blocks/threads params)
        if (argIdx < numHostUserArgs) {
          userArgMapping[i] = argIdx;
        }
      }
    }
  }

  return userArgMapping;
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
/// Uses launch_func analysis to correctly identify which args are user args vs synthetic.
/// Creates a new GPU function with user arguments in original order and updates call sites.
/// Returns true if reordering was performed.
static bool reorderGPUKernelArgumentsV2(
    gpu::GPUModuleOp gpuModule,
    gpu::GPUFuncOp gpuFunc,
    func::FuncOp hostWrapper,
    unsigned numHostUserArgs,
    gpu::LaunchFuncOp launchOp,
    ModuleOp module) {

  auto argTypes = gpuFunc.getArgumentTypes();
  unsigned numKernelArgs = argTypes.size();

  // Analyze launch_func to find which kernel args are user args
  // userArgMapping[kernel_idx] = host_user_arg_idx, or -1 if synthetic
  auto userArgMapping = analyzeKernelArgsFromLaunch(launchOp, hostWrapper, numHostUserArgs);

  // Count user args and build mapping
  std::vector<unsigned> userArgKernelIndices;  // kernel indices that are user args
  std::vector<int> kernelToUserIdx(numKernelArgs, -1);  // kernel_idx -> index in userArgKernelIndices

  for (unsigned i = 0; i < numKernelArgs; ++i) {
    if (userArgMapping[i] >= 0) {
      kernelToUserIdx[i] = userArgKernelIndices.size();
      userArgKernelIndices.push_back(i);
    }
  }

  unsigned numUserArgs = userArgKernelIndices.size();
  if (numUserArgs != numHostUserArgs) {
    llvm::errs() << "Warning: User arg count mismatch for " << gpuFunc.getName()
                 << " - expected " << numHostUserArgs << ", found " << numUserArgs << "\n";
    return false;
  }

  // Build deviceToOriginal: for user args in kernel order, what's their original host order?
  // deviceToOriginal[user_arg_index_in_kernel] = host_user_arg_index
  std::vector<unsigned> deviceToOriginal(numUserArgs);
  for (unsigned i = 0; i < numUserArgs; ++i) {
    unsigned kernelIdx = userArgKernelIndices[i];
    deviceToOriginal[i] = userArgMapping[kernelIdx];
  }

  // Check if reordering is needed
  bool needsReorder = false;
  for (unsigned i = 0; i < numUserArgs; ++i) {
    if (deviceToOriginal[i] != i) {
      needsReorder = true;
      break;
    }
  }

  if (!needsReorder) {
    llvm::outs() << "No reordering needed for: " << gpuFunc.getName() << "\n";
    return false;
  }

  // Compute inverse: originalToDevice[orig_idx] = device_user_idx
  std::vector<unsigned> originalToDevice(numUserArgs);
  for (unsigned devIdx = 0; devIdx < numUserArgs; ++devIdx) {
    originalToDevice[deviceToOriginal[devIdx]] = devIdx;
  }

  // Build new argument types: keep kernel structure but reorder user args in place
  // We want user args in their original host order while keeping synthetic args
  // in their relative positions.
  SmallVector<Type> newArgTypes(numKernelArgs);
  std::vector<int> oldIdxToNewIdx(numKernelArgs, -1);  // mapping for block arg reorder

  // First, place synthetic args at their original positions
  for (unsigned i = 0; i < numKernelArgs; ++i) {
    if (userArgMapping[i] < 0) {
      // Synthetic arg - keep in same position
      newArgTypes[i] = argTypes[i];
      oldIdxToNewIdx[i] = i;
    }
  }

  // Now place user args in their original host order, filling the remaining positions
  // User args will be placed contiguously where the first user arg was
  unsigned firstUserArgPos = userArgKernelIndices[0];
  for (unsigned origIdx = 0; origIdx < numUserArgs; ++origIdx) {
    unsigned devUserIdx = originalToDevice[origIdx];
    unsigned oldKernelIdx = userArgKernelIndices[devUserIdx];
    unsigned newKernelIdx = firstUserArgPos + origIdx;
    newArgTypes[newKernelIdx] = argTypes[oldKernelIdx];
    oldIdxToNewIdx[oldKernelIdx] = newKernelIdx;
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

  // Collect current block args
  SmallVector<BlockArgument> currentArgs;
  for (auto arg : clonedEntry.getArguments()) {
    currentArgs.push_back(arg);
  }

  // Add new arguments with the CORRECT types (from newArgTypes)
  SmallVector<BlockArgument> newArgs(numKernelArgs);
  for (unsigned newIdx = 0; newIdx < numKernelArgs; ++newIdx) {
    Type argType = newArgTypes[newIdx];
    auto newArg = clonedEntry.addArgument(argType, clonedFunc.getLoc());
    newArgs[newIdx] = newArg;
  }

  // RAUW old args with new args at their new positions
  // oldIdxToNewIdx[oldIdx] = newIdx means old arg at oldIdx moves to newIdx
  for (unsigned oldIdx = 0; oldIdx < numKernelArgs; ++oldIdx) {
    unsigned newIdx = oldIdxToNewIdx[oldIdx];
    currentArgs[oldIdx].replaceAllUsesWith(newArgs[newIdx]);
  }

  // Remove old arguments (in reverse order)
  for (int i = numKernelArgs - 1; i >= 0; --i) {
    clonedEntry.eraseArgument(i);
  }

  // Update function type
  clonedFunc.setFunctionType(FunctionType::get(gpuFunc.getContext(), newArgTypes, {}));

  // Update launch_func operands
  auto oldOperands = launchOp.getKernelOperands();
  SmallVector<Value> newOperands(numKernelArgs);

  // Build new operands matching new arg order
  for (unsigned oldIdx = 0; oldIdx < numKernelArgs; ++oldIdx) {
    newOperands[oldIdxToNewIdx[oldIdx]] = oldOperands[oldIdx];
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

  // Erase old function
  gpuFunc.erase();

  // Rename cloned function to original name
  clonedFunc.setName(funcNameStr);

  // Update all launch_func to use the final name
  module.walk([&](gpu::LaunchFuncOp lop) {
    auto callee = lop.getKernelAttr();
    if (callee.getLeafReference() == tempName) {
      auto newKernelAttr = SymbolRefAttr::get(
          lop.getContext(),
          callee.getRootReference(),
          {FlatSymbolRefAttr::get(lop.getContext(), funcNameStr)});
      lop.setKernelAttr(newKernelAttr);
    }
  });

  // Add attribute to mark user arg range for metadata generation
  // User args are now at positions [firstUserArgPos, firstUserArgPos + numUserArgs)
  SmallVector<int64_t> userArgRange = {
    static_cast<int64_t>(firstUserArgPos),
    static_cast<int64_t>(numUserArgs)
  };
  clonedFunc->setAttr("vortex.user_arg_range",
      DenseI64ArrayAttr::get(clonedFunc.getContext(), userArgRange));

  llvm::outs() << "Reordered kernel arguments (V2): " << funcNameStr << "\n";
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

    // Build map of host wrapper functions by name
    // Maps base name (e.g. "__polygeist_launch_basic_kernel") -> func::FuncOp
    llvm::StringMap<func::FuncOp> hostWrappers;
    llvm::StringMap<unsigned> hostWrapperUserArgCount;

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      StringRef funcName = funcOp.getName();
      if (!funcName.startswith("__polygeist_launch_"))
        continue;

      // Host wrapper args: user args... + blocks + threads (last 2 are launch params)
      auto hostArgTypes = funcOp.getArgumentTypes();
      unsigned numHostUserArgs = hostArgTypes.size() > 2 ? hostArgTypes.size() - 2 : 0;

      hostWrappers[funcName] = funcOp;
      hostWrapperUserArgCount[funcName] = numHostUserArgs;
    }

    // Find and process each kernel
    // For each kernel, we need to find:
    // 1. The gpu.func in the gpu.module
    // 2. The gpu.launch_func that calls it
    // 3. The host wrapper that contains the launch_func

    module.walk([&](gpu::GPUModuleOp gpuModule) {
      // Collect kernels and their launches
      struct KernelInfo {
        gpu::GPUFuncOp gpuFunc;
        gpu::LaunchFuncOp launchOp;
        func::FuncOp hostWrapper;
        unsigned numHostUserArgs;
      };
      SmallVector<KernelInfo> kernelsToProcess;

      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (!gpuFunc.isKernel())
          continue;

        std::string funcName = gpuFunc.getName().str();
        std::string baseName = extractBaseKernelName(funcName).str();

        // Find host wrapper
        auto wrapperIt = hostWrappers.find(baseName);
        if (wrapperIt == hostWrappers.end()) {
          llvm::errs() << "Warning: No host wrapper found for " << funcName << "\n";
          continue;
        }

        func::FuncOp hostWrapper = wrapperIt->second;
        unsigned numHostUserArgs = hostWrapperUserArgCount[baseName];

        // Find launch_func inside host wrapper that calls this kernel
        gpu::LaunchFuncOp foundLaunch = nullptr;
        hostWrapper.walk([&](gpu::LaunchFuncOp launchOp) {
          if (launchOp.getKernelName().getValue() == funcName) {
            foundLaunch = launchOp;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (!foundLaunch) {
          llvm::errs() << "Warning: No launch_func found for " << funcName << "\n";
          continue;
        }

        kernelsToProcess.push_back({gpuFunc, foundLaunch, hostWrapper, numHostUserArgs});
      }

      // Process each kernel
      for (auto &info : kernelsToProcess) {
        reorderGPUKernelArgumentsV2(
            gpuModule, info.gpuFunc, info.hostWrapper,
            info.numHostUserArgs, info.launchOp, module);
      }

      // After all kernels are processed, collect the user arg ranges from
      // the newly created kernel functions (the originals were erased)
      SmallVector<NamedAttribute> kernelArgRanges;
      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (auto rangeAttr = gpuFunc->getAttrOfType<DenseI64ArrayAttr>("vortex.user_arg_range")) {
          kernelArgRanges.push_back(
            NamedAttribute(
              StringAttr::get(module.getContext(), gpuFunc.getName()),
              rangeAttr
            )
          );
        }
      }

      // Add module-level attribute with all kernel arg ranges
      if (!kernelArgRanges.empty()) {
        // Get existing attribute or create new one
        auto existingAttr = module->getAttrOfType<DictionaryAttr>("vortex.kernel_arg_ranges");
        SmallVector<NamedAttribute> allRanges;
        if (existingAttr) {
          for (auto attr : existingAttr)
            allRanges.push_back(attr);
        }
        for (auto &attr : kernelArgRanges)
          allRanges.push_back(attr);
        module->setAttr("vortex.kernel_arg_ranges",
                        DictionaryAttr::get(module.getContext(), allRanges));
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
