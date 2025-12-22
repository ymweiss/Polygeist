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

  // Count leading synthetic args (thread count indices):
  // - index: total thread count or loop bounds hoisted into kernel parameters
  unsigned numLeadingSynthetic = 0;
  for (unsigned i = 0; i < argTypes.size(); ++i) {
    Type argType = argTypes[i];
    if (argType.isa<IndexType>()) {
      numLeadingSynthetic++;
    } else {
      break;  // Stop at first non-index arg from the front
    }
  }

  // Count trailing synthetic args (not user args):
  // - !llvm.ptr: captured globals (e.g., printf format strings)
  // - index: loop bounds that Polygeist hoists into kernel parameters (already counted above)
  unsigned numTrailingSynthetic = 0;
  for (int i = argTypes.size() - 1; i >= (int)(numLeadingSynthetic); --i) {
    Type argType = argTypes[i];
    if (argType.isa<LLVM::LLVMPointerType>()) {
      numTrailingSynthetic++;
    } else {
      break;  // Stop at first non-synthetic arg from the end
    }
  }

  // User args are between leading and trailing synthetic args
  unsigned argsToSkip = numLeadingSynthetic;
  unsigned numUserArgs = argTypes.size() - numLeadingSynthetic - numTrailingSynthetic;

  if (numUserArgs != originalIsPointer.size()) {
    llvm::errs() << "Warning: Argument count mismatch for reordering "
                 << gpuFunc.getName() << " - expected " << originalIsPointer.size()
                 << " user args, got " << numUserArgs
                 << " (leading synthetic: " << numLeadingSynthetic
                 << ", trailing synthetic: " << numTrailingSynthetic << ")\n";
    return false;
  }

  // Check if reordering is actually needed by comparing actual GPU arg types
  // with the wrapper arg types in order.
  // We DON'T assume Polygeist reorders - we check the actual types.
  bool needsReorder = false;
  for (unsigned i = 0; i < numUserArgs; ++i) {
    Type gpuArgType = argTypes[argsToSkip + i];
    bool gpuIsPtr = gpuArgType.isa<MemRefType>();
    bool wrapperIsPtr = originalIsPointer[i];

    if (gpuIsPtr != wrapperIsPtr) {
      needsReorder = true;
      break;
    }
  }

  if (!needsReorder)
    return false;

  // Compute permutation: deviceToOriginal[device_idx] = original_idx
  // This is only used if we actually need to reorder.
  auto deviceToOriginal = computeArgPermutation(originalIsPointer);

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
  // Append trailing captured globals (llvm.ptr args) unchanged
  for (unsigned i = 0; i < numTrailingSynthetic; ++i) {
    newArgTypes.push_back(argTypes[argTypes.size() - numTrailingSynthetic + i]);
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
  SmallVector<BlockArgument> newCapturedArgs(numTrailingSynthetic);

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

  // For captured args, add them after user args (they stay in relative order)
  for (unsigned i = 0; i < numTrailingSynthetic; ++i) {
    unsigned oldIdx = argsToSkip + numUserArgs + i;
    Type argType = currentArgs[oldIdx].getType();
    auto newArg = clonedEntry.addArgument(argType, clonedFunc.getLoc());
    newCapturedArgs[i] = newArg;
  }

  // Step 3: RAUW - redirect uses of old args to new args
  // Old arg at device position devIdx should be replaced with new arg at original position
  // deviceToOriginal[devIdx] = origIdx means device pos devIdx came from original pos origIdx
  // So uses of oldArg[argsToSkip + devIdx] should use newUserArgs[origIdx]
  for (unsigned devIdx = 0; devIdx < numUserArgs; ++devIdx) {
    unsigned origIdx = deviceToOriginal[devIdx];
    currentArgs[argsToSkip + devIdx].replaceAllUsesWith(newUserArgs[origIdx]);
  }
  // Also redirect uses of old captured args to new captured args
  for (unsigned i = 0; i < numTrailingSynthetic; ++i) {
    unsigned oldIdx = argsToSkip + numUserArgs + i;
    currentArgs[oldIdx].replaceAllUsesWith(newCapturedArgs[i]);
  }

  // Step 4: Remove old user AND captured arguments (in reverse order to maintain indices)
  unsigned numArgsToRemove = numUserArgs + numTrailingSynthetic;
  for (int i = numArgsToRemove - 1; i >= 0; --i) {
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
    // Append captured args (trailing llvm.ptr) unchanged
    for (unsigned i = 0; i < numTrailingSynthetic; ++i) {
      newOperands.push_back(oldOperands[argsToSkip + numUserArgs + i]);
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

  // After reordering, the kernel args are in wrapper order.
  // Set the kernel_arg_mapping to identity: kernel arg i -> host arg i
  SmallVector<int64_t> identityMapping;
  for (unsigned i = 0; i < argTypes.size(); ++i) {
    if (i < argsToSkip || i >= argTypes.size() - numTrailingSynthetic) {
      // Synthetic args (leading index or trailing llvm.ptr)
      identityMapping.push_back(-1);
    } else {
      // User args now in identity order
      identityMapping.push_back(static_cast<int64_t>(i - argsToSkip));
    }
  }
  clonedFunc->setAttr("kernel_arg_mapping",
                       DenseI64ArrayAttr::get(clonedFunc.getContext(), identityMapping));

  llvm::outs() << "Reordered kernel arguments: " << funcNameStr << "\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_REORDERGPUKERNELARGS
#define GEN_PASS_DEF_REORDERGPUKERNELARGS
#include "polygeist/Passes/Passes.h.inc"

/// Check if a type is a dim3 struct (memref<?x3xi32>)
static bool isDim3Type(Type type) {
  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return false;
  // dim3 is lowered to memref<?x3xi32> or similar
  auto shape = memrefType.getShape();
  if (shape.size() != 2)
    return false;
  // Check for shape [?, 3] with i32 element type
  if (shape[1] != 3)
    return false;
  return memrefType.getElementType().isInteger(32);
}

/// Check if a type is a pointer type (memref or LLVM pointer)
static bool isPointerType(Type type) {
  return type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>();
}

/// Check if function name looks like a wrapper function
/// Wrappers are named like: __launch_<kernel_name>, _Z<digits>__launch_<kernel_name>...
static bool isWrapperFunctionName(StringRef name) {
  // Direct match: __launch_
  if (name.contains("__launch_"))
    return true;
  return false;
}

/// Extract kernel name from wrapper function name
/// E.g., "_Z22__launch_vecadd_kernelPfS_S_j4dim3S0_" -> "vecadd_kernel"
static std::string extractKernelNameFromWrapper(StringRef wrapperName) {
  // Find "__launch_" in the name
  size_t pos = wrapperName.find("__launch_");
  if (pos == StringRef::npos)
    return "";

  // Skip "__launch_" prefix
  StringRef afterLaunch = wrapperName.substr(pos + 9);

  // The kernel name continues until a type suffix in the mangled name.
  // Find the EARLIEST occurrence of any type suffix.
  // We focus on pointer types (P prefix) as they're unambiguous.
  size_t endPos = afterLaunch.size();

  // Helper to update endPos to the earliest position
  auto updateEndPos = [&](size_t pos) {
    if (pos != StringRef::npos && pos < endPos)
      endPos = pos;
  };

  // Check pointer types (P followed by type code) - these are unambiguous
  updateEndPos(afterLaunch.find("Pf"));  // pointer to float
  updateEndPos(afterLaunch.find("Pd"));  // pointer to double
  updateEndPos(afterLaunch.find("Pi"));  // pointer to int
  updateEndPos(afterLaunch.find("Pj"));  // pointer to uint32_t
  updateEndPos(afterLaunch.find("Pl"));  // pointer to long
  updateEndPos(afterLaunch.find("Pm"));  // pointer to unsigned long (uint64_t)
  updateEndPos(afterLaunch.find("Px"));  // pointer to long long
  updateEndPos(afterLaunch.find("Py"));  // pointer to unsigned long long
  updateEndPos(afterLaunch.find("Pc"));  // pointer to char
  updateEndPos(afterLaunch.find("Ph"));  // pointer to unsigned char
  updateEndPos(afterLaunch.find("Ps"));  // pointer to short
  updateEndPos(afterLaunch.find("Pt"));  // pointer to unsigned short
  updateEndPos(afterLaunch.find("Pv"));  // pointer to void
  updateEndPos(afterLaunch.find("PP")); // pointer to pointer
  updateEndPos(afterLaunch.find("PK")); // pointer to const
  // Check struct/class types (digit followed by name)
  updateEndPos(afterLaunch.find("4dim3"));  // dim3 type
  updateEndPos(afterLaunch.find("S_"));  // substitution
  updateEndPos(afterLaunch.find("S0_")); // substitution 0
  updateEndPos(afterLaunch.find("S1_")); // substitution 1

  return afterLaunch.substr(0, endPos).str();
}

struct ReorderGPUKernelArgsPass
    : public impl::ReorderGPUKernelArgsBase<ReorderGPUKernelArgsPass> {

  ReorderGPUKernelArgsPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Map: original kernel name -> is_pointer array for each arg
    llvm::StringMap<std::vector<bool>> kernelArgIsPointer;
    // Map: original kernel name -> wrapper function name (for debugging)
    llvm::StringMap<std::string> kernelToWrapper;

    llvm::outs() << "[ReorderGPUKernelArgs] Starting pass\n";

    // Find wrapper functions by name pattern (they may have empty bodies after inlining)
    // Wrapper functions have correct arg order in their signature
    module.walk([&](func::FuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      if (!isWrapperFunctionName(funcName))
        return;

      std::string kernelName = extractKernelNameFromWrapper(funcName);
      if (kernelName.empty()) {
        llvm::outs() << "[ReorderGPUKernelArgs] Could not extract kernel name from: "
                     << funcName << "\n";
        return;
      }

      llvm::outs() << "[ReorderGPUKernelArgs] Found wrapper " << funcName
                   << " for kernel " << kernelName << "\n";

      // Extract user args from wrapper signature
      // Wrapper args are: [user_arg0, user_arg1, ..., dim3_grid, dim3_block]
      auto argTypes = funcOp.getArgumentTypes();
      std::vector<bool> isPointer;

      // Find where user args end (where dim3 args begin)
      unsigned numUserArgs = 0;
      for (unsigned i = 0; i < argTypes.size(); ++i) {
        if (isDim3Type(argTypes[i])) {
          numUserArgs = i;
          break;
        }
        numUserArgs = i + 1;
      }

      llvm::outs() << "[ReorderGPUKernelArgs] Wrapper has " << numUserArgs
                   << " user args out of " << argTypes.size() << " total\n";

      // Build is_pointer array from wrapper's user args
      for (unsigned i = 0; i < numUserArgs; ++i) {
        bool isPtr = isPointerType(argTypes[i]) && !isDim3Type(argTypes[i]);
        isPointer.push_back(isPtr);
        llvm::outs() << "[ReorderGPUKernelArgs]   arg" << i << ": "
                     << (isPtr ? "pointer" : "scalar") << "\n";
      }

      if (!isPointer.empty()) {
        kernelArgIsPointer[kernelName] = isPointer;
        kernelToWrapper[kernelName] = funcName.str();
      }
    });

    if (kernelArgIsPointer.empty()) {
      llvm::outs() << "[ReorderGPUKernelArgs] No wrapper functions found\n";
      return;
    }

    // Reorder GPU kernel arguments
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      SmallVector<gpu::GPUFuncOp> kernelsToReorder;
      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>())
        if (gpuFunc.isKernel())
          kernelsToReorder.push_back(gpuFunc);

      for (auto gpuFunc : kernelsToReorder) {
        StringRef gpuFuncName = gpuFunc.getName();
        llvm::outs() << "[ReorderGPUKernelArgs] Checking gpu.func: "
                     << gpuFuncName << "\n";

        // Calculate number of user args in gpu.func
        // All args are user args except:
        // - Leading index: thread count or loop bounds hoisted into kernel parameters
        // - Trailing !llvm.ptr: captured globals (e.g., printf format strings)
        auto argTypes = gpuFunc.getArgumentTypes();

        // Count leading synthetic args (index types at the front)
        unsigned numLeadingSynthetic = 0;
        for (unsigned i = 0; i < argTypes.size(); ++i) {
          Type argType = argTypes[i];
          if (argType.isa<IndexType>()) {
            numLeadingSynthetic++;
          } else {
            break;  // Stop at first non-index arg from the front
          }
        }

        // Count trailing synthetic args (llvm.ptr at the end)
        unsigned numTrailingSynthetic = 0;
        for (int i = argTypes.size() - 1; i >= (int)numLeadingSynthetic; --i) {
          Type argType = argTypes[i];
          if (argType.isa<LLVM::LLVMPointerType>()) {
            numTrailingSynthetic++;
          } else {
            break;  // Stop at first non-ptr from the end
          }
        }

        // User args are between leading and trailing synthetic args
        unsigned argsToSkip = numLeadingSynthetic;
        unsigned numSyntheticArgs = numLeadingSynthetic + numTrailingSynthetic;
        unsigned numGpuUserArgs = argTypes.size() - numSyntheticArgs;

        llvm::outs() << "[ReorderGPUKernelArgs]   gpu.func has " << numGpuUserArgs
                     << " user args (total: " << argTypes.size()
                     << ", leading synthetic: " << numLeadingSynthetic
                     << ", trailing synthetic: " << numTrailingSynthetic << ")\n";

        // Try to match gpu.func name to wrapper's kernel name
        // 1. Try exact match first
        auto it = kernelArgIsPointer.find(gpuFuncName);

        // 2. If no exact match, try name-based matching
        if (it == kernelArgIsPointer.end()) {
          for (auto &entry : kernelArgIsPointer) {
            if (gpuFuncName.contains(entry.first()) ||
                entry.first().contains(gpuFuncName)) {
              llvm::outs() << "[ReorderGPUKernelArgs] Name match: " << gpuFuncName
                           << " <-> " << entry.first() << "\n";
              it = kernelArgIsPointer.find(entry.first());
              break;
            }
          }
        }

        // 3. If still no match, try matching by arg count
        // This handles the case where gpu.func is named "main_kernel" but
        // the wrapper has a different kernel name
        if (it == kernelArgIsPointer.end()) {
          for (auto &entry : kernelArgIsPointer) {
            if (entry.second.size() == numGpuUserArgs) {
              llvm::outs() << "[ReorderGPUKernelArgs] Arg count match: "
                           << gpuFuncName << " (" << numGpuUserArgs << " args) <-> "
                           << entry.first() << "\n";
              it = kernelArgIsPointer.find(entry.first());
              break;
            }
          }
        }

        // 4. If still no match and there's exactly one wrapper and one kernel, match them
        // This is a fallback for generic kernel names like "main_kernel"
        if (it == kernelArgIsPointer.end() &&
            kernelArgIsPointer.size() == 1 && kernelsToReorder.size() == 1) {
          it = kernelArgIsPointer.begin();
          llvm::outs() << "[ReorderGPUKernelArgs] Single-kernel fallback match: "
                       << gpuFuncName << " <-> " << it->first() << "\n";
        }

        if (it != kernelArgIsPointer.end()) {
          llvm::outs() << "[ReorderGPUKernelArgs] Found arg order for "
                       << gpuFuncName << " (wrapper kernel: " << it->first() << ")\n";

          // Set vortex.kernel_name attribute with the original kernel name
          // This is used by ConvertGPUToVortex to name the metadata files correctly
          std::string originalKernelName = it->first().str();
          gpuFunc->setAttr("vortex.kernel_name",
                          StringAttr::get(gpuFunc.getContext(), originalKernelName));
          llvm::outs() << "[ReorderGPUKernelArgs] Set vortex.kernel_name='"
                       << originalKernelName << "' on " << gpuFuncName << "\n";

          // Set vortex.num_synthetic_args to indicate how many trailing args
          // are synthetic (not user args). ConvertGPUToVortex uses this to
          // skip these args when generating the host stub.
          if (numSyntheticArgs > 0) {
            gpuFunc->setAttr("vortex.num_synthetic_args",
                            IntegerAttr::get(
                              IntegerType::get(gpuFunc.getContext(), 32),
                              numSyntheticArgs));
            llvm::outs() << "[ReorderGPUKernelArgs] Set vortex.num_synthetic_args="
                         << numSyntheticArgs << " on " << gpuFuncName << "\n";
          }

          // Update kernel_arg_mapping to reflect synthetic vs user args.
          // This is critical for GenerateVortexMain to correctly load user args.
          //
          // Polygeist reorders args: scalars first, then pointers.
          // It may also add synthetic scalar args (e.g., total_threads for bounds checking).
          // Synthetic scalars appear AFTER user scalars in the scalar block.
          //
          // Example for mstress:
          // - Wrapper args (original): addr_ptr(0), src_ptr(1), dst_ptr(2), stride(3)
          // - Wrapper isPointer: [true, true, true, false] => 1 scalar, 3 pointers
          // - GPU kernel args: (i32, i32, memref, memref, memref) => 2 scalars, 3 pointers
          // - Extra scalar = synthetic. User scalar comes first.
          // - Expected mapping: [3, -1, 0, 1, 2]

          const auto& wrapperIsPointer = it->second;

          // Count wrapper scalars and pointers
          unsigned numWrapperScalars = 0;
          unsigned numWrapperPointers = 0;
          for (bool isPtr : wrapperIsPointer) {
            if (isPtr) numWrapperPointers++;
            else numWrapperScalars++;
          }

          // Count GPU kernel scalar args at the front (before memrefs)
          unsigned numGpuScalars = 0;
          for (unsigned i = 0; i < argTypes.size(); ++i) {
            Type argType = argTypes[i];
            if (argType.isa<MemRefType>() || argType.isa<LLVM::LLVMPointerType>()) {
              break;  // Hit first pointer/memref type
            }
            numGpuScalars++;
          }

          // Synthetic scalars = GPU scalars - wrapper scalars
          unsigned numSyntheticScalars = 0;
          if (numGpuScalars > numWrapperScalars) {
            numSyntheticScalars = numGpuScalars - numWrapperScalars;
          }

          llvm::outs() << "[ReorderGPUKernelArgs]   wrapper: " << numWrapperScalars
                       << " scalars, " << numWrapperPointers << " pointers\n";
          llvm::outs() << "[ReorderGPUKernelArgs]   GPU: " << numGpuScalars
                       << " leading scalars, " << numSyntheticScalars << " synthetic\n";

          // Build the kernel_arg_mapping by checking the ACTUAL type at each position.
          // Don't assume Polygeist reordered args - it may or may not have.
          //
          // The wrapper has args in the original HIP order.
          // We need to map each GPU kernel arg to the corresponding wrapper arg index.
          //
          // Strategy:
          // - Count scalars and pointers encountered so far in the GPU kernel
          // - Map each GPU arg to the Nth occurrence in wrapper of that type

          SmallVector<int64_t> newMapping(argTypes.size(), -1);

          // Build lists of wrapper scalar and pointer indices (in wrapper order)
          SmallVector<unsigned> wrapperScalarIndices;
          SmallVector<unsigned> wrapperPointerIndices;
          for (unsigned i = 0; i < wrapperIsPointer.size(); ++i) {
            if (wrapperIsPointer[i]) {
              wrapperPointerIndices.push_back(i);
            } else {
              wrapperScalarIndices.push_back(i);
            }
          }

          unsigned scalarsSeen = 0;
          unsigned pointersSeen = 0;

          for (unsigned i = 0; i < argTypes.size(); ++i) {
            if (i >= argTypes.size() - numTrailingSynthetic) {
              // Trailing synthetic args (llvm.ptr) stay as -1
              newMapping[i] = -1;
            } else if (i < numLeadingSynthetic) {
              // Leading synthetic args (index types) stay as -1
              newMapping[i] = -1;
            } else {
              // User args - check actual type
              Type argType = argTypes[i];
              bool isPtr = argType.isa<MemRefType>();

              if (isPtr) {
                // Pointer arg - map to next available wrapper pointer
                if (pointersSeen < wrapperPointerIndices.size()) {
                  newMapping[i] = wrapperPointerIndices[pointersSeen++];
                } else {
                  newMapping[i] = -1;  // Extra pointer = synthetic
                }
              } else {
                // Scalar arg - map to next available wrapper scalar
                if (scalarsSeen < wrapperScalarIndices.size()) {
                  newMapping[i] = wrapperScalarIndices[scalarsSeen++];
                } else {
                  newMapping[i] = -1;  // Extra scalar = synthetic
                }
              }
            }
          }

          // Try to reorder the GPU kernel args to match wrapper order
          bool didReorder = reorderGPUKernelArguments(gpuModule, gpuFunc, it->second, module);

          // If reordering happened, the identity mapping was set inside reorderGPUKernelArguments
          // on the cloned function (gpuFunc is now invalid/erased).
          // If no reordering, set the computed mapping on the original gpuFunc.
          if (!didReorder) {
            gpuFunc->setAttr("kernel_arg_mapping",
                             DenseI64ArrayAttr::get(gpuFunc.getContext(), newMapping));
            llvm::outs() << "[ReorderGPUKernelArgs] Updated kernel_arg_mapping: [";
            for (size_t i = 0; i < newMapping.size(); ++i) {
              if (i > 0) llvm::outs() << ", ";
              llvm::outs() << newMapping[i];
            }
            llvm::outs() << "]\n";
          }
        } else {
          llvm::outs() << "[ReorderGPUKernelArgs] No matching wrapper for "
                       << gpuFuncName << " (has " << numGpuUserArgs << " user args)\n";
        }
      }
    });

    llvm::outs() << "[ReorderGPUKernelArgs] Pass complete\n";
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
