//===- GenerateVortexMain.cpp - Generate Vortex main() wrapper ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates the Vortex-specific main() entry point and kernel_body
// wrapper function. It should run AFTER gpu-to-llvm lowering has converted
// gpu.func to llvm.func.
//
// The Vortex execution model requires:
// 1. A main() function that reads args from VX_CSR_MSCRATCH and calls
//    vx_spawn_threads() with the kernel callback
// 2. A kernel_body() wrapper that unpacks arguments and calls the kernel
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <limits>

using namespace mlir;
using namespace mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// Vortex CSR Address
//===----------------------------------------------------------------------===//

// VX_CSR_MSCRATCH - Machine scratch register used to pass kernel arguments
// From vortex/hw/rtl/VX_types.vh: `define VX_CSR_MSCRATCH 12'h340
constexpr uint32_t VX_CSR_MSCRATCH = 0x340;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Count user args (non -1 values) in kernel_arg_mapping
/// User args have mapping value >= 0 indicating which wrapper arg they map to
/// A kernel with more user args is generally better for arg unpacking
static unsigned countUserArgs(LLVM::LLVMFuncOp func) {
  auto mappingAttr = func->getAttrOfType<DenseI64ArrayAttr>("kernel_arg_mapping");
  if (!mappingAttr)
    return 0;

  unsigned count = 0;
  for (int64_t idx : mappingAttr.asArrayRef()) {
    if (idx >= 0)
      count++;
  }
  return count;
}

/// Find the kernel function in the module (lowered from gpu.func)
/// After gpu-to-llvm, the kernel is an llvm.func with a mangled name
///
/// Preference order:
/// 1. main_kernel IF it has valid user args (mapping not all -1)
/// 2. Kernel with most user args (most non -1 values in kernel_arg_mapping)
/// 3. First kernel found as fallback
///
/// The main_kernel is preferred when it has valid args because Polygeist inlines
/// constants (blockDim.x, gridDim.x, etc.) and ConvertGPUToVortex lowers GPU
/// intrinsics to accessor function calls (vx_get_threadIdx, vx_get_blockIdx).
/// However, if main_kernel has all synthetic args (e.g., due to captured globals
/// like printf format strings), prefer the wrapper kernel instead.
static LLVM::LLVMFuncOp findKernelFunction(ModuleOp module) {
  LLVM::LLVMFuncOp kernelFunc = nullptr;
  LLVM::LLVMFuncOp mainKernel = nullptr;
  unsigned bestUserArgCount = 0;

  module.walk([&](LLVM::LLVMFuncOp func) {
    StringRef name = func.getName();

    // Skip kernel_body wrappers (these are what we generate, not inputs)
    if (name.startswith("kernel_body"))
      return;

    // Primary check: has kernel_arg_mapping attribute (most reliable)
    // This attribute is set by ConvertGPUToVortex/ReorderGPUKernelArgs passes
    bool isKernel = func->hasAttr("kernel_arg_mapping");

    // Fallback check: name contains "kernel" with optional leading underscore
    // Matches: "_kernel", "kernel_", "foo_kernel", "kernel_iadd", etc.
    if (!isKernel) {
      isKernel = name.contains("_kernel") || name.contains("kernel_");
    }

    if (isKernel) {
      // Check for main_kernel
      if (name == "main_kernel") {
        mainKernel = func;
        return;
      }

      // Track kernel with most user args (best for arg unpacking)
      unsigned userArgCount = countUserArgs(func);
      if (!kernelFunc || userArgCount > bestUserArgCount) {
        kernelFunc = func;
        bestUserArgCount = userArgCount;
      }
    }
  });

  // Prefer wrapper kernel as it has dimension=3 which works with multi-block launches.
  // main_kernel has dimension=1 which causes simulator exceptions with multi-block.
  // The wrapper kernel needs synthetic args to be fixed, but at least runs.
  if (kernelFunc) {
    return kernelFunc;
  }
  return mainKernel;
}

/// Declare vx_spawn_threads external function
/// Signature: int vx_spawn_threads(uint32_t dimension, const uint32_t* grid_dim,
///                                  const uint32_t* block_dim,
///                                  vx_kernel_func_cb kernel_func, const void* arg)
static LLVM::LLVMFuncOp
getOrDeclareVxSpawnThreads(ModuleOp module, OpBuilder &builder) {
  MLIRContext *ctx = module.getContext();

  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>("vx_spawn_threads"))
    return existing;

  auto i32Type = IntegerType::get(ctx, 32);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);

  // int vx_spawn_threads(uint32_t, uint32_t*, uint32_t*, void(*)(void*), void*)
  auto funcType = LLVM::LLVMFunctionType::get(
      i32Type, {i32Type, ptrType, ptrType, ptrType, ptrType},
      /*isVarArg=*/false);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "vx_spawn_threads",
                                          funcType, LLVM::Linkage::External);
}

/// Check if a sequence of parameter types represents a memref descriptor
/// LLVM memref descriptor format: { ptr, ptr, i64, [1 x i64], [1 x i64] }
/// After flattening, this becomes: ptr, ptr, i64, i64, i64 (for 1D memref)
static bool isMemrefDescriptorStart(LLVM::LLVMFunctionType funcType,
                                     unsigned startIdx) {
  unsigned numParams = funcType.getNumParams();
  // Need at least 5 params remaining for a 1D memref descriptor
  if (startIdx + 5 > numParams)
    return false;

  Type t0 = funcType.getParamType(startIdx);
  Type t1 = funcType.getParamType(startIdx + 1);
  Type t2 = funcType.getParamType(startIdx + 2);
  Type t3 = funcType.getParamType(startIdx + 3);
  Type t4 = funcType.getParamType(startIdx + 4);

  // Check pattern: ptr, ptr, i64, i64, i64
  return t0.isa<LLVM::LLVMPointerType>() &&
         t1.isa<LLVM::LLVMPointerType>() && t2.isInteger(64) &&
         t3.isInteger(64) && t4.isInteger(64);
}

/// Generate kernel_body wrapper function
/// This function unpacks arguments from the void* args pointer and calls
/// the original kernel function
///
/// Uses kernel_arg_mapping attribute to determine which args are user args
/// vs launch config args (from block_dim). The mapping format:
///   - Each entry corresponds to an original kernel arg (before LLVM expansion)
///   - Value >= 0: index into host args (0=first user arg, 1=second, etc.)
///   - Value = -1: synthetic arg (computed at launch time, e.g., comparisons)
///
/// Args with mapping value >= num_user_args are launch config params
/// (typically block_dim.x as i32 and index types).
///
/// This handles the memref descriptor expansion that happens during
/// LLVM lowering. Each memref<?xf32> in the original kernel becomes 5 params
/// (ptr, ptr, i64, i64, i64) after lowering. The host passes simple device
/// pointers, so we must construct the full descriptor from each pointer.
static LLVM::LLVMFuncOp
generateKernelBodyWrapper(ModuleOp module, LLVM::LLVMFuncOp kernelFunc,
                          OpBuilder &builder) {
  MLIRContext *ctx = module.getContext();
  Location loc = kernelFunc.getLoc();

  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  auto i32Type = IntegerType::get(ctx, 32);
  auto i64Type = IntegerType::get(ctx, 64);

  // Create function: void kernel_body(void* args)
  auto funcType =
      LLVM::LLVMFunctionType::get(voidType, {ptrType}, /*isVarArg=*/false);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(kernelFunc);

  auto bodyFunc = builder.create<LLVM::LLVMFuncOp>(loc, "kernel_body", funcType,
                                                   LLVM::Linkage::External);

  // Create entry block
  Block *entryBlock = bodyFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Value argsPtr = bodyFunc.getArgument(0);

  // Get kernel argument types
  auto kernelFuncType = kernelFunc.getFunctionType();
  unsigned numParams = kernelFuncType.getNumParams();

  // Standard Vortex args layout (matches runtime/hip_vortex_runtime.h):
  //   uint32_t grid_dim[3];   // 12 bytes (offsets 0, 4, 8)
  //   uint32_t block_dim[3];  // 12 bytes (offsets 12, 16, 20)
  //   <user args>             // starting at offset 24
  constexpr unsigned BLOCK_DIM_OFFSET = 12;
  constexpr unsigned USER_ARGS_OFFSET = 24;

  SmallVector<Value> unpackedArgs;
  unsigned currentOffset = USER_ARGS_OFFSET;
  auto i8Type = IntegerType::get(ctx, 8);

  // Parse kernel_arg_mapping attribute to understand which args are user args
  // vs launch config (from block_dim)
  SmallVector<int64_t> argMapping;
  unsigned numUserArgs = 0;

  if (auto mappingAttr = kernelFunc->getAttrOfType<DenseI64ArrayAttr>("kernel_arg_mapping")) {
    argMapping = SmallVector<int64_t>(mappingAttr.asArrayRef());
    // Determine numUserArgs by finding the boundary between user args and launch config
    // User args have consecutive indices 0,1,2,3...
    // Launch config args have indices that jump (e.g., index 5 when only 4 user args)
    // We find the threshold by looking for gaps or the first repeated index
    llvm::SmallSet<int64_t, 8> seenIndices;
    int64_t maxIdx = -1;
    for (int64_t idx : argMapping) {
      if (idx >= 0) {
        seenIndices.insert(idx);
        if (idx > maxIdx) maxIdx = idx;
      }
    }

    // If all mappings are -1 (maxIdx stays -1), tracing completely failed.
    // This happens when kernel is launched from main() with local variables.
    // In this case, treat ALL kernel args as user args by clearing the mapping.
    if (maxIdx < 0) {
      argMapping.clear();  // No mapping info - all args will be loaded from args buffer
      // numUserArgs stays 0, but argMapping being empty means no arg will be marked synthetic
    } else {
      // numUserArgs is the count of unique indices from 0 to maxIdx that exist
      // If indices are 0,1,2,3,5,5 then unique set is {0,1,2,3,5}, so numUserArgs = 4 (not 6)
      // because index 4 is missing, indicating indices >= 4 are launch config
      numUserArgs = 0;
      for (int64_t i = 0; i <= maxIdx; ++i) {
        if (seenIndices.contains(i)) {
          numUserArgs++;
        } else {
          // Gap found - everything before this is user args
          break;
        }
      }
    }
  }

  // Pre-load block_dim[0] for launch config args (user args that came from blockDim.x)
  // Note: After constant sinking, synthetic args should be rare. This is only for
  // kernel args that explicitly reference blockDim.x in the kernel signature.
  Value blockDimX_i32 = nullptr;
  Value blockDimX_i64 = nullptr;
  {
    SmallVector<LLVM::GEPArg> blockGepIndices;
    blockGepIndices.push_back(static_cast<int32_t>(BLOCK_DIM_OFFSET));
    auto blockDimPtr =
        builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, blockGepIndices);
    blockDimX_i32 = builder.create<LLVM::LoadOp>(loc, i32Type, blockDimPtr);
    blockDimX_i64 = builder.create<LLVM::ZExtOp>(loc, i64Type, blockDimX_i32);
  }

  // Track which original args we've processed (for mapping lookup)
  unsigned origArgIdx = 0;

  for (unsigned i = 0; i < numParams; ) {
    Type argType = kernelFuncType.getParamType(i);

    // Check if this is the start of a memref descriptor (5 consecutive params)
    if (isMemrefDescriptorStart(kernelFuncType, i)) {
      // Get the mapping for this memref arg
      int64_t hostIdx = -1;
      if (origArgIdx < argMapping.size()) {
        hostIdx = argMapping[origArgIdx];
      }

      // Compute offset from mapping
      unsigned argOffset;
      if (hostIdx >= 0) {
        argOffset = USER_ARGS_OFFSET + static_cast<unsigned>(hostIdx) * 4;
      } else {
        argOffset = currentOffset;  // Fallback
      }

      // This is a memref descriptor - load single pointer from args and expand
      SmallVector<LLVM::GEPArg> gepIndices;
      gepIndices.push_back(static_cast<int32_t>(argOffset));
      auto argBytePtr = builder.create<LLVM::GEPOp>(loc, ptrType, i8Type,
                                                     argsPtr, gepIndices);

      // Load the device pointer (4 bytes on RV32)
      auto rawPtr = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
      auto devicePtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, rawPtr);

      // Construct memref descriptor values:
      // param 0: allocated pointer (same as device ptr)
      // param 1: aligned pointer (same as device ptr)
      // param 2: offset (0)
      // param 3: size (use large value, kernel will bounds check)
      // param 4: stride (1 for contiguous)
      unpackedArgs.push_back(devicePtr);                    // allocated ptr
      unpackedArgs.push_back(devicePtr);                    // aligned ptr

      auto zeroI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 0);
      auto maxI64 = builder.create<LLVM::ConstantOp>(
          loc, i64Type, std::numeric_limits<int64_t>::max());
      auto oneI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 1);

      unpackedArgs.push_back(zeroI64);  // offset = 0
      unpackedArgs.push_back(maxI64);   // size = MAX (kernel has bounds check)
      unpackedArgs.push_back(oneI64);   // stride = 1

      currentOffset += 4; // Single pointer in args buffer
      i += 5;             // Skip 5 params (the whole memref descriptor)
      origArgIdx++;       // Advance original arg index
      continue;
    }

    // Scalar argument handling
    Value argVal;

    // Check if this is a launch config arg (from block_dim) or synthetic arg
    bool isLaunchConfigArg = false;
    bool isSyntheticArg = false;
    int64_t hostIdx = -1;

    if (origArgIdx < argMapping.size()) {
      hostIdx = argMapping[origArgIdx];
      if (hostIdx == -1) {
        isSyntheticArg = true;
      } else if (static_cast<unsigned>(hostIdx) >= numUserArgs) {
        isLaunchConfigArg = true;
      }
    }

    if (isSyntheticArg) {
      // Synthetic args (mapping = -1) should be rare after constant sinking.
      // The createGpuLauchSinkIndexComputationsPass embeds constants (loop bounds,
      // initial values) directly into the kernel body before outlining.
      //
      // If we still have synthetic args, emit a warning and use simple defaults.
      // Complex heuristics were removed - fix at source level instead.
      llvm::errs() << "Warning: Synthetic arg at kernel param index " << i
                   << " (type: ";
      argType.print(llvm::errs());
      llvm::errs() << "). Constants should be sunk at source level.\n";

      if (argType.isInteger(1)) {
        argVal = builder.create<LLVM::ConstantOp>(loc, argType, 1);
      } else if (argType.isInteger(32)) {
        argVal = builder.create<LLVM::ConstantOp>(loc, i32Type, 0);
      } else if (argType.isInteger(64)) {
        argVal = builder.create<LLVM::ConstantOp>(loc, i64Type, 0);
      } else if (argType.isa<LLVM::LLVMPointerType>()) {
        argVal = builder.create<LLVM::ZeroOp>(loc, ptrType);
      } else if (argType.isF32()) {
        argVal = builder.create<LLVM::ConstantOp>(loc, argType,
            builder.getFloatAttr(builder.getF32Type(), 0.0));
      } else if (argType.isF64()) {
        argVal = builder.create<LLVM::ConstantOp>(loc, argType,
            builder.getFloatAttr(builder.getF64Type(), 0.0));
      } else {
        argVal = builder.create<LLVM::ConstantOp>(loc, i32Type, 0);
      }
    } else if (isLaunchConfigArg) {
      // Launch config arg - load from block_dim
      if (argType.isInteger(64)) {
        argVal = blockDimX_i64;
      } else {
        argVal = blockDimX_i32;
      }
    } else {
      // Regular user argument from user args buffer
      // Use hostIdx from kernel_arg_mapping to compute offset
      // In RV32, all args are 4 bytes (pointers are 4 bytes, i32/f32 are 4 bytes)
      // Exception: i64/f64 are 8 bytes
      //
      // Compute offset for this host arg. Since args may have different sizes,
      // we need to track the cumulative offset. For simplicity, assume 4 bytes
      // per arg for now (RV32 pointers and common scalar types).
      unsigned argOffset;
      if (hostIdx >= 0) {
        // Use mapping: offset = USER_ARGS_OFFSET + hostIdx * 4
        argOffset = USER_ARGS_OFFSET + static_cast<unsigned>(hostIdx) * 4;
      } else {
        // Fallback to sequential (shouldn't happen for user args)
        argOffset = currentOffset;
      }

      SmallVector<LLVM::GEPArg> gepIndices;
      gepIndices.push_back(static_cast<int32_t>(argOffset));
      auto argBytePtr =
          builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, gepIndices);

      if (argType.isa<LLVM::LLVMPointerType>()) {
        // For pointers: load as i32 (RV32 pointer), then inttoptr
        auto rawPtr = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        argVal = builder.create<LLVM::IntToPtrOp>(loc, ptrType, rawPtr);
        currentOffset += 4;
      } else if (argType.isInteger(32)) {
        argVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        currentOffset += 4;
      } else if (argType.isInteger(64)) {
        // On RV32, i64 args (often from Polygeist's index type conversion) are
        // actually stored as 4-byte values in the host args buffer.
        // Load as i32 and zero-extend to i64.
        auto loadedVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        argVal = builder.create<LLVM::ZExtOp>(loc, i64Type, loadedVal);
        currentOffset += 4;
      } else if (argType.isF32()) {
        auto f32Type = Float32Type::get(ctx);
        argVal = builder.create<LLVM::LoadOp>(loc, f32Type, argBytePtr);
        currentOffset += 4;
      } else if (argType.isF64()) {
        auto f64Type = Float64Type::get(ctx);
        argVal = builder.create<LLVM::LoadOp>(loc, f64Type, argBytePtr);
        currentOffset += 8;
      } else if (argType.isInteger(1)) {
        // Boolean (i1): Load as i32, then truncate to i1
        // In the args buffer, booleans are stored as 4-byte values
        auto loadedVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        argVal = builder.create<LLVM::TruncOp>(loc, argType, loadedVal);
        currentOffset += 4;
      } else if (argType.isInteger(8)) {
        auto i8LoadType = IntegerType::get(ctx, 8);
        argVal = builder.create<LLVM::LoadOp>(loc, i8LoadType, argBytePtr);
        currentOffset += 1;
      } else if (argType.isInteger(16)) {
        auto i16Type = IntegerType::get(ctx, 16);
        argVal = builder.create<LLVM::LoadOp>(loc, i16Type, argBytePtr);
        currentOffset += 2;
      } else {
        // Default: treat as 4-byte value
        argVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        currentOffset += 4;
      }
    }

    unpackedArgs.push_back(argVal);
    ++i;
    ++origArgIdx;
  }

  // Call the original kernel function
  builder.create<LLVM::CallOp>(loc, kernelFunc, unpackedArgs);

  // Return void
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  return bodyFunc;
}

/// Generate main() entry point function
/// This function:
/// 1. Reads args from VX_CSR_MSCRATCH via inline assembly
/// 2. Extracts grid_dim pointer from args struct
/// 3. Calls vx_spawn_threads() with kernel_body callback
/// @param dimension 1, 2, or 3 for grid/block dimensionality
static LLVM::LLVMFuncOp generateMainFunction(ModuleOp module,
                                              LLVM::LLVMFuncOp bodyFunc,
                                              LLVM::LLVMFuncOp spawnFunc,
                                              unsigned dimension,
                                              OpBuilder &builder) {
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();

  auto i32Type = IntegerType::get(ctx, 32);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);

  // Create function: int main()
  auto funcType =
      LLVM::LLVMFunctionType::get(i32Type, {}, /*isVarArg=*/false);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  auto mainFunc =
      builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType, LLVM::Linkage::External);

  // Create entry block
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // 1. Read args from VX_CSR_MSCRATCH using inline assembly
  // csrr rd, 0x340
  // Note: LLVM 18+ requires direct result type, not struct-wrapped
  auto inlineAsm = builder.create<LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/i32Type,
      /*operands=*/ValueRange{},
      /*asm_string=*/"csrr $0, 0x340",
      /*constraints=*/"=r",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/LLVM::AsmDialectAttr{},
      /*operand_attrs=*/ArrayAttr{});

  // Get the result directly (no struct extraction needed for single output)
  auto argsRaw = inlineAsm.getRes();

  // Convert to pointer
  auto argsPtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, argsRaw);

  // 2. Get grid_dim pointer (first field of args struct, offset 0)
  // The args struct layout is: grid_dim[3], block_dim[3], user_args...
  // grid_dim is at offset 0 - for offset 0, argsPtr itself is the grid_dim pointer
  Value gridDimPtr = argsPtr;

  // 3. Get block_dim pointer (offset 12 = 3 * sizeof(uint32_t))
  auto i8Type = IntegerType::get(ctx, 8);
  SmallVector<LLVM::GEPArg> blockDimIndices;
  blockDimIndices.push_back(12);
  auto blockDimPtr =
      builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, blockDimIndices);

  // 4. Get kernel_body function pointer
  auto kernelPtr = builder.create<LLVM::AddressOfOp>(loc, ptrType, bodyFunc.getName());

  // 5. Call vx_spawn_threads(dimension, grid_dim, block_dim, kernel_body, args)
  // dimension is 1, 2, or 3 for 1D/2D/3D grids (read from vortex.kernel_dimension attr)
  auto dim = builder.create<LLVM::ConstantOp>(loc, i32Type, dimension);

  SmallVector<Value> spawnArgs = {dim, gridDimPtr, blockDimPtr, kernelPtr,
                                   argsPtr};
  auto result = builder.create<LLVM::CallOp>(loc, spawnFunc, spawnArgs);

  // 6. Return the result from vx_spawn_threads
  builder.create<LLVM::ReturnOp>(loc, result.getResult());

  return mainFunc;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_GENERATEVORTEXMAIN
#define GEN_PASS_DEF_GENERATEVORTEXMAIN
#include "polygeist/Passes/Passes.h.inc"

struct GenerateVortexMainPass
    : public impl::GenerateVortexMainBase<GenerateVortexMainPass> {

  GenerateVortexMainPass() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // 1. Find the kernel function
    LLVM::LLVMFuncOp kernelFunc = findKernelFunction(module);
    if (!kernelFunc) {
      // No kernel found - this might be a host-only module, skip silently
      return;
    }

    // 2. Check if main() already exists
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("main")) {
      // main() already exists, skip generation
      return;
    }

    // 3. Declare vx_spawn_threads
    auto spawnFunc = getOrDeclareVxSpawnThreads(module, builder);

    // 4. Generate kernel_body wrapper
    auto bodyFunc = generateKernelBodyWrapper(module, kernelFunc, builder);

    // 5. Read dimension from kernel function attribute (default to 1 for backward compatibility)
    // Dimension is set by ConvertGPUToVortex pass based on grid sizes
    unsigned dimension = 1;
    if (auto dimAttr = kernelFunc->getAttrOfType<IntegerAttr>("vortex.kernel_dimension")) {
      dimension = dimAttr.getInt();
    }

    // 6. Generate main() entry point
    generateMainFunction(module, bodyFunc, spawnFunc, dimension, builder);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createGenerateVortexMainPass() {
  return std::make_unique<GenerateVortexMainPass>();
}

} // namespace polygeist
} // namespace mlir
