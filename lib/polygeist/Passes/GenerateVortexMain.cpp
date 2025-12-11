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

/// Find the kernel function in the module (lowered from gpu.func)
/// After gpu-to-llvm, the kernel is an llvm.func with a mangled name
static LLVM::LLVMFuncOp findKernelFunction(ModuleOp module) {
  LLVM::LLVMFuncOp kernelFunc = nullptr;

  module.walk([&](LLVM::LLVMFuncOp func) {
    StringRef name = func.getName();
    // Look for functions with "_kernel" in the name (Polygeist naming convention)
    // or functions that were marked as kernels
    if (name.contains("_kernel") && !name.startswith("kernel_body")) {
      // Prefer the first kernel found
      if (!kernelFunc) {
        kernelFunc = func;
      }
    }
  });

  return kernelFunc;
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

  // Pre-load block_dim[0] for launch config args
  Value blockDimX_i32 = nullptr;
  Value blockDimX_i64 = nullptr;
  {
    SmallVector<LLVM::GEPArg> gepIndices;
    gepIndices.push_back(static_cast<int32_t>(BLOCK_DIM_OFFSET));
    auto blockDimPtr =
        builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, gepIndices);
    blockDimX_i32 = builder.create<LLVM::LoadOp>(loc, i32Type, blockDimPtr);
    blockDimX_i64 = builder.create<LLVM::ZExtOp>(loc, i64Type, blockDimX_i32);
  }

  // Track which original args we've processed (for mapping lookup)
  unsigned origArgIdx = 0;

  for (unsigned i = 0; i < numParams; ) {
    Type argType = kernelFuncType.getParamType(i);

    // Check if this is the start of a memref descriptor (5 consecutive params)
    if (isMemrefDescriptorStart(kernelFuncType, i)) {
      // This is a memref descriptor - load single pointer from args and expand
      SmallVector<LLVM::GEPArg> gepIndices;
      gepIndices.push_back(static_cast<int32_t>(currentOffset));
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

    if (origArgIdx < argMapping.size()) {
      int64_t hostIdx = argMapping[origArgIdx];
      if (hostIdx == -1) {
        isSyntheticArg = true;
      } else if (static_cast<unsigned>(hostIdx) >= numUserArgs) {
        isLaunchConfigArg = true;
      }
    }

    if (isSyntheticArg) {
      // Synthetic args (mapping = -1) are computed at launch time
      // For i1 args, this is typically a comparison result - use true as default
      // For pointer args, this is often a format string - find the global
      // For other types, use appropriate defaults
      if (argType.isInteger(1)) {
        // Boolean: default to true (common for bounds checks)
        argVal = builder.create<LLVM::ConstantOp>(loc, argType, 1);
      } else if (argType.isInteger(32)) {
        argVal = builder.create<LLVM::ConstantOp>(loc, argType, 0);
      } else if (argType.isInteger(64)) {
        argVal = builder.create<LLVM::ConstantOp>(loc, argType, 0);
      } else if (argType.isa<LLVM::LLVMPointerType>()) {
        // For pointer types (e.g., printf format strings), try to find a global
        // string constant. Look for globals like @str0, @str1, etc.
        LLVM::GlobalOp foundGlobal = nullptr;
        module.walk([&](LLVM::GlobalOp globalOp) {
          if (globalOp.getName().startswith("str")) {
            foundGlobal = globalOp;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (foundGlobal) {
          argVal = builder.create<LLVM::AddressOfOp>(loc, ptrType, foundGlobal.getName());
        } else {
          // Fallback: create null pointer
          auto nullVal = builder.create<LLVM::ZeroOp>(loc, ptrType);
          argVal = nullVal;
        }
      } else {
        // Default: 0
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
      SmallVector<LLVM::GEPArg> gepIndices;
      gepIndices.push_back(static_cast<int32_t>(currentOffset));
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
        argVal = builder.create<LLVM::LoadOp>(loc, i64Type, argBytePtr);
        currentOffset += 8;
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
static LLVM::LLVMFuncOp generateMainFunction(ModuleOp module,
                                              LLVM::LLVMFuncOp bodyFunc,
                                              LLVM::LLVMFuncOp spawnFunc,
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

  // 5. Call vx_spawn_threads(dimension=1, grid_dim, block_dim, kernel_body, args)
  // dimension=1 for 1D grid (most common case)
  // TODO: Support multi-dimensional grids by reading dimension from metadata
  auto dim = builder.create<LLVM::ConstantOp>(loc, i32Type, 1);

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

    // 5. Generate main() entry point
    generateMainFunction(module, bodyFunc, spawnFunc, builder);
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
