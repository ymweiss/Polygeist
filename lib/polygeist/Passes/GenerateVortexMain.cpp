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

/// Analyze kernel function to find the user arg range.
/// Looks for vortex.kernel_arg_ranges on the module, then falls back to heuristic.
/// Returns {userArgStartLLVM, userArgCount} where userArgStartLLVM is the
/// position in the LLVM function signature (accounting for memref expansion).
static std::pair<unsigned, unsigned>
findUserArgRange(LLVM::LLVMFuncOp kernelFunc, ModuleOp module) {
  auto kernelFuncType = kernelFunc.getFunctionType();
  unsigned numLLVMArgs = kernelFuncType.getNumParams();

  StringRef kernelName = kernelFunc.getName();

  // Look for module-level vortex.kernel_arg_ranges attribute
  unsigned userArgStart = 0;
  unsigned userArgCount = 0;
  bool foundAttr = false;

  if (auto rangesDict = module->getAttrOfType<DictionaryAttr>("vortex.kernel_arg_ranges")) {
    if (auto rangeAttr = rangesDict.getAs<DenseI64ArrayAttr>(kernelName)) {
      auto range = rangeAttr.asArrayRef();
      if (range.size() == 2) {
        userArgStart = static_cast<unsigned>(range[0]);
        userArgCount = static_cast<unsigned>(range[1]);
        foundAttr = true;
      }
    }
  }

  if (foundAttr) {
    // Convert from MLIR arg positions to LLVM arg positions
    // MLIR memrefs become 5 LLVM params each
    // We need to count how many LLVM params come before userArgStart
    unsigned llvmArgPos = 0;
    unsigned mlirArgPos = 0;

    // Count LLVM params for args before userArgStart
    while (mlirArgPos < userArgStart && llvmArgPos < numLLVMArgs) {
      if (isMemrefDescriptorStart(kernelFuncType, llvmArgPos)) {
        llvmArgPos += 5;  // memref expands to 5 params
      } else {
        llvmArgPos += 1;
      }
      mlirArgPos += 1;
    }

    return {llvmArgPos, userArgCount};
  }

  // Fallback: use heuristic based on signature pattern
  // After ReorderGPUKernelArgs, user args are contiguous and include all pointers
  // Synthetic args are i64 scalars at the boundaries

  // Count leading i64 scalars (before first memref/pointer)
  unsigned numLeadingScalars = 0;
  for (unsigned i = 0; i < numLLVMArgs; ++i) {
    Type argType = kernelFuncType.getParamType(i);
    if (isMemrefDescriptorStart(kernelFuncType, i) ||
        argType.isa<LLVM::LLVMPointerType>()) {
      break;
    }
    if (argType.isInteger(64)) {
      ++numLeadingScalars;
    } else {
      break; // Non-i64 scalar is a user arg
    }
  }

  // Count trailing i64 scalars (after last memref/pointer)
  unsigned numTrailingScalars = 0;
  for (unsigned i = numLLVMArgs; i > 0; --i) {
    Type argType = kernelFuncType.getParamType(i - 1);
    if (argType.isInteger(64)) {
      ++numTrailingScalars;
    } else {
      break;
    }
  }

  // User args are everything in between
  unsigned argsToSkip = numLeadingScalars;

  // Count MLIR-level user args (group memref descriptors)
  userArgCount = 0;
  for (unsigned i = argsToSkip; i < numLLVMArgs - numTrailingScalars; ) {
    if (isMemrefDescriptorStart(kernelFuncType, i)) {
      userArgCount += 1;
      i += 5;
    } else {
      userArgCount += 1;
      i += 1;
    }
  }

  return {argsToSkip, userArgCount};
}

/// Generate kernel_body wrapper function
/// This function unpacks arguments from the void* args pointer and calls
/// the original kernel function
///
/// IMPORTANT: After ReorderGPUKernelArgs pass, user args are in a contiguous
/// range marked by the vortex.user_arg_range attribute. Synthetic args
/// (derived from block_dim) are outside this range.
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
  unsigned numArgs = kernelFuncType.getNumParams();

  // Standard Vortex args layout:
  //   uint32_t grid_dim[3];   // 12 bytes (offsets 0, 4, 8)
  //   uint32_t block_dim[3];  // 12 bytes (offsets 12, 16, 20)
  //   <user args>             // starting at offset 24
  constexpr unsigned BLOCK_DIM_OFFSET = 12;
  constexpr unsigned USER_ARGS_OFFSET = 24;

  // Find user arg range (accounts for ReorderGPUKernelArgs)
  auto [userArgStartLLVM, userArgCount] = findUserArgRange(kernelFunc, module);

  SmallVector<Value> unpackedArgs;
  unsigned currentOffset = USER_ARGS_OFFSET;
  auto i8Type = IntegerType::get(ctx, 8);

  // Pre-load block_dim[0] for synthetic args
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

  // Track MLIR arg index for user arg range check
  unsigned mlirArgIdx = 0;

  for (unsigned i = 0; i < numArgs; ) {
    Type argType = kernelFuncType.getParamType(i);

    // Check if this LLVM arg position is within the user arg range
    bool isUserArg = (mlirArgIdx >= userArgStartLLVM &&
                      mlirArgIdx < userArgStartLLVM + userArgCount);

    // Check if this is the start of a memref descriptor (5 consecutive params)
    if (isMemrefDescriptorStart(kernelFuncType, i)) {
      if (isUserArg) {
        // User memref - load from user args buffer
        SmallVector<LLVM::GEPArg> gepIndices;
        gepIndices.push_back(static_cast<int32_t>(currentOffset));
        auto argBytePtr = builder.create<LLVM::GEPOp>(loc, ptrType, i8Type,
                                                       argsPtr, gepIndices);

        // Load the device pointer (4 bytes on RV32)
        auto rawPtr = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        auto devicePtr = builder.create<LLVM::IntToPtrOp>(loc, ptrType, rawPtr);

        // Construct memref descriptor values
        unpackedArgs.push_back(devicePtr);  // allocated ptr
        unpackedArgs.push_back(devicePtr);  // aligned ptr

        auto zeroI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 0);
        auto maxI64 = builder.create<LLVM::ConstantOp>(
            loc, i64Type, std::numeric_limits<int64_t>::max());
        auto oneI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 1);

        unpackedArgs.push_back(zeroI64);  // offset = 0
        unpackedArgs.push_back(maxI64);   // size = MAX
        unpackedArgs.push_back(oneI64);   // stride = 1

        currentOffset += 4;  // Single pointer in args buffer
      } else {
        // Synthetic memref - shouldn't happen, but handle gracefully
        // Use null pointer
        auto nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
        unpackedArgs.push_back(nullPtr);  // allocated ptr
        unpackedArgs.push_back(nullPtr);  // aligned ptr
        auto zeroI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 0);
        auto maxI64 = builder.create<LLVM::ConstantOp>(
            loc, i64Type, std::numeric_limits<int64_t>::max());
        auto oneI64 = builder.create<LLVM::ConstantOp>(loc, i64Type, 1);
        unpackedArgs.push_back(zeroI64);
        unpackedArgs.push_back(maxI64);
        unpackedArgs.push_back(oneI64);
      }

      i += 5;  // Skip 5 params (the whole memref descriptor)
      mlirArgIdx += 1;
      continue;
    }

    // Scalar argument handling
    Value argVal;

    if (isUserArg) {
      // User scalar - load from user args buffer
      SmallVector<LLVM::GEPArg> gepIndices;
      gepIndices.push_back(static_cast<int32_t>(currentOffset));
      auto argBytePtr =
          builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, gepIndices);

      if (argType.isa<LLVM::LLVMPointerType>()) {
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
      } else {
        argVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
        currentOffset += 4;
      }
    } else {
      // Synthetic scalar - derive from block_dim
      if (argType.isInteger(64)) {
        argVal = blockDimX_i64;
      } else {
        argVal = blockDimX_i32;
      }
    }

    unpackedArgs.push_back(argVal);
    ++i;
    mlirArgIdx += 1;
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
