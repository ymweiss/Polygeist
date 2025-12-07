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

/// Generate kernel_body wrapper function
/// This function unpacks arguments from the void* args pointer and calls
/// the original kernel function
static LLVM::LLVMFuncOp
generateKernelBodyWrapper(ModuleOp module, LLVM::LLVMFuncOp kernelFunc,
                          OpBuilder &builder) {
  MLIRContext *ctx = module.getContext();
  Location loc = kernelFunc.getLoc();

  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  auto i32Type = IntegerType::get(ctx, 32);

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

  // Calculate offset for user args (skip grid_dim[3] and block_dim[3])
  // Standard Vortex args layout:
  //   uint32_t grid_dim[3];   // 12 bytes (offsets 0, 4, 8)
  //   uint32_t block_dim[3];  // 12 bytes (offsets 12, 16, 20)
  //   <user args>             // starting at offset 24
  constexpr unsigned USER_ARGS_OFFSET = 24;

  SmallVector<Value> unpackedArgs;
  unsigned currentOffset = USER_ARGS_OFFSET;

  for (unsigned i = 0; i < numArgs; ++i) {
    Type argType = kernelFuncType.getParamType(i);

    // Create GEP to access the argument at currentOffset
    // We treat the args struct as an array of bytes and use byte offsets
    auto i8Type = IntegerType::get(ctx, 8);

    // GEP to byte offset - use GEPArg for indices
    SmallVector<LLVM::GEPArg> gepIndices;
    gepIndices.push_back(static_cast<int32_t>(currentOffset));
    auto argBytePtr =
        builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, argsPtr, gepIndices);

    // Load the argument value
    Value argVal;
    if (argType.isa<LLVM::LLVMPointerType>()) {
      // For pointers: load as i32 (RV32 pointer), then inttoptr
      auto rawPtr = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
      argVal = builder.create<LLVM::IntToPtrOp>(loc, ptrType, rawPtr);
      currentOffset += 4; // Pointers are 4 bytes on RV32
    } else if (argType.isInteger(32) || argType.isIndex()) {
      argVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
      currentOffset += 4;
    } else if (argType.isInteger(64)) {
      auto i64Type = IntegerType::get(ctx, 64);
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
      // Default: treat as 4-byte value
      argVal = builder.create<LLVM::LoadOp>(loc, i32Type, argBytePtr);
      currentOffset += 4;
    }

    unpackedArgs.push_back(argVal);
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
  auto asmResultType = LLVM::LLVMStructType::getLiteral(ctx, {i32Type});

  auto inlineAsm = builder.create<LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/asmResultType,
      /*operands=*/ValueRange{},
      /*asm_string=*/"csrr $0, 0x340",
      /*constraints=*/"=r",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*asm_dialect=*/LLVM::AsmDialectAttr{},
      /*operand_attrs=*/ArrayAttr{});

  // Extract the result from the struct
  auto argsRaw = builder.create<LLVM::ExtractValueOp>(loc, i32Type,
                                                       inlineAsm.getRes(), 0);

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
