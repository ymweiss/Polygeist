//===- ConvertGPUToVortex.cpp - Lower GPU dialect to Vortex intrinsics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that lowers GPU dialect operations to LLVM
// dialect with Vortex-specific intrinsics (CSR reads, custom instructions).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace mlir::gpu;

namespace {

//===----------------------------------------------------------------------===//
// Vortex CSR Addresses (from vx_intrinsics.h)
//===----------------------------------------------------------------------===//

constexpr uint32_t VX_CSR_THREAD_ID = 0xCC0;
constexpr uint32_t VX_CSR_WARP_ID = 0xCC1;
constexpr uint32_t VX_CSR_CORE_ID = 0xCC2;
constexpr uint32_t VX_CSR_NUM_THREADS = 0xFC0;
constexpr uint32_t VX_CSR_NUM_WARPS = 0xFC1;
constexpr uint32_t VX_CSR_NUM_CORES = 0xFC2;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get or create a TLS global variable for dim3_t type (threadIdx/blockIdx)
/// Returns the address of the TLS variable
static LLVM::GlobalOp getOrCreateDim3TLSGlobal(ModuleOp module,
                                               OpBuilder &builder,
                                               StringRef name) {
  MLIRContext *context = module.getContext();

  // Check if global already exists
  if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(name)) {
    return existing;
  }

  // Create dim3_t struct type: { i32, i32, i32 }
  auto i32Type = builder.getI32Type();
  auto dim3Type = LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type});

  // Create external thread-local global variable
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<LLVM::GlobalOp>(
      module.getLoc(),
      dim3Type,
      /*isConstant=*/false,
      LLVM::Linkage::External,
      name,
      /*value=*/Attribute(),
      /*alignment=*/0,
      /*addrSpace=*/0,
      /*dsoLocal=*/false,
      /*threadLocal=*/true);
}

/// Access a field of a TLS dim3_t variable (threadIdx or blockIdx)
/// dimension: gpu::Dimension::x (0), y (1), or z (2)
static Value createDim3TLSAccess(ModuleOp module,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 StringRef varName,
                                 gpu::Dimension dimension) {
  MLIRContext *context = module.getContext();

  // Get or create the TLS global variable
  auto globalVar = getOrCreateDim3TLSGlobal(module, rewriter, varName);

  // Get the address of the global
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto globalAddr = rewriter.create<LLVM::AddressOfOp>(
      loc, ptrType, globalVar.getSymName());

  // Create GEP to access the specific field (x=0, y=1, z=2)
  auto i32Type = rewriter.getI32Type();
  auto dim3Type = LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type});

  // GEP indices: [0, dimension]
  // First 0 is to dereference the pointer
  // Second index selects the struct field
  SmallVector<LLVM::GEPArg> indices;
  indices.push_back(0);  // Base index
  indices.push_back(static_cast<int32_t>(dimension));  // Field index (0=x, 1=y, 2=z)

  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, ptrType, dim3Type, globalAddr, indices);

  // Load the value from the computed address
  auto result = rewriter.create<LLVM::LoadOp>(loc, i32Type, gep);

  return result.getResult();
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower gpu.thread_id to TLS variable access
/// Accesses the threadIdx TLS variable set by vx_spawn_threads()
struct ThreadIdOpLowering : public ConvertOpToLLVMPattern<ThreadIdOp> {
  using ConvertOpToLLVMPattern<ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access threadIdx.{x,y,z} from TLS
    auto result = createDim3TLSAccess(module, rewriter, loc,
                                      "threadIdx", dimension);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower gpu.block_id to TLS variable access
/// Accesses the blockIdx TLS variable set by vx_spawn_threads()
struct BlockIdOpLowering : public ConvertOpToLLVMPattern<BlockIdOp> {
  using ConvertOpToLLVMPattern<BlockIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access blockIdx.{x,y,z} from TLS
    auto result = createDim3TLSAccess(module, rewriter, loc,
                                      "blockIdx", dimension);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertGPUToVortexPass
    : public PassWrapper<ConvertGPUToVortexPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUToVortexPass)

  StringRef getArgument() const final { return "convert-gpu-to-vortex"; }

  StringRef getDescription() const final {
    return "Lower GPU dialect operations to Vortex RISC-V intrinsics";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Set up type converter for GPU to LLVM types
    LLVMTypeConverter typeConverter(context);

    // Set up conversion target
    LLVMConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<ThreadIdOp, BlockIdOp>();

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<ThreadIdOpLowering, BlockIdOpLowering>(typeConverter);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createConvertGPUToVortexPass() {
  return std::make_unique<ConvertGPUToVortexPass>();
}

} // namespace polygeist
} // namespace mlir
