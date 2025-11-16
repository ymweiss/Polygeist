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
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower gpu.thread_id to RISC-V CSR read via inline assembly
struct ThreadIdOpLowering : public ConvertOpToLLVMPattern<ThreadIdOp> {
  using ConvertOpToLLVMPattern<ThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // For now, all dimensions map to thread ID within warp
    // TODO: Proper 3D thread ID calculation from spawn framework
    uint32_t csrAddr = VX_CSR_THREAD_ID;

    // Create RISC-V inline assembly: csrr $0, <csr_addr>
    // This reads a Control Status Register and returns the value
    std::string asmStr = "csrr $0, " + std::to_string(csrAddr);

    // Create the inline assembly operation
    // Inputs: none
    // Outputs: one i32 value
    // Constraints: "$0" means first output register
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc,
        /*resultTypes=*/rewriter.getI32Type(),
        /*operands=*/ValueRange{},
        /*asm_string=*/asmStr,
        /*constraints=*/"=r",  // Output: any register
        /*has_side_effects=*/false,
        /*is_align_stack=*/false,
        /*asm_dialect=*/LLVM::AsmDialectAttr{},
        /*operand_attrs=*/ArrayAttr{});

    rewriter.replaceOp(op, asmOp.getRes());
    return success();
  }
};

/// Lower gpu.block_id to threadIdx from TLS (Thread Local Storage)
/// In Vortex spawn framework, blockIdx is a __thread variable
/// For now, we use CSR as placeholder - proper TLS access needs more work
struct BlockIdOpLowering : public ConvertOpToLLVMPattern<BlockIdOp> {
  using ConvertOpToLLVMPattern<BlockIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Placeholder: use warp ID as block ID for now
    // TODO: Access blockIdx TLS variable from vx_spawn framework
    uint32_t csrAddr = VX_CSR_WARP_ID;

    std::string asmStr = "csrr $0, " + std::to_string(csrAddr);

    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc,
        rewriter.getI32Type(),
        ValueRange{},
        asmStr,
        "=r",
        false, false,
        LLVM::AsmDialectAttr{},
        ArrayAttr{});

    rewriter.replaceOp(op, asmOp.getRes());
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
