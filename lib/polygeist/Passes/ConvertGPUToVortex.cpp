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
#include "llvm/ADT/StringMap.h"

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
// Preprocessing: Consolidate Polygeist Alternatives
//===----------------------------------------------------------------------===//

/// Extract base kernel name by removing Polygeist variant suffix
/// Example: _Z12launch_basicPiS_ji_kernel94565344022848 -> _Z12launch_basicPiS_ji
static StringRef extractBaseKernelName(StringRef mangledName) {
  size_t pos = mangledName.find("_kernel");
  if (pos != StringRef::npos) {
    // Find where the numeric suffix starts after "_kernel"
    size_t suffixStart = pos + 7; // Length of "_kernel"
    if (suffixStart < mangledName.size() &&
        std::isdigit(mangledName[suffixStart])) {
      return mangledName.substr(0, pos);
    }
  }
  return mangledName;
}

/// Consolidate polygeist.alternatives to first variant only
/// This preprocessing step simplifies downstream processing by:
/// 1. Replacing polygeist.alternatives with content of first alternative
/// 2. Ensuring single canonical launch configuration for Vortex
static void consolidatePolygeistAlternatives(ModuleOp module) {
  SmallVector<Operation *> altOps;

  // Collect all polygeist.alternatives operations
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "polygeist.alternatives") {
      altOps.push_back(op);
    }
  });

  // Replace each alternatives op with content of its first region
  for (Operation *altOp : altOps) {
    if (altOp->getNumRegions() == 0 || altOp->getRegion(0).empty())
      continue;

    OpBuilder builder(altOp);
    Region &firstRegion = altOp->getRegion(0);
    Block &firstBlock = firstRegion.front();

    // Move all operations from first region to parent block (before alternatives op)
    // This inlines the first alternative's content
    auto &ops = firstBlock.getOperations();
    for (Operation &innerOp : llvm::make_early_inc_range(ops)) {
      // Skip the terminator (polygeist.polygeist_yield)
      if (innerOp.getName().getStringRef() == "polygeist.polygeist_yield")
        continue;
      innerOp.moveBefore(altOp);
    }

    // Erase the now-empty alternatives operation
    altOp->erase();
  }
}

/// Remove duplicate GPU kernel functions, keeping only the first variant
/// After Polygeist auto-tuning, multiple kernel variants exist but only
/// the first one is referenced after consolidating alternatives.
static void removeDuplicateKernels(ModuleOp module) {
  // Track seen kernel base names
  llvm::StringMap<gpu::GPUFuncOp> seenKernels;
  SmallVector<gpu::GPUFuncOp> toErase;

  // Walk all GPU modules
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    // Collect all kernel functions
    for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
      if (!gpuFunc.isKernel())
        continue;

      StringRef funcName = gpuFunc.getName();
      StringRef baseName = extractBaseKernelName(funcName);

      // Check if we've seen this kernel base name before
      auto it = seenKernels.find(baseName);
      if (it != seenKernels.end()) {
        // Duplicate found - mark for deletion
        toErase.push_back(gpuFunc);
      } else {
        // First occurrence - keep it
        seenKernels[baseName] = gpuFunc;
      }
    }
  });

  // Erase duplicate kernels
  for (auto func : toErase) {
    func.erase();
  }
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Declare an external function to access TLS dim3_t variables
/// For thread-local variables like blockIdx/threadIdx, we generate helper
/// functions that return pointers to the TLS variables
/// Returns an LLVM function declaration
/// The function is declared within the gpu.module where it's being used
static LLVM::LLVMFuncOp getOrCreateDim3TLSAccessor(Operation *op,
                                                    OpBuilder &builder,
                                                    StringRef varName) {
  // Find the gpu.module containing this operation
  auto gpuModule = op->getParentOfType<gpu::GPUModuleOp>();
  MLIRContext *context = gpuModule.getContext();

  // Create function name: e.g., "vx_get_blockIdx"
  std::string funcName = ("vx_get_" + varName).str();

  // Check if function already exists in gpu.module
  if (auto existing = gpuModule.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    return existing;
  }

  // Create function type: () -> !llvm.ptr (returns pointer to dim3_t)
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto funcType = LLVM::LLVMFunctionType::get(ptrType, {}, /*isVarArg=*/false);

  // Declare external function within gpu.module
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(gpuModule.getBody());

  return builder.create<LLVM::LLVMFuncOp>(
      gpuModule.getLoc(),
      funcName,
      funcType,
      LLVM::Linkage::External);
}

/// Access a field of a TLS dim3_t variable (threadIdx or blockIdx)
/// dimension: gpu::Dimension::x (0), y (1), or z (2)
static Value createDim3TLSAccess(Operation *op,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 StringRef varName,
                                 gpu::Dimension dimension) {
  auto module = op->getParentOfType<ModuleOp>();
  MLIRContext *context = module.getContext();

  // Get or create the TLS accessor function
  auto accessorFunc = getOrCreateDim3TLSAccessor(op, rewriter, varName);

  // Call the accessor function to get pointer to TLS variable
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto callResult = rewriter.create<LLVM::CallOp>(
      loc, accessorFunc, ValueRange{});
  Value dim3Ptr = callResult.getResult();

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
      loc, ptrType, dim3Type, dim3Ptr, indices);

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

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access threadIdx.{x,y,z} from TLS
    auto result = createDim3TLSAccess(op, rewriter, loc,
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

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access blockIdx.{x,y,z} from TLS
    auto result = createDim3TLSAccess(op, rewriter, loc,
                                      "blockIdx", dimension);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower gpu.block_dim to TLS variable access
/// Accesses the blockDim global variable set by vx_spawn_threads()
struct BlockDimOpLowering : public ConvertOpToLLVMPattern<gpu::BlockDimOp> {
  using ConvertOpToLLVMPattern<gpu::BlockDimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access blockDim.{x,y,z} from global variable
    // Note: blockDim is NOT thread-local, it's a regular global
    auto result = createDim3TLSAccess(op, rewriter, loc,
                                      "blockDim", dimension);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower gpu.grid_dim to TLS variable access
/// Accesses the gridDim global variable set by vx_spawn_threads()
struct GridDimOpLowering : public ConvertOpToLLVMPattern<gpu::GridDimOp> {
  using ConvertOpToLLVMPattern<gpu::GridDimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::GridDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the dimension (X, Y, or Z)
    auto dimension = op.getDimension();

    // Access gridDim.{x,y,z} from global variable
    // Note: gridDim is NOT thread-local, it's a regular global
    auto result = createDim3TLSAccess(op, rewriter, loc,
                                      "gridDim", dimension);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower gpu.barrier to Vortex vx_barrier call
/// Synchronizes all threads in a block using Vortex hardware barriers
struct BarrierOpLowering : public ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();

    // Allocate barrier ID (simple counter for now)
    // TODO: Proper barrier ID allocation to avoid conflicts
    static int barrierIdCounter = 0;
    int barrierId = barrierIdCounter++;

    // Create barrier ID constant
    auto i32Type = rewriter.getI32Type();
    auto barIdConstant = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(barrierId));

    // Get blockDim to calculate total threads
    // We need blockDim.x * blockDim.y * blockDim.z
    auto blockDimX = createDim3TLSAccess(op, rewriter, loc,
                                         "blockDim", gpu::Dimension::x);
    auto blockDimY = createDim3TLSAccess(op, rewriter, loc,
                                         "blockDim", gpu::Dimension::y);
    auto blockDimZ = createDim3TLSAccess(op, rewriter, loc,
                                         "blockDim", gpu::Dimension::z);

    // Calculate total threads: x * y * z
    // blockDimX/Y/Z are already i32 from TLS load
    auto tempXY = rewriter.create<LLVM::MulOp>(loc, i32Type,
                                                blockDimX, blockDimY);
    auto numThreads = rewriter.create<LLVM::MulOp>(loc, i32Type,
                                                     tempXY, blockDimZ);

    // Declare vx_barrier function if not already declared
    auto vxBarrierFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("vx_barrier");
    if (!vxBarrierFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(context),
          {i32Type, i32Type},
          /*isVarArg=*/false);

      vxBarrierFunc = rewriter.create<LLVM::LLVMFuncOp>(
          module.getLoc(), "vx_barrier", funcType);
    }

    // Call vx_barrier(bar_id, num_threads)
    SmallVector<Value> args;
    args.push_back(barIdConstant.getResult());
    args.push_back(numThreads.getResult());

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, vxBarrierFunc, args);

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

    // PREPROCESSING: Consolidate Polygeist auto-tuning artifacts
    // This must happen before any conversion patterns are applied
    consolidatePolygeistAlternatives(module);
    removeDuplicateKernels(module);

    // Set up type converter for GPU to LLVM types
    LLVMTypeConverter typeConverter(context);

    // Set up conversion target
    // Mark only the Vortex-specific GPU operations as illegal
    // All other operations (including GPU structural ops) remain legal
    // A subsequent --gpu-to-llvm pass will handle gpu.module/gpu.func conversion
    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<ThreadIdOp, BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp,
                        gpu::BarrierOp>();

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<ThreadIdOpLowering, BlockIdOpLowering, BlockDimOpLowering,
                 GridDimOpLowering, BarrierOpLowering>(typeConverter);

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
