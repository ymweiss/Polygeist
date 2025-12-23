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
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

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
constexpr uint32_t VX_CSR_LOCAL_MEM_BASE = 0xFC3;

//===----------------------------------------------------------------------===//
// Preprocessing: Consolidate Polygeist Alternatives
//===----------------------------------------------------------------------===//

/// Extract base kernel name by removing Polygeist variant suffix
/// Example: _Z12launch_basicPiS_ji_kernel94565344022848 -> _Z12launch_basicPiS_ji
/// Example: __polygeist_launch_vecadd_kernel_kernel94... -> __polygeist_launch_vecadd_kernel
static StringRef extractBaseKernelName(StringRef mangledName) {
  // Search from the end for "_kernel" followed by digits
  // This handles cases like "vecadd_kernel_kernel94..." where the kernel name
  // itself contains "_kernel"
  size_t searchStart = 0;
  size_t lastValidPos = StringRef::npos;

  while (true) {
    size_t pos = mangledName.find("_kernel", searchStart);
    if (pos == StringRef::npos)
      break;

    size_t suffixStart = pos + 7; // Length of "_kernel"
    if (suffixStart < mangledName.size() &&
        std::isdigit(mangledName[suffixStart])) {
      // Found "_kernel" followed by digit - this is a potential suffix
      lastValidPos = pos;
    }
    searchStart = pos + 1;
  }

  if (lastValidPos != StringRef::npos) {
    return mangledName.substr(0, lastValidPos);
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
    // Declare functions in gpu.module (not top-level module) so they're visible
    auto gpuModule = op->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
      return failure();
    MLIRContext *context = gpuModule.getContext();

    // Allocate barrier ID (simple counter for now)
    // TODO: Proper barrier ID allocation to avoid conflicts
    static int barrierIdCounter = 0;
    int barrierId = barrierIdCounter++;

    // Create barrier ID constant
    auto i32Type = rewriter.getI32Type();
    auto barIdConstant = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(barrierId));

    // Declare vx_num_warps_abi function in gpu.module if not already declared
    // Using _abi suffix to call the non-inline wrapper in vx_intrinsics_abi.c
    auto vxNumWarpsFunc = gpuModule.lookupSymbol<LLVM::LLVMFuncOp>("vx_num_warps_abi");
    if (!vxNumWarpsFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(gpuModule.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          i32Type, {}, /*isVarArg=*/false);

      vxNumWarpsFunc = rewriter.create<LLVM::LLVMFuncOp>(
          gpuModule.getLoc(), "vx_num_warps_abi", funcType);
    }

    // Call vx_num_warps_abi() to get number of warps
    auto numWarps = rewriter.create<LLVM::CallOp>(
        loc, vxNumWarpsFunc, ValueRange{});

    // Declare vx_barrier_abi function in gpu.module if not already declared
    // Using _abi suffix to call the non-inline wrapper in vx_intrinsics_abi.c
    auto vxBarrierFunc = gpuModule.lookupSymbol<LLVM::LLVMFuncOp>("vx_barrier_abi");
    if (!vxBarrierFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(gpuModule.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(context),
          {i32Type, i32Type},
          /*isVarArg=*/false);

      vxBarrierFunc = rewriter.create<LLVM::LLVMFuncOp>(
          gpuModule.getLoc(), "vx_barrier_abi", funcType);
    }

    // Call vx_barrier_abi(barrier_id, num_warps)
    SmallVector<Value> args;
    args.push_back(barIdConstant.getResult());
    args.push_back(numWarps.getResult());

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, vxBarrierFunc, args);

    return success();
  }
};

/// Lower printf calls to vx_printf
/// Matches: llvm.call @printf(format, args...)
/// Replaces with: llvm.call @vx_printf(format, args...)
/// vx_printf has the same signature as standard printf
struct PrintfOpLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                 PatternRewriter &rewriter) const override {
    // Only match calls to 'printf'
    auto callee = callOp.getCalleeAttr();
    if (!callee)
      return failure();

    auto flatSymbolRef = callee.dyn_cast<FlatSymbolRefAttr>();
    if (!flatSymbolRef || flatSymbolRef.getValue() != "printf")
      return failure();

    // Only lower printf calls inside GPU modules
    auto gpuModule = callOp->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
      return failure();

    Location loc = callOp.getLoc();
    MLIRContext *context = gpuModule.getContext();
    auto i32Type = rewriter.getI32Type();

    // Declare vx_printf function in gpu.module if not already declared
    auto vxPrintfFunc = gpuModule.lookupSymbol<LLVM::LLVMFuncOp>("vx_printf");
    if (!vxPrintfFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(gpuModule.getBody());

      auto ptrType = LLVM::LLVMPointerType::get(context);
      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType}, /*isVarArg=*/true);
      vxPrintfFunc = rewriter.create<LLVM::LLVMFuncOp>(
          gpuModule.getLoc(), "vx_printf", funcType);
    }

    // Build argument list: pass all original arguments unchanged
    SmallVector<Value> newArgs;
    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      newArgs.push_back(callOp.getOperand(i));
    }

    // Replace with call to vx_printf
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        callOp, vxPrintfFunc, newArgs);

    return success();
  }
};

/// Lower __threadfence* calls to RISC-V fence instructions
/// Matches: func.call @__threadfence* (void)
/// Replaces with: llvm.inline_asm "fence rw, rw" or "fence iorw, iorw"
///
/// Fence semantics:
/// - __threadfence_block(): Orders memory accesses for all threads within a block
/// - __threadfence(): Orders memory accesses for all threads on the device
/// - __threadfence_system(): Orders memory accesses visible to other devices and host
struct ThreadFenceOpLowering : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                 PatternRewriter &rewriter) const override {
    StringRef name = callOp.getCallee();

    // Match fence function names and determine instruction
    StringRef fenceAsm;
    if (name == "__threadfence" || name == "__threadfence_block") {
      // Block and device scope use same instruction on Vortex (single-core)
      fenceAsm = "fence rw, rw";
    } else if (name == "__threadfence_system") {
      // System scope includes I/O for visibility to host/other devices
      fenceAsm = "fence iorw, iorw";
    } else {
      return failure();
    }

    // Only lower inside GPU modules
    auto gpuModule = callOp->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
      return failure();

    // Emit RISC-V fence inline assembly
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        callOp,
        /*res=*/TypeRange{},
        /*operands=*/ValueRange{},
        /*asm_string=*/fenceAsm,
        /*constraints=*/"",
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*asm_dialect=*/LLVM::AsmDialectAttr{},
        /*operand_attrs=*/ArrayAttr{});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Lowering (Address Space 3)
//===----------------------------------------------------------------------===//

/// Track shared memory allocations for computing offsets
/// Maps global name to (offset, size) within the shared memory region
static llvm::StringMap<std::pair<unsigned, unsigned>> sharedMemoryLayout;
static unsigned totalSharedMemorySize = 0;

/// Get or declare the __local_group_id TLS variable accessor
/// Returns a function that provides access to the per-warp group ID
static LLVM::LLVMFuncOp getOrCreateLocalGroupIdAccessor(Operation *op,
                                                         OpBuilder &builder) {
  // Find the parent module (works for both gpu.module and regular module)
  Operation *symbolTableOp = op->getParentOfType<gpu::GPUModuleOp>();
  if (!symbolTableOp)
    symbolTableOp = op->getParentOfType<ModuleOp>();
  if (!symbolTableOp)
    return nullptr;

  std::string funcName = "vx_get_local_group_id";

  // Check if function already exists
  if (auto existing = SymbolTable::lookupSymbolIn(symbolTableOp,
                                                   builder.getStringAttr(funcName))) {
    return cast<LLVM::LLVMFuncOp>(existing);
  }

  // Create function type: () -> i32 (returns the local group ID)
  auto i32Type = builder.getI32Type();
  auto funcType = LLVM::LLVMFunctionType::get(i32Type, {}, /*isVarArg=*/false);

  // Declare external function
  OpBuilder::InsertionGuard guard(builder);
  if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(symbolTableOp)) {
    builder.setInsertionPointToStart(gpuModule.getBody());
  } else if (auto module = dyn_cast<ModuleOp>(symbolTableOp)) {
    builder.setInsertionPointToStart(module.getBody());
  }

  return builder.create<LLVM::LLVMFuncOp>(
      symbolTableOp->getLoc(),
      funcName,
      funcType,
      LLVM::Linkage::External);
}

/// Lower memref.global with address space 3 (shared memory)
/// These become placeholders - the actual allocation is done by vx_spawn_threads
/// We record the size and assign offsets for memref.get_global to use
struct SharedMemoryGlobalOpLowering : public OpRewritePattern<memref::GlobalOp> {
  using OpRewritePattern<memref::GlobalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::GlobalOp globalOp,
                                 PatternRewriter &rewriter) const override {
    // Only handle address space 3 (shared memory)
    auto memrefType = globalOp.getType();
    if (memrefType.getMemorySpaceAsInt() != 3)
      return failure();

    // Skip if already processed
    if (globalOp->hasAttr("vortex.shared_memory_offset"))
      return failure();

    // Calculate the size of this shared memory allocation
    unsigned elementSize = 4;  // Default to 4 bytes
    Type elemType = memrefType.getElementType();
    if (elemType.isF32() || elemType.isInteger(32))
      elementSize = 4;
    else if (elemType.isF64() || elemType.isInteger(64))
      elementSize = 8;
    else if (elemType.isInteger(8))
      elementSize = 1;
    else if (elemType.isInteger(16) || elemType.isF16())
      elementSize = 2;

    unsigned numElements = 1;
    for (int64_t dim : memrefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        // Dynamic shared memory - can't compute static offset
        globalOp.emitWarning("Dynamic shared memory size not supported");
        return failure();
      }
      numElements *= dim;
    }
    unsigned totalBytes = numElements * elementSize;

    // Assign offset in shared memory layout
    unsigned offset = totalSharedMemorySize;
    sharedMemoryLayout[globalOp.getSymName()] = {offset, totalBytes};
    totalSharedMemorySize += totalBytes;

    // Mark as processed with offset attribute
    rewriter.startRootUpdate(globalOp);
    globalOp->setAttr("vortex.shared_memory_offset",
                      rewriter.getI32IntegerAttr(offset));
    globalOp->setAttr("vortex.shared_memory_size",
                      rewriter.getI32IntegerAttr(totalBytes));
    rewriter.finalizeRootUpdate(globalOp);

    return success();
  }
};

/// Lower memref.get_global with address space 3 to Vortex local memory access
/// Generates: (int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * total_size + offset
/// Returns a proper memref descriptor for use with memref load/store operations
struct SharedMemoryGetGlobalOpLowering
    : public ConvertOpToLLVMPattern<memref::GetGlobalOp> {
  using ConvertOpToLLVMPattern<memref::GetGlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle address space 3 (shared memory)
    auto memrefType = getGlobalOp.getType();
    if (memrefType.getMemorySpaceAsInt() != 3)
      return failure();

    // Only handle static shapes (required for MemRefDescriptor::fromStaticShape)
    if (!memrefType.hasStaticShape()) {
      getGlobalOp.emitError("Dynamic shared memory shapes not supported");
      return failure();
    }

    Location loc = getGlobalOp.getLoc();
    MLIRContext *context = getGlobalOp.getContext();

    // Look up the offset for this global
    StringRef globalName = getGlobalOp.getName();
    auto it = sharedMemoryLayout.find(globalName);
    if (it == sharedMemoryLayout.end()) {
      // Not found - this can happen if GlobalOp lowering hasn't run yet
      // Try to find the GlobalOp and compute offset
      auto module = getGlobalOp->getParentOfType<ModuleOp>();
      auto gpuModule = getGlobalOp->getParentOfType<gpu::GPUModuleOp>();
      Operation *symbolTable = gpuModule ? (Operation*)gpuModule : (Operation*)module;

      if (auto globalOp = SymbolTable::lookupSymbolIn(symbolTable,
                                                       getGlobalOp.getNameAttr())) {
        if (auto memGlobalOp = dyn_cast<memref::GlobalOp>(globalOp)) {
          if (auto offsetAttr = memGlobalOp->getAttrOfType<IntegerAttr>(
                  "vortex.shared_memory_offset")) {
            unsigned offset = offsetAttr.getInt();
            unsigned size = 0;
            if (auto sizeAttr = memGlobalOp->getAttrOfType<IntegerAttr>(
                    "vortex.shared_memory_size")) {
              size = sizeAttr.getInt();
            }
            sharedMemoryLayout[globalName] = {offset, size};
            it = sharedMemoryLayout.find(globalName);
          }
        }
      }

      if (it == sharedMemoryLayout.end()) {
        getGlobalOp.emitError("Shared memory global not found in layout: ")
            << globalName;
        return failure();
      }
    }

    unsigned offset = it->second.first;
    auto i32Type = rewriter.getI32Type();

    // Get pointer type with address space 3 for shared memory
    unsigned addressSpace = memrefType.getMemorySpaceAsInt();
    Type elementType = getTypeConverter()->convertType(memrefType.getElementType());
    auto ptrType = getTypeConverter()->getPointerType(elementType, addressSpace);

    // Generate CSR read for local memory base
    // csrr %0, 0xFC3 (VX_CSR_LOCAL_MEM_BASE)
    std::string csrAsmStr = "csrr $0, " + std::to_string(VX_CSR_LOCAL_MEM_BASE);

    auto csrRead = rewriter.create<LLVM::InlineAsmOp>(
        loc,
        i32Type,             // result type (single i32, not a struct)
        ValueRange{},        // operands
        csrAsmStr,           // asm string
        "=r",                // constraints: output to register
        /*has_side_effects=*/false,
        /*is_align_stack=*/false,
        /*asm_dialect=*/nullptr,
        /*operand_attrs=*/nullptr);

    // The CSR read returns the base address directly
    Value baseAddr = csrRead.getResult(0);

    // Get __local_group_id via external function call
    auto localGroupIdFunc = getOrCreateLocalGroupIdAccessor(
        getGlobalOp, rewriter);
    if (!localGroupIdFunc)
      return failure();

    auto localGroupIdCall = rewriter.create<LLVM::CallOp>(
        loc, localGroupIdFunc, ValueRange{});
    Value localGroupId = localGroupIdCall.getResult();

    // Calculate final address:
    // base + localGroupId * totalSharedMemorySize + offset
    //
    // Note: For now, we use totalSharedMemorySize computed so far.
    // A more robust approach would compute this in a second pass.
    Value totalSizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(totalSharedMemorySize));
    Value groupOffset = rewriter.create<LLVM::MulOp>(
        loc, i32Type, localGroupId, totalSizeVal);
    Value baseWithGroup = rewriter.create<LLVM::AddOp>(
        loc, i32Type, baseAddr, groupOffset);

    // Add the static offset for this specific global
    Value offsetVal = rewriter.create<LLVM::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(offset));
    Value finalAddr = rewriter.create<LLVM::AddOp>(
        loc, i32Type, baseWithGroup, offsetVal);

    // Convert to pointer
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, finalAddr);

    // Create a memref descriptor from the computed pointer
    // This creates a proper LLVM struct with allocated_ptr, aligned_ptr, offset, sizes, strides
    Value descr = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memrefType, ptr);

    rewriter.replaceOp(getGlobalOp, descr);

    return success();
  }
};

/// Extract metadata from gpu.launch_func for Vortex kernel argument struct
/// For RV32, all arguments are 4 bytes (scalars and pointers)
struct LaunchFuncMetadataExtraction : public OpRewritePattern<gpu::LaunchFuncOp> {
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchOp,
                                 PatternRewriter &rewriter) const override {
    // Skip if metadata already exists (avoid infinite loop in greedy rewriter)
    if (launchOp->hasAttr("vortex.kernel_metadata"))
      return failure();

    Location loc = launchOp.getLoc();

    // Get kernel name
    StringRef kernelName = launchOp.getKernelName().getValue();

    // Get kernel arguments
    auto kernelOperands = launchOp.getKernelOperands();
    unsigned numArgs = kernelOperands.size();

    // For RV32: all arguments are 4 bytes (scalars and pointers)
    // Calculate total struct size: numArgs * 4
    unsigned totalSize = numArgs * 4;

    // Build metadata string for debugging/documentation
    std::string metadataStr = "Kernel: " + kernelName.str() +
                              "\nNum args: " + std::to_string(numArgs) +
                              "\nTotal size (RV32): " + std::to_string(totalSize) + " bytes\nArguments:\n";

    unsigned offset = 0;
    for (auto [idx, arg] : llvm::enumerate(kernelOperands)) {
      Type argType = arg.getType();
      bool isPointer = argType.isa<MemRefType>();

      metadataStr += "  [" + std::to_string(idx) + "] offset=" + std::to_string(offset) +
                     ", size=4, type=" + (isPointer ? "pointer" : "scalar") + "\n";
      offset += 4;
    }

    // Emit metadata as a comment for now (can be enhanced to create LLVM metadata)
    rewriter.startRootUpdate(launchOp);
    launchOp->setAttr("vortex.kernel_metadata",
                      rewriter.getStringAttr(metadataStr));
    rewriter.finalizeRootUpdate(launchOp);

    // Note: We don't replace the op, just annotate it with metadata
    // The actual launch lowering will be handled separately
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel Metadata JSON Emission
//===----------------------------------------------------------------------===//

/// Structure to hold kernel argument metadata
struct KernelArgInfo {
  std::string name;
  std::string type;  // "ptr", "i32", "u32", "f32", "f64", etc.
  unsigned size;     // Size in bytes
  unsigned offset;   // Offset in args struct
  bool isPointer;
  bool isSynthetic = false;  // True for Polygeist-generated args (loop bounds)
  int sourceArg = -1;        // For synthetic args: which user arg to derive value from
};

/// Structure to hold complete kernel metadata
struct KernelMetadata {
  std::string kernelName;
  std::vector<KernelArgInfo> arguments;
  unsigned totalArgsSize;
};

/// Convert MLIR type to metadata type string
static std::string getMetadataTypeString(Type type) {
  if (type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>())
    return "ptr";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(64))
    return "i64";
  if (type.isF32())
    return "f32";
  if (type.isF64())
    return "f64";
  if (type.isIndex())
    return "i64";  // Index maps to i64 by default in LLVM lowering
  return "unknown";
}

/// Get size in bytes for a type on RV32
static unsigned getTypeSizeRV32(Type type) {
  // On RV32 Vortex, pointers are 4 bytes
  if (type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>())
    return 4;
  if (type.isInteger(32) || type.isF32())
    return 4;
  if (type.isInteger(64) || type.isF64() || type.isIndex())
    return 8;  // Index maps to i64 in LLVM lowering
  return 4;  // Default
}

/// Convert metadata type string to C type
static std::string getCTypeString(const std::string &metaType) {
  if (metaType == "ptr") return "uint32_t";  // RV32 pointer = 32-bit device address
  if (metaType == "i32") return "int32_t";
  if (metaType == "u32") return "uint32_t";
  if (metaType == "i64") return "int64_t";
  if (metaType == "u64") return "uint64_t";
  if (metaType == "f32") return "float";
  if (metaType == "f64") return "double";
  return "uint32_t";  // Default
}

/// Convert metadata type to C type for function parameters (host ABI)
/// Unlike getCTypeString (for packed struct), this uses void* for pointers
static std::string getHostParamTypeString(const std::string &metaType, bool isPointer) {
  if (isPointer) return "const void*";  // Host pointers are void*
  if (metaType == "i32") return "int32_t";
  if (metaType == "u32") return "uint32_t";
  if (metaType == "i64") return "int64_t";
  if (metaType == "u64") return "uint64_t";
  if (metaType == "f32") return "float";
  if (metaType == "f64") return "double";
  if (metaType == "ptr") return "const void*";
  return "uint32_t";  // Default
}

/// Generate C++ header string with complete host stub (Vortex-compatible)
/// Includes: args struct, metadata array, and inline launcher function
static std::string generateKernelArgsHeader(const KernelMetadata &meta) {
  std::ostringstream header;

  // Generate include guard
  std::string guardName = meta.kernelName;
  std::transform(guardName.begin(), guardName.end(), guardName.begin(), ::toupper);
  std::replace(guardName.begin(), guardName.end(), '-', '_');

  header << "// Auto-generated host stub for " << meta.kernelName << "\n";
  header << "// Generated by Polygeist ConvertGPUToVortex pass\n";
  header << "// Include after hip_vortex_runtime.h\n";
  header << "#ifndef " << guardName << "_ARGS_H\n";
  header << "#define " << guardName << "_ARGS_H\n\n";
  header << "#include <stdint.h>\n\n";

  // Check that runtime header was included
  header << "#ifndef HIP_VORTEX_RUNTIME_H\n";
  header << "#error \"" << meta.kernelName << "_args.h must be included after hip_vortex_runtime.h\"\n";
  header << "#endif\n\n";

  // 1. Generate packed args struct (using void* for pointers on host)
  header << "// Argument structure for " << meta.kernelName << "\n";
  header << "// Pointers stored as void* (host format, converted by vortexLaunchKernel)\n";
  header << "typedef struct __attribute__((packed)) {\n";

  // Compute host offsets (void* is 8 bytes on 64-bit)
  unsigned hostOffset = 0;
  for (const auto &arg : meta.arguments) {
    std::string cType;
    unsigned hostSize;
    if (arg.isPointer) {
      cType = "void*";
      hostSize = 8;  // 64-bit host pointer
    } else {
      cType = getCTypeString(arg.type);
      hostSize = arg.size;
    }
    header << "  " << cType << " " << arg.name << ";";
    header << "  // host_offset=" << hostOffset << ", host_size=" << hostSize;
    if (arg.isPointer) header << ", device pointer";
    header << "\n";
    hostOffset += hostSize;
  }
  header << "} " << meta.kernelName << "_args_t;\n\n";

  // 2. Generate metadata array for vortexLaunchKernel
  header << "// Metadata array for vortexLaunchKernel\n";
  header << "// Offsets are for host struct (sizeof(void*) = 8 on 64-bit)\n";
  header << "static const VortexKernelArgMeta " << meta.kernelName << "_metadata[] = {\n";

  hostOffset = 0;
  for (const auto &arg : meta.arguments) {
    unsigned hostSize = arg.isPointer ? 8 : arg.size;  // void* is 8 bytes on host
    header << "  { .offset = " << hostOffset
           << ", .size = " << hostSize
           << ", .is_pointer = " << (arg.isPointer ? "1" : "0")
           << " },  // " << arg.name << "\n";
    hostOffset += hostSize;
  }
  header << "};\n";
  header << "#define " << guardName << "_NUM_ARGS " << meta.arguments.size() << "\n\n";

  // 3. Generate inline launcher function
  header << "// Type-safe launcher for " << meta.kernelName << "\n";
  header << "// Call this from host code with original argument types\n";
  header << "static inline hipError_t launch_" << meta.kernelName << "(\n";
  header << "    dim3 gridDim, dim3 blockDim";

  // Function parameters
  for (const auto &arg : meta.arguments) {
    std::string paramType = getHostParamTypeString(arg.type, arg.isPointer);
    header << ",\n    " << paramType << " " << arg.name;
  }
  header << ") {\n\n";

  // Pack arguments into struct
  header << "  " << meta.kernelName << "_args_t args;\n";
  for (const auto &arg : meta.arguments) {
    if (arg.isPointer) {
      header << "  args." << arg.name << " = (void*)" << arg.name << ";\n";
    } else {
      header << "  args." << arg.name << " = " << arg.name << ";\n";
    }
  }
  header << "\n";

  // Call vortexLaunchKernel
  header << "  return vortexLaunchKernel(\n";
  header << "    \"" << meta.kernelName << "\",\n";
  header << "    gridDim, blockDim,\n";
  header << "    &args, sizeof(args),\n";
  header << "    " << meta.kernelName << "_metadata, " << guardName << "_NUM_ARGS);\n";
  header << "}\n\n";

  header << "#endif // " << guardName << "_ARGS_H\n";

  return header.str();
}

/// Generate JSON string for kernel metadata (for runtime dynamic loading)
static std::string generateMetadataJSON(const KernelMetadata &meta,
                                         const std::vector<unsigned> &originalOrder = {}) {
  std::ostringstream json;
  json << "{\n";
  json << "  \"kernel_name\": \"" << meta.kernelName << "\",\n";
  json << "  \"arguments\": [\n";

  for (size_t i = 0; i < meta.arguments.size(); ++i) {
    const auto &arg = meta.arguments[i];
    json << "    {\n";
    json << "      \"name\": \"" << arg.name << "\",\n";
    json << "      \"type\": \"" << arg.type << "\",\n";
    json << "      \"size\": " << arg.size << ",\n";
    json << "      \"offset\": " << arg.offset << ",\n";
    json << "      \"is_pointer\": " << (arg.isPointer ? "true" : "false") << "\n";
    json << "    }";
    if (i < meta.arguments.size() - 1)
      json << ",";
    json << "\n";
  }

  json << "  ],\n";
  json << "  \"total_args_size\": " << meta.totalArgsSize << ",\n";

  // Include original argument order mapping if available
  // This maps from original (hipLaunchKernelGGL) order to device order
  if (!originalOrder.empty()) {
    json << "  \"original_arg_order\": [";
    for (size_t i = 0; i < originalOrder.size(); ++i) {
      json << originalOrder[i];
      if (i < originalOrder.size() - 1)
        json << ", ";
    }
    json << "],\n";
  }

  json << "  \"architecture\": \"rv32\"\n";
  json << "}\n";

  return json.str();
}

/// Extract metadata from a GPU function and write metadata files
/// Generates both .meta.json (for runtime) and _args.h (for compile-time)
/// If outputDir is empty, uses current working directory
/// Prioritizes vortex.kernel_args attribute (original HIP order from AST)
/// Falls back to kernel_arg_mapping + MLIR types if not available
static void emitKernelMetadata(gpu::GPUFuncOp funcOp, StringRef outputDir) {
  if (!funcOp.isKernel())
    return;

  KernelMetadata meta;
  meta.kernelName = funcOp.getName().str();

  // PRIORITY 0: Check for vortex.kernel_name attribute (set by ReorderGPUKernelArgsPass)
  // This contains the ORIGINAL HIP kernel name before inlining/outlining mangled it
  if (auto kernelNameAttr = funcOp->getAttrOfType<StringAttr>("vortex.kernel_name")) {
    meta.kernelName = kernelNameAttr.getValue().str();
    llvm::outs() << "Using vortex.kernel_name attribute: " << meta.kernelName << "\n";
  } else {
    // Extract base kernel name (remove Polygeist suffix if present)
    StringRef baseName = extractBaseKernelName(funcOp.getName());
    meta.kernelName = baseName.str();
    llvm::outs() << "Using function name: " << meta.kernelName << "\n";
  }

  // PRIORITY 1: Use vortex.kernel_args attribute if available
  // This contains the ORIGINAL HIP argument order from AST, which is what the
  // host code expects. The MLIR function signature may have reordered args.
  if (auto vortexArgsAttr = funcOp->getAttrOfType<ArrayAttr>("vortex.kernel_args")) {
    llvm::outs() << "Using vortex.kernel_args attribute for " << meta.kernelName
                 << " (original HIP argument order)\n";

    unsigned offset = 0;
    for (auto argAttr : vortexArgsAttr) {
      auto argDict = argAttr.dyn_cast<DictionaryAttr>();
      if (!argDict) continue;

      KernelArgInfo argInfo;

      // Extract name
      if (auto nameAttr = argDict.getAs<StringAttr>("name"))
        argInfo.name = nameAttr.getValue().str();
      else
        argInfo.name = "arg" + std::to_string(meta.arguments.size());

      // Extract type string
      if (auto typeAttr = argDict.getAs<StringAttr>("type"))
        argInfo.type = typeAttr.getValue().str();
      else
        argInfo.type = "unknown";

      // Extract size (from AST, this is host size - we need device size)
      // For pointers: host=8 bytes (64-bit), device=4 bytes (RV32)
      if (auto sizeAttr = argDict.getAs<IntegerAttr>("size"))
        argInfo.size = sizeAttr.getInt();
      else
        argInfo.size = 4;

      // Check if pointer
      if (auto isPtrAttr = argDict.getAs<BoolAttr>("is_pointer"))
        argInfo.isPointer = isPtrAttr.getValue();
      else
        argInfo.isPointer = (argInfo.type.find('*') != std::string::npos);

      // For device struct, pointers are 4 bytes (RV32), not 8 bytes
      unsigned deviceSize = argInfo.isPointer ? 4 : argInfo.size;

      // Compute offset (recalculate for device layout)
      argInfo.offset = offset;
      offset += deviceSize;
      // Update size to device size for the metadata
      argInfo.size = deviceSize;

      meta.arguments.push_back(argInfo);
    }
    meta.totalArgsSize = offset;

  } else {
    // PRIORITY 2: Fall back to kernel_arg_mapping + MLIR function types
    // This path is used when vortex.kernel_args attribute is not propagated
    llvm::outs() << "Falling back to kernel_arg_mapping for " << meta.kernelName << "\n";

    auto argTypes = funcOp.getArgumentTypes();
    unsigned totalArgs = argTypes.size();

    // Check for synthetic args (not user args):
    // - llvm.ptr: captured globals (e.g., printf format strings) - SKIP these
    // - index: loop bounds that Polygeist hoists into kernel parameters - INCLUDE these
    //
    // Captured globals are handled by Polygeist and don't need host data.
    // Index args (loop bounds) need buffer space and values derived from user args.
    //
    // Count synthetic args by type:
    unsigned numSyntheticPtrArgs = 0;  // llvm.ptr - skip from metadata
    unsigned numSyntheticIndexArgs = 0;  // index - include in metadata
    for (unsigned i = 0; i < totalArgs; ++i) {
      Type argType = argTypes[i];
      if (argType.isa<LLVM::LLVMPointerType>()) {
        numSyntheticPtrArgs++;
      } else if (argType.isa<IndexType>()) {
        numSyntheticIndexArgs++;
      }
    }
    llvm::outs() << "  Synthetic args: " << numSyntheticPtrArgs << " ptr (skip), "
                 << numSyntheticIndexArgs << " index (include)\n";

    // Only skip ptr-type synthetic args (captured globals)
    // Index args need to be included for proper buffer allocation
    unsigned effectiveTotalArgs = (totalArgs > numSyntheticPtrArgs) ? (totalArgs - numSyntheticPtrArgs) : totalArgs;

    // Use kernel_arg_mapping attribute to identify user args
    // The mapping tells us which host wrapper arg each kernel arg came from.
    // Host wrapper signature is: (user_arg0, user_arg1, ..., __blocks, __threads)
    // The last 2 host args are grid/block dimensions, NOT user args.
    SmallVector<unsigned> userArgIndices;

    // Track which args are index-type (synthetic loop bounds)
    SmallVector<bool, 8> isIndexArg(totalArgs, false);
    for (unsigned i = 0; i < totalArgs; ++i) {
      if (argTypes[i].isa<IndexType>()) {
        isIndexArg[i] = true;
      }
    }

    if (auto mappingAttr = funcOp->getAttrOfType<DenseI64ArrayAttr>("kernel_arg_mapping")) {
      auto mapping = mappingAttr.asArrayRef();

      // Find the maximum host arg index to determine dim arg boundary
      int64_t maxHostIdx = -1;
      for (int64_t idx : mapping) {
        if (idx > maxHostIdx) maxHostIdx = idx;
      }

      // If all mappings are -1, tracing completely failed
      if (maxHostIdx < 0) {
        for (unsigned i = 0; i < effectiveTotalArgs; ++i) {
          userArgIndices.push_back(i);
        }
      } else {
        // Trust the mapping: if mapping[i] >= 0, it's a valid user arg.
        // Don't assume grid/block dims are at the end - they may have been
        // specialized away by Polygeist (e.g., in main_kernel with constant dims).
        llvm::SmallSet<int64_t, 8> seenHostArgs;
        for (unsigned i = 0; i < mapping.size() && i < effectiveTotalArgs; ++i) {
          int64_t hostIdx = mapping[i];
          // Include arg if it has a valid mapping (non-negative means user arg)
          if (hostIdx >= 0) {
            if (seenHostArgs.insert(hostIdx).second) {
              userArgIndices.push_back(i);
            }
          } else if (isIndexArg[i]) {
            // Synthetic index arg (loop bound) - include for buffer allocation
            // The runtime will initialize this based on a heuristic
            userArgIndices.push_back(i);
            llvm::outs() << "  Including synthetic index arg " << i << " for buffer allocation\n";
          }
        }
      }
    } else {
      // Fallback: assume all non-captured args are user args
      for (unsigned i = 0; i < effectiveTotalArgs; ++i) {
        userArgIndices.push_back(i);
      }
    }

    unsigned offset = 0;

    for (unsigned i = 0; i < userArgIndices.size(); ++i) {
      unsigned argIndex = userArgIndices[i];
      Type argType = argTypes[argIndex];

      KernelArgInfo argInfo;
      argInfo.name = "arg" + std::to_string(i);  // Renumber from 0
      argInfo.type = getMetadataTypeString(argType);
      argInfo.size = getTypeSizeRV32(argType);
      argInfo.offset = offset;
      argInfo.isPointer = argType.isa<MemRefType>() ||
                          argType.isa<LLVM::LLVMPointerType>();

      meta.arguments.push_back(argInfo);
      offset += argInfo.size;
    }

    meta.totalArgsSize = offset;
  }  // End of fallback (else) block

  // Determine output directory
  SmallString<256> outDir;
  if (outputDir.empty()) {
    llvm::sys::fs::current_path(outDir);
  } else {
    outDir = outputDir;
  }

  // Write JSON metadata file (with original order mapping if available)
  {
    SmallString<256> jsonPath(outDir);
    llvm::sys::path::append(jsonPath, meta.kernelName + ".meta.json");

    std::error_code ec;
    llvm::raw_fd_ostream outFile(jsonPath, ec);
    if (ec) {
      llvm::errs() << "Error writing metadata file " << jsonPath << ": "
                   << ec.message() << "\n";
    } else {
      outFile << generateMetadataJSON(meta);
      outFile.close();
      llvm::outs() << "Wrote kernel metadata: " << jsonPath << "\n";
    }
  }

  // NOTE: _args.h stub generation is now handled by HIPSourceTransform (clang AST pass)
  // which generates stubs with correct argument order BEFORE Polygeist reorders them.
  // We only generate .meta.json here for runtime metadata.
  // The _args.h generation code has been disabled to avoid overwriting correct stubs.
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

// Use the tablegen-generated base class which handles the pass options correctly
#define GEN_PASS_DECL_CONVERTGPUTOVORTEX
#define GEN_PASS_DEF_CONVERTGPUTOVORTEX
#include "polygeist/Passes/Passes.h.inc"

struct ConvertGPUToVortexPass
    : public impl::ConvertGPUToVortexBase<ConvertGPUToVortexPass> {

  ConvertGPUToVortexPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // PREPROCESSING: Consolidate Polygeist auto-tuning artifacts
    // This must happen before any conversion patterns are applied
    consolidatePolygeistAlternatives(module);
    removeDuplicateKernels(module);

    // Always emit kernel metadata for each kernel
    // Files are written to current working directory:
    //   - <kernel_name>.meta.json (for runtime dynamic loading)
    //   - <kernel_name>_args.h (for compile-time type-safe usage)
    // Uses kernel_arg_mapping attribute to identify user args
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (gpuFunc.isKernel()) {
          emitKernelMetadata(gpuFunc, "" /* use current directory */);
        }
      }
    });

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

    // Apply metadata extraction, printf lowering, and shared memory global annotation
    // as separate greedy rewrites (these patterns don't replace ops, just annotate)
    RewritePatternSet metadataPatterns(context);
    metadataPatterns.add<LaunchFuncMetadataExtraction, PrintfOpLowering,
                         ThreadFenceOpLowering, SharedMemoryGlobalOpLowering>(context);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(metadataPatterns)))) {
      signalPassFailure();
    }

    // Lower memref.get_global for address space 3 (shared memory) to Vortex intrinsics
    // This must run after SharedMemoryGlobalOpLowering has annotated the globals
    {
      ConversionTarget sharedMemTarget(*context);
      sharedMemTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
      sharedMemTarget.addDynamicallyLegalOp<memref::GetGlobalOp>(
          [](memref::GetGlobalOp op) {
            // Only make shared memory (address space 3) get_global ops illegal
            return op.getType().getMemorySpaceAsInt() != 3;
          });

      RewritePatternSet sharedMemPatterns(context);
      sharedMemPatterns.add<SharedMemoryGetGlobalOpLowering>(typeConverter);

      if (failed(applyPartialConversion(module, sharedMemTarget,
                                         std::move(sharedMemPatterns)))) {
        signalPassFailure();
      }
    }

    // Extract kernel dimension from gpu.launch_func and set as attribute on kernel
    // This determines whether vx_spawn_threads uses 1D, 2D, or 3D dispatch
    // Dimension is determined by checking if gridSizeY/Z are constant 1
    module.walk([&](gpu::LaunchFuncOp launchOp) {
      // Find the corresponding gpu.func
      auto kernelSymbol = launchOp.getKernel();
      auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(
          kernelSymbol.getRootReference());
      if (!gpuModule)
        return;
      auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(
          kernelSymbol.getLeafReference());
      if (!gpuFunc || gpuFunc->hasAttr("vortex.kernel_dimension"))
        return;  // Already processed or not found

      // Determine dimension from grid sizes
      // 1D: gridSizeY == 1 && gridSizeZ == 1
      // 2D: gridSizeZ == 1
      // 3D: otherwise
      unsigned dimension = 3;  // Default to 3D

      auto isConstantOne = [](Value v) -> bool {
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
            return intAttr.getInt() == 1;
          }
        }
        return false;
      };

      bool gridYIsOne = isConstantOne(launchOp.getGridSizeY());
      bool gridZIsOne = isConstantOne(launchOp.getGridSizeZ());

      if (gridYIsOne && gridZIsOne) {
        dimension = 1;
      } else if (gridZIsOne) {
        dimension = 2;
      }

      // Set dimension attribute on the kernel function
      auto dimAttr = IntegerAttr::get(IntegerType::get(context, 32), dimension);
      gpuFunc->setAttr("vortex.kernel_dimension", dimAttr);
    });

    // Remove gpu.launch_func operations - they were needed for Polygeist
    // to generate proper MLIR but are not needed for Vortex kernel compilation.
    // The host code handles kernel launching through the Vortex runtime separately.
    SmallVector<gpu::LaunchFuncOp> launchOps;
    module.walk([&](gpu::LaunchFuncOp launchOp) {
      launchOps.push_back(launchOp);
    });
    for (auto launchOp : launchOps) {
      launchOp.erase();
    }

    // Remove host-side functions (those outside gpu.module)
    // Keep only the kernel code inside gpu.module for kernel binary compilation
    SmallVector<func::FuncOp> hostFuncs;
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      hostFuncs.push_back(funcOp);
    }
    for (auto funcOp : hostFuncs) {
      funcOp.erase();
    }

    // Extract kernel functions from gpu.module and convert to func.func
    // This allows standard MLIR lowering passes to work on the kernel code
    OpBuilder builder(context);
    SmallVector<gpu::GPUModuleOp> gpuModulesToErase;

    module.walk([&](gpu::GPUModuleOp gpuModule) {
      // Clone kernel functions as func.func at module level
      for (auto gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        // Create func.func with same name and type
        builder.setInsertionPointToEnd(module.getBody());

        auto funcOp = builder.create<func::FuncOp>(
            gpuFunc.getLoc(),
            gpuFunc.getName(),
            gpuFunc.getFunctionType());

        // Don't copy GPU-specific attributes - they're not relevant for Vortex
        // Skipped attributes: gpu.kernel, gpu.known_block_size, nvvm.*, rocdl.*
        // The kernel will use Vortex runtime conventions instead

        // However, DO preserve kernel_arg_mapping - it's needed by GenerateVortexMain
        // to identify synthetic arguments that need to be computed at runtime
        if (auto mappingAttr = gpuFunc->getAttr("kernel_arg_mapping")) {
          funcOp->setAttr("kernel_arg_mapping", mappingAttr);
        }

        // Preserve kernel dimension (1D, 2D, 3D) for vx_spawn_threads
        if (auto dimAttr = gpuFunc->getAttr("vortex.kernel_dimension")) {
          funcOp->setAttr("vortex.kernel_dimension", dimAttr);
        }

        // Clone the function body
        IRMapping mapping;
        gpuFunc.getBody().cloneInto(&funcOp.getBody(), mapping);

        // Replace gpu.return with func.return in the cloned body
        funcOp.walk([&](gpu::ReturnOp returnOp) {
          OpBuilder returnBuilder(returnOp);
          returnBuilder.create<func::ReturnOp>(returnOp.getLoc(),
                                                returnOp.getOperands());
          returnOp.erase();
        });

        // Replace unrealized_conversion_cast (i32 -> index) with arith.index_cast
        // These come from Polygeist's type conversions and can't be reconciled as-is
        SmallVector<UnrealizedConversionCastOp> castsToReplace;
        funcOp.walk([&](UnrealizedConversionCastOp castOp) {
          // Only replace i32 -> index casts
          if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
            auto srcType = castOp.getOperand(0).getType();
            auto dstType = castOp.getResult(0).getType();
            if (srcType.isInteger(32) && dstType.isIndex()) {
              castsToReplace.push_back(castOp);
            }
          }
        });
        for (auto castOp : castsToReplace) {
          OpBuilder castBuilder(castOp);
          auto indexCast = castBuilder.create<arith::IndexCastOp>(
              castOp.getLoc(), castOp.getResult(0).getType(), castOp.getOperand(0));
          castOp.getResult(0).replaceAllUsesWith(indexCast.getResult());
          castOp.erase();
        }
      }

      // Also clone any llvm.func declarations (like vx_get_threadIdx)
      for (auto llvmFunc : gpuModule.getOps<LLVM::LLVMFuncOp>()) {
        builder.setInsertionPointToEnd(module.getBody());
        // Check if already exists at module level
        if (!module.lookupSymbol(llvmFunc.getName())) {
          llvmFunc.clone();
          builder.clone(*llvmFunc.getOperation());
        }
      }

      gpuModulesToErase.push_back(gpuModule);
    });

    // Erase the gpu.module after extracting all functions
    for (auto gpuModule : gpuModulesToErase) {
      gpuModule.erase();
    }

    // Remove gpu.container_module attribute since we no longer have gpu.module
    module->removeAttr("gpu.container_module");
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
