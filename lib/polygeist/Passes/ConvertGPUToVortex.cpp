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

/// Lower printf calls to vx_printf with core ID as first argument
/// Matches: llvm.call @printf(format, args...)
/// Replaces with: llvm.call @vx_printf(format, cid, args...)
/// where cid = vx_core_id()
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

    // Declare vx_core_id function in gpu.module if not already declared
    auto vxCoreIdFunc = gpuModule.lookupSymbol<LLVM::LLVMFuncOp>("vx_core_id");
    if (!vxCoreIdFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(gpuModule.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(i32Type, {}, /*isVarArg=*/false);
      vxCoreIdFunc = rewriter.create<LLVM::LLVMFuncOp>(
          gpuModule.getLoc(), "vx_core_id", funcType);
    }

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

    // Call vx_core_id() to get core ID
    auto coreIdCall = rewriter.create<LLVM::CallOp>(loc, vxCoreIdFunc, ValueRange{});
    Value coreId = coreIdCall.getResult();

    // Build new argument list: format, cid, original_args...
    SmallVector<Value> newArgs;
    newArgs.push_back(callOp.getOperand(0)); // format string (first arg)
    newArgs.push_back(coreId);                // core ID (new second arg)

    // Add remaining original arguments (skip format which is operand 0)
    for (unsigned i = 1; i < callOp.getNumOperands(); ++i) {
      newArgs.push_back(callOp.getOperand(i));
    }

    // Replace with call to vx_printf
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        callOp, vxPrintfFunc, newArgs);

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
    return "i32";  // Index maps to i32 on RV32
  return "unknown";
}

/// Get size in bytes for a type on RV32
static unsigned getTypeSizeRV32(Type type) {
  // On RV32 Vortex, pointers are 4 bytes
  if (type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>())
    return 4;
  if (type.isInteger(32) || type.isF32() || type.isIndex())
    return 4;
  if (type.isInteger(64) || type.isF64())
    return 8;
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

/// Generate C header string for kernel args struct (Vortex-compatible)
static std::string generateKernelArgsHeader(const KernelMetadata &meta) {
  std::ostringstream header;

  // Generate include guard
  std::string guardName = meta.kernelName;
  std::transform(guardName.begin(), guardName.end(), guardName.begin(), ::toupper);
  std::replace(guardName.begin(), guardName.end(), '-', '_');

  header << "// Auto-generated kernel argument structure for " << meta.kernelName << "\n";
  header << "// Generated by Polygeist ConvertGPUToVortex pass\n";
  header << "#ifndef " << guardName << "_ARGS_H\n";
  header << "#define " << guardName << "_ARGS_H\n\n";
  header << "#include <stdint.h>\n\n";

  header << "typedef struct {\n";
  for (const auto &arg : meta.arguments) {
    std::string cType = getCTypeString(arg.type);
    header << "  " << cType << " " << arg.name << ";";
    header << "  // offset=" << arg.offset << ", size=" << arg.size;
    if (arg.isPointer) header << ", device pointer";
    header << "\n";
  }
  header << "} " << meta.kernelName << "_args_t;\n\n";

  header << "#define " << guardName << "_ARGS_SIZE " << meta.totalArgsSize << "\n\n";
  header << "#endif // " << guardName << "_ARGS_H\n";

  return header.str();
}

/// Generate JSON string for kernel metadata (for runtime dynamic loading)
static std::string generateMetadataJSON(const KernelMetadata &meta) {
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
  json << "  \"architecture\": \"rv32\"\n";
  json << "}\n";

  return json.str();
}

/// Extract metadata from a GPU function and write metadata files
/// Generates both .meta.json (for runtime) and _args.h (for compile-time)
/// If outputDir is empty, uses current working directory
static void emitKernelMetadata(gpu::GPUFuncOp funcOp,
                                StringRef outputDir) {
  if (!funcOp.isKernel())
    return;

  KernelMetadata meta;
  meta.kernelName = funcOp.getName().str();

  // Extract base kernel name (remove Polygeist suffix if present)
  StringRef baseName = extractBaseKernelName(funcOp.getName());
  meta.kernelName = baseName.str();

  unsigned offset = 0;
  unsigned argIndex = 0;

  for (auto argType : funcOp.getArgumentTypes()) {
    KernelArgInfo argInfo;
    argInfo.name = "arg" + std::to_string(argIndex);
    argInfo.type = getMetadataTypeString(argType);
    argInfo.size = getTypeSizeRV32(argType);
    argInfo.offset = offset;
    argInfo.isPointer = argType.isa<MemRefType>() ||
                        argType.isa<LLVM::LLVMPointerType>();

    meta.arguments.push_back(argInfo);
    offset += argInfo.size;
    argIndex++;
  }

  meta.totalArgsSize = offset;

  // Determine output directory
  SmallString<256> outDir;
  if (outputDir.empty()) {
    llvm::sys::fs::current_path(outDir);
  } else {
    outDir = outputDir;
  }

  // Write JSON metadata file
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

  // Write C header file
  {
    SmallString<256> headerPath(outDir);
    llvm::sys::path::append(headerPath, meta.kernelName + "_args.h");

    std::error_code ec;
    llvm::raw_fd_ostream outFile(headerPath, ec);
    if (ec) {
      llvm::errs() << "Error writing header file " << headerPath << ": "
                   << ec.message() << "\n";
    } else {
      outFile << generateKernelArgsHeader(meta);
      outFile.close();
      llvm::outs() << "Wrote kernel args header: " << headerPath << "\n";
    }
  }
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

    // Apply metadata extraction and printf lowering as separate greedy rewrites
    RewritePatternSet metadataPatterns(context);
    metadataPatterns.add<LaunchFuncMetadataExtraction, PrintfOpLowering>(context);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(metadataPatterns)))) {
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
