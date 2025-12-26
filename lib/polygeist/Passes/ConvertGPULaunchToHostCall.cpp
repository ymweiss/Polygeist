//===- ConvertGPULaunchToHostCall.cpp - Convert gpu.launch_func to host calls -===//
//
// This pass converts gpu.launch_func operations to calls to the Vortex HIP
// runtime. This allows compiling host launch wrappers as a native library.
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

/// Pattern to convert gpu.launch_func to calls to vortex_launch_kernel_direct
struct ConvertLaunchFuncToHostCall : public OpRewritePattern<gpu::LaunchFuncOp> {
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = launchOp.getLoc();
    ModuleOp module = launchOp->getParentOfType<ModuleOp>();

    // Get LLVM types
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Declare external runtime functions if not already declared
    auto getOrInsertFunc = [&](StringRef name, Type resultType,
                               ArrayRef<Type> argTypes) -> LLVM::LLVMFuncOp {
      if (auto existingFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return existingFunc;

      auto funcType = LLVM::LLVMFunctionType::get(resultType, argTypes);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      return rewriter.create<LLVM::LLVMFuncOp>(loc, name, funcType);
    };

    // hip_ptr_to_device_addr: (ptr) -> i32
    getOrInsertFunc("hip_ptr_to_device_addr", i32Type, {ptrType});

    // vortex_launch_with_args: (ptr, ptr, i64) -> i32
    // Args: kernel_name, vortex_args, args_size
    getOrInsertFunc("vortex_launch_with_args", i32Type, {ptrType, ptrType, i64Type});

    // Get kernel name
    StringRef kernelName = launchOp.getKernelName().getValue();

    // Create kernel name string constant
    std::string kernelNameStr = kernelName.str();
    kernelNameStr += ".vxbin";  // Append extension for lookup
    kernelNameStr.push_back('\0');  // Null terminator

    // Create global string for kernel name
    std::string globalName = ("__vortex_kernel_name_" + kernelName).str();
    LLVM::GlobalOp nameGlobal;
    if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
      nameGlobal = existing;
    } else {
      auto stringType = LLVM::LLVMArrayType::get(
          IntegerType::get(rewriter.getContext(), 8), kernelNameStr.size());
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      nameGlobal = rewriter.create<LLVM::GlobalOp>(
          loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal,
          globalName, rewriter.getStringAttr(kernelNameStr));
    }

    // Get pointer to kernel name
    Value kernelNamePtr = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, globalName);

    // Get grid and block dimensions
    Value gridX = launchOp.getGridSizeX();
    Value gridY = launchOp.getGridSizeY();
    Value gridZ = launchOp.getGridSizeZ();
    Value blockX = launchOp.getBlockSizeX();
    Value blockY = launchOp.getBlockSizeY();
    Value blockZ = launchOp.getBlockSizeZ();

    // Convert index types to i32 if needed
    auto toI32 = [&](Value v) -> Value {
      if (v.getType().isIndex()) {
        return rewriter.create<arith::IndexCastOp>(loc, i32Type, v);
      }
      if (v.getType() != i32Type) {
        return rewriter.create<LLVM::TruncOp>(loc, i32Type, v);
      }
      return v;
    };

    gridX = toI32(gridX);
    gridY = toI32(gridY);
    gridZ = toI32(gridZ);
    blockX = toI32(blockX);
    blockY = toI32(blockY);
    blockZ = toI32(blockZ);

    // Get kernel operands
    auto kernelOperands = launchOp.getKernelOperands();
    unsigned numArgs = kernelOperands.size();

    // Calculate buffer size: 6 * i32 (grid + block) + numArgs * i32
    unsigned headerSize = 6 * 4;  // 24 bytes
    unsigned argsSize = numArgs * 4;
    unsigned totalSize = headerSize + argsSize;

    // Allocate buffer on stack
    Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, 1);
    auto i8Type = rewriter.getI8Type();
    auto bufferType = LLVM::LLVMArrayType::get(i8Type, totalSize);
    Value buffer = rewriter.create<LLVM::AllocaOp>(loc, ptrType, bufferType, one);

    // Store grid dimensions (offset 0, 4, 8)
    auto storeI32AtOffset = [&](Value val, unsigned offset) {
      Value offsetVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type, offset);
      Value ptr = rewriter.create<LLVM::GEPOp>(loc, ptrType, i8Type, buffer,
                                               ValueRange{offsetVal});
      rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    };

    storeI32AtOffset(gridX, 0);
    storeI32AtOffset(gridY, 4);
    storeI32AtOffset(gridZ, 8);
    storeI32AtOffset(blockX, 12);
    storeI32AtOffset(blockY, 16);
    storeI32AtOffset(blockZ, 20);

    // Convert and store kernel arguments at offset 24+
    unsigned argOffset = 24;
    for (auto [idx, arg] : llvm::enumerate(kernelOperands)) {
      Type argType = arg.getType();
      Value argVal;

      if (argType.isa<MemRefType>()) {
        // Pointer argument - convert to device address
        // First extract the base pointer from memref
        Value basePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
            loc, arg);
        Value ptrAsInt = rewriter.create<arith::IndexCastOp>(loc, i64Type, basePtr);
        Value ptrVal = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrAsInt);

        // Call hip_ptr_to_device_addr
        argVal = rewriter.create<LLVM::CallOp>(
            loc, i32Type, "hip_ptr_to_device_addr", ValueRange{ptrVal}).getResult();
      } else if (argType.isInteger(32)) {
        argVal = arg;
      } else if (argType.isInteger(64)) {
        // Truncate to 32-bit for device
        argVal = rewriter.create<LLVM::TruncOp>(loc, i32Type, arg);
      } else if (argType.isIndex()) {
        // Index -> i32
        argVal = rewriter.create<arith::IndexCastOp>(loc, i32Type, arg);
      } else if (argType.isF32()) {
        // Bitcast float to i32
        argVal = rewriter.create<LLVM::BitcastOp>(loc, i32Type, arg);
      } else {
        // Fallback: try to convert via index
        argVal = rewriter.create<arith::IndexCastOp>(loc, i32Type, arg);
      }

      storeI32AtOffset(argVal, argOffset);
      argOffset += 4;
    }

    // Call vortex_launch_with_args
    Value totalSizeVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type, totalSize);
    rewriter.create<LLVM::CallOp>(
        loc, i32Type, "vortex_launch_with_args",
        ValueRange{kernelNamePtr, buffer, totalSizeVal});

    // Erase the original launch operation
    rewriter.eraseOp(launchOp);

    return success();
  }
};

/// Pass to convert gpu.launch_func to host runtime calls
struct ConvertGPULaunchToHostCallPass
    : public PassWrapper<ConvertGPULaunchToHostCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPULaunchToHostCallPass)

  StringRef getArgument() const override { return "convert-gpu-launch-to-host-call"; }

  StringRef getDescription() const override {
    return "Convert gpu.launch_func to host-side runtime calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
  }

  // Structure to hold extracted launch info
  struct LaunchInfo {
    StringRef kernelName;
    Value gridX, gridY, gridZ;
    Value blockX, blockY, blockZ;
    SmallVector<Value> operands;
    Operation *launchOp;  // Original launch op (for location)
    func::FuncOp parentFunc;
    Operation *insertionPoint;  // Where to insert the runtime call
    Region *asyncRegion;  // Region containing the launch (for cloning)
    SmallVector<Operation *> toErase;  // Operations to delete
  };

  // Check if a value is defined inside a given region (e.g., async.execute body)
  bool isDefinedInRegion(Value v, Region *region) {
    if (auto blockArg = v.dyn_cast<BlockArgument>()) {
      return region->isAncestor(blockArg.getOwner()->getParent());
    }
    if (auto defOp = v.getDefiningOp()) {
      return region->isAncestor(defOp->getParentRegion());
    }
    return false;
  }

  // Clone a value and its dependencies to a target location
  // Returns the cloned value
  Value cloneValueToLocation(Value v, OpBuilder &builder,
                             DenseMap<Value, Value> &cloneMap,
                             Region *sourceRegion) {
    // Check if already cloned
    if (auto it = cloneMap.find(v); it != cloneMap.end()) {
      return it->second;
    }

    // If not defined in the source region, use as-is
    if (!isDefinedInRegion(v, sourceRegion)) {
      cloneMap[v] = v;
      return v;
    }

    // Clone the defining operation
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      // Block argument - can't clone, use as-is
      cloneMap[v] = v;
      return v;
    }

    // First clone all operands
    SmallVector<Value> newOperands;
    for (Value operand : defOp->getOperands()) {
      newOperands.push_back(cloneValueToLocation(operand, builder, cloneMap, sourceRegion));
    }

    // Clone the operation
    Operation *clonedOp = builder.clone(*defOp);

    // Update operands of cloned op
    for (unsigned i = 0; i < newOperands.size(); ++i) {
      clonedOp->setOperand(i, newOperands[i]);
    }

    // Map the result
    Value clonedResult = clonedOp->getResult(0);
    cloneMap[v] = clonedResult;
    return clonedResult;
  }

  // Find the outermost parent that should be replaced (async.execute or the polygeist ops)
  Operation* findInsertionPointAndOpsToErase(gpu::LaunchFuncOp launchOp,
                                              SmallVector<Operation*> &toErase) {
    Operation *op = launchOp;
    Operation *insertionPoint = nullptr;

    // Walk up the parent chain looking for async.execute or the first func-level op
    while (op->getParentOp()) {
      Operation *parent = op->getParentOp();

      // If we hit a func, use the current op as insertion point
      if (isa<func::FuncOp>(parent)) {
        insertionPoint = op;
        break;
      }

      op = parent;
    }

    // Find all async/polygeist operations in the parent function to erase
    if (insertionPoint) {
      Operation *current = insertionPoint;
      // If insertion point is async.execute, also mark polygeist ops before it
      while (current != nullptr) {
        // Look for polygeist ops that produce async tokens (they precede async.execute)
        if (auto prevOp = current->getPrevNode()) {
          if (prevOp->getDialect() &&
              prevOp->getDialect()->getNamespace() == "polygeist") {
            toErase.push_back(prevOp);
            current = prevOp;
            continue;
          }
        }
        break;
      }
      toErase.push_back(insertionPoint);
    }

    return insertionPoint;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Get types
    auto i32Type = IntegerType::get(&getContext(), 32);
    auto i64Type = IntegerType::get(&getContext(), 64);
    auto i8Type = IntegerType::get(&getContext(), 8);
    auto ptrType = LLVM::LLVMPointerType::get(&getContext());

    // Declare external runtime functions at the start of the module
    OpBuilder moduleBuilder(&getContext());
    moduleBuilder.setInsertionPointToStart(module.getBody());

    auto getOrInsertFunc = [&](StringRef name, Type resultType,
                               ArrayRef<Type> argTypes) -> LLVM::LLVMFuncOp {
      if (auto existingFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return existingFunc;

      auto funcType = LLVM::LLVMFunctionType::get(resultType, argTypes);
      OpBuilder::InsertionGuard guard(moduleBuilder);
      moduleBuilder.setInsertionPointToStart(module.getBody());
      return moduleBuilder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
    };

    // Collect launch operations and their context
    SmallVector<LaunchInfo> launches;
    module.walk([&](gpu::LaunchFuncOp launchOp) {
      LaunchInfo info;
      info.launchOp = launchOp;
      info.kernelName = launchOp.getKernelName().getValue();
      info.gridX = launchOp.getGridSizeX();
      info.gridY = launchOp.getGridSizeY();
      info.gridZ = launchOp.getGridSizeZ();
      info.blockX = launchOp.getBlockSizeX();
      info.blockY = launchOp.getBlockSizeY();
      info.blockZ = launchOp.getBlockSizeZ();
      for (auto arg : launchOp.getKernelOperands()) {
        info.operands.push_back(arg);
      }
      info.parentFunc = launchOp->getParentOfType<func::FuncOp>();
      info.insertionPoint = findInsertionPointAndOpsToErase(launchOp, info.toErase);
      // Find the async region containing the launch (for cloning values out)
      info.asyncRegion = nullptr;
      Operation *parent = launchOp->getParentOp();
      while (parent) {
        if (parent->getDialect() &&
            parent->getDialect()->getNamespace() == "async") {
          info.asyncRegion = &parent->getRegion(0);
          break;
        }
        parent = parent->getParentOp();
      }
      launches.push_back(info);
    });

    if (launches.empty()) {
      goto cleanup;
    }

    getOrInsertFunc("hip_ptr_to_device_addr", i32Type, {ptrType});
    getOrInsertFunc("vortex_launch_with_args", i32Type, {ptrType, ptrType, i64Type});

    // Process each launch
    for (auto &info : launches) {
      if (!info.insertionPoint) continue;

      Location loc = info.launchOp->getLoc();
      OpBuilder builder(info.insertionPoint);

      // Map for cloned values
      DenseMap<Value, Value> cloneMap;

      // Helper to get a value, cloning it if necessary (if defined inside async region)
      auto getOrCloneValue = [&](Value v) -> Value {
        if (!info.asyncRegion) return v;
        return cloneValueToLocation(v, builder, cloneMap, info.asyncRegion);
      };

      // Create kernel name string constant
      std::string kernelNameStr = info.kernelName.str();
      kernelNameStr += ".vxbin";
      kernelNameStr.push_back('\0');

      std::string globalName = ("__vortex_kernel_name_" + info.kernelName).str();
      if (!module.lookupSymbol<LLVM::GlobalOp>(globalName)) {
        auto stringType = LLVM::LLVMArrayType::get(i8Type, kernelNameStr.size());
        OpBuilder::InsertionGuard guard(moduleBuilder);
        moduleBuilder.setInsertionPointToStart(module.getBody());
        moduleBuilder.create<LLVM::GlobalOp>(
            loc, stringType, /*isConstant=*/true, LLVM::Linkage::Internal,
            globalName, moduleBuilder.getStringAttr(kernelNameStr));
      }

      // Get pointer to kernel name
      Value kernelNamePtr = builder.create<LLVM::AddressOfOp>(loc, ptrType, globalName);

      // Get grid/block dimensions, cloning from async region if needed
      Value gridXVal = getOrCloneValue(info.gridX);
      Value gridYVal = getOrCloneValue(info.gridY);
      Value gridZVal = getOrCloneValue(info.gridZ);
      Value blockXVal = getOrCloneValue(info.blockX);
      Value blockYVal = getOrCloneValue(info.blockY);
      Value blockZVal = getOrCloneValue(info.blockZ);

      // Convert grid/block dimensions to i32
      auto toI32 = [&](Value v) -> Value {
        if (v.getType().isIndex()) {
          return builder.create<arith::IndexCastOp>(loc, i32Type, v);
        }
        if (v.getType() != i32Type && v.getType().isa<IntegerType>()) {
          return builder.create<LLVM::TruncOp>(loc, i32Type, v);
        }
        return v;
      };

      Value gridX = toI32(gridXVal);
      Value gridY = toI32(gridYVal);
      Value gridZ = toI32(gridZVal);
      Value blockX = toI32(blockXVal);
      Value blockY = toI32(blockYVal);
      Value blockZ = toI32(blockZVal);

      // Calculate buffer size
      unsigned numArgs = info.operands.size();
      unsigned headerSize = 6 * 4;  // 24 bytes for grid+block
      unsigned argsSize = numArgs * 4;
      unsigned totalSize = headerSize + argsSize;

      // Allocate buffer on stack
      Value one = builder.create<LLVM::ConstantOp>(loc, i64Type, 1);
      auto bufferType = LLVM::LLVMArrayType::get(i8Type, totalSize);
      Value buffer = builder.create<LLVM::AllocaOp>(loc, ptrType, bufferType, one);

      // Helper to store i32 at offset
      auto storeI32AtOffset = [&](Value val, unsigned offset) {
        Value offsetVal = builder.create<LLVM::ConstantOp>(loc, i64Type, offset);
        Value ptr = builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, buffer,
                                                 ValueRange{offsetVal});
        builder.create<LLVM::StoreOp>(loc, val, ptr);
      };

      // Store grid and block dimensions
      storeI32AtOffset(gridX, 0);
      storeI32AtOffset(gridY, 4);
      storeI32AtOffset(gridZ, 8);
      storeI32AtOffset(blockX, 12);
      storeI32AtOffset(blockY, 16);
      storeI32AtOffset(blockZ, 20);

      // Convert and store kernel arguments
      unsigned argOffset = 24;
      for (auto origArg : info.operands) {
        // Clone argument if needed
        Value arg = getOrCloneValue(origArg);
        Type argType = arg.getType();
        Value argVal;

        if (argType.isa<MemRefType>()) {
          // Pointer argument - convert to device address
          Value basePtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, arg);
          Value ptrAsInt = builder.create<arith::IndexCastOp>(loc, i64Type, basePtr);
          Value ptrVal = builder.create<LLVM::IntToPtrOp>(loc, ptrType, ptrAsInt);
          argVal = builder.create<LLVM::CallOp>(
              loc, i32Type, "hip_ptr_to_device_addr", ValueRange{ptrVal}).getResult();
        } else if (argType.isInteger(32)) {
          argVal = arg;
        } else if (argType.isInteger(64)) {
          argVal = builder.create<LLVM::TruncOp>(loc, i32Type, arg);
        } else if (argType.isIndex()) {
          argVal = builder.create<arith::IndexCastOp>(loc, i32Type, arg);
        } else if (argType.isF32()) {
          argVal = builder.create<LLVM::BitcastOp>(loc, i32Type, arg);
        } else {
          argVal = builder.create<arith::IndexCastOp>(loc, i32Type, arg);
        }

        storeI32AtOffset(argVal, argOffset);
        argOffset += 4;
      }

      // Call vortex_launch_with_args
      Value totalSizeVal = builder.create<LLVM::ConstantOp>(loc, i64Type, totalSize);
      builder.create<LLVM::CallOp>(
          loc, i32Type, "vortex_launch_with_args",
          ValueRange{kernelNamePtr, buffer, totalSizeVal});

      // Erase the old operations
      // Dependencies: async.execute uses stream2token, stream2token uses pointer2memref
      // Need to handle the circular dependency by dropping uses first
      for (auto *op : info.toErase) {
        // Drop all uses of this op's results
        for (auto result : op->getResults()) {
          result.dropAllUses();
        }
      }
      // Now erase in any order
      for (auto *op : info.toErase) {
        op->erase();
      }
    }

cleanup:
    // Remove gpu.module operations (device code handled separately)
    SmallVector<gpu::GPUModuleOp> gpuModules;
    module.walk([&](gpu::GPUModuleOp gpuModule) {
      gpuModules.push_back(gpuModule);
    });
    for (auto gpuModule : gpuModules) {
      gpuModule.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createConvertGPULaunchToHostCallPass() {
  return std::make_unique<ConvertGPULaunchToHostCallPass>();
}

} // namespace polygeist
} // namespace mlir
