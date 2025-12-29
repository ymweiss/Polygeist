//===- ConvertGPULaunchToHostCall.cpp - Convert gpu.launch_func to host calls -===//
//
// This pass converts gpu.launch_func operations to calls to the Vortex HIP
// runtime. This allows compiling host launch wrappers as a native library.
//
// The argument buffer layout must match what the device expects:
// - Header: 6 x i32 for grid/block dimensions (always 24 bytes)
// - Args: Each argument sized according to device type:
//   - Pointers: pointerWidth/8 bytes (4 for RV32, 8 for RV64)
//   - i32/f32: 4 bytes
//   - i64/f64: 8 bytes
//   - i16: 2 bytes
//   - i8: 1 byte
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

/// Get the size in bytes for a type on the device
/// pointerWidth: 32 for RV32, 64 for RV64
static unsigned getDeviceTypeSize(Type type, unsigned pointerWidth) {
  // Pointer types (memref, ptr)
  if (type.isa<MemRefType>() || type.isa<LLVM::LLVMPointerType>())
    return pointerWidth / 8;  // 4 for RV32, 8 for RV64

  // Integer types
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned bits = intType.getWidth();
    if (bits <= 8) return 1;
    if (bits <= 16) return 2;
    if (bits <= 32) return 4;
    return 8;  // i64 and larger
  }

  // Floating point types
  if (type.isF16() || type.isBF16()) return 2;
  if (type.isF32()) return 4;
  if (type.isF64()) return 8;

  // Index type - same as pointer width
  if (type.isIndex())
    return pointerWidth / 8;

  // Default to pointer size for unknown types
  return pointerWidth / 8;
}

/// Pass to convert gpu.launch_func to host runtime calls
/// Supports RV32 (pointerWidth=32) and RV64 (pointerWidth=64) targets.
struct ConvertGPULaunchToHostCallPass
    : public PassWrapper<ConvertGPULaunchToHostCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPULaunchToHostCallPass)

  ConvertGPULaunchToHostCallPass() = default;
  ConvertGPULaunchToHostCallPass(const ConvertGPULaunchToHostCallPass &other)
      : PassWrapper(other), pointerWidthValue(other.pointerWidthValue) {}
  explicit ConvertGPULaunchToHostCallPass(unsigned ptrWidth)
      : pointerWidthValue(ptrWidth) {}

  // Pointer width value (32 or 64), set by command-line option
  unsigned pointerWidthValue = 32;

  // Command-line option for pointer width
  Pass::Option<unsigned> pointerWidthOpt{*this, "pointer-width",
      llvm::cl::desc("Target device pointer width in bits (32 for RV32, 64 for RV64)"),
      llvm::cl::init(32)};

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

    // Get effective pointer width (from option or constructor)
    unsigned ptrWidth = pointerWidthOpt.hasValue() ? pointerWidthOpt.getValue()
                                                   : pointerWidthValue;

    // Get types based on pointer width
    auto i8Type = IntegerType::get(&getContext(), 8);
    auto i16Type = IntegerType::get(&getContext(), 16);
    auto i32Type = IntegerType::get(&getContext(), 32);
    auto i64Type = IntegerType::get(&getContext(), 64);
    auto ptrType = LLVM::LLVMPointerType::get(&getContext());

    // Device address type: i32 for RV32, i64 for RV64
    Type deviceAddrType = (ptrWidth == 64) ? (Type)i64Type : (Type)i32Type;

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

    // hip_ptr_to_device_addr returns device address (i32 for RV32, i64 for RV64)
    getOrInsertFunc("hip_ptr_to_device_addr", deviceAddrType, {ptrType});
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

      // Calculate buffer size based on actual type sizes
      unsigned headerSize = 6 * 4;  // 24 bytes for grid+block (always i32)
      unsigned argsSize = 0;
      for (auto origArg : info.operands) {
        argsSize += getDeviceTypeSize(origArg.getType(), ptrWidth);
      }
      unsigned totalSize = headerSize + argsSize;

      // Allocate buffer on stack
      Value one = builder.create<LLVM::ConstantOp>(loc, i64Type, 1);
      auto bufferType = LLVM::LLVMArrayType::get(i8Type, totalSize);
      Value buffer = builder.create<LLVM::AllocaOp>(loc, ptrType, bufferType, one);

      // Helper to store a value at offset
      auto storeAtOffset = [&](Value val, unsigned offset) {
        Value offsetVal = builder.create<LLVM::ConstantOp>(loc, i64Type, offset);
        Value ptr = builder.create<LLVM::GEPOp>(loc, ptrType, i8Type, buffer,
                                                 ValueRange{offsetVal});
        builder.create<LLVM::StoreOp>(loc, val, ptr);
      };

      // Store grid and block dimensions (always i32)
      storeAtOffset(gridX, 0);
      storeAtOffset(gridY, 4);
      storeAtOffset(gridZ, 8);
      storeAtOffset(blockX, 12);
      storeAtOffset(blockY, 16);
      storeAtOffset(blockZ, 20);

      // Convert and store kernel arguments with proper sizes
      unsigned argOffset = 24;
      for (auto origArg : info.operands) {
        // Clone argument if needed
        Value arg = getOrCloneValue(origArg);
        Type argType = arg.getType();
        Value argVal;
        unsigned argSize = getDeviceTypeSize(argType, ptrWidth);

        if (argType.isa<MemRefType>()) {
          // Pointer argument - convert to device address
          Value basePtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, arg);
          Value ptrAsInt = builder.create<arith::IndexCastOp>(loc, i64Type, basePtr);
          Value ptrVal = builder.create<LLVM::IntToPtrOp>(loc, ptrType, ptrAsInt);
          // hip_ptr_to_device_addr returns deviceAddrType (i32 or i64)
          argVal = builder.create<LLVM::CallOp>(
              loc, deviceAddrType, "hip_ptr_to_device_addr", ValueRange{ptrVal}).getResult();
        } else if (argType.isInteger(64)) {
          // i64 - store as 8 bytes
          argVal = arg;
        } else if (argType.isInteger(32)) {
          argVal = arg;
        } else if (argType.isInteger(16)) {
          // i16 - store as 2 bytes
          argVal = arg;
        } else if (argType.isInteger(8)) {
          // i8 - store as 1 byte
          argVal = arg;
        } else if (argType.isIndex()) {
          // Index type - convert to device address size
          if (ptrWidth == 64) {
            argVal = builder.create<arith::IndexCastOp>(loc, i64Type, arg);
          } else {
            argVal = builder.create<arith::IndexCastOp>(loc, i32Type, arg);
          }
        } else if (argType.isF64()) {
          // f64 - store as 8 bytes (bitcast to i64)
          argVal = builder.create<LLVM::BitcastOp>(loc, i64Type, arg);
        } else if (argType.isF32()) {
          // f32 - bitcast to i32
          argVal = builder.create<LLVM::BitcastOp>(loc, i32Type, arg);
        } else if (argType.isF16() || argType.isBF16()) {
          // f16/bf16 - bitcast to i16
          argVal = builder.create<LLVM::BitcastOp>(loc, i16Type, arg);
        } else {
          // Fallback: try to convert to device address size
          if (ptrWidth == 64) {
            argVal = builder.create<arith::IndexCastOp>(loc, i64Type, arg);
          } else {
            argVal = builder.create<arith::IndexCastOp>(loc, i32Type, arg);
          }
        }

        storeAtOffset(argVal, argOffset);
        argOffset += argSize;
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
