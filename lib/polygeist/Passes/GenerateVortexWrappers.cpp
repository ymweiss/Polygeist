//===- GenerateVortexWrappers.cpp - Generate launch wrappers for Vortex -===//
//
// This pass runs AFTER kernel outlining and generates launch wrapper functions
// that call gpu.launch_func with the correct argument order.
//
// The wrappers preserve the original kernel argument order and can be compiled
// as host code to call the Vortex runtime.
//
//===----------------------------------------------------------------------===//

#include "polygeist/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {

/// Pass to generate launch wrapper functions for Vortex kernels
/// Runs AFTER kernel outlining to generate wrappers with gpu.launch_func
struct GenerateVortexWrappersPass
    : public PassWrapper<GenerateVortexWrappersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateVortexWrappersPass)

  StringRef getArgument() const override { return "generate-vortex-wrappers"; }

  StringRef getDescription() const override {
    return "Generate launch wrapper functions for Vortex kernels (post-outlining)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
  }

  // Information about a kernel to generate wrapper for
  struct KernelInfo {
    StringRef kernelName;
    StringRef moduleName;
    SmallVector<Type> argTypes;
    ArrayAttr kernelArgsAttr;  // vortex.kernel_args metadata
    Attribute kernelArgsSizeAttr;
  };

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // Collect kernel information from gpu.launch_func operations
    llvm::StringMap<KernelInfo> kernelInfos;

    module.walk([&](gpu::LaunchFuncOp launchOp) {
      // Extract kernel name and GPU module name
      StringRef kernelName = launchOp.getKernelName().getValue();
      StringRef moduleName = launchOp.getKernelModuleName().getValue();

      // Skip if we've already seen this kernel
      if (kernelInfos.count(kernelName))
        return;

      // Get argument types from the launch_func operands
      KernelInfo info;
      info.kernelName = kernelName;
      info.moduleName = moduleName;

      // Get kernel argument types
      for (Value arg : launchOp.getKernelOperands()) {
        info.argTypes.push_back(arg.getType());
      }

      // Get vortex.kernel_args metadata if present
      if (auto attr = launchOp->getAttr("vortex.kernel_args")) {
        info.kernelArgsAttr = attr.cast<ArrayAttr>();
      }
      if (auto attr = launchOp->getAttr("vortex.kernel_args_size")) {
        info.kernelArgsSizeAttr = attr;
      }

      // Try to get metadata from the kernel function itself
      if (auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(moduleName)) {
        if (auto gpuFunc = gpuModule.lookupSymbol<gpu::GPUFuncOp>(kernelName)) {
          if (!info.kernelArgsAttr) {
            if (auto attr = gpuFunc->getAttr("vortex.kernel_args")) {
              info.kernelArgsAttr = attr.cast<ArrayAttr>();
            }
          }
          if (!info.kernelArgsSizeAttr) {
            info.kernelArgsSizeAttr = gpuFunc->getAttr("vortex.kernel_args_size");
          }
        }
      }

      kernelInfos[kernelName] = info;
    });

    if (kernelInfos.empty()) {
      llvm::errs() << "[GenerateVortexWrappers] No gpu.launch_func found\n";
      return;
    }

    llvm::errs() << "[GenerateVortexWrappers] Found " << kernelInfos.size()
                 << " kernel(s)\n";

    // Generate wrapper for each kernel
    for (auto &entry : kernelInfos) {
      KernelInfo &info = entry.second;
      generateWrapper(module, builder, info);
    }
  }

  void generateWrapper(ModuleOp module, OpBuilder &builder, KernelInfo &info) {
    MLIRContext *context = module.getContext();
    Location loc = module.getLoc();

    // Generate wrapper function name: __vortex_launch_<kernel>
    std::string wrapperName = ("__vortex_launch_" + info.kernelName).str();

    // Check if wrapper already exists
    if (module.lookupSymbol(wrapperName)) {
      llvm::errs() << "[GenerateVortexWrappers] Wrapper " << wrapperName
                   << " already exists, skipping\n";
      return;
    }

    // Build function signature:
    // Args: kernel args + gridX, gridY, gridZ, blockX, blockY, blockZ (all index)
    SmallVector<Type> argTypes;
    argTypes.append(info.argTypes.begin(), info.argTypes.end());

    // Add grid and block dimensions as index types
    auto indexType = IndexType::get(context);
    for (int i = 0; i < 6; ++i) {
      argTypes.push_back(indexType);
    }

    // Create function type (void return)
    auto funcType = FunctionType::get(context, argTypes, {});

    // Create wrapper function at module level
    builder.setInsertionPointToEnd(module.getBody());
    auto wrapperFunc = builder.create<func::FuncOp>(loc, wrapperName, funcType);

    // Set external linkage for host compilation
    wrapperFunc->setAttr("llvm.linkage",
                         LLVM::LinkageAttr::get(context, LLVM::Linkage::External));

    // Copy kernel args metadata to wrapper
    if (info.kernelArgsAttr) {
      wrapperFunc->setAttr("vortex.kernel_args", info.kernelArgsAttr);
    }
    if (info.kernelArgsSizeAttr) {
      wrapperFunc->setAttr("vortex.kernel_args_size", info.kernelArgsSizeAttr);
    }

    // Mark as host function (not for device compilation)
    wrapperFunc->setAttr("vortex.host_wrapper", builder.getUnitAttr());

    // Create entry block
    Block *entryBlock = wrapperFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Get function arguments
    auto funcArgs = entryBlock->getArguments();
    unsigned numKernelArgs = info.argTypes.size();

    // Kernel arguments
    SmallVector<Value> kernelArgs;
    for (unsigned i = 0; i < numKernelArgs; ++i) {
      kernelArgs.push_back(funcArgs[i]);
    }

    // Grid and block dimensions
    Value gridX = funcArgs[numKernelArgs + 0];
    Value gridY = funcArgs[numKernelArgs + 1];
    Value gridZ = funcArgs[numKernelArgs + 2];
    Value blockX = funcArgs[numKernelArgs + 3];
    Value blockY = funcArgs[numKernelArgs + 4];
    Value blockZ = funcArgs[numKernelArgs + 5];

    // Create gpu.launch_func operation
    // This will be converted to runtime calls by ConvertGPULaunchToHostCall
    auto kernelSymbol = SymbolRefAttr::get(
        context, info.moduleName,
        {SymbolRefAttr::get(context, info.kernelName)});

    builder.create<gpu::LaunchFuncOp>(
        loc,
        kernelSymbol,
        gpu::KernelDim3{gridX, gridY, gridZ},
        gpu::KernelDim3{blockX, blockY, blockZ},
        /*dynamicSharedMemorySize=*/Value(),
        kernelArgs);

    // Return from wrapper
    builder.create<func::ReturnOp>(loc);

    llvm::errs() << "[GenerateVortexWrappers] Created wrapper: " << wrapperName
                 << " with " << numKernelArgs << " kernel args\n";
  }
};

} // namespace

namespace mlir {
namespace polygeist {

std::unique_ptr<Pass> createGenerateVortexWrappersPass() {
  return std::make_unique<GenerateVortexWrappersPass>();
}

} // namespace polygeist
} // namespace mlir
