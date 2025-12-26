#ifndef POLYGEIST_PASSES
#define POLYGEIST_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def PolyhedralOpt : Pass<"polyhedral-opt"> {
  let summary = "Optimize affine regions with pluto";
  let constructor = "mlir::polygeist::createPolyhedralOptPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "affine::AffineDialect",
    // This vector dialect is needed because lower-affine needs it. We launch
    // a pass manager to use lower-affine from this pass itself, and that
    // causes the multi-threaded mlir context to try to load the vector dialect
    // which is unsupported. Instead we preload it here.
    "vector::VectorDialect",
  ];
}

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::replaceAffineCFGPass()";
}

def PolygeistMem2Reg : Pass<"polygeist-mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::createPolygeistMem2RegPass()";
}

def SCFParallelLoopUnroll : Pass<"scf-parallel-loop-unroll"> {
  let summary = "Unroll and interleave scf parallel loops";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
  let constructor = "mlir::polygeist::createSCFParallelLoopUnrollPass()";
  let options = [
  Option<"unrollFactor", "unrollFactor", "int", /*default=*/"2", "Unroll factor">
  ];
}

def CollectKernelStatistics : Pass<"collect-kernel-statistics", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createCollectKernelStatisticsPass()";
}

def LowerAlternatives : Pass<"lower-alternatives", "mlir::ModuleOp"> {
  let summary = "Lower alternatives if in opt mode";
  let dependentDialects = [];
  let constructor = "mlir::polygeist::createLowerAlternativesPass()";
}

def ConvertCudaRTtoCPU : Pass<"convert-cudart-to-cpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to cpu versions";
  let dependentDialects = [
    "memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect",
    "cf::ControlFlowDialect",
  ];
  let constructor = "mlir::polygeist::createConvertCudaRTtoCPUPass()";
}

def FixGPUFunc : Pass<"fix-gpu-func", "mlir::gpu::GPUModuleOp"> {
  let summary = "Fix nested calls to gpu functions we generate in the frontend";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createFixGPUFuncPass()";
}

def ConvertCudaRTtoGPU : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ConvertCudaRTtoHipRT : Pass<"convert-cudart-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower cudart functions to generic gpu versions";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let constructor = "mlir::polygeist::createConvertCudaRTtoGPUPass()";
}

def ParallelLower : Pass<"parallel-lower", "mlir::ModuleOp"> {
  let summary = "Lower gpu launch op to parallel ops";
  let dependentDialects = [
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
  ];
  let constructor = "mlir::polygeist::createParallelLowerPass()";
}

def AffineReduction : Pass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::polygeist::detectReductionPass()";
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createCPUifyPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def ConvertParallelToGPU1 : Pass<"convert-parallel-to-gpu1"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass1()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
  let options = [
  Option<"arch", "arch", "std::string", /*default=*/"\"sm_60\"", "Target GPU architecture">
  ];
}

def ConvertParallelToGPU2 : Pass<"convert-parallel-to-gpu2"> {
  let summary = "Convert parallel loops to gpu";
  let constructor = "mlir::polygeist::createConvertParallelToGPUPass2()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
}

def ConvertToOpaquePtrPass : Pass<"convert-to-opaque-ptr"> {
  let summary = "Convert typed llvm pointers to opaque";
  let constructor = "mlir::polygeist::createConvertToOpaquePtrPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def MergeGPUModulesPass : Pass<"merge-gpu-modules", "mlir::ModuleOp"> {
  let summary = "Merge all gpu modules into one";
  let constructor = "mlir::polygeist::createMergeGPUModulesPass()";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
}

def InnerSerialization : Pass<"inner-serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createInnerSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def Serialization : Pass<"serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect", "func::FuncDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
  ];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::polygeist::createCanonicalizeForPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "math::MathDialect",
  ];
}

def ForBreakToWhile : Pass<"for-break-to-while"> {
  let summary = "Rewrite scf.for(scf.if) to scf.while";
  let constructor = "mlir::polygeist::createForBreakToWhilePass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
  ];
}

def ParallelLICM : Pass<"parallel-licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
  let constructor = "mlir::polygeist::createParallelLICMPass()";
}

def OpenMPOptPass : Pass<"openmp-opt"> {
  let summary = "Optimize OpenMP";
  let constructor = "mlir::polygeist::createOpenMPOptPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "omp::OpenMPDialect",
    "LLVM::LLVMDialect",
  ];
}

def PolygeistCanonicalize : Pass<"canonicalize-polygeist"> {
  let constructor = "mlir::polygeist::createPolygeistCanonicalizePass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
    "polygeist::PolygeistDialect",
  ];
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Perform control flow optimizations to the region tree">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Max. iterations between applying patterns / simplifying regions">,
    Option<"maxNumRewrites", "max-num-rewrites", "int64_t", /*default=*/"-1",
           "Max. number of pattern rewrites within an iteration">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
  ] # RewritePassUtils.options;
}

def LoopRestructure : Pass<"loop-restructure"> {
  let constructor = "mlir::polygeist::createLoopRestructurePass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "polygeist::PolygeistDialect",
  ];
}

def RemoveTrivialUse : Pass<"trivialuse"> {
  let constructor = "mlir::polygeist::createRemoveTrivialUsePass()";
}

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::polygeist::createConvertPolygeistToLLVMPass()";
  let dependentDialects = [
    "polygeist::PolygeistDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
  ];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useCStyleMemRef", "use-c-style-memref", "bool",
           /*default=*/"true",
           "Use C-style nested-array lowering of memref instead of "
           "the default MLIR descriptor structure">
  ];
}

def SinkIndexCastsIntoGPULaunch : Pass<"sink-index-casts-into-gpu-launch"> {
  let summary = "Sink arith.index_cast operations into gpu.launch regions";
  let description = [{
    This pass sinks arith.index_cast operations into gpu.launch regions to
    reduce the number of kernel parameters. When a value is captured by a
    gpu.launch and then index_cast'd for use in an scf.for loop bound,
    both the original value and the cast result become separate kernel parameters.

    This pass identifies such cases and sinks the index_cast into the launch
    region, so only the original value needs to be captured.

    Example:
      Before:
        %count = ...  // i32
        %bound = arith.index_cast %count : i32 to index
        gpu.launch ... {
          scf.for %i = 0 to %bound step 1 { ... }
        }

      After:
        %count = ...  // i32
        gpu.launch ... {
          %bound = arith.index_cast %count : i32 to index
          scf.for %i = 0 to %bound step 1 { ... }
        }

    This pass should run BEFORE gpu-kernel-outlining.
  }];
  let constructor = "mlir::polygeist::createSinkIndexCastsIntoGPULaunchPass()";
  let dependentDialects = [
    "gpu::GPUDialect",
    "arith::ArithDialect",
  ];
}

def SinkGpuDimsIntoLaunch : Pass<"sink-gpu-dims-into-launch"> {
  let summary = "Sink GPU dimension operations into launch regions";
  let description = [{
    This pass sinks gpu.block_dim and gpu.grid_dim operations (and their
    dependent pure computations) into gpu.launch regions before kernel
    outlining. This eliminates synthetic kernel arguments for dimension values.

    Values that would become synthetic kernel arguments are instead cloned
    into the launch body, where they become local computations using the
    gpu.block_dim and gpu.grid_dim operations available inside the launch.

    This pass should run BEFORE gpu-kernel-outlining.

    Example:
      Before:
        %bd = gpu.block_dim x
        %total = arith.muli %gd, %bd : index
        gpu.launch ... {
          // uses %total (captured as synthetic arg)
        }

      After:
        gpu.launch ... {
          %bd = gpu.block_dim x
          %total = arith.muli %gd, %bd : index
          // uses %total (local computation)
        }
  }];
  let constructor = "mlir::polygeist::createSinkGpuDimsIntoLaunchPass()";
  let dependentDialects = [
    "gpu::GPUDialect",
    "arith::ArithDialect",
  ];
}

def ReorderGPUKernelArgs : Pass<"reorder-gpu-kernel-args", "ModuleOp"> {
  let summary = "Reorder GPU kernel arguments to match original HIP kernel signature";
  let description = [{
    Polygeist reorders GPU kernel arguments, placing scalars before pointers while
    preserving relative order within each group. This transformation optimizes
    memory access patterns but changes the ABI.

    This pass undoes that reordering, restoring arguments to their original order
    as declared in the HIP kernel signature. This is necessary for interoperability
    with host code that expects arguments in the original order.

    The pass:
    1. Extracts original argument order from host wrapper functions
    2. Creates new GPU functions with arguments in original order
    3. Updates all gpu.launch_func call sites to pass arguments in the new order
    4. Replaces the original GPU functions with the reordered versions

    This pass should run BEFORE convert-gpu-to-vortex.
  }];
  let constructor = "mlir::polygeist::createReorderGPUKernelArgsPass()";
  let dependentDialects = [
    "gpu::GPUDialect",
    "func::FuncDialect",
  ];
}

def ConvertGPUToVortex : Pass<"convert-gpu-to-vortex", "ModuleOp"> {
  let summary = "Lower GPU dialect operations to Vortex RISC-V intrinsics";
  let description = [{
    This pass converts GPU dialect operations to LLVM dialect with Vortex-specific
    intrinsics. It lowers operations like gpu.thread_id, gpu.block_id to RISC-V
    CSR reads via inline assembly, preparing the code for the Vortex GPGPU backend.

    Example:
      %tid = gpu.thread_id x
    becomes:
      %tid = llvm.inline_asm "csrr $0, 0xCC0" : () -> i32

    The pass automatically emits JSON metadata files describing kernel argument
    layouts for runtime argument marshaling. Files are written to the current
    working directory as <kernel_name>.meta.json.
  }];
  let constructor = "mlir::polygeist::createConvertGPUToVortexPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "gpu::GPUDialect",
  ];
}

def GenerateVortexMain : Pass<"generate-vortex-main", "ModuleOp"> {
  let summary = "Generate Vortex main() wrapper for kernel execution";
  let description = [{
    This pass generates the Vortex-specific main() entry point and kernel_body
    wrapper function. It should run AFTER gpu-to-llvm lowering has converted
    gpu.func to llvm.func.

    The generated code:
    1. main() function that:
       - Reads kernel args from VX_CSR_MSCRATCH (0x340) via inline assembly
       - Calls vx_spawn_threads() with grid dimensions and kernel callback
    2. kernel_body() wrapper that:
       - Takes void* args pointer
       - Unpacks individual arguments from the struct
       - Calls the original lowered kernel function

    This matches the Vortex kernel execution model where kernels are launched
    via vx_spawn_threads() with a callback function.
  }];
  let constructor = "mlir::polygeist::createGenerateVortexMainPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
}

def InsertVortexDivergence : Pass<"insert-vortex-divergence", "ModuleOp"> {
  let summary = "Insert vx_split/vx_join around divergent branches for Vortex";
  let description = [{
    This pass handles SIMT divergence for the Vortex GPGPU. Unlike NVIDIA GPUs
    which handle divergence in hardware, Vortex requires explicit vx_split()
    and vx_join() calls around divergent branches.

    The pass:
    1. Performs divergence analysis to find values dependent on thread ID
       (values derived from vx_get_threadIdx calls)
    2. Identifies conditional branches with divergent conditions
    3. Inserts vx_split_abi(condition) before each divergent branch
    4. Inserts vx_join_abi(stack_ptr) at the convergence point (post-dominator)

    This should run AFTER convert-gpu-to-vortex (which lowers gpu.thread_id to
    vx_get_threadIdx calls) and BEFORE generate-vortex-main.

    Example transformation:
      // Before:
      cf.cond_br %divergent_cond, ^true, ^false
      ^true: ...
      ^false: ...
      ^merge: ...

      // After:
      %sp = llvm.call @vx_split_abi(%divergent_cond)
      cf.cond_br %divergent_cond, ^true, ^false
      ^true: ...
      ^false: ...
      ^merge:
        llvm.call @vx_join_abi(%sp)
        ...
  }];
  let constructor = "mlir::polygeist::createInsertVortexDivergencePass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
  ];
}

def ConvertGPULaunchToHostCall : Pass<"convert-gpu-launch-to-host-call", "ModuleOp"> {
  let summary = "Convert gpu.launch_func to host-side runtime calls";
  let description = [{
    This pass converts gpu.launch_func operations in HOST functions to calls
    to the Vortex HIP runtime. This is used to compile host launch wrappers
    as a native library that can be linked with the host application.

    The pass transforms:
      gpu.launch_func @kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
        args(%ptr0 : memref<?xf32>, %ptr1 : memref<?xf32>, %n : i32)

    Into:
      %addr0 = llvm.call @hip_ptr_to_device_addr(%ptr0)
      %addr1 = llvm.call @hip_ptr_to_device_addr(%ptr1)
      // Build Vortex args buffer: grid + block + [addr0, addr1, n]
      llvm.call @vortex_launch_with_args("kernel", %args_buffer, %size)

    This allows compiling the launch wrapper to x86_64 host code that calls
    the HIP runtime, instead of generating C header stubs.

    Note: This pass should be used for HOST compilation, not device compilation.
    For device compilation, use --convert-gpu-to-vortex instead.
  }];
  let constructor = "mlir::polygeist::createConvertGPULaunchToHostCallPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
  ];
}

def GenerateVortexWrappers : Pass<"generate-vortex-wrappers", "ModuleOp"> {
  let summary = "Generate launch wrapper functions for Vortex kernels";
  let description = [{
    This pass runs AFTER kernel outlining and generates launch wrapper functions
    that call gpu.launch_func with the correct argument order.

    The wrappers:
    - Accept kernel arguments in the original HIP source order
    - Accept grid and block dimensions as additional index parameters
    - Call gpu.launch_func to invoke the outlined kernel
    - Are marked for host compilation (not device)

    The generated wrapper function has the signature:
      func @__vortex_launch_<kernel>(kernel_args..., gridX, gridY, gridZ, blockX, blockY, blockZ)

    This pass should run:
    - AFTER kernel outlining (so gpu.launch_func ops exist)
    - BEFORE convert-gpu-launch-to-host-call (which lowers the wrapper to runtime calls)

    Example:
      // Input: gpu.launch_func @kernel_module::@my_kernel blocks in (...) threads in (...) args(...)
      // Output: Additional wrapper function:
      // func @__vortex_launch_my_kernel(..., gridX, gridY, gridZ, blockX, blockY, blockZ) {
      //   gpu.launch_func @kernel_module::@my_kernel ...
      //   return
      // }
  }];
  let constructor = "mlir::polygeist::createGenerateVortexWrappersPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
  ];
}

def StripHostOnlyFunctions : Pass<"strip-host-only-functions", "ModuleOp"> {
  let summary = "Remove functions marked with polygeist.host_only_func attribute";
  let description = [{
    This pass removes host-only functions from the module before lowering to
    device code. Functions with the polygeist.host_only_func="1" attribute
    are erased.

    This is used in conjunction with cgeist's --emit-host-functions flag, which
    emits both host and device functions but marks host-only functions with
    the polygeist.host_only_func attribute. This pass then strips out the
    host-only functions before the device compilation pipeline.

    Example usage:
      cgeist input.hip --emit-host-functions ... -o gpu.mlir
      polygeist-opt gpu.mlir --convert-gpu-to-vortex --strip-host-only-functions ...

    The pass handles both func.func and llvm.func operations, so it can be
    run before or after LLVM lowering.
  }];
  let constructor = "mlir::polygeist::createStripHostOnlyFunctionsPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
  ];
}

#endif // POLYGEIST_PASSES
