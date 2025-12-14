# HIP Kernel Metadata Flow in Polygeist

This document explains how kernel argument metadata flows from HIP source through cgeist to the Vortex runtime.

## Summary of Changes

The following modifications enable proper HIP-to-Vortex compilation:

1. **Phase 1-2**: AST-based kernel discovery in `clang-mlir.cc` attaches `vortex.kernel_args` metadata
2. **Phase 3**: `CGCall.cc` copies metadata to `gpu.launch` ops; `KernelOutlining.cpp` propagates to `gpu.func`
3. **Kernel Naming**: `KernelOutlining.cpp` uses `vortex.kernel_name` for kernel naming
4. **MergeGPUModules Fix**: Fixed collision detection to preserve original kernel names

## Overview

When compiling HIP code for Vortex, we need to:
1. **Device side**: Generate a `.vxbin` binary with the correct kernel entry point
2. **Host side**: Ensure kernel launches use the correct argument marshalling

The key challenge is that the host and device compilations are separate, but must agree on:
- Kernel name (for lookup at runtime)
- Argument layout (offsets, sizes, alignment)

## Compilation Flow

```
HIP Source (.cu/.hip)
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼                                         ▼
   Device Compilation                        Host Compilation
   (via cgeist → MLIR)                       (via GCC/clang)
        │                                         │
        ▼                                         │
   vortex.kernel_args metadata               hipLaunchKernelGGL macro
   attached to gpu.func                      packs args with std::make_tuple
        │                                         │
        ▼                                         │
   ConvertGPUToVortex pass                        │
   uses metadata for marshalling                  │
        │                                         │
        ▼                                         ▼
   kernel.vxbin                              libhip_vortex.so
   (named by vortex.kernel_name)             (calls hipLaunchKernelByName)
        │                                         │
        └─────────────────────────────────────────┘
                            │
                            ▼
                    Runtime Execution
                    (vx_start + kernel binary)
```

## Phase 1: AST-Based Kernel Discovery

**Purpose**: Extract kernel argument information from the Clang AST during parsing.

**Implementation**: `Polygeist/tools/cgeist/Lib/clang-mlir.cc` (GetOrCreateMLIRFunction)

When cgeist encounters a `__global__` function:
1. Check for `CUDAGlobalAttr` on the FunctionDecl
2. Iterate over parameters using `FunctionDecl::parameters()`
3. For each parameter, compute:
   - **name**: `param->getNameAsString()`
   - **type**: `param->getType().getAsString()`
   - **size**: `ASTContext::getTypeSizeInChars()`
   - **alignment**: `ASTContext::getTypeAlignInChars()`
   - **is_pointer**: `QualType::isPointerType()`
   - **offset**: Accumulated with alignment padding

**Why this approach**: The AST has complete type information including sizes and alignments that match what the device will use. This is more reliable than trying to infer sizes from MLIR types.

## Phase 2: MLIR Attribute Attachment

**Purpose**: Store kernel metadata as MLIR attributes so it survives transformations.

**Implementation**: `clang-mlir.cc` lines ~5240-5260

Attributes attached to the kernel `func.func`:
```mlir
func.func @kernel_name(...) attributes {
  vortex.kernel_args = [
    {name = "arg0", type = "float *", offset = 0, size = 8, align = 8, is_pointer = true},
    {name = "arg1", type = "int", offset = 8, size = 4, align = 4, is_pointer = false}
  ],
  vortex.kernel_args_size = 12
}
```

**Why attributes**: MLIR attributes are preserved through most passes and can be queried by downstream transformations.

## Phase 3: gpu.launch Metadata Propagation

**Purpose**: Ensure metadata survives kernel outlining by attaching it to gpu.launch.

**Implementation**: `CGCall.cc` (CUDAKernelCallExpr handling)

When a kernel launch `<<<>>>` is encountered:
1. Create `gpu.launch` operation
2. Copy `vortex.kernel_args`, `vortex.kernel_args_size` from the kernel function
3. Attach `vortex.kernel_name` with the unmangled kernel name

```mlir
gpu.launch blocks(...) threads(...) {
  func.call @kernel(...)
} {vortex.kernel_args = [...], vortex.kernel_name = "simple_kernel"}
```

**Why on gpu.launch**: The kernel outlining pass extracts the gpu.launch body into a gpu.func. By putting metadata on gpu.launch, we ensure it's available when the outlined kernel is created.

## Phase 4: Kernel Outlining with Metadata

**Purpose**: Transfer metadata from gpu.launch to the outlined gpu.func.

**Implementation**: `llvm-project/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp`

Modifications to `outlineKernelFuncImpl`:
1. If `vortex.kernel_name` exists on gpu.launch, use it as the kernel name
2. Copy all `vortex.*` attributes to the outlined gpu.func

Result:
```mlir
gpu.func @simple_kernel(...) kernel attributes {
  vortex.kernel_args = [...],
  vortex.kernel_args_size = 12,
  vortex.kernel_name = "simple_kernel"
}
```

**Why modify outlining**: The default outlining names kernels based on the containing function (e.g., `host_function_kernel123`). We need the original kernel name for runtime lookup.

## Phase 5: Host-Side Compilation (Source-to-Source)

**Purpose**: The host side uses standard GCC/clang with HIP runtime macros.

**Implementation**: `runtime/host/hip_vortex_runtime.h`

The `hipLaunchKernelGGL` macro:
```cpp
#define hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, ...) \
    do { \
        auto _args = std::make_tuple(__VA_ARGS__); \
        hipLaunchKernelByName(#kernel, ..., &_args, sizeof(_args)); \
    } while(0)
```

**Why source-to-source**:
1. The HIP `<<<>>>` syntax is preprocessed to `hipLaunchKernelGGL`
2. `std::make_tuple` automatically packs arguments with correct C++ layout
3. `sizeof(_args)` gives the exact packed size
4. `#kernel` stringifies to the kernel name (e.g., `"simple_kernel"`)

This matches what the device expects because both use the same C++ ABI for argument layout.

## Runtime Flow

1. Host calls `simple_kernel<<<1,32>>>(ptr, n)`
2. Preprocessor expands to `hipLaunchKernelGGL(simple_kernel, ...)`
3. Macro packs args into tuple and calls `hipLaunchKernelByName("simple_kernel", ...)`
4. Runtime looks for `simple_kernel.vxbin` (or registered kernel)
5. Runtime loads kernel binary and copies packed args to device
6. `vx_start()` executes the kernel

## Key Files

| File | Purpose |
|------|---------|
| `cgeist/Lib/clang-mlir.cc` | AST traversal, metadata attachment |
| `cgeist/Lib/CGCall.cc` | CUDAKernelCallExpr handling, gpu.launch creation |
| `llvm-project/mlir/.../KernelOutlining.cpp` | Metadata transfer during outlining |
| `runtime/host/hip_vortex_runtime.h` | hipLaunchKernelGGL macro |
| `runtime/.../hip_kernel.cpp` | hipLaunchKernelByName implementation |

## Vortex Single Kernel Mode

When targeting Vortex, use `--vortex-single-kernel` to emit a single kernel variant instead of multiple block-size alternatives:

```bash
./bin/cgeist test.cu --cuda-lower --emit-cuda --vortex-single-kernel \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -I/path/to/runtime/device \
    --function='*' -S
```

This flag:
- Emits a single kernel with 16 threads (Vortex default: NUM_WARPS=4 x NUM_THREADS=4)
- Skips the `polygeist.alternatives` block-size specialization
- Produces simpler output suitable for Vortex targets

**Note**: When using `--cuda-lower`, the vortex metadata is currently not preserved through the block-size specialization pass. For full metadata preservation, omit `--cuda-lower` and handle lowering passes separately.

## Testing

Test without `--cuda-lower` (preserves vortex metadata):
```bash
./bin/cgeist test.cu --emit-cuda --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -I/path/to/runtime/device \
    --function='*' -S
```

Verify output contains:
```mlir
gpu.func @<kernel_name>(...) kernel attributes {
  vortex.kernel_args = [...],
  vortex.kernel_name = "<kernel_name>"
}
```

Test with `--cuda-lower --vortex-single-kernel` (single kernel, no alternatives):
```bash
./bin/cgeist test.cu --cuda-lower --emit-cuda --vortex-single-kernel \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -I/path/to/runtime/device \
    --function='*' -S
```

Verify output contains a single gpu.func with `gpu.known_block_size = array<i32: 16, 1, 1>`
