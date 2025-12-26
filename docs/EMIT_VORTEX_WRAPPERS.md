# HIP-to-Vortex GPU Launch Generation

## Overview

When compiling HIP/CUDA code for Vortex, cgeist needs to:
1. Generate `gpu.launch` operations that contain the kernel body
2. Attach `vortex.kernel_args` metadata for argument marshalling
3. Create wrapper functions that can be processed by kernel outlining

There are two approaches to achieve this:

| Approach | Description | Recommended |
|----------|-------------|-------------|
| **Macro Expansion** | `hipLaunchKernelGGL` expands to `<<<>>>` syntax | ✅ Yes |
| **`--emit-vortex-wrappers`** | cgeist auto-generates wrappers from kernel signatures | For raw kernels only |

---

## Recommended Approach: hipLaunchKernelGGL Macro

### Why This Approach is Better

The macro expansion approach leverages existing clang CUDA infrastructure:

1. **Captures exact call-site arguments**: Any pointer arithmetic, casts, or expressions passed to the kernel are visible in the generated MLIR
2. **Uses proven code path**: CGCall.cc's `CUDAKernelCallExpr` handling is well-tested
3. **Works with standard HIP code**: No source transformation needed
4. **Explicit launch configuration**: Grid/block dimensions come from the actual call site

### How It Works

The device header (`runtime/device/hip_runtime.h`) defines:

```cpp
#ifdef __CUDA__
#define hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, ...) \
    kernel<<<(gridDim), (blockDim), (sharedMem), (stream)>>>(__VA_ARGS__)
#endif
```

When cgeist compiles device code, the macro expands to `<<<>>>` syntax, which triggers clang's `CUDAKernelCallExpr` handling. This creates a `gpu.launch` operation with:
- Proper grid/block dimensions from the call site
- `vortex.kernel_args` metadata for argument marshalling
- `vortex.kernel_name` attribute for binary naming

### Usage

**Input (vecadd.hip)**:
```cpp
#include "hip_runtime.h"

__global__ void vecadd_kernel(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void launch_vecadd(const float* a, const float* b, float* c, unsigned int n) {
    dim3 grid((n + 255) / 256);
    dim3 block(256);
    hipLaunchKernelGGL(vecadd_kernel, grid, block, 0, 0, a, b, c, n);
}
```

**Compile**:
```bash
./build/bin/cgeist vecadd.hip \
    --cuda-lower \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -resource-dir=$(clang -print-resource-dir) \
    -I /path/to/runtime/device \
    --function='*' \
    --immediate
```

**Output** (abbreviated MLIR):
```mlir
module {
  // Kernel function with GPU intrinsics
  func.func @_Z13vecadd_kernelPKfS0_Pfj(...) attributes {
    vortex.kernel_args = [
      {name = "a", type = "const float *", size = 8, offset = 0, ...},
      {name = "b", type = "const float *", size = 8, offset = 8, ...},
      {name = "c", type = "float *", size = 8, offset = 16, ...},
      {name = "n", type = "unsigned int", size = 4, offset = 24, ...}
    ],
    vortex.kernel_args_size = 28 : i32
  }

  // Launch function with gpu.launch
  func.func @_Z13launch_vecaddPKfS0_Pfj(...) {
    // ... grid/block dim setup ...
    gpu.launch async blocks(...) in (...) threads(...) in (...) {
      func.call @_Z27__device_stub__vecadd_kernel(...)
      gpu.terminator
    } {
      vortex.kernel_args = [...],
      vortex.kernel_name = "vecadd_kernel"
    }
    return
  }
}
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--cuda-lower` | Enable CUDA/HIP mode compilation |
| `--immediate` | Output MLIR before pass pipeline (GPU dialect) |
| `--function='*'` | Compile all functions in the source |
| `-nocudalib -nocudainc` | Don't include system CUDA libraries/headers |
| `-resource-dir=<path>` | **CRITICAL**: Path to clang resource directory (for stddef.h, etc.) |
| `-I <path>` | Include path for `runtime/device/hip_runtime.h` |

---

## Critical: Include Directory Setup

**The `-resource-dir` flag is essential** for cgeist to find standard C headers (`stddef.h`, `stdint.h`, etc.). Without it, compilation will fail with "file not found" errors.

### Finding the Resource Directory

For a Polygeist build, the resource directory is typically:
```bash
# Standard location after Polygeist build
RESOURCE_DIR="$POLYGEIST_ROOT/llvm-project/build/lib/clang/18"

# Or use system clang to find it
RESOURCE_DIR=$(clang -print-resource-dir)
```

### Required Include Paths

| Include | Purpose |
|---------|---------|
| `-I runtime/device` | Device-side HIP header (`hip_runtime.h` with GPU builtins) |
| `-resource-dir=...` | Standard C headers (`stddef.h`, `stdint.h`) |

### Example with Full Paths

```bash
REPO_ROOT="/path/to/vortex_hip"
POLYGEIST="$REPO_ROOT/Polygeist"
RESOURCE_DIR="$POLYGEIST/llvm-project/build/lib/clang/18"

$POLYGEIST/build/bin/cgeist input.hip \
    --cuda-lower \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -resource-dir="$RESOURCE_DIR" \
    -I "$REPO_ROOT/runtime/device" \
    --function='*' \
    --immediate
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `'stddef.h' file not found` | Missing `-resource-dir` | Add `-resource-dir=<clang-resource-path>` |
| `'hip_runtime.h' file not found` | Missing include path | Add `-I runtime/device` |
| `'__clang_cuda_builtin_vars.h' not found` | Wrong resource dir | Use correct clang version's resource dir |

---

### Argument Visibility

The macro approach captures exact call-site expressions. For example:

```cpp
hipLaunchKernelGGL(kernel, grid, block, 0, 0, ptr + offset, n * 2);
```

The generated MLIR shows `ptr + offset` and `n * 2` as computed values, making argument transformations visible to downstream passes.

---

## Alternative: `--emit-vortex-wrappers` Flag

For raw kernel files without launch functions, use the `--emit-vortex-wrappers` flag.

### When to Use

- Kernel-only source files (no `hipLaunchKernelGGL` calls)
- Testing individual kernels
- When you want cgeist to synthesize launch wrappers

### Usage

```bash
./build/bin/cgeist raw_kernel.hip \
    --cuda-lower \
    --emit-vortex-wrappers \
    --cuda-gpu-arch=sm_60 \
    -nocudalib -nocudainc \
    -resource-dir=$(clang -print-resource-dir) \
    -I /path/to/runtime/device \
    --function='*' \
    --immediate
```

### How It Works

1. **Device Mode Only**: Wrappers generated when `CUDAIsDevice=1`
2. **HIPKernelCollector**: AST traversal finds all `__global__` functions
3. **getOrCreateLaunchWrapper()**: Creates wrapper function with `gpu.launch`
4. **Metadata Propagation**: `vortex.kernel_args` attached to kernel, wrapper, and `gpu.launch`

### Limitations

- Synthesizes wrapper from kernel signature (doesn't see actual call-site arguments)
- Grid/block dimensions are parameters (not concrete values from source)
- Cannot capture argument transformations (casts, pointer arithmetic)

### Debug Output

When using `--emit-vortex-wrappers`, stderr shows:
- "Skipping Vortex wrapper generation in host mode" (during host pass)
- "Generating Vortex launch wrappers for N kernel(s)..." (during device pass)
- "Created wrapper: __polygeist_launch_..." for each kernel

---

## Next Steps in Pipeline

After cgeist generates GPU MLIR:

1. **polygeist-opt** runs kernel outlining and Vortex conversion
2. **convert-gpu-to-vortex** pass generates:
   - `<kernel_name>.meta.json` - JSON metadata for runtime
   - `<kernel_name>_args.h` - Complete C++ host stub with launcher function
3. Host code is compiled with generated stubs

---

## Host Code Compilation

### Using Generated Host Stubs

The `convert-gpu-to-vortex` pass generates complete host stubs with:
- Packed argument struct
- Metadata array for `vortexLaunchKernel`
- Type-safe inline launcher function

**Example generated stub** (`vecadd_kernel_args.h`):
```cpp
typedef struct __attribute__((packed)) {
  void* a;      // host_offset=0, host_size=8, device pointer
  void* b;      // host_offset=8, host_size=8, device pointer
  void* c;      // host_offset=16, host_size=8, device pointer
  uint32_t n;   // host_offset=24, host_size=4
} vecadd_kernel_args_t;

static const VortexKernelArgMeta vecadd_kernel_metadata[] = {
  { .offset = 0, .size = 8, .is_pointer = 1 },
  { .offset = 8, .size = 8, .is_pointer = 1 },
  { .offset = 16, .size = 8, .is_pointer = 1 },
  { .offset = 24, .size = 4, .is_pointer = 0 },
};

static inline hipError_t launch_vecadd_kernel(
    dim3 gridDim, dim3 blockDim,
    const void* a, const void* b, const void* c, uint32_t n) {
  vecadd_kernel_args_t args = {(void*)a, (void*)b, (void*)c, n};
  return vortexLaunchKernel("vecadd_kernel", gridDim, blockDim,
    &args, sizeof(args), vecadd_kernel_metadata, 4);
}
```

### Host Include Pattern

Use conditional includes to support both device and host compilation:

```cpp
#ifdef __CUDA__
#include "hip_runtime.h"      // Device compilation (cgeist)
#else
#include "hip_vortex_host.h"  // Host compilation (GCC/Clang)
#include "vecadd_kernel_args.h"  // Generated kernel stub
#endif

void run_vecadd(float* d_a, float* d_b, float* d_c, unsigned int n) {
    dim3 grid((n + 255) / 256);
    dim3 block(256);

#ifdef __CUDA__
    // Device code path (for cgeist to capture metadata)
    hipLaunchKernelGGL(vecadd_kernel, grid, block, 0, 0, d_a, d_b, d_c, n);
#else
    // Host code path (uses generated launcher)
    launch_vecadd_kernel(grid, block, d_a, d_b, d_c, n);
#endif
}
```

---

## Related Files

| File | Purpose |
|------|---------|
| `runtime/device/hip_runtime.h` | Device header with `hipLaunchKernelGGL` macro |
| `tools/cgeist/Lib/clang-mlir.cc` | cgeist implementation (wrapper generation) |
| `tools/cgeist/Lib/HIPKernelAnalysis.cc` | Kernel metadata extraction |
| `lib/polygeist/Passes/ConvertGPUToVortex.cpp` | Downstream Vortex pass |
| `docs/HIP_KERNEL_METADATA_FLOW.md` | Metadata flow documentation |
