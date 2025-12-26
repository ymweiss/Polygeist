# Host Launch Library Approach

## Overview

When compiling HIP code for Vortex, the host needs to:
1. **Manage device** - Open/close Vortex device
2. **Manage memory** - Allocate device buffers, copy data to/from device
3. **Launch kernels** - Load binary, marshal arguments, start execution, wait for completion

This document describes compiling the MLIR host launch wrappers into a native library.

## Vortex Kernel Launch Sequence

A complete kernel launch involves these Vortex API calls:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEVICE INITIALIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_dev_open(&device)           - Open Vortex device handle                 │
│  vx_dev_caps(device, ...)       - Query device capabilities                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY ALLOCATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_mem_alloc(device, size, flags, &buffer)  - Allocate device memory       │
│  vx_mem_address(buffer, &dev_addr)           - Get 32-bit device address    │
│  vx_copy_to_dev(buffer, host_ptr, offset, size)  - H2D transfer            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KERNEL LOADING                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_upload_kernel_file(device, "kernel.vxbin", &kernel_buffer)              │
│  -- OR --                                                                   │
│  vx_upload_kernel_bytes(device, binary_data, size, &kernel_buffer)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARGUMENT MARSHALLING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Build arg buffer:                                                          │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │ Offset │ Content                                                   │   │
│    ├────────┼───────────────────────────────────────────────────────────┤   │
│    │   0    │ grid_dim.x, grid_dim.y, grid_dim.z  (3x uint32 = 12B)    │   │
│    │  12    │ block_dim.x, block_dim.y, block_dim.z (3x uint32 = 12B)  │   │
│    │  24    │ arg0 (device pointer → 32-bit address)                   │   │
│    │  28    │ arg1 (device pointer → 32-bit address)                   │   │
│    │  32    │ arg2 (scalar int32)                                      │   │
│    │  ...   │ ...                                                       │   │
│    └────────┴───────────────────────────────────────────────────────────┘   │
│                                                                             │
│  vx_upload_bytes(device, arg_buffer, size, &arg_buffer_device)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KERNEL EXECUTION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_start(device, kernel_buffer, arg_buffer_device)  - Begin execution      │
│  vx_ready_wait(device, timeout)                      - Wait for completion  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESULT RETRIEVAL                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_copy_from_dev(host_ptr, buffer, offset, size)    - D2H transfer         │
│  vx_mem_free(buffer)                                  - Free device memory  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLEANUP                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  vx_dev_close(device)                                - Close device         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Current Runtime Architecture

The HIP runtime (`libhip_vortex.so`) wraps Vortex intrinsics as HIP APIs:

| HIP API | Vortex Intrinsic(s) |
|---------|---------------------|
| `hipInit()` | `vx_dev_open()` |
| `hipMalloc()` | `vx_mem_alloc()`, `vx_mem_address()` |
| `hipMemcpy(H2D)` | `vx_copy_to_dev()` |
| `hipMemcpy(D2H)` | `vx_copy_from_dev()` |
| `hipFree()` | `vx_mem_free()` |
| `hipDeviceSynchronize()` | `vx_ready_wait()` |
| `vortexLaunchKernel()` | `vx_upload_bytes()`, `vx_start()` |

The `vortexLaunchKernel` function performs:
1. Look up kernel by name (from registered kernels)
2. Ensure kernel binary is uploaded (`vx_upload_kernel_bytes` if needed)
3. Build argument buffer (grid dims + block dims + user args)
4. Convert host pointers to device addresses
5. Upload args (`vx_upload_bytes`)
6. Start kernel (`vx_start`)

## Proposed Host Library Approach

### Option A: Call Runtime Functions

The MLIR launch wrapper calls existing runtime functions:

```mlir
// Input: GPU MLIR launch wrapper
func.func @launch_vecadd(%a: memref<?xf32>, %b: memref<?xf32>,
                         %c: memref<?xf32>, %n: i32, %bs: i32) {
    gpu.launch_func @kernel blocks in (%grid) threads in (%block)
        args(%a, %b, %c, %n)
}

// Output: Host-callable function using runtime
func.func @launch_vecadd(%a: !llvm.ptr, %b: !llvm.ptr,
                         %c: !llvm.ptr, %n: i32, %bs: i32) {
    // Build args struct
    %args = llvm.alloca ...
    llvm.store %a, %args[0]    // Store pointer (64-bit on host)
    llvm.store %b, %args[1]
    llvm.store %c, %args[2]
    llvm.store %n, %args[3]

    // Build grid/block dims
    %grid = llvm.alloca !llvm.struct<(i32, i32, i32)>
    %block = llvm.alloca !llvm.struct<(i32, i32, i32)>
    // ... fill in dims

    // Call runtime (handles device init, kernel loading, arg marshalling)
    llvm.call @vortexLaunchKernel("vecadd_kernel", %grid, %block,
                                   %args, %args_size, %metadata, %num_args)
}
```

**Pros:**
- Minimal changes - reuse existing runtime
- Pointer-to-address conversion handled by runtime
- Kernel loading handled by runtime

**Cons:**
- Requires runtime library at link time
- Metadata array must be generated

### Option B: Direct Vortex Intrinsics

Generate direct calls to Vortex intrinsics:

```mlir
func.func @launch_vecadd(%a: !llvm.ptr, %b: !llvm.ptr,
                         %c: !llvm.ptr, %n: i32, %bs: i32) {
    // Get device handle (global state)
    %device = llvm.load @vortex_device : !llvm.ptr

    // Look up device addresses for pointers
    %a_dev = llvm.call @vortex_get_device_addr(%a) : (!llvm.ptr) -> i32
    %b_dev = llvm.call @vortex_get_device_addr(%b) : (!llvm.ptr) -> i32
    %c_dev = llvm.call @vortex_get_device_addr(%c) : (!llvm.ptr) -> i32

    // Build argument buffer (Vortex format)
    %arg_buf = llvm.alloca 36 x i8  // 12 (grid) + 12 (block) + 12 (args)

    // Grid dims at offset 0
    llvm.store %grid_x, %arg_buf[0]
    llvm.store %grid_y, %arg_buf[4]
    llvm.store %grid_z, %arg_buf[8]

    // Block dims at offset 12
    llvm.store %block_x, %arg_buf[12]
    llvm.store %block_y, %arg_buf[16]
    llvm.store %block_z, %arg_buf[20]

    // User args at offset 24 (32-bit device addresses!)
    llvm.store %a_dev, %arg_buf[24]
    llvm.store %b_dev, %arg_buf[28]
    llvm.store %c_dev, %arg_buf[32]
    llvm.store %n, %arg_buf[36]

    // Upload args to device
    %arg_device = llvm.call @vx_upload_bytes(%device, %arg_buf, 40)

    // Get kernel buffer (must be pre-loaded)
    %kernel_buf = llvm.load @vecadd_kernel_buffer : !llvm.ptr

    // Start kernel
    llvm.call @vx_start(%device, %kernel_buf, %arg_device)
}
```

**Pros:**
- No runtime dependency for launch code
- Full control over marshalling
- Can inline/optimize

**Cons:**
- Must handle pointer→address mapping
- Kernel must be pre-loaded
- More complex code generation

### Option C: Hybrid Approach (Recommended)

Use runtime for complex operations, direct intrinsics for performance-critical path:

```mlir
func.func @launch_vecadd(%a: !llvm.ptr, %b: !llvm.ptr,
                         %c: !llvm.ptr, %n: i32, %bs: i32) {
    // Use runtime for pointer translation (handles buffer lookup)
    %a_dev = llvm.call @hip_ptr_to_device_addr(%a) : (!llvm.ptr) -> i32
    %b_dev = llvm.call @hip_ptr_to_device_addr(%b) : (!llvm.ptr) -> i32
    %c_dev = llvm.call @hip_ptr_to_device_addr(%c) : (!llvm.ptr) -> i32

    // Build Vortex arg buffer directly (fast path)
    %arg_buf = llvm.alloca ...
    // ... pack grid, block, and converted args

    // Use runtime for upload + launch (handles device state)
    llvm.call @vortex_launch_with_args("vecadd_kernel", %arg_buf, %arg_size)
}
```

## Required Runtime Support Functions

For Option C, the runtime needs:

```cpp
// Convert host buffer pointer to 32-bit device address
// Looks up in allocation table
extern "C" uint32_t hip_ptr_to_device_addr(void* host_ptr);

// Launch kernel with pre-packed Vortex argument buffer
// Handles kernel lookup, upload, and start
extern "C" hipError_t vortex_launch_with_args(
    const char* kernel_name,
    const void* vortex_args,  // Already in Vortex format (grid+block+args)
    size_t args_size
);

// Variant that takes pre-converted device addresses
extern "C" hipError_t vortex_launch_kernel_direct(
    const char* kernel_name,
    dim3 grid, dim3 block,
    const uint32_t* device_args,  // Already converted to 32-bit
    size_t num_args
);
```

## Implementation Plan

### Phase 1: Runtime Extensions

Add new functions to `libhip_vortex.so`:
- `hip_ptr_to_device_addr()` - expose pointer translation
- `vortex_launch_with_args()` - direct arg buffer launch

### Phase 2: MLIR Pass

Create `--convert-gpu-launch-to-host-call` pass:

**Input:**
```mlir
gpu.launch_func @kernel blocks in (%gx, %gy, %gz)
                        threads in (%bx, %by, %bz)
    args(%a: memref<?xf32>, %b: memref<?xf32>, %n: i32)
```

**Output:**
```mlir
// Extract base pointers from memrefs
%a_ptr = memref.extract_aligned_pointer_as_index %a
%b_ptr = memref.extract_aligned_pointer_as_index %b

// Convert to device addresses
%a_dev = llvm.call @hip_ptr_to_device_addr(%a_ptr)
%b_dev = llvm.call @hip_ptr_to_device_addr(%b_ptr)

// Build Vortex arg buffer
%buf = llvm.alloca 40 x i8
// ... pack grid (12B) + block (12B) + args (16B)

// Launch
llvm.call @vortex_launch_with_args("kernel", %buf, 40)
```

### Phase 3: Build Integration

Update `compile_hip_v2.sh`:

```bash
# Device compilation (unchanged)
... → kernel.vxbin

# Host library compilation (NEW)
polygeist-opt gpu.mlir \
    --convert-gpu-launch-to-host-call \
    --convert-memref-to-llvm \
    --convert-func-to-llvm \
    -o host.mlir

mlir-translate --mlir-to-llvmir host.mlir -o host.ll

clang -shared -fPIC host.ll \
    -L$RUNTIME_DIR -lhip_vortex \
    -o libkernels.so
```

## Argument Marshalling Details

### Pointer Conversion

Host pointers (64-bit `void*`) must be converted to device addresses (32-bit):

```
Host View:                    Device View:
┌──────────────────┐         ┌──────────────────┐
│ void* a = 0x7fff │   ──►   │ uint32_t a_dev = │
│   12345678       │         │   0x80001000     │
└──────────────────┘         └──────────────────┘
        │                            │
        │ hipMalloc returned         │ vx_mem_address returned
        │ this to user               │ this device address
        ▼                            ▼
┌──────────────────────────────────────────────┐
│        Runtime Allocation Table              │
│  host_ptr → {vx_buffer_h, device_addr}       │
└──────────────────────────────────────────────┘
```

### Arg Buffer Layout

Vortex expects this layout:

```
Offset  Size  Content
──────  ────  ─────────────────────────
0       4     grid_dim.x (uint32)
4       4     grid_dim.y (uint32)
8       4     grid_dim.z (uint32)
12      4     block_dim.x (uint32)
16      4     block_dim.y (uint32)
20      4     block_dim.z (uint32)
24      4     arg0 (uint32 device addr or scalar)
28      4     arg1 (uint32 device addr or scalar)
...     ...   ...
```

All values are 32-bit aligned for RISC-V.

## Files to Modify/Create

| File | Change |
|------|--------|
| `runtime/src/vortex_hip_runtime.cpp` | Add `hip_ptr_to_device_addr`, `vortex_launch_with_args` |
| `runtime/host/hip_vortex_runtime.h` | Declare new functions |
| `lib/polygeist/Passes/ConvertGPULaunchToHostCall.cpp` | **NEW** - MLIR pass |
| `lib/polygeist/Passes/Passes.td` | Add pass definition |
| `scripts/compile_hip_v2.sh` | Add host library compilation |

## Testing Strategy

1. **Unit test**: Verify pointer conversion in runtime
2. **Integration test**: Compile vecadd, link host library, verify execution
3. **Comparison test**: Same results as stub-based approach
