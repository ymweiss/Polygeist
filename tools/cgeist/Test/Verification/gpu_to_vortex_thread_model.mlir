// RUN: polygeist-opt %s -convert-gpu-to-vortex | FileCheck %s

// Test Developer A: Thread Model & Synchronization operations
// Tests for blockDim, gridDim, and gpu.barrier operations

module {
  //===----------------------------------------------------------------------===//
  // Block Dimension Tests (blockDim.x, blockDim.y, blockDim.z)
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @test_block_dim_x
  func.func @test_block_dim_x() -> index {
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 0]
    // CHECK: llvm.load
    %bdim = gpu.block_dim x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bdim : index
  }

  // CHECK-LABEL: func @test_block_dim_y
  func.func @test_block_dim_y() -> index {
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 1]
    // CHECK: llvm.load
    %bdim = gpu.block_dim y
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bdim : index
  }

  // CHECK-LABEL: func @test_block_dim_z
  func.func @test_block_dim_z() -> index {
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 2]
    // CHECK: llvm.load
    %bdim = gpu.block_dim z
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bdim : index
  }

  //===----------------------------------------------------------------------===//
  // Grid Dimension Tests (gridDim.x, gridDim.y, gridDim.z)
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @test_grid_dim_x
  func.func @test_grid_dim_x() -> index {
    // CHECK: llvm.mlir.addressof @gridDim
    // CHECK: llvm.getelementptr {{.*}}[0, 0]
    // CHECK: llvm.load
    %gdim = gpu.grid_dim x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %gdim : index
  }

  // CHECK-LABEL: func @test_grid_dim_y
  func.func @test_grid_dim_y() -> index {
    // CHECK: llvm.mlir.addressof @gridDim
    // CHECK: llvm.getelementptr {{.*}}[0, 1]
    // CHECK: llvm.load
    %gdim = gpu.grid_dim y
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %gdim : index
  }

  // CHECK-LABEL: func @test_grid_dim_z
  func.func @test_grid_dim_z() -> index {
    // CHECK: llvm.mlir.addressof @gridDim
    // CHECK: llvm.getelementptr {{.*}}[0, 2]
    // CHECK: llvm.load
    %gdim = gpu.grid_dim z
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %gdim : index
  }

  //===----------------------------------------------------------------------===//
  // Barrier Synchronization Tests
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @test_simple_barrier
  func.func @test_simple_barrier() {
    // CHECK: %[[BAR_ID:.*]] = llvm.mlir.constant({{[0-9]+}} : i32)
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 0]
    // CHECK: llvm.load
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 1]
    // CHECK: llvm.load
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr {{.*}}[0, 2]
    // CHECK: llvm.load
    // CHECK: llvm.mul
    // CHECK: %[[NUM_THREADS:.*]] = llvm.mul
    // CHECK: llvm.call @vx_barrier(%[[BAR_ID]], %[[NUM_THREADS]])
    gpu.barrier
    // CHECK: return
    return
  }

  // CHECK-LABEL: func @test_multiple_barriers
  func.func @test_multiple_barriers() {
    // First barrier
    // CHECK: %[[BAR_ID_0:.*]] = llvm.mlir.constant({{[0-9]+}} : i32)
    // CHECK: llvm.call @vx_barrier(%[[BAR_ID_0]]
    gpu.barrier

    // Second barrier - ID should be different from first
    // CHECK: %[[BAR_ID_1:.*]] = llvm.mlir.constant({{[0-9]+}} : i32)
    // CHECK: llvm.call @vx_barrier(%[[BAR_ID_1]]
    gpu.barrier

    // CHECK: return
    return
  }

  //===----------------------------------------------------------------------===//
  // Combined Test: Global ID Computation Pattern
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @test_global_id_pattern
  func.func @test_global_id_pattern() -> index {
    // Get threadIdx.x
    // CHECK: llvm.mlir.addressof @threadIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %tid = gpu.thread_id x

    // Get blockIdx.x
    // CHECK: llvm.mlir.addressof @blockIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %bid = gpu.block_id x

    // Get blockDim.x
    // CHECK: llvm.mlir.addressof @blockDim
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %bdim = gpu.block_dim x

    // Compute: blockIdx.x * blockDim.x + threadIdx.x
    // CHECK: arith.muli
    %temp = arith.muli %bid, %bdim : index
    // CHECK: arith.addi
    %gid = arith.addi %temp, %tid : index

    // CHECK: return
    return %gid : index
  }

  //===----------------------------------------------------------------------===//
  // Realistic Kernel Pattern with Barrier
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @test_kernel_with_barrier
  func.func @test_kernel_with_barrier() -> index {
    // Compute global ID
    // CHECK: llvm.mlir.addressof @threadIdx
    %tid = gpu.thread_id x

    // CHECK: llvm.mlir.addressof @blockIdx
    %bid = gpu.block_id x

    // CHECK: llvm.mlir.addressof @blockDim
    %bdim = gpu.block_dim x

    // CHECK: arith.muli
    %temp = arith.muli %bid, %bdim : index
    // CHECK: arith.addi
    %gid = arith.addi %temp, %tid : index

    // Synchronize threads
    // CHECK: llvm.mlir.constant({{[0-9]+}} : i32)
    // CHECK: llvm.call @vx_barrier
    gpu.barrier

    // CHECK: return
    return %gid : index
  }

}
