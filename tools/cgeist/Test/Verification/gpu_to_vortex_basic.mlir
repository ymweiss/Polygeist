// RUN: mlir-opt %s -convert-gpu-to-vortex | FileCheck %s

// Test basic gpu.thread_id and gpu.block_id lowering to Vortex TLS access

module {
  // CHECK-LABEL: func @test_thread_id_x
  func.func @test_thread_id_x() -> index {
    // CHECK: llvm.mlir.global external thread_local @threadIdx
    // CHECK: llvm.mlir.addressof @threadIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %tid = gpu.thread_id x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %tid : index
  }

  // CHECK-LABEL: func @test_thread_id_y
  func.func @test_thread_id_y() -> index {
    // CHECK: llvm.mlir.addressof @threadIdx
    // CHECK: llvm.getelementptr
    // CHECK-SAME: 1
    // CHECK: llvm.load
    %tid = gpu.thread_id y
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %tid : index
  }

  // CHECK-LABEL: func @test_thread_id_z
  func.func @test_thread_id_z() -> index {
    // CHECK: llvm.mlir.addressof @threadIdx
    // CHECK: llvm.getelementptr
    // CHECK-SAME: 2
    // CHECK: llvm.load
    %tid = gpu.thread_id z
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %tid : index
  }

  // CHECK-LABEL: func @test_block_id_x
  func.func @test_block_id_x() -> index {
    // CHECK: llvm.mlir.global external thread_local @blockIdx
    // CHECK: llvm.mlir.addressof @blockIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %bid = gpu.block_id x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bid : index
  }

  // CHECK-LABEL: func @test_block_id_y
  func.func @test_block_id_y() -> index {
    // CHECK: llvm.mlir.addressof @blockIdx
    // CHECK: llvm.getelementptr
    // CHECK-SAME: 1
    // CHECK: llvm.load
    %bid = gpu.block_id y
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bid : index
  }

  // CHECK-LABEL: func @test_combined
  func.func @test_combined() -> index {
    // CHECK: llvm.mlir.addressof @threadIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %tid = gpu.thread_id x
    // CHECK: llvm.mlir.addressof @blockIdx
    // CHECK: llvm.getelementptr
    // CHECK: llvm.load
    %bid = gpu.block_id x
    // CHECK: arith.addi
    %sum = arith.addi %tid, %bid : index
    // CHECK: return
    return %sum : index
  }
}
