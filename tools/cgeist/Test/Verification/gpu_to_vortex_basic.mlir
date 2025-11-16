// RUN: mlir-opt %s -convert-gpu-to-vortex | FileCheck %s

// Test basic gpu.thread_id lowering to Vortex CSR read

module {
  // CHECK-LABEL: func @test_thread_id_x
  func.func @test_thread_id_x() -> index {
    // CHECK: %[[TID:.*]] = llvm.inline_asm "csrr $0, 3264"
    // CHECK-SAME: : () -> i32
    %tid = gpu.thread_id x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %tid : index
  }

  // CHECK-LABEL: func @test_block_id_x
  func.func @test_block_id_x() -> index {
    // CHECK: %[[BID:.*]] = llvm.inline_asm "csrr $0, 3265"
    // CHECK-SAME: : () -> i32
    %bid = gpu.block_id x
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: return
    return %bid : index
  }

  // CHECK-LABEL: func @test_combined
  func.func @test_combined() -> index {
    // CHECK: %[[TID:.*]] = llvm.inline_asm "csrr $0, 3264"
    %tid = gpu.thread_id x
    // CHECK: %[[BID:.*]] = llvm.inline_asm "csrr $0, 3265"
    %bid = gpu.block_id x
    // CHECK: arith.addi
    %sum = arith.addi %tid, %bid : index
    // CHECK: return
    return %sum : index
  }
}
