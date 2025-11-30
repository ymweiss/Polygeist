// Basic HIP kernel test
#include "Inputs/cuda.h"
#include "__clang_cuda_builtin_vars.h"

__global__ void basic_kernel(int32_t* src, int32_t* dst, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    dst[tid] = src[tid];
  }
}

void launch_basic(int32_t* d_src, int32_t* d_dst, uint32_t count, int threads_per_block) {
  int num_blocks = (count + threads_per_block - 1) / threads_per_block;
  basic_kernel<<<num_blocks, threads_per_block>>>(d_src, d_dst, count);
}
