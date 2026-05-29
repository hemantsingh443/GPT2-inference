#include "../include/residual.h"
#include "../include/cuda_utils.h"

__global__
void residual_kernel(
    float* out,
    const float* residual,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] += residual[idx];
    }
}

void residual_add(
    float* out,
    const float* residual,
    int size
) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    residual_kernel<<<blocks, threads_per_block>>>(out, residual, size);
}
