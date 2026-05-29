#include "../include/gelu.h"
#include "../include/cuda_utils.h"
#include <math.h>

__global__
void gelu_kernel(
    float* out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = out[idx];
        // Standard GPT GELU approximation formula
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void gelu_forward(
    float* out,
    int size
) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<blocks, threads_per_block>>>(out, size);
}
