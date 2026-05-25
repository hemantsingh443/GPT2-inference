#include "../include/attention.h"
#include "../include/cuda_utils.h"
#include <math.h>

__global__
void attention_kernel(
    float* qkv,
    float* output,
    int seq_len,
    int num_heads,
    int head_dim
) {
    // blockIdx.x is query token index (q_idx)
    // blockIdx.y is head index (head_idx)
    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int C = num_heads * head_dim;

    // Shared memory to hold scores for causal tokens (max sequence length 1024)
    __shared__ float s_scores[1024];

    // Compute Q @ K^T / sqrt(D)
    // We only compute scores for keys up to current query token (j <= q_idx)
    // Pointer to Q[q_idx, head_idx]
    float* q_ptr = qkv + q_idx * 3 * C + 0 * C + head_idx * head_dim;

    for (int j = tid; j <= q_idx; j += blockDim.x) {
        // Pointer to K[j, head_idx]
        float* k_ptr = qkv + j * 3 * C + 1 * C + head_idx * head_dim;

        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += q_ptr[d] * k_ptr[d];
        }
        s_scores[j] = sum / sqrtf((float)head_dim);
    }
    __syncthreads();

    //Parallel Causal Softmax
    // Find max score for numerical stability
    float local_max = -1e20f;
    for (int j = tid; j <= q_idx; j += blockDim.x) {
        if (s_scores[j] > local_max) {
            local_max = s_scores[j];
        }
    }

    __shared__ float s_max;
    __shared__ float s_reduce[256];
    s_reduce[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_reduce[0];
    __syncthreads();

    // Sum exponents
    float local_sum = 0.0f;
    for (int j = tid; j <= q_idx; j += blockDim.x) {
        local_sum += expf(s_scores[j] - s_max);
    }

    s_reduce[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_sum;
    if (tid == 0) s_sum = s_reduce[0];
    __syncthreads();

    // Normalize probabilities
    for (int j = tid; j <= q_idx; j += blockDim.x) {
        s_scores[j] = expf(s_scores[j] - s_max) / s_sum;
    }
    __syncthreads();

    //Weighted Sum over V values
    // Let each thread handle a subset of head dimensions (D = 64)
    if (tid < head_dim) {
        float val = 0.0f;
        for (int j = 0; j <= q_idx; ++j) {
            // Pointer to V[j, head_idx]
            float* v_ptr = qkv + j * 3 * C + 2 * C + head_idx * head_dim;
            val += s_scores[j] * v_ptr[tid];
        }
        // Write to output[q_idx, head_idx, tid]
        output[q_idx * C + head_idx * head_dim + tid] = val;
    }
}

void attention_forward(
    float* qkv,
    float* output,
    int seq_len,
    int num_heads,
    int head_dim
) {
    dim3 grid(seq_len, num_heads);
    dim3 block(256); // 256 threads is perfect for reduction
    attention_kernel<<<grid, block>>>(qkv, output, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}
