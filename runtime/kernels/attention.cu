#include "../include/attention.h"
#include "../include/cuda_utils.h"
#include <math.h>

__global__
void update_kv_cache_kernel(
    float* qkv,
    float* key_cache,
    float* value_cache,
    int n_embd,
    int seq_len,
    int past_seq_len
) {
    int token_idx = blockIdx.y;
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx >= seq_len || feat_idx >= n_embd) return;

    int cache_slot = past_seq_len + token_idx;
    
    float k_val = qkv[token_idx * 3 * n_embd + n_embd + feat_idx];
    float v_val = qkv[token_idx * 3 * n_embd + 2 * n_embd + feat_idx];

    key_cache[cache_slot * n_embd + feat_idx] = k_val;
    value_cache[cache_slot * n_embd + feat_idx] = v_val;
}

void update_kv_cache(
    float* qkv,
    float* key_cache,
    float* value_cache,
    int n_embd,
    int seq_len,
    int past_seq_len
) {
    int threads = 256;
    dim3 grid((n_embd + threads - 1) / threads, seq_len);
    update_kv_cache_kernel<<<grid, threads>>>(qkv, key_cache, value_cache, n_embd, seq_len, past_seq_len);
}

__global__
void attention_kernel(
    float* qkv,
    float* key_cache,
    float* value_cache,
    float* output,
    int seq_len,
    int past_seq_len,
    int num_heads,
    int head_dim
) {
    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int C = num_heads * head_dim;
    int global_q_idx = past_seq_len + q_idx;

    __shared__ float s_scores[1024];

    float* q_ptr = qkv + q_idx * 3 * C + head_idx * head_dim;

    for (int j = tid; j <= global_q_idx; j += blockDim.x) {
        float* k_ptr = key_cache + j * C + head_idx * head_dim;

        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += q_ptr[d] * k_ptr[d];
        }
        s_scores[j] = sum / sqrtf((float)head_dim);
    }
    __syncthreads();

    float local_max = -1e20f;
    for (int j = tid; j <= global_q_idx; j += blockDim.x) {
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

    float local_sum = 0.0f;
    for (int j = tid; j <= global_q_idx; j += blockDim.x) {
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

    for (int j = tid; j <= global_q_idx; j += blockDim.x) {
        s_scores[j] = expf(s_scores[j] - s_max) / s_sum;
    }
    __syncthreads();

    if (tid < head_dim) {
        float val = 0.0f;
        for (int j = 0; j <= global_q_idx; ++j) {
            float* v_ptr = value_cache + j * C + head_idx * head_dim;
            val += s_scores[j] * v_ptr[tid];
        }
        output[q_idx * C + head_idx * head_dim + tid] = val;
    }
}

void attention_forward(
    float* qkv,
    float* key_cache,
    float* value_cache,
    float* output,
    int seq_len,
    int past_seq_len,
    int num_heads,
    int head_dim
) {
    dim3 grid(seq_len, num_heads);
    dim3 block(256);
    attention_kernel<<<grid, block>>>(qkv, key_cache, value_cache, output, seq_len, past_seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}
