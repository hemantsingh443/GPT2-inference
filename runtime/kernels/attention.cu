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
void flash_attention_prefill_kernel(
    float* qkv,
    float* key_cache,
    float* value_cache,
    float* output,
    int seq_len,
    int past_seq_len,
    int num_heads,
    int head_dim
) {
    int q_block = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.y * 32 + threadIdx.x;

    int C = num_heads * head_dim;
    int q_local = threadIdx.x;
    int total_seq_len = past_seq_len + seq_len;

    // Transpose s_Q to avoid shared memory bank conflicts
    __shared__ float s_Q[64][32];
    __shared__ float s_K[32][64];
    __shared__ float s_V[32][64];
    __shared__ float s_scores[32][32];
    __shared__ float s_row_max[32];
    __shared__ float s_row_sum[32];

    // Shared memory for parallel reductions along threadIdx.y (dimension of size eight)
    __shared__ float s_reduce_max[8][32];
    __shared__ float s_reduce_sum[8][32];

    for (int i = tid; i < 32 * 64; i += 256) {
        int r = i / 64;
        int c = i % 64;
        int target_q = q_block * 32 + r;
        if (target_q < seq_len) {
            s_Q[c][r] = qkv[target_q * 3 * C + head_idx * head_dim + c];
        } else {
            s_Q[c][r] = 0.0f;
        }
    }
    __syncthreads();

    float r_m = -1e20f;
    float r_d = 0.0f;
    float r_O[8] = {0.0f};

    for (int kv_block = 0; kv_block <= q_block; ++kv_block) {
        for (int i = tid; i < 32 * 64; i += 256) {
            int r = i / 64;
            int c = i % 64;
            int k_idx = kv_block * 32 + r;
            if (k_idx < total_seq_len) {
                s_K[r][c] = key_cache[k_idx * C + head_idx * head_dim + c];
                s_V[r][c] = value_cache[k_idx * C + head_idx * head_dim + c];
            } else {
                s_K[r][c] = 0.0f;
                s_V[r][c] = 0.0f;
            }
        }
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            int k_local = threadIdx.y * 4 + i;
            int k_idx = kv_block * 32 + k_local;
            int target_q = q_block * 32 + q_local;

            float score = 0.0f;
            if (target_q < seq_len && k_idx <= target_q && k_idx < total_seq_len) {
                for (int d = 0; d < 64; ++d) {
                    score += s_Q[d][q_local] * s_K[k_local][d];
                }
                score /= 8.0f;
            } else {
                score = -1e20f;
            }
            s_scores[q_local][k_local] = score;
        }
        __syncthreads();

        // Parallel reduction along threadIdx.y to find max
        float thread_max = -1e20f;
        for (int i = 0; i < 4; ++i) {
            int k_local = threadIdx.y * 4 + i;
            float s = s_scores[q_local][k_local];
            if (s > thread_max) thread_max = s;
        }
        s_reduce_max[threadIdx.y][q_local] = thread_max;
        __syncthreads();

        if (threadIdx.y < 4) {
            s_reduce_max[threadIdx.y][q_local] = fmaxf(s_reduce_max[threadIdx.y][q_local], s_reduce_max[threadIdx.y + 4][q_local]);
        }
        __syncthreads();
        if (threadIdx.y < 2) {
            s_reduce_max[threadIdx.y][q_local] = fmaxf(s_reduce_max[threadIdx.y][q_local], s_reduce_max[threadIdx.y + 2][q_local]);
        }
        __syncthreads();
        if (threadIdx.y == 0) {
            s_row_max[q_local] = fmaxf(s_reduce_max[0][q_local], s_reduce_max[1][q_local]);
        }
        __syncthreads();

        float block_max = s_row_max[q_local];

        // Precompute exponential values and store in s_scores to avoid redundant expf calls in accumulation loop
        for (int i = 0; i < 4; ++i) {
            int k_local = threadIdx.y * 4 + i;
            float s = s_scores[q_local][k_local];
            if (s > -1e9f) {
                s_scores[q_local][k_local] = expf(s - block_max);
            } else {
                s_scores[q_local][k_local] = 0.0f;
            }
        }
        __syncthreads();

        // Parallel reduction along threadIdx.y to find sum
        float thread_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int k_local = threadIdx.y * 4 + i;
            thread_sum += s_scores[q_local][k_local];
        }
        s_reduce_sum[threadIdx.y][q_local] = thread_sum;
        __syncthreads();

        if (threadIdx.y < 4) {
            s_reduce_sum[threadIdx.y][q_local] += s_reduce_sum[threadIdx.y + 4][q_local];
        }
        __syncthreads();
        if (threadIdx.y < 2) {
            s_reduce_sum[threadIdx.y][q_local] += s_reduce_sum[threadIdx.y + 2][q_local];
        }
        __syncthreads();
        if (threadIdx.y == 0) {
            s_row_sum[q_local] = s_reduce_sum[0][q_local] + s_reduce_sum[1][q_local];
        }
        __syncthreads();

        float block_sum = s_row_sum[q_local];

        float old_m = r_m;
        float old_d = r_d;

        r_m = fmaxf(old_m, block_max);
        float scale_prev = expf(old_m - r_m);
        float scale_curr = expf(block_max - r_m);

        r_d = old_d * scale_prev + block_sum * scale_curr;

        for (int d = 0; d < 8; ++d) {
            r_O[d] *= scale_prev;
        }

        int d_offset = threadIdx.y * 8;
        for (int k = 0; k < 32; ++k) {
            float weight = s_scores[q_local][k] * scale_curr;
            for (int d = 0; d < 8; ++d) {
                r_O[d] += weight * s_V[k][d_offset + d];
            }
        }
        __syncthreads();
    }

    int target_q = q_block * 32 + q_local;
    if (target_q < seq_len) {
        int d_offset = threadIdx.y * 8;
        for (int d = 0; d < 8; ++d) {
            output[target_q * C + head_idx * head_dim + d_offset + d] = r_O[d] / r_d;
        }
    }
}

__global__
void flash_decoding_map_kernel(
    float* qkv,
    float* key_cache,
    float* value_cache,
    float* temp_output,
    float* temp_stats,
    int past_seq_len,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int tid = threadIdx.x;

    int C = num_heads * head_dim;
    int total_seq_len = past_seq_len + seq_len;
    int num_chunks = (total_seq_len + 128 - 1) / 128;

    if (chunk_idx >= num_chunks) return;

    __shared__ float s_Q[64];
    __shared__ float s_scores[128];
    __shared__ float s_reduce[128];

    if (tid < 64) {
        s_Q[tid] = qkv[head_idx * head_dim + tid];
    }
    __syncthreads();

    int k_idx = chunk_idx * 128 + tid;
    float r_score = -1e20f;

    if (k_idx < total_seq_len) {
        float* k_ptr = key_cache + k_idx * C + head_idx * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < 64; ++d) {
            sum += s_Q[d] * k_ptr[d];
        }
        r_score = sum / 8.0f;
    }

    s_scores[tid] = r_score;
    s_reduce[tid] = r_score;
    __syncthreads();

    for (int stride = 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        }
        __syncthreads();
    }
    __shared__ float s_block_max;
    if (tid == 0) s_block_max = s_reduce[0];
    __syncthreads();

    // Compute softmax weights and store them in s_scores to avoid redundant expf in final loop
    float local_exp = (k_idx < total_seq_len) ? expf(r_score - s_block_max) : 0.0f;
    s_scores[tid] = local_exp;
    s_reduce[tid] = local_exp;
    __syncthreads();

    for (int stride = 64; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_block_sum;
    if (tid == 0) s_block_sum = s_reduce[0];
    __syncthreads();

    if (tid < 64) {
        float val = 0.0f;
        for (int i = 0; i < 128; ++i) {
            int target_k = chunk_idx * 128 + i;
            if (target_k < total_seq_len) {
                float weight = s_scores[i];
                float* v_ptr = value_cache + target_k * C + head_idx * head_dim;
                val += weight * v_ptr[tid];
            }
        }
        int out_offset = head_idx * 8 * 64 + chunk_idx * 64 + tid;
        temp_output[out_offset] = val;
    }

    if (tid == 0) {
        int stats_offset = head_idx * 8 * 2 + chunk_idx * 2;
        temp_stats[stats_offset] = s_block_max;
        temp_stats[stats_offset + 1] = s_block_sum;
    }
}

__global__
void flash_decoding_reduce_kernel(
    float* temp_output,
    float* temp_stats,
    float* output,
    int past_seq_len,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    int total_seq_len = past_seq_len + seq_len;
    int num_chunks = (total_seq_len + 128 - 1) / 128;

    float r_m = -1e20f;
    float r_d = 0.0f;
    float r_val = 0.0f;

    for (int c = 0; c < num_chunks; ++c) {
        int stats_offset = head_idx * 8 * 2 + c * 2;
        float chunk_max = temp_stats[stats_offset];
        float chunk_sum = temp_stats[stats_offset + 1];

        int out_offset = head_idx * 8 * 64 + c * 64 + tid;
        float chunk_val = temp_output[out_offset];

        float old_m = r_m;
        r_m = fmaxf(r_m, chunk_max);
        float scale_prev = expf(old_m - r_m);
        float scale_curr = expf(chunk_max - r_m);

        r_d = r_d * scale_prev + chunk_sum * scale_curr;
        r_val = r_val * scale_prev + chunk_val * scale_curr;
    }

    output[head_idx * 64 + tid] = r_val / r_d;
}

void attention_forward(
    float* qkv,
    float* key_cache,
    float* value_cache,
    float* temp_output,
    float* temp_stats,
    float* output,
    int seq_len,
    int past_seq_len,
    int num_heads,
    int head_dim
) {
    if (seq_len > 1) {
        dim3 grid((seq_len + 32 - 1) / 32, num_heads);
        dim3 block(32, 8);
        flash_attention_prefill_kernel<<<grid, block>>>(
            qkv, key_cache, value_cache, output, seq_len, past_seq_len, num_heads, head_dim
        );
    } else {
        int total_seq_len = past_seq_len + seq_len;
        int num_chunks = (total_seq_len + 128 - 1) / 128;
        dim3 grid(num_heads, num_chunks);
        dim3 block(128);
        flash_decoding_map_kernel<<<grid, block>>>(
            qkv, key_cache, value_cache, temp_output, temp_stats, past_seq_len, seq_len, num_heads, head_dim
        );

        flash_decoding_reduce_kernel<<<num_heads, 64>>>(
            temp_output, temp_stats, output, past_seq_len, seq_len, num_heads, head_dim
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
