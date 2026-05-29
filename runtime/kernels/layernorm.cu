#include "../include/layernorm.h" 
#include "../include/cuda_utils.h" 
#include <math.h>

__global__ 
void layernorm_kernel( 
    float* input, 
    float* residual, 
    float* gamma, 
    float* beta, 
    float* output, 
    int hidden_size, 
    bool add_residual,
    float eps
) { 
    int token_idx = blockIdx.x; 
    int tid = threadIdx.x; 

    float* token_input = input + token_idx * hidden_size;
    float* token_output = output + token_idx * hidden_size;
    float* token_residual = (residual != nullptr) ? (residual + token_idx * hidden_size) : nullptr;

    // Sum input and residual in-place if requested
    if (token_residual != nullptr) {
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            float val = token_input[i];
            if (add_residual) {
                val += token_residual[i];
            }
            token_input[i] = val;
            token_residual[i] = val;
        }
        __syncthreads();
    }

    // Compute mean
    float local_sum = 0.0f; 
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        local_sum += token_input[i]; 
    } 

    // Block-level reduction for sum using shared memory
    __shared__ float shared_val[256]; 
    shared_val[tid] = local_sum; 
    __syncthreads(); 

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) { 
        if (tid < stride) { 
            shared_val[tid] += shared_val[tid + stride]; 
        } 
        __syncthreads(); 
    } 

    __shared__ float mean; 
    if (tid == 0) { 
        mean = shared_val[0] / hidden_size; 
    } 
    __syncthreads(); 

    // Compute variance
    float local_sq_diff = 0.0f; 
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        float diff = token_input[i] - mean; 
        local_sq_diff += diff * diff; 
    } 

    shared_val[tid] = local_sq_diff; 
    __syncthreads(); 

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) { 
        if (tid < stride) { 
            shared_val[tid] += shared_val[tid + stride]; 
        } 
        __syncthreads(); 
    } 

    __shared__ float inv_std; 
    if (tid == 0) { 
        float variance = shared_val[0] / hidden_size; 
        inv_std = 1.0f / sqrtf(variance + eps); 
    } 
    __syncthreads(); 

    // Normalize and scale/shift
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        float norm = (token_input[i] - mean) * inv_std; 
        token_output[i] = norm * gamma[i] + beta[i]; 
    } 
} 

void layernorm_forward( 
    float* input, 
    float* residual, 
    float* gamma, 
    float* beta, 
    float* output, 
    int hidden_size, 
    int seq_len,
    bool add_residual,
    float eps
) { 
    // Launch kernel with block count equal to sequence length
    int threads_per_block = 256; 
    layernorm_kernel<<<seq_len, threads_per_block>>>(
        input, residual, gamma, beta, output, hidden_size, add_residual, eps
    ); 
}
