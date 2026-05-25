#include "../include/layernorm.h" 
#include "../include/cuda_utils.h" 
#include <math.h>

__global__ 
void layernorm_kernel( 
    float* input, 
    float* gamma, 
    float* beta, 
    float* output, 
    int hidden_size, 
    float eps
) { 
    int tid = threadIdx.x; 

    //Compute mean
    float local_sum = 0.0f; 
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        local_sum += input[i]; 
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

    //Compute variance
    float local_sq_diff = 0.0f; 
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        float diff = input[i] - mean; 
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

    //Normalize and scale/shift
    for (int i = tid; i < hidden_size; i += blockDim.x) { 
        float norm = (input[i] - mean) * inv_std; 
        output[i] = norm * gamma[i] + beta[i]; 
    } 
} 

void layernorm_forward( 
    float* input, 
    float* gamma, 
    float* beta, 
    float* output, 
    int hidden_size, 
    float eps
) { 
    // Launch kernel with 1 block and 256 threads.
    // 256 is a power of 2, ensuring correct block reduction.
    int threads_per_block = 256; 
    layernorm_kernel<<<1, threads_per_block>>>(input, gamma, beta, output, hidden_size, eps); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
}
