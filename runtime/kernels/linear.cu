#include "../include/linear.h" 
#include "../include/cuda_utils.h" 

__global__ 
void linear_kernel( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features  
){ 

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (out_idx >= out_features) return;  

    float sum = 0.0f; 

    for (int k = 0; k < in_features; ++k) { 
        sum += input[k] * weights[k * out_features + out_idx]; 
    } 

    if(bias != nullptr) { 
        sum += bias[out_idx]; 
    } 
    output[out_idx] = sum;
} 

void linear_forward( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features
){ 
    // Launch kernel with 256 threads per block to support out_features > 1024
    int threads_per_block = 256;  
    int blocks = (out_features + threads_per_block - 1) / threads_per_block; 
    linear_kernel<<<blocks, threads_per_block>>>(input, weights, bias, output, in_features, out_features); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
} 

__global__ 
void linear_transposed_kernel( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features  
){ 
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (out_idx >= out_features) return;  

    float sum = 0.0f; 
    // Notice the difference in weight memory indexing: out_idx * in_features + k
    for (int k = 0; k < in_features; ++k) { 
        sum += input[k] * weights[out_idx * in_features + k]; 
    } 

    if (bias != nullptr) { 
        sum += bias[out_idx]; 
    } 
    output[out_idx] = sum;
}

void linear_forward_transposed( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features
){ 
    int threads_per_block = 256;  
    int blocks = (out_features + threads_per_block - 1) / threads_per_block; 
    linear_transposed_kernel<<<blocks, threads_per_block>>>(input, weights, bias, output, in_features, out_features); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
}
