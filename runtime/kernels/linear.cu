#include "../include/linear.h" 
#include "../include/cuda_utils.h" 

__global__
void linear_gemv_warp_reduced(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int in_features,
    int out_features
) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_idx = threadIdx.x % 32;

    if (warp_idx >= out_features) return;

    float sum = 0.0f;
    for (int k = lane_idx; k < in_features; k += 32) {
        sum += input[k] * weights[k * out_features + warp_idx];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_idx == 0) {
        if (bias != nullptr) {
            sum += bias[warp_idx];
        }
        output[warp_idx] = sum;
    }
}

__global__
void linear_gemv_transposed_warp_reduced(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int in_features,
    int out_features
) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_idx = threadIdx.x % 32;

    if (warp_idx >= out_features) return;

    float sum = 0.0f;
    for (int k = lane_idx; k < in_features; k += 32) {
        sum += input[k] * weights[warp_idx * in_features + k];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_idx == 0) {
        if (bias != nullptr) {
            sum += bias[warp_idx];
        }
        output[warp_idx] = sum;
    }
}

__global__
void linear_gemm_batched(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int in_features,
    int out_features,
    int seq_len
) {
    __shared__ float s_input[16][16];
    __shared__ float s_weights[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (in_features + 16 - 1) / 16; ++ph) {
        if (row < seq_len && (ph * 16 + threadIdx.x) < in_features) {
            s_input[threadIdx.y][threadIdx.x] = input[row * in_features + ph * 16 + threadIdx.x];
        } else {
            s_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((ph * 16 + threadIdx.y) < in_features && col < out_features) {
            s_weights[threadIdx.y][threadIdx.x] = weights[(ph * 16 + threadIdx.y) * out_features + col];
        } else {
            s_weights[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += s_input[threadIdx.y][k] * s_weights[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < seq_len && col < out_features) {
        float b = (bias != nullptr) ? bias[col] : 0.0f;
        output[row * out_features + col] = sum + b;
    }
}

__global__
void linear_gemm_transposed_batched(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int in_features,
    int out_features,
    int seq_len
) {
    __shared__ float s_input[16][16];
    __shared__ float s_weights[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (in_features + 16 - 1) / 16; ++ph) {
        if (row < seq_len && (ph * 16 + threadIdx.x) < in_features) {
            s_input[threadIdx.y][threadIdx.x] = input[row * in_features + ph * 16 + threadIdx.x];
        } else {
            s_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int w_row = blockIdx.x * 16 + threadIdx.y;
        int w_col = ph * 16 + threadIdx.x;
        if (w_row < out_features && w_col < in_features) {
            s_weights[threadIdx.y][threadIdx.x] = weights[w_row * in_features + w_col];
        } else {
            s_weights[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += s_input[threadIdx.y][k] * s_weights[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (row < seq_len && col < out_features) {
        float b = (bias != nullptr) ? bias[col] : 0.0f;
        output[row * out_features + col] = sum + b;
    }
}

void linear_forward( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features,
    int seq_len
){ 
    if (seq_len == 1) {
        int threads_per_block = 256;  
        int blocks = (out_features * 32 + threads_per_block - 1) / threads_per_block; 
        linear_gemv_warp_reduced<<<blocks, threads_per_block>>>(input, weights, bias, output, in_features, out_features); 
    } else {
        dim3 block(16, 16);
        dim3 grid((out_features + 16 - 1) / 16, (seq_len + 16 - 1) / 16);
        linear_gemm_batched<<<grid, block>>>(input, weights, bias, output, in_features, out_features, seq_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); 
} 

void linear_forward_transposed( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features,
    int seq_len
){ 
    if (seq_len == 1) {
        int threads_per_block = 256;  
        int blocks = (out_features * 32 + threads_per_block - 1) / threads_per_block; 
        linear_gemv_transposed_warp_reduced<<<blocks, threads_per_block>>>(input, weights, bias, output, in_features, out_features); 
    } else {
        dim3 block(16, 16);
        dim3 grid((out_features + 16 - 1) / 16, (seq_len + 16 - 1) / 16);
        linear_gemm_transposed_batched<<<grid, block>>>(input, weights, bias, output, in_features, out_features, seq_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); 
}
