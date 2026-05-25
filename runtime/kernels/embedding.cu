#include "../include/cuda_utils.h" 

__global__  
void embedding_lookup_kernel( 
    float* embedding_table, 
    int* token_ids, 
    float* output, 
    int seq_len,
    int hidden_size
)   { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * hidden_size;
    
    if (idx < total_elements) {
        int pos = idx / hidden_size;
        int emb_dim = idx % hidden_size;
        int token = token_ids[pos]; 
        
        output[idx] = embedding_table[token * hidden_size + emb_dim];
    }
} 

void embedding_lookup(
    float* embedding_table,
    int* token_ids,
    float* output,
    int seq_len,
    int hidden_size
) {
    int total_elements = seq_len * hidden_size;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    embedding_lookup_kernel<<<blocks, threads_per_block>>>(
        embedding_table,
        token_ids,
        output,
        seq_len,
        hidden_size
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}