#include "../include/cuda_utils.h" 

__global__  
void embedding_lookup_kernel( 
    float* embedding_table, 
    int* token_ids, 
    float* output, 
    int hidden_size
)   { 
   
    int idx = threadIdx.x;

    int token = token_ids[0]; 

    output[idx] =  
        embedding_table[token * hidden_size + idx];
} 

void embedding_lookup(
    float* embedding_table,
    int* token_ids,
    float* output,
    int hidden_size
) {

    embedding_lookup_kernel<<<1, hidden_size>>>(
        embedding_table,
        token_ids,
        output,
        hidden_size
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}