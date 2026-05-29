#pragma once 

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
);