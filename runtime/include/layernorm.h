#pragma once 

void layernorm_forward( 
    float* input, 
    float* gamma, 
    float* beta, 
    float* output, 
    int hidden_size, 
    float eps
);