#pragma once 

void linear_forward( 
    float* input, 
    float* weights, 
    float* bias, 
    float* output, 
    int in_features, 
    int out_features
); 

void linear_forward_transposed(
    float* input,
    float* weights,
    float* bias,
    float* output,
    int in_features,
    int out_features
);
