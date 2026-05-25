#pragma once 

void attention_forward( 
    float* qkv, 
    float* output, 
    int seq_len, 
    int num_heads, 
    int head_dim 
); 