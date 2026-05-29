#pragma once 

void attention_forward( 
    float* qkv, 
    float* key_cache,
    float* value_cache,
    float* temp_output,
    float* temp_stats,
    float* output, 
    int seq_len, 
    int past_seq_len,
    int num_heads, 
    int head_dim 
); 

void update_kv_cache(
    float* qkv,
    float* key_cache,
    float* value_cache,
    int n_embd,
    int seq_len,
    int past_seq_len
); 