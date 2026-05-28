#pragma once 

#include <string> 
#include <vector> 
#include "tensor.h" 

//architectural params for gpt2 (124m) 
struct GPT2Config { 
    int vocab_size = 50257; 
    int n_positions = 1024; 
    int n_embd = 768; 
    int n_layer = 12; 
    int n_head = 12;
}; 

//weights for single transformer layer (block) 
struct BlockWeights {
    // Attention sub-layer
    Tensor ln_1_weight; //layernorm
    Tensor ln_1_bias;
    Tensor c_attn_weight;
    Tensor c_attn_bias;
    Tensor c_proj_weight;
    Tensor c_proj_bias;
    
    // MLP sub-layer
    Tensor ln_2_weight; //layernorm
    Tensor ln_2_bias;
    Tensor c_fc_weight;
    Tensor c_fc_bias;
    Tensor c_proj_weight_mlp;
    Tensor c_proj_bias_mlp;
}; 

//complete model weights
struct GPT2Weights {
    Tensor wte;                     // Token embeddings: [vocab_size, n_embd]
    Tensor wpe;                     // Position embeddings: [n_positions, n_embd]
    std::vector<BlockWeights> blocks; // Transformer layers
    Tensor ln_f_weight;             // Final LayerNorm: [n_embd]
    Tensor ln_f_bias;               // Final LayerNorm: [n_embd]
    Tensor lm_head;                 // LM head projection: [vocab_size, n_embd]
};


//pre-allocating intermediate gpu buffer(activation cache) 
//it allocates cudamalloc once during model load 
struct GPT2Activations {
    Tensor x;               // Main stream: [batch_size, seq_len, n_embd]
    Tensor residual;        // Temp buffer for residual adds: [batch_size, seq_len, n_embd]
    Tensor ln_out;          // Temp buffer for layernorms: [batch_size, seq_len, n_embd]
    Tensor qkv;             // QKV projections: [batch_size, seq_len, 3 * n_embd]
    Tensor attn_scores;     // Query-Key similarity scores: [n_head, seq_len, seq_len]
    Tensor attn_out;        // Attention output: [batch_size, seq_len, n_embd]
    Tensor mlp_hidden;      // MLP intermediate state: [batch_size, seq_len, 4 * n_embd]
    Tensor logits;          // Output vocabulary distribution: [batch_size, seq_len, vocab_size]
}; 


struct KVCache {
    Tensor key_cache;
    Tensor value_cache;
};

//model class
class GPT2Model {
public:
    GPT2Config config;
    GPT2Weights weights;
    GPT2Activations activations;
    std::vector<KVCache> kv_caches;
    int past_seq_len;

    // Constructor/Destructor handles setting up GPU memory
    GPT2Model(const GPT2Config& cfg);
    ~GPT2Model();
    // Loads weights from the exported directory of binary weights
    void load_weights(const std::string& weights_dir);
    // Executes the forward pass graph and returns the GPU pointer to the logits
    float* forward(const int* input_tokens, int batch_size, int seq_len);
private:
    void allocate_weights();
    void allocate_activations(int max_batch_size, int max_seq_len);
    void free_weights();
    void free_activations();
};