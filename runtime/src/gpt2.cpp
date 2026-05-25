#include "../include/gpt2.h"
#include "../include/cuda_utils.h"
#include <fstream>
#include <iostream>

// Helper function to read binary weights directly to GPU memory
void load_weight_helper(Tensor& tensor, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open weight file: " << path << "\n";
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (size != tensor.numel * sizeof(float)) {
        std::cerr << "Weight file size mismatch for " << path 
                  << " (expected " << tensor.numel * sizeof(float) 
                  << " bytes, got " << size << ")\n";
        exit(1);
    }
    
    std::vector<float> host_data(tensor.numel);
    file.read((char*)host_data.data(), size);
    
    CUDA_CHECK(cudaMemcpy(tensor.data, host_data.data(), size, cudaMemcpyHostToDevice));
}

// Helper to free GPU memory
void free_gpu_tensor(Tensor& t) {
    if (t.data != nullptr) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

GPT2Model::GPT2Model(const GPT2Config& cfg) {
    config = cfg;
    allocate_weights();
    // Pre-allocate activations for batch_size = 1, max_seq_len = 1024
    allocate_activations(1, config.n_positions);
}

GPT2Model::~GPT2Model() {
    free_weights();
    free_activations();
}

void GPT2Model::allocate_weights() {
    //Embeddings
    weights.wte = create_gpu_tensor({config.vocab_size, config.n_embd});
    weights.wpe = create_gpu_tensor({config.n_positions, config.n_embd});
    
    //Transformer blocks
    weights.blocks.resize(config.n_layer);
    for (int l = 0; l < config.n_layer; ++l) {
        BlockWeights& block = weights.blocks[l];
        
        // Attention
        block.ln_1_weight = create_gpu_tensor({config.n_embd});
        block.ln_1_bias = create_gpu_tensor({config.n_embd});
        block.c_attn_weight = create_gpu_tensor({config.n_embd, 3 * config.n_embd});
        block.c_attn_bias = create_gpu_tensor({3 * config.n_embd});
        block.c_proj_weight = create_gpu_tensor({config.n_embd, config.n_embd});
        block.c_proj_bias = create_gpu_tensor({config.n_embd});
        
        // MLP
        block.ln_2_weight = create_gpu_tensor({config.n_embd});
        block.ln_2_bias = create_gpu_tensor({config.n_embd});
        block.c_fc_weight = create_gpu_tensor({config.n_embd, 4 * config.n_embd});
        block.c_fc_bias = create_gpu_tensor({4 * config.n_embd});
        block.c_proj_weight_mlp = create_gpu_tensor({4 * config.n_embd, config.n_embd});
        block.c_proj_bias_mlp = create_gpu_tensor({config.n_embd});
    }
    
    //  Final outputs
    weights.ln_f_weight = create_gpu_tensor({config.n_embd});
    weights.ln_f_bias = create_gpu_tensor({config.n_embd});
    weights.lm_head = create_gpu_tensor({config.vocab_size, config.n_embd});
}

void GPT2Model::allocate_activations(int max_batch_size, int max_seq_len) {
    activations.x = create_gpu_tensor({max_batch_size, max_seq_len, config.n_embd});
    activations.residual = create_gpu_tensor({max_batch_size, max_seq_len, config.n_embd});
    activations.ln_out = create_gpu_tensor({max_batch_size, max_seq_len, config.n_embd});
    activations.qkv = create_gpu_tensor({max_batch_size, max_seq_len, 3 * config.n_embd});
    activations.attn_scores = create_gpu_tensor({config.n_head, max_seq_len, max_seq_len});
    activations.attn_out = create_gpu_tensor({max_batch_size, max_seq_len, config.n_embd});
    activations.mlp_hidden = create_gpu_tensor({max_batch_size, max_seq_len, 4 * config.n_embd});
    activations.logits = create_gpu_tensor({max_batch_size, max_seq_len, config.vocab_size});
}

void GPT2Model::free_weights() {
    free_gpu_tensor(weights.wte);
    free_gpu_tensor(weights.wpe);
    
    for (int l = 0; l < config.n_layer; ++l) {
        BlockWeights& block = weights.blocks[l];
        free_gpu_tensor(block.ln_1_weight);
        free_gpu_tensor(block.ln_1_bias);
        free_gpu_tensor(block.c_attn_weight);
        free_gpu_tensor(block.c_attn_bias);
        free_gpu_tensor(block.c_proj_weight);
        free_gpu_tensor(block.c_proj_bias);
        
        free_gpu_tensor(block.ln_2_weight);
        free_gpu_tensor(block.ln_2_bias);
        free_gpu_tensor(block.c_fc_weight);
        free_gpu_tensor(block.c_fc_bias);
        free_gpu_tensor(block.c_proj_weight_mlp);
        free_gpu_tensor(block.c_proj_bias_mlp);
    }
    
    free_gpu_tensor(weights.ln_f_weight);
    free_gpu_tensor(weights.ln_f_bias);
    free_gpu_tensor(weights.lm_head);
}

void GPT2Model::free_activations() {
    free_gpu_tensor(activations.x);
    free_gpu_tensor(activations.residual);
    free_gpu_tensor(activations.ln_out);
    free_gpu_tensor(activations.qkv);
    free_gpu_tensor(activations.attn_scores);
    free_gpu_tensor(activations.attn_out);
    free_gpu_tensor(activations.mlp_hidden);
    free_gpu_tensor(activations.logits);
}

void GPT2Model::load_weights(const std::string& weights_dir) {
    load_weight_helper(weights.wte, weights_dir + "/transformer_wte_weight.bin");
    load_weight_helper(weights.wpe, weights_dir + "/transformer_wpe_weight.bin");
    
    for (int l = 0; l < config.n_layer; ++l) {
        std::string prefix = weights_dir + "/transformer_h_" + std::to_string(l);
        BlockWeights& block = weights.blocks[l];
        
        load_weight_helper(block.ln_1_weight, prefix + "_ln_1_weight.bin");
        load_weight_helper(block.ln_1_bias, prefix + "_ln_1_bias.bin");
        load_weight_helper(block.c_attn_weight, prefix + "_attn_c_attn_weight.bin");
        load_weight_helper(block.c_attn_bias, prefix + "_attn_c_attn_bias.bin");
        load_weight_helper(block.c_proj_weight, prefix + "_attn_c_proj_weight.bin");
        load_weight_helper(block.c_proj_bias, prefix + "_attn_c_proj_bias.bin");
        
        load_weight_helper(block.ln_2_weight, prefix + "_ln_2_weight.bin");
        load_weight_helper(block.ln_2_bias, prefix + "_ln_2_bias.bin");
        load_weight_helper(block.c_fc_weight, prefix + "_mlp_c_fc_weight.bin");
        load_weight_helper(block.c_fc_bias, prefix + "_mlp_c_fc_bias.bin");
        load_weight_helper(block.c_proj_weight_mlp, prefix + "_mlp_c_proj_weight.bin");
        load_weight_helper(block.c_proj_bias_mlp, prefix + "_mlp_c_proj_bias.bin");
    }
    
    load_weight_helper(weights.ln_f_weight, weights_dir + "/transformer_ln_f_weight.bin");
    load_weight_helper(weights.ln_f_bias, weights_dir + "/transformer_ln_f_bias.bin");
    load_weight_helper(weights.lm_head, weights_dir + "/lm_head_weight.bin");
}
