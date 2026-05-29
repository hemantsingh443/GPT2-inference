#include "../include/gpt2.h"
#include "../include/cuda_utils.h"
#include "../include/embedding.h"
#include "../include/linear.h"
#include "../include/layernorm.h"
#include "../include/attention.h"
#include "../include/residual.h"
#include "../include/gelu.h"

float* GPT2Model::forward(const int* input_tokens, int batch_size, int seq_len) {
    // Note: Assumes batch_size = 1 for simplicity in this implementation
    int n_embd = config.n_embd;
    
    // Copy input tokens to temporary GPU memory
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, input_tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    // Token embeddings lookup -> activations.x
    embedding_lookup(
        weights.wte.data,
        d_tokens,
        activations.x.data,
        seq_len,
        n_embd
    );

    // Position embeddings lookup -> activations.residual
    std::vector<int> pos_ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        pos_ids[i] = i;
    }
    int* d_pos_ids;
    CUDA_CHECK(cudaMalloc(&d_pos_ids, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pos_ids, pos_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));

    embedding_lookup(
        weights.wpe.data,
        d_pos_ids,
        activations.residual.data,
        seq_len,
        n_embd
    );
    
    CUDA_CHECK(cudaFree(d_pos_ids));

    // Sum WTE + WPE -> activations.x
    residual_add(
        activations.x.data,
        activations.residual.data,
        seq_len * n_embd
    );
    
    // Free the temporary token ID GPU memory
    cudaFree(d_tokens);
    
    // Loop through the transformer layers (blocks)
    for (int l = 0; l < config.n_layer; ++l) {
        BlockWeights& block = weights.blocks[l];
        
        // Attention block LayerNorm (fused residual add and copy)
        layernorm_forward(
            activations.x.data,
            activations.residual.data,
            block.ln_1_weight.data,
            block.ln_1_bias.data,
            activations.ln_out.data,
            n_embd,
            seq_len,
            l > 0,
            1e-5f
        );
        
        // Projection to QKV: ln_out @ c_attn -> qkv
        linear_forward(
            activations.ln_out.data,
            block.c_attn_weight.data,
            block.c_attn_bias.data,
            activations.qkv.data,
            n_embd,
            3 * n_embd,
            seq_len
        );
        
        update_kv_cache(
            activations.qkv.data,
            kv_caches[l].key_cache.data,
            kv_caches[l].value_cache.data,
            n_embd,
            seq_len,
            past_seq_len
        );

        // Self-Attention calculation: qkv -> attn_out
        attention_forward(
            activations.qkv.data,
            kv_caches[l].key_cache.data,
            kv_caches[l].value_cache.data,
            activations.flash_decoding_temp_output.data,
            activations.flash_decoding_temp_stats.data,
            activations.attn_out.data,
            seq_len,
            past_seq_len,
            config.n_head,
            n_embd / config.n_head
        );
        
        // Attention Output Projection: attn_out @ c_proj -> x
        linear_forward(
            activations.attn_out.data,
            block.c_proj_weight.data,
            block.c_proj_bias.data,
            activations.x.data,
            n_embd,
            n_embd,
            seq_len
        );
        
        // MLP block LayerNorm (fused residual add and copy)
        layernorm_forward(
            activations.x.data,
            activations.residual.data,
            block.ln_2_weight.data,
            block.ln_2_bias.data,
            activations.ln_out.data,
            n_embd,
            seq_len,
            true,
            1e-5f
        );
        
        // MLP First Projection: ln_out @ c_fc -> mlp_hidden
        linear_forward(
            activations.ln_out.data,
            block.c_fc_weight.data,
            block.c_fc_bias.data,
            activations.mlp_hidden.data,
            n_embd,
            4 * n_embd,
            seq_len
        );
        
        // GELU activation: mlp_hidden = gelu(mlp_hidden)
        gelu_forward(
            activations.mlp_hidden.data,
            seq_len * 4 * n_embd
        );
        
        // MLP Output Projection: mlp_hidden @ c_proj_mlp -> x
        linear_forward(
            activations.mlp_hidden.data,
            block.c_proj_weight_mlp.data,
            block.c_proj_bias_mlp.data,
            activations.x.data,
            4 * n_embd,
            n_embd,
            seq_len
        );
    }
    
    // Final layer normalization: x -> ln_out (fused with the final MLP residual addition)
    layernorm_forward(
        activations.x.data,
        activations.residual.data,
        weights.ln_f_weight.data,
        weights.ln_f_bias.data,
        activations.ln_out.data,
        n_embd,
        seq_len,
        true,
        1e-5f
    );
    
    // Vocabulary output projection (LM Head): ln_out @ lm_head^T -> logits
    linear_forward_transposed(
        activations.ln_out.data,
        weights.lm_head.data,
        nullptr,
        activations.logits.data,
        n_embd,
        config.vocab_size,
        seq_len
    );
    
    past_seq_len += seq_len;
    return activations.logits.data;
}
