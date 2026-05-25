#include "../include/gpt2.h"
#include "../include/cuda_utils.h"
#include "../include/embedding.h"
#include "../include/linear.h"
#include "../include/layernorm.h"
#include "../include/attention.h"
#include "../include/residual.h"
#include "../include/gelu.h"

// CUDA kernel to perform embedding lookup and sum token + position embeddings
__global__
void embedding_lookup_and_sum_kernel(
    const int* token_ids,
    float* wte,
    float* wpe,
    float* output,
    int seq_len,
    int n_embd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * n_embd;
    if (idx < total_elements) {
        int pos = idx / n_embd;
        int emb_dim = idx % n_embd;
        int token_id = token_ids[pos];
        
        float token_val = wte[token_id * n_embd + emb_dim];
        float pos_val = wpe[pos * n_embd + emb_dim];
        output[idx] = token_val + pos_val;
    }
}

float* GPT2Model::forward(const int* input_tokens, int batch_size, int seq_len) {
    // Note: Assumes batch_size = 1 for simplicity in this implementation
    int n_embd = config.n_embd;
    
    // Copy input tokens to temporary GPU memory
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, input_tokens, seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    // Compute token + position embeddings: lookup & sum
    int total_elements = seq_len * n_embd;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    embedding_lookup_and_sum_kernel<<<blocks, threads_per_block>>>(
        d_tokens,
        weights.wte.data,
        weights.wpe.data,
        activations.x.data,
        seq_len,
        n_embd
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Free the temporary token ID GPU memory
    cudaFree(d_tokens);
    
    // Loop through the transformer layers (blocks)
    for (int l = 0; l < config.n_layer; ++l) {
        BlockWeights& block = weights.blocks[l];
        
        // Attention block
        // Save residual input: x -> residual
        CUDA_CHECK(cudaMemcpy(
            activations.residual.data,
            activations.x.data,
            seq_len * n_embd * sizeof(float),
            cudaMemcpyDeviceToDevice
        ));
        
        // LayerNorm 1: x -> ln_out
        for (int s = 0; s < seq_len; ++s) {
            layernorm_forward(
                activations.x.data + s * n_embd,
                block.ln_1_weight.data,
                block.ln_1_bias.data,
                activations.ln_out.data + s * n_embd,
                n_embd,
                1e-5f
            );
        }
        
        // Projection to QKV: ln_out @ c_attn -> qkv
        for (int s = 0; s < seq_len; ++s) {
            linear_forward(
                activations.ln_out.data + s * n_embd,
                block.c_attn_weight.data,
                block.c_attn_bias.data,
                activations.qkv.data + s * 3 * n_embd,
                n_embd,
                3 * n_embd
            );
        }
        
        // Self-Attention calculation: qkv -> attn_out
        attention_forward(
            activations.qkv.data,
            activations.attn_out.data,
            seq_len,
            config.n_head,
            n_embd / config.n_head
        );
        
        // Attention Output Projection: attn_out @ c_proj -> x
        for (int s = 0; s < seq_len; ++s) {
            linear_forward(
                activations.attn_out.data + s * n_embd,
                block.c_proj_weight.data,
                block.c_proj_bias.data,
                activations.x.data + s * n_embd,
                n_embd,
                n_embd
            );
        }
        
        // Add residual: x += residual
        residual_add(
            activations.x.data,
            activations.residual.data,
            seq_len * n_embd
        );
        
        // MLP block
        // Save residual input: x -> residual
        CUDA_CHECK(cudaMemcpy(
            activations.residual.data,
            activations.x.data,
            seq_len * n_embd * sizeof(float),
            cudaMemcpyDeviceToDevice
        ));
        
        // LayerNorm 2: x -> ln_out
        for (int s = 0; s < seq_len; ++s) {
            layernorm_forward(
                activations.x.data + s * n_embd,
                block.ln_2_weight.data,
                block.ln_2_bias.data,
                activations.ln_out.data + s * n_embd,
                n_embd,
                1e-5f
            );
        }
        
        // MLP First Projection: ln_out @ c_fc -> mlp_hidden
        for (int s = 0; s < seq_len; ++s) {
            linear_forward(
                activations.ln_out.data + s * n_embd,
                block.c_fc_weight.data,
                block.c_fc_bias.data,
                activations.mlp_hidden.data + s * 4 * n_embd,
                n_embd,
                4 * n_embd
            );
        }
        
        // GELU activation: mlp_hidden = gelu(mlp_hidden)
        gelu_forward(
            activations.mlp_hidden.data,
            seq_len * 4 * n_embd
        );
        
        // MLP Output Projection: mlp_hidden @ c_proj_mlp -> x
        for (int s = 0; s < seq_len; ++s) {
            linear_forward(
                activations.mlp_hidden.data + s * 4 * n_embd,
                block.c_proj_weight_mlp.data,
                block.c_proj_bias_mlp.data,
                activations.x.data + s * n_embd,
                4 * n_embd,
                n_embd
            );
        }
        
        // Add residual: x += residual
        residual_add(
            activations.x.data,
            activations.residual.data,
            seq_len * n_embd
        );
    }
    
    // Final layer normalization: x -> ln_out
    for (int s = 0; s < seq_len; ++s) {
        layernorm_forward(
            activations.x.data + s * n_embd,
            weights.ln_f_weight.data,
            weights.ln_f_bias.data,
            activations.ln_out.data + s * n_embd,
            n_embd,
            1e-5f
        );
    }
    
    // Vocabulary output projection (LM Head): ln_out @ lm_head^T -> logits
    for (int s = 0; s < seq_len; ++s) {
        linear_forward_transposed(
            activations.ln_out.data + s * n_embd,
            weights.lm_head.data,
            nullptr,
            activations.logits.data + s * config.vocab_size,
            n_embd,
            config.vocab_size
        );
    }
    
    return activations.logits.data;
}
