#include <fstream> 
#include <iostream> 
#include <vector> 
#include <cmath>

#include "../include/tensor.h"  
#include "../include/cuda_utils.h" 
#include "../include/embedding.h"
#include "../include/linear.h"
#include "../include/layernorm.h"
#include "../include/attention.h"  
#include "../include/residual.h"
#include "../include/gpt2.h"
#include "../include/tokenizer.h"

// Forward declaration of create_gpu_tensor (since it's defined in tensor.cpp)
Tensor create_gpu_tensor( 
    std::vector<int> shape
); 

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << path << "\n";
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read((char*)data.data(), size);
    return data;
}

bool check_close(float val, float ref, float tol = 1e-4) {
    return std::abs(val - ref) < tol;
}

int main() {
    std::cout << "=== Running GPT-2 Inference Runtime Tests ===\n\n";

    //Load Token Embedding Table & Test Embedding Lookup
    std::ifstream file("../../weights/transformer_wte_weight.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: weights/transformer_wte_weight.bin not found. Are you running from build/ directory?\n";
        return 1;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> host_wte(size / sizeof(float));
    file.read((char*)host_wte.data(), size);

    Tensor wte = create_gpu_tensor({50257, 768});
    CUDA_CHECK(cudaMemcpy(wte.data, host_wte.data(), size, cudaMemcpyHostToDevice));

    int token = 15496;
    int* d_token;
    CUDA_CHECK(cudaMalloc(&d_token, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token, &token, sizeof(int), cudaMemcpyHostToDevice));

    Tensor output = create_gpu_tensor({768});
    embedding_lookup(wte.data, d_token, output.data, 1, 768);

    std::vector<float> verify_emb(768);
    CUDA_CHECK(cudaMemcpy(verify_emb.data(), output.data, 768 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref_emb = {
        -0.0686508f, -0.132694f, 0.0112033f, -0.146658f, -0.18417f,
        -0.0357969f, -0.217333f, -0.171337f, -0.0103861f, -0.0141989f
    };
    bool emb_passed = true;
    for (int i = 0; i < 10; ++i) {
        if (!check_close(verify_emb[i], ref_emb[i])) {
            emb_passed = false;
        }
    }
    std::cout << "Test 1: Embedding Lookup -> " << (emb_passed ? "PASSED" : "FAILED") << "\n";

    //Test GEMM (linear_forward)
    std::vector<float> host_fc_weight = load_binary_file("../../weights/transformer_h_0_mlp_c_fc_weight.bin");
    std::vector<float> host_fc_bias = load_binary_file("../../weights/transformer_h_0_mlp_c_fc_bias.bin");
    Tensor fc_weight = create_gpu_tensor({768, 3072});
    Tensor fc_bias = create_gpu_tensor({3072});
    Tensor fc_output = create_gpu_tensor({3072});

    CUDA_CHECK(cudaMemcpy(fc_weight.data, host_fc_weight.data(), fc_weight.numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc_bias.data, host_fc_bias.data(), fc_bias.numel * sizeof(float), cudaMemcpyHostToDevice));

    linear_forward(output.data, fc_weight.data, fc_bias.data, fc_output.data, 768, 3072);

    std::vector<float> verify_fc(3072);
    CUDA_CHECK(cudaMemcpy(verify_fc.data(), fc_output.data, 3072 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref_fc = {
        -0.099145f, 0.179045f, 0.868187f, 1.144703f, 0.128677f,
        0.314243f, -0.196578f, 0.317911f, -0.367978f, 0.233335f
    };
    bool fc_passed = true;
    for (int i = 0; i < 10; ++i) {
        if (!check_close(verify_fc[i], ref_fc[i], 1e-3)) {
            fc_passed = false;
        }
    }
    std::cout << "Test 2: GEMM (Linear Forward) -> " << (fc_passed ? "PASSED" : "FAILED") << "\n";

    //Test LayerNorm
    std::vector<float> host_ln_weight = load_binary_file("../../weights/transformer_h_0_ln_1_weight.bin");
    std::vector<float> host_ln_bias = load_binary_file("../../weights/transformer_h_0_ln_1_bias.bin");
    Tensor ln_weight = create_gpu_tensor({768});
    Tensor ln_bias = create_gpu_tensor({768});
    Tensor ln_output = create_gpu_tensor({768});

    CUDA_CHECK(cudaMemcpy(ln_weight.data, host_ln_weight.data(), ln_weight.numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ln_bias.data, host_ln_bias.data(), ln_bias.numel * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward(output.data, ln_weight.data, ln_bias.data, ln_output.data, 768, 1e-5f);

    std::vector<float> verify_ln(768);
    CUDA_CHECK(cudaMemcpy(verify_ln.data(), ln_output.data, 768 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref_ln = {
        -0.114096f, -0.146124f, -0.052343f, -0.206682f, -0.284546f,
        -0.062076f, -0.026583f, -0.193440f, -0.018746f, -0.028776f
    };
    bool ln_passed = true;
    for (int i = 0; i < 10; ++i) {
        if (!check_close(verify_ln[i], ref_ln[i], 1e-3)) {
            ln_passed = false;
        }
    }
    std::cout << "Test 3: LayerNorm -> " << (ln_passed ? "PASSED" : "FAILED") << "\n\n"; 

    // Test Attention (attention_forward) + Output Projection
    // Load sequence [15496, 50256, 1234]
    std::vector<int> tokens = {15496, 50256, 1234};
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), 3 * sizeof(int), cudaMemcpyHostToDevice));
    // Lookup embeddings for 3 tokens
    Tensor seq_output = create_gpu_tensor({3, 768});
    embedding_lookup(wte.data, d_tokens, seq_output.data, 3, 768);
    // Load Attention QKV Projection Weights & Biases
    std::vector<float> host_attn_w = load_binary_file("../../weights/transformer_h_0_attn_c_attn_weight.bin");
    std::vector<float> host_attn_b = load_binary_file("../../weights/transformer_h_0_attn_c_attn_bias.bin");
    Tensor attn_w = create_gpu_tensor({768, 2304});
    Tensor attn_b = create_gpu_tensor({2304});
    Tensor attn_qkv = create_gpu_tensor({3, 2304});
    CUDA_CHECK(cudaMemcpy(attn_w.data, host_attn_w.data(), attn_w.numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(attn_b.data, host_attn_b.data(), attn_b.numel * sizeof(float), cudaMemcpyHostToDevice));
    // Batched linear projection to QKV
    for (int s = 0; s < 3; ++s) {
        linear_forward(seq_output.data + s * 768, attn_w.data, attn_b.data, attn_qkv.data + s * 2304, 768, 2304);
    }
    // Run Causal Self-Attention
    Tensor attn_out = create_gpu_tensor({3, 768});
    attention_forward(attn_qkv.data, attn_out.data, 3, 12, 64);
    // Project output using c_proj
    std::vector<float> host_proj_w = load_binary_file("../../weights/transformer_h_0_attn_c_proj_weight.bin");
    std::vector<float> host_proj_b = load_binary_file("../../weights/transformer_h_0_attn_c_proj_bias.bin");
    Tensor proj_w = create_gpu_tensor({768, 768});
    Tensor proj_b = create_gpu_tensor({768});
    Tensor final_attn_out = create_gpu_tensor({3, 768});
    CUDA_CHECK(cudaMemcpy(proj_w.data, host_proj_w.data(), proj_w.numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(proj_b.data, host_proj_b.data(), proj_b.numel * sizeof(float), cudaMemcpyHostToDevice));
    for (int s = 0; s < 3; ++s) {
        linear_forward(attn_out.data + s * 768, proj_w.data, proj_b.data, final_attn_out.data + s * 768, 768, 768);
    }
    // Verify first 5 values for token 0 and token 2
    std::vector<float> verify_attn(3 * 768);
    CUDA_CHECK(cudaMemcpy(verify_attn.data(), final_attn_out.data, 3 * 768 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> ref_token0 = {0.5459095f, -0.82332975f, 0.39609146f, -0.06768971f, 0.07372971f};
    std::vector<float> ref_token2 = {0.9198164f, -0.41184098f, 0.63222086f, -0.04071165f, 0.0521681f};
    bool attn_passed = true;
    for (int i = 0; i < 5; ++i) {
        if (!check_close(verify_attn[0 * 768 + i], ref_token0[i], 1e-3)) attn_passed = false;
        if (!check_close(verify_attn[2 * 768 + i], ref_token2[i], 1e-3)) attn_passed = false;
    }
    std::cout << "Test 4: Causal Self-Attention + Projection -> " << (attn_passed ? "PASSED" : "FAILED") << "\n";


        // Test Full Model Forward Pass
    std::cout << "Loading full GPT-2 model weights...\n";
    GPT2Config cfg;
    GPT2Model model(cfg);
    model.load_weights("../../weights");
    
    std::cout << "Running full model forward pass...\n";
    float* d_logits = model.forward(tokens.data(), 1, 3);
    
    std::vector<float> verify_logits(3 * cfg.vocab_size);
    CUDA_CHECK(cudaMemcpy(verify_logits.data(), d_logits, 3 * cfg.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::vector<float> ref_logits0 = {-35.236275f, -35.326614f, -38.975384f, -39.390717f, -37.65322f};
    std::vector<float> ref_logits2 = {-76.941086f, -79.28483f, -80.43526f, -80.43415f, -80.22852f};
    
    bool model_passed = true;
    for (int i = 0; i < 5; ++i) {
        if (!check_close(verify_logits[0 * cfg.vocab_size + i], ref_logits0[i], 1e-2)) model_passed = false;
        if (!check_close(verify_logits[2 * cfg.vocab_size + i], ref_logits2[i], 1e-2)) model_passed = false;
    }
    
    // Argmax check for predicted next tokens
    int pred_token0 = 0;
    int pred_token1 = 0;
    int pred_token2 = 0;
    
    float max_val0 = -1e20f;
    float max_val1 = -1e20f;
    float max_val2 = -1e20f;
    
    for (int v = 0; v < cfg.vocab_size; ++v) {
        if (verify_logits[0 * cfg.vocab_size + v] > max_val0) {
            max_val0 = verify_logits[0 * cfg.vocab_size + v];
            pred_token0 = v;
        }
        if (verify_logits[1 * cfg.vocab_size + v] > max_val1) {
            max_val1 = verify_logits[1 * cfg.vocab_size + v];
            pred_token1 = v;
        }
        if (verify_logits[2 * cfg.vocab_size + v] > max_val2) {
            max_val2 = verify_logits[2 * cfg.vocab_size + v];
            pred_token2 = v;
        }
    }
    
    std::cout << "\nPredicted next tokens:\n";
    std::cout << "Position 0 -> predicted: " << pred_token0 << " (expected: 11)\n";
    std::cout << "Position 1 -> predicted: " << pred_token1 << " (expected: 464)\n";
    std::cout << "Position 2 -> predicted: " << pred_token2 << " (expected: 262)\n";
    
    if (pred_token0 != 11 || pred_token1 != 464 || pred_token2 != 262) {
        model_passed = false;
    }
    
    std::cout << "Test 5: Full Model Forward Pass -> " << (model_passed ? "PASSED" : "FAILED") << "\n";

    // 6. Test Tokenizer
    std::cout << "\nTesting Tokenizer...\n";
    Tokenizer tokenizer;
    tokenizer.load("../../weights/vocab.bin", "../../weights/merges.bin");
    
    std::string test_str = "GPT2 inference runtime in C++ and CUDA";
    std::vector<int> encoded = tokenizer.encode(test_str);
    std::vector<int> expected_ids = {38, 11571, 17, 32278, 19124, 287, 327, 4880, 290, 29369, 5631};
    
    bool tok_passed = (encoded == expected_ids);
    std::cout << "Encoded token IDs: ";
    for (int id : encoded) std::cout << id << " ";
    std::cout << "\n";
    
    std::string decoded = tokenizer.decode(encoded);
    std::cout << "Decoded string: \"" << decoded << "\"\n";
    if (decoded != test_str) tok_passed = false;
    
    std::cout << "Test 6: Tokenizer -> " << (tok_passed ? "PASSED" : "FAILED") << "\n\n";

    // Clean up
    cudaFree(d_token);
    cudaFree(wte.data);
    cudaFree(output.data);
    cudaFree(fc_weight.data);
    cudaFree(fc_bias.data);
    cudaFree(fc_output.data);
    cudaFree(ln_weight.data);
    cudaFree(ln_bias.data);
    cudaFree(ln_output.data); 
    cudaFree(seq_output.data); 
    cudaFree(attn_w.data); 
    cudaFree(attn_b.data); 
    cudaFree(attn_qkv.data); 
    cudaFree(attn_out.data); 
    cudaFree(proj_w.data); 
    cudaFree(proj_b.data); 
    cudaFree(final_attn_out.data); 

    if (emb_passed && fc_passed && ln_passed && attn_passed && model_passed && tok_passed) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED!\n";
        return 1;
    }
}
