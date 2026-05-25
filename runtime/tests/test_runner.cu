#include <fstream> 
#include <iostream> 
#include <vector> 
#include <cmath>

#include "../include/tensor.h"  
#include "../include/cuda_utils.h" 
#include "../include/embedding.h"
#include "../include/linear.h"
#include "../include/layernorm.h"

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
    embedding_lookup(wte.data, d_token, output.data, 768);

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

    if (emb_passed && fc_passed && ln_passed) {
        std::cout << "All tests PASSED!\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED!\n";
        return 1;
    }
}
