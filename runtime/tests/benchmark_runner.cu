#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"
#include "../include/linear.h"
#include "../include/attention.h"
#include "../include/gpt2.h"

// Struct to store benchmark parameters
struct BenchmarkConfig {
    std::string name;
    int seq_len;
    int in_features;
    int out_features;
    bool transposed;
};

void run_benchmark(const BenchmarkConfig& config) {
    // Total runs of the benchmark
    int num_runs = 50;
    
    // Allocate host memory
    size_t input_size = config.seq_len * config.in_features * sizeof(float);
    size_t weights_size = config.in_features * config.out_features * sizeof(float);
    size_t bias_size = config.out_features * sizeof(float);
    size_t output_size = config.seq_len * config.out_features * sizeof(float);
    
    std::vector<float> h_input(config.seq_len * config.in_features, 0.5f);
    std::vector<float> h_weights(config.in_features * config.out_features, 0.01f);
    std::vector<float> h_bias(config.out_features, 0.1f);
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_weights = nullptr;
    float* d_bias = nullptr;
    float* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_size));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weights_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));
    
    // Warmup execution
    if (config.transposed) {
        linear_forward_transposed(
            d_input,
            d_weights,
            d_bias,
            d_output,
            config.in_features,
            config.out_features,
            config.seq_len
        );
    } else {
        linear_forward(
            d_input,
            d_weights,
            d_bias,
            d_output,
            config.in_features,
            config.out_features,
            config.seq_len
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Execute benchmark loop
    for (int run = 0; run < num_runs; ++run) {
        if (config.transposed) {
            linear_forward_transposed(
                d_input,
                d_weights,
                d_bias,
                d_output,
                config.in_features,
                config.out_features,
                config.seq_len
            );
        } else {
            linear_forward(
                d_input,
                d_weights,
                d_bias,
                d_output,
                config.in_features,
                config.out_features,
                config.seq_len
            );
        }
    }
    
    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));
    
    // Wait for all operations to complete
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate total and average elapsed time
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_runs;
    
    // Compute total floating point operations
    double flops = 2.0 * config.seq_len * config.in_features * config.out_features;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    
    // Print row of benchmark table
    std::cout << std::left << std::setw(15) << config.name
              << std::setw(10) << config.seq_len
              << std::setw(15) << config.in_features
              << std::setw(15) << config.out_features
              << std::setw(12) << (config.transposed ? "Yes" : "No")
              << std::setw(15) << avg_ms
              << std::setw(15) << gflops << "\n";
              
    // Clean up resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

void run_attention_benchmark(int seq_len, int past_seq_len) {
    int num_runs = 50;
    int num_heads = 12;
    int head_dim = 64;
    int n_embd = num_heads * head_dim;
    int total_seq_len = past_seq_len + seq_len;

    size_t qkv_size = seq_len * 3 * n_embd * sizeof(float);
    size_t key_cache_size = total_seq_len * n_embd * sizeof(float);
    size_t value_cache_size = total_seq_len * n_embd * sizeof(float);
    size_t output_size = seq_len * n_embd * sizeof(float);

    std::vector<float> h_qkv(seq_len * 3 * n_embd, 0.5f);
    std::vector<float> h_key(total_seq_len * n_embd, 0.1f);
    std::vector<float> h_val(total_seq_len * n_embd, 0.2f);

    float* d_qkv = nullptr;
    float* d_key = nullptr;
    float* d_val = nullptr;
    float* d_output = nullptr;
    float* d_temp_output = nullptr;
    float* d_temp_stats = nullptr;

    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_key, key_cache_size));
    CUDA_CHECK(cudaMalloc(&d_val, value_cache_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_temp_output, num_heads * 8 * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_stats, num_heads * 8 * 2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv.data(), qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), key_cache_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), value_cache_size, cudaMemcpyHostToDevice));

    // Warmup execution
    attention_forward(d_qkv, d_key, d_val, d_temp_output, d_temp_stats, d_output, seq_len, past_seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs; ++run) {
        attention_forward(d_qkv, d_key, d_val, d_temp_output, d_temp_stats, d_output, seq_len, past_seq_len, num_heads, head_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_runs;

    std::cout << std::left << std::setw(15) << "Attention"
              << std::setw(10) << seq_len
              << std::setw(15) << past_seq_len
              << std::setw(15) << "-"
              << std::setw(12) << "-"
              << std::setw(15) << avg_ms
              << std::setw(15) << "-" << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp_output));
    CUDA_CHECK(cudaFree(d_temp_stats));
}

void run_model_forward_benchmark() {
    std::cout << "\n=========================================================================================\n";
    std::cout << "                         Full Model Forward Pass Benchmark                              \n";
    std::cout << "=========================================================================================\n";
    
    // Detect weights directory path automatically
    std::string weights_dir = "../../weights";
    {
        std::ifstream test_file("weights/vocab.bin");
        if (test_file.good()) {
            weights_dir = "weights";
        }
    }

    GPT2Config cfg;
    GPT2Model model(cfg);
    model.load_weights(weights_dir);
    
    std::vector<int> tokens(1024, 15496);
    
    // Warmup model execution
    float* d_logits = model.forward(tokens.data(), 1, 1024);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int num_runs = 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Measure prefill latency
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs; ++run) {
        model.past_seq_len = 0;
        model.forward(tokens.data(), 1, 1024);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float prefill_avg_ms = total_ms / num_runs;
    
    // Warm up model cache to prepare for decoding
    model.past_seq_len = 0;
    model.forward(tokens.data(), 1, 1023);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure generation/decoding step latency
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs * 10; ++run) {
        model.forward(&tokens[0], 1, 1);
        model.past_seq_len = 1023;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float gen_avg_ms = total_ms / (num_runs * 10);
    
    std::cout << "Full Model Prefill (1024 tokens): " << prefill_avg_ms << " ms\n";
    std::cout << "Full Model Decode Step (1024th token): " << gen_avg_ms << " ms\n";
    std::cout << "=========================================================================================\n";
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "=========================================================================================\n";
    std::cout << "                         Model GEMM/GEMV Performance Benchmark                            \n";
    std::cout << "=========================================================================================\n";
    std::cout << std::left << std::setw(15) << "Layer"
              << std::setw(10) << "SeqLen"
              << std::setw(15) << "InFeatures"
              << std::setw(15) << "OutFeatures"
              << std::setw(12) << "Transposed"
              << std::setw(15) << "Latency (ms)"
              << std::setw(15) << "GFLOPS" << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    // Standard configurations representing the model layers
    std::vector<BenchmarkConfig> configs = {
        // Generation phase configurations
        {"Attention QKV",  1, 768, 2304,  false},
        {"Attention Proj", 1, 768, 768,   false},
        {"MLP Expansion",  1, 768, 3072,  false},
        {"Vocabulary Proj",1, 768, 50257, true},

        // Short sequence configurations
        {"Attention QKV",  8, 768, 2304,  false},
        {"Attention Proj", 8, 768, 768,   false},
        {"MLP Expansion",  8, 768, 3072,  false},
        {"Vocabulary Proj",8, 768, 50257, true},

        // Medium sequence configurations
        {"Attention QKV",  128, 768, 2304,  false},
        {"Attention Proj", 128, 768, 768,   false},
        {"MLP Expansion",  128, 768, 3072,  false},
        {"Vocabulary Proj",128, 768, 50257, true},

        // Large sequence configurations
        {"Attention QKV",  512, 768, 2304,  false},
        {"Attention Proj", 512, 768, 768,   false},
        {"MLP Expansion",  512, 768, 3072,  false},
        {"Vocabulary Proj",512, 768, 50257, true},

        // Full context size configurations
        {"Attention QKV",  1024, 768, 2304,  false},
        {"Attention Proj", 1024, 768, 768,   false},
        {"MLP Expansion",  1024, 768, 3072,  false},
        {"Vocabulary Proj",1024, 768, 50257, true}
    };

    for (const auto& config : configs) {
        run_benchmark(config);
    }

    std::cout << "-----------------------------------------------------------------------------------------\n";
    
    run_attention_benchmark(1024, 0);
    run_attention_benchmark(1, 1023);

    std::cout << "=========================================================================================\n";
    
    run_model_forward_benchmark();
    return 0;
}
