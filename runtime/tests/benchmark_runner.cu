#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"
#include "../include/linear.h"

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
        for (int s = 0; s < config.seq_len; ++s) {
            linear_forward_transposed(
                d_input + s * config.in_features,
                d_weights,
                d_bias,
                d_output + s * config.out_features,
                config.in_features,
                config.out_features
            );
        }
    } else {
        for (int s = 0; s < config.seq_len; ++s) {
            linear_forward(
                d_input + s * config.in_features,
                d_weights,
                d_bias,
                d_output + s * config.out_features,
                config.in_features,
                config.out_features
            );
        }
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
            for (int s = 0; s < config.seq_len; ++s) {
                linear_forward_transposed(
                    d_input + s * config.in_features,
                    d_weights,
                    d_bias,
                    d_output + s * config.out_features,
                    config.in_features,
                    config.out_features
                );
            }
        } else {
            for (int s = 0; s < config.seq_len; ++s) {
                linear_forward(
                    d_input + s * config.in_features,
                    d_weights,
                    d_bias,
                    d_output + s * config.out_features,
                    config.in_features,
                    config.out_features
                );
            }
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

    std::cout << "=========================================================================================\n";
    return 0;
}
