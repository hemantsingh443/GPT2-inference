#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>

#include "../include/tensor.h"  
#include "../include/cuda_utils.h" 
#include "../include/gpt2.h"
#include "../include/tokenizer.h"

Tensor create_gpu_tensor(std::vector<int> shape);

int main(int argc, char* argv[]) {
    std::string prompt = "GPT2 inference runtime in C++ and CUDA is";
    if (argc > 1) {
        prompt = "";
        for (int i = 1; i < argc; ++i) {
            prompt += argv[i];
            if (i < argc - 1) prompt += " ";
        }
    }

    std::cout << "GPT-2 Autoregressive Text Generation\n";

    std::cout << "Loading tokenizer (vocab.bin and merges.bin)..." << std::endl;
    Tokenizer tokenizer;
    tokenizer.load("../../weights/vocab.bin", "../../weights/merges.bin");

    std::cout << "Loading GPT-2 model weights from weights/ directory..." << std::endl;
    GPT2Config cfg;
    GPT2Model model(cfg);
    model.load_weights("../../weights");
    std::cout << "Model loaded successfully!\n\n";

    // Encode the prompt
    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Encoded token IDs: ";
    for (int id : tokens) {
        std::cout << id << " ";
    }
    std::cout << "\n\n";

    std::cout << "--- Generated Text ---\n";
    std::cout << prompt << std::flush;

    // Initialize random number generator for sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Autoregressive Generation Loop
    int max_new_tokens = 30;
    std::vector<float> host_logits(cfg.vocab_size);

    for (int step = 0; step < max_new_tokens; ++step) {
        float* d_logits = nullptr;
        if (step == 0) {
            d_logits = model.forward(tokens.data(), 1, tokens.size());
        } else {
            d_logits = model.forward(&tokens.back(), 1, 1);
        }

        float* last_token_logits = nullptr;
        if (step == 0) {
            last_token_logits = d_logits + (tokens.size() - 1) * cfg.vocab_size;
        } else {
            last_token_logits = d_logits;
        }

        CUDA_CHECK(cudaMemcpy(
            host_logits.data(),
            last_token_logits,
            cfg.vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        float temperature = 0.8f;
        int K = 40;

        if (temperature > 0.0f) {
            for (int v = 0; v < cfg.vocab_size; ++v) {
                host_logits[v] /= temperature;
            }
        }

        struct TokenLogit {
            int id;
            float logit;
        };
        std::vector<TokenLogit> token_logits(cfg.vocab_size);
        for (int v = 0; v < cfg.vocab_size; ++v) {
            token_logits[v] = {v, host_logits[v]};
        }

        // Get the top K tokens
        std::partial_sort(
            token_logits.begin(),
            token_logits.begin() + K,
            token_logits.end(),
            [](const TokenLogit& a, const TokenLogit& b) {
                return a.logit > b.logit;
            }
        );

        // Compute softmax over top K
        float max_logit = token_logits[0].logit;
        float sum_exp = 0.0f;
        std::vector<float> probs(K);
        for (int k = 0; k < K; ++k) {
            probs[k] = expf(token_logits[k].logit - max_logit);
            sum_exp += probs[k];
        }
        for (int k = 0; k < K; ++k) {
            probs[k] /= sum_exp;
        }

        // Categorical sampling from the top K candidates
        float r = dis(gen);
        float cumulative_prob = 0.0f;
        int next_token_id = token_logits[K - 1].id; // Fallback
        for (int k = 0; k < K; ++k) {
            cumulative_prob += probs[k];
            if (r <= cumulative_prob) {
                next_token_id = token_logits[k].id;
                break;
            }
        }

        // Check for end-of-text token (50256)
        if (next_token_id == 50256) {
            break;
        }

        // Decode and print the single token string
        std::string next_token_str = tokenizer.decode({next_token_id});
        std::cout << next_token_str << std::flush;

        // Append the new token to the sequence for the next step
        tokens.push_back(next_token_id);
    }


    std::cout << "\nGeneration finished.\n";

    return 0;
}
