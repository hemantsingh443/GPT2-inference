#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

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

    // Autoregressive Generation Loop
    int max_new_tokens = 30;
    std::vector<float> host_logits(cfg.vocab_size);

    for (int step = 0; step < max_new_tokens; ++step) {
        float* d_logits = model.forward(tokens.data(), 1, tokens.size());

        float* last_token_logits = d_logits + (tokens.size() - 1) * cfg.vocab_size;

        CUDA_CHECK(cudaMemcpy(
            host_logits.data(),
            last_token_logits,
            cfg.vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // Greedy decoding: find the token with the highest probability (argmax)
        int next_token_id = 0;
        float max_val = -1e20f;
        for (int v = 0; v < cfg.vocab_size; ++v) {
            if (host_logits[v] > max_val) {
                max_val = host_logits[v];
                next_token_id = v;
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
