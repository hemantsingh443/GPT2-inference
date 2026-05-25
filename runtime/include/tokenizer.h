#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer() = default;

    // Loads vocabulary and merges from binary files
    void load(const std::string& vocab_path, const std::string& merges_path);

    // Encodes a text string into a list of token IDs
    std::vector<int> encode(const std::string& text);

    // Decodes a list of token IDs back into a text string
    std::string decode(const std::vector<int>& tokens);

private:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> bpe_ranks;
    std::unordered_map<std::string, uint8_t> utf8_to_byte;

    // Helper to generate byte-to-utf8 mapping
    std::string byte_to_utf8(uint8_t b);
};
