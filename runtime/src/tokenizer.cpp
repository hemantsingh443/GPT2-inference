#include "../include/tokenizer.h"
#include <fstream>
#include <iostream>
#include <cstdint>

Tokenizer::Tokenizer() {
    // Populate the reverse UTF-8 mapping
    for (int i = 0; i < 256; ++i) {
        utf8_to_byte[byte_to_utf8(i)] = (uint8_t)i;
    }
}

// Maps raw bytes (0-255) to the special unicode characters used in GPT-2 BPE
std::string Tokenizer::byte_to_utf8(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        return std::string(1, (char)b);
    }
    int n = 0;
    for (int i = 0; i < b; ++i) {
        if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255))) {
            n++;
        }
    }
    int cp = 256 + n;
    std::string s = "";
    s += (char)(0xC0 | (cp >> 6));
    s += (char)(0x80 | (cp & 0x3F));
    return s;
}

void Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // 1. Load Vocab binary file
    std::ifstream v_file(vocab_path, std::ios::binary);
    if (!v_file) {
        std::cerr << "Failed to open vocab file: " << vocab_path << "\n";
        exit(1);
    }
    int32_t vocab_size;
    v_file.read((char*)&vocab_size, sizeof(vocab_size));
    
    id_to_token.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        int32_t len;
        v_file.read((char*)&len, sizeof(len));
        std::string s(len, '\0');
        v_file.read(&s[0], len);
        id_to_token[i] = s;
        token_to_id[s] = i;
    }

    // 2. Load Merges binary file
    std::ifstream m_file(merges_path, std::ios::binary);
    if (!m_file) {
        std::cerr << "Failed to open merges file: " << merges_path << "\n";
        exit(1);
    }
    int32_t merges_count;
    m_file.read((char*)&merges_count, sizeof(merges_count));
    
    for (int i = 0; i < merges_count; ++i) {
        int32_t p_len, c_len;
        
        m_file.read((char*)&p_len, sizeof(p_len));
        std::string parent(p_len, '\0');
        m_file.read(&parent[0], p_len);
        
        m_file.read((char*)&c_len, sizeof(c_len));
        std::string child(c_len, '\0');
        m_file.read(&child[0], c_len);
        
        std::string pair_str = parent + " " + child;
        bpe_ranks[pair_str] = i;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<std::string> parts;
    // Map each raw byte in input text to its unicode character
    for (size_t i = 0; i < text.size(); ++i) {
        uint8_t b = (uint8_t)text[i];
        parts.push_back(byte_to_utf8(b));
    }
    
    if (parts.size() < 2) {
        std::vector<int> ids;
        for (const auto& p : parts) {
            ids.push_back(token_to_id[p]);
        }
        return ids;
    }
    
    while (true) {
        // Find the adjacent pair of tokens with the lowest merge rank
        int min_rank = 1e9;
        int min_idx = -1;
        
        for (size_t i = 0; i < parts.size() - 1; ++i) {
            std::string pair_str = parts[i] + " " + parts[i+1];
            auto it = bpe_ranks.find(pair_str);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                min_idx = i;
            }
        }
        
        if (min_idx == -1) break; // Stop if no merge rules apply
        
        // Merge the selected pair
        std::vector<std::string> new_parts;
        std::string p0 = parts[min_idx];
        std::string p1 = parts[min_idx+1];
        
        for (size_t i = 0; i < parts.size(); ) {
            if (i < parts.size() - 1 && parts[i] == p0 && parts[i+1] == p1) {
                new_parts.push_back(p0 + p1);
                i += 2;
            } else {
                new_parts.push_back(parts[i]);
                i += 1;
            }
        }
        parts = new_parts;
    }
    
    // Map merged tokens to IDs
    std::vector<int> ids;
    for (const auto& p : parts) {
        ids.push_back(token_to_id[p]);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text = "";
    for (int t_id : tokens) {
        if (t_id < 0 || t_id >= (int)id_to_token.size()) continue;
        const std::string& token_str = id_to_token[t_id];
        
        // Parse token string byte-by-byte (either 1 or 2 bytes per BPE character)
        for (size_t i = 0; i < token_str.size(); ) {
            size_t len = 1;
            if ((unsigned char)token_str[i] >= 0xC0) {
                len = 2; // UTF-8 code point > 127
            }
            std::string char_utf8 = token_str.substr(i, len);
            text += (char)utf8_to_byte[char_utf8];
            i += len;
        }
    }
    return text;
}
