#include <fstream>
#include <vector>
#include <iostream>

int main() {

    std::ifstream file(
        "../weights/transformer_wte_weight.bin",
        std::ios::binary
    );

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();

    file.seekg(0, std::ios::beg);

    std::vector<float> data(size / sizeof(float));

    file.read((char*)data.data(), size);

    std::cout << "Loaded "
              << data.size()
              << " floats\n";
}