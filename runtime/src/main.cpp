#include <fstream> 
#include <iostream> 
#include <vector> 

#include "../include/tensor.h"  
#include "../include/cuda_utils.h" 
#include "../include/embedding.h"
#include "../include/linear.h"

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

int main() { 
    
    std::ifstream file( 
        "../../weights/transformer_wte_weight.bin", 
        std::ios::binary
    ); 

    if(!file) { 
        std::cerr << "Failed to open file\n"; 
        return 1; 
    } 

    file.seekg(0, std::ios::end); 
    size_t size = file.tellg(); 
    file.seekg(0, std::ios::beg); 

    std::vector<float> host_data(size / sizeof(float)); 

    file.read( 
        (char*)host_data.data(), 
        size    
    ); 

    std::cout << "loaded " 
              << host_data.size() 
              << " floats\n"; 

    Tensor wte = create_gpu_tensor(
    {50257, 768});  

    CUDA_CHECK(
            cudaMemcpy(
                wte.data,
                host_data.data(),
                size,
                cudaMemcpyHostToDevice
            )
        ); 
    
    //token lookup 
    int token = 15496;

    int* d_token;

    CUDA_CHECK(
        cudaMalloc(
            &d_token,
            sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_token,
            &token,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));  

    Tensor output = create_gpu_tensor({768}); 

        embedding_lookup(
        wte.data,
        d_token,
        output.data,
        768
    );

     //verify by copying back to host and printing first 10 values
    std::vector<float> verify(10);

        CUDA_CHECK(
            cudaMemcpy(
                verify.data(),
                output.data,
                10 * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );

        std::cout << "Embedding output first 10 values:\n";
        for (float x : verify)
           std::cout << x << "\n"; 
        
        //Load MLP projection weights and bias
        std::vector<float> host_fc_weight = load_binary_file("../../weights/transformer_h_0_mlp_c_fc_weight.bin");
        std::vector<float> host_fc_bias = load_binary_file("../../weights/transformer_h_0_mlp_c_fc_bias.bin");
        
        //Allocate GPU tensors for weight, bias, and output
        Tensor fc_weight = create_gpu_tensor({768, 3072});
        Tensor fc_bias = create_gpu_tensor({3072});
        Tensor fc_output = create_gpu_tensor({3072});
        
        //Copy weight and bias to GPU
        CUDA_CHECK(
            cudaMemcpy(
                fc_weight.data,
                host_fc_weight.data(),
                fc_weight.numel * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
        CUDA_CHECK(
            cudaMemcpy(
                fc_bias.data,
                host_fc_bias.data(),
                fc_bias.numel * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
        
        //Run GEMM (linear_forward)
        linear_forward(
            output.data,
            fc_weight.data,
            fc_bias.data,
            fc_output.data,
            768,
            3072
        );
        
        //Copy back to host and verify the first 10 values
        std::vector<float> verify_fc(10);
        CUDA_CHECK(
            cudaMemcpy(
                verify_fc.data(),
                fc_output.data,
                10 * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );
        
        std::cout << "\nGEMM Output first 10 values:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "[" << i << "]: " << verify_fc[i] << "\n";
        }
}
