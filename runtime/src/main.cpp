#include <fstream> 
#include <iostream> 
#include <vector> 

#include "../include/tensor.h"  
#include "../include/cuda_utils.h" 
#include "../include/embedding.h"

Tensor create_gpu_tensor( 
    std::vector<int> shape
); 

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

        for (float x : verify)
           std::cout << x << "\n"; 
}