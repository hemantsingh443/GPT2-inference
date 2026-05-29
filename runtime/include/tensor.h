#pragma once 

#include <vector> 
#include <string> 

enum Device { 
    CPU, 
    GPU
}; 

struct Tensor { 
    float* data; 
    std::vector<int> shape; 
    size_t numel;  
    Device device;
}; 

Tensor create_gpu_tensor(std::vector<int> shape);
