#include "../include/tensor.h" 
#include "../include/cuda_utils.h" 

Tensor create_gpu_tensor( 
    std::vector<int> shape
){ 
    Tensor t;  
    t.shape = shape; 
    t.numel = 1; 

    for (int s : shape) 
        t.numel *= s;  

    t.device = GPU;  

    CUDA_CHECK(
                cudaMalloc(&t.data,  
                            t.numel * sizeof(float))); 

    return t;

}