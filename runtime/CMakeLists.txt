cmake_minimum_required(VERSION 3.18)

project(gpt_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)
enable_language(CUDA) 

add_executable(
    gpt_runtime

    src/main.cpp
    src/tensor.cpp 
    src/gpt2.cpp

    kernels/embedding.cu
    kernels/linear.cu
    kernels/layernorm.cu  
    kernels/attention.cu
    kernels/residual.cu 
    kernels/gelu.cu 
    

)

target_link_libraries(
    gpt_runtime
    cudart
)

add_executable(
    test_runner

    tests/test_runner.cu
    src/tensor.cpp 
    src/gpt2.cpp

    kernels/embedding.cu
    kernels/linear.cu
    kernels/layernorm.cu 
    kernels/attention.cu
    kernels/residual.cu 
    kernels/gelu.cu
)

target_link_libraries(
    test_runner
    cudart
)

set_target_properties(
    gpt_runtime test_runner
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional

set_target_properties(
    gpt_runtime test_runner
    PROPERTIES
    CUDA_ARCHITECTURES native
)