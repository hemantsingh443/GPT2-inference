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

    kernels/embedding.cu
    kernels/linear.cu
)

target_link_libraries(
    gpt_runtime
    cudart
)

set_target_properties(
    gpt_runtime
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional

set_target_properties(
    gpt_runtime
    PROPERTIES
    CUDA_ARCHITECTURES native
)