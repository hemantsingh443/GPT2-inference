cmake_minimum_required(VERSION 3.18)

project(gpt_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(
    gpt_runtime
    src/main.cpp
    src/tensor.cpp
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