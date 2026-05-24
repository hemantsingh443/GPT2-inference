#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(x)                         \
do {                                          \
    cudaError_t err = x;                      \
    if (err != cudaSuccess) {                 \
        std::cerr << "CUDA Error: "           \
                  << cudaGetErrorString(err)  \
                  << std::endl;               \
        exit(1);                              \
    }                                         \
} while(0)