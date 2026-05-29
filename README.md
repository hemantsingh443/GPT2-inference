# GPT-2 Inference from Scratch in C++ and CUDA

This is a learning project to build a high-performance GPT-2 (124M parameter) inference engine completely from the ground up in C++ and CUDA. 

Instead of relying on heavy frameworks like PyTorch or ONNX, all layers (embeddings, layernorm, attention, gelu, and linear projections) are written in raw C++ and custom CUDA kernels.

To learn more about the step-by-step optimization journey and benchmark results, check out the live write-up: [https://hemantsingh443.github.io/blogs/](https://hemantsingh443.github.io/blogs/)

---

## How to Run It

### Prerequisites
* CUDA Toolkit 11.x or higher
* C++17 compatible compiler (e.g., MSVC, GCC)
* CMake 3.18 or higher
* Python 3.x with `transformers`, `numpy`, and `torch` (for weight extraction)

### 1. Download and Convert Weights
Run the weight initialization script to download pre-trained weights from Hugging Face and convert them into flat binary float32 arrays:
```bash
pip install numpy torch transformers
python init_model.py
```
This will generate a `weights/` directory containing the serialized parameter weights and the vocabulary/merges binary files.

### 2. Build the C++ Runtime
Compile the runtime using CMake:
```bash
cd runtime
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
This builds three executables:
* `gpt_runtime`: The main autoregressive generation loop.
* `test_runner`: Unit tests verifying kernel correctness.
* `benchmark_runner`: Performance benchmarks for individual layers and the end-to-end model.

### 3. Run Inference
To generate text, run the executable from the build directory (since paths to weights are relative to the execution location):

**On Linux / macOS:**
```bash
./build/gpt_runtime "Your prompt goes here"
```

**On Windows:**
```cmd
.\build\Release\gpt_runtime.exe "Your prompt goes here"
```
