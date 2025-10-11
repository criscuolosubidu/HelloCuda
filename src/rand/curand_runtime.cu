#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <thread>
#include <curand.h>

// Scale uniform [0,1) to [min, max)
__global__ void scale_uniform_kernel(float *data, int N, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * (max_val - min_val) + min_val;
    }
}

int main() {
    constexpr int N = 1073741824; // 2 ^ 30
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    float h_y;
    std::vector<float> h_x(N);

    auto start_gen = std::chrono::high_resolution_clock::now();

    float *d_data, *h_data;

    cudaMalloc(&d_data, N * sizeof(float));
    h_data = new float[N];

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1233ULL);
    curandGenerateUniform(gen, d_data, N);

    // choose to scale
    constexpr float MIN_VAL = 0.0f;
    constexpr float MAX_VAL = 0.01f;
    int num_blocks = (N + 255) / 256;
    scale_uniform_kernel<<<num_blocks, 256>>>(d_data, N, MIN_VAL, MAX_VAL);

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    curandDestroyGenerator(gen);
    cudaFree(d_data);
    delete[] h_data;

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

    std::cout << "GPU Random Number Generate Time : " << gen_time << " ms" << std::endl; // 943 ms
}