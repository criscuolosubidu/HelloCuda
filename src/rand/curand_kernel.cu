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
#include <curand_kernel.h>


__global__ void generate_random(float *out, int n, int m, unsigned long long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x * m + y;
    if (x < n && y < m) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        out[idx] = curand_uniform(&state);
    }
}

int main() {
    std::cout << "Start Generating!" << std::endl;
    constexpr int N = 32768; // 2 ^ 15

    std::vector<float> h_x(N * N);

    auto start_gen = std::chrono::high_resolution_clock::now();

    dim3 gridDim(N / 16, N / 16);
    dim3 blockDim(16, 16);

    float *d_x;
    cudaMalloc(&d_x, N * N * sizeof(float));

    generate_random<<<gridDim, blockDim>>>(d_x, N, N, 123456);
    cudaDeviceSynchronize();

    cudaMemcpy(h_x.data(), d_x, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

    std::cout << "GPU Random Number Generate Time : " << gen_time << " ms" << std::endl; // 4513ms
}