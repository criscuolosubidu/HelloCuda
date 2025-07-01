#include <assert.h>
#include <iostream>
#include "cuda_memory.h"
#include <vector>
#include <random>
#include <chrono>

static constexpr int M = 256;
static constexpr int K = 512;
static constexpr int N = 256;
static constexpr int BLOCK_SIZE = 32;

// [M, K] @ [K, N] => [M, N]

void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int m, int k, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0;
            for (int x = 0; x < k; ++x)
            {
                sum += A[i * k + x] * B[x * n + j]; // A[i][x] * B[x][j]
            }
            C[i * n + j] = sum;
        }
    }
}


__global__ void matmul_kernel(const float *A, const float *B, float *C, int m, int k, int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col]; // A[row][i], B[i][col]
        }
        C[row * n + col] = sum;
    }
}


void matmul_gpu(const CudaMemory<float>& d_A, const CudaMemory<float>& d_B, CudaMemory<float>& d_C, int m, int k, int n)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE); // gpu 先分 x, 再分 y
    matmul_kernel<<<gridDim, blockDim>>>(d_A.get(), d_B.get(), d_C.get(), m, k, n);
    cudaDeviceSynchronize();
}


void matmul_init_vector(std::vector<float>& vec)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : vec) v = dis(gen);
}


void matrix_mul()
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_h(M * N), C_d(M * N);
    matmul_init_vector(A);
    matmul_init_vector(B);

    CudaMemory<float> d_A(M * K);
    CudaMemory<float> d_B(K * N);
    CudaMemory<float> d_C(M * N);

    cudaMemcpy(d_A.get(), A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.get(), B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        matmul_cpu(A, B, C_h, M, K, N);
        matmul_gpu(d_A, d_B, d_C, M, K, N);
    }

    std::cout << "测试CPU计算速度" << std::endl;
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_cpu(A, B, C_h, M, K, N);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_time += std::chrono::duration<double>(end - start).count();
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    std::cout << "测试GPU计算速度" << std::endl;
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_gpu(d_A, d_B, d_C, M, K, N);
        auto end = std::chrono::high_resolution_clock::now();
        gpu_total_time += std::chrono::duration<double>(end - start).count();
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(C_d.data(), d_C.get(), sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M * N; ++i)
    {
        if (fabs(C_h[i] - C_d[i]) > 1e-3)
        {
            std::cout << "GPU计算结果和CPU计算结果不一样，GPU : " << C_d[i] << " CPU : " << C_h[i] << std::endl;
            return;
        }
    }

    std::cout << "CPU平均时间: " << (cpu_avg_time * 1e6) << " 微秒" << std::endl;
    std::cout << "GPU平均时间: " << (gpu_avg_time * 1e6) << " 微秒" << std::endl;
    std::cout << "加速比: " << (cpu_avg_time / gpu_avg_time) << "x" << std::endl;
}
