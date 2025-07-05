#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <random>

static constexpr int BLOCK_SIZE = 32;
static constexpr int MAXN = 2048;

// this file just for test nsys

__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.x + threadIdx.y;
    if (row < N && col < N)
    {
        for (int i = 0; i < N; ++i)
        {
            C[row * N + col] += A[row * N + i] * B[i * N + col];
        }
    }
}


void NvtxMatmul()
{
    float* A = new float[MAXN * MAXN];
    float* B = new float[MAXN * MAXN];
    float* C = new float[MAXN * MAXN];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < MAXN * MAXN; ++i) A[i] = dis(gen);
    for (int i = 0; i < MAXN * MAXN; ++i) B[i] = dis(gen);

    nvtxRangePush("Matrix Multiplication");

    float *d_A, *d_B, *d_C;

    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, MAXN * MAXN * sizeof(float));
    cudaMalloc(&d_B, MAXN * MAXN * sizeof(float));
    cudaMalloc(&d_C, MAXN * MAXN * sizeof(float));
    nvtxRangePop();

    nvtxRangePush("Memory Copy, Host -> Device");
    cudaMemcpy(d_A, A, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((MAXN + BLOCK_SIZE - 1) / BLOCK_SIZE, (MAXN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, MAXN);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy, Device -> Host");
    cudaMemcpy(C, d_C, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    delete[] A;
    delete[] B;
    delete[] C;
}
