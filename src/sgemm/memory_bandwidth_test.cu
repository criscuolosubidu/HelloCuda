#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);          \
            return -1;                                                     \
        }                                                                  \
    } while (0)

// nvcc -O3 memory_bandwidth_test.cu test
// 优化版本：使用 float4 进行向量化读写
// 强制 16 字节对齐，确保生成 LD.E.128 / ST.E.128 指令
__global__ void bandwidth_kernel_vectorized(const float4* __restrict__ in,
                                            float4* __restrict__ out,
                                            size_t N_float4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N_float4; i += stride) {
        out[i] = in[i];
    }
}

int main() {
    // 增加数据量到 512MB，确保 GPU 频率跑满，且掩盖启动开销
    const size_t N = (size_t)1 << 27; // 128M floats
    const size_t bytes = N * sizeof(float);

    int dev_id = 0;
    cudaSetDevice(dev_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    printf("Device: %s\n", prop.name);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("Theoretical Peak: %.2f GB/s\n",
           (double)prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0 / 1e9);
           // 注意：memoryClockRate 返回的是 kHz，DDR 需乘 2

    printf("Allocating %.2f MB on device...\n", bytes / (1024.0 * 1024.0));

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in,  0, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // 配置 Kernel
    int blockSize = 256;
    // 针对 4090 (128 SMs)，给足 Block 数以保证 Occupancy
    // 通常设为 SM 数量的倍数
    int numBlocks = prop.multiProcessorCount * 8;

    // 数据量转换为 float4 的数量
    size_t N_float4 = N / 4;

    printf("Kernel Config: <<<%d, %d>>> using float4 vectorization\n", numBlocks, blockSize);

    // 预热
    bandwidth_kernel_vectorized<<<numBlocks, blockSize>>>((float4*)d_in, (float4*)d_out, N_float4);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 20; // 增加迭代次数

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        bandwidth_kernel_vectorized<<<numBlocks, blockSize>>>((float4*)d_in, (float4*)d_out, N_float4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= iters;

    double total_bytes = (double)N * sizeof(float) * 2.0;
    double bandwidth_GBps = total_bytes / (elapsed_ms / 1000.0) / 1e9;

    printf("------------------------------------------------\n");
    printf("Avg elapsed:     %.3f ms\n", elapsed_ms);
    printf("Measured BW:     %.2f GB/s\n", bandwidth_GBps);
    printf("------------------------------------------------\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}