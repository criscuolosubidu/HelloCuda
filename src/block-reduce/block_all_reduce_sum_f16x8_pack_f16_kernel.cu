#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>
#include <thread>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

// grid(N / 256), block(256 / 8>
template<const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *a, float *v, int N) {
    int tid = threadIdx.x;
    int idx = 8 * (blockIdx.x * NUM_THREADS + threadIdx.x);
    constexpr int NUM_WAPRS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WAPRS];
    half pack_a[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    const half zero = __float2half(0.0f);
#pragma unroll
    half sum_f16 = 0.0f;
    for (auto i : pack_a) {
        if (idx < N) sum_f16 = __hadd(sum_f16, i);
    }
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0) reduce_sum[warp] = __half2float(sum_f16); // warp's first thread to write
    __syncthreads();
    float sum = lane < NUM_WAPRS ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WAPRS>(sum);
    if (tid == 0) atomicAdd(v, sum); // global first thread to write
}

int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 256 * 256 * 256;

    // CPU part
    std::vector<float> h_x(N);
    std::vector<half> h_x_half(N);
    float h_y = 0.0f;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 0.001f);
    for (int i = 0; i < N; ++i) {
        h_x[i] = dist(gen);
        h_x_half[i] = __float2half(h_x[i]);
    }

    float *d_x, *d_y;
    half *d_x_half;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMalloc(&d_x_half, N * sizeof(half));

    auto start = std::chrono::high_resolution_clock::now();

    // cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_half, h_x_half.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(float));

    block_all_reduce_sum_f16x8_pack_f16_kernel<<<256 * 256, 32>>>(d_x_half, d_y, N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    std::cout << "GPU operations took: " << elapsed_ms.count() << " ms\n";

    float ans = accumulate(h_x.begin(), h_x.end(), 0.0f);
    std::cout << "CPU sum: " << ans << ", GPU sum: " << h_y << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}