#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) reduce_sum[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}

// a: N x 1, y = sum(a)
template<const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + threadIdx.x) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    // read as float4 type
    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? reg_a.x + reg_a.y + reg_a.z + reg_a.w : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) reduce_sum[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
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

// grid(N / 256), block(256)
// a: Nx1, y = sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half *a, float *v, int N) {
    int tid = threadIdx.x; // block threadIdx
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x; // global threadIdx
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    half sum_f16 = idx < N ? a[idx] : __float2half(0.0f);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (lane == 0) reduce_sum[warp] = sum_f16;
    __syncthreads();
    float sum = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(v, sum);
}

int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 256 * 256;

    // CPU part
    std::vector<float> h_x(N);
    std::vector<half> h_x_half(N);
    float h_y = 0.0f;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
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

    // block_all_reduce_sum_f32x4_f32_kernel<<<256, 256/4>>>(d_x, d_y, N); // 1.3512 ms
    // block_all_reduce_sum_f32_f32_kernel<<<256, 256>>>(d_x, d_y, N); //  1.4372 ms
    block_all_reduce_sum_f16_f16_kernel<<<256, 256>>>(d_x_half, d_y, N); // 1.3897 ms

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