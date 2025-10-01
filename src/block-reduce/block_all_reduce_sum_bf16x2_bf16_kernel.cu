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
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

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

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {
    // half = 1 + 5 + 10, bfloat = 1 + 8 + 7
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {
    float val_f32 = __bfloat162float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

// grid(N / 256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16* a, float *v, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_sum[NUM_WARPS];
    __nv_bfloat16 sum_b16 = idx < N ? a[idx] : __float2bfloat16(0.0f);
    sum_b16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_b16);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) reduce_sum[warp] = sum_b16;
    __syncthreads();
    __nv_bfloat16 sum = lane < NUM_WARPS ? reduce_sum[lane] : __float2bfloat16(0.0f);
    if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(v, __bfloat162float(sum));
}


// grid(N / 256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16* a, float *v, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    __nv_bfloat16 sum_bf16 = idx < N ? a[idx] : __float2bfloat16(0.0f);
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) reduce_sum[warp] = sum_f32;
    __syncthreads();
    float sum = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f; // WARP_SIZE is enough, 1 block most have 1024 threads
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum); // first warp to calculate the sum
    if (tid == 0) atomicAdd(v, sum); // first thread to write
}

// grid(N / 256), block(256 / 2)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16* a, float *v, int N) {
    int tid = threadIdx.x;
    int idx = 2 * (blockIdx.x * NUM_THREADS + threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_sum[NUM_WARPS];
    __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_b16 = idx < N ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
    sum_b16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_b16);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) reduce_sum[warp] = sum_b16;
    __syncthreads();
    __nv_bfloat16 sum = lane < NUM_WARPS ? reduce_sum[lane] : __float2bfloat16(0.0f);
    if (warp == 0) sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(v, __bfloat162float(sum));
}

// grid(N / 256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16* a, float *v, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    __nv_bfloat162 reg_a = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_bf16 = idx < N ? __hadd(reg_a.x, reg_a.y) : __float2bfloat16(0.0f);
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) reduce_sum[warp] = sum_f32;
    __syncthreads();
    float sum = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f; // WARP_SIZE is enough, 1 block most have 1024 threads
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum); // first warp to calculate the sum
    if (tid == 0) atomicAdd(v, sum); // first thread to write
}

int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 256 * 256 * 256;

    // CPU part
    float h_y = 0.0f;
    std::vector<float> h_x(N);
    std::vector<__nv_bfloat16> h_x_bf16(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 0.1f);

    for (int i = 0; i < N; ++i) {
        h_x[i] = dist(gen);
        h_x_bf16[i] = __float2bfloat16(h_x[i]);
    }

    float *d_y;
    __nv_bfloat16 *d_x;
    cudaMalloc(&d_x, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_y, sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_x, h_x_bf16.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(float));

    // block_all_reduce_sum_bf16_bf16_kernel<<<256 * 256, 256>>>(d_x, d_y, N); // 4.3514 ms
    // block_all_reduce_sum_bf16_f32_kernel<<<256 * 256, 256>>>(d_x, d_y, N); // 4.3604 ms
    block_all_reduce_sum_bf16x2_bf16_kernel<<<256 * 256, 256 / 2>>>(d_x, d_y, N); // 4.3899 ms , 精度有点差

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