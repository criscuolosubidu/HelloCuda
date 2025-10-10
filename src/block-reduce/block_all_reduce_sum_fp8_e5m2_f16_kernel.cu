#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

// FP8 -> FP16/FP32
// __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t, nv_fp8_enum);
// __nv_cvt_fp8_to_float(__nv_fp8_storage_t, nv_fp8_enum);

// FP16/FP32 -> FP8
// __nv_cvt_halfraw_to_fp8(__half, nv_fp8_enum);
// __nv_cvt_float_to_fp8(float, nv_fp8_enum);

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
    // storage type and calculate type is seperated
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    }
    return val_f16;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) {
    // storage type and calculate type is seperated
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    }
    return val_f16;
}


template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(const __nv_fp8_storage_t *a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_sum[NUM_WARPS];
    __nv_fp8_storage_t sum_f8 = idx < N ? a[idx] : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_f8);
    if (lane == 0) reduce_sum[warp] = sum_f16;
    __syncthreads();
    half sum = lane < NUM_WARPS ? reduce_sum[lane] : __float2half(0.0f);
    if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, __half2float(sum));
}


template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(const __nv_fp8_storage_t *a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_sum[NUM_WARPS];
    __nv_fp8_storage_t sum_f8 = idx < N ? a[idx] : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E5M2);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_f8);
    if (lane == 0) reduce_sum[warp] = sum_f16;
    __syncthreads();
    half sum = lane < NUM_WARPS ? reduce_sum[lane] : __float2half(0.0f);
    if (warp == 0) sum = warp_reduce_sum_f16_f16<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, __half2float(sum));
}


int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 256 * 256 * 256;
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // CPU part
    float h_y = 0.0f;
    std::vector<float> h_x(N);
    std::vector<__nv_fp8_storage_t> h_x_fp8(N);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 0.1f);

    for (int i = 0; i < N; ++i) {
        h_x[i] = dist(gen);
        h_x_fp8[i] = __nv_cvt_float_to_fp8(h_x[i], __NV_SATFINITE, __NV_E5M2);
    }

    // GPU part
    float *d_y;
    __nv_fp8_storage_t *d_x_fp8;

    cudaMalloc(&d_x_fp8, N * sizeof(__nv_fp8_storage_t));
    cudaMalloc(&d_y, sizeof(float));

    std::cout << "\nTesting FP8 E5M2 kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_x_fp8, h_x_fp8.data(), N * sizeof(__nv_fp8_storage_t), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(float));

    block_all_reduce_sum_fp8_e5m2_f16_kernel<NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(d_x_fp8, d_y, N); // 2.6ms

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "FP8 GPU operations took: " << elapsed_ms << " ms\n";

    float ans = std::accumulate(h_x.begin(), h_x.end(), 0.0f);
    std::cout << "\nResults:" << std::endl;
    std::cout << "CPU sum: " << ans << std::endl;
    std::cout << "FP8 GPU sum: " << h_y << std::endl;

    float fp8_error = std::abs(h_y - ans) / ans;
    std::cout << "FP8 relative error: " << fp8_error * 100 << "%" << std::endl;

    cudaFree(d_x_fp8);
    cudaFree(d_y);

    return 0;
}