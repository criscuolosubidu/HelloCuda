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

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
    auto val_i32 = static_cast<int32_t>(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
    }
    return val_i32;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(const int8_t *a, int32_t *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ int32_t reduce_sum[NUM_WARPS];
    int8_t sum_i8 = idx < N ? a[idx] : 0;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(sum_i8);
    if (lane == 0) reduce_sum[warp] = sum_i32;
    __syncthreads();
    int32_t sum = lane < NUM_WARPS ? reduce_sum[lane] : 0;
    if (warp == 0) sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}

template<const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t *a, int32_t *y, int N) {
    // 8 * 16 = 128 = 32 * 4
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ int32_t reduce_sum[NUM_WARPS];
    int8_t pack_a[16];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    int32_t sum_i32 = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        sum_i32 += static_cast<int32_t>(pack_a[i]);
    }
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
    if (lane == 0) reduce_sum[warp] = sum_i32;
    __syncthreads();
    int32_t sum = lane < NUM_WARPS ? reduce_sum[lane] : 0;
    if (warp == 0) sum = warp_reduce_sum_i32_i32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}


int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 1073741824; // 2 ^ 30
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // CPU part
    int32_t h_y;
    std::vector<int8_t> h_x(N);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 10);

    for (int i = 0; i < N; ++i) {
        int tmp = dist(gen);
        h_x[i] = static_cast<int8_t>(tmp);
    }

    // GPU part
    int32_t *d_y;
    int8_t *d_x;

    cudaMalloc(&d_x, N * sizeof(int8_t));
    cudaMalloc(&d_y, sizeof(int32_t));

    std::cout << "\nTesting INT8 kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_x, h_x.data(), N * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(int32_t));

    block_all_reduce_sum_i8_i32_kernel<NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(d_x, d_y, N); // 83 ms
    // block_all_reduce_sum_i8x16_pack_i32_kernel<NUM_THREADS / 16><<<NUM_BLOCKS, NUM_THREADS / 16>>>(d_x, d_y, N); // 81 ms

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(&h_y, d_y, sizeof(int32_t), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "INT8 GPU operations took: " << elapsed_ms << " ms\n";

    int32_t ans = std::accumulate(h_x.begin(), h_x.end(), 0);
    std::cout << "\nResults:" << std::endl;
    std::cout << "CPU sum: " << ans << std::endl;
    std::cout << "INT8 GPU sum: " << h_y << std::endl;

    double i8_error = std::abs(h_y - ans) * 1.0 / ans;
    std::cout << "INT8 relative error: " << i8_error * 100 << "%" << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}