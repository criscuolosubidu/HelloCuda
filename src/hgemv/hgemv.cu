#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// FP16
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// HGEMV: Warp HGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
    int tx = threadIdx.x; // 0~31
    int ty = threadIdx.y; // 0~4
    int bx = blockIdx.x; // 0~M/4
    int lane = tx % WARP_SIZE; // 0~31
    int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
    if (m < M) {
        half sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            // 若NUM_WARPS>=2，先将当前行的数据累加到第一个warp中
            int k = w * WARP_SIZE + lane;
            sum += a[m * K + k] * x[k];
        }
        sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}

// HGEMV: Warp HGEMV K128 + half2x2
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M,
                                        int K) {
    // 每个线程负责4个元素，一个warp覆盖128个元素
    int tx = threadIdx.x; // 0~31
    int ty = threadIdx.y; // 0~3
    int bx = blockIdx.x; // 0~M/4
    int lane = tx % WARP_SIZE; // 0~31
    int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

    if (m < M) {
        half sum = 0.0f;
        // process 4*WARP_SIZE elements per warp.
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = (w * WARP_SIZE + lane) * 4;
            half2 reg_x_0 = HALF2(x[k + 0]);
            half2 reg_x_1 = HALF2(x[k + 2]);
            half2 reg_a_0 = HALF2(a[m * K + k + 0]);
            half2 reg_a_1 = HALF2(a[m * K + k + 2]);
            sum += (reg_x_0.x * reg_a_0.x + reg_x_0.y * reg_a_0.y +
                    reg_x_1.x * reg_a_1.x + reg_x_1.y * reg_a_1.y);
        }
        sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}

// HGEMV: Warp HGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template<const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *A, half *x, half *y, int M, int K) {
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
    int tx = threadIdx.x; // 0~31
    int ty = threadIdx.y; // 0~NUM_WARPS
    int bx = blockIdx.x; // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
    int lane = tx % WARP_SIZE; // 0~31
    int k = lane % K_WARP_SIZE; // 0~15
    // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
    int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
    if (m < M) {
        half sum = A[m * K + k] * x[k];
        sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
        // 注意是k == 0，而不是lane == 0
        if (k == 0)
            y[m] = sum;
    }
}
