#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// 定义阈值
#define THRESHOLD_A 3.0
#define THRESHOLD_B (-3.0)

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
__device__ __forceinline__ float hardswish(float x) {
    if (x >= THRESHOLD_A) return x;
    else if (x <= THRESHOLD_B) return 0;
    else return x * (x + 3) / 6;
}

// fp16
__device__ __forceinline__ half hardswish(half x) {
    const half threshold_a = __float2half(THRESHOLD_A);
    const half threshold_b = __float2half(THRESHOLD_B);
    const half h0 = __float2half(0.f), h3 = __float2half(3.0f), h6 = __float2half(6.0f);
    if (x >= threshold_a) return x;
    else if (x <= threshold_b) return h0;
    else return x * (x + h3) / h6;
}

// fp32
__global__ void hardswish_fp32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardswish(x[idx]);
    }
}

// fp32x4
__global__ void hardswish_fp32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = hardswish(reg_x.x);
        reg_y.y = hardswish(reg_x.y);
        reg_y.z = hardswish(reg_x.z);
        reg_y.w = hardswish(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// fp16
__global__ void hardswish_fp16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardswish(x[idx]);
    }
}

// fp16x2
__global__ void hardswish_fp16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = hardswish(reg_x.x);
        reg_y.y = hardswish(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

// fp16x8
__global__ void hardswish_f16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = hardswish(reg_x_0.x);
    reg_y_0.y = hardswish(reg_x_0.y);
    reg_y_1.x = hardswish(reg_x_1.x);
    reg_y_1.y = hardswish(reg_x_1.y);
    reg_y_2.x = hardswish(reg_x_2.x);
    reg_y_2.y = hardswish(reg_x_2.y);
    reg_y_3.x = hardswish(reg_x_3.x);
    reg_y_3.y = hardswish(reg_x_3.y);
    if ((idx + 0) < N) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if ((idx + 2) < N) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if ((idx + 4) < N) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if ((idx + 6) < N) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

// fp16x8 pack
__global__ void hardswish_fp16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half x_pack[8], y_pack[8];
    LDST128BITS(x_pack[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        y_pack[i] = hardswish(x_pack[i]);
    }
    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(y_pack[0]);
    }
}




