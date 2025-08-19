#include <algorithm>
#include <cuda_fp16.h>
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
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define LAMBD 0.5f

// fp32
__device__ __forceinline__ float hardshrink(float x) {
    return (x > LAMBD || x < -LAMBD) ? x : 0;
}

// fp16
__device__ __forceinline__ half hardshrink(half x) {
    const half lambd = __float2half(LAMBD);
    return (x > lambd || x < -lambd) ? x : __float2half(0.f);
}

// fp32
// hardshrink, x:N, y:N, y=(abs(x) > lambd ? x : 0)
__global__ void hardshrink_fp32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardshrink(x[idx]);
    }
}

// fp32x4
__global__ void hardshrink_fp32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = hardshrink(reg_x.x);
        reg_y.y = hardshrink(reg_x.y);
        reg_y.z = hardshrink(reg_x.z);
        reg_y.w = hardshrink(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// fp16
__global__ void hardshrink_fp16_kernel(half *x, half *y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardshrink(x[idx]);
    }
}

// fp16x2
__global__ void hardshrink_fp16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = hardshrink(reg_x.x);
        reg_y.y = hardshrink(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

// fp16x8 unpack
__global__ void hardshrink_f16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = hardshrink(reg_x_0.x);
    reg_y_0.y = hardshrink(reg_x_0.y);
    reg_y_1.x = hardshrink(reg_x_1.x);
    reg_y_1.y = hardshrink(reg_x_1.y);
    reg_y_2.x = hardshrink(reg_x_2.x);
    reg_y_2.y = hardshrink(reg_x_2.y);
    reg_y_3.x = hardshrink(reg_x_3.x);
    reg_y_3.y = hardshrink(reg_x_3.y);
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
__global__ void hardshrink_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half x_pack[8], y_pack[8];
    LDST128BITS(x_pack[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        y_pack[i] = hardshrink(x_pack[i]);
    }
    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(y_pack[0]);
    }
}







