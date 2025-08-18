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


// fp32
// swish x:N, y:N, y=x*sigmoid(x)
// grid(N/256), block(256)
__device__ __forceinline__ float swish(float x) {
    return x / (1 + expf(-x));
}

__global__ void swish_fp32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = swish(x[idx]);
    }
}

// fp32x4
// grid(N/256),block(256/4)
__global__ void swish_fp32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = swish(reg_x.x);
        reg_y.y = swish(reg_x.y);
        reg_y.z = swish(reg_x.z);
        reg_y.w = swish(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// fp16
__device__ __forceinline__ half swish(half x) {
    const half f = __float2half(x);
    return x * (f / (f + hexp(-x))); // 这里是为了数值稳定性
}

__global__ void swish_fp16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = swish(x[idx]);
    }
}

// fp16x4
__global__ void swish_fp16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = swish(reg_x.x);
        reg_y.y = swish(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

// f16x8
__global__ void swish_f16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = swish(reg_x_0.x);
    reg_y_0.y = swish(reg_x_0.y);
    reg_y_1.x = swish(reg_x_1.x);
    reg_y_1.y = swish(reg_x_1.y);
    reg_y_2.x = swish(reg_x_2.x);
    reg_y_2.y = swish(reg_x_2.y);
    reg_y_3.x = swish(reg_x_3.x);
    reg_y_3.y = swish(reg_x_3.y);
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

// fp16x8 pack kernel
__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half x_pack[8], y_pack[8];
    LDST128BITS(x_pack[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        y_pack[i] = swish(x_pack[i]);
    }
    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(y_pack[0]);
    }
}








