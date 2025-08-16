#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
// Relu x:N, y:N, y=max(0,x)
// grid(N/256), block(256)
__global__ void relu_fp32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

// fp32 x 4
// grid(N/256), block(256)
__global__ void relu_fp32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    if (idx < N) {
        reg_y.x = fmaxf(0.0f, reg_x.x);
        reg_y.y = fmaxf(0.0f, reg_x.y);
        reg_y.z = fmaxf(0.0f, reg_x.z);
        reg_y.w = fmaxf(0.0f, reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// fp16
// Relu x:N, y:N, y=max(0,x)
// grid(N/256), block(256)
__global__ void relu_fp16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = __hmax(__float2half(0.0f), x[idx]);
    }
}

// fp16 x 2
__global__ void relu_fp16x2_kernel(half *x, half *y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    if (idx < N) {
        reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
        reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

// fp16 x 8
__global__ void relu_fp16x8_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
    reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
    reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
    reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
    reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
    reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
    reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
    reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
    if (idx + 0 < N) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if (idx + 2 < N) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if (idx + 4 < N) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if (idx + 6 < N) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

// fp16 x 8
// pack version, 1 memory issue
__global__ void relu_fp16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half reg_x[8], reg_y[8];
    const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
    LDST128BITS(reg_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        HALF2(reg_y[i]) = __hmax2(z2, HALF2(reg_x[i]));
    }
    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(reg_y[0]);
    }
}
