#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
// ElementWise Add grid(N/256)
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4=64)
// a: Nx1, b: Nx1, c:Nx1, c = elementwirse_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) { // 向量化内存访问技巧
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c; // *((float4*)(&c[idx])) = reg_c
    }
}

// ElementWise Add grid(N/256)
// block(256) a: Nx1, b : Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = __hadd(a[idx], b[idx]); // 注意这里不能直接相加，否则会隐式转换为float进行运算然后再做类型转换，这里用原生的加法
}

// ElementWise Add grid(N/256)
// block(256/2), a: Nx1, b: Nx1, c:Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(c[idx]) = reg_c;
    }
}

// ElementWise Add grid(N / 256)
// block(256 / 8), a: Nx1, b: Nx1, c:Nx1, c = element_wise(a, b)
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) { // 手动得循环展开，速度应该是更加快
        half2 a_0 = HALF2(a[idx]);
        half2 a_1 = HALF2(a[idx+2]);
        half2 a_2 = HALF2(a[idx+4]);
        half2 a_3 = HALF2(a[idx+6]);
        half2 b_0 = HALF2(b[idx]);
        half2 b_1 = HALF2(b[idx+2]);
        half2 b_2 = HALF2(b[idx+4]);
        half2 b_3 = HALF2(b[idx+6]);
        half2 c_0, c_1, c_2, c_3;
        c_0.x = __hadd(a_0.x, b_0.x);
        c_0.y = __hadd(a_0.y, b_0.y);
        c_1.x = __hadd(a_1.x, b_1.x);
        c_1.y = __hadd(a_1.y, b_1.y);
        c_2.x = __hadd(a_2.x, b_2.x);
        c_2.y = __hadd(a_2.y, b_2.y);
        c_3.x = __hadd(a_3.x, b_3.x);
        c_3.y = __hadd(a_3.y, b_3.y);
        if (idx < N) HALF2(c[idx]) = c_0;
        if (idx + 2 < N) HALF2(c[idx+2]) = c_1;
        if (idx + 4 < N) HALF2(c[idx+4]) = c_2;
        if (idx + 6 < N) HALF2(c[idx+6]) = c_3;
    }
}

// 向量化加载，非常trick！
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_a[8], pack_b[8], pack_c[8];
    // 将 f16 * 8 看成一个 float4去加载，只需要一次加载内存
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
#pragma unroll // 提示后面得循环进行展开或者部分展开
    for (int i = 0; i < 8; i += 2) {
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    if (idx + 7 < N) {
        // 依然是加载一次内存
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
}

