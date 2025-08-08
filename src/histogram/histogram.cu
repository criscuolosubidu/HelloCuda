#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


// Histogram
// grid(N/256), block(256)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32_kernel(int* a, int* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&y[a[idx]], 1);
    }
}

// struct __device_builtin__ __builtin_align__(16) int4
// {
//     int x, y, z, w;
// };

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y : count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int* a, int* y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        int4 reg_a = INT4(a[idx]);
        atomicAdd(&y[reg_a.x], 1);
        atomicAdd(&y[reg_a.y], 1);
        atomicAdd(&y[reg_a.z], 1);
        atomicAdd(&y[reg_a.w], 1);
    }
}