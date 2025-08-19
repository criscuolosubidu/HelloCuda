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
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
// output[i] = weight[idx[i]], 查表操作
// grid(N), block(emb / 4)
__global__ void embedding_fp32_kernel(const int *idx, const float *weight, float *output, int emb_size) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

// fp32x4 unroll kernel
// grid(N), block(emb / 4)
__global__ void embedding_fp32x4_kernel(const int *idx, const float *weight, float *output, int emb_size) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
    output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
    output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
    output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}

// fp32x4 pack
// grid(N), block(emb / 4)
__global__ void embedding_fp32x4_pack_kernel(const int *idx, float *weight, float *output, int emb_size) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    LDST128BITS(output[bx * emb_size + 4 * tx]) = LDST128BITS(weight[offset + 4 * tx]);
}

// fp16
// grid(N), block(emb)
__global__ void embedding_fp16_kernel(const int *idx, const half *weight, half *output, int emb_size) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

// fp16x8 unroll
// grid(N), block(emb/8)
__global__ void embedding_fp16x8_kernel(const int *idx, const half *weight, half *output, int emb_size) {
    int tx = threadIdx.x * 8;
    int bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
    output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
    output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
    output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
    output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
    output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
    output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
    output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

// fp16x8 pack
// grid(N), block(emb/8)
__global__ void embedding_fp16x8_pack_kernel(const int *idx, half *weight, half *output, int emb_size) {
    int tx = threadIdx.x * 8, bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    LDST128BITS(output[bx * emb_size + tx]) = LDST128BITS(weight[offset + tx]);
}
