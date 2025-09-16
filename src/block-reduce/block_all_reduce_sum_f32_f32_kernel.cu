#include <iostream>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>

#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) reduce_sum[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}


int main() {
    std::cout << "test cuda kernels!" << std::endl;
    constexpr int N = 256 * 256;

    // CPU part
    std::vector<float> h_x(N);
    float h_y = 0.0f;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_x[i] = dist(gen);
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));

    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(float));

    block_all_reduce_sum_f32_f32_kernel<<<256, 256>>>(d_x, d_y, N);
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    float ans = accumulate(h_x.begin(), h_x.end(), 0.0f);
    std::cout << ans << ' ' << h_y << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
}

