#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <thread>
#include <curand.h>

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

// Scale uniform [0,1) to [min, max)
__global__ void scale_uniform_kernel(float *data, int N, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * (max_val - min_val) + min_val;
    }
}


// grid(N / 256), block(256 / 4)
// a: N x 1, b x 1, y = sum(elementwise_mul(a, b))
template<const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float prod = 0.0f;
    if (idx < N) prod = reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w;
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    if (lane == 0) reduce_sum[warp] = prod;
    __syncthreads();
    prod = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    if (warp == 0) prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    if (tid == 0) atomicAdd(y, prod);
}


int main() {
    std::cout << "Start Test dot-product kernels!" << std::endl;
    constexpr int N = 2048 * 2048; // 2 ^ 22

    float h_y;
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);

    auto start_gen = std::chrono::high_resolution_clock::now();

    float *d_a, *d_b;
    float *d_y;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1233ULL);
    curandGenerateUniform(gen, d_a, N);  // [0, 1)
    curandGenerateUniform(gen, d_b, N);  // [0, 1)

    constexpr float MIN_VAL = 0.0f;
    constexpr float MAX_VAL = 0.1f;  // -> [0, 0.001)
    int num_blocks = (N + 255) / 256;
    scale_uniform_kernel<<<num_blocks, 256>>>(d_a, N, MIN_VAL, MAX_VAL);
    scale_uniform_kernel<<<num_blocks, 256>>>(d_b, N, MIN_VAL, MAX_VAL);

    cudaMemset(d_y, 0, sizeof(float)); // sum

    cudaDeviceSynchronize();

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();
    std::cout << "GPU Random Number Generate Time : " << gen_time << " ms" << std::endl;

    std::cout << "Start Copy data to host" << std::endl;
    auto start_copy = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_a.data(), d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto end_copy = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration<double, std::milli>(end_copy - start_copy).count();
    std::cout << "GPU Copy data Time : " << copy_time << " ms" << std::endl;

    std::cout << "Start Compute dot-product" << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();

    dot_prod_f32x4_f32_kernel<256 / 4><<<N / 256, 256 / 4>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();
    std::cout << "GPU Compute dot-product Time : " << compute_time << " ms" << std::endl;

    float ans = 0.0f;
    for (int i = 0; i < N; i++) {
        ans += h_a[i] * h_b[i];
    }
    std::cout << "CPU Compute dot-product Result : " << ans << std::endl;

    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU Compute dot-product Result : " << h_y << std::endl;

    double f32_error = std::abs(ans - h_y) / std::abs(ans);
    std::cout << "GPU Compute dot-product Error : " << f32_error << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);
}