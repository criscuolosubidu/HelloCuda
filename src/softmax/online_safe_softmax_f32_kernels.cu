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


// Scale uniform [0,1) to [min, max)
__global__ void scale_uniform_kernel(float *data, int N, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * (max_val - min_val) + min_val;
    }
}

struct __align__(8) MD {
    float m;
    float d;
};

// Warp Reduce for Online Softmax
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int stride = kWarpSize >> 1; stride > 0; stride >>= 1) {
        MD other{};
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);
        bool is_big = value.m > other.m;
        MD bigger_md = is_big ? value : other;
        MD smaller_md = is_big ? other : value;
        value.m = bigger_md.m;
        value.d = bigger_md.d + smaller_md.d * __expf(smaller_md.m - bigger_md.m);
    }
    return value;
}

// reference: https://arxiv.org/pdf/1805.02867
// widely used in flash-attn
template<const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ MD reduce_md[NUM_WARPS];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    MD val;
    val.m = idx < N ? x[idx] : -FLT_MAX;
    val.d = idx < N ? 1.0f : 0.0f; // e^(xi-xi) = e^0 = 1
    val = warp_reduce_md_op<WARP_SIZE>(val);
    if (lane == 0) reduce_md[warp] = val;
    __syncthreads();
    // calculate the block reduce MD
    MD block_res = lane < NUM_WARPS ? reduce_md[lane] : MD{-FLT_MAX, 0};
    if (warp == 0) block_res = warp_reduce_md_op<NUM_WARPS>(block_res);
    if (tid == 0) reduce_md[0] = block_res;
    __syncthreads();
    // calculate the softmax value
    MD final_res = reduce_md[0];
    float d_total_inverse = __fdividef(1.0, final_res.d);
    if (idx < N) y[idx] = __expf(x[idx] - final_res.m) * d_total_inverse;
}

template<const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ MD reduce_md[NUM_WARPS];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float4 reg_x = FLOAT4(x[idx]);
    float mx_val = fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w));
    float d_val = __expf(reg_x.x - mx_val) + __expf(reg_x.y - mx_val) + __expf(reg_x.z - mx_val) + __expf(reg_x.w - mx_val);
    MD md{mx_val, d_val};
    MD res = warp_reduce_md_op<WARP_SIZE>(md);
    if (lane == 0) reduce_md[warp] = res;
    __syncthreads();
    MD block_res = lane < NUM_WARPS ? reduce_md[lane] : MD{-FLT_MAX, 0};
    if (warp == 0) block_res = warp_reduce_md_op<NUM_WARPS>(block_res);
    if (tid == 0) reduce_md[0] = block_res;
    __syncthreads();
    MD final_res = reduce_md[0];
    float d_total_inverse = __fdividef(1.0, final_res.d);
    float4 reg_y;
    reg_y.x = __expf(reg_x.x - final_res.m) * d_total_inverse;
    reg_y.y = __expf(reg_x.y - final_res.m) * d_total_inverse;
    reg_y.z = __expf(reg_x.z - final_res.m) * d_total_inverse;
    reg_y.w = __expf(reg_x.w - final_res.m) * d_total_inverse;
    if (idx + 3 < N) FLOAT4(y[idx]) = reg_y;
}


int main() {
    std::cout << "Start Testing Softmax kernels" << std::endl;
    constexpr int TOKENS = 2048;      // Number of tokens
    constexpr int TOKEN_DIM = 256;    // Dimension per token
    constexpr int N = TOKENS * TOKEN_DIM; // Total elements

    std::vector<float> h_y(N);
    std::vector<float> h_x(N);

    auto start_gen = std::chrono::high_resolution_clock::now();

    float *d_x;
    float *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // gen
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1233ULL);
    curandGenerateUniform(gen, d_x, N);

    // scale to [0, 1]
    constexpr float MIN_VAL = 0.0f;
    constexpr float MAX_VAL = 2;
    scale_uniform_kernel<<<TOKENS, TOKEN_DIM>>>(d_x, N, MIN_VAL, MAX_VAL);

    cudaDeviceSynchronize();

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();
    std::cout << "GPU Random Number Generate Time : " << gen_time << " ms" << std::endl;

    // copy to host
    auto start_copy = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto end_copy = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration<double, std::milli>(end_copy - start_copy).count();
    std::cout << "GPU Copy data Time : " << copy_time << " ms" << std::endl;

    std::cout << "Start Compute softmax" << std::endl;
    auto start_compute = std::chrono::high_resolution_clock::now();

    // kernel functions
    // online_safe_softmax_f32_per_token_kernel<<<TOKENS, TOKEN_DIM>>>(d_x, d_y, N); // 40.0868x speed up
    online_safe_softmax_f32x4_per_token_kernel<<<TOKENS, TOKEN_DIM / 4>>>(d_x, d_y, N); // 45.2075x speed up
    cudaDeviceSynchronize();

    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();
    std::cout << "GPU Compute softmax Time : " << compute_time << " ms" << std::endl;

    // CPU reference computation
    std::cout << "Start CPU Reference Compute" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Per-token softmax: each token independently normalized
    for (int token = 0; token < TOKENS; ++token) {
        int base_idx = token * TOKEN_DIM;

        // Calculate sum of exp for this token
        float exp_sum = 0.0f;
        for (int i = 0; i < TOKEN_DIM; ++i) {
            exp_sum += expf(h_x[base_idx + i]);
        }

        // Normalize for this token
        for (int i = 0; i < TOKEN_DIM; ++i) {
            h_y[base_idx + i] = expf(h_x[base_idx + i]) / exp_sum;
        }
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU Reference Compute Time : " << cpu_time << " ms" << std::endl;

    // Copy GPU result back to host
    std::vector<float> h_y_gpu(N);
    cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Compare results and calculate error
    std::cout << "\n=== Result Comparison ===" << std::endl;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    float max_relative_error = 0.0f;
    int error_count = 0;
    constexpr float ERROR_THRESHOLD = 1e-5f;

    for (int i = 0; i < N; ++i) {
        float error = fabsf(h_y_gpu[i] - h_y[i]);
        float relative_error = h_y[i] != 0.0f ? error / fabsf(h_y[i]) : error;

        max_error = fmaxf(max_error, error);
        max_relative_error = fmaxf(max_relative_error, relative_error);
        avg_error += error;

        if (error > ERROR_THRESHOLD) {
            error_count++;
            if (error_count <= 5) { // Show first 5 errors
                std::cout << "Error at index " << i << ": GPU=" << h_y_gpu[i]
                          << ", CPU=" << h_y[i] << ", diff=" << error << std::endl;
            }
        }
    }

    avg_error /= N;

    std::cout << "\nError Statistics:" << std::endl;
    std::cout << "  Max Error        : " << max_error << std::endl;
    std::cout << "  Average Error    : " << avg_error << std::endl;
    std::cout << "  Max Relative Err : " << max_relative_error * 100.0f << "%" << std::endl;
    std::cout << "  Error Count (>" << ERROR_THRESHOLD << ") : " << error_count << " / " << N << std::endl;

    if (max_error < 1e-4f) {
        std::cout << "\nTest PASSED! Results match within tolerance." << std::endl;
    } else {
        std::cout << "\nTest FAILED! Errors exceed tolerance." << std::endl;
    }

    // Performance summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "CPU Time  : " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time  : " << compute_time << " ms" << std::endl;
    std::cout << "Speedup   : " << cpu_time / compute_time << "x" << std::endl;

    // Verify sum of softmax output (each token should sum to ~1.0)
    std::cout << "\nSoftmax sum check (per token, should be ~1.0):" << std::endl;
    float max_sum_error = 0.0f;
    for (int token = 0; token < TOKENS; ++token) {
        int base_idx = token * TOKEN_DIM;
        float token_sum = 0.0f;
        for (int i = 0; i < TOKEN_DIM; ++i) {
            token_sum += h_y_gpu[base_idx + i];
        }
        float sum_error = fabsf(token_sum - 1.0f);
        max_sum_error = fmaxf(max_sum_error, sum_error);
        if (token < 3) { // Show first 3 tokens
            std::cout << "  Token " << token << " sum: " << token_sum << std::endl;
        }
    }
    std::cout << "  Max sum error: " << max_sum_error << std::endl;

    // Cleanup - free GPU memory
    std::cout << "\n=== Cleanup ===" << std::endl;
    cudaFree(d_x);
    cudaFree(d_y);
    curandDestroyGenerator(gen);

    std::cout << "GPU memory freed successfully." << std::endl;
    std::cout << "Test completed!" << std::endl;

    return 0;
}