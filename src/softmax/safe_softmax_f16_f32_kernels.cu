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

// transform f32 to f16
__global__ void transform_f32_f16_kernel(float *data, half *data_f16, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data_f16[idx] = __float2half(data[idx]);
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

// Warp Reduce for sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float value) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride > 0; stride >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, stride);
    }
    return value;
}

// Warp Reduce for max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float value) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride > 0; stride >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, stride));
    }
    return value;
}

// grid 1D
// grid(N / 256), block(256)
template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WAPRS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ float reduce_sum[NUM_WAPRS];
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = value;
    __syncthreads();
    value = lane < NUM_WAPRS ? reduce_sum[lane] : 0.0f;
    value = warp_reduce_sum_f32<NUM_WAPRS>(value);
    // if you choose `value = warp_reduce_sum_f32<WARP_SIZE>(value)` then you don't need to sync them explicitly
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}

// grid 1D
// grid(N / 256), block(256)
template<const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
    constexpr int NUM_WAPRS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ float reduce_max[NUM_WAPRS];
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float value = warp_reduce_max_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_max[warp] = value;
    __syncthreads();
    value = lane < NUM_WAPRS ? reduce_max[lane] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WAPRS>(value);
    // if you choose `value = warp_reduce_sum_f32<WARP_SIZE>(value)` then you don't need to sync them explicitly
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}


template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mx_val = idx < N ? __half2float(x[idx]) : -FLT_MAX;
    mx_val = block_reduce_max_f32<NUM_THREADS>(mx_val);
    float exp_val = idx < N ? expf(__half2float(x[idx]) - mx_val) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (idx < N) y[idx] = __float2half(exp_val / exp_sum);
}


template<const int NUM_THREADS = 256 / 2>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + tid) * 2;

    // read 2 half, turn into 2 float
    float2 reg_x = __half22float2(HALF2(x[idx]));
    float max_val = -FLT_MAX;
    max_val = idx + 0 < N ? fmaxf(reg_x.x, max_val) : -FLT_MAX;
    max_val = idx + 1 < N ? fmaxf(reg_x.y, max_val) : -FLT_MAX;
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

    float2 reg_exp;
    reg_exp.x = idx + 0 < N ? expf(reg_x.x - max_val) : 0.0f;
    reg_exp.y = idx + 1 < N ? expf(reg_x.y - max_val) : 0.0f;
    float exp_val = reg_exp.x + reg_exp.y;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum

    float2 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    if (idx + 1 < N)
        HALF2(y[idx]) = __float22half2_rn(reg_y);
}


template<const int NUM_THREADS = 256 / 8>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y, int N) {
    const int tid = threadIdx.x;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    float mx_val = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        mx_val = fmaxf(mx_val, __half2float(pack_x[i]));
    }
    mx_val = block_reduce_max_f32<NUM_THREADS>(mx_val);
    float exp_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        exp_sum += expf(__half2float(pack_x[i]) - mx_val);
    }
    exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float exp_val = expf(__half2float(pack_x[i]) - mx_val);
        pack_y[i] = __float2half_rn(exp_val / exp_sum);
    }
    if (idx + 7 < N) LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
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
    half *d_x_half;
    half *d_y_half;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_x_half, N * sizeof(half));
    cudaMalloc(&d_y_half, N * sizeof(half));

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

    // transform fp32 -> fp16
    transform_f32_f16_kernel<<<TOKENS, TOKEN_DIM>>>(d_x, d_x_half, N);

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
    // safe_softmax_f16_f32_per_token_kernel<<<TOKENS, TOKEN_DIM>>>(d_x_half, d_y_half, N); // 82.97x speed up
    // safe_softmax_f16x2_f32_per_token_kernel<<<TOKENS, TOKEN_DIM / 2>>>(d_x_half, d_y_half, N); // 78.134x speed up
    safe_softmax_f16x8_pack_f32_per_token_kernel<<<TOKENS, TOKEN_DIM / 8>>>(d_x_half, d_y_half, N); // 90.3038x speed up

    cudaDeviceSynchronize();

    auto end_compute = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();
    std::cout << "GPU Compute softmax Time : " << compute_time << " ms" << std::endl;

    // CPU reference computation
    std::cout << "Start CPU Reference Compute" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Per-token safe softmax: each token independently normalized
    for (int token = 0; token < TOKENS; ++token) {
        int base_idx = token * TOKEN_DIM;

        // Step 1: Find maximum value for this token (for numerical stability)
        float max_val = h_x[base_idx];
        for (int i = 1; i < TOKEN_DIM; ++i) {
            max_val = fmaxf(max_val, h_x[base_idx + i]);
        }

        // Step 2: Calculate sum of exp(x - max_val) for this token
        float exp_sum = 0.0f;
        for (int i = 0; i < TOKEN_DIM; ++i) {
            exp_sum += expf(h_x[base_idx + i] - max_val);
        }

        // Step 3: Normalize for this token
        for (int i = 0; i < TOKEN_DIM; ++i) {
            h_y[base_idx + i] = expf(h_x[base_idx + i] - max_val) / exp_sum;
        }
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU Reference Compute Time : " << cpu_time << " ms" << std::endl;

    // Copy GPU result back to host
    std::vector<float> h_y_gpu(N);
    std::vector<half> h_y_half(N);
    cudaMemcpy(h_y_half.data(), d_y_half, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) h_y_gpu[i] = __half2float(h_y_half[i]);

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