#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thread>

static constexpr int WARP_SIZE = 32;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// fp32 -> fp32
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// fp16 -> fp16
template<const int kWarpSize = 32>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// fp16 -> fp32
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

// block reduce sum f16 -> f32
template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = val_f32;
    __syncthreads(); // wait all the warps done!
    val_f32 = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32); // WARNING: not all the threads get the same correct value, lane >= NUM_WARPS get 0
    return val_f32;
}

template<const int NUM_THREADS = 256 / 8>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g, float b, int N, int K) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    __shared__ float s_mean;
    __shared__ float s_var;
    constexpr float eps = 1e-5;
    half reg_x[8], reg_y[8];
    reinterpret_cast<float4*>(&reg_x[0])[0] = reinterpret_cast<float4*>(&x[idx])[0];
    float value = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value += idx + i < N * K ? __half2float(reg_x[i]) : 0.0f;
    }
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_mean = sum / K;
    __syncthreads();
    sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v_hat = __half2float(reg_x[i]) - s_mean;
        sum += idx + i < N * K ? v_hat * v_hat : 0.0f;
    }
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    if (threadIdx.x == 0) s_var = rsqrtf(sum / K + eps);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg_y[i] = __float2half((__half2float(reg_x[i]) - s_mean) * s_var * g + b);
    }
    if (idx + 7 < N * K) {
        reinterpret_cast<float4*>(&y[idx])[0] = reinterpret_cast<float4*>(&reg_y[0])[0];
    }
}


void layer_norm_cpu(const float *x, float *y, float g, float b, int N, int K) {
    constexpr float epsilon = 1e-5f;

    for (int i = 0; i < N; ++i) {
        const float *row_x = x + i * K;
        float *row_y = y + i * K;

        float mean = 0.0f;
        for (int j = 0; j < K; ++j) {
            mean += row_x[j];
        }
        mean /= K;

        float var = 0.0f;
        for (int j = 0; j < K; ++j) {
            float diff = row_x[j] - mean;
            var += diff * diff;
        }
        var /= K;

        float rstd = 1.0f / sqrtf(var + epsilon);

        for (int j = 0; j < K; ++j) {
            row_y[j] = (row_x[j] - mean) * rstd * g + b;
        }
    }
}


int main() {
    constexpr int N = 1024 * 256; // seqlen * batch
    constexpr int K = 256;
    constexpr int NUM_THREADS = 256;

    static_assert(NUM_THREADS == K, "Kernel implementation requires K == NUM_THREADS");

    constexpr float g = 1.0f;
    constexpr float b = 0.0f;
    constexpr auto num_elements = static_cast<size_t>(N * K);
    constexpr size_t size_bytes = num_elements * sizeof(half);

    std::cout << "Running LayerNorm Benchmark..." << std::endl;
    std::cout << "Parameters: N(rows)=" << N << ", K(cols/hidden_size)=" << K << std::endl;
    std::cout << "Number of elements: " << num_elements << ", Size: " << (double)size_bytes / (1024 * 1024) << " MB" << std::endl;

    float *h_x = new float[num_elements];
    float *h_y_cpu = new float[num_elements];
    half *h_x_f16 = new half[num_elements];
    half *h_y_gpu = new half[num_elements];

    half *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, size_bytes));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; i++) h_x[i] = distribution(generator);
    for (size_t i = 0; i < num_elements; i++) h_x_f16[i] = __float2half(h_x[i]);

    CUDA_CHECK(cudaMemcpy(d_x, h_x_f16, size_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(NUM_THREADS / 8);
    dim3 gridDim(N); // N * K / K = N

    std::cout << "Running GPU kernel..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    int num_runs = 10;
    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        layer_norm_f16x8_pack_f32_kernel<NUM_THREADS / 8><<<gridDim, blockDim>>>(d_x, d_y, g, b, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    CUDA_CHECK(cudaGetLastError());

    float milliseconds_gpu_total = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_gpu_total, start_gpu, stop_gpu));
    float milliseconds_gpu_avg = milliseconds_gpu_total / num_runs;

    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Running CPU implementation for verification..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    layer_norm_cpu(h_x, h_y_cpu, g, b, N, K);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> milliseconds_cpu = stop_cpu - start_cpu;

    std::cout << "Verifying results..." << std::endl;
    float max_abs_error = 0.0f;
    float sum_abs_error = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float diff = std::fabs(h_y_cpu[i] - __half2float(h_y_gpu[i]));
        sum_abs_error += diff;
        if (diff > max_abs_error) max_abs_error = diff;
    }

    float mae = sum_abs_error / num_elements;

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Verification:" << std::endl;
    std::cout << "  Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "  Mean Absolute Error (MAE): " << mae << std::endl;

    if (mae > 1e-3) {
        std::cout << "  [WARNING] High error detected. Check implementation." << std::endl;
    } else {
        std::cout << "  [SUCCESS] Results match." << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << std::setprecision(4);
    std::cout << "  CPU Time: " << milliseconds_cpu.count() << " ms" << std::endl;
    std::cout << "  GPU Time (avg over " << num_runs << " runs): " << milliseconds_gpu_avg << " ms" << std::endl;
    std::cout << "  Speedup (CPU Time / GPU Time): " << (milliseconds_cpu.count() / milliseconds_gpu_avg) << " X" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    delete[] h_x;
    delete[] h_x_f16;
    delete[] h_y_gpu;
    delete[] h_y_cpu;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    std::cout << "Done!" << std::endl;
    return 0;
}
