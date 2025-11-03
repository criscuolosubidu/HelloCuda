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

// block reduce sum f16 -> f16
template<const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ half reduce_sum[NUM_WARPS];
    val = warp_reduce_sum_f16<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = val;
    __syncthreads();
    val = lane < NUM_WARPS ? reduce_sum[lane] : __float2half(0.0f);
    val = warp_reduce_sum_f16<NUM_WARPS>(val); // WARNING: not all the threads get the same correct value, lane >= NUM_WARPS get 0
    return val;
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

// N = seqlen * batch, K is the dimension, N * K
// one block for K, so we need N blocks
// https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
template<const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half epsilon = __float2half(1e-5f);
    __shared__ half s_mean;
    __shared__ half s_variance;
    half value = idx < N * K ? x[idx] : __float2half(0.0f);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_mean = sum / __int2half_rn(K); // the first thread to calculate
    __syncthreads(); // wait until s_mean is ready
    value = (value - s_mean) * (value - s_mean);
    value = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_variance = hrsqrt(__hadd(value / __int2half_rn(K), epsilon));
    __syncthreads(); // wait until s_variance is ready
    if (idx < N * K) y[idx] = (x[idx] - s_mean) * s_variance * __float2half(g) + __float2half(b);
}

template<const int NUM_THREADS = 256 / 2>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float g, float b, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half epsilon = __float2half(1e-5f);
    __shared__ half s_mean;
    __shared__ half s_variance;
    half2 reg_x = reinterpret_cast<half2*>(&x[idx * 2])[0];
    half value = idx * 2 < N * K ? reg_x.x + reg_x.y : __float2half(0.0f);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_mean = sum / __int2half_rn(K);
    __syncthreads();
    value = idx * 2 < N * K ? (reg_x.x - s_mean) * (reg_x.x - s_mean) + (reg_x.y - s_mean) * (reg_x.y - s_mean) : __float2half(0.0f);
    value = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_variance = hrsqrt(__hadd(value / __int2half_rn(K), epsilon));
    __syncthreads();
    if (2 * idx < N * K) {
        reg_x.x = (reg_x.x - s_mean) * s_variance * __float2half(g) + __float2half(b);
        reg_x.y = (reg_x.y - s_mean) * s_variance * __float2half(g) + __float2half(b);
        reinterpret_cast<half2*>(&y[idx * 2])[0] = reg_x;
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

    const float g = 1.0f;
    const float b = 0.0f;
    const size_t num_elements = (size_t)N * K;
    const size_t size_bytes = num_elements * sizeof(half);

    std::cout << "Running LayerNorm Benchmark..." << std::endl;
    std::cout << "Parameters: N(rows)=" << N << ", K(cols/hidden_size)=" << K << std::endl;
    std::cout << "Number of elements: " << num_elements << ", Size: " << (double)size_bytes / (1024 * 1024) << " MB" << std::endl;

    float *h_x, *h_y_cpu;
    half *h_x_f16, *h_y_gpu;
    h_x = new float[num_elements];
    h_y_cpu = new float[num_elements];
    h_y_gpu = new half[num_elements];
    h_x_f16 = new half[num_elements];

    half *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, size_bytes));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; i++) h_x[i] = distribution(generator);
    for (size_t i = 0; i < num_elements; i++) h_x_f16[i] = __float2half(h_x[i]);

    CUDA_CHECK(cudaMemcpy(d_x, h_x_f16, size_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(NUM_THREADS / 2);
    dim3 gridDim(N); // N * K / K = N

    std::cout << "Running GPU kernel..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    int num_runs = 10;
    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        // layer_norm_f16_f16_kernel<NUM_THREADS><<<gridDim, blockDim>>>(d_x, d_y, g, b, N, K);
        layer_norm_f16x2_f16_kernel<NUM_THREADS / 2><<<gridDim, blockDim>>>(d_x, d_y, g, b, N, K);
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
