#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    unsigned int warp = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS]; // block shared memory
    float sum = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = sum;
    __syncthreads();
    sum = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum); // only warp 0 will use this sum value
    return sum;
}

// layer norm, x: NxK, y: NxK, y' = (x - mean(x))/std(x)
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf(sum((x - mean(x))^2) / K + epsilon)
// grid(N*K/K), block(K < 1024), N = batch_size * seq_len, K = hidden_size
// y = y' * g + b (g: scale, b: bias)
template<const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b, int N, int K) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float epsilon = 1e-5f;
    __shared__ float s_mean;
    __shared__ float s_var;
    float value = idx < N * K ? x[idx] : 0.0f; // use 0 to pad
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_mean = sum / static_cast<float>(K);
    __syncthreads(); // wait for s_mean in shared memory in all threads
    float var = (value - s_mean) * (value - s_mean);
    var = block_reduce_sum_f32<NUM_THREADS>(var);
    if (threadIdx.x == 0) s_var = rsqrtf(var / static_cast<float>(K) + epsilon);
    __syncthreads();
    if (idx < N * K) y[idx] = (value - s_mean) * s_var * g + b;
}

template<const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b, int N, int K) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float epsilon = 1e-5f;
    __shared__ float s_mean;
    __shared__ float s_var;
    float4 reg_x = reinterpret_cast<float4*>(&x[idx * 4])[0];
    float value = idx < N * K ? reg_x.x + reg_x.y + reg_x.z + reg_x.w : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) s_mean = sum / static_cast<float>(K);
    __syncthreads();
    float var = (reg_x.x - s_mean) * (reg_x.x - s_mean) + \
                (reg_x.y - s_mean) * (reg_x.y - s_mean) + \
                (reg_x.z - s_mean) * (reg_x.z - s_mean) + \
                (reg_x.w - s_mean) * (reg_x.w - s_mean);
    var = block_reduce_sum_f32<NUM_THREADS>(var);
    if (threadIdx.x == 0) s_var = rsqrtf(var / static_cast<float>(K) + epsilon);
    __syncthreads();
    float4 reg_y;
    if (idx < N * K) {
        reg_y.x = (reg_x.x - s_mean) * s_var * g + b;
        reg_y.y = (reg_x.y - s_mean) * s_var * g + b;
        reg_y.z = (reg_x.z - s_mean) * s_var * g + b;
        reg_y.w = (reg_x.w - s_mean) * s_var * g + b;
        reinterpret_cast<float4*>(&y[idx * 4])[0] = reg_y;
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
    const int N = 8192; // BatchSize * SeqLen
    const int K = 256;  // HiddenSize
    const int NUM_THREADS = 256;

    static_assert(K == NUM_THREADS, "Kernel implementation requires K == NUM_THREADS");

    const float g = 1.0f; // scale
    const float b = 0.0f; // bias
    const size_t num_elements = (size_t)N * K;
    const size_t size_bytes = num_elements * sizeof(float);

    std::cout << "Running LayerNorm Benchmark..." << std::endl;
    std::cout << "Parameters: N (rows) = " << N << ", K (cols/hidden_size) = " << K << std::endl;
    std::cout << "Total elements: " << num_elements << ", Size: " << (double)size_bytes / (1024 * 1024) << " MB" << std::endl;

    float *h_x, *h_y_gpu, *h_y_cpu;
    h_x = new float[num_elements];
    h_y_gpu = new float[num_elements];
    h_y_cpu = new float[num_elements];

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, size_bytes));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i) {
        h_x[i] = distribution(generator);
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(NUM_THREADS / 4);
    dim3 gridDim(N);

    std::cout << "Running GPU kernel..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    int num_runs = 100;
    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        // layer_norm_f32_kernel<NUM_THREADS><<<gridDim, blockDim>>>(d_x, d_y, g, b, N, K);
        layer_norm_f32x4_kernel<NUM_THREADS / 4><<<gridDim, blockDim>>>(d_x, d_y, g, b, N, K);
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
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        double diff = std::fabs(static_cast<double>(h_y_cpu[i]) - static_cast<double>(h_y_gpu[i]));
        sum_abs_error += diff;
        if (diff > max_abs_error) {
            max_abs_error = diff;
        }
    }
    double mae = sum_abs_error / num_elements;

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Verification:" << std::endl;
    std::cout << "  Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "  Mean Absolute Error (MAE): " << mae << std::endl;

    if (mae > 1e-5) {
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
    delete[] h_y_gpu;
    delete[] h_y_cpu;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    std::cout << "Benchmark complete." << std::endl;

    return 0;
}