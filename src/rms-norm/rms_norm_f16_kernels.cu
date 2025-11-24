#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
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

// block reduce sum f32 -> f32
template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    unsigned int warp = threadIdx.x / WARP_SIZE;
    unsigned int lane = threadIdx.x % WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float sum = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = sum;
    __syncthreads();
    sum = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    return sum;
}

// block reduce sum f16 -> f32
template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float reduce_sum[NUM_WARPS];
    float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
    if (lane == 0) reduce_sum[warp] = val_f32;
    __syncthreads();
    val_f32 = lane < NUM_WARPS ? reduce_sum[lane] : 0.0f;
    val_f32 = warp_reduce_sum_f32<NUM_WARPS>(val_f32);
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
    val = warp_reduce_sum_f16<NUM_WARPS>(val);
    return val;
}

// RMS Norm Kernel (Scalar FP16)
template<const int NUM_THREADS = 256>
__global__ void rms_norm_f16_f16_kernel(half *x, half *y, float g, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float epsilon = 1e-5f;
    __shared__ half s_var;
    half value = idx < N * K ? x[idx] : __float2half(0.0f);
    // Note: multiplying half*half might overflow if values are large, but ok for typical RMSNorm inputs
    half variance = block_reduce_sum_f16_f16<NUM_THREADS>(value * value);
    if (threadIdx.x == 0) s_var = hrsqrt(variance / __int2half_rn(K) + __float2half(epsilon));
    __syncthreads();
    if (idx < N * K) y[idx] = value * s_var * __float2half(g);
}

// RMS Norm Kernel (Vectorized FP16x2)
// NOTE: I fixed the 'hsqrt' to 'hrsqrt' here.
template<const int NUM_THREADS = 256 / 2>
__global__ void rms_norm_f16x2_f16_kernel(half *x, half *y, float g, int N, int K) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const float epsilon = 1e-5f;
    __shared__ half s_var;

    // Load 2 halves at once
    half2 reg_x = reinterpret_cast<half2*>(&x[idx])[0];

    // Compute squared sum for this thread (dot product with itself)
    half variance = idx < N * K ? reg_x.x * reg_x.x + reg_x.y * reg_x.y : __float2half(0.0f);

    // Reduce across block
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);

    // Calculate inverse RMS (rsqrt), NOT sqrt
    if (threadIdx.x == 0) s_var = hrsqrt(variance / __int2half_rn(K) + __float2half(epsilon));
    __syncthreads();

    if (idx < N * K) {
        half2 reg_y;
        reg_y.x = reg_x.x * s_var * __float2half(g);
        reg_y.y = reg_x.y * s_var * __float2half(g);
        reinterpret_cast<half2*>(&y[idx])[0] = reg_y;
    }
}

// Helper to convert array float -> half
void float2half_array(const float* src, half* dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

// Helper to convert array half -> float
void half2float_array(const half* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = __half2float(src[i]);
    }
}

// CPU Implementation (Ground Truth using Float)
void rms_norm_cpu(const float *x, float *y, float g, int N, int K) {
    constexpr float epsilon = 1e-5f;

    for (int i = 0; i < N; ++i) {
        const float *row_x = x + i * K;
        float *row_y = y + i * K;

        float sum_sq = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum_sq += row_x[j] * row_x[j];
        }

        float rms_inv = 1.0f / sqrtf(sum_sq / K + epsilon);

        for (int j = 0; j < K; ++j) {
            row_y[j] = row_x[j] * rms_inv * g;
        }
    }
}

int main() {
    const int N = 8192; // BatchSize * SeqLen
    const int K = 256;  // HiddenSize
    const int NUM_THREADS = 256;

    static_assert(K % 2 == 0, "K must be divisible by 2 for half2 vectorization");
    static_assert(K == NUM_THREADS, "Kernel implementation currently assumes K == NUM_THREADS");

    const float g = 1.0f; // scale
    const size_t num_elements = (size_t)N * K;
    const size_t size_bytes = num_elements * sizeof(half);

    std::cout << "Running FP16 RMSNorm Benchmark..." << std::endl;
    std::cout << "Parameters: N (rows) = " << N << ", K (cols/hidden_size) = " << K << std::endl;
    std::cout << "Total elements: " << num_elements << ", Size: " << (double)size_bytes / (1024 * 1024) << " MB" << std::endl;

    // Allocate host memory
    // We use float on host for generation and verification for convenience and precision
    float *h_x_float = new float[num_elements];
    float *h_y_cpu_float = new float[num_elements]; // CPU result
    float *h_y_gpu_float = new float[num_elements]; // GPU result converted back

    // Host side buffers for copying to/from device
    half *h_x_half = new half[num_elements];
    half *h_y_gpu_half = new half[num_elements];

    // Allocate device memory
    half *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, size_bytes));

    // Initialize input data (Random Floats -> Half)
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i) {
        h_x_float[i] = distribution(generator);
    }
    // Convert to half
    float2half_array(h_x_float, h_x_half, num_elements);

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_x, h_x_half, size_bytes, cudaMemcpyHostToDevice));

    // Define Grid/Block
    dim3 gridDim(N);
    // Block size depends on kernel (Scalar: K threads, Vectorized: K/2 threads)

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    int num_runs = 100;

    // --- 1. Test Vectorized Kernel (f16x2) ---
    std::cout << "Running GPU Kernel (Vectorized half2)..." << std::endl;
    dim3 blockDimVec(NUM_THREADS / 2);

    // Warmup
    rms_norm_f16x2_f16_kernel<NUM_THREADS / 2><<<gridDim, blockDimVec>>>(d_x, d_y, g, N, K);

    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        rms_norm_f16x2_f16_kernel<NUM_THREADS / 2><<<gridDim, blockDimVec>>>(d_x, d_y, g, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    float milliseconds_vec = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_vec, start_gpu, stop_gpu));
    float avg_time_vec = milliseconds_vec / num_runs;

    // Copy result back for verification (from vectorized run)
    CUDA_CHECK(cudaMemcpy(h_y_gpu_half, d_y, size_bytes, cudaMemcpyDeviceToHost));

    // --- Verification ---
    std::cout << "Running CPU implementation (Float32) for verification..." << std::endl;
    rms_norm_cpu(h_x_float, h_y_cpu_float, g, N, K);

    // Convert GPU result to float for comparison
    half2float_array(h_y_gpu_half, h_y_gpu_float, num_elements);

    std::cout << "Verifying Vectorized FP16 results..." << std::endl;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        double diff = std::fabs(static_cast<double>(h_y_cpu_float[i]) - static_cast<double>(h_y_gpu_float[i]));
        sum_abs_error += diff;
        if (diff > max_abs_error) {
            max_abs_error = diff;
        }
    }
    double mae = sum_abs_error / num_elements;

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "  Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "  Mean Absolute Error (MAE): " << mae << std::endl;

    // Precision warning: fp16 has lower precision, so tolerance is higher (e.g., 1e-3)
    if (mae > 1e-2) {
         std::cout << "  [WARNING] High error detected. Check implementation or precision issues." << std::endl;
    } else {
         std::cout << "  [SUCCESS] Results match (within FP16 tolerance)." << std::endl;
    }

    std::cout << "  Vectorized Kernel Time: " << avg_time_vec << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- 2. Test Scalar Kernel (Optional, for comparison) ---
    /*
    std::cout << "Running GPU Kernel (Scalar half)..." << std::endl;
    dim3 blockDimScalar(NUM_THREADS);
    rms_norm_f16_f16_kernel<NUM_THREADS><<<gridDim, blockDimScalar>>>(d_x, d_y, g, N, K); // Warmup

    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        rms_norm_f16_f16_kernel<NUM_THREADS><<<gridDim, blockDimScalar>>>(d_x, d_y, g, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    float milliseconds_scalar = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_scalar, start_gpu, stop_gpu));
    std::cout << "  Scalar Kernel Time: " << milliseconds_scalar / num_runs << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    */

    // Cleanup
    delete[] h_x_float;
    delete[] h_y_cpu_float;
    delete[] h_y_gpu_float;
    delete[] h_x_half;
    delete[] h_y_gpu_half;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    std::cout << "Benchmark complete." << std::endl;

    return 0;
}