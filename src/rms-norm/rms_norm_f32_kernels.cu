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


// block reduce sum f32 -> f32
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

// block reduce sum f16 -> f32
template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
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

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float epsilon = 1e-5f;
    __shared__ float s_var;

    float val = idx < N * K ? x[idx] : 0.0f;
    float sq_val = val * val; // Compute square for reduction

    float variance = block_reduce_sum_f32<NUM_THREADS>(sq_val);

    if (threadIdx.x == 0) s_var = rsqrtf(variance / K + epsilon);
    __syncthreads();

    if (idx < N * K) y[idx] = val * s_var * g; // Use original val, not sq_val
}

template<const int NUM_THREADS = 256 / 4>
__global__ void rms_norm_f32x4_kernel(float *x, float *y, float g, int N, int K) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    constexpr float epsilon = 1e-5f;
    __shared__ float s_var;

    // Use float4 to load data
    float4 reg_x = reinterpret_cast<float4*>(&x[idx])[0];

    // Compute sum of squares for this thread
    float sum_sq = (idx < N * K) ? (reg_x.x * reg_x.x +
                                    reg_x.y * reg_x.y +
                                    reg_x.z * reg_x.z +
                                    reg_x.w * reg_x.w) : 0.0f;

    // Block reduction to get sum of squares for the row
    float variance = block_reduce_sum_f32<NUM_THREADS>(sum_sq);

    if (threadIdx.x == 0) s_var = rsqrtf(variance / K + epsilon);
    __syncthreads();

    if (idx < N * K) {
        float4 reg_y;
        reg_y.x = reg_x.x * s_var * g;
        reg_y.y = reg_x.y * s_var * g;
        reg_y.z = reg_x.z * s_var * g;
        reg_y.w = reg_x.w * s_var * g;
        reinterpret_cast<float4*>(&y[idx])[0] = reg_y;
    }
}

// CPU Verification
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

    // Check assumption for vectorized kernel
    static_assert(K % 4 == 0, "K must be divisible by 4 for float4");
    static_assert(K == NUM_THREADS, "Kernel implementation currently assumes K == NUM_THREADS (for simplest block reduction)");

    const float g = 1.0f; // scale
    const size_t num_elements = (size_t)N * K;
    const size_t size_bytes = num_elements * sizeof(float);

    std::cout << "Running RMSNorm Benchmark..." << std::endl;
    std::cout << "Parameters: N (rows) = " << N << ", K (cols/hidden_size) = " << K << std::endl;
    std::cout << "Total elements: " << num_elements << ", Size: " << (double)size_bytes / (1024 * 1024) << " MB" << std::endl;

    // Allocate host memory
    float *h_x, *h_y_gpu, *h_y_cpu;
    h_x = new float[num_elements];
    h_y_gpu = new float[num_elements];
    h_y_cpu = new float[num_elements];

    // Allocate device memory
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, size_bytes));

    // Initialize input data
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < num_elements; ++i) {
        h_x[i] = distribution(generator);
    }

    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_bytes, cudaMemcpyHostToDevice));

    // Configuration for Vectorized (x4) Kernel
    // Each thread handles 4 elements, so we need 1/4th the threads per block
    dim3 blockDim(NUM_THREADS / 4);
    dim3 gridDim(N);

    std::cout << "Running GPU kernel (vectorized float4)..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    int num_runs = 100;

    // Warmup
    rms_norm_f32x4_kernel<NUM_THREADS / 4><<<gridDim, blockDim>>>(d_x, d_y, g, N, K);

    CUDA_CHECK(cudaEventRecord(start_gpu));
    for (int i = 0; i < num_runs; ++i) {
        // Uncomment to test scalar kernel (remember to change blockDim to NUM_THREADS)
        // rms_norm_f32_kernel<NUM_THREADS><<<gridDim, NUM_THREADS>>>(d_x, d_y, g, N, K);

        // Running Vectorized Kernel
        rms_norm_f32x4_kernel<NUM_THREADS / 4><<<gridDim, blockDim>>>(d_x, d_y, g, N, K);
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
    rms_norm_cpu(h_x, h_y_cpu, g, N, K);
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

    // RMSNorm can be slightly less stable than LayerNorm due to rsqrt, slightly relax tolerance if needed
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

    // Cleanup
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