#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define WARP_SIZE 32

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

// ----------------------------------------------------------------------
// Kernels provided
// ----------------------------------------------------------------------

// a: MxK, x: Kx1, y: Mx1, y = a * x
// K = 32
// Launch: block(32, NUM_WARPS), grid(1, (M + NUM_WARPS - 1) / NUM_WARPS)
// Each warp handles 1 row
__global__ void sgemv_k32_f32_kernel(float *a, float *x, float *y, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        int lane = threadIdx.x % WARP_SIZE;
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE; // Should be 1 for K=32
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            sum += a[row * K + k] * x[k];
        }
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
        if (lane == 0) y[row] = sum;
    }
}

// a: MxK, x: Kx1, y: Mx1, y = a * x
// K = 128
// Launch: block(32, NUM_WARPS), grid(1, (M + NUM_WARPS - 1) / NUM_WARPS)
// Each warp handles 1 row (using float4, 32 threads * 4 = 128 elements)
__global__ void sgemv_k128_f32x4_kernel(float *a, float *x, float *y, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
        NUM_WARPS = (NUM_WARPS + 3) / 4; // 1 warp 128 elements via float4
        int lane = threadIdx.x % WARP_SIZE;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_a = reinterpret_cast<float4*>(&a[row * K + k])[0];
            float4 reg_x = reinterpret_cast<float4*>(&x[k])[0];
            sum += reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z + reg_a.w * reg_x.w;
        }
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
        if (lane == 0) y[row] = sum;
    }
}

// assume K = 16, 1 warp for 2 rows
// NUM_THREADS = 128, NUM_WARPS = NUM_THREADS / WARP_SIZE
// NUM_ROWS = NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32, NUM_WARPS)
template<const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *a, float *x, float *y, int M, int K) {
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP; // 16
    int lane = threadIdx.x % WARP_SIZE;
    // Calculation: (warp_index) * rows_per_warp + (0 or 1 depending on lane)
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_PER_WARP + lane / K_WARP_SIZE;

    if (row < M) {
        int k = threadIdx.x % K_WARP_SIZE; // 0..15
        float sum = a[row * K + k] * x[k];
        sum = warp_reduce_sum_f32<K_WARP_SIZE>(sum);
        // Only the first thread of the sub-group writes the result
        if (k == 0) y[row] = sum;
    }
}

// ----------------------------------------------------------------------
// CPU Verification
// ----------------------------------------------------------------------
void sgemv_cpu(const float *a, const float *x, float *y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            sum += a[i * K + j] * x[j];
        }
        y[i] = sum;
    }
}

// ----------------------------------------------------------------------
// Test Runner Helper
// ----------------------------------------------------------------------
void run_test(int M, int K, const std::string& kernel_name) {
    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_x = (size_t)K * sizeof(float);
    size_t size_y = (size_t)M * sizeof(float);

    std::cout << "Testing " << kernel_name << " [M=" << M << ", K=" << K << "]..." << std::endl;

    // Host allocation
    float *h_a = new float[M * K];
    float *h_x = new float[K];
    float *h_y_gpu = new float[M];
    float *h_y_cpu = new float[M];

    // Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) h_a[i] = distribution(generator);
    for (int i = 0; i < K; ++i) h_x[i] = distribution(generator);

    // Device allocation
    float *d_a, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_x, size_x));
    CUDA_CHECK(cudaMalloc(&d_y, size_y));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));

    // Define Grid/Block dimensions based on K
    dim3 blockDim, gridDim;

    // Config logic
    int WARPS_PER_BLOCK = 4; // Arbitrary choice, usually 4 or 8 works well
    blockDim = dim3(WARP_SIZE, WARPS_PER_BLOCK);

    if (K == 16) {
        // K=16: 1 warp handles 2 rows.
        // Rows per block = WARPS_PER_BLOCK * 2
        int rows_per_block = WARPS_PER_BLOCK * 2;
        gridDim = dim3(1, (M + rows_per_block - 1) / rows_per_block);
    } else {
        // K=32 and K=128: 1 warp handles 1 row.
        // Rows per block = WARPS_PER_BLOCK
        gridDim = dim3(1, (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    }

    // Warmup
    if (K == 16) sgemv_k16_f32_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    else if (K == 32) sgemv_k32_f32_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    else if (K == 128) sgemv_k128_f32x4_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < num_runs; ++i) {
        if (K == 16) sgemv_k16_f32_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
        else if (K == 32) sgemv_k32_f32_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
        else if (K == 128) sgemv_k128_f32x4_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / num_runs;

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size_y, cudaMemcpyDeviceToHost));

    // Verification
    sgemv_cpu(h_a, h_x, h_y_cpu, M, K);

    double max_diff = 0.0;
    double total_diff = 0.0;
    for (int i = 0; i < M; ++i) {
        double diff = std::fabs(static_cast<double>(h_y_cpu[i]) - static_cast<double>(h_y_gpu[i]));
        total_diff += diff;
        if (diff > max_diff) max_diff = diff;
    }
    double mae = total_diff / M;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  GPU Avg Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Max Abs Error: " << max_diff << std::endl;
    std::cout << "  MAE: " << mae << std::endl;

    if (mae < 1e-4) std::cout << "  [SUCCESS] Verification passed." << std::endl;
    else std::cout << "  [WARNING] Verification failed (high error)." << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // Cleanup
    delete[] h_a; delete[] h_x; delete[] h_y_gpu; delete[] h_y_cpu;
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "Running SGEMV Kernel Benchmarks" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // 1. Test K=32 (1 warp per row)
    run_test(4096, 32, "sgemv_k32_f32_kernel");

    // 2. Test K=128 (1 warp per row, vectorized float4)
    run_test(4096, 128, "sgemv_k128_f32x4_kernel");

    // 3. Test K=16 (1 warp per 2 rows)
    run_test(4096, 16, "sgemv_k16_f32_kernel");

    return 0;
}