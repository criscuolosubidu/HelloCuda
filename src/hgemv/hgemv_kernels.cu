#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
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

// ----------------------------------------------------------------------
// Device Helper & Kernels (User Provided)
// ----------------------------------------------------------------------

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// grid(M/4), block(32,4) blockDim.x = 32 = K, blockDim.y = 4
// a: MxK, x: Kx1, y: Mx1, compute y = a * x
// assume that K % 32 == 0 and 1 warp for 1 row
__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;
        half sum = __float2half(0.0f);
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            sum += a[row * K + k] * x[k];
        }
        sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
        if (threadIdx.x == 0) y[row] = sum;
    }
}

// grid(M/4), block(32,4) blockDim.x = 32 = K, blockDim.y = 4
// a: MxK, x: Kx1, y: Mx1, compute y = a * x
// assume that k % 128 == 0 and 1 warp for 1 row
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M) {
        int NUM_WARPS = ((K + WARP_SIZE - 1) / WARP_SIZE + 3) / 4;
        int lane = threadIdx.x % WARP_SIZE;
        half sum = __float2half(0.0f);
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * 4 * WARP_SIZE + lane * 4;
            // NOTE: Original code used explicit casts. Ensure alignment or correctness in logic.
            half2 reg_a_0 = reinterpret_cast<half2 *>(&a[row * K + k])[0];
            half2 reg_a_1 = reinterpret_cast<half2 *>(&a[row * K + k + 2])[0];
            half2 reg_x_0 = reinterpret_cast<half2 *>(&x[k])[0];
            half2 reg_x_1 = reinterpret_cast<half2 *>(&x[k + 2])[0];
            sum += (reg_a_0.x * reg_x_0.x + reg_a_0.y * reg_x_0.y + reg_a_1.x * reg_x_1.x + reg_a_1.y * reg_x_1.y);
        }
        sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
        if (threadIdx.x == 0) y[row] = sum;
    }
}

// NUM_THREADS = 32 * 4 = 128, NUM_WARPS = NUM_THREADS / WARP_SIZE = 4
// NUM_ROWS = NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute y = a * x
// assume that k % 16 == 0 and 1 warp for 2 row
template<const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *a, half *x, half *y, int M, int K) {
    // NOTE: Based on provided code.
    // In block(32, 4), threadIdx.y is 0..3. threadIdx.x is 0..31.
    int lane = threadIdx.x % WARP_SIZE;
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP; // 16

    // Calculation:
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_PER_WARP + lane / K_WARP_SIZE;

    if (row < M) {
        int k = threadIdx.x % K_WARP_SIZE;
        half sum = a[row * K + k] * x[k];
        sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
        if (k == 0) y[row] = sum;
    }
}

// ----------------------------------------------------------------------
// CPU Verification (Simulation in Float for accuracy)
// ----------------------------------------------------------------------
void hgemv_cpu(const half *a, const half *x, float *y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            // Convert half to float for CPU calculation reference
            float val_a = __half2float(a[i * K + j]);
            float val_x = __half2float(x[j]);
            sum += val_a * val_x;
        }
        y[i] = sum;
    }
}

// ----------------------------------------------------------------------
// Test Runner Helper
// ----------------------------------------------------------------------
void run_test(int M, int K, const std::string &kernel_name) {
    size_t size_a = (size_t) M * K * sizeof(half);
    size_t size_x = (size_t) K * sizeof(half);
    size_t size_y = (size_t) M * sizeof(half);

    std::cout << "Testing " << kernel_name << " [M=" << M << ", K=" << K << "]..." << std::endl;

    // Host allocation (using half)
    std::vector<half> h_a(M * K);
    std::vector<half> h_x(K);
    std::vector<half> h_y_gpu(M);
    std::vector<float> h_y_cpu(M); // CPU result computed in float

    // Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    // Initialize with random floats, converted to half
    for (int i = 0; i < M * K; ++i) h_a[i] = __float2half(distribution(generator));
    for (int i = 0; i < K; ++i) h_x[i] = __float2half(distribution(generator));

    // Device allocation
    half *d_a, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_x, size_x));
    CUDA_CHECK(cudaMalloc(&d_y, size_y));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice));

    // Define Grid/Block dimensions based on Kernel logic
    dim3 blockDim, gridDim;

    if (K == 16) {
        // K=16 logic: 1 warp handles 2 rows.
        // User Comment: block(32, NUM_WARPS). Let's pick NUM_WARPS=4.
        // blockDim = (32, 4)
        int WARPS_PER_BLOCK = 4;
        blockDim = dim3(32, WARPS_PER_BLOCK);

        // Rows handled per block: Each warp handles 2 rows.
        int rows_per_warp = 2;
        int rows_per_block = WARPS_PER_BLOCK * rows_per_warp; // 8 rows per block

        // Grid logic: User comment said grid(M/NUM_ROWS)
        gridDim = dim3(1, (M + rows_per_block - 1) / rows_per_block);
    } else {
        // K=32 and K=128 logic: 1 warp handles 1 row.
        // User Comment: grid(M/4), block(32, 4)
        // blockDim.y = 4 means 4 warps per block (since x=32).
        // Each warp handles 1 row -> 4 rows per block.
        blockDim = dim3(32, 4);
        int rows_per_block = 4;
        gridDim = dim3(1, (M + rows_per_block - 1) / rows_per_block);
    }

    // Warmup
    if (K == 16) hgemv_k16_f16_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    else if (K == 32) hgemv_k32_f16_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    else if (K == 128) hgemv_k128_f16x4_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        if (K == 16) hgemv_k16_f16_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
        else if (K == 32) hgemv_k32_f16_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
        else if (K == 128) hgemv_k128_f16x4_kernel<<<gridDim, blockDim>>>(d_a, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / num_runs;

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, size_y, cudaMemcpyDeviceToHost));

    // Verification
    // Compute expected result on CPU
    hgemv_cpu(h_a.data(), h_x.data(), h_y_cpu.data(), M, K);

    double max_diff = 0.0;
    double total_diff = 0.0;
    for (int i = 0; i < M; ++i) {
        // Convert GPU half result back to float for comparison
        float gpu_val = __half2float(h_y_gpu[i]);
        float cpu_val = h_y_cpu[i];

        double diff = std::fabs(static_cast<double>(cpu_val) - static_cast<double>(gpu_val));
        total_diff += diff;
        if (diff > max_diff) max_diff = diff;
    }
    double mae = total_diff / M;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  GPU Avg Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Max Abs Error: " << max_diff << std::endl;
    std::cout << "  MAE: " << mae << std::endl;

    // FP16 has lower precision, so we relax the threshold slightly compared to FP32 (e.g., 1e-3)
    if (mae < 1e-2) std::cout << "  [SUCCESS] Verification passed." << std::endl;
    else std::cout << "  [WARNING] Verification failed (high error). Check kernel logic." << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "Running HGEMV (FP16) Kernel Benchmarks" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // 1. Test K=32 (1 warp per row)
    run_test(4096, 32, "hgemv_k32_f16_kernel");

    // 2. Test K=128 (1 warp per row, vectorized half2)
    run_test(4096, 128, "hgemv_k128_f16x4_kernel");

    // 3. Test K=16 (1 warp per 2 rows)
    run_test(4096, 16, "hgemv_k16_f16_kernel");

    return 0;
}
