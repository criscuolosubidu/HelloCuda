#include <iostream>
#include <vector>
#include <chrono>
#include <cmath> // For std::sin, std::cos, std::pow, std::abs
#include <random> // For generating random test data
#include <iomanip> // For std::fixed and std::setprecision

#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static constexpr float theta = 10000.0f;

/**
 * @brief RoPE CUDA Kernel for float32
 * @param x Input tensor of shape (seq_len, N_dim)
 * @param out Output tensor of shape (seq_len, N_dim)
 * @param seq_len Sequence length, number of tokens
 * @param N_dim_half Half of the feature dimension (N_dim / 2), as we process pairs.
 */
__global__ void rope_f32_kernel(const float *x, float *out, int seq_len, int N_dim_half) {
    const unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= seq_len * N_dim_half) return;

    // every token should have N_dim_half pairs
    const unsigned int token_pos = pair_idx / N_dim_half;
    // the feature idx int current token
    const unsigned int feature_pair_idx = pair_idx % N_dim_half;

    // calculate the sin and cos value
    const float inv_freq = 1.0f / powf(theta, static_cast<float>(feature_pair_idx) * 2.0f / (static_cast<float>(N_dim_half) * 2.0f));
    const float freq = static_cast<float>(token_pos) * inv_freq;
    const float sin_v = sinf(freq);
    const float cos_v = cosf(freq);

    const unsigned int data_idx_base = token_pos * (N_dim_half * 2) + feature_pair_idx * 2;
    const float x1 = x[data_idx_base];
    const float x2 = x[data_idx_base + 1];
    const float out1 = x1 * cos_v - x2 * sin_v;
    const float out2 = x1 * sin_v + x2 * cos_v;

    out[data_idx_base] = out1;
    out[data_idx_base + 1] = out2;
}

__global__ void rope_f32x4_kernel(float *x, float *out, int seq_len, int N_dim_half) {
    // we guarantee that d can be divided by 4
    const unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= seq_len * N_dim_half) return;

    const unsigned int token_pos = pair_idx * 2 / N_dim_half;
    const unsigned int feature_pair_idx = pair_idx * 2 % N_dim_half;

    const float inv_freq_1 = 1.0f / powf(theta, static_cast<float>(feature_pair_idx) * 2.0f / (static_cast<float>(N_dim_half) * 2.0f));
    const float inv_freq_2 = 1.0f / powf(theta, static_cast<float>(feature_pair_idx + 1) * 2.0f / (static_cast<float>(N_dim_half) * 2.0f));
    const float freq1 = static_cast<float>(token_pos) * inv_freq_1;
    const float freq2 = static_cast<float>(token_pos) * inv_freq_2;
    const float sin_v_1 = sinf(freq1);
    const float sin_v_2 = sinf(freq2);
    const float cos_v_1 = cosf(freq1);
    const float cos_v_2 = cosf(freq2);

    const unsigned int data_idx_base = token_pos * (N_dim_half * 2) + feature_pair_idx * 2;
    float4 reg_x = reinterpret_cast<float4*>(&x[data_idx_base])[0];
    const float out1 = reg_x.x * cos_v_1 - reg_x.y * sin_v_1;
    const float out2 = reg_x.x * sin_v_1 + reg_x.y * cos_v_1;
    const float out3 = reg_x.z * cos_v_2 - reg_x.w * sin_v_2;
    const float out4 = reg_x.z * sin_v_2 + reg_x.w * cos_v_2;

    out[data_idx_base] = out1;
    out[data_idx_base + 1] = out2;
    out[data_idx_base + 2] = out3;
    out[data_idx_base + 3] = out4;
}

/**
 * @brief CPU version of RoPE for verification
 */
void rope_cpu(const std::vector<float>& x, std::vector<float>& out, int seq_len, int N_dim) {
    int N_dim_half = N_dim / 2;
    out.resize(x.size());

    // Iterate over each token in the sequence
    for (int pos = 0; pos < seq_len; ++pos) {
        // Iterate over each pair of features in the dimension
        for (int i = 0; i < N_dim_half; ++i) {
            float inv_freq = 1.0f / std::pow(theta, static_cast<float>(i) * 2.0f / static_cast<float>(N_dim));
            float freq = static_cast<float>(pos) * inv_freq;
            float cos_v = std::cos(freq);
            float sin_v = std::sin(freq);

            int data_idx_base = pos * N_dim + i * 2;
            float x1 = x[data_idx_base];
            float x2 = x[data_idx_base + 1];

            out[data_idx_base] = x1 * cos_v - x2 * sin_v;
            out[data_idx_base + 1] = x1 * sin_v + x2 * cos_v;
        }
    }
}


int main() {
    // set test parameters
    constexpr int seq_len = 4096; // sequence length
    constexpr int N_dim = 1024;   // feature dim, in traditional transformers is 1024 or 2048

    if (N_dim % 2 != 0) {
        std::cerr << "N_dim must be an even number." << std::endl;
        return 1;
    }

    constexpr int N_dim_half = N_dim / 2;
    constexpr size_t total_elements = static_cast<size_t>(seq_len) * N_dim;
    constexpr size_t total_pairs = total_elements / 2;
    constexpr size_t bytes = total_elements * sizeof(float);

    std::cout << "--- Test Configuration ---" << std::endl;
    std::cout << "Sequence Length: " << seq_len << std::endl;
    std::cout << "Feature Dimension: " << N_dim << std::endl;
    std::cout << "Total float elements: " << total_elements << std::endl;
    std::cout << "Memory size: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "--------------------------" << std::endl << std::endl;


    // generate random float numbers
    std::vector<float> h_x(total_elements);
    std::vector<float> h_out_gpu(total_elements);
    std::vector<float> h_out_cpu(total_elements);

    std::mt19937 gen(12345); // a fixed seed for reproducibility
    std::uniform_real_distribution dis(-1.0f, 1.0f);
    for (size_t i = 0; i < total_elements; ++i) {
        h_x[i] = dis(gen);
    }

    float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Running GPU Kernel..." << std::endl;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    constexpr int block_size = 256;
    constexpr int grid_size = (total_pairs + block_size - 1) / block_size;

    CHECK_CUDA(cudaEventRecord(start));
    // rope_f32_kernel<<<grid_size, block_size>>>(d_x, d_out, seq_len, N_dim_half);
    rope_f32x4_kernel<<<grid_size, block_size / 2>>>(d_x, d_out, seq_len, N_dim_half);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_milliseconds, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Running CPU implementation for verification..." << std::endl;
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    rope_cpu(h_x, h_out_cpu, seq_len, N_dim);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_time - cpu_start_time);
    double cpu_milliseconds = cpu_duration.count() / 1000.0;

    // calculate the float eps
    std::cout << "Verifying results and calculating error..." << std::endl;
    double total_error = 0.0;
    bool mismatch_found = false;
    constexpr float tolerance = 1e-3f;

    for (size_t i = 0; i < total_elements; ++i) {
        float diff = std::abs(h_out_cpu[i] - h_out_gpu[i]);
        total_error += diff;
        if (diff > tolerance && !mismatch_found) {
            std::cerr << "Mismatch found at index " << i << "!" << std::endl;
            std::cerr << "CPU value: " << h_out_cpu[i] << ", GPU value: " << h_out_gpu[i] << ", Diff: " << diff << std::endl;
            mismatch_found = true;
        }
    }
    double mean_absolute_error = total_error / total_elements;
    std::cout << "\n--- Performance and Accuracy Report ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU execution time: " << cpu_milliseconds << " ms" << std::endl;
    std::cout << "GPU execution time: " << gpu_milliseconds << " ms" << std::endl;
    if (gpu_milliseconds > 0) {
        std::cout << "Performance Speedup (CPU/GPU): " << cpu_milliseconds / gpu_milliseconds << "x" << std::endl;
    }
    std::cout << "Verification Status: " << (mismatch_found ? "FAILED" : "PASSED") << std::endl;
    std::cout << std::scientific;
    std::cout << "Mean Absolute Error (MAE): " << mean_absolute_error << std::endl;
    std::cout << std::fixed;
    std::cout << "---------------------------------------" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}