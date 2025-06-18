#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_memory.h"

static constexpr size_t N = 10'000'000;
static constexpr size_t BLOCK_SIZE = 256;

// 计时工具函数
template<typename Func>
double measure_time(Func&& func, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << name << " time: " << elapsed.count() * 1000 << " ms\n";
    return elapsed.count();
}

void init_vector(std::vector<float>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : vec) v = dis(gen);
}

void vector_add_cpu_1d(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(const float* a, const float* b, float* c, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_1d() {
    // 初始化数据
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);
    init_vector(h_a);
    init_vector(h_b);

    // 分配设备内存
    CudaMemory<float> d_a(N);
    CudaMemory<float> d_b(N);
    CudaMemory<float> d_c(N);

    // 拷贝数据到设备
    cudaMemcpy(d_a.get(), h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.get(), h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 预热运行
    std::cout << "Performing warm-up runs..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        vector_add_cpu_1d(h_a, h_b, h_c_cpu);
        vector_add_gpu_1d<<<numBlocks, BLOCK_SIZE>>>(d_a.get(), d_b.get(), d_c.get(), N);
        cudaDeviceSynchronize();
    }

    // 测量CPU时间
    double cpu_time = measure_time([&]() {
        vector_add_cpu_1d(h_a, h_b, h_c_cpu);
    }, "CPU");

    // 测量GPU时间（包含内存传输）
    double gpu_total_time = measure_time([&]() {
        vector_add_gpu_1d<<<numBlocks, BLOCK_SIZE>>>(d_a.get(), d_b.get(), d_c.get(), N);
        cudaDeviceSynchronize(); // 确保内核执行完成
        cudaMemcpy(h_c_gpu.data(), d_c.get(), N * sizeof(float), cudaMemcpyDeviceToHost);
    }, "GPU (total)");

    // 仅测量GPU计算时间（不包含内存传输）
    double gpu_compute_time = measure_time([&]() {
        vector_add_gpu_1d<<<numBlocks, BLOCK_SIZE>>>(d_a.get(), d_b.get(), d_c.get(), N);
        cudaDeviceSynchronize();
    }, "GPU (compute only)");

    // 验证结果
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_c_cpu[i] - h_c_gpu[i]) > 0.000001f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i]
                      << ", GPU=" << h_c_gpu[i] << std::endl;
            break;
        }
    }

    // 输出性能对比
    std::cout << "\nPerformance Summary:\n";
    std::cout << "CPU time:          " << cpu_time * 1000 << " ms\n";
    std::cout << "GPU total time:    " << gpu_total_time * 1000 << " ms (includes memory transfer)\n";
    std::cout << "GPU compute time:  " << gpu_compute_time * 1000 << " ms (kernel only)\n";
    std::cout << "Speedup (compute): " << cpu_time / gpu_compute_time << "x\n";
    std::cout << "Speedup (total):   " << cpu_time / gpu_total_time << "x\n";

    std::cout << "\nVector addition in 1d array completed successfully!\n";
}
