#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_memory.h"

static constexpr size_t N = 10'000'000;
static constexpr size_t BLOCK_SIZE = 256;
static constexpr size_t BLOCK_SIZE_3D_X = 10;
static constexpr size_t BLOCK_SIZE_3D_Y = 10;
static constexpr size_t BLOCK_SIZE_3D_Z = 10;

// 计时工具函数
template<typename Func>
double measure_time(Func&& func, const std::string& name, int iterations = 1) {
    double total_time = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }
    double avg_time = total_time / iterations;
    std::cout << name << " average time: " << avg_time * 1000 << " ms (" << iterations << " runs)\n";
    return avg_time;
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

__global__ void vector_add_gpu_3d(const float* a, const float* b, float* c, int nx, int ny, int nz)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // the x idx in grid
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // the y idx in grid
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z; // the z idx in grid

    if (i < nx && j < ny && k < nz)
    {
        unsigned int idx = i + j * nx + k * nx * ny; // map to 1d position
        if (idx < nx * ny * nz)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}

bool verify_results(const std::vector<float>& cpu_result, const std::vector<float>& gpu_result)
{
    constexpr float epsilon = 1e-5f;
    for (size_t i = 0; i < cpu_result.size(); ++i)
    {
        if (std::abs(cpu_result[i] - gpu_result[i]) > epsilon)
        {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpu_result[i] << ", GPU = " << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
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
    verify_results(h_c_cpu, h_c_gpu);

    // 输出性能对比
    std::cout << "\nPerformance Summary:\n";
    std::cout << "CPU time:          " << cpu_time * 1000 << " ms\n";
    std::cout << "GPU total time:    " << gpu_total_time * 1000 << " ms (includes memory transfer)\n";
    std::cout << "GPU compute time:  " << gpu_compute_time * 1000 << " ms (kernel only)\n";
    std::cout << "Speedup (compute): " << cpu_time / gpu_compute_time << "x\n";
    std::cout << "Speedup (total):   " << cpu_time / gpu_total_time << "x\n";

    std::cout << "\nVector addition in 1d array completed successfully!\n";
}

void vector_add_3d()
{
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c_cpu(N), h_c_gpu(N);

    init_vector(h_a);
    init_vector(h_b);

    CudaMemory<float> d_a(N);
    CudaMemory<float> d_b(N);
    CudaMemory<float> d_c(N);

    cudaMemcpy(d_a.get(), h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.get(), h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int nx = 100, ny = 100, nz = 1000;
    dim3 block_size(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks(
        (nx + block_size.x - 1) / block_size.x,
        (ny + block_size.y - 1) / block_size.y,
        (nz + block_size.z - 1) / block_size.z
    );

    // warm up runs
    std::cout << "Performing warm-up runs..." << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        vector_add_cpu_1d(h_a, h_b, h_c_cpu);
        vector_add_gpu_3d<<<num_blocks, block_size>>>(d_a.get(), d_b.get(), d_c.get(), nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // benchmark cpu implementation (5 runs)
    double cpu_time = measure_time([&]()
    {
        vector_add_cpu_1d(h_a, h_b, h_c_cpu);
    }, "CPU vector addition", 5);

    // benchmark gpu implementation (100 runs)
    double gpu_3d_time = measure_time([&]()
    {
        cudaMemset(d_c.get(), 0, N * sizeof(float));
        vector_add_gpu_3d<<<num_blocks, block_size>>>(d_a.get(), d_b.get(), d_c.get(), nx, ny, nz);
        cudaDeviceSynchronize();
    }, "GPU 3D vector addition", 100);

    cudaMemcpy(h_c_gpu.data(), d_c.get(), N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = verify_results(h_c_gpu, h_c_cpu);

    std::cout << "3D Results are " << (correct ? "correct" : "incorrect") << "\n";

    // Performance summary
    std::cout << "\nPerformance Summary:\n";
    std::cout << "CPU average time:       " << cpu_time * 1000 << " ms\n";
    std::cout << "GPU 3D average time:    " << gpu_3d_time * 1000 << " ms\n";
    std::cout << "Speedup (CPU vs 3D):    " << cpu_time / gpu_3d_time << "x\n";
}






