#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <iostream>

#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask); // __shfl_xor_sync is a warp-level operation, 硬件自动知道线程所处的warp
    }
    return val;
}

__global__ void test_warp_reduce_sum_f32_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        float warp_sum = warp_reduce_sum_f32(val);
        int lane_id = idx / WARP_SIZE;
        if (lane_id == 0) {
            output[idx / WARP_SIZE] = warp_sum;
        }
    }
}

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void test_warp_reduce_sum_f32() {
    const int N = 256;
    std::vector<float> h_input(N);
    std::fill(h_input.begin(), h_input.end(), 1.0f);

    float *d_input = nullptr;
    float *d_output = nullptr;
    size_t input_bytes = N * sizeof(float);

    int num_warps = (N + WARP_SIZE - 1) / WARP_SIZE;
    size_t output_bytes = num_warps * sizeof(float);
    std::vector<float> h_output(num_warps);

    checkCudaErrors(cudaMalloc(&d_input, input_bytes));
    checkCudaErrors(cudaMalloc(&d_output, output_bytes));
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    int block_size = 128;
    int grid_size = (N + block_size - 1) / block_size;

    std::cout << "Launching Kernel..." << std::endl;
    std::cout << "Grid Size: " << grid_size << ", Block Size: " << block_size << std::endl;
    std::cout << "Total Warps: " << num_warps << std::endl << std::endl;

    test_warp_reduce_sum_f32_kernel<<<grid_size, block_size>>>(d_input, d_output, N);

    checkCudaErrors(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < num_warps; ++i) {
        float expected_val = 0.0f;
        int start_idx = i * WARP_SIZE;
        int end_idx = std::min((i + 1) * WARP_SIZE, N);
        for (int j = start_idx; j < end_idx; ++j) {
            expected_val += h_input[j];
        }
        std::cout << "Warp " << i << ": "
                << "GPU Sum = " << h_output[i]
                << ", Expected CPU Sum = " << expected_val
                << ((h_output[i] == expected_val) ? " -> [OK]" : " -> [FAIL]")
                << std::endl;
    }
    cudaFree(d_input);
    cudaFree(d_output);
}
