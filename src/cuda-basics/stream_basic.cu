#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sstream>
#include <vector>
#include <random>

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, std::string_view func, std::string_view file, int line)
        : std::runtime_error(format(code, func, file, line)), code_(code) {}

    [[nodiscard]] cudaError_t code() const { return code_; }

private:
    cudaError_t code_;
    static std::string format(cudaError_t code, std::string_view func, std::string_view file, int line) {
        std::ostringstream oss;
        oss << "CUDA error [" << cudaGetErrorName(code) << "]: "
            << cudaGetErrorString(code) << "\n  in " << func
            << "\n  at " << file << ":" << line;
        return oss.str();
    }
};

inline void checkCuda(cudaError_t err, std::string_view func, std::string_view file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(err, func, file, line);
    }
}

__global__ void vectorAdd(const float* A, const float* B, float* C, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int streamBasicsDemo() {
    try {
        constexpr unsigned int n = (1u << 20);
        constexpr size_t sz = n * sizeof(float);

        std::vector<float> h_A(n), h_B(n), h_C(n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto& v : h_A) v = dis(gen);
        for (auto& v : h_B) v = dis(gen);

        float *d_A, *d_B, *d_C;
        checkCuda(cudaMalloc((void**)&d_A, sz), "cudaMalloc", __FILE__, __LINE__);
        checkCuda(cudaMalloc((void**)&d_B, sz), "cudaMalloc", __FILE__, __LINE__);
        checkCuda(cudaMalloc((void**)&d_C, sz), "cudaMalloc", __FILE__, __LINE__);

        cudaStream_t stream1, stream2;
        checkCuda(cudaStreamCreate(&stream1), "cudaStreamCreate", __FILE__, __LINE__);
        checkCuda(cudaStreamCreate(&stream2), "cudaStreamCreate", __FILE__, __LINE__);

        checkCuda(cudaMemcpyAsync(d_A, h_A.data(), sz, cudaMemcpyHostToDevice, stream1), "cudaMemcpyAsync", __FILE__, __LINE__);
        checkCuda(cudaMemcpyAsync(d_B, h_B.data(), sz, cudaMemcpyHostToDevice, stream2), "cudaMemcpyAsync", __FILE__, __LINE__);

        constexpr unsigned int threadsPerBlock = 256;
        constexpr unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, n);
        checkCuda(cudaMemcpyAsync(h_C.data(), d_C, sz, cudaMemcpyDeviceToHost, stream1), "cudaMemcpyAsync", __FILE__, __LINE__);

        checkCuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize", __FILE__, __LINE__);
        checkCuda(cudaStreamSynchronize(stream2), "cudaStreamSynchronize", __FILE__, __LINE__);

        for (size_t i = 0; i < n; ++i) {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
                throw std::runtime_error("Verification failed at index" + std::to_string(i));
            }
        }

        std::cout << "Test Passed!" << std::endl;

    } catch (const CudaError& e) {
        std::cerr << "CUDA Error:\n" << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}



