// sgemm_bench.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)               \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// ===================== 函数原型（你要在各自 .cu 里实现） =====================

// 纯 FP32 CUDA 核心
using GemmFunc =
    void (*)(float* A, float* B, float* C, int M, int N, int K);

// 带 stages、swizzle 的 WMMA 核心
using GemmStagesFunc = void (*)(float* A, float* B, float* C,
                                int M, int N, int K,
                                int stages, bool swizzle, int swizzle_stride);

// 这些名字对应你 Python 里的调用：
//   lib.sgemm_t_8x8_sliced_k_f32x4
//   lib.sgemm_t_8x8_sliced_k_f32x4_bcf
//   lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf
//   lib.sgemm_cublas
//   lib.sgemm_cublas_tf32
//   lib.sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages
//   lib.sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem
//
// 你可以在各自的 .cu 里实现这些 *_host 函数，
// 负责设置 grid/block，然后 launch kernel 或调用 cuBLAS。
// 这里先声明，后面 main() 里直接用。
extern void sgemm_t_8x8_sliced_k_f32x4_host(float* A, float* B, float* C,
                                            int M, int N, int K);

extern void sgemm_t_8x8_sliced_k_f32x4_bcf_host(float* A, float* B, float* C,
                                                int M, int N, int K);

extern void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_host(float* A, float* B,
                                                     float* C,
                                                     int M, int N, int K);

extern void sgemm_cublas_host(float* A, float* B, float* C,
                              int M, int N, int K);

extern void sgemm_cublas_tf32_host(float* A, float* B, float* C,
                                   int M, int N, int K);

extern void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_host(
    float* A, float* B, float* C,
    int M, int N, int K,
    int stages, bool swizzle, int swizzle_stride);

extern void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_host(
    float* A, float* B, float* C,
    int M, int N, int K,
    int stages, bool swizzle, int swizzle_stride);

// ===================== benchmark 公共逻辑 =====================

static double g_max_tflops = -1.0;

// 打印前几个 C 的元素，方便和 Python 结果对比
static void print_head_C(float* dC) {
  float h[4] = {0};
  CHECK_CUDA(cudaMemcpy(h, dC, 4 * sizeof(float), cudaMemcpyDeviceToHost));
  std::cout << "[" << std::setw(10) << h[0]
            << ", " << std::setw(10) << h[1] << "]";
}

// 不带 stages 的版本，对应：
//   run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4, ...)
//   run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf, ...)
//   run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf, ...)
//   run_benchmark(lib.sgemm_cublas, ...)
//   run_benchmark(lib.sgemm_cublas_tf32, ...)
void run_benchmark(GemmFunc func,
                   float* dA, float* dB, float* dC,
                   int M, int N, int K,
                   const std::string& tag,
                   int warmup = 2,
                   int iters = 20) {
  int eff_iters = iters;
  if (M > 1024 || K >= 1024 || N > 1024) {
    eff_iters = 10;
  }

  // 清零输出
  size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));

  // 预热
  for (int i = 0; i < warmup; ++i) {
    func(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // 计时
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < eff_iters; ++i) {
    func(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= eff_iters;  // mean time

  double flops = 2.0 * static_cast<double>(M) * N * K;
  double tflops = flops * 1e-9 / ms;  // 和你 python 一样：2MNK * 1e-9 / ms

  // 计算 TFLOPS 提升
  double improve = 0.0;
  bool is_new_max = false;
  if (tflops > g_max_tflops) {
    if (g_max_tflops > 0) {
      improve = (tflops - g_max_tflops) / g_max_tflops * 100.0;
    }
    g_max_tflops = tflops;
    is_new_max = true;
  }

  std::cout << std::right << std::setw(35)
            << ("out_" + tag) << ": ";
  print_head_C(dC);

  std::cout << ", time:" << std::setw(8) << std::fixed << std::setprecision(4)
            << ms << "ms, swizzle: " << std::setw(4) << "NOOP"
            << ", TFLOPS: " << std::setw(8) << std::setprecision(2)
            << tflops;
  if (is_new_max) {
    std::cout << "(+" << std::fixed << std::setprecision(2)
              << improve << "%)";
  }
  std::cout << std::endl;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}

// 带 stages / swizzle 的版本，对应 python 里的 WMMA 那几组 run_benchmark
void run_benchmark_stages(GemmStagesFunc func,
                          float* dA, float* dB, float* dC,
                          int M, int N, int K,
                          const std::string& tag,
                          int stages,
                          bool swizzle,
                          int warmup = 2,
                          int iters = 20) {
  int eff_iters = iters;
  if (M > 1024 || K >= 1024 || N > 1024) {
    eff_iters = 10;
  }

  int swizzle_stride = 1;
  if (swizzle) {
    // 与 python 同样的规则：
    // swizzle_stride = int((int(N / 8) // 256) * 256)
    // if swizzle_stride < 256 -> NOOP
    swizzle_stride = static_cast<int>((N / 8) / 256) * 256;
    if (swizzle_stride < 256) {
      swizzle_stride = 1;
      swizzle = false;
    }
  }

  size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));

  for (int i = 0; i < warmup; ++i) {
    func(dA, dB, dC, M, N, K, stages, swizzle, swizzle_stride);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < eff_iters; ++i) {
    func(dA, dB, dC, M, N, K, stages, swizzle, swizzle_stride);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= eff_iters;

  double flops = 2.0 * static_cast<double>(M) * N * K;
  double tflops = flops * 1e-9 / ms;

  double improve = 0.0;
  bool is_new_max = false;
  if (tflops > g_max_tflops) {
    if (g_max_tflops > 0) {
      improve = (tflops - g_max_tflops) / g_max_tflops * 100.0;
    }
    g_max_tflops = tflops;
    is_new_max = true;
  }

  std::cout << std::right << std::setw(35)
            << ("out_" + tag) << ": ";
  print_head_C(dC);

  std::cout << ", time:" << std::setw(8) << std::fixed << std::setprecision(4)
            << ms << "ms, ";

  if (swizzle_stride == 1) {
    std::cout << "swizzle: " << std::setw(4) << "NOOP";
  } else {
    std::cout << "swizzle: " << std::setw(4) << swizzle_stride;
  }

  std::cout << ", TFLOPS: " << std::setw(8) << std::setprecision(2)
            << tflops;
  if (is_new_max) {
    std::cout << "(+" << std::fixed << std::setprecision(2)
              << improve << "%)";
  }
  std::cout << std::endl;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}

// ===================== main：照着 python 的循环写 =====================

int main() {
  // 和 python 一致的尺寸
  const int Ms[] = {4096, 8192, 16384};
  const int Ns[] = {4096, 8192, 16384};
  const int Ks[] = {2048, 4096, 8192};

  const int MAX_M = 16384;
  const int MAX_N = 16384;
  const int MAX_K = 8192;

  // Host buffer（其实只用来初始化一次）
  std::vector<float> hA(static_cast<size_t>(MAX_M) * MAX_K);
  std::vector<float> hB(static_cast<size_t>(MAX_K) * MAX_N);
  std::vector<float> hC(static_cast<size_t>(MAX_M) * MAX_N);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : hA) v = dist(rng);
  for (auto& v : hB) v = dist(rng);
  for (auto& v : hC) v = dist(rng);

  // Device buffer（一次最大值，后面都共用）
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(),
                        hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(),
                        hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(),
                        hC.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());

  for (int Mi : Ms) {
    for (int Ni : Ns) {
      for (int Ki : Ks) {
        g_max_tflops = -1.0;

        std::cout << std::string(130, '-') << "\n";
        std::cout << std::string(55, ' ')
                  << "M=" << Mi << ", N=" << Ni << ", K=" << Ki << "\n";

        int M = Mi, N = Ni, K = Ki;
        // 注意：这里直接用 dA/dB/dC 的前 M*K / K*N / M*N 区域，
        // kernel 按 row-major 访问即可，不需要偏移。

        // 纯 CUDA Cores FP32
        run_benchmark(sgemm_t_8x8_sliced_k_f32x4_host,
                      dA, dB, dC, M, N, K,
                      "f32x4(t8x8sk)");

        run_benchmark(sgemm_t_8x8_sliced_k_f32x4_bcf_host,
                      dA, dB, dC, M, N, K,
                      "f32x4(t8x8bcf)");

        run_benchmark(sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_host,
                      dA, dB, dC, M, N, K,
                      "f32x4(t8x8dbuf)");

        run_benchmark(sgemm_cublas_host,
                      dA, dB, dC, M, N, K,
                      "f32(cublas)");

        // 这里没有 torch.matmul，对标的话你可以留着或者换成别的参考实现

        std::cout << std::string(62, '-') << "WMMA"
                  << std::string(64, '-') << "\n";

        // WMMA + stages + swizzle/dsmem
        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+warp2x4+stage3)",
            3, false);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+warp2x4+stage2)",
            2, false);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+...+stage3+dsmem)",
            3, false);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+...+stage2+dsmem)",
            2, false);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+...+stage3+swizzle)",
            3, true);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_host,
            dA, dB, dC, M, N, K,
            "tf32(mma2x4+...+stage2+swizzle)",
            2, true);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_host,
            dA, dB, dC, M, N, K,
            "tf32(...+stage3+dsmem+swizzle)",
            3, true);

        run_benchmark_stages(
            sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_host,
            dA, dB, dC, M, N, K,
            "tf32(...+stage2+dsmem+swizzle)",
            2, true);

        run_benchmark(sgemm_cublas_tf32_host,
                      dA, dB, dC, M, N, K,
                      "tf32(cublas+tf32)");

        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << std::string(130, '-') << "\n";
      }
    }
  }

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));

  return 0;
}
