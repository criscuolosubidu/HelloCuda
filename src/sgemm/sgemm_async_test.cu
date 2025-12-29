#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


template<const int BM = 64, const int BN = 64, const int BK = 16, const int TM = 8, const int TN = 4, const int OFFSET =
        0>
__global__ void sgemm_t_8x4_sliced_k_f32x4_bcf_dbuf_kernel(float *a, float *b, float *c, int M, int N, int K) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // share memory
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    // temp arrays
    float r_load_a[8]; // 64 * 64 / (8 * 4) = 8 * 16 threads, 64 * 16 / 8 * 16 = 8 elements per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    // 128 threads
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 3;
    int load_b_smem_n = tid / 8;
    int load_b_smem_k = (tid & 7) << 3;
    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    // load data at the first time
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            // 加载两次，因为一次最多只能加载4个float
            reinterpret_cast<float4 *>(&s_b[0][load_b_smem_k][load_b_smem_n])[0] = reinterpret_cast<float4 *>(&b[
                load_b_gmem_addr + i])[0];
            reinterpret_cast<float4 *>(&r_load_a[i])[0] = reinterpret_cast<float4 *>(&a[load_a_gmem_addr + i])[0];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
    }
    __syncthreads();
    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        // 加载下一次需要使用的数据
        int smem_sel = bk - 1 & 1;
        int smem_sel_next = bk & 1;
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            // 加载两次，因为一次最多只能加载4个float
            reinterpret_cast<float4 *>(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n])[0] = reinterpret_cast<float4
                *>(&b[load_b_gmem_addr + i])[0];
            reinterpret_cast<float4 *>(&r_load_a[i])[0] = reinterpret_cast<float4 *>(&a[load_a_gmem_addr + i])[0];
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 根据tid找到对应的在C矩阵的位置，然后去加载对应的数据
            reinterpret_cast<float4 *>(&r_comp_a[0])[0] = reinterpret_cast<float4 *>(&s_a[smem_sel][k][
                threadIdx.y * TM])[0];
            reinterpret_cast<float4 *>(&r_comp_a[4])[0] = reinterpret_cast<float4 *>(&s_a[smem_sel][k][
                threadIdx.y * TM + 4])[0];
            reinterpret_cast<float4 *>(&r_comp_b[0])[0] = reinterpret_cast<float4 *>(&s_b[smem_sel][k][
                threadIdx.x * TN])[0];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < BK; ++k) {
        reinterpret_cast<float4 *>(&r_comp_a[0])[0] = reinterpret_cast<float4 *>(&s_a[1][k][threadIdx.y * TM])[0];
        reinterpret_cast<float4 *>(&r_comp_a[4])[0] = reinterpret_cast<float4 *>(&s_a[1][k][threadIdx.y * TM + 4])[0];
        reinterpret_cast<float4 *>(&r_comp_b[0])[0] = reinterpret_cast<float4 *>(&s_b[1][k][threadIdx.x * TN])[0];
        for (int m = 0; m < TM; ++m) {
            for (int n = 0; n < TN; ++n) {
                r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + m;
            int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            reinterpret_cast<float4 *>(&r_c[m][n])[0] = reinterpret_cast<float4 *>(&c[store_c_gmem_addr])[0];
        }
    }
}


// PTX COMMAND: cp.async.<cache_policy>.shared.global.L2::128B [dst_shared], [src_global], bytes
// ca: cache all level, cg: cache global
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile(\
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),\
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile(\
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),\
        "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

template<const int BM = 64, const int BN = 64, const int BK = 16, const int TM = 8, const int TN = 4, const int OFFSET =
        0>
__global__ void sgemm_t_8x4_sliced_k_f32x4_bcf_dbuf_async_kernel(float *a, float *b, float *c, int M, int N, int K) {
    // async load b data
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // the thread idx in the block
    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];
    // float arrays
    float r_load_a[8]; // 64 * 64 / (8 * 4) = 8 * 16 threads, 64 * 16 / 8 * 16 = 8 elements per thread
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    // 128 threads,
    // [0,7][8,15] 1 row 2 threads
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 3;
    // 64/8=8, 1 row 8 threads
    int load_b_smem_k = tid / 8;
    int load_b_smem_n = (tid & 7) << 3;
    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    // load the first batch data to calculate
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[0][load_b_smem_k][load_b_smem_n]);
        // get shared memory 32-bit address, only need it when it is target address
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16);
            // 4 floats, 1 thread for 8 floats, load 2 times
        }
        CP_ASYNC_COMMIT_GROUP(); // auto increment to 0
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            reinterpret_cast<float4 *>(&r_load_a[i])[0] = reinterpret_cast<float4 *>(&a[load_a_gmem_addr + i])[0];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        CP_ASYNC_WAIT_GROUP(0); // wait all the group number >= N to be done
    }
    __syncthreads();

    for (int k = 1; k < (K + BK - 1) / BK; ++k) {
        int smem_sel = k - 1 & 1;
        int smem_sel_next = k & 1;
        int load_a_gmem_k = k * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = k * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            CP_ASYNC_CA(load_b_smem_ptr + i * 4, &b[load_b_gmem_addr + i], 16); // 4 floats
        }
        CP_ASYNC_COMMIT_GROUP();
#pragma unroll
        for (int i = 0; i < 8; i += 4) {
            reinterpret_cast<float4 *>(&r_load_a[i])[0] = reinterpret_cast<float4 *>(&a[load_a_gmem_addr + i])[0];
        }
#pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            reinterpret_cast<float4 *>(&r_comp_a[0])[0] = reinterpret_cast<float4 *>(&s_a[smem_sel][tk][
                threadIdx.y * TM])[0];
            reinterpret_cast<float4 *>(&r_comp_a[4])[0] = reinterpret_cast<float4 *>(&s_a[smem_sel][tk][
                threadIdx.y * TM + 4])[0];
            reinterpret_cast<float4 *>(&r_comp_b[0])[0] = reinterpret_cast<float4 *>(&s_b[smem_sel][tk][
                threadIdx.x * TN])[0];
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) s_a[smem_sel_next][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
#pragma unroll
    for (int tk = 0; tk < BK; ++tk) {
        reinterpret_cast<float4 *>(&r_comp_a[0])[0] = reinterpret_cast<float4 *>(&s_a[1][tk][threadIdx.y * TM])[0];
        reinterpret_cast<float4 *>(&r_comp_a[4])[0] = reinterpret_cast<float4 *>(&s_a[1][tk][threadIdx.y * TM + 4])[0];
        reinterpret_cast<float4 *>(&r_comp_b[0])[0] = reinterpret_cast<float4 *>(&s_b[1][tk][threadIdx.x * TN])[0];
#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
            int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + m;
            int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            reinterpret_cast<float4 *>(&c[store_c_gmem_addr])[0] = reinterpret_cast<float4 *>(&r_c[m][n])[0];
        }
    }
}

// 错误检查宏
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) \
{ \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS API failed at line %d with error: %d\n", \
               __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

void randomize_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = (float) (rand() % 100) / 100.0f;
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double max_diff = 0.0;
    double max_rel_diff = 0.0; // 记录最大相对误差

    for (int i = 0; i < N; i++) {
        float v1 = mat1[i];
        float v2 = mat2[i];
        float diff = std::abs(v1 - v2);

        if (diff > max_diff) max_diff = diff;

        // 计算相对误差: |v1 - v2| / max(|v1|, |v2|, 1e-5)
        // 加上 1e-5 是为了防止除以 0
        float rel_err = diff / (std::max(std::abs(v1), std::abs(v2)) + 1e-5f);

        if (rel_err > max_rel_diff) max_rel_diff = rel_err;

        // 阈值设为 1e-4 (即 0.01% 的误差是允许的)
        if (rel_err > 1e-4) {
            printf("Verification failed at index %d: %f vs %f (diff: %f, rel_err: %f)\n",
                   i, v1, v2, diff, rel_err);
            return false;
        }
    }
    printf("Verification passed! Max Abs Diff: %f, Max Rel Diff: %f\n", max_diff, max_rel_diff);
    return true;
}


int main() {
    // 1. 设置矩阵大小 (必须是 block size 的倍数，因为你的 kernel 没有边界检查)
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 2. 分配主机内存
    float *h_A = (float *) malloc(bytes_A);
    float *h_B = (float *) malloc(bytes_B);
    float *h_C = (float *) malloc(bytes_C);
    float *h_C_ref = (float *) malloc(bytes_C);

    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_C_ref, bytes_C)); // 用于 cuBLAS 结果

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // 4. 配置 Kernel 参数
    // BM=64, BN=64. Grid = (N/BN, M/BM)
    // Threads = 128 (你的代码逻辑基于 128 线程)
    dim3 blockDim(16, 8);
    dim3 gridDim(N / 64, M / 64);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ==========================================
    // 5. 正确性验证 (使用 cuBLAS)
    // ==========================================
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    // 注意：cuBLAS 是列优先 (Column Major)。
    // 我们想要计算行优先的 C = A * B。
    // 在显存中，行优先的 A 看起来像是列优先的 A^T。
    // 因此调用 cuBLAS 计算 C^T = B^T * A^T，即对应 cublasSgemm(..., d_B, d_A, ...)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C_ref, N));

    // 运行你的 Async Kernel 一次进行验证
    sgemm_t_8x4_sliced_k_f32x4_bcf_dbuf_async_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    printf("Verifying Async Kernel correctness...\n");
    if (!verify_matrix(h_C, h_C_ref, M * N)) {
        printf("Verification Failed.\n");
        // exit(1); // 可选：如果验证失败是否继续 Benchmark
    }

    // ==========================================
    // 6. 性能测试 (Benchmark)
    // ==========================================
    int num_runs = 20;
    int num_warmup = 5;

    // Warm up
    for (int i = 0; i < num_warmup; i++) {
        sgemm_t_8x4_sliced_k_f32x4_bcf_dbuf_async_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Record Timing
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; i++) {
        sgemm_t_8x4_sliced_k_f32x4_bcf_dbuf_async_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_latency = milliseconds / num_runs;
    // 2 * M * N * K ops
    float tflops = (2.0f * (float) M * (float) N * (float) K) / (avg_latency * 1e-3f) / 1e12f;

    printf("\nPerformance (Async Kernel):\n");
    printf("Avg Latency: %.4f ms\n", avg_latency);
    printf("Throughput : %.4f TFLOPS\n", tflops);

    // ==========================================
    // 清理
    // ==========================================
    CHECK_CUBLAS(cublasDestroy(handle));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);

    return 0;
}
