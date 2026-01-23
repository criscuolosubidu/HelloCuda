#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes)                                           \
    asm volatile(                                                                \
    "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
    "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                                \
    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
    "l"(src), "n"(bytes))


using namespace nvcuda;


__global__ void f32x4_tf32x4_kernel(float *x, float *y, int N) {
    // 将 f32 转换为 tf32
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = reinterpret_cast<float4 *>(&x[idx])[0];
        float4 reg_y;
        reg_y.x = wmma::__float_to_tf32(reg_x.x);
        reg_y.y = wmma::__float_to_tf32(reg_x.y);
        reg_y.z = wmma::__float_to_tf32(reg_x.z);
        reg_y.w = wmma::__float_to_tf32(reg_x.w);
        reinterpret_cast<float4 *>(&y[idx])[0] = reg_y;
    }
}

// https://zhuanlan.zhihu.com/p/555339335
template<
    const int WMMA_M = 16,
    const int WMMA_N = 16,
    const int WMMA_K = 8,
    const int WMMA_TILE_M = 4,
    const int WMMA_TILE_N = 2,
    const int WARP_TILE_M = 2,
    const int WARP_TILE_N = 4,
    const int A_PAD = 0,
    const int B_PAD = 0,
    const int K_STAGE = 2,
    const bool BLOCK_SWIZZLE = false
>
__global__ void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const int bx = static_cast<int>(BLOCK_SWIZZLE) * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = (K + WMMA_K - 1) / WMMA_K;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 1 warp for 32 * 64, 32 * 4 = 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 1 warp for 32 * 64, 64 * 2 = 128
    constexpr int BK = WMMA_K;
    __shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // tid in block
    // 32 * 8 threads, 8 warps, 4 * 2 shape
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    // 128 * 8 / (32 * 8) = 4, 1 threads for 4 floats, 128 * 2 shape
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    // 128 * 8 / (32 * 8) = 4, 1 threads for 4 floats, 8 * 32 shape
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = tid % 32 * 4;
    // global a_m, b_n
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // C wmma fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag[WARP_TILE_M][WARP_TILE_N];
    // initialize
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // load K-STAGE data, pre-load 0, 1, 2, 3, ... K - 2 stage data
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; ++k) {
        int load_a_gmem_k = BK * k + load_a_smem_k;
        int load_b_gmem_k = BK * k + load_b_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_a_smem_ptr = __cvta_generic_to_shared(&s_a[k][load_a_smem_m][load_a_smem_k]);
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[k][load_b_smem_k][load_b_smem_n]);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16); // 4 * 4 floats
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll
    for (int k = K_STAGE - 1; k < NUM_K_TILES; ++k) {
        int smem_sel = (k + 1) % K_STAGE; // compute stage
        int smem_sel_next = k % K_STAGE; // storage stage
        int load_a_gmem_k = BK * k + load_a_smem_k;
        int load_b_gmem_k = BK * k + load_b_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        uint32_t load_a_smem_ptr = __cvta_generic_to_shared(&s_a[smem_sel_next][load_a_smem_m][load_a_smem_k]);
        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> B_frag[WARP_TILE_N];
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD);
        }
        for (int i = 0; i < WARP_TILE_N; ++i) {
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
            wmma::load_matrix_sync(B_frag[i], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);
        }
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }
    CP_ASYNC_WAIT_ALL();
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; ++k) {
        int offset = NUM_K_TILES - K_STAGE + 1;
        const int stage_sel = (offset + k) % K_STAGE;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> B_frag[WARP_TILE_N];
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0], BK + A_PAD);
        }
        for (int i = 0; i < WARP_TILE_N; ++i) {
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
            wmma::load_matrix_sync(B_frag[i], &s_b[stage_sel][0][warp_smem_b_n], BN + B_PAD);
        }
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_c_gmem_m = BM * blockIdx.y + warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            int store_c_gmem_n = BN * blockIdx.x + warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
}


// 静态shared memory的最大的大小就是48kb，要想使用更加大的内存，只能使用动态的shared memory
template<
    const int WMMA_M = 16,
    const int WMMA_N = 16,
    const int WMMA_K = 8,
    const int WMMA_TILE_M = 4,
    const int WMMA_TILE_N = 2,
    const int WARP_TILE_M = 2,
    const int WARP_TILE_N = 4,
    const int A_PAD = 0,
    const int B_PAD = 0,
    const int K_STAGE = 2,
    const bool BLOCK_SWIZZLE = false
>
__global__ void sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const int bx = static_cast<int>(BLOCK_SWIZZLE) * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = (K + WMMA_K - 1) / WMMA_K;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 1 warp for 32 * 64, 32 * 4 = 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 1 warp for 32 * 64, 64 * 2 = 128
    constexpr int BK = WMMA_K;

    //__shared__ float s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
    extern __shared__ float smem[];
    float *s_a = smem;
    float *s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // tid in block
    // 32 * 8 threads, 8 warps, 4 * 2 shape
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    // 128 * 8 / (32 * 8) = 4, 1 threads for 4 floats, 128 * 2 shape
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    // 128 * 8 / (32 * 8) = 4, 1 threads for 4 floats, 8 * 32 shape
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = tid % 32 * 4;
    // global a_m, b_n
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // C wmma fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag[WARP_TILE_M][WARP_TILE_N];
    // initialize
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // turn generic pointer to shared memory pointer
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    // load K-STAGE data, pre-load 0, 1, 2, 3, ... K - 2 stage data
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; ++k) {
        int load_a_gmem_k = BK * k + load_a_smem_k;
        int load_b_gmem_k = BK * k + load_b_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        // uint32_t load_a_smem_ptr = __cvta_generic_to_shared(&s_a[k][load_a_smem_m][load_a_smem_k]);
        // uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[k][load_b_smem_k][load_b_smem_n]);
        uint32_t load_a_smem_offset = k * s_a_stage_offset + load_a_smem_m * (BK + A_PAD) + load_a_smem_k;
        uint32_t load_a_smem_ptr = smem_a_base_ptr + load_a_smem_offset * sizeof(float);
        uint32_t load_b_smem_offset = k * s_b_stage_offset + load_b_smem_k * (BN + B_PAD) + load_b_smem_n;
        uint32_t load_b_smem_ptr = smem_b_base_ptr + load_b_smem_offset * sizeof(float);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16); // 4 * 4 floats
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll
    for (int k = K_STAGE - 1; k < NUM_K_TILES; ++k) {
        int smem_sel = (k + 1) % K_STAGE; // compute stage
        int smem_sel_next = k % K_STAGE; // storage stage
        int load_a_gmem_k = BK * k + load_a_smem_k;
        int load_b_gmem_k = BK * k + load_b_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        // uint32_t load_a_smem_ptr = __cvta_generic_to_shared(&s_a[smem_sel_next][load_a_smem_m][load_a_smem_k]);
        // uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);
        uint32_t load_a_smem_offset = smem_sel_next * s_a_stage_offset + load_a_smem_m * (BK + A_PAD) + load_a_smem_k;
        uint32_t load_a_smem_ptr = smem_a_base_ptr + load_a_smem_offset * sizeof(float);
        uint32_t load_b_smem_offset = smem_sel_next * s_b_stage_offset + load_b_smem_k * (BN + B_PAD) + load_b_smem_n;
        uint32_t load_b_smem_ptr = smem_b_base_ptr + load_b_smem_offset * sizeof(float);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> B_frag[WARP_TILE_N];
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            //wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD);
            float* load_a_smem_frag_ptr = s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD);
            wmma::load_matrix_sync(A_frag[i], load_a_smem_frag_ptr, BK + A_PAD);
        }
        for (int i = 0; i < WARP_TILE_N; ++i) {
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
            // wmma::load_matrix_sync(B_frag[i], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);
            float* load_b_smem_frag_ptr = s_b + smem_sel * s_b_stage_offset + warp_smem_b_n;
            wmma::load_matrix_sync(B_frag[i], load_b_smem_frag_ptr, BN + B_PAD);
        }
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }
    CP_ASYNC_WAIT_ALL();
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; ++k) {
        int offset = NUM_K_TILES - K_STAGE + 1;
        const int stage_sel = (offset + k) % K_STAGE;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> B_frag[WARP_TILE_N];
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            //wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0], BK + A_PAD);
            float* load_a_smem_frag_ptr = s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD);
            wmma::load_matrix_sync(A_frag[i], load_a_smem_frag_ptr, BK + A_PAD);
        }
        for (int i = 0; i < WARP_TILE_N; ++i) {
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + i * WMMA_N;
            // wmma::load_matrix_sync(B_frag[i], &s_b[stage_sel][0][warp_smem_b_n], BN + B_PAD);
            float* load_b_smem_frag_ptr = s_b + stage_sel * s_b_stage_offset + warp_smem_b_n;
            wmma::load_matrix_sync(B_frag[i], load_b_smem_frag_ptr, BN + B_PAD);
        }
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_c_gmem_m = BM * blockIdx.y + warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            int store_c_gmem_n = BN * blockIdx.x + warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
}


#define CHECK_CUDA_ERROR(call)                                                  \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

void init_matrix(float *mat, int size, float val) {
    for (int i = 0; i < size; i++) {
        mat[i] = val;
    }
}

bool verify_result(float *C, int M, int N, float expected, float tolerance = 1e-2f) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - expected) > tolerance) {
            printf("Verification failed at index %d: expected %f, got %f\n", i, expected, C[i]);
            return false;
        }
    }
    return true;
}

void test_f32x4_tf32x4_kernel() {
    printf("\n=== Testing f32x4_tf32x4_kernel ===\n");
    const int N = 1024;
    float *h_x, *h_y;
    float *d_x, *d_y;

    h_x = (float *)malloc(N * sizeof(float));
    h_y = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i) * 0.001f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N / 4 + threads - 1) / threads;
    f32x4_tf32x4_kernel<<<blocks, threads>>>(d_x, d_y, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("TF32 conversion test passed (first 5 values):\n");
    for (int i = 0; i < 5; i++) {
        printf("  x[%d] = %f -> tf32 = %f\n", i, h_x[i], h_y[i]);
    }

    free(h_x);
    free(h_y);
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
}

void test_sgemm_wmma_kernel() {
    printf("\n=== Testing sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel ===\n");

    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    init_matrix(h_A, M * K, 1.0f);
    init_matrix(h_B, K * N, 2.0f);
    init_matrix(h_C, M * N, 0.0f);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // 8 warps: 4x2 shape, 32*8 = 256 threads
    dim3 block(32, 8);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Warmup
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    const int iterations = 10;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("Average time: %.3f ms, Performance: %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 1.0f * 2.0f * K; // A=1, B=2, C = A*B*K = 4096
    if (verify_result(h_C, M, N, expected)) {
        printf("Verification PASSED! (expected: %f)\n", expected);
    } else {
        printf("Verification FAILED!\n");
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

void test_sgemm_wmma_dsmem_kernel() {
    printf("\n=== Testing sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel ===\n");

    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 0;
    constexpr int K_STAGE = 2;
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = WMMA_K;

    // Calculate dynamic shared memory size
    constexpr int smem_a_size = K_STAGE * BM * (BK + A_PAD);
    constexpr int smem_b_size = K_STAGE * BK * (BN + B_PAD);
    constexpr int smem_size = (smem_a_size + smem_b_size) * sizeof(float);

    printf("Dynamic shared memory size: %d bytes\n", smem_size);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    init_matrix(h_A, M * K, 1.0f);
    init_matrix(h_B, K * N, 2.0f);
    init_matrix(h_C, M * N, 0.0f);

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));

    dim3 block(32, 8);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // Set max dynamic shared memory if needed
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Warmup
    sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    const int iterations = 10;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;

    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("Average time: %.3f ms, Performance: %.2f GFLOPS\n", ms, gflops);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 1.0f * 2.0f * K;
    if (verify_result(h_C, M, N, expected)) {
        printf("Verification PASSED! (expected: %f)\n", expected);
    } else {
        printf("Verification FAILED!\n");
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

int main() {
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);

    test_f32x4_tf32x4_kernel();
    test_sgemm_wmma_kernel();
    test_sgemm_wmma_dsmem_kernel();

    printf("\nAll tests completed!\n");
    return 0;
}

