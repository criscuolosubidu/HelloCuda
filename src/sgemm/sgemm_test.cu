#include <cuda_runtime.h>
#include <cstdio>
#include <thread>
#include <vector>
#include <cstdlib>

#define WARP_SIZE 32

// 错误检查宏
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        std::exit(EXIT_FAILURE);
    }
}

// ==========================================
// Kernel 1: Naive 实现
// ==========================================

// SGEMM NAIVE: element per thread, all row major
__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float p_sum = 0.0f;
        // c[m, n] = SUM(a[m, k] * b[k, n])
        for (int k = 0; k < K; ++k) p_sum += a[m * K + k] * b[k * N + n];
        c[m * N + n] = p_sum;
    }
}

// ==========================================
// Kernel 2: Tiled 实现 (Shared Memory)
// ==========================================

// SGEMM: block tile + K tile, with smem
// Block Tile(BM, BN) + K Tile(BK = 32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute c = a * b, all row major
template<const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M, int N, int K) {
    __shared__ float s_a[BM][BK], s_b[BK][BN];
    // 这里就是直接映射，因为都是方阵
    int load_smem_a_m = threadIdx.y;
    int load_smem_a_k = threadIdx.x;
    int load_smem_b_k = threadIdx.y;
    int load_smem_b_n = threadIdx.x;
    int load_gmem_a_m = BM * blockIdx.y + load_smem_a_m;;
    int load_gmem_b_n = BN * blockIdx.x + load_smem_b_n;;
    float sum = 0.0f;
    // 枚举其中K维度的分块
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_m * K + load_gmem_a_k];
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_k * N + load_gmem_b_n];
        __syncthreads(); // 等待块内所有的线程都加载完毕
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads(); // 等待所有的线程计算完毕
    }
    int store_gmem_a_m = load_gmem_a_m;
    int store_gmem_b_n = load_gmem_b_n;
    c[store_gmem_a_m * N + store_gmem_b_n] = sum;
}


// SGEMM: block tile + thread tile + K tile + vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8, BM/TM=16, BN/TN=16
// dim3 blockDim(BN/TN, BM/TM)
// dim3 gridDim((N+BN-1)/BN, (M+BM-1)/BM)
template<const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c, int M, int N, int K) {
    // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
    // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
    // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
    //                  每次计算TM*TN个元素各自的部分乘累加
    // [4]   Vectorize: 减少load和store指令，使用float4
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // thread ID in block
    // RTX 4090也就是实验的显卡，单个SM差不多100KB，但是静态的smem最大还是48KB，超过这个值需要申请动态的smem
    __shared__ float s_a[BM][BK], s_b[BK][BN]; // 128*2*8*4=8KB
    // 首先计算在块内的具体位置
    // 总共划分了 256 个线程，那么加载 s_a 时候，也就是一个线程负责加载4个float，刚好可以向量化，而且一行就是8个float，恰好对应两个thread
    int load_smem_a_m = tid / 2, load_smem_a_k = tid % 2 == 0 ? 0 : 4;
    // 同样的，对于 s_b 而言，也是每个线程负责4个float，但现在是一行128个，恰好对应32线程，总共8行
    int load_smem_b_k = tid / 32, load_smem_b_n = tid % 32 * 4;
    // 计算对应的全局的位置
    int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m; // global row in a
    int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n; // global col in b

    float r_c[TM][TN] = {0.0}; // 8x8
    // 开始加载数据
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        reinterpret_cast<float4*>(&s_a[load_smem_a_m][load_smem_a_k])[0] = reinterpret_cast<float4*>(&a[load_gmem_a_addr])[0];
        reinterpret_cast<float4*>(&s_b[load_smem_b_k][load_smem_b_n])[0] = reinterpret_cast<float4*>(&b[load_gmem_b_addr])[0];
        __syncthreads(); // 等待所有的线程把数据加载完毕才能进行下一步矩阵乘法的计算，因为依赖于 s_a, s_b
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    int comp_smem_a_m = threadIdx.y * TM + m;
                    int comp_smem_b_n = threadIdx.x * TN + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int store_gmem_c_m = blockIdx.y * BM + threadIdx.y * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = blockIdx.x * BN + threadIdx.x * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            reinterpret_cast<float4*>(&c[store_gmem_c_addr])[0] = reinterpret_cast<float4*>(&r_c[m][n])[0];
        }
    }
}


template<const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    /* 将 s_a 进行转置，有两个很大的好处：
     * 1 能够遍历行而不是遍历列，遍历列的时候访问的stride恰好是 8个 float，每隔4行bank就会重复产生，那么对于128行而言，就有32个地方是重复同一个bank的，计算的时候会读取列，这就导致了bank conflict
     * 2 能够支持向量化读取了，因为要读取的数组都是连续的
     */
    __shared__ float s_a[BK][BM + OFFSET], s_b[BK][BN + OFFSET]; // OFFSET其实就是修改了stride来错开这些bank

    // 使用TM / 2和TN / 2可以使用一条指令加载完4个float
    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};
    /*
     * 第一步依然是映射到对应的位置然后加载数据，和前面的过程其实没有区别
     * 首先计算对于当前的block而言的位置
    */
    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = (tid & 31) << 2;
    // 全局的位置
    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        reinterpret_cast<float4*>(&r_load_a[0])[0] = reinterpret_cast<float4*>(&a[load_a_gmem_addr])[0];
        reinterpret_cast<float4*>(&r_load_b[0])[0] = reinterpret_cast<float4*>(&b[load_b_gmem_addr])[0];

        /*
            0. bank布局分析: s_a[8][128]
            每个bank 4字节（共32个bank，总共128字节，32个float），
            每个bank包含1个float。s_a[8][128]的smem banks布局如下:
            8*(128/32)=32个bank层，每个k-th行有4层。

            [k=0][m=  [0],   [1],   [2],...,    [31]]
            layer_0   [b0],  [b1],  [b2],...,   [b31]
            [k=0][m=  [32],  [33],  [34],...,   [63]]
            layer_1   [b0],  [b1],  [b2],...,   [b31]
            [k=0][m=  [64],  [65],  [66],...,   [95]]
            layer_2   [b0],  [b1],  [b2],...,   [b31]
            [k=0][m=  [96],  [97],  [98],...,   [127]]
            layer_3   [b0],  [b1],  [b2],...,   [b31]
            ...       ...               ...
            [k=7][m=  [0],   [1],   [2],...,    [31]]
            layer_28  [b0],  [b1],  [b2],...,   [b31]
            [k=7][m=  [32],  [33],  [34],...,   [63]]
            layer_29  [b0],  [b1],  [b2],...,   [b31]
            [k=7][m=  [64],  [65],  [66],...,   [95]]
            layer_30  [b0],  [b1],  [b2],...,   [b31]
            [k=7][m=  [96],  [97],  [98],...,   [127]]
            layer_31  [b0],  [b1],  [b2],...,   [b31]

            1. bank冲突分析: s_a[8][128]
            tid 0   -> m 0,   k 0 -> 全部访问bank 0   (layer_0/4/8/12)
            tid 1   -> m 0,   k 4 -> 全部访问bank 0   (layer_16/20/24/28)
            tid 2   -> m 1,   k 0 -> 全部访问bank 1   (layer_0/4/8/12)
            tid 3   -> m 1,   k 4 -> 全部访问bank 1   (layer_16/20/24/28)
            tid 4   -> m 2,   k 0 -> 全部访问bank 2   (layer_0/4/8/12)
            tid 5   -> m 2,   k 4 -> 全部访问bank 2   (layer_16/20/24/28)
            tid 6   -> m 3,   k 0 -> 全部访问bank 3   (layer_0/4/8/12)
            tid 7   -> m 3,   k 4 -> 全部访问bank 3   (layer_16/20/24/28)
            ...        ...           ...                ...
            tid 28  -> m 14,  k 0 -> 全部访问bank 14  (layer_0/4/8/12)
            tid 29  -> m 14,  k 4 -> 全部访问bank 14  (layer_16/20/24/28)
            tid 30  -> m 15,  k 0 -> 全部访问bank 15  (layer_0/2/4/6)
            tid 31  -> m 15,  k 4 -> 全部访问bank 15  (layer_16/20/24/28)

            结论：对于smem_a写入访问，我们依然还有bank冲突，
            每个warp内连续的2个thread会访问同一个bank！
            所以，每个warp里至少还有2次内存访问冲突。
        */

        // load s_a
        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        // load s_b vectorized
        reinterpret_cast<float4*>(&s_b[load_b_smem_k][load_b_smem_n])[0] = reinterpret_cast<float4*>(&r_load_b[0])[0];

        __syncthreads(); // 等待所有的线程都把数据加载完毕

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            int row = threadIdx.y * TM / 2;
            int col = threadIdx.x * TN / 2;
            reinterpret_cast<float4*>(&r_comp_a[0])[0] = reinterpret_cast<float4*>(&s_a[k][row])[0];
            reinterpret_cast<float4*>(&r_comp_a[4])[0] = reinterpret_cast<float4*>(&s_a[k][row + BM / 2])[0];
            reinterpret_cast<float4*>(&r_comp_b[0])[0] = reinterpret_cast<float4*>(&s_b[k][col])[0];
            reinterpret_cast<float4*>(&r_comp_b[4])[0] = reinterpret_cast<float4*>(&s_b[k][col + BN / 2])[0];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    // compute and write to r_c
                    r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
                }
            }
        }
        __syncthreads(); // 这里需要等待所有的线程都在当前的BK块计算完毕再循环到下一个BK块
    }
#pragma unroll
    // 把结果写入到矩阵c中
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + i;
        int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        reinterpret_cast<float4*>(&c[store_c_gmem_addr])[0] = reinterpret_cast<float4*>(&r_c[i][0])[0];
        reinterpret_cast<float4*>(&c[store_c_gmem_addr + BN / 2])[0] = reinterpret_cast<float4*>(&r_c[i][4])[0];
    }
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        reinterpret_cast<float4*>(&c[store_c_gmem_addr])[0] = reinterpret_cast<float4*>(&r_c[i + TM / 2][0])[0];
        reinterpret_cast<float4*>(&c[store_c_gmem_addr + BN / 2])[0] = reinterpret_cast<float4*>(&r_c[i + TM / 2][4])[0];
    }
}


// 比较之前在计算的时候少了一个 sync 的过程，从而达到了优化加速的效果
template<const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ float s_a[2][BK][BM + OFFSET];
    __shared__ float s_b[2][BK][BN + OFFSET];

    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = (tid & 31) << 2;
    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    // 启动pipeline之前需要首先加载一次s_a和s_b
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        reinterpret_cast<float4*>(&r_load_a[0])[0] = reinterpret_cast<float4*>(&a[load_a_gmem_addr])[0];
        reinterpret_cast<float4*>(&r_load_b[0])[0] = reinterpret_cast<float4*>(&b[load_b_gmem_addr])[0];
        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        reinterpret_cast<float4*>(&s_b[0][load_b_smem_k][load_b_smem_n])[0] = reinterpret_cast<float4*>(&r_load_b[0])[0];
    }

    __syncthreads(); // 等待所有的线程都加载完毕了

    int smem_sel = 0; // 用于追踪当前使用的 buffer
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        reinterpret_cast<float4*>(&r_load_a[0])[0] = reinterpret_cast<float4*>(&a[load_a_gmem_addr])[0];
        reinterpret_cast<float4*>(&r_load_b[0])[0] = reinterpret_cast<float4*>(&b[load_b_gmem_addr])[0];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            int row = threadIdx.y * TM / 2;
            int col = threadIdx.x * TN / 2;
            reinterpret_cast<float4*>(&r_comp_a[0])[0] = reinterpret_cast<float4*>(&s_a[smem_sel][k][row])[0];
            reinterpret_cast<float4*>(&r_comp_a[4])[0] = reinterpret_cast<float4*>(&s_a[smem_sel][k][row + BM / 2])[0];
            reinterpret_cast<float4*>(&r_comp_b[0])[0] = reinterpret_cast<float4*>(&s_b[smem_sel][k][col])[0];
            reinterpret_cast<float4*>(&r_comp_b[4])[0] = reinterpret_cast<float4*>(&s_b[smem_sel][k][col + BN / 2])[0];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
                }
            }
        }

        // 加载下一次要使用的s_a和s_b
        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        reinterpret_cast<float4*>(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n])[0] = reinterpret_cast<float4*>(&r_load_b[0])[0];
        __syncthreads(); // 等待所有的线程都加载完毕了
    }
    // 计算最后一个BK块，此时数据在 smem_sel_next 中，即 ((K + BK - 1) / BK - 1) & 1
    smem_sel = ((K + BK - 1) / BK - 1) & 1;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
        int row = threadIdx.y * TM / 2;
        int col = threadIdx.x * TN / 2;
        reinterpret_cast<float4*>(&r_comp_a[0])[0] = reinterpret_cast<float4*>(&s_a[smem_sel][k][row])[0];
        reinterpret_cast<float4*>(&r_comp_a[4])[0] = reinterpret_cast<float4*>(&s_a[smem_sel][k][row + BM / 2])[0];
        reinterpret_cast<float4*>(&r_comp_b[0])[0] = reinterpret_cast<float4*>(&s_b[smem_sel][k][col])[0];
        reinterpret_cast<float4*>(&r_comp_b[4])[0] = reinterpret_cast<float4*>(&s_b[smem_sel][k][col + BN / 2])[0];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                r_c[m][n] = __fmaf_rn(r_comp_a[m], r_comp_b[n], r_c[m][n]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + i;
        int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        reinterpret_cast<float4*>(&c[store_c_gmem_addr])[0] = reinterpret_cast<float4*>(&r_c[i][0])[0];
        reinterpret_cast<float4*>(&c[store_c_gmem_addr + BN / 2])[0] = reinterpret_cast<float4*>(&r_c[i][4])[0];
    }
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + BM / 2 + i;
        int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        reinterpret_cast<float4*>(&c[store_c_gmem_addr])[0] = reinterpret_cast<float4*>(&r_c[i + TM / 2][0])[0];
        reinterpret_cast<float4*>(&c[store_c_gmem_addr + BN / 2])[0] = reinterpret_cast<float4*>(&r_c[i + TM / 2][4])[0];
    }
}



// ==========================================
// 辅助函数：CPU 验证
// ==========================================
void verify_result(float *h_c, int M, int N, float expected_val) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_c[i] - expected_val);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max Error: %f", max_diff);
    if (max_diff < 1e-4) printf(" => PASS\n");
    else printf(" => FAIL\n");
}


int main() {
    int M = 1024 * 2;
    int N = 1024 * 2;
    int K = 1024 * 2;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    printf("Matrix Size: %dx%dx%d\n", M, N, K);

    // Host Memory
    std::vector<float> h_a(M * K, 1.0f); // A 全是 1
    std::vector<float> h_b(K * N, 2.0f); // B 全是 2
    std::vector<float> h_c(M * N, 0.0f);

    // 理论结果： 1 * 2 * K = 2 * 1024 = 2048
    float expected_val = 2.0f * K;

    // Device Memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size_b));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size_c));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // ================= Naive Test =================
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    printf("\n=== Testing Naive SGEMM ===\n");
    // Warmup
    sgemm_naive_f32_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_naive_f32_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= Tiled Test =================
    // 清空 d_c 以防万一
    cudaMemset(d_c, 0, size_c);

    printf("\n=== Testing Tiled SGEMM (Sliced K) ===\n");
    // Warmup
    sgemm_sliced_k_f32_kernel<32, 32, 32><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_sliced_k_f32_kernel<32, 32, 32><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= Optimized Test (New Added) =================
    // 清空 d_c
    cudaMemset(d_c, 0, size_c);

    // 配置参数与 Template 保持一致: BM=128, BN=128, BK=8, TM=8, TN=8
    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;

    // Block Dim 计算: (BN/TN, BM/TM) => (128/8, 128/8) => (16, 16)
    dim3 block_opt(BN / TN, BM / TM);
    // Grid Dim 计算: M和N维度分别按照 BM 和 BN 切分
    dim3 grid_opt((N + BN - 1) / BN, (M + BM - 1) / BM);

    printf("\n=== Testing Optimized SGEMM (128x128 Tile + Vectorize) ===\n");
    // Warmup
    // 显式指定模板参数，确保与上方参数变量一致
    sgemm_t_8x8_sliced_k_f32x4_kernel<128, 128, 8, 8, 8><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_t_8x8_sliced_k_f32x4_kernel<128, 128, 8, 8, 8><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= BCF (Bank Conflict Free) Test =================
    // 清空 d_c
    cudaMemset(d_c, 0, size_c);

    printf("\n=== Testing BCF SGEMM (128x128 Tile + Vectorize + Transpose s_a) ===\n");
    // Warmup - 使用 OFFSET=0 版本
    sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<128, 128, 8, 8, 8, 0><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<128, 128, 8, 8, 8, 0><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms (OFFSET=0)\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= BCF with OFFSET Test =================
    // 清空 d_c
    cudaMemset(d_c, 0, size_c);

    printf("\n=== Testing BCF SGEMM with Padding (OFFSET=4) ===\n");
    // Warmup - 使用 OFFSET=4 版本，进一步避免 bank conflict
    sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<128, 128, 8, 8, 8, 4><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_t_8x8_sliced_k_f32x4_bcf_kernel<128, 128, 8, 8, 8, 4><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms (OFFSET=4)\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= Double Buffer Test =================
    // 清空 d_c
    cudaMemset(d_c, 0, size_c);

    printf("\n=== Testing Double Buffer SGEMM (BCF + Double Buffering) ===\n");
    // Warmup - 使用 OFFSET=0 版本
    sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<128, 128, 8, 8, 8, 0><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<128, 128, 8, 8, 8, 0><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms (OFFSET=0)\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // ================= Double Buffer with OFFSET Test =================
    // 清空 d_c
    cudaMemset(d_c, 0, size_c);

    printf("\n=== Testing Double Buffer SGEMM with Padding (OFFSET=4) ===\n");
    // Warmup - 使用 OFFSET=4 版本
    sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<128, 128, 8, 8, 8, 4><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel<128, 128, 8, 8, 8, 4><<<grid_opt, block_opt>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("Avg Time: %.3f ms (OFFSET=4)\n", ms / 10.0f);

    // Check Result
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    verify_result(h_c.data(), M, N, expected_val);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}