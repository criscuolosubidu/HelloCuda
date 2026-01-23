# WMMA 基础和 K-STAGE

WMMA (Warp Matrix Multiply Accumulate)，英伟达CUDA里面提供的一个面向Tensor Core的API，实现了硬件上的GEMM算法的加速，最大区别在于它使用WARP视角而不是传统的Thread视角。

---

### 第一步：WMMA (Warp Matrix Multiply Accumulate) 基础扫盲

你之前的内核是让每个线程计算一个 `float` 或 `float4`。但在 Tensor Core 的世界里，最小的计算单位不是“线程”，而是 **“Warp (线程束)”**。

#### 1. 核心概念：Fragment (片段)

* **把它想象成一个“黑盒”寄存器块**：你不能直接像数组那样 `frag[i]` 随意访问数据（虽然有办法，但通常不建议），它专门是为了配合 Tensor Core 硬件指令设计的。
* **三种角色**：
* `matrix_a`：存放矩阵 A 的块。
* `matrix_b`：存放矩阵 B 的块。
* `accumulator`：存放矩阵 C（累加结果）的块。

#### 2. 核心操作：Warp 协同

不需要自己写 `FMA` 指令了，WMMA API 是一组 **Warp 级别的函数**，意味着同一个 Warp 里的 32 个线程必须同时调用它们，Tensor Core 会在硬件层面把任务分发给这 32 个线程。

* `wmma::load_matrix_sync`：全 Warp 合作，从 Shared Memory (Smem) 或 Global Memory (Gmem) 把数据搬进 Fragment。
* `wmma::mma_sync`：核心计算 `D = A * B + C`。全 Warp 合作完成一次矩阵乘法。
* `wmma::store_matrix_sync`：把 Fragment 里的结果写回内存。

#### 3. 维度约束

WMMA 是有形状要求的，最经典的是 `16x16x16` (半精度) 或 `16x16x8` (TF32)。

*  `WMMA_M=16, WMMA_N=16, WMMA_K=8` 说明它使用的是 **TF32 (TensorFloat-32)** 精度。这是 Ampere 架构 (RTX 30系/A100) 引入的，专门用于加速 FP32 计算。

#### 4. API代码示例

下面代码使用一个WARP计算一个16x16的矩阵乘法。

```cpp
#include <cuda_runtime.h>
#include <mma.h> // 必须包含这个头文件
#include <stdio.h>

// 使用 nvcuda 命名空间，减少代码长度
using namespace nvcuda;

// 定义 Tensor Core 的形状
// 常见的有 16x16x16 (FP16), 16x16x8 (TF32)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_hello_world(half *a, half *b, float *c, int M, int N, int K) {
    // 假设 grid(1,1), block(32,1)。只有一个 Warp 在工作。
    
    // 1. 声明 Fragment (片段)
    // fragment<角色, M, N, K, 数据类型, 布局>
    // matrix_a: 矩阵A的片段
    // matrix_b: 矩阵B的片段
    // accumulator: 累加器C的片段 (通常是 float 以保证精度)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 2. 初始化累加器
    // 将 c_frag 清零。类似 float sum = 0;
    wmma::fill_fragment(c_frag, 0.0f);

    // 3. 循环 K 维 (类似于普通 GEMM 的最内层循环)
    // 每次步进 WMMA_K (这里是16)
    for (int k = 0; k < K; k += WMMA_K) {
        
        // --- Load ---
        // 从 Global Memory 加载数据到 Fragment (寄存器)
        // 参数: (目标frag, 源指针, 跨度stride)
        // 跨度(Stride): 也就是 Leading Dimension。
        // 对于行主序矩阵 A (MxK)，下一行的数据在内存中隔了 K 个元素，所以 stride = K
        // 对于行主序矩阵 B (KxN)，下一行的数据在内存中隔了 N 个元素，所以 stride = N
        
        // a + k: 指针偏移。因为我们要读 A 的第 k 列开始的块
        // b + k * N: 指针偏移。因为我们要读 B 的第 k 行开始的块
        wmma::load_matrix_sync(a_frag, a + k, K); 
        wmma::load_matrix_sync(b_frag, b + k * N, N);

        // --- Compute ---
        // 执行矩阵乘加: D = A * B + C
        // 全 Warp 32个线程协同完成
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4. Store
    // 将计算结果从 Fragment 写回 Global Memory
    // 这里的 stride 是 N，因为 C 矩阵的大小是 MxN
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}
```

---

### 第二步：什么是 K_STAGE？(从 Double Buffer 到 N-Stage)

之前的写的代码里用了 `__shared__ float s_a[2][BK][BM]`，这就是 **Double Buffer (双缓冲)**，也就是 `K_STAGE = 2`。

* **Double Buffer (Stage=2)**：
* Buffer 0: 正在被计算 (Compute)。
* Buffer 1: 正在从 Global Memory 加载 (Load)。
* *问题*：如果计算太快，加载太慢，计算单元还是要停下来等加载。


* **Multi-Stage (Stage=N)**：
* 我们不仅想预取“下一块”，还想预取“下下块”、“下下下块”。
* 比如 `K_STAGE = 4`：
* Stage 0: 计算中 (Compute)。
* Stage 1: 数据已就绪 (Ready)。
* Stage 2: 正在传输中 (In-flight)。
* Stage 3: 刚刚发出加载指令 (Issued)。


* **目的**：利用 `cp.async` 的能力，让数据在总线上“飞”得更久一点，最大程度掩盖 Global Memory 的延迟。

---

### 第三步：深度解析大佬的内核代码

让我们带着上面的知识，把那个内核拆解开。

#### 1. 动态的 Shared Memory 布局

```cpp
// K_STAGE 是模板参数，比如 3, 4, 5...
__shared__ float s_a[K_STAGE][BM][BK + A_PAD];
```

这里的 `s_a` 变成了一个**环形缓冲区 (Circular Buffer)**。

#### 2. Prologue (序幕)：预加载

在进入主循环之前，代码先启动了前 `K_STAGE - 1` 个块的加载：

```cpp
#pragma unroll
for (int k = 0; k < (K_STAGE - 1); ++k) {
    // ... 计算地址 ...
    CP_ASYNC_CG(...); // 发出异步加载指令
    CP_ASYNC_COMMIT_GROUP(); // 把这批加载打包成一个 Group
}
CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // 关键！
__syncthreads();
```

* **`COMMIT_GROUP`**：打个标记，“这批货是一组的”。
* **`WAIT_GROUP(N)`**：这是异步流水线的灵魂。它的意思是：“阻塞在这里，直到**还没完成**的 Group 数量只剩下 `N` 个”。
* 如果你发出了 `K_STAGE - 1` 个组（比如 3 个），而你 `WAIT_GROUP(K_STAGE - 2)`（保留 2 个），意味着你只需要**第 1 个组**加载完成就可以继续了。第 2、3 个组可以还在路上。

#### 3. Main Loop (主循环)：计算与预取并行

```cpp
for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
    // 1. 算出当前要计算谁 (smem_sel)，下一波加载要填到哪 (smem_sel_next)
    // 这是一个环形索引：0 -> 1 -> 2 -> 0 ...
    int smem_sel = (k + 1) % K_STAGE;
    int smem_sel_next = k % K_STAGE;

    // 2. 发起【未来】数据的加载指令 (写到 smem_sel_next)
    CP_ASYNC_CG(...); 
    CP_ASYNC_COMMIT_GROUP();

    // 3. 真正干活：从 Shared Memory 加载到寄存器 (利用 smem_sel)
    wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel]...);
    
    // 4. 计算
    wmma::mma_sync(...);

    // 5. 流水线推进
    // 我们刚提交了一个新组，现在未完成的组多了1个。
    // 我们需要确保下一轮计算用的数据准备好，所以再次 Wait
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2); 
    __syncthreads();
}
```

**逻辑流：**

* 我正在计算 `T` 时刻的数据。
* 我发出了 `T + (K_STAGE-1)` 时刻的加载请求。
* 我确认 `T + 1` 时刻的数据已经到货（通过 `WAIT_GROUP`），这样下一次循环就能直接算，不用等。

#### 4. Epilogue (尾声)：收尾工作

主循环结束后，还有最后几块数据已经在 Shared Memory 里了（或者还在路上），但还没计算。

```cpp
CP_ASYNC_WAIT_GROUP(0); // 等待所有在途数据全部到齐
__syncthreads();
// ... 剩下的计算循环，不再发出新的加载指令 ...
```
