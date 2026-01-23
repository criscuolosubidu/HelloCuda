# CUDA 地址空间详解

本文档介绍 CUDA 中地址空间的概念，以及为什么在使用 PTX 内联汇编时需要进行地址转换。

## 1. GPU 物理内存布局

GPU 有**多个物理上独立的内存区域**，它们有不同的硬件访问路径和特性：

```
┌─────────────────────────────────────────────────┐
│                    GPU                          │
│  ┌─────────────────────────────────────────┐   │
│  │              SM (流处理器)               │   │
│  │  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ 寄存器文件   │  │  共享内存 (SMEM) │   │   │
│  │  │ (每线程私有) │  │  (Block内共享)   │   │   │
│  │  └─────────────┘  └─────────────────┘   │   │
│  └─────────────────────────────────────────┘   │
│                      ↓                          │
│  ┌─────────────────────────────────────────┐   │
│  │           L1/L2 缓存                     │   │
│  └─────────────────────────────────────────┘   │
│                      ↓                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 全局内存  │  │ 常量内存  │  │  纹理内存    │  │
│  │ (GMEM)   │  │(只读,缓存)│  │ (只读,缓存)  │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### 各内存区域特性

| 内存类型 | 作用域 | 生命周期 | 访问速度 | 典型大小 |
|---------|--------|---------|---------|---------|
| 寄存器 | 线程私有 | 线程 | 最快 | 255个/线程 |
| 共享内存 | Block内共享 | Block | 很快 | 48KB-228KB |
| 全局内存 | 所有线程 | 应用程序 | 慢 | 数GB-数十GB |
| 常量内存 | 所有线程(只读) | 应用程序 | 快(有缓存) | 64KB |
| 纹理内存 | 所有线程(只读) | 应用程序 | 快(有缓存) | 取决于全局内存 |

**关键点**：这些内存区域不是同一块内存的不同名字，而是**物理上独立的硬件单元**，有各自独立的地址编号！

### 常量内存 vs 纹理内存

常量内存和纹理内存都是**只读**的，但它们在缓存机制、访问模式、功能特性上有很大区别：

| 特性 | 常量内存 (Constant Memory) | 纹理内存 (Texture Memory) |
|------|---------------------------|--------------------------|
| **大小** | 64 KB（硬件限制） | 取决于全局内存大小 |
| **缓存** | 常量缓存（每SM约8KB） | 纹理缓存（每SM约12-48KB） |
| **最优访问模式** | 所有线程读**同一地址** | 线程读**相邻地址**（空间局部性） |
| **缓存优化** | 广播优化 | 2D/3D 空间局部性优化 |
| **特殊功能** | 无 | 插值、边界处理、归一化坐标 |
| **声明方式** | `__constant__` | `texture` / `cudaTextureObject_t` |

#### 常量内存的广播机制

```cuda
__constant__ float coefficients[256];

__global__ void kernel(float *data) {
    int idx = threadIdx.x;
    // ✅ 好：所有线程读取同一个 coefficients[0]，一次读取后广播
    float result = data[idx] * coefficients[0];
    
    // ❌ 差：每个线程读取不同地址，访问会被串行化
    float bad = data[idx] * coefficients[idx];
}
```

#### 纹理内存的 2D 空间局部性

```
全局内存/常量内存缓存（线性布局）：
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  ← 一维连续
└───┴───┴───┴───┴───┴───┴───┴───┘

纹理缓存（2D 空间局部性布局，Morton/Z-order）：
┌───┬───┬───┬───┐
│ 0 │ 1 │ 4 │ 5 │    一个缓存行包含
├───┼───┼───┼───┤    2D 相邻的元素
│ 2 │ 3 │ 6 │ 7 │
├───┼───┼───┼───┤
│ 8 │ 9 │12 │13 │
├───┼───┼───┼───┤
│10 │11 │14 │15 │
└───┴───┴───┴───┘
```

#### 纹理内存的特殊功能

```cuda
cudaTextureObject_t texObj;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

cudaTextureDesc texDesc = {};
texDesc.addressMode[0] = cudaAddressModeWrap;   // 边界处理：环绕
texDesc.addressMode[1] = cudaAddressModeClamp;  // 边界处理：钳制
texDesc.filterMode = cudaFilterModeLinear;       // 双线性插值（硬件免费）
texDesc.normalizedCoords = true;                 // 归一化坐标 [0,1]

cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

__global__ void texture_kernel(cudaTextureObject_t tex, float *output) {
    float u = threadIdx.x / (float)blockDim.x;  // 归一化坐标
    float v = threadIdx.y / (float)blockDim.y;
    
    // 硬件自动进行双线性插值！
    float value = tex2D<float>(tex, u, v);
    
    // 即使坐标越界也不会出错，硬件自动处理边界
    float wrapped = tex2D<float>(tex, u + 2.5f, v);  // 自动 wrap
}
```

#### 访问模式性能对比

```
                    常量内存              纹理内存
                    ────────              ────────
所有线程同地址：     ████████ 最优        ████ 一般
                    (广播)               
                    
线程访问相邻1D：     ████ 一般            ████████ 优秀
                    (可能串行)           (缓存命中)
                    
线程访问相邻2D：     ██ 较差              ████████████ 最优
                    (缓存不友好)         (空间局部性)
                    
随机访问：          █ 很差               ████ 一般
                   (串行化)              (比全局内存好)
```

#### 选择建议

| 选择依据 | 使用常量内存 | 使用纹理内存 |
|---------|-------------|-------------|
| 数据大小 | ≤ 64KB | 任意大小 |
| 访问模式 | 所有线程读同一地址 | 2D/3D 空间相邻访问 |
| 是否需要插值 | 不需要 | 需要（免费硬件插值） |
| 是否需要边界处理 | 不需要 | 需要（自动处理） |
| 典型应用 | 常量参数、广播数据 | 图像、体数据、不规则访问 |

简单记忆：
- **常量内存** = 广播器（一对多）
- **纹理内存** = 图像专用缓存（2D 友好 + 插值）

## 2. 地址空间冲突问题

假设：
- 共享内存的地址范围：`0x0000` - `0xFFFF`（64KB）
- 全局内存的地址范围：`0x00000000` - `0xFFFFFFFF`（4GB）

那么地址 `0x1000` 是什么意思？

- 在共享内存中，它是共享内存的第 4096 字节
- 在全局内存中，它是全局内存的第 4096 字节

**同一个数字地址，在不同内存中指向完全不同的物理位置！**

因此，仅靠一个整数地址无法唯一确定内存位置，还需要知道"这是哪个地址空间的地址"。

## 3. PTX 的强类型地址空间

PTX（Parallel Thread Execution）是 CUDA 的低级中间表示语言，类似于 CPU 的汇编语言。

在 PTX 中，必须**明确指定**访问的是哪个内存区域。同样是加载数据，PTX 有不同的指令：

```ptx
// 从全局内存加载
ld.global.f32 %f1, [%rd1];    // rd1 是全局内存地址

// 从共享内存加载  
ld.shared.f32 %f1, [%r1];     // r1 是共享内存地址

// 从常量内存加载
ld.const.f32 %f1, [%r1];      // r1 是常量内存地址

// 异步拷贝（从全局到共享）
cp.async.cg.shared.global [%r1], [%rd1], 16;  // 明确指定 shared 和 global
```

这就是**"强类型地址空间"**的含义：

> **地址 + 地址空间类型** 才能唯一确定一个内存位置

## 4. C++ 的通用指针（Generic Pointer）

CUDA C++ 为了编程方便，做了一层抽象，让你可以用统一的指针语法：

```cuda
__shared__ float s_data[1024];
__device__ float g_data[1024];

__global__ void kernel() {
    float* ptr1 = &s_data[0];  // 指向共享内存
    float* ptr2 = &g_data[0];  // 指向全局内存
    
    // 都可以用相同的语法解引用
    float val1 = *ptr1;  // 编译器自动知道从共享内存读取
    float val2 = *ptr2;  // 编译器自动知道从全局内存读取
}
```

### 编译器是如何实现的？

编译器使用**"胖指针"**（Fat Pointer），用更长的地址编码了地址空间信息：

```
通用指针格式（简化示意）：
┌────────────────┬─────────────────────────────────┐
│  地址空间标记   │         实际地址                 │
│  (高位 bits)   │        (低位 bits)               │
└────────────────┴─────────────────────────────────┘

例如：
0x0001_0000_1000  →  共享内存的 0x1000
0x0000_0000_1000  →  全局内存的 0x1000
```

当你用 C++ 解引用指针时，编译器会：

1. 检查高位标记，判断是哪个地址空间
2. 生成对应的 PTX 指令（`ld.global` 或 `ld.shared` 等）

## 5. 共享内存：静态分配 vs 动态分配

共享内存有两种分配方式：**静态分配**和**动态分配**（Dynamic Shared Memory，简称 dsmem）。

### 静态共享内存

编译时确定大小，使用多维数组语法：

```cuda
__global__ void kernel_static() {
    // 编译时确定大小
    __shared__ float s_a[128][8];
    __shared__ float s_b[8][128];
    
    // 多维数组寻址
    s_a[threadIdx.y][threadIdx.x] = 1.0f;
}
```

### 动态共享内存

运行时确定大小，使用一维数组语法：

```cuda
__global__ void kernel_dynamic() {
    // 运行时确定大小，使用 extern 声明
    extern __shared__ float smem[];
    
    // 手动划分内存区域
    float *s_a = smem;                    // 从 0 开始
    float *s_b = smem + 128 * 8;          // 从 s_a 之后开始
    
    // 一维线性寻址，需要手动计算偏移
    s_a[threadIdx.y * 8 + threadIdx.x] = 1.0f;
}

// 调用时需要指定共享内存大小
size_t smem_size = (128 * 8 + 8 * 128) * sizeof(float);
kernel_dynamic<<<grid, block, smem_size>>>();
```

### 对比

| 特性 | 静态共享内存 | 动态共享内存 |
|-----|-------------|-------------|
| 大小限制 | 默认 48KB | 可超过 48KB（需配置） |
| 声明方式 | `__shared__ T arr[N]` | `extern __shared__ T smem[]` |
| 寻址方式 | 多维数组 `arr[i][j]` | 一维线性 `smem + offset` |
| 灵活性 | 编译时固定 | 运行时可变 |
| 调用方式 | 直接调用 | 需指定第三个参数 |

### 为什么需要动态共享内存？

**默认共享内存上限是 48KB**。当需要更大的共享内存时，必须使用动态共享内存并显式配置：

```cuda
// 查询设备支持的最大共享内存
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max shared memory per block: %zu bytes\n", 
       prop.sharedMemPerBlockOpaque);

// 设置 kernel 允许使用更大的动态共享内存
size_t smem_size = 80 * 1024;  // 例如 80KB
cudaFuncSetAttribute(
    kernel_dynamic,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size
);

// 启动 kernel
kernel_dynamic<<<grid, block, smem_size>>>();
```

### 各架构共享内存上限

| GPU 架构 | Compute Capability | 每 SM 最大共享内存 | 每 Block 可配置最大 |
|----------|-------------------|-------------------|-------------------|
| Volta | SM 7.0 | 96 KB | 96 KB |
| Turing | SM 7.5 | 64 KB | 64 KB |
| Ampere | SM 8.0 | 164 KB | 163 KB |
| Ampere | SM 8.6 | 100 KB | 99 KB |
| Ada Lovelace | SM 8.9 | 100 KB | 99 KB |
| Hopper | SM 9.0 | 228 KB | 227 KB |

### 共享内存与 L1 缓存的权衡

共享内存和 L1 缓存**共享同一块物理 SRAM**，可以配置分配比例：

```cuda
// 优先共享内存（适合 GEMM 等需要大量共享内存的场景）
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared  // 100% 给共享内存
);

// 或者优先 L1 缓存
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxL1  // 100% 给 L1
);
```

### 注意事项

- ✅ 动态共享内存可以**突破 48KB 的默认限制**
- ❌ 但**不能超过 GPU 硬件的物理上限**
- ⚠️ 使用超过 48KB 时，必须调用 `cudaFuncSetAttribute` 显式设置
- ⚠️ 共享内存越大，每个 SM 能同时运行的 block 数量越少（occupancy 下降）

## 6. 地址转换函数

当使用 PTX 内联汇编时，**绕过了编译器的自动处理**，需要手动提取"裸地址"。

### CUDA 提供的地址转换函数

| 函数 | 作用 | 返回类型 |
|------|------|---------|
| `__cvta_generic_to_shared(ptr)` | 转换为共享内存地址 | `uint32_t` |
| `__cvta_generic_to_global(ptr)` | 转换为全局内存地址 | `uint64_t` |
| `__cvta_generic_to_constant(ptr)` | 转换为常量内存地址 | `uint32_t` |
| `__cvta_generic_to_local(ptr)` | 转换为本地内存地址 | `uint32_t` |

### 为什么共享内存地址是 32 位？

因为共享内存大小有限（通常最大 48KB-164KB），32 位（可寻址 4GB）足够了。而全局内存可能有数十 GB，需要 64 位地址。

## 7. 实际应用：cp.async 异步拷贝

`cp.async` 是 Ampere 架构（SM80+）引入的异步内存拷贝指令，可以绕过寄存器直接将数据从全局内存复制到共享内存。

### 定义宏

```cuda
#define CP_ASYNC_CG(dst, src, bytes)                                    \
    asm volatile(                                                       \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"         \
        :: "r"(dst), "l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
```

### 使用示例

```cuda
__shared__ float s_a[128][8];

__global__ void kernel(float* A, int M, int K) {
    int tid = threadIdx.x;
    int load_smem_m = tid / 2;
    int load_smem_k = (tid % 2) * 4;
    
    // 计算全局内存地址（C++ 指针可直接使用）
    float* gmem_ptr = &A[load_smem_m * K + load_smem_k];
    
    // 计算共享内存地址（需要转换为裸地址）
    uint32_t smem_ptr = __cvta_generic_to_shared(&s_a[load_smem_m][load_smem_k]);
    
    // 异步拷贝 16 字节（4 个 float）
    CP_ASYNC_CG(smem_ptr, gmem_ptr, 16);
    
    // 提交并等待
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_ALL();
    
    __syncthreads();
    
    // 现在可以使用 s_a 中的数据了
}
```

### PTX 指令解析

```ptx
cp.async.cg.shared.global.L2::128B [%0], [%1], 16;
│      │  │      │      │          │     │     │
│      │  │      │      │          │     │     └─ 拷贝字节数
│      │  │      │      │          │     └─ 源地址（全局内存，64位）
│      │  │      │      │          └─ 目标地址（共享内存，32位）
│      │  │      │      └─ L2 缓存策略提示（128字节缓存行）
│      │  │      └─ 源地址空间：global
│      │  └─ 目标地址空间：shared
│      └─ 缓存策略：cg = cache global（缓存到 L2）
└─ 异步拷贝指令
```

## 8. 总结

| 层级 | 地址表示 | 特点 |
|------|---------|------|
| 硬件层 | 各内存区域独立编址 | 同一数字可能指向不同物理位置 |
| PTX 层 | 地址 + 显式地址空间标记 | 必须用 `.global`、`.shared` 等显式指定 |
| C++ 层 | 通用指针（胖指针） | 编译器自动处理，对程序员透明 |
| 内联汇编 | 需要裸地址 | 使用 `__cvta_generic_to_*` 转换 |

### 记忆要点

1. **GPU 有多个独立的物理内存区域**，它们有各自的地址空间
2. **PTX 是强类型地址空间**，必须明确指定访问哪个内存
3. **CUDA C++ 使用通用指针**，编译器自动处理地址空间
4. **PTX 内联汇编绕过编译器**，需要手动用 `__cvta_generic_to_*` 提取裸地址
5. **共享内存地址是 32 位**，全局内存地址是 64 位
6. **常量内存**适合广播访问（所有线程读同一地址），**纹理内存**适合 2D 空间局部性访问
7. **静态共享内存**默认上限 48KB，**动态共享内存**可突破限制但需显式配置
8. **共享内存和 L1 缓存共享物理 SRAM**，可配置分配比例

