# CUDA cp.async Group 机制详解

## 概述

`cp.async` 是 CUDA 中用于异步内存复制的 PTX 指令，可以实现从 Global Memory 到 Shared Memory 的异步数据传输，从而实现**计算与数据加载的重叠**。

## 核心指令

```cpp
// 异步复制（不同缓存策略）
#define CP_ASYNC_CA(dst, src, bytes)  // cache all levels
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ...)

#define CP_ASYNC_CG(dst, src, bytes)  // cache global level (跳过 L1)
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ...)

// 提交当前组
#define CP_ASYNC_COMMIT_GROUP() 
    asm volatile("cp.async.commit_group;\n" ::)

// 等待：队列中未完成组数量 ≤ N
#define CP_ASYNC_WAIT_GROUP(n) 
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// 等待所有组完成（等价于 WAIT_GROUP(0)）
#define CP_ASYNC_WAIT_ALL() 
    asm volatile("cp.async.wait_all;\n" ::)
```

## Group 机制

### 基本概念

- **Group 编号是隐式的**，由硬件自动管理
- 采用 **FIFO 队列模型**
- 每次 `COMMIT_GROUP()` 将当前所有未提交的 `cp.async` 操作打包成一个组

### 工作流程

```
cp.async ...          ─┐
cp.async ...           │ 属于"当前未提交组"
cp.async ...          ─┘
COMMIT_GROUP()           ← 提交，形成 Group 0，入队

cp.async ...          ─┐
cp.async ...           │ 属于"当前未提交组"  
cp.async ...          ─┘
COMMIT_GROUP()           ← 提交，形成 Group 1，入队

队列状态: [Group 0, Group 1]
```

## WAIT_GROUP 语义

### 核心定义

```
WAIT_GROUP(N) = 等待直到 FIFO 队列中未完成的组数量 ≤ N
```

| 调用 | 含义 |
|------|------|
| `WAIT_GROUP(0)` | 等待所有组完成 |
| `WAIT_GROUP(1)` | 等待直到 ≤ 1 个组在飞行中 |
| `WAIT_GROUP(2)` | 等待直到 ≤ 2 个组在飞行中 |

### 示例

```
初始队列: [G0, G1, G2]  (3个组在飞行)

WAIT_GROUP(1) 执行后:
  - G0 完成 ✓
  - G1 完成 ✓
  - 队列变为 [G2]（大小=1，满足 ≤1）
  
此时可安全使用 G0, G1 的数据
G2 继续异步加载
```

## FIFO 顺序保证

### 硬件保证

**先提交的组一定先完成**，这是硬件的强保证：

```
提交顺序:  Group 0 → Group 1 → Group 2 → Group 3
完成顺序:  Group 0 → Group 1 → Group 2 → Group 3  ✓ 保证！
```

### 为什么重要？

FIFO 保证让你可以**精确控制哪些数据已就绪**：

```cpp
// 提交了 3 个组: G0(stage0), G1(stage1), G2(stage2)
CP_ASYNC_WAIT_GROUP(1);

// 100% 确定:
// - G0 (stage 0) 已完成 ✓
// - G1 (stage 1) 已完成 ✓  
// - G2 (stage 2) 可能还在进行
```

没有 FIFO 保证，就无法知道哪个 stage 的数据准备好了，流水线无法正确工作。

## 多级流水线应用

### 典型模式（K_STAGE 级流水线）

```cpp
__shared__ float buffer[K_STAGE][...];  // 多级缓冲

// ===== 预加载阶段 =====
// 加载 K_STAGE - 1 个组
for (int k = 0; k < K_STAGE - 1; ++k) {
    CP_ASYNC_CG(..., &data[k * stride], 16);
    CP_ASYNC_COMMIT_GROUP();  // 每次迭代提交一个组
}
// 队列: [G0, G1, ..., G(K_STAGE-2)]

// ===== 等待第一个 stage 就绪 =====
CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
// 含义: 允许 ≤ (K_STAGE - 2) 个组还在飞行
// 结果: 至少 1 个组(G0)已完成

// ===== 主计算循环 =====
for (int k = K_STAGE - 1; k < NUM_K_TILES; ++k) {
    int stage = k % K_STAGE;
    
    // 使用当前 stage 数据进行计算（数据已就绪）
    compute(buffer[stage]);
    
    // 异步加载下一批数据
    CP_ASYNC_CG(..., &data[k * stride], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // 等待下一个要用的 stage 就绪
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
}
```

### K_STAGE - 2 的含义

| 参数 | 含义 |
|------|------|
| `K_STAGE` | 共享内存缓冲区数量 |
| `K_STAGE - 1` | 预加载的组数 |
| `K_STAGE - 2` | 允许还在飞行中的组数 |

**计算方式**：
- 预加载了 `K_STAGE - 1` 个组
- 要用第一个组的数据，所以至少等 1 个组完成
- 允许剩下的 `(K_STAGE - 1) - 1 = K_STAGE - 2` 个组继续飞行

### 具体例子

#### K_STAGE = 2（双缓冲）

```
预加载: 1 个组 (G0)
WAIT_GROUP(0): 等待所有完成，G0 就绪
主循环: 用 stage 0，加载 stage 1，交替进行
```

#### K_STAGE = 3（三级缓冲）

```
预加载: 2 个组 (G0, G1)
WAIT_GROUP(1): 允许 ≤1 个飞行，G0 就绪，G1 可能还在加载
主循环: 用 stage 0，加载 stage 2；用 stage 1，加载 stage 0；循环
```

#### K_STAGE = 4（四级缓冲）

```
预加载: 3 个组 (G0, G1, G2)
WAIT_GROUP(2): 允许 ≤2 个飞行，G0 就绪，G1/G2 可能还在加载
主循环: 更深的流水线，更好地隐藏延迟
```

## 流水线时间线示意

以 K_STAGE = 3 为例：

```
时间 →
────────────────────────────────────────────────────────

Load:   [G0 加载][G1 加载][G2 加载][G0 加载][G1 加载]...
                  ↓
Compute:         [G0 计算][G1 计算][G2 计算][G0 计算]...

重叠！计算 G0 时，G1/G2 在加载
```

## 注意事项

1. **GROUP 编号不是显式指定的**，而是按 COMMIT 顺序自动分配
2. **WAIT_GROUP 的参数是"允许飞行的数量"**，不是"等待的组编号"
3. **必须在使用 shared memory 数据前确保对应的组已完成**
4. **K_STAGE 越大，流水线越深**，但 shared memory 占用也越大

## 参考资料

- [NVIDIA PTX ISA - Data Movement and Conversion Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
- [CUDA C++ Programming Guide - Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)

