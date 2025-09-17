# 使用 Nsight Systems 和 Nsight Compute 来进行性能分析

---

### 第一步：系统级分析 (找到瓶颈**在哪里**)

* **工具**: **Nsight Systems** (命令: `nsys`)
* **目的**: 鸟瞰整个程序，了解时间都花在了哪里。
* **做什么**: 运行 `nsys profile ./your_app`，它会生成一个时间线报告。
* **看什么**: 在报告中查看 CPU 和 GPU 的活动，重点关注 **内存拷贝 (Memcpy)** 和 **内核执行 (Kernel)** 哪个占用了大部分时间。
* **结果**: 帮你定位到最大的性能瓶颈，例如：“数据传输太耗时了”，或者“`my_kernel` 这个内核运行得太慢”。

---

### 第二步：内核级分析 (搞清楚**为什么**慢)

* **工具**: **Nsight Compute** (命令: `ncu`)
* **目的**: 深入到第一步发现的那个慢内核内部，分析其硬件性能。
* **做什么**: 运行 `ncu -o report ./your_app`，它会专门收集内核的详细数据。
* **看什么**: 在报告中，首先看 **GPU Speed of Light** 部分。它会直接告诉你内核的主要限制因素。
* **结果**: 帮你找到内核慢的根本原因，通常是两种：
    * **访存受限 (Memory Bound)**: 内核大部分时间在等待数据从内存中读写，计算单元很闲。
    * **计算受限 (Compute Bound)**: 数据供应很快，但内核的计算任务太繁重，计算单元一直在忙。

-----

### 准备阶段: 编译代码

为了让性能分析工具能将结果关联到你的源代码，编译时需要加入调试信息。

```bash
# -g: 为 CPU 代码加调试信息
# -G: 为 GPU 设备代码加调试信息
nvcc -g -G -o block_all_reduce.exe your_program.cu
```

-----

### 第一步: Nsight Systems (nsys) 系统级分析

**目标**: 找到程序的宏观瓶颈，比如是内存拷贝慢还是某个内核慢。

#### 示例 1: 基本的性能分析

这是最常用的命令，它会追踪 CUDA、操作系统调用等一系列活动。

```bash
# 对你的程序进行性能分析，默认输出名为 report.nsys-rep
nsys profile ./block_all_reduce.exe
```

#### 示例 2: 指定输出名称并覆盖旧文件

在反复测试时，这个命令非常方便，可以避免生成一堆 `report1`, `report2` ... 的文件。

```bash
# -o: 指定输出文件名
# --force-overwrite true: 如果文件已存在，则直接覆盖
nsys profile -o my_report --force-overwrite true ./block_all_reduce.exe
```

#### 示例 3: 聚焦于 CUDA 相关的活动

如果你的程序很复杂，只想关注 GPU 部分，可以用 `-t` (trace) 标志来过滤。

```bash
# -t cuda,nvtx: 只追踪 CUDA API调用 和 NVTX 自定义范围
# NVTX 是一个高级功能，可以让你在代码中打上自定义标签，显示在时间线上
nsys profile -t cuda,nvtx -o cuda_focused_report ./block_all_reduce.exe
```

-----

### 第二步: Nsight Compute (ncu) 内核级分析

**目标**: 深入分析某个特定内核，找出其性能瓶颈（访存或计算）。

#### 示例 1: 分析程序中的所有内核

默认情况下，`ncu` 会分析程序执行过程中遇到的每一个内核。

```bash
# -o: 指定输出文件名
ncu -o all_kernels_report ./block_all_reduce.exe
```

#### 示例 2: 只分析一个特定的内核（强烈推荐）

如果你的程序有多个内核，但你只关心其中一个，用 `-k` (kernel-name) 标志可以节省大量时间，并使报告更清晰。内核名称可以用正则表达式匹配。

```bash
# -k: 后跟一个正则表达式，匹配你想分析的内核名称
# 这里我们精确匹配你的内核 block_all_reduce_sum_f32x4_f32_kernel
ncu -k block_all_reduce_sum_f32x4_f32_kernel -o specific_kernel_report --force-overwrite true ./block_all_reduce.exe
```

#### 示例 3: 收集更全面的性能指标

默认情况下, `ncu` 会收集一组常用的指标。如果你想进行更深入的分析，可以指定收集一个完整的集合。

```bash
# --set full: 收集所有可用的硬件性能指标，分析会慢一些
ncu --set full -k your_kernel_name -o full_metrics_report ./block_all_reduce.exe
```
