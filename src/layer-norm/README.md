## 🧩 1. 背景：为什么需要归一化（Normalization）

在神经网络中，输入的特征往往在不同维度上数值尺度差异巨大，这会导致：

* 梯度传播不稳定（梯度爆炸或消失）；
* 学习速度慢；
* 对初始化敏感。

**归一化（Normalization）** 的核心思想就是：

> 在每层计算中，让特征分布更“标准化”，从而让训练更稳、更快。

归一化最早的突破是 **Batch Normalization（Ioffe & Szegedy, 2015）**。它后来催生了 LayerNorm、InstanceNorm、GroupNorm、RMSNorm 等一系列变体。

---

## 🧮 2. Batch Normalization（BN）

### (1) 数学定义

对每个 mini-batch 的输入特征 $x$，对每个通道（feature map）进行标准化：

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

其中：
$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
$$

然后再引入可学习参数 $\gamma$ 和 $\beta$：
$$
y_i = \gamma \hat{x}_i + \beta
$$

### (2) 直觉解释

BN 相当于在每个通道上，减去 batch 内的均值，除以标准差，让激活值分布接近标准正态分布（0 均值，1 方差）。
→ **优点：** 梯度稳定，网络更容易训练。
→ **缺点：** 对 batch size 敏感，在 RNN / Transformer 这种时间序列模型里效果不好（因为每步输入独立）。

---

## 📏 3. Layer Normalization（LN）

### (1) 诞生背景

LayerNorm（Ba et al., 2016）提出的目的就是：

> 让归一化与 batch 大小无关，只依赖单个样本的特征维度。

这使得它更适合 **序列模型（RNN、Transformer）**。

### (2) 数学定义

对于某个样本的输入向量 $x = [x_1, x_2, ..., x_H]$：

$$
\mu = \frac{1}{H}\sum_{i=1}^H x_i, \quad \sigma^2 = \frac{1}{H}\sum_{i=1}^H (x_i - \mu)^2
$$

$$
\text{LayerNorm}(x_i) = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

### (3) 与 BatchNorm 的区别

| 特性                     | BatchNorm       | LayerNorm       |
| ---------------------- | --------------- | --------------- |
| 归一化维度                  | batch + channel | feature 维度（单样本） |
| 是否依赖 batch 大小          | 是               | 否               |
| 是否适合 RNN / Transformer | 否               | 是               |
| 是否在推理时需要统计均值方差         | 需要              | 不需要             |

### (4) 几何直觉

LayerNorm 把每个样本的向量“投影”到单位球上（均值为0，方差为1）。
换句话说，它规范了样本在特征空间的尺度和偏移，使得优化器能更稳定地更新方向。

---

## 📉 4. RMS Normalization（RMSNorm）

### (1) 背景

RMSNorm（Zhang & Sennrich, 2019）是在 LayerNorm 基础上去掉了“减均值”这一步。

### (2) 数学定义

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$
其中
$$
\text{RMS}(x) = \sqrt{\frac{1}{H}\sum_{i=1}^H x_i^2 + \epsilon}
$$

### (3) 直觉解释

* LayerNorm 中的“减均值”会改变方向信息（因为平移了向量中心）。
* 而 RMSNorm 只缩放，不平移。
* 在 Transformer 这种主要依赖方向关系的模型中，这样可能更稳。

所以 RMSNorm 通常更轻量，计算开销略小，效果与 LayerNorm 接近或更好（尤其在大型语言模型中，LLaMA 就用了 RMSNorm）。

---

## 🔬 5. 使用与代码示例（PyTorch）

```python
import torch
import torch.nn as nn

# LayerNorm 示例
x = torch.randn(3, 5)  # batch=3, hidden=5
ln = nn.LayerNorm(5)
print(ln(x))

# RMSNorm 实现（PyTorch 里没有内置）
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * norm_x

rms = RMSNorm(5)
print(rms(x))
```

---

## 🧠 6. 小结与记忆线索

| 方法        | 归一化维度           | 是否减均值 | 是否依赖 batch | 常用场景                  |
| --------- | --------------- | ----- | ---------- | --------------------- |
| BatchNorm | across batch    | ✅     | ✅          | CNN                   |
| LayerNorm | across features | ✅     | ❌          | RNN, Transformer      |
| RMSNorm   | across features | ❌     | ❌          | Transformer (LLaMA 等) |

> 📌 记忆口诀：
> **“Batch 用批，Layer 用维，RMS 不减均值。”**

---

# 问题1：为什么transformer不使用BatchNorm?

这是一个非常经典的问题。

简单来说，**Batch Normalization (BatchNorm) 不适用于 Transformer，主要是因为它依赖于“批次”的统计数据（均值和方差），而这在处理序列数据（尤其是NLP任务）时会带来严重问题。**

Transformer 中使用的是 **Layer Normalization (LayerNorm)**。

下面详细解释一下为什么 Batch Norm 不行，而 LayerNorm 可以：

### 1\. Batch Normalization (BN) 的工作原理

BatchNorm 的核心思想是，在网络的每一层，它都会计算**当前 mini-batch 中**所有样本在*同一个特征维度*上的均值和方差，然后用这个均值和方差来归一化该特征。

* **计算对象**：$N$ 个样本的 $C$ 个通道中的第 $c$ 个通道。
* **依赖性**：**强依赖于 Batch Size。**

### 2\. Batch Norm 在 Transformer 上的“水土不服”

当 Batch Norm 遇到 Transformer（特别是用于NLP任务时），会产生几个关键问题：

#### (1) 序列长度不一致 (Variable Sequence Lengths)

* 在NLP任务中，一个批次（batch）里的句子（序列）长度几乎总是不一样的。
* 为了把它们放进一个张量（tensor）中，我们必须用特殊的 `[PAD]`（填充）标记将短句子“填充”到和最长的句子一样长。
* **问题所在**：Batch Norm 会把这些 `[PAD]` 标记也算进去，用来计算整个批次的均值和方差。这些 `[PAD]` 标记是“假”数据，它们本不应具有统计意义。这会导致计算出来的均值和方差是“失真”的、被污染的，从而严重影响模型的训练效果。

#### (2) 小批量大小 (Small Batch Sizes)

* Transformer 模型（如 BERT, GPT）非常巨大，消耗显存极多。
* 这导致在实际训练中，我们（尤其是用单张GPU时）往往只能使用很小
  的 Batch Size（比如 2, 4, 8）。
* **问题所在**：Batch Norm 严重依赖“批次统计数据”。如果批次太小，那么这个批次的均值和方差就无法准确代表“全局”数据的分布。这个统计数据会充满噪声、非常不稳定，导致模型训练困难，甚至崩溃。

#### (3) 训练 (Training) 和 推理 (Inference) 时的不一致

* **训练时**：BN 使用当前 mini-batch 的均值和方差。
* **推理时**：通常我们一次只处理一个句子（Batch Size = 1），根本没有“批次”可言。此时 BN 使用的是在训练时计算并保存下来的“全局”均值和方差的*滑动平均值*。
* **问题所在**：如果训练时的批次很小（问题2）或者被 `[PAD]` 污染（问题1），那么这个“滑动平均值”本身就是不准确的。这会导致模型在训练和推理时的行为严重不一致，性能大幅下降。
