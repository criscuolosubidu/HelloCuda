## 一、基本思想

RoPE 的核心思想是：

> 将位置编码转化为向量空间中的**旋转操作**，使得不同位置的向量在高维空间中通过"相位旋转"来表达位置信息。

对于每个 token 的 embedding 向量 $\mathbf{x} \in \mathbb{R}^d$，
RoPE 将它分为 $d/2$ 个二维子空间：

$$
(x_1, x_2), (x_3, x_4), \dots, (x_{d-1}, x_d)
$$

每个子空间都对应一个独立的旋转角度（频率）。

---

## 二、二维旋转的基本形式

二维旋转矩阵定义为：

$$
R(\theta) =
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
$$

旋转操作为：

$$
\begin{pmatrix}
x'_1 \\ x'_2
\end{pmatrix}
=
R(\theta)
\begin{pmatrix}
x_1 \\ x_2
\end{pmatrix}
$$

---

## 三、RoPE 的多维旋转系统

将 $d$ 维向量分成 $d/2$ 个二维平面后，
整体旋转矩阵为：

$$
R_p =
\operatorname{blockdiag}\big( R(\theta_{p,1}), R(\theta_{p,2}), \dots, R(\theta_{p,d/2}) \big)
$$

其中每个子块角度为：

$$
\theta_{p,k} = \frac{p}{10000^{2k/d}}
$$

于是带位置信息的向量为：

$$
\mathbf{x}_p = R_p \mathbf{x}
$$

---

## 四、每个子平面的具体计算

对于第 $k$ 个二维分量 $(x_{2k-1}, x_{2k})$：

$$
\begin{aligned}
x'_{2k-1} &= x_{2k-1} \cos\theta_{p,k} - x_{2k} \sin\theta_{p,k}, \\
x'_{2k}   &= x_{2k-1} \sin\theta_{p,k} + x_{2k} \cos\theta_{p,k}.
\end{aligned}
$$

这就是在每个局部二维平面上"旋转一个角度"的过程。

---

## 五、复数等价形式

若将每对维度看作复数：

$$
z_k = x_{2k-1} + i x_{2k},
$$

则旋转可简化为复数乘法：

$$
z'_k = z_k \, e^{i \theta_{p,k}}.
$$

这种形式在理论上与矩阵乘法完全等价，但在实现上更简洁。

---

## 六、RoPE 的高效实现（Elementwise 实现）

在实践中，不直接构造稀疏矩阵 $R_p$，
而是使用**逐元素乘法（elementwise multiply）**实现旋转，效率更高：

1. 预先计算两个向量：
   $$
   \mathbf{c}_p = [\cos\theta_{p,1}, \cos\theta_{p,2}, \dots, \cos\theta_{p,d/2}],
   $$
   $$
   \mathbf{s}_p = [\sin\theta_{p,1}, \sin\theta_{p,2}, \dots, \sin\theta_{p,d/2}].
   $$

2. 将 $\mathbf{x}$ 按奇偶位分开：
   $$
   \mathbf{x}_{\text{even}} = [x_1, x_3, \dots], \quad
   \mathbf{x}_{\text{odd}} = [x_2, x_4, \dots].
   $$

3. 使用逐元素计算：
   $$
   \mathbf{x}'_{\text{even}} = \mathbf{x}_{\text{even}} \cdot \mathbf{c}_p - \mathbf{x}_{\text{odd}} \cdot \mathbf{s}_p,
   $$
   $$
   \mathbf{x}'_{\text{odd}} = \mathbf{x}_{\text{even}} \cdot \mathbf{s}_p + \mathbf{x}_{\text{odd}} \cdot \mathbf{c}_p.
   $$

4. 最后拼接还原：
   $$
   \mathbf{x}' = \operatorname{concat}(\mathbf{x}'_{\text{even}}, \mathbf{x}'_{\text{odd}}).
   $$

这样就实现了与旋转矩阵完全等价的操作，但无需矩阵乘法，计算复杂度从 $O(d^2)$ 降为 $O(d)$。

---

## 七、频率的控制

每个二维平面的旋转频率为：

$$
\omega_k = \frac{1}{10000^{2k/d}},
\quad \theta_{p,k} = p \cdot \omega_k.
$$

* 较小的 $k$（前层维度） → 高频旋转 → 捕捉局部变化；
* 较大的 $k$（后层维度） → 低频旋转 → 捕捉长程依赖。

可扩展版本会引入缩放或学习参数：
$$
\theta_{p,k} = \alpha \, p \, / \, 10000^{2k/d}, \quad \text{或} \quad \omega_k = \text{learnable}.
$$

---

## 八、RoPE 的相对位置性质

RoPE 的一个核心性质是：

> “相对位置可以通过旋转矩阵的乘法来组合，而不需要显式计算绝对位置差。”

数学上，这意味着：

$$
R_{p+q} = R_p R_q
$$

我们来**证明**它。

### ✳️ 对二维子空间：

令 $R_\alpha$ 和 $R_\beta$ 分别为两个角度的旋转矩阵：

$$
R_\alpha R_\beta
=
\left(
\begin{array}{cc}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{array}
\right)
\left(
\begin{array}{cc}
\cos\beta & -\sin\beta \\
\sin\beta & \cos\beta
\end{array}
\right)
=
\left(
\begin{array}{cc}
\cos(\alpha+\beta) & -\sin(\alpha+\beta) \\
\sin(\alpha+\beta) & \cos(\alpha+\beta)
\end{array}
\right)
= R_{\alpha+\beta}
$$


✅ 因此，二维旋转矩阵具有**角度可加性**（群性质）：

$$
R_{\alpha+\beta} = R_\alpha R_\beta
$$

这个性质在 RoPE 的高维情况下仍然成立，因为高维的 $R_p$ 是由这些二维旋转块组成的“直积”。

点积定义为：
$$
\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a}^\top \mathbf{b}
$$

代入 $\mathbf{a} = R_p \mathbf{q}$，$\mathbf{b} = R_q \mathbf{k}$：

$$
\langle \mathbf{a}, \mathbf{b} \rangle
= (R_p \mathbf{q})^\top (R_q \mathbf{k})
= \mathbf{q}^\top R_p^\top R_q \mathbf{k}
$$

而旋转矩阵满足正交性：
$$
R_p^\top = R_p^{-1} = R_{-p}
$$

因此：

$$
R_p^\top R_q = R_{-p} R_q = R_{q-p}
$$

这说明：

> RoPE 编码下，两个位置 $p$ 和 $q$ 的相对位置差 $q-p$ 自然地通过矩阵乘法体现出来。


---

## 九、总结要点表

| 概念     | 数学形式                                                                                         | 含义      |
| ------ | -------------------------------------------------------------------------------------------- | ------- |
| 局部旋转矩阵 | $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ | 二维平面旋转  |
| 旋转角度   | $\theta_{p,k} = p / 10000^{2k/d}$                                                            | 控制频率    |
| 高维旋转矩阵 | $R_p = \operatorname{blockdiag}(R(\theta_{p,1}),\dots)$                                      | 多平面旋转系统 |
| 高效实现   | 逐元素乘法计算 $\cos\theta, \sin\theta$                                                             | 避免矩阵运算  |
| 相对位置特性 | $R_p^\top R_q = R_{p-q}$                                                                     | 位置差自然编码 |
