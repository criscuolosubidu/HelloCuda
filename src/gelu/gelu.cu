#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// ===================================================================================
// 1. 内存操作宏 (Memory Operation Macros)
//
// 核心是 `reinterpret_cast`，它是一种类型强转，告诉编译器“将这块内存中的
// 二进制数据当作另一种类型来处理”。在 GPU 编程中，这通常是为了利用128位（16字节）
// 的内存加载/存储指令，从而最大化内存带宽，提升性能。
// 一次性读写更多数据远比多次读写零散数据要快。
// ===================================================================================

#define WARP_SIZE 32
/**
 * @brief 将一个地址重新解释为 int4* 类型的指针，并解引用获取这个 int4 值。
 * `int4` 是一个包含4个int成员的向量类型。
 * 目的是一次性加载 4 * sizeof(int) = 16 字节（128位）的数据。
 */
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

/**
 * @brief 同上，但目标类型是 float4（4个 float），同样是为了128位内存操作。
 * 目的是一次性加载 4 * sizeof(float) = 16 字节（128位）的数据。
 */
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * @brief 目标类型是 half2（2个 half）。half 类型是16位浮点数。
 * 目的是一次性加载 2 * sizeof(half) = 4 字节（32位）的数据。
 */
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

/**
 * @brief 目标类型是 __nv_bfloat162（2个 bfloat16）。bfloat16 也是16位浮点数。
 * 目的是一次性加载 2 * sizeof(__nv_bfloat16) = 4 字节（32位）的数据。
 */
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

/**
 * @brief 一个更明确的别名，直接表明其意图是加载/存储128位数据。
 * 底层实现和 FLOAT4 完全一样。
 */
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


// ===================================================================================
// 2. 指数函数边界常量 (Exponential Function Boundary Constants)
//
// 在计算 exp(x) 时，如果 x 过大或过小，结果会**上溢**（变成无穷大 `Inf`）或
// **下溢**（变成 0 或非规格化数），这可能导致后续计算出现 `NaN` (Not a Number) 等问题。
// 因此，通常需要将 x 的值限制在一个安全范围内。
// ===================================================================================

/**
 * @brief 32位浮点数 (float) 进行 expf(x) 计算时的安全上限。
 * `expf(88.37626)` 约等于 3.4e38，接近 float 类型的最大可表示值 (FLT_MAX)。
 * 超出此值会导致上溢。
 */
#define MAX_EXP_F32 88.3762626647949f

/**
 * @brief 32位浮点数 (float) 进行 expf(x) 计算时的安全下限。
 * `expf(-88.37626)` 的结果非常接近于0。
 * 使用此下限可以避免产生非规格化数 (denormalized numbers)，从而保持计算性能。
 */
#define MIN_EXP_F32 -88.3762626647949f

/**
 * @brief 16位半精度浮点数 (half) 进行 h2exp(x) 计算时的安全上限。
 * `__float2half` 是一个 CUDA 内置函数，将 float 转为 half。
 * 11.089... 约等于 ln(65504)，而 65504 是 half 类型的最大值。
 */
#define MAX_EXP_F16 __float2half(11.089866488461016f)

/**
 * @brief 16位半精度浮点数 (half) 进行 h2exp(x) 计算时的安全下限。
 * -9.704... 约等于 ln(2^-14)，而 2^-14 (约 6.1e-5) 是 half 类型能表示的最小正规格化数。
 * 使用此下限可以避免下溢到非规格化数。
 */
#define MIN_EXP_F16 __float2half(-9.704060527839234f)


// ===================================================================================
// 3. 数学与GELU激活函数常量 (Mathematical & GELU Activation Constants)
//
// 这些常量主要用于实现 GELU (Gaussian Error Linear Unit) 激活函数，
// 特别是它的 tanh 近似版本，这在 Transformer 模型中非常常见。
// GELU 的 tanh 近似公式为：
// GELU(x) ≈ 0.5x * (1 + tanh[sqrt(2/π) * (x + 0.044715 * x^3)])
// ===================================================================================

// CUDA math.h 中定义的常量
// #define M_SQRT2    1.41421356237309504880 // sqrt(2)
// #define M_2_SQRTPI 1.12837916709551257390 // 2/sqrt(pi)

/**
 * @brief 计算 GELU 公式中的常量 sqrt(2/π) ≈ 0.79788。
 * @note 【注意】这里的原始注释和代码实现不完全匹配。
 * 代码 M_SQRT2 * M_2_SQRTPI * 0.5f 计算的是 sqrt(2) * (2/sqrt(pi)) * 0.5 = sqrt(2/pi)。
 * 而原始注释中描述的 sqrt(2*pi)/2 计算的是 sqrt(pi/2)。
 * GELU 函数中使用的是 sqrt(2/pi)，所以代码实现是正确的。
 */
#define M_2_SQRTPI	1.12837916709551257390
#define M_SQRT2		1.41421356237309504880
#define SQRT_2_PI M_SQRT2 * M_2_SQRTPI * 0.5f

/**
 * @brief 将常用的浮点数 1.0, 2.0, 0.5 预先转换为 half 类型。
 * 在 kernel 中直接使用这些常量可以避免运行时的类型转换开销。
 */
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)

/**
 * @brief 这是上面 SQRT_2_PI 的 half 精度版本，用于 half 类型的计算。
 * 它将构成 sqrt(2/pi) 的各个部分先转为 half，再进行乘法。
 * 目的是为了在全半精度计算流程中保持一致的精度。
 */
#define HALF_SQRT_2_PI                                                         \
    __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2

/**
 * @brief 这是 GELU tanh 近似公式中 x^3 项的系数 0.044715 的 half 精度版本。
 */
#define HALF_V_APP __float2half(0.044715f)


// ===================================================================================
// 4. 实现别名 (Implementation Aliases)
//
// 这是一种常见的软件工程实践，通过宏定义来选择使用哪个具体的函数实现。
// 这样做的好处是，如果将来想切换 GELU 的实现（例如，从 `tanh` 近似版
// 换成更精确的 `erf` 版），只需修改这两行宏定义即可，而无需改动调用它的代码。
// ===================================================================================

/**
 * @brief 定义一个别名，使得代码中调用的 HALF_GELU_OPS 实际上是在调用 gelu_tanh_approximate 函数。
 * 这通常用于 half 类型的计算。
 */
#define HALF_GELU_OPS gelu_tanh_approximate

/**
 * @brief 同上，用于 float 类型的计算。
 */
#define GELU_OPS gelu_tanh_approximate

// 没有原生的半精度的计算函数 sinh, cosh, tanh
// $$ tanh(x) = \frac{exp^{2x} - 1}{exp^{2x} + 1}$$
// 直接使用tanh的数学公式进行计算也是不可行的，需要按照pytorch的方法，先在更大的精度中计算，然后在downcast
__inline__ __device__ half gelu_tanh_approximate(half x) {
    half x_cube = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
    return HALF_DIV2 * x * (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));
}

__inline__ __device__ float gelu_tanh_approximate(float x) {
    return 0.5f * x * (1.0f + tanh(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

__inline__ __device__ float gelu_none_approximate(float x) {
    return x * 0.5 * (1 + erff(x * M_2_SQRTPI)); // 使用误差函数 erf 计算，没有近似
}

// FP32
// GELU tanh approximate: x, y:x 0.5 * x
// * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))) grid(N/256),
// block(K=256)
__global__ void gelu_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
        y[idx] = GELU_OPS(v);
    }
}

// grid(N/256), block(256/4)
__global__ void gelu_f32x4_kernel(float *x, float *y, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
        reg_x.z = fminf(fmaxf(reg_x.z, MAX_EXP_F32), MAX_EXP_F32);
        reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);
        reg_y.x = GELU_OPS(reg_x.x);
        reg_y.y = GELU_OPS(reg_x.y);
        reg_y.z = GELU_OPS(reg_x.z);
        reg_y.w = GELU_OPS(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// fp16
__global__ void gelu_fp16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        half v = __hmin(__hmax(x[idx], MIN_EXP_F16), MAX_EXP_F16);
        y[idx] = HALF_GELU_OPS(v);
    }
}

__global__ void gelu_fp16x2_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_x.x = __hmin(__hmax(reg_x.x, MIN_EXP_F16), MAX_EXP_F16);
        reg_x.y = __hmin(__hmax(reg_x.y, MIN_EXP_F16), MAX_EXP_F16);
        reg_y.x = HALF_GELU_OPS(reg_x.x);
        reg_y.y = HALF_GELU_OPS(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

// unpack f16x8
__global__ void gelu_f16x8_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);

    reg_x_0.x = __hmin(__hmax(reg_x_0.x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_0.y = __hmin(__hmax(reg_x_0.y, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_1.x = __hmin(__hmax(reg_x_1.x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_1.y = __hmin(__hmax(reg_x_1.y, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_2.x = __hmin(__hmax(reg_x_2.x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_2.y = __hmin(__hmax(reg_x_2.y, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_3.x = __hmin(__hmax(reg_x_3.x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x_3.y = __hmin(__hmax(reg_x_3.y, MIN_EXP_F16), MAX_EXP_F16);

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

    reg_x_0.x = HALF_GELU_OPS(reg_x_0.x);
    reg_x_0.y = HALF_GELU_OPS(reg_x_0.y);
    reg_x_1.x = HALF_GELU_OPS(reg_x_1.x);
    reg_x_1.y = HALF_GELU_OPS(reg_x_1.y);
    reg_x_2.x = HALF_GELU_OPS(reg_x_2.x);
    reg_x_2.y = HALF_GELU_OPS(reg_x_2.y);
    reg_x_3.x = HALF_GELU_OPS(reg_x_3.x);
    reg_x_3.y = HALF_GELU_OPS(reg_x_3.y);

    if ((idx + 0) < N) {
        HALF2(y[idx + 0]) = reg_x_0;
    }
    if ((idx + 2) < N) {
        HALF2(y[idx + 2]) = reg_x_1;
    }
    if ((idx + 4) < N) {
        HALF2(y[idx + 4]) = reg_x_2;
    }
    if ((idx + 6) < N) {
        HALF2(y[idx + 6]) = reg_x_3;
    }
}

// pack fp16x8
__global__ void gelu_fp16x8_pack_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    half reg_x[8], reg_y[8];
    LDST128BITS(reg_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        half2 x_pack = HALF2(reg_x[i]);
        half2 y_pack;
        x_pack.x = __hmin(__hmax(x_pack.x, MIN_EXP_F16), MAX_EXP_F16);
        x_pack.y = __hmin(__hmax(x_pack.y, MIN_EXP_F16), MAX_EXP_F16);
        y_pack.x = HALF_GELU_OPS(x_pack.x);
        y_pack.y = HALF_GELU_OPS(x_pack.y);
        HALF2(reg_y[i]) = y_pack;
    }
    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(reg_y[0]);
    }
}


