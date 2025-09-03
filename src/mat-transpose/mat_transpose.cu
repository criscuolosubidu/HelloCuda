#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// fp32
// col2row: read x[row][col] and write to y[col][row]
// row2col: read x[col][row] and write to y[row][col]
// 矩阵转置
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y, const int row, const int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = idx / col, c = idx % col;
    if (idx < row * col) {
        y[c * row + r] = x[idx];
    }
}

__global__ void mat_transpose_f32_row2col_kernel(float *x, float *y, const int row, const int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = idx / row, c = idx % row;
    if (idx < row * col) {
        y[idx] = x[r * col + c];
    }
}

// f32x4
// col2row
__global__ void mat_transpose_f32x4_col2row_kernel(float *x, float *y, const int row, const int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = 4 * idx / col, c = 4 * idx % col;
    if (r < row && c + 3 < col) {
        float4 reg_x = reinterpret_cast<float4 *>(x)[idx]; // sizeof(float4)，这个写法真是简单啊
        // [r, c], [r, c + 1], [r, c + 2], [r, c + 3]
        y[c * row + r] = reg_x.x;
        y[(c + 1) * row + c + r] = reg_x.y;
        y[(c + 2) * row + c + r] = reg_x.z;
        y[(c + 3) * row + c + r] = reg_x.w;
    }
}

// f32x4
// row2col
__global__ void mat_transpose_f32x4_row2col_kernel(float *x, float *y, const int row, const int col) {
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    // c * row + r
    const int c = idx / row, r = idx % row;
    if (r < row && c < col) {
        float4 reg_x;
        reg_x.x = x[r * col + c];
        reg_x.y = x[(r + 1) * col + c];
        reg_x.z = x[(r + 2) * col + c];
        reg_x.w = x[(r + 3) * col + c];
        FLOAT4(y[idx]) = reg_x;
    }
}


// work for row == col ，这种情况下优化效果最为显著
// block(i, j), read x[i,j] and write y[j,i]
// block(j, i), read x[j,i] and write y[i,j]
// x[j,i] and y[j,i] or x[i,j] and y[i,j] has a high probability at the same memory tile (due to mod operation)
// row2col 方式
__global__ void mat_transpose_f32_diagonal2d_kernel(float *x, float *y, const int row, const int col) {
    const int block_y = blockIdx.x;
    // (x, y) -> (y, x) -> (y, (x + y) % M) , 首先转置然后进行统一的偏移
    const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int r = threadIdx.x + block_x * blockDim.x, c = threadIdx.y + block_y * blockDim.y;
    if (r < row && c < col) {
        y[r * col + c] = x[c * row + r];
    }
}

// fp32
// col2row2d
__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y, const int row, const int col) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < col && r < row) {
        y[c * row + r] = x[r * col + c];
    }
}

// row2col2d
// 注意这里的block, row, col都是针对x而言的，所以其实本质上都要按照x来写
__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y, const int row, const int col) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < row && r < col) {
        y[r * row + c] = x[c * col + r];
    }
}

// col2row2d
// f32x4
// 表达式本身是 r * col + c，然后就是变为了每一行，其实本质上只有 col / 4 个 而不是 col 个了
// grid(16, 16) -> grid(16, 4)
__global__ void mat_transpose_f32x4_col2row2d_kernel(float *x, float *y, const int row, const int col) {
    // 按照列方向四个四个读取
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < row && c * 4 + 3 < col) {
        // r * col + c * 4，本质上对应的地址，但是按照 float4 来读取的话，那么一行只有 col/4 个 float4 了
        // 扁平化之后也可以直接理解为除以4步长
        float4 reg_x = reinterpret_cast<float4 *>(x)[r * col / 4 + c];
        y[c * 4 * row + r] = reg_x.x;
        y[(c * 4 + 1) * row + r] = reg_x.y;
        y[(c * 4 + 2) * row + r] = reg_x.z;
        y[(c * 4 + 3) * row + r] = reg_x.w;
    }
}

__global__ void mat_transpose_f32x4_row2col2d_kernel(float *x, float *y, const int row, const int col) {
    // 按照y矩阵的列方向连续存4个
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c * 4 + 3 < row && r < col) {
        // r * row + c * 4 -> c * 4 * col + r
        float4 reg_x;
        reg_x.x = x[c * 4 * col + r];
        reg_x.y = x[(c * 4 + 1) * col + r];
        reg_x.z = x[(c * 4 + 2) * col + r];
        reg_x.w = x[(c * 4 + 3) * col + r];
        reinterpret_cast<float4*>(y)[r * row / 4 + c] = reg_x;
    }
}

// 内存层次结构：按照速度从快到慢
// SM中的寄存器 -> shared memory(shared in 1 block) -> L1/L2 cache -> Constant Memory(read-only, constant data), Texture Memory(optimize for 2d/3d)
// -> global memory(slow but large) -> local memory(a part of global memory)

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float *x, float *y, const int row, const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.x + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    // 在共享内存中声明一个 tile 数组
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4]; // 32 * 128 = 32 * (4 * 32) 刚好可以装下 32 * 32 的 float4 类型的 2d网格
    if (global_x * 4 + 3 < col && global_y < row) {
        float4 x_val = reinterpret_cast<float4*>(x)[global_y * col /4 + global_x];
        FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);

        // 确保同一个块内的所有的线程的数据都加载到了 tile 上面
        __syncthreads();

        float4 smem_val;
        constexpr int STRIDE = WARP_SIZE_S / 4;
        // 分组，然后每个程序负责一小列
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y =
            tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z =
            tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w =
            tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];

        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
    if (global_y * 4 < row && global_x < col) {
        // load value from x to shared memory
        float4 x_val;
        x_val.x = x[(global_y * 4) * col + global_x];
        x_val.y = x[(global_y * 4 + 1) * col + global_x];
        x_val.z = x[(global_y * 4 + 2) * col + global_x];
        x_val.w = x[(global_y * 4 + 3) * col + global_x];
        tile[local_y * 4][local_x] = x_val.x;
        tile[local_y * 4 + 1][local_x] = x_val.y;
        tile[local_y * 4 + 2][local_x] = x_val.z;
        tile[local_y * 4 + 3][local_x] = x_val.w;
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        // map index n*n to (n/4)*(n*4)
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
        smem_val.y =
            tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
        smem_val.z =
            tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
        smem_val.w =
            tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
        const int bid_x = blockIdx.x * blockDim.x;
        const int bid_y = blockIdx.y * blockDim.y;

        const int out_y = bid_x + (local_y % STRIDE) * 4;
        const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
        y[out_y * row + out_x] = smem_val.x;
        y[(out_y + 1) * row + out_x] = smem_val.y;
        y[(out_y + 2) * row + out_x] = smem_val.z;
        y[(out_y + 3) * row + out_x] = smem_val.w;
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];  // 巧妙的引入了一个offset，避免bank冲突
    if (global_x * 4 + 3 < col + 3 && global_y < row) {
        // load value from x to shared memory
        float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];
        tile[local_y][local_x * 4] = x_val.x;
        tile[local_y][local_x * 4 + 1] = x_val.y;
        tile[local_y][local_x * 4 + 2] = x_val.z;
        tile[local_y][local_x * 4 + 3] = x_val.w;
        __syncthreads();
        float4 smem_val;
        // load value from shared memory to y.
        // add STRIDE to satisfied different block size.
        constexpr int STRIDE = WARP_SIZE_S / 4;
        smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
        smem_val.y =
            tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
        smem_val.z =
            tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
        smem_val.w =
            tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
        // map index n*n to (n/4)*(n*4)
        const int bid_y = blockIdx.y * blockDim.y;
        const int out_y = global_x * 4 + local_y / STRIDE;
        const int out_x = (local_y % STRIDE) * 4 + bid_y;
        reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
    }
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
        tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}