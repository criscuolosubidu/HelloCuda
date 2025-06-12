#include <iostream>
#include <cuda_runtime.h>

__global__ void whoami()
{
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", id, blockIdx.x, blockIdx.y, blockIdx.z, block_id, threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

void cuda_basics_01_idxing()
{
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // the max wrap size is 32

    int block_per_grid = b_x * b_y * b_z;
    int thread_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", block_per_grid);
    printf("%d threads/block\n", thread_per_block);
    printf("%d total threads\n", block_per_grid * thread_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}