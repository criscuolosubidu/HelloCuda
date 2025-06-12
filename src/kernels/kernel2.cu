#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel2()
{
    printf("Hello World from kernel2!\n");
}

void launch_kernel2()
{
    kernel2<<<32,32>>>();
    cudaDeviceSynchronize();
}