#include <iostream>
#include <cuda_runtime.h>


__global__ void kernel1()
{
    printf("hello world from kernel1!\n");
}

void launch_kernel1()
{
    kernel1<<<32,32>>>();
    cudaDeviceSynchronize();
}