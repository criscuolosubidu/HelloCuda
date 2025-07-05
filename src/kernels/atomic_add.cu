#include <cuda_runtime.h>
#include <iostream>

static constexpr int NUM_THREADS = 1024;
static constexpr int NUM_BLOCKS = 1024;

__global__ void incrementCounterNonAtomic(int* counter)
{
    int old = *counter;
    *counter = old + 1;
}

__global__ void incrementCounterAtomic(int* counter)
{
    atomicAdd(counter, 1);
}

void compareIncrementCounter()
{
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtmoic;

    cudaMalloc(reinterpret_cast<void**>(&d_counterNonAtomic), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_counterAtmoic), sizeof(int));

    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtmoic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtmoic);

    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtmoic, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Non-atomic counter value : " << h_counterNonAtomic << std::endl;
    std::cout << "Atomic counter value : " << h_counterAtomic << std::endl;

    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtmoic);
}

