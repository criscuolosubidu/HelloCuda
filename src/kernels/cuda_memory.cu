#include <iostream>
#include <cuda_runtime.h>
#include "cuda_memory.h"


template <typename T>
CudaMemory<T>::CudaMemory(size_t count) : count_(count)
{
    if (count > 0)
    {
        cudaError_t err = cudaMalloc(&data_, sizeof(T) * count_);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate CUDA memory.");
        }
    }
}


template <typename T>
CudaMemory<T>::~CudaMemory()
{
    if (data_)
    {
        cudaFree(data_);
    }
}


template <typename T>
CudaMemory<T>::CudaMemory(CudaMemory&& other) noexcept : data_(other.data_), count_(other.count_)
{
    other.data_ = nullptr;
    other.count_ = 0;
}


template <typename T>
CudaMemory<T>& CudaMemory<T>::operator=(CudaMemory&& other) noexcept
{
    if (this != &other)
    {
        if (data_)
        {
            cudaFree(data_);
        }
        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
    }
    return *this;
}

template class CudaMemory<float>;
template class CudaMemory<double>;
template class CudaMemory<int>;