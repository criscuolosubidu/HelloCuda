//
// Created by ubuntu on 2025/6/18.
//

#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

template <typename T>
class CudaMemory
{
public:
    explicit CudaMemory(size_t count);

    ~CudaMemory();

    T* get() const
    {
        return data_;
    }

    size_t size() const
    {
        return count_;
    }

    // 禁止拷贝和复制
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // 允许移动
    CudaMemory(CudaMemory&& other) noexcept;
    CudaMemory& operator=(CudaMemory&& other) noexcept;

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

#endif //CUDA_MEMORY_H