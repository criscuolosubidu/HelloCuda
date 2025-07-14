cmake_minimum_required(VERSION 3.27)  # 建议升级到 3.27 或更高
project(hello_cuda LANGUAGES CXX CUDA)

file(GLOB CUDA_KERNELS "src/kernels/*.cu")
file(GLOB CUDA_BASICS  "src/cuda-basics/*.cu")
add_executable(
        hello_cuda
        src/main.cpp
        ${CUDA_KERNELS}
        ${CUDA_BASICS}
        src/kernels/kernels.h
        src/cuda-basics/CudaBasics.h
        src/kernels/cuda_memory.h
)

add_compile_options(/std:c++20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 -Wall -Xcompiler")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_target_properties(hello_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(hello_cuda PRIVATE cudart cublas)