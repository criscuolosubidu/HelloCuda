cmake_minimum_required(VERSION 3.27)
project(hello_cuda LANGUAGES CXX CUDA)

set(CUDA_SOURCE_DIRS
    kernels
    cuda-basics
    elementwise
    histogram
    sigmoid
    relu
    gelu
    swish
    hardshrink
    embedding
    mat-transpose
)

set(CUDA_SOURCES "")

foreach(DIR ${CUDA_SOURCE_DIRS})
    file(GLOB_RECURSE DIR_SOURCES "src/${DIR}/*.cu")
    list(APPEND CUDA_SOURCES ${DIR_SOURCES})
endforeach()


add_executable(
    hello_cuda
    src/main.cpp
    ${CUDA_SOURCES}
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 标准设置
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")  不推荐使用这种全局的方式

if(MSVC)
    target_compile_options(hello_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler,/Wall>)
else()
    target_compile_options(hello_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler,-Wall>)
endif()


set_target_properties(hello_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(hello_cuda PRIVATE cudart cublas)