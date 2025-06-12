#include <iostream>
#include "kernels/kernels.h"
#include "cuda-basics/CudaBasics.h"


int main()
{
    std::cout << "Launching CUDA kernels..." << std::endl;
    std::cout << "Warming up..." << std::endl;
    launch_kernel1();
    launch_kernel2();
    std::cout << "Warming up finish!" << std::endl;

    std::cout << "run cuda basics of 01_idxing.cu ..." << std::endl;
    cuda_basics_01_idxing();
    std::cout << "run cuda basics of 01_idxing.cu is finish!" << std::endl;

    return 0;
}
