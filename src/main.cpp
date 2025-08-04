#include <iostream>
#include "kernels/kernels.h"
#include "cuda-basics/CudaBasics.h"


int main()
{
    int a = 123;
    *(&a) = 5;
    std::cout << a << std::endl;
}
