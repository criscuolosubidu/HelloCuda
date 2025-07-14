#include <iostream>
#include "kernels/kernels.h"
#include "cuda-basics/CudaBasics.h"


int main()
{
    auto result = streamBasicsDemo();
    if (result != EXIT_SUCCESS) {
        std::cout << "Stream BasicsDemo failed with error. " << std::endl;
    }
    std::cout << __cplusplus << std::endl;
    return 0;
}
