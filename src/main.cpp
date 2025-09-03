#include <iostream>


int main()
{
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            std::cout << "(" << i % 8 + 4 << "," << j * 4 + i / 8 << ")" << ' ';
        }
        std::cout << std::endl;
    }
}
