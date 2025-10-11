#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <thread>


int main() {
    std::cout << "Test CPU MultiThreads Rand Numbers Generate!" << std::endl;
    constexpr int N = 1073741824; // 2 ^ 30

    std::vector<float> h_x(N);

    auto start_gen = std::chrono::high_resolution_clock::now();

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    const int chunk_size = N / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&h_x, t, chunk_size, num_threads, N]() {
            std::mt19937 gen(std::random_device{}() + t);
            std::uniform_real_distribution<float> dist(0, 1);

            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? N : (t + 1) * chunk_size;

            for (int i = start; i < end; ++i) {
                h_x[i] = dist(gen);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

    std::cout << "CPU Random Number Generate Time : " << gen_time << " ms" << std::endl; // 1218.44 ms
}