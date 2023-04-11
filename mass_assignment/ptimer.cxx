#include "ptimer.h"
#include <iostream>

PTimer::PTimer() {
    lap_names = std::list<std::string>();
    lap_times = std::list<int>();
}

void PTimer::start() {
    reset();

    t = std::chrono::high_resolution_clock::now();
}

void PTimer::lap(std::string name) {
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    lap_names.push_back(name);
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t).count();
    std::cout << name << ": " << duration << "ms" << std::endl;
    lap_times.push_back(duration);
    t = t2;
}

void PTimer::reset() {
    lap_names.clear();
    lap_times.clear();
}