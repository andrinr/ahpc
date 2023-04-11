#include "ptimer.h"

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
    lap_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2-t).count());
    t = t2;
}

void PTimer::reset() {
    lap_names.clear();
    lap_times.clear();
}