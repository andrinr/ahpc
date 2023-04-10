#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <list>
#include <chrono>

class Timer {
public:
    std::list<std::string> lap_names;
    std::list<int> lap_times;

    Timer();
    void start();
    void lap(std::string name);
    void reset();

private:
    std::chrono::high_resolution_clock::time_point t;
};

#endif // TIMER_H