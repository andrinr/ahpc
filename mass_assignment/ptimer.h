#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <string>
#include <list>
#include <chrono>

class PTimer {
public:
    PTimer(int rank);
    void start();
    void lap(std::string name);
    void reset();

private:
    int rank;
    std::list<std::string> lap_names;
    std::list<int> lap_times;
    std::chrono::high_resolution_clock::time_point t;
};

#endif // TIMER_H_INCLUDED