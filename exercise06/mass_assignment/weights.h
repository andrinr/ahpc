#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <cmath>
#include <iostream>

inline int ngp_weights(float x, float *W) {
    int i = std::floor(x);
    W[0] = 1.0;
    return i;
}

inline int cic_weights(float x, float *W) {
    int i = std::floor(x-0.5f);
    W[0] = 1.0 - std::abs(x-i-0.5f);
    W[1] = 1.0 - std::abs(x-i-1.5f);
    return i;
}

inline int tsc_weights(float x, float *W) {
    int i = std::floor(x-1.0);
    W[0] = 1.0f/2.0f * std::pow(3.0/2.0 - std::abs(x-i-0.5f), 2);
    W[1] = 3.0f/4.0f - std::pow(x-i-1.5f, 2);
    W[2] = 1.0f/2.0f * std::pow(3.0/2.0 - std::abs(x-i-2.5f), 2);
    return i;
}

inline int pcs_weights(float x, float *W) {
    int i = std::floor(x-1.5);
    W[0] = 1.0f/6.0f * (
        std::pow(2.0f - std::abs(x-i-0.5f), 3)
    );
    W[1] = 1.0f/6.0f * (
        4.0f - 6.0f * std::pow(x-i-1.5f, 2) 
        + 3.0f * std::abs(std::pow(x-i-1.5f, 3))
    );
    W[2] = 1.0f/6.0f * (
        4.0f - 6.0f * std::pow(x-i-2.5f, 2) 
        + 3.0f * std::abs(std::pow(x-i-2.5f, 3))
    );
    W[3] = 1.0f/6.0f * (
        std::pow(2.0f - std::abs(x-i-3.5f), 3)
    );
    return i;
}

#endif // WEIGHTS_H