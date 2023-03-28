#include <cmath>

inline int ngp_weights(float x, float *W) {
    int i = std::floor(x);
    W[0] = 1.0;
    return i;
}

inline int cic_weights(float x, float *W) {
    int i = std::floor(x-0.5);
    W[0] = 0; // fix me
    W[1] = 0; // fix me
    return i;
}

inline int tsc_weights(float x, float *W) {
    int i = std::floor(x-1.0);
    W[0] = 0; // fix me
    W[1] = 0; // fix me
    W[2] = 0; // fix me
    return i;
}

inline int pcs_weights(float x, float *W) {
    int i = std::floor(x-1.5);
    W[0] = 0; // fix me
    W[1] = 0; // fix me
    W[2] = 0; // fix me
    W[3] = 0; // fix me
    return i;
}

