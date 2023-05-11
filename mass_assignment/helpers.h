#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED

#include "blitz/array.h"
#include <string>

class WrapX {
public:
    int n0;
    std::string method;
    std::map<std::string, int> range;
    typedef int (*kernel)(float, float*);
    std::map<std::string, kernel> kernels;

    float * tmp;

    WrapX(int n0, std::string method);

    float wrap(float x);
};

void assign(
    blitz::Array<float, 2> particles, 
    blitz::Array<float, 3> grid,
    blitz::TinyVector<int, 3> grid_size,
    std::string method,
    int rank,
    int np);

void project(
    blitz::Array<float, 3> grid_3d,
    blitz::Array<float, 2> grid_2d);

void bin(
    blitz::Array<std::complex<float>, 3> grid,
    blitz::TinyVector<int, 3> grid_size,
    blitz::Array<float, 1> fPower,
    blitz::Array<int, 1> nPower,
    int nBins,
    bool log);

int getK(
    int x, int y, int z);

int getIndex (int k, int kmax, int nBins, bool log);

void sortParticles(blitz::Array<float, 2> particles, int n0, std::string method);

blitz::Array<float, 2> reshuffleParticles (
    blitz::Array<float, 2> particlesLocallySorted, 
    int slabStart, int nSlabs, 
    std::string method,
    int nGrid, int rank, int np);

blitz::Array<float, 2> loadParticles(std::string location, int rank, int np);

#endif // HELPERS_H_INCLUDED

