#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED

#include "blitz/array.h"
#include <string>

void assign(
    blitz::Array<float, 2> particles, 
    blitz::Array<float, 3> grid,
    blitz::TinyVector<int, 3> grid_size,
    std::string method);

void project(
    blitz::Array<float, 3> grid_3d,
    blitz::Array<float, 2> grid_2d);

void bin(
    blitz::Array<std::complex<float>, 3> grid,
    blitz::Array<float, 1> fPower,
    blitz::Array<int, 1> nPower,
    int nBins,
    bool log);

int getK(
    int x, int y, int z);

int getIndex (int k, int kmax, int nBins, bool log);

void sortParticles(blitz::Array<float, 2> particles);

#endif // HELPERS_H_INCLUDED

