#include "blitz/array.h"
#include <map>
#include <string>
#include "weights.h"
#include "helpers.h"
#include <cmath>
#include <stdlib.h>
#include "blitz/array.h"
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void assign(
    blitz::Array<float, 2> particles, 
    blitz::Array<float, 3> grid,
    std::string method) 
{
    // Get size of grid
    int nGrid = grid.extent(0);

    // Get number of particles
    int N = particles.extent(0);

    std::map<std::string, int> range = {
        { "ngp", 1 },
        { "cic", 2 },
        { "tsc", 3 },
        { "pcs", 4 }
    };

    typedef int (*kernel)(float, float*);
    std::map<std::string, kernel> kernels = {
        { "ngp", &ngp_weights },
        { "cic", &cic_weights },
        { "tsc", &tsc_weights },
        { "pcs", &pcs_weights }
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=particles.lbound(0); i<N + particles.lbound(0); ++i) {
        float x = (particles(i,0) + 0.5) * nGrid;
        float y = (particles(i,1) + 0.5) * nGrid;
        float z = (particles(i,2) + 0.5) * nGrid;

        float* weightsX = new float[range[method]];
        float* weightsY = new float[range[method]];
        float* weightsZ = new float[range[method]];

        int startX = kernels[method](x, weightsX);
        int startY = kernels[method](y, weightsY);
        int startZ = kernels[method](z, weightsZ);

        for (int j=0; j<range[method]; ++j) {
            for (int k=0; k<range[method]; ++k) {
                for (int l=0; l<range[method]; ++l) {

                    float weight = weightsX[j] * weightsY[k] * weightsZ[l];

                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    grid(
                        (startX + j + nGrid) % nGrid, 
                        (startY + k + nGrid) % nGrid, 
                        (startZ + l + nGrid) % nGrid) 
                        += weight * 1.0f;
                }
            }
        }
    }
}

void project(
    blitz::Array<float, 3> grid_3d,
    blitz::Array<float, 2> grid_2d)
{
    // Get size of grid
    int nGrid = grid_3d.extent(0);

    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int k=0; k<nGrid; ++k) {
                grid_2d(i,j) = std::max(grid_3d(i,j,k), grid_3d(i,j));
            }
        }
    }
}

void bin(
    blitz::Array<std::complex<float>, 3> grid,
    blitz::Array<float, 1> bins,
    int nBins,
    bool log)
{
    int nGrid = grid.extent(0);
    blitz::Array<float, 1> kx(nGrid);
    blitz::Array<float, 1> ky(nGrid);
    blitz::Array<float, 1> kz(nGrid/2 + 1);

    for (int i = 0; i <= nGrid/2; i++) {
        kx(i) = i;
        ky(i) = i;
        kz(i) = i;
    }

    for (int i = 0; i < nGrid/2-1; i++) {
        kx(i+nGrid/2+1) = -nGrid/2 + i + 1;
        ky(i+nGrid/2+1) = -nGrid/2 + i + 1;
    }
    
    int kMax = getK(nGrid/2, nGrid/2, nGrid/2);
    std::cout << "kMax: " << kMax << std::endl;

    blitz::Array<float, 1> fPower(nBins);
    fPower = 0;
    blitz::Array<int, 1> nPower(nBins);
    nPower = 0;

    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int l=0; l<nGrid/2; ++l) {
                int k = getK(kx(i), ky(j), kz(l));
                int index = getIndex(k, kMax, nBins, log);

                if (index >= nBins) {
                    continue;
                }

                fPower(index) += std::norm(grid(i, j, l));
                nPower(index) += 1;
            }
        }
    }

    std::cout << "FPower: " << fPower << std::endl;
    std::cout << "NPower: " << nPower << std::endl;

    for(int i=0; i<nBins; ++i) {
        bins(i) = fPower(i) / nPower(i);
    }
}

int getK(int x, int y, int z) {
    return int(std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2)));
}

int getIndex(int k, int kmax, int nBins, bool log) {
    if (log) {
        return int(std::log((float)k) / std::log((float) kmax) * nBins);
    } else {
        return int((float)k / kmax * nBins);
    }
}