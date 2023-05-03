#include "blitz/array.h"
#include <map>
#include <string>
#include "weights.h"
#include "helpers.h"
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm> // sort
#include <fstream> // std::ifstream
#include <iostream> // std::cout

#ifdef _OPENMP
#include <omp.h>
#endif

void assign(
    blitz::Array<float, 2> particles, 
    blitz::Array<float, 3> grid,
    std::string method) 
{
    std::cout << "Assigning " << particles.extent(0) << " particles to a " << grid.extent(0) << "x" << grid.extent(1) << "x" << grid.extent(2) << " grid" << std::endl;

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
    for (int i=particles.lbound(0); i< particles.extent(0) + particles.lbound(0); ++i) {
        float x = (particles(i,0) + 0.5) * grid.extent(0);
        float y = (particles(i,1) + 0.5) * grid.extent(1);
        float z = (particles(i,2) + 0.5) * grid.extent(2);

        //std::cout << "Particle " << i << " at " << x << ", " << y << ", " << z << std::endl;
        //std::cout << "from " << i << " at " << particles(i,0) << ", " << particles(i,1) << ", " << particles(i,2) << std::endl;

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
                    /*grid(
                        (startX + j + grid.extent(0)) % grid.extent(1), 
                        (startY + k + grid.extent(1)) % grid.extent(2), 
                        (startZ + l + grid.extent(2)) % grid.extent(3)) 
                        += weight * 1.0f;*/

                    grid (
                        startX + j,
                        startY + k,
                        startZ + l) += weight * 1.0f;
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
    blitz::Array<float, 1> fPower,
    blitz::Array<int, 1> nPower,
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

void sortParticles(blitz::Array<float, 2> particles) {
    struct Particle {
        float x;
        float y;
        float z;
    };

    Particle* particle_Object = reinterpret_cast<Particle*>(particles.data());

    std::sort(
        particle_Object, 
        particle_Object + particles.rows(),
        [&](Particle a, Particle b){ 
            return a.x < b.x;
        }
    );
}