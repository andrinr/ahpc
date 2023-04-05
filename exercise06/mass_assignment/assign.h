#include "blitz/array.h"
#include <map>
#include <string>
#include "weights.h"

inline void assign(
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
    for (int i=0; i<N; ++i) {
        float x = (r(i,0) + 0.5) * nGrid;
        float y = (r(i,1) + 0.5) * nGrid;
        float z = (r(i,2) + 0.5) * nGrid;

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