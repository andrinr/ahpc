// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
#include <chrono>
#include <string>
#include <map>
#include <stdio.h>
#include <new>
#include <fftw3.h>
#include "weights.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace blitz;

int main(int argc, char *argv[]) {
    if (argc<=1) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }

    // Load particle positions
    std::uint64_t N = io.count();
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    io.load(r);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Reading file took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

    // Create Mass Assignment Grid
    typedef std::complex<float> cplx;
    int mem_size = nGrid * nGrid * (nGrid+2);
    float *memory_in = new (std::align_val_t(64)) float[mem_size];

    TinyVector<int, 3> in_shape(nGrid, nGrid, nGrid+2);
    Array<float,3> in(memory_in, in_shape, deleteDataWhenDone);
    Array<float,3> in_no_pad = in(Range::all(), Range::all(), Range(0,nGrid));

    std::cerr << "Mass assignment method: " << argv[3] << std::endl;

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

    std::string m = (std::string) argv[3];

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i<N; ++i) {
        float x = (r(i,0) + 0.5) * nGrid;
        float y = (r(i,1) + 0.5) * nGrid;
        float z = (r(i,2) + 0.5) * nGrid;

        float* weightsX = new float[range[m]];
        float* weightsY = new float[range[m]];
        float* weightsZ = new float[range[m]];

        int startX = kernels[m](x, weightsX);
        int startY = kernels[m](y, weightsY);
        int startZ = kernels[m](z, weightsZ);

        for (int j=0; j<range[m]; ++j) {
            for (int k=0; k<range[m]; ++k) {
                for (int l=0; l<range[m]; ++l) {
                    float weight = weightsX[j] * weightsY[k] * weightsZ[l];
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    in_no_pad(
                        (startX + j + nGrid) % nGrid, 
                        (startY + k + nGrid) % nGrid, 
                        (startZ + l + nGrid) % nGrid) 
                        += weight * 1.0f;
                }
            }
        }
    }

    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Mass assignment took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << " ms" << std::endl;

    // compute the sum over all particles
    float sum = blitz::sum(in_no_pad);
    std::cout << "Sum: " << sum << std::endl;

    
    // Project the grid onto the xy-plane
    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
    Array<float,2> projected(nGrid,nGrid);
    projected = 0;
    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int k=0; k<nGrid; ++k) {
                projected(i,j) = max(in_no_pad(i,j,k), projected(i,j));
            }
        }
    }

    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
    std::cout << "Projection took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count() << " ms" << std::endl;

    ofstream myfile;
    std::string filename = "out_" + (std::string) argv[3] + ".txt";
    myfile.open(filename);
    for (int i=0; i<nGrid; ++i) {
        for (int j=0; j<nGrid; ++j) {
            myfile << projected(i,j);
            if (j<nGrid-1) myfile << " ";
        }
        myfile << "\n";
    }

    myfile.close();
    
    cplx *memory_out = reinterpret_cast <cplx*>( memory_in );
    TinyVector<int, 3> out_shape(nGrid, nGrid, nGrid/2+1);
    Array<cplx,3> out(memory_out, out_shape, neverDeleteData);

    fftwf_plan plan = fftwf_plan_dft_r2c_3d(
        nGrid, nGrid, nGrid, memory_in, (fftwf_complex*) memory_out, FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

}