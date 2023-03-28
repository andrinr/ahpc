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

    std::map<std::string, int> max_distance = {
        { "ngp", 0 },
        { "cic", 1 },
        { "tsc", 2 },
        { "psc", 2 }
    };

    typedef float (*kernel)(float);
    std::map<std::string, kernel> kernels = {
        { "ngp", [](float x) { 
            if (x < 0.5) {
                return 1.0f;
            }
            else {
                return 0.0f;
            }
        } },
        { "cic", [](float x) { 
            if (x < 1) {
                return 1.0f - x;
            }
            else {
                return 0.0f;
            }
        } },
        { "tsc", [](float x) { 
            if (x < 0.5f) {
                return (3.0f / 4.0f - x * x);
            }
            else if (x < 1.5f) {
                return 0.5f * (1.5f - x) * (1.5f - x);
            }
            else {
                return 0.0f;
            }
         } },
        { "psc", [](float x) { 
            if (x < 1.0f) {
                return 1.0f / 6.0f * (4.0f - 6.0f * x * x + 3 * x * x * x);
            }
            else if (x < 2.0f) {
                return (float) 1.0f / 6.0f * (2.0f - x) * (2.0f - x) * (2.0f - x);
            }
            else {
                return (float) 0.0f;
            }
         } }
    };

    std::string m = (std::string) argv[3];

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i<N; ++i) {
        float x = (r(i,0) + 0.5) * nGrid;
        float y = (r(i,1) + 0.5) * nGrid;
        float z = (r(i,2) + 0.5) * nGrid;

        int gridX = int((r(i,0) + 0.5) * nGrid);
        int gridY = int((r(i,1) + 0.5) * nGrid);
        int gridZ = int((r(i,2) + 0.5) * nGrid);

        for (int j=-max_distance[m]; j<=max_distance[m]; ++j) {
            for (int k=-max_distance[m]; k<=max_distance[m]; ++k) {
                for (int l=-max_distance[m]; l<=max_distance[m]; ++l) {

                    int coordX = gridX + j;
                    int coordY = gridY + k;
                    int coordZ = gridZ + l;

                    float dx = x - (coordX + 0.5);
                    float dy = y - (coordY + 0.5);
                    float dz = z - (coordZ + 0.5);

                    float weight = kernels[m](std::abs(dx));
                    weight *= kernels[m](std::abs(dy));
                    weight *= kernels[m](std::abs(dz));

                    coordX = (coordX + nGrid) % nGrid;
                    coordY = (coordY + nGrid) % nGrid;
                    coordZ = (coordZ + nGrid) % nGrid;

                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    in_no_pad(coordX, coordY, coordZ) += weight * 1.0f;
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