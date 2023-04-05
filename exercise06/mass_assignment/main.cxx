// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
#include <chrono>
#include <stdio.h>
#include <new>
#include <fftw3.h>
#include <cmath>
#include <assign.h>

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

    assign(r, in_no_pad, (string) argv[3]);
    
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

    Array<float, 1> kx(nGrid);
    Array<float, 1> ky(nGrid);
    Array<float, 1> kz(nGrid/2);

    for (int i = 0; i < nGrid/2; i++) {
        kx(i) = i;
        kx(nGrid/2 + i) = nGrid/2 - i;
        ky(i) = i;
        ky(nGrid/2 + i) = nGrid/2 - i;
        kz(i) = i;
    }

    int nBins = 30;
    int kMax = int(std::sqrt(
        std::pow(kx(nGrid-1), 2) + 
        std::pow(ky(nGrid-1), 2) + 
        std::pow(kz(nGrid/2-1), 2)));
    int dBin = kMax / nBins;

    Array<float, 1> fPower(nBins);
    Array<int, 1> nPower(nBins);

    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int l=0; l<nGrid/2; ++l) {
                int k = int(std::sqrt(
                    std::pow(kx(i), 2) + 
                    std::pow(ky(j), 2) + 
                    std::pow(kz(l), 2)) / kMax * nBins);

                if (k >= nBins) {
                    continue;
                }

                fPower(k) += std::norm(out(i, j, l));
                nPower(k) += 1;
            }
        }
    }

    for(int i=0; i<nGrid; ++i) {
        fPower(i) /= nPower(i);
    }   

}