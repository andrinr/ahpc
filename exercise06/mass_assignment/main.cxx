// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include <chrono>
#include <stdio.h>
#include <new>
#include <fftw3.h>
#include <cmath>
#include <string>

#include "tipsy.h"
#include "helpers.h"
#include "timer.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace blitz;

int main(int argc, char *argv[]) {
    if (argc<=2) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size] [method]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

    Timer timer = Timer();
    timer.start();

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

    timer.lap("Loading particles");
   
    // Create Mass Assignment Grid
    typedef std::complex<float> cplx;
    int mem_size = nGrid * nGrid * (nGrid+2);
    float *memory_in = new (std::align_val_t(64)) float[mem_size];

    TinyVector<int, 3> in_shape(nGrid, nGrid, nGrid+2);
    Array<float,3> in(memory_in, in_shape, deleteDataWhenDone);
    Array<float,3> in_no_pad = in(Range::all(), Range::all(), Range(0,nGrid));

    assign(r, in_no_pad, (std::string) argv[3]);
    
    timer.lap("Mass assignment");
    
    // compute the sum over all particles
    float sum = blitz::sum(in_no_pad);
    std::cout << "Sum: " << sum << std::endl;

    // Project the grid onto the xy-plane

    Array<float,2> projected(nGrid,nGrid);
    projected = 0;
    project(in_no_pad, projected);

    timer.lap("Projection");

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

    blitz::Array<float,1> bins(nGrid);

    int nBins = 30;
    bin(out, bins, nBins, true);

}