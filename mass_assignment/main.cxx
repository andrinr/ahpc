// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include <stdio.h>
#include <new>
#include <fftw3.h>
#include <string>
#include <assert.h>
#include "main.h"
#include "tipsy.h"
#include "helpers.h"
#include "ptimer.h"
#include "blitz/array.h"

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

    blitz::Array<float,2> particles = load(argv[1]);

    PTimer timer;
    timer.start();
    timer.lap("Loading particles");
   
    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int mem_size = nGrid * nGrid * (nGrid+2);
    float *memory_in = new (std::align_val_t(64)) float[mem_size];

    // Create a blitz array that points to the memory
    TinyVector<int, 3> in_shape(nGrid, nGrid, nGrid+2);
    Array<float,3> in(memory_in, in_shape, deleteDataWhenDone);
    // Create a blitz array that points to the memory without padding
    Array<float,3> in_no_pad = in(Range::all(), Range::all(), Range(0,nGrid));

    // Assign the particles to the grid
    assign(particles, in_no_pad, (std::string) argv[3]);
    
    timer.lap("Mass assignment");

    // Compute the sum over all particles
    float sum = blitz::sum(in_no_pad);
    std::cout << "Sum of all particles: " << sum << std::endl;

    // Project the grid onto the xy-plane
    blitz::Array<float,2> projected(nGrid,nGrid);
    projected = 0;
    project(in_no_pad, projected);

    // Output the projected grid
    write<float>("projected", projected);
    timer.lap("Projection");
    
    // Compute the FFT of the grid
    cplx *memory_out = reinterpret_cast <cplx*>( memory_in );
    TinyVector<int, 3> out_shape(nGrid, nGrid, nGrid/2+1);
    Array<cplx,3> out(memory_out, out_shape, neverDeleteData);

    fftwf_plan plan = fftwf_plan_dft_r2c_3d(
        nGrid, nGrid, nGrid, memory_in, (fftwf_complex*) memory_out, FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Create bins for the power spectrum
    int nBins = 30;
    blitz::Array<float,2> bins(2, nBins);
    bins = 0;

    for (int i=0; i<nBins; ++i) {
        bins(1,i) = (i+1) * 0.5;
    }

    blitz::Array<float,1> fPower = bins(0, Range::all());
    bin(out, fPower, nBins, false);
    // Output the power spectrum
    write<float>("power", bins);
}

blitz::Array<float,2> load(std::string location) {
    TipsyIO io;
    io.open(location);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << location << std::endl;
    }
    // Load particle positions
    std::uint64_t N = io.count();
    Array<float,2> r(N,3);
    io.load(r);

    return r;
}

template <typename T> void write(std::string location, blitz::Array<T,2> data) {
    ofstream myfile;
    std::string filename = "out_" + location + ".txt";
    myfile.open(filename);
    int n = data.extent(0);
    int m = data.extent(1);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            myfile << data(i,j);
            if (j<m-1) myfile << " ";
        }
        myfile << "\n";
    }

    myfile.close();
}