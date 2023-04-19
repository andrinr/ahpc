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
#include <mpi.h>

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

    int np, rank;
    int errs = 0;
    int provided, flag, claimed;
    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);

    MPI_Is_thread_main( &flag );
    if (!flag) {
        errs++;
        printf( "This thread called init_thread but Is_thread_main gave false\n" );fflush(stdout);
    }

    MPI_Query_thread( &claimed );
    if (claimed != provided) {
        errs++;
        printf( "Query thread gave thread level %d but Init_thread gave %d\n", claimed, provided );fflush(stdout);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &np );
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PTimer timer(rank);
    timer.start();

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
    }

    // Load particle positions
    std::uint64_t N = io.count();


    int N_per = floor(((float)N + np - 1) / np);
    int i_start = rank * N_per;
    int i_end = (rank+1) * N_per;

    TinyVector<int, 2> lBound(i_start, 0);
    TinyVector<int, 2> extent(N_per, 3);

    Array<float,2> particles(lBound, extent);
    io.load(particles);

    timer.lap("Loading particles"); 

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int mem_size = nGrid * nGrid * (nGrid+2);
    float *memory_in = new (std::align_val_t(64)) float[mem_size];

    // Create a blitz array that points to the memory
    TinyVector<int, 3> in_shape(nGrid, nGrid, nGrid+2);
    Array<float,3> grid(memory_in, in_shape, deleteDataWhenDone);
    // Create a blitz array that points to the memory without padding
    Array<float,3> grid_no_pad = grid(Range::all(), Range::all(), Range(0,nGrid));

    std::cout << "Rank: " << rank << " of " << np << " allocated memory" << std::endl;
    // Assign the particles to the grid using the given mass assignment method
    assign(particles, grid_no_pad, (std::string) argv[3]);
    
    timer.lap("Mass assignment");

    float sum = blitz::sum(grid);
    std::cout << "Sum of all particles before reduction: " << sum << std::endl;
    // Reduce the grid over all processes
    if (rank != 0) {
        MPI_Reduce(
            grid.data(),
            NULL,
            grid.size(),
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);
        finalize();
        return 0;
    }

    MPI_Reduce(
        MPI_IN_PLACE,
        grid.data(),
        grid.size(),
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD);

    // Compute the sum over all particles to verify the mass assignment
    float sum2 = blitz::sum(grid);
    std::cout << "Sum of all particles after reduction: " << sum2 << std::endl;

    // Project the grid onto the xy-plane (3d -> 2d)
    blitz::Array<float,2> projected(nGrid,nGrid);
    projected = 0;
    project(grid_no_pad, projected);

    // Output the projected grid
    write<float>("projected", projected);
    timer.lap("Projection");
    
    // Prepare memory to compute the FFT of the 3D grid
    cplx *memory_out = reinterpret_cast <cplx*>( memory_in );
    TinyVector<int, 3> out_shape(nGrid, nGrid, nGrid/2+1);
    Array<cplx,3> out(memory_out, out_shape, neverDeleteData);

    // Compute the FFT of the grid
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

    // Compute bins for the power spectrum
    blitz::Array<float,1> fPower = bins(0, Range::all());
    bin(out, fPower, nBins, false);
    
    // Output the power spectrum
    write<float>("power", bins);

    timer.lap("Power spectrum");

    finalize();
}

void finalize() {
    MPI_Finalize();
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