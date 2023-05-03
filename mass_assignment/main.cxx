// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream> // std::cout
#include <cstdint> // uint32_t
#include <stdlib.h>
#include <stdio.h> 
#include <new> //std::align_val_t
#include <string> // std::string
#include <fftw3.h> // FFTW
#include <mpi.h> // MPI
#include <fftw3-mpi.h> // MPI FFTW
#include <assert.h>
#include <map> // std::map

#include "comm.h"
#include "main.h"
#include "tipsy.h"
#include "ptimer.h"
#include "helpers.h"
#include "blitz/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[]) {

    if (argc<=2) {
        throw std::invalid_argument("Not enough arguments");
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

    int nBins = 100;
    if (argc>3) nBins = atoi(argv[3]);

    std::string method = "ngp";
    if (argc>4) method = argv[4];

    bool logBining = false;
    if (argc>5) logBining = atoi(argv[5]);

    Communicator comm;
    int rank = comm.rank;
    int np = comm.np;

    PTimer timer(rank);
    timer.start();

    blitz::Array<float,2> particlesUnsorted = loadParticles(argv[1], rank, np);
    timer.lap("reading particles from input file"); 

    // Sort particles by their x position
    sortParticles(particlesUnsorted);
    timer.lap("sorting particles");

    // Get slab decomposition of grid
    long alloc_local, n_slabs_long, slab_start_long, i, j;
    alloc_local = fftwf_mpi_local_size_3d(
        nGrid, nGrid, nGrid, MPI_COMM_WORLD,
        &n_slabs_long, &slab_start_long);

    int nSlabs = n_slabs_long;
    int slabStart = slab_start_long;

    blitz::Array<float, 2> particles = 
        reshuffleParticles(particlesUnsorted, slabStart, nSlabs,  nGrid, rank, np);
    timer.lap("reshuffling particles");

    // Determine the number of ghost cells in each direction
    std::map<std::string, int> nGhost = {
        { "ngp", 0 },
        { "cic", 1 },
        { "tsc", 2 },
        { "pcs", 2 }
    };
    int nGhostCells = nGhost[method];

    if (nGhostCells > 0) {
        
        int upperGhostStart = slabStart;
        int upperGhostEnd = slabStart + nGhostCells;
        blitz::Array<float, 2> upperGhostParticles = getGhostParticles(
            particlesUnsorted, nGrid, upperGhostStart, upperGhostEnd, rank, (rank + 1 + np) % np, np);

        int lowerGhostStart = slabStart + nSlabs - nGhostCells;
        int lowerGhostEnd = slabStart + nSlabs;
        blitz::Array<float, 2> lowerGhostParticles = getGhostParticles(
            particlesUnsorted, nGrid, lowerGhostStart, lowerGhostEnd, rank, (rank - 1 + np) % np, np);
        
        timer.lap("getting ghost particles");
    }

    blitz::GeneralArrayStorage<3> storage;
    storage.ordering() = blitz::firstRank, blitz::secondRank, blitz::thirdRank;
    storage.base() = slabStart, 0, 0;

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int mem_size = nSlabs * nGrid * (nGrid+2);
    float *grid_data = new (std::align_val_t(64)) float[mem_size];

    // Create a blitz array that points to the memory
    blitz::Array<float,3> grid(
        grid_data, blitz::shape(nSlabs, nGrid, nGrid *2), blitz::deleteDataWhenDone, storage);
    grid = 0;
 
    // Create a blitz array that points to the memory without padding
    blitz::Array<float,3> grid_no_pad = grid(blitz::Range::all(), blitz::Range::all(), blitz::Range(0,nGrid-1));

    // Assign the particles to the grid using the given mass assignment method
    assign(particles, grid_no_pad, blitz::shape(nGrid, nGrid, nGrid), method, rank, np);
    
    assert(int(blitz::sum(grid)) == particles.rows());
    timer.lap("Mass assignment");

    // Get overdensity
    float mean = blitz::mean(grid);
    grid_no_pad -= mean;
    grid_no_pad /= mean;
    
    // Prepare memory to compute the FFT of the 3D grid
    cplx *memory_density_grid = reinterpret_cast <cplx*>( grid_data );
    blitz::TinyVector<int, 3> density_grid_shape(nGrid, nGrid, nGrid/2+1);
    blitz::Array<cplx,3> density(memory_density_grid, density_grid_shape, blitz::neverDeleteData);
    
    // Compute the FFT of the grid
    fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(
        nGrid, nGrid, nGrid, grid_data, (fftwf_complex*) memory_density_grid, 
        MPI_COMM_WORLD, FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Compute bins for the power spectrum
    blitz::Array<float, 1> fPower_local(nBins);
    blitz::Array<float, 1> fPower_global(nBins);
    fPower_local = 0;
    blitz::Array<int, 1> nPower_local(nBins);
    blitz::Array<int, 1> nPower_global(nBins);
    nPower_local = 0;

    bin(density, fPower_local, nPower_local, nBins, log);
    timer.lap("Power spectrum");

    MPI_Reduce(
        fPower_local.data(), fPower_global.data(), nBins, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(
        nPower_local.data(), nPower_global.data(), nBins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return 0;
    }
    
    // Compute the average power spectrum
    fPower_global /= nPower_global;

    std::cout << "Average power spectrum: " << fPower_global << std::endl;

    // Output the power spectrum
    //write<float>("power" + std::to_string(N_total), fPower);
}

template <typename T> void write(std::string location, blitz::Array<T,2> data) {
    std::ofstream myfile;
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