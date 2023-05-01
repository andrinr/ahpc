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

#include "comm.h"
#include "main.h"
#include "tipsy.h"
#include "helpers.h"
#include "ptimer.h"
#include "blitz/array.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[]) {
    if (argc<=2) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size] [method]"
                  << std::endl;
        return 1;
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

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
    }

    // Load particle positions
    std::uint64_t N_total = io.count();

    int N_load = floor(((float)N_total + np - 1) / np);

    // Handle odd number of particles
    if (rank == np-1) {
        N_load = N_total - (np-1) * N_load;
    }

    // Handle more processes than particles
    if (N_load == 0) {
        throw std::runtime_error("More processes than particles");
        return 0;
    }

    int i_start_load = rank * N_load;
    int i_end_load = (rank+1) * N_load;

    blitz::TinyVector<int, 2> lBound(i_start_load, 0);
    blitz::TinyVector<int, 2> extent(N_load, 3);

    blitz::Array<float,2> particlesUnsorted(lBound, extent);
    io.load(particlesUnsorted);

    timer.lap("Loading particles"); 

    // Sort particles by their x position
    sortParticles(particlesUnsorted);
    timer.lap("Sorting particles");

    // Get slab decomposition of grid
    long alloc_local, slab_size_long, slab_start_long, i, j;
    alloc_local = fftw_mpi_local_size_3d(
        nGrid, nGrid, nGrid, MPI_COMM_WORLD,
        &slab_size_long, &slab_start_long);

    int slab_size = slab_size_long;
    int slab_start = slab_start_long;

    // Communicate the number of particles in each slab to all processes
    blitz::Array<int,1> slab_sizes(nGrid);
    MPI_Allgather(
        &slab_size, 1, MPI_INT, slab_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    blitz::Array<int,1> slab_starts(nGrid);
    MPI_Allgather(
        &slab_start, 1, MPI_INT, slab_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute slab to rank mapping
    blitz::Array<int,1> slabToRank(nGrid);
    int current_rank = 0;
    for (int i = 0; i < nGrid; i++) {
        if (current_rank < np-1 && i == slab_starts(current_rank+1)) {
            current_rank++;
        }
        slabToRank(i) = current_rank;
    } 

    // Count the number of particles to send to each process
    blitz::Array<int,1> sendcounts(np);
    blitz::Array<int,1> recvcounts(np);
    sendcounts = 0;
    recvcounts = 0;

    for (int i = i_start_load; i < i_end_load; i++) {
        int slab = floor((particlesUnsorted(i,0)+0.5) * nGrid);
        int particleRank = slabToRank(slab);
        sendcounts(slabToRank(slab))++;
    }

    // Communicate the number of particles to send to each process
    MPI_Alltoall(
        sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    blitz::Array<int,1> senddispls(np);
    blitz::Array<int,1> recvdispls(np);

    // Compute send and receive displacements
    senddispls(0) = 0;
    recvdispls(0) = 0;

    for (int i = 1; i < np; i++) {
        senddispls(i) = senddispls(i-1) + sendcounts(i-1);
        recvdispls(i) = recvdispls(i-1) + recvcounts(i-1);
    }

    // Find total number of particles for this rank
    int N_particles = blitz::sum(recvcounts);

    std::cout << "Rank " << rank << " has " << N_particles << " particles" << std::endl;
    // Allocate memory for particles
    blitz::Array<float,2> particles(N_particles, 3);

    // Communicate the particles to each process
    MPI_Alltoallv(
        particlesUnsorted.data(), sendcounts.data(), senddispls.data(), MPI_FLOAT,
        particles.data(), recvcounts.data(), recvdispls.data(), MPI_FLOAT,
        MPI_COMM_WORLD);
    
    timer.lap("Communicating particles");

    blitz::GeneralArrayStorage<3> storage;
    storage.ordering() = blitz::firstRank, blitz::secondRank, blitz::thirdRank;
    storage.base() = slab_start, 0, 0;

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int mem_size = slab_size * nGrid * (nGrid+2);
    float *grid_data = new (std::align_val_t(64)) float[mem_size];

    // Create a blitz array that points to the memory
    blitz::Array<float,3> grid(
        grid_data, blitz::shape(slab_size, nGrid, nGrid *2), blitz::deleteDataWhenDone, storage);
    // Create a blitz array that points to the memory without padding
    blitz::Array<float,3> grid_no_pad = grid(blitz::Range::all(), blitz::Range::all(), blitz::Range(0,nGrid));

    std::cout << "grid size: " << grid.size() << std::endl;
    // Assign the particles to the grid using the given mass assignment method
    assign(particles, grid_no_pad, method);
    
    float sum = blitz::sum(grid);
    std::cout << "Sum of all particles before reduction: " << sum << std::endl;

    timer.lap("Mass assignment");

    // Get overdensity
    float mean = blitz::mean(grid);
    grid_no_pad -= mean;
    grid_no_pad /= mean;


    // // Project the grid onto the xy-plane (3d -> 2d)
    // blitz::Array<float,2> projected(nGrid,nGrid);
    // projected = 0;
    // project(grid_no_pad, projected);

    // // Output the projected grid
    // write<float>("projected", projected);
    // timer.lap("Projection");
    
    // Prepare memory to compute the FFT of the 3D grid
    cplx *memory_density_grid = reinterpret_cast <cplx*>( grid_data );
    blitz::TinyVector<int, 3> density_grid_shape(nGrid, nGrid, nGrid/2+1);
    blitz::Array<cplx,3> density(memory_density_grid, density_grid_shape, blitz::neverDeleteData);
    
    // Compute the FFT of the grid
    fftwf_plan plan = fftwf_mpi_plan_dft_3d(
        nGrid, nGrid, nGrid, grid_data, (fftwf_complex*) memory_density_grid, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Create bins for the power spectrum
    blitz::Array<float,2> bins(2, nBins);
    bins = 0;

    for (int i=0; i<nBins; ++i) {
        bins(1,i) = i;
    }

    // Compute bins for the power spectrum
    blitz::Array<float,1> fPower = bins(0, blitz::Range::all());
    bin(density, fPower, nBins, logBining);
    
    // Output the power spectrum
    write<float>("power" + std::to_string(N_total), bins);

    timer.lap("Power spectrum");
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