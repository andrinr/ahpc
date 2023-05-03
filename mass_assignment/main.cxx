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
#include "helpers.h"
#include "ptimer.h"
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

    int upperGhostStart = slabStart;
    int upperGhostEnd = slabStart + nGhostCells;

    blitz::Array<float, 2> upperGhostParticles = getGhostParticles(
        particlesUnsorted, upperGhostStart, upperGhostEnd, nGrid, (rank + 1 + np) % np,  rank, np);

    int lowerGhostStart = slabStart + nSlabs - nGhostCells;
    int lowerGhostEnd = slabStart + nSlabs;
    blitz::Array<float, 2> lowerGhostParticles = getGhostParticles(
        particlesUnsorted, lowerGhostStart, lowerGhostEnd, nGrid, (rank - 1 + np) % np,  rank, np);
    
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
    assign(particles, grid_no_pad, blitz::shape(nGrid, nGrid, nGrid), method);
    
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
blitz::Array<float, 2> getGhostParticles(
    blitz::Array<float, 2> particles, 
    int nGrid, int regionStart, int regionEnd, int rank, int otherRank, int np
) {

    int startIndex = particles.rows();
    int endIndex = 0;

    // We could use a binary search to speed this up ( in some cases)
    for (int i = 0; i < particles.rows(); i++) {
        int slab = (particles(i,0) + 0.5) * nGrid;
        if (slab >= regionEnd) { break; }

        if (slab >= regionStart && slab >= regionEnd) {
            startIndex = std::min(i, startIndex);
            endIndex = std::max(i, endIndex);
        }
    }

    int nSend = endIndex - startIndex + 1;
    int nRecv = 0;

    MPI_Send(&nSend, 1, MPI_INT, otherRank, 0, MPI_COMM_WORLD);
    MPI_Recv(&nRecv, 1, MPI_INT, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send the particles to the previous rank
    blitz::Array<float,2> upperGhostParticles(nSend, 3);
    MPI_Send(
        particles.data() + startIndex, nSend * 3, MPI_FLOAT, otherRank, 0, MPI_COMM_WORLD);
    MPI_Recv(
        upperGhostParticles.data(), nRecv, MPI_FLOAT, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

blitz::Array<float, 2> reshuffleParticles (
    blitz::Array<float, 2> particlesUnsorted, 
    int slabStart, int nSlabs, 
    int nGrid, int rank, int np) {

    // Communicate the slab sizes and starts to all processes
    blitz::Array<int,1> nSlabsGlobal(nGrid);
    MPI_Allgather(
        &nSlabs, 1, MPI_INT, nSlabsGlobal.data(), 1, MPI_INT, MPI_COMM_WORLD);

    blitz::Array<int,1> slabStartGlobal(nGrid);
    MPI_Allgather(
        &slabStart, 1, MPI_INT, slabStartGlobal.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute slab to rank mapping
    blitz::Array<int,1> slabToRank(nGrid);
    int current_rank = 0;
    for (int i = 0; i < nGrid; i++) {
        if (current_rank < np-1 && i == slabStartGlobal(current_rank+1)) {
            current_rank++;
        }
        slabToRank(i) = current_rank;
    } 

    // Count the number of particles to send to each process
    blitz::Array<int,1> sendcounts(np);
    blitz::Array<int,1> recvcounts(np);
    sendcounts = 0;
    recvcounts = 0;
    int start = particlesUnsorted.lbound(0);
    int end = particlesUnsorted.extent(0) + start;
    for (int i = start; i < end; i++) {
        int slab = floor((particlesUnsorted(i,0)+0.5) * nGrid);
        int particleRank = slabToRank(slab);
        sendcounts(slabToRank(slab))++;
    }
    // acount for xzy axes
    sendcounts = sendcounts * 3;

    // Communicate the number of particles to send to each process
    MPI_Alltoall(
        sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute send and receive displacements
    blitz::Array<int,1> senddispls(np);
    blitz::Array<int,1> recvdispls(np);
    senddispls(0) = 0;
    recvdispls(0) = 0;

    for (int i = 1; i < np; i++) {
        senddispls(i) = senddispls(i-1) + sendcounts(i-1);
        recvdispls(i) = recvdispls(i-1) + recvcounts(i-1);
    }

    // Find total number of particles for this rank
    int N_particles = blitz::sum(recvcounts) / 3;

    // Allocate memory for particles
    blitz::Array<float,2> particles(N_particles, 3);
    particles = 0;

    // Communicate the particles to each process
    MPI_Alltoallv(
        particlesUnsorted.data(), sendcounts.data(), senddispls.data(), MPI_FLOAT,
        particles.data(), recvcounts.data(), recvdispls.data(), MPI_FLOAT,
        MPI_COMM_WORLD);

    return particles;
}

blitz::Array<float, 2> loadParticles(std::string location, int rank, int np) {
    TipsyIO io;
    io.open(location);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << location << std::endl;
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
    }

    int i_start_load = rank * N_load;
    int i_end_load = (rank+1) * N_load;

    blitz::TinyVector<int, 2> lBound(i_start_load, 0);
    blitz::TinyVector<int, 2> extent(N_load, 3);

    blitz::Array<float,2> particlesUnsorted(lBound, extent);
    io.load(particlesUnsorted);

    return particlesUnsorted;
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