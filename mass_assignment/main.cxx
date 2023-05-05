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

    // Determine the number of ghost cells
    std::map<std::string, int> nGhost = {
        { "ngp", 0 },
        { "cic", 1 },
        { "tsc", 2 },
        { "pcs", 2 }
    };
    int nGhostCells = nGhost[method];

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int gridMemorySize = (nSlabs+nGhostCells*2) * nGrid * (nGrid+2);
    float *gridData = new (std::align_val_t(64)) float[gridMemorySize];

    // Create a blitz array that points to the memory
    blitz::GeneralArrayStorage<3> gridStorageLayout;
    gridStorageLayout.ordering() = blitz::firstRank, blitz::secondRank, blitz::thirdRank;
    gridStorageLayout.base() = slabStart, 0, 0;
    blitz::Array<float,3> grid(
        gridData, blitz::shape((nSlabs+nGhostCells*2), nGrid, nGrid+2), blitz::deleteDataWhenDone, gridStorageLayout);
    grid = 0;
    blitz::Array<float,3> gridNoPad = grid(blitz::Range::all(), blitz::Range::all(), blitz::Range(0,nGrid-1));

    // Assign the particles to the grid using the given mass assignment method
    assign(particles, gridNoPad, blitz::shape(nGrid, nGrid, nGrid), method, rank, np);
    
    //assert(int(blitz::sum(grid)) == particles.rows());
    timer.lap("mass assignment");

    if (nGhostCells > 0) {
        blitz::Array<float, 3> upperGhostRegion(nGhostCells, nGrid, nGrid);
        blitz::Array<float, 3> lowerGhostRegion(nGhostCells, nGrid, nGrid);
        upperGhostRegion = 0;
        lowerGhostRegion = 0;

        int dimensions_full_array[3] = {nSlabs + nGhostCells * 2, nGrid, nGrid + 2};
        int dimensions_subarray[3] = {nGhostCells, nGrid, nGrid};

        int start_coordinates_upper_send[3] = {0, 0, 0};
        MPI_Datatype upperGhostType;
        MPI_Type_create_subarray(
            3, 
            dimensions_full_array, 
            dimensions_subarray,
            start_coordinates_upper_send,
            MPI_ORDER_C, 
            MPI_FLOAT, 
            &upperGhostType);
        MPI_Type_commit(&upperGhostType);

        int start_coordinates_upper_recv[3] = {nGhostCells, 0, 0};
        MPI_Datatype upperReceive;
        MPI_Type_create_subarray(
            3, 
            dimensions_full_array,
            dimensions_subarray,
            start_coordinates_upper_recv,
            MPI_ORDER_C, 
            MPI_FLOAT, 
            &upperReceive);
        MPI_Type_commit(&upperReceive);

        int start_coordinates_lower_send[3] = {nSlabs-1 + nGhostCells, 0, 0};
        MPI_Datatype lowerGhost;
        MPI_Type_create_subarray(
            3, 
            dimensions_full_array,
            dimensions_subarray,
            start_coordinates_lower_send,
            MPI_ORDER_C, 
            MPI_FLOAT, 
            &lowerGhost);
        MPI_Type_commit(&lowerGhost);

        int start_coordinates_lower_recv[3] = {nSlabs-1, 0, 0};
        MPI_Datatype lowerReceive;
        MPI_Type_create_subarray(
            3, 
            dimensions_full_array,
            dimensions_subarray,
            start_coordinates_lower_recv,
            MPI_ORDER_C, 
            MPI_FLOAT, 
            &lowerReceive);
        MPI_Type_commit(&lowerReceive);

        MPI_Request upperSendRequest;
        MPI_Isend(
            grid.data(), 
            1, 
            upperGhostType, 
            comm.up(), 
            0, 
            MPI_COMM_WORLD, 
            &upperSendRequest);

        MPI_Request upperReceiveRequest;
        MPI_Irecv(
            upperGhostRegion.data(), 
            nGhostCells * nGrid * nGrid,
            MPI_FLOAT, 
            comm.up(), 
            0, 
            MPI_COMM_WORLD, 
            &upperReceiveRequest);

        MPI_Request lowerSendRequest;
        MPI_Isend(
            grid.data(), 
            1, 
            lowerGhost, 
            comm.down(), 
            0, 
            MPI_COMM_WORLD, 
            &lowerSendRequest);

        MPI_Request lowerReceiveRequest;
        MPI_Irecv(
            lowerGhostRegion.data(), 
            nGhostCells * nGrid * nGrid,
            MPI_FLOAT, 
            comm.down(), 
            0, 
            MPI_COMM_WORLD, 
            &lowerReceiveRequest);

        MPI_Wait(&upperSendRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&upperReceiveRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&upperSendRequest, MPI_STATUS_IGNORE);
        MPI_Wait(&upperReceiveRequest, MPI_STATUS_IGNORE);

        // Add the upper ghost region to the grid
        grid(blitz::Range(slabStart + nGhostCells, slabStart + nGhostCells * 2), blitz::Range::all(), blitz::Range::all()) += upperGhostRegion;

        // Add the lower ghost region to the grid
        grid(blitz::Range(slabStart + nSlabs, slabStart + nSlabs + nGhostCells), blitz::Range::all(), blitz::Range::all()) += lowerGhostRegion;

        timer.lap("reducing ghost regions");
    }

    float sum = blitz::sum(grid);

    MPI_Reduce(&sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sum of grid: " << sum << std::endl;
    }

    // We can remove the ghost region and the grid still remains contiguous (due to x axis being the primary axis in the data layout)
    blitz::Array<float, 3> gridNoGhost = grid(blitz::Range(nGhostCells, nGhostCells+nSlabs-1), blitz::Range::all(), blitz::Range::all());
    float mean = blitz::mean(grid);
    grid -= mean;
    grid /= mean;

    // Prepare memory to compute the FFT of the 3D grid
    cplx *memory_density_grid = reinterpret_cast <cplx*>( gridNoGhost.data() );
    // Compute the FFT of the grid
    fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(
        nGrid, nGrid, nGrid, gridNoGhost.data(), (fftwf_complex*) memory_density_grid, 
        MPI_COMM_WORLD, FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    timer.lap("3d FFT");

    // Create blitz array that wraps around the result of the FFT
    blitz::TinyVector<int, 3> density_grid_shape(nGrid, nGrid, nGrid/2+1);
    blitz::GeneralArrayStorage<3> densityStorageLayout;
    densityStorageLayout.ordering() = blitz::firstRank, blitz::secondRank, blitz::thirdRank;
    densityStorageLayout.base() = slabStart, 0, 0;
    blitz::Array<cplx,3> density(memory_density_grid, density_grid_shape, blitz::neverDeleteData, densityStorageLayout);

    // Compute bins for the power spectrum
    blitz::Array<float, 1> fPower_local(nBins);
    blitz::Array<float, 1> fPower_global(nBins);
    fPower_local = 0;
    blitz::Array<int, 1> nPower_local(nBins);
    blitz::Array<int, 1> nPower_global(nBins);
    nPower_local = 0;

    bin(density, blitz::shape(nGrid, nGrid, nGrid), fPower_local, nPower_local, nBins, log);
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