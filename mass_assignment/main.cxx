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

    blitz::TinyVector<int, 3> g_gridSize(nGrid, nGrid, nGrid);

    blitz::Array<float,2> g_particles = loadParticles(argv[1], rank, np);
    timer.lap("reading particles from input file"); 

    // Sort particles by their x position
    sortParticles(g_particles, g_gridSize(0), method);
    timer.lap("sorting particles");

    // Get slab decomposition of grid
    long l_alloc, n_slabs_long, slab_start_long, i, j;
    l_alloc = fftwf_mpi_local_size_3d(
        nGrid, nGrid, nGrid, MPI_COMM_WORLD,
        &n_slabs_long, &slab_start_long);

    int l_nSlabs = n_slabs_long;
    int l_slabStart = slab_start_long;

    blitz::Array<float, 2> l_particles = 
        reshuffleParticles(g_particles, l_slabStart, l_nSlabs, method, g_gridSize(0), rank, np);

    timer.lap("reshuffling particles");

    // Determine the number of ghost cells
    std::map<std::string, int> nGhost = {
        { "ngp", 0 },
        { "cic", 1 },
        { "tsc", 2 },
        { "pcs", 3 }
    };
    int nGhostCells = nGhost[method];
    blitz::TinyVector<int, 3> l_gridSizeGhostPad(l_nSlabs+nGhostCells*2, nGrid, nGrid*2);
    blitz::TinyVector<int, 3> l_gridSizeGhost(l_nSlabs, nGrid, nGrid*2);
    blitz::TinyVector<int, 3> l_gridSize(l_nSlabs, nGrid, nGrid);
    blitz::TinyVector<int, 3> l_ghostSize(nGhostCells, nGrid, nGrid);

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int gridMemorySize = blitz::product(l_gridSizeGhostPad);
    float *gridData = new (std::align_val_t(64)) float[gridMemorySize];

    // Create a blitz array that points to the memory
    blitz::GeneralArrayStorage<3> gridStorageLayout;
    gridStorageLayout.base() = l_slabStart, 0, 0;
    blitz::Array<float,3> l_gridGhostPad(
        gridData, l_gridSizeGhostPad, blitz::deleteDataWhenDone, gridStorageLayout);
    l_gridGhostPad = 0;

    // Create slices
    blitz::Array<float,3> l_gridGhost = l_gridGhostPad(blitz::Range::all(), blitz::Range::all(), blitz::Range(0,nGrid-1));
    blitz::Array<float,3> l_gridPad = l_gridGhostPad(blitz::Range(0, l_nSlabs-1), blitz::Range::all(), blitz::Range::all());
    blitz::Array<float,3> l_grid = l_gridGhost(blitz::Range(0, l_nSlabs-1), blitz::Range::all(), blitz::Range::all());

    // Ghost Cells can be removed from grid without destroying continuity of underlying memory array
    assert(!l_gridGhost.isStorageContiguous());
    assert(l_gridPad.isStorageContiguous());
    assert(!l_grid.isStorageContiguous());

    // Assign the particles to the grid using the given mass assignment method
    assign(l_particles, l_gridGhost, g_gridSize, method, rank, np);

    timer.lap("mass assignment");

    if (nGhostCells > 0) {

        blitz::Array ghostSendBuffer = 
            l_gridGhostPad(blitz::Range(l_nSlabs, l_nSlabs + nGhostCells - 1), blitz::Range::all(), blitz::Range::all());

        blitz::Array ghostRecvBuffer = 
            l_gridGhostPad(blitz::Range(0, nGhostCells - 1), blitz::Range::all(), blitz::Range::all());

        assert(ghostSendBuffer.isStorageContiguous());
        assert(ghostRecvBuffer.isStorageContiguous());
        assert(ghostRecvBuffer.size() == ghostRecvBuffer.size());

        bool odd = np % 2 != 0;
        bool last = rank == np - 1;
        bool first = rank == 0;

        std::cout << "odd  " << odd << " last " << last << " first " << first <<  " rank " << rank << std::endl;

        MPI_Request req[3];

        // Create new communicators for ghost cell exchange
        if (!odd || !last) {
            MPI_Comm ghostCommA;
            MPI_Comm_split(MPI_COMM_WORLD, int(rank / 2), rank, &ghostCommA);

            std::cout << "rank " << rank << " commA " << int(rank / 2) << std::endl;

            MPI_Ireduce(
                ghostSendBuffer.data(),
                ghostRecvBuffer.data(),
                ghostRecvBuffer.size(), 
                MPI_FLOAT, 
                MPI_SUM, 
                1, 
                ghostCommA,
                &req[0]);
        }

        if (!odd || !first) {
            MPI_Comm ghostCommB;
            MPI_Comm_split(MPI_COMM_WORLD, int(((rank + 1 + np) % np) / 2), (rank + 1 + np) % np, &ghostCommB);

            std::cout << "rank " << rank << " commB " << int(((rank + 1 + np) % np) / 2) << std::endl;

            MPI_Ireduce(
                ghostSendBuffer.data(),
                ghostRecvBuffer.data(),
                ghostRecvBuffer.size(), 
                MPI_FLOAT, 
                MPI_SUM, 
                1, 
                ghostCommB,
                &req[1]);
        }

        if (odd && (last || first)) {
            MPI_Comm ghostCommC;
            MPI_Comm_split(MPI_COMM_WORLD, 0, 0, &ghostCommC);

            std::cout << "rank " << rank << " commC rank " << 0 << std::endl;

            MPI_Ireduce(
                ghostSendBuffer.data(),
                ghostRecvBuffer.data(),
                ghostRecvBuffer.size(), 
                MPI_FLOAT, 
                MPI_SUM, 
                0, 
                ghostCommC, 
                &req[2]);
        }

        if (odd) {
            MPI_Waitall(3, &req[0], MPI_STATUS_IGNORE);
        } else {
            MPI_Waitall(2, &req[0], MPI_STATUS_IGNORE);
        }

        timer.lap("reducing ghost regions");
    }

    int sum_local = blitz::sum(l_gridGhostPad);
    int sum_global = 0;
    std::cout << "Sum of local grid: " << sum_local << std::endl;

    MPI_Reduce(&sum_local, &sum_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sum of global grid: " << sum_global << std::endl;
    }

    // We can remove the ghost region and the grid still remains contiguous (due to x axis being the primary axis in the data layout)
    float mean = blitz::mean(l_grid);
    l_grid -= mean;
    l_grid /= mean;

    std::cout << "Mean of grid: " << mean << std::endl;

    // Prepare memory to compute the FFT of the 3D grid
    cplx *memory_density_grid = reinterpret_cast <cplx*>( l_gridPad.data() );
    // Compute the FFT of the grid
    fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(
        nGrid, nGrid, nGrid, l_gridPad.data(), (fftwf_complex*) memory_density_grid, 
        MPI_COMM_WORLD, FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    timer.lap("3d FFT");

    // Create blitz array that wraps around the result of the FFT
    blitz::TinyVector<int, 3> density_grid_shape(nGrid, nGrid, nGrid/2+1);
    blitz::GeneralArrayStorage<3> densityStorageLayout;
    densityStorageLayout.ordering() = blitz::firstRank, blitz::secondRank, blitz::thirdRank;
    densityStorageLayout.base() = l_slabStart, 0, 0;
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