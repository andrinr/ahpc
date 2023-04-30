// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream> // std::cout
#include <fstream> // std::ifstream
#include <cstdint> // uint32_t
#include <stdlib.h>
#include <stdio.h> 
#include <new> //std::align_val_t
#include <string> // std::string
#include <fftw3.h> // FFTW
#include <mpi.h> // MPI
#include <fftw3-mpi.h> // MPI FFTW

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

    int nBins = 100;
    if (argc>3) nBins = atoi(argv[3]);

    std::string method = "ngp";
    if (argc>4) method = argv[4];

    bool logBining = false;
    if (argc>5) logBining = atoi(argv[5]);

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

    // Handle odd number of particles
    if (rank == np-1) {
        N_per = N - (np-1) * N_per;
    }

    // Handle more processes than particles
    if (N_per == 0) {
        throw std::runtime_error("More processes than particles");
        finalize();
        return 0;
    }

    int i_start = rank * N_per;
    int i_end = (rank+1) * N_per;

    TinyVector<int, 2> lBound(i_start, 0);
    TinyVector<int, 2> extent(N_per, 3);

    Array<float,2> particles(lBound, extent);
    io.load(particles);

    timer.lap("Loading particles"); 

    sortParticles(particles);

    timer.lap("Sorting particles");

    long alloc_local, local_n0, local_0_start, i, j;

    alloc_local = fftw_mpi_local_size_3d(
        nGrid, nGrid, nGrid, MPI_COMM_WORLD,
        &local_n0, &local_0_start);

    //long* all_local_n0 = new long[np];
    blitz::Array<long,1> all_local_n0(nGrid);

    MPI_Allgather(
        &local_n0,
        1,
        MPI_LONG,
        all_local_n0.data(),
        1,
        MPI_LONG,
        MPI_COMM_WORLD);

    blitz::Array<long,1> all_local_0_start(nGrid);
    MPI_Allgather(
        &local_0_start,
        1,
        MPI_LONG,
        all_local_0_start.data(),
        1,
        MPI_LONG,
        MPI_COMM_WORLD);

    blitz::Array<int,1> slabToRank(nGrid);
    int current_rank = 0;
    int total = 0;
    for (int i = 0; i < nGrid; i++) {
        slabToRank(i) = current_rank;
        total += all_local_n0(i);

        if (total >= N) {
            current_rank++;
            total = 0;
        }
    } 

    blitz::Array<int,1> sendcounts(np);
    blitz::Array<int,1> recvcounts(np);
    sendcounts = 0;
    recvcounts = 0;

    for (int i = i_start; i < i_end; i++) {
        int slab = floor((particles(i,0)+0.5) * nGrid);
        int particleRank = slabToRank(slab);
        if (particleRank == rank) {
            continue;
        }
        sendcounts(slabToRank(slab))++;
    }

    for (int i = 0; i < np; i++) {
        std::cout << "from " << i << " to " << " send " << sendcounts(i) << std::endl;
    }

    timer.lap("Counting particles");

    MPI_Alltoall(
        sendcounts.data(),
        1,
        MPI_INT,
        recvcounts.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD);

    for (int i = 0; i < np; i++) {
        std::cout << "from " << i << " to " << rank << " rec " << recvcounts(i) << std::endl;
    }

    GeneralArrayStorage<3> storage;
    storage.ordering() = firstRank, secondRank, thirdRank;
    storage.base() = local_0_start, 0, 0;

    // Create Mass Assignment Grid with padding for fft
    typedef std::complex<float> cplx;
    int mem_size = local_n0 * nGrid * (nGrid+2);
    std::cout << "mem_size: " << mem_size << std::endl;
    float *grid_data = new (std::align_val_t(64)) float[mem_size];

    // Create a blitz array that points to the memory
    Array<float,3> grid(grid_data, shape(local_n0, nGrid, nGrid *2), deleteDataWhenDone, storage);
    // Create a blitz array that points to the memory without padding
    Array<float,3> grid_no_pad = grid(Range::all(), Range::all(), Range(0,nGrid));

    // Assign the particles to the grid using the given mass assignment method
    assign(particles, grid_no_pad, method);
    
    float sum = blitz::sum(grid);
    std::cout << "Sum of all particles before reduction: " << sum << std::endl;

    timer.lap("Mass assignment");

    // Reduce the grid over all processes
    if (rank != 0) {
        MPI_Reduce(
            grid.data(),
            grid.data(),
            grid.size(),
            MPI_FLOAT,
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
        MPI_FLOAT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD);

    // Compute the sum over all particles to verify the mass assignment
    float sum2 = blitz::sum(grid);
    std::cout << "Sum of all particles after reduction: " << sum2 << std::endl;

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
    TinyVector<int, 3> density_grid_shape(nGrid, nGrid, nGrid/2+1);
    Array<cplx,3> density(memory_density_grid, density_grid_shape, neverDeleteData);
    
    // Compute the FFT of the grid
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(
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
    blitz::Array<float,1> fPower = bins(0, Range::all());
    bin(density, fPower, nBins, logBining);
    
    // Output the power spectrum
    write<float>("power" + to_string(N), bins);

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
