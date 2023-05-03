#include "blitz/array.h"
#include <map> // std::map
#include <string>
#include "weights.h"
#include "helpers.h"
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm> // sort
#include <fstream> // std::ifstream
#include <iostream> // std::cout
#include <mpi.h> // MPI
#include <fftw3-mpi.h> // MPI FFTW
#include "tipsy.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void assign(
    blitz::Array<float, 2> particles, 
    blitz::Array<float, 3> grid,
    blitz::TinyVector<int, 3> grid_size,
    std::string method,
    int rank,
    int np) 
{
    std::map<std::string, int> range = {
        { "ngp", 1 },
        { "cic", 2 },
        { "tsc", 3 },
        { "pcs", 4 }
    };

    typedef int (*kernel)(float, float*);
    std::map<std::string, kernel> kernels = {
        { "ngp", &ngp_weights },
        { "cic", &cic_weights },
        { "tsc", &tsc_weights },
        { "pcs", &pcs_weights }
    };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=particles.lbound(0); i< particles.extent(0) + particles.lbound(0); ++i) {
        float x = (particles(i,0) + 0.5) * grid_size(0);
        float y = (particles(i,1) + 0.5) * grid_size(1);
        float z = (particles(i,2) + 0.5) * grid_size(2);

        float* weightsX = new float[range[method]];
        float* weightsY = new float[range[method]];
        float* weightsZ = new float[range[method]];

        int startX = kernels[method](x, weightsX);
        int startY = kernels[method](y, weightsY);
        int startZ = kernels[method](z, weightsZ);

        for (int j=0; j<range[method]; ++j) {
            for (int k=0; k<range[method]; ++k) {
                for (int l=0; l<range[method]; ++l) {

                    float weight = weightsX[j] * weightsY[k] * weightsZ[l];

                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    // Check if 
                    if (startX + j < grid.lbound(0)|| 
                        startX + j >= grid.extent(0) +  grid.lbound(0)) {
                        continue;
                    }
    
                    grid (
                        startX + j, // Should not go out of bounds
                        (startY + k + grid_size(1)) % grid_size(1), 
                        (startZ + l + grid_size(2)) % grid_size(2)) 
                        += weight * 1.0f;
                }
            }
        }
    }
}

void project(
    blitz::Array<float, 3> grid_3d,
    blitz::Array<float, 2> grid_2d)
{
    // Get size of grid
    int nGrid = grid_3d.extent(0);

    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int k=0; k<nGrid; ++k) {
                grid_2d(i,j) = std::max(grid_3d(i,j,k), grid_3d(i,j));
            }
        }
    }
}

void bin(
    blitz::Array<std::complex<float>, 3> grid,
    blitz::Array<float, 1> fPower,
    blitz::Array<int, 1> nPower,
    int nBins,
    bool log)
{
    int nGrid = grid.extent(0);
    blitz::Array<float, 1> kx(nGrid);
    blitz::Array<float, 1> ky(nGrid);
    blitz::Array<float, 1> kz(nGrid/2 + 1);

    for (int i = 0; i <= nGrid/2; i++) {
        kx(i) = i;
        ky(i) = i;
        kz(i) = i;
    }

    for (int i = 0; i < nGrid/2-1; i++) {
        kx(i+nGrid/2+1) = -nGrid/2 + i + 1;
        ky(i+nGrid/2+1) = -nGrid/2 + i + 1;
    }
    
    int kMax = getK(nGrid/2, nGrid/2, nGrid/2);
    for(int i=0; i<nGrid; ++i) {
        for(int j=0; j<nGrid; ++j) {
            for(int l=0; l<nGrid/2; ++l) {
                int k = getK(kx(i), ky(j), kz(l));
                int index = getIndex(k, kMax, nBins, log);
           
                if (index >= nBins) {
                    continue;
                }

                fPower(index) += std::norm(grid(i, j, l));
                nPower(index) += 1;
            }
        }
    }
}

int getK(int x, int y, int z) {
    return int(std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2)));
}

int getIndex(int k, int kmax, int nBins, bool log) {
    if (log) {
        return int(std::log((float)k) / std::log((float) kmax) * nBins);
    } else {
        return int((float)k / kmax * nBins);
    }
}

void sortParticles(blitz::Array<float, 2> particles) {
    struct Particle {
        float x;
        float y;
        float z;
    };

    Particle* particle_Object = reinterpret_cast<Particle*>(particles.data());

    std::sort(
        particle_Object, 
        particle_Object + particles.rows(),
        [&](Particle a, Particle b){ 
            return a.x < b.x;
        }
    );
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
    blitz::Array<float,2> ghostParticles(nSend, 3);
    MPI_Send(
        particles.data() + startIndex, nSend * 3, MPI_FLOAT, otherRank, 0, MPI_COMM_WORLD);
    MPI_Recv(
        ghostParticles.data(), nRecv, MPI_FLOAT, otherRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return ghostParticles;
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