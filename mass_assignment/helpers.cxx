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

WrapX::WrapX(int n0, std::string method) : n0(n0), method(method) {
    kernels = {
        { "ngp", &ngp_weights },
        { "cic", &cic_weights },
        { "tsc", &tsc_weights },
        { "pcs", &pcs_weights }
    };

    range = {
        { "ngp", 1 },
        { "cic", 2 },
        { "tsc", 3 },
        { "pcs", 4 }
    };

    tmp = new float[range[method]];
};

float WrapX::wrap(float x) {
    return (kernels[method]((x + 0.5) * n0, tmp) + n0) % n0;
};

void assign(
    blitz::Array<float, 2> l_particles, 
    blitz::Array<float, 3> l_grid,
    blitz::TinyVector<int, 3> g_grid_size,
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
    for (int i=l_particles.lbound(0); i< l_particles.extent(0) + l_particles.lbound(0); ++i) {
        float x = (l_particles(i,0) + 0.5) * g_grid_size(0);
        float y = (l_particles(i,1) + 0.5) * g_grid_size(1);
        float z = (l_particles(i,2) + 0.5) * g_grid_size(2);

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

                    // if (startX + j > l_grid.extent(0) + l_grid.lbound(0) || startX + j < l_grid.lbound(0)) {
                    //     std::cout << "x " << l_particles(i,0) << " y " << l_particles(i,1) << " z " << l_particles(i,2) << std::endl;
                    //     std::cout << "rank " << rank << " : " <<  startX + j  << " out of bounds. pos x " << x  << " gsx " << g_grid_size(0) << std::endl;
                    // }
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    l_grid (
                        (startY + j + g_grid_size(0)) % g_grid_size(1), // Should not go out of bounds
                        (startY + k + g_grid_size(1)) % g_grid_size(1), 
                        (startZ + l + g_grid_size(2)) % g_grid_size(2)) 
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
    blitz::TinyVector<int, 3> grid_size,
    blitz::Array<float, 1> fPower,
    blitz::Array<int, 1> nPower,
    int nBins,
    bool log)
{
    int nGrid = grid_size(0);

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
    for(int i=grid.lbound(0); i<grid.lbound(0) + grid.extent(0); ++i) {
        for(int j=grid.lbound(1); j<grid.lbound(1) + grid.extent(1); ++j) {
            for(int l=grid.lbound(2); l<grid.lbound(2) + grid.extent(2); ++l) {

                std::cout << i << " " << j << " " << l << std::endl;
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

void sortParticles(blitz::Array<float, 2> particles, int n0, std::string method) {
    struct Particle {
        float x;
        float y;
        float z;
    };

    Particle* particle_Object = reinterpret_cast<Particle*>(particles.data());

    WrapX wrap(n0, method);

    std::sort(
        particle_Object, 
        particle_Object + particles.rows(),
        [&wrap](Particle a, Particle b){ 
            return wrap.wrap(a.x) < wrap.wrap(b.x);
        }
    );
}

blitz::Array<float, 2> reshuffleParticles (
    blitz::Array<float, 2> particlesLocallySorted, 
    int slabStart, int nSlabs, 
    std::string method,
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

    WrapX wrap(nGrid, method);
    // Count the number of particles to send to each process
    blitz::Array<int,1> sendcounts(np);
    blitz::Array<int,1> recvcounts(np);
    sendcounts = 0;
    recvcounts = 0;
    int start = particlesLocallySorted.lbound(0);
    int end = particlesLocallySorted.extent(0) + start;
    for (int i = start; i < end; i++) {
        int slab = floor(wrap.wrap(particlesLocallySorted(i, 0)));
        int particleRank = slabToRank(slab);
        sendcounts(slabToRank(slab))++;
    }

    for (int i = start; i < end; i+=1000) {
        int slab = floor(wrap.wrap(particlesLocallySorted(i, 0)));
        std::cout << slab << " slab from " << particlesLocallySorted(i, 0) << std::endl;
        int particleRank = slabToRank(slab);
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

    std::cout << "sendcounts: " << sendcounts << std::endl;
    std::cout << "recvcounts: " << recvcounts << std::endl;
    std::cout << "senddispls: " << senddispls << std::endl;
    std::cout << "recvdispls: " << recvdispls << std::endl;

    // Find total number of particles for this rank
    int N_particles = blitz::sum(recvcounts) / 3;

    // Allocate memory for particles
    blitz::Array<float,2> particles(N_particles, 3);
    particles = 0;

    // Communicate the particles to each process
    MPI_Alltoallv(
        particlesLocallySorted.data(), sendcounts.data(), senddispls.data(), MPI_FLOAT,
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