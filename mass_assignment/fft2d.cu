#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

using namespace blitz;
using std::complex;

extern "C++" void fft2d(
    Array<float, 3> grid, 
    int nGrid, 
    int grid_start, 
    int grid_end, 
    int order, 
    int n_streams) {

    int grid_size[2] = {grid.rows(), grid.cols()}; // 2D FFT of length NxN
    int inembed[2] = {grid.rows(), 2 * (grid.cols() / 2 + 1)};
    int onembed[2] = {grid.rows(), (grid.cols() / 2 + 1)};
    int batch = 1;
    int odist = grid.rows() * (grid.cols() / 2 + 1); // Output distance is in "complex"
    int idist = 2 * odist;                           // Input distance is in "real"
    int istride = 1;                                 // Elements of each FFT are adjacent
    int ostride = 1;

    size_t slab_size = sizeof(cufftComplex) * batch * grid.rows() * (grid.cols() / 2 + 1);
    size_t work_size;

    cufftHandle plan;
    cufftCreate(&plan);
    cufftSetAutoAllocation(plan, 0);
    cufftMakePlanMany(
        plan, 2, grid_size,
        inembed, istride, idist, onembed, ostride, odist, 
        CUFFT_R2C, batch, &work_size);

    void *d_slab_data;
    void *d_work_data;
    cudaMalloc(&d_slab_data, slab_size);
    cudaMalloc(&d_work_data, work_size);

    // create streams / malloc memory
    cudaStream_t stream[n_streams];
    void * d_slab_data_stream[n_streams];
    void * d_work_data_stream[n_streams];

    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&d_slab_data_stream[i], slab_size);
        cudaMalloc(&d_work_data_stream[i], work_size);
    }

    for (int i = grid_start; i < grid_end - order; i++)
    {   
        int stream_index = i % n_streams;

        Array<float, 2> slab = grid(i, Range::all(), Range::all());

        cudaMemcpyAsync(
            d_slab_data, slab.dataFirst(), 
            slab_size, cudaMemcpyHostToDevice, stream[stream_index]);

        cufftSetStream(plan, stream[stream_index]);
        cufftSetWorkArea(plan, d_work_data);

        cufftExecR2C(
            plan, (cufftReal*)d_slab_data, 
            (cufftComplex*)d_slab_data);

        cudaMemcpyAsync(
            slab.dataFirst(), d_slab_data, 
            slab_size, cudaMemcpyDeviceToHost, stream[stream_index]);
    }

    cudaDeviceSynchronize();

    // destroy streams / free memory
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(stream[i]);
        cudaFree(d_slab_data_stream[i]);
        cudaFree(d_work_data_stream[i]);
    }

    std::cout << "FFT 2D CUDA Done" << std::endl;

    cufftDestroy(plan);
}