#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>

using namespace blitz;
using std::complex;

extern "C++" void fft2d(Array<float, 3> grid, int nGrid, int grid_start, int grid_end, int order){
    size_t size_in_bytes = nGrid * (nGrid+2) * sizeof(float);

    int grid_size[2] = {nGrid, nGrid};
    int inembed[2] = {nGrid, nGrid};
    int onembed[2] = {nGrid, nGrid/2 + 1};
    int batch = 1;
    int odist = nGrid/2 + 1;
    int idist = 2 * odist;
    int istride = 1;
    int ostride = 1;

    cufftHandle plan;
    cufftPlanMany(
        &plan, 2, grid_size,
        inembed, istride, idist, onembed, ostride, odist, 
        CUFFT_R2C, batch);

    for (int i = grid_start; i < grid_end - order; i++)
    {   
        void *device_data;

        cudaMalloc(&device_data, size_in_bytes);
        cudaMemcpy(device_data, &grid(i, 0, 0), size_in_bytes, cudaMemcpyHostToDevice);

        cufftExecR2C(plan, (cufftReal*)device_data, (cufftComplex*)device_data);

        cudaMemcpy(&grid(i, 0, 0), device_data, size_in_bytes, cudaMemcpyDeviceToHost);

        cudaFree(device_data);
    }

    cudaDeviceSynchronize();
}