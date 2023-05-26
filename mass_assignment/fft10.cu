#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>

using namespace blitz;
using std::complex;

void fill_array(Array<float,2> &data) {
    // Set the grid to the sum of two sine functions
    for (int i=0; i < data.rows(); i++) {
        for (int j=0; j < data.cols(); j++) {
            float x = (float)i / 25.0; // Period of 1/4 of the box in x
            float y = (float)j / 10.0; // Period of 1/10 of the box in y
            data(i,j) = sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y);
        }
    }
}

// Verify the FFT (kdata) of data by performing a reverse transform and comparing
bool validate(Array<float,2> &data,Array<std::complex<float>, 2> kdata) {
    Array<float,2> rdata(data.extent());
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(data.rows(), data.cols(),
        reinterpret_cast<fftwf_complex*>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}

int main() {
    int n = 10000;

    // Out of place
    Array<float,2> rdata1(n,n);
    Array<std::complex<float>, 2> kdata1(n, n/2 + 1);
    fftwf_plan plan1  = fftwf_plan_dft_r2c_2d(n, n,
        rdata1.data(), reinterpret_cast<fftwf_complex*>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);
    fftwf_execute(plan1);
    fftwf_destroy_plan(plan1);
    std::cout << ">>> Out of place FFT " << (validate(rdata1,kdata1)?"match":"MISMATCH") << endl;

    // in-place
    Array<float,2> raw_data2(n,n+2);
    Array<float,2> rdata2 = raw_data2(Range::all(),Range(0,n-1));
    fftwf_plan plan2  = fftwf_plan_dft_r2c_2d(n, n,
        rdata2.data(), reinterpret_cast<fftwf_complex*>(rdata2.data()), FFTW_ESTIMATE);
    fill_array(rdata2);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float>*>(rdata2.data()),
        shape(n, n/2 + 1),neverDeleteData);
    std::cout << ">>> In-place FFT " << (validate(rdata1,kdata2)?"match":"MISMATCH") << endl;

    // Ex 3
    Array<float,2> raw_data3(n,n);
    Array<float,2> rdata3 = raw_data3(Range::all(),Range(0,n-1));   
    fill_array(rdata3);

    Array<float,2> raw_data4(n,n);
    Array<float,2> rdata4 = raw_data4(Range::all(),Range(0,n-1));

    void *d_data3;
    size_t size_in_bytes = rdata3.size() * sizeof(float);

    cudaMalloc(&d_data3, size_in_bytes);

    cudaMemcpy(d_data3, rdata3.data(), size_in_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(rdata4.data(), d_data3, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_data3);

    std::cout << ">>> CUDA Memory Copy " << (all(abs(rdata3 - rdata4) < 1e-5)?"match":"mismatch") << endl;

    // Ex 4
    Array<float,2> raw_data5(n,n+2);
    Array<float,2> rdata5 = raw_data5(Range::all(),Range(0,n-1));   
    fill_array(rdata5);
    Array<std::complex<float>, 2> kdata5(reinterpret_cast<std::complex<float>*>(rdata5.data()),
        shape(n, n/2 + 1),neverDeleteData);

    void *d_data5;
    size_in_bytes = n * (n+2) * sizeof(float);

    cudaMalloc(&d_data5, size_in_bytes);

    cudaMemcpy(d_data5, rdata5.data(), size_in_bytes, cudaMemcpyHostToDevice);

    int grid_size[2] = {n, n};
    int inembed[2] = {n, n+2};
    int onembed[2] = {n, n/2 + 1};
    int batch = 1;
    int odist = n * (n/2 + 1);
    int idist = 2 * odist;
    int istride = 1;
    int ostride = 1;

    cufftHandle plan;
    cufftPlanMany(
        &plan, 2, grid_size,
        inembed, istride, idist, onembed, ostride, odist, 
        CUFFT_R2C, batch);

    cufftExecR2C(plan, (cufftReal*)d_data5, (cufftComplex*)d_data5);

    cudaMemcpy(rdata5.data(), d_data5, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_data5);

    std::cout << ">>> CUDA FFT " << (validate(rdata1,kdata5)?"match":"mismatch") << endl;

    return 0;
}