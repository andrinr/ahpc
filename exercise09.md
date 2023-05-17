# Exercise Sheet 09

## Exercise 01

Adding new varibales to store start1, local1:

```c++
ptrdiff_t start0, local0, start1, local1;
auto alloc_local = fftwf_mpi_local_size_3d_transposed(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &local0, &start0, &local1, &start1);
```

Adapt the flags passed to the plan creation functions: 

```c++
fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(
    nGrid, nGrid, nGrid, 
    data, (fftwf_complex *)complex_data, 
    MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_ESTIMATE);
```

Finally I adjusted the size of the komplex grid view:

```c++
int grid_start1 = start1;
int grid_end1 = int(start1 + local1 - 1);

blitz::Array<std::complex<float>, 3> kdata(
    complex_data, blitz::shape(grid_end1 - grid_start1, nGrid, nGrid / 2 + 1));
```

The code still runs as expected.

## Exercise 02

I changed the ``fftw_mpi_local_size_2d_transposed`` to ``fftw_mpi_local_size_3d_transposed`` as follows:

```c++
// Init FFTW
fftwf_mpi_init();
ptrdiff_t start0, local0, start1, local1;
ptrdiff_t block0, block1;
ptrdiff_t* sizes = new ptrdiff_t[2];
sizes[0] = nGrid;
sizes[1] = nGrid;

auto alloc_tranposed = fftwf_mpi_local_size_many_transposed(
    2, sizes, nGrid, 
    FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, 
    &local0, &start0, &local1, &start1);

```

## Exercise 03

I changed to previous creationg and exection of the plan to the following:

```c++
printf("[Rank %d] 2D FFT plan created\n", i_rank);
fftwf_plan plan = fftwf_mpi_plan_dft_r2c_2d(
    nGrid, nGrid, 
    data, (fftwf_complex *)complex_data, 
    MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_ESTIMATE);

for (int i = start0; i < start0 + local0; i++) {
    
    float * slab_data = grid_data(i, blitz::Range::all(), blitz::Range::all()).data();
    std::complex<float> *complex_slab_data = reinterpret_cast<std::complex<float> *>(slab_data);

    fftwf_execute_dft_r2c(
        plan, slab_data, (fftwf_complex *)complex_slab_data);
}

printf("[Rank %d] 2D FFT plan executed for all slabs \n", i_rank);

fftwf_destroy_plan(plan);
printf("[Rank %d] 2D FFT plan destroyed\n", i_rank);
```

## Exercise 04

