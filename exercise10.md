# Exercise Sheet 10

## Exercise 01

This worked for me on my local machine (since I aave cuda installed)


```bash
g++ -O3 -o fft10 fft10.cxx -lfftw3f
```

then I ran it with

```bash
./fft10
```

which returned

```txt
>>> Out of place FFT match
>>> In-place FFT match
```

## Exercise 02


```bash 
nvcc -O3 -o fft10 fft10.cu -lfftw3f
```

worked.

```bash
./fft10
```

returned

```txt
>>> Out of place FFT match
>>> In-place FFT match
```

## Exercise 03

See fft10.cu file. 

the program returned 

```txt
>>> Out of place FFT match
>>> In-place FFT match
>>> Cuda Memory Copy match
```

## Exercise 04

See fft10.cu file.

compiled locally with the command:

```bash
nvcc -O3 -o fft10 fft10.cu -lfftw3f -lcufft
```

and ran with 

```bash
./fft10
```

which returned

```bash
>>> Out of place FFT match
>>> In-place FFT match
>>> CUDA Memory Copy match
>>> CUDA FFT match
```

## Exercise 05

See ```fft2d.cu``` file.

also had to adapt the makefile:

```makefile	
CXX	= mpiCC

assign	: assign.o tipsy.o fft2d.o
	$(CXX) -O3 -o assign assign.o tipsy.o fft2d.o -lfftw3 -lfftw3f -lfftw3f_mpi -lcufft -lcudart -lmpi -lm 

assign.o : assign.cxx tipsy.h
	$(CXX) -O3 -std=c++17 -c -o assign.o assign.cxx 

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) -O3 -std=c++17 -c -o tipsy.o tipsy.cxx 

fft2d.o : fft2d.cu
	nvcc -c -O3 -o fft2d.o fft2d.cu -lfftw3f -lcufft -lcudart

clean:
	rm -f assign assign.o tipsy.o

```


