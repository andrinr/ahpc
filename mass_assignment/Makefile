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

