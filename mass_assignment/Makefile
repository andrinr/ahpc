CXX	= mpicxx

assign	: assign.o tipsy.o
	$(CXX) -O3 -o assign assign.o tipsy.o  -lfftw3 -lmpi -lfftw3f -lfftw3f_mpi -lm 

assign.o : assign.cxx tipsy.h
	$(CXX) -O3 -std=c++17 -c -o assign.o assign.cxx

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) -O3 -std=c++17 -c -o tipsy.o tipsy.cxx

clean:
	rm -f assign assign.o tipsy.o

