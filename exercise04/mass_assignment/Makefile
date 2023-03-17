CXX	= g++

CFLAGS = -O3 -std=c++17
LDFLAGS = -lfftw3f -lm -fopenmp

assign	: assign.o tipsy.o
	$(CXX) $(CFLAGS) -o assign assign.o tipsy.o $(LDFLAGS)

assign.o : assign.cxx tipsy.h
	$(CXX) $(CFLAGS) -c -o assign.o assign.cxx $(LDFLAGS)

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CFLAGS) -c -o tipsy.o tipsy.cxx $(LDFLAGS)

clean:
	rm -f assign assign.o tipsy.o

