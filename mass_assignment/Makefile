CXX = mpiCC
CFLAGS = -O3 -std=c++17
LDFLAGS = -lfftw3f_mpi -lfftw3f -lm -fopenmp
BUILDDIR = build/
SRCDIR = src/

all : main test_weights test

# Main program
MAINOBJS = main.o tipsy.o ptimer.o helpers.o comm.o
# prepend the build directory to the object files
MAINOBJS := $(addprefix $(BUILDDIR), $(MAINOBJS))

main : main.o tipsy.o ptimer.o helpers.o comm.o
	$(CXX) $(CFLAGS) -o main $(MAINOBJS) $(LDFLAGS)

main.o : main.cxx tipsy.h ptimer.h helpers.h weights.h comm.h
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)main.o main.cxx

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)tipsy.o tipsy.cxx

helpers.o : helpers.cxx helpers.h	
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)helpers.o helpers.cxx

ptimer.o : ptimer.cxx ptimer.h	
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)ptimer.o ptimer.cxx

comm.o : comm.cxx comm.h	
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)comm.o comm.cxx

# Test program
test_weights :  $(BUILDDIR)test_weights.o
	$(CXX) $(CFLAGS) -o test_weights  $(BUILDDIR)test_weights.o  $(LDFLAGS)

$(BUILDDIR)test_weights.o : test_weights.cxx weights.h
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)test_weights.o test_weights.cxx $(LDFLAGS)

clean:
	rm -f main test_weights build/*

test : test_weights
	./test_weights