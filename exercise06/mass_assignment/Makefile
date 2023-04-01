CFLAGS = -O3 -std=c++17
LDFLAGS = -lfftw3f -lm -fopenmp
BUILDDIR = build/

all : assign test_weights test

# Main program
assign : $(BUILDDIR)assign.o  $(BUILDDIR)tipsy.o
	$(CXX) $(CFLAGS) -o assign  $(BUILDDIR)assign.o  $(BUILDDIR)tipsy.o $(LDFLAGS)

 $(BUILDDIR)assign.o : assign.cxx tipsy.h weights.h
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)assign.o assign.cxx $(LDFLAGS)

 $(BUILDDIR)tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)tipsy.o tipsy.cxx $(LDFLAGS)

# Test program
test_weights :  $(BUILDDIR)test_weights.o
	$(CXX) $(CFLAGS) -o test_weights  $(BUILDDIR)test_weights.o  $(LDFLAGS)

 $(BUILDDIR)test_weights.o : test_weights.cxx weights.h
	$(CXX) $(CFLAGS) -c -o  $(BUILDDIR)test_weights.o test_weights.cxx $(LDFLAGS)

clean:
	rm -f assign test_weights build/*

test : test_weights
	./test_weights