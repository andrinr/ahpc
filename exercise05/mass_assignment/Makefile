CFLAGS = -O3 -std=c++17
LDFLAGS = -lfftw3f -lm #-fopenmp

all : assign test_weights

# Main program
assign : assign.o tipsy.o
	$(CXX) $(CFLAGS) -o assign assign.o tipsy.o $(LDFLAGS)

assign.o : assign.cxx tipsy.h
	$(CXX) $(CFLAGS) -c -o assign.o assign.cxx $(LDFLAGS)

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(CFLAGS) -c -o tipsy.o tipsy.cxx $(LDFLAGS)

# Test program
test_weights : test_weights.o
	$(CXX) $(CFLAGS) -o test_weights test_weights.o  $(LDFLAGS)

test_weights.o : test_weights.cxx weights.h
	$(CXX) $(CFLAGS) -c -o test_weights.o test_weights.cxx $(LDFLAGS)

clean:
	rm -f assign assign.o tipsy.o test_weights test_weights.o
