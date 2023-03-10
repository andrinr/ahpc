CXX	= g++

FLAGS = -O3 -fopenmp

assign	: assign.o tipsy.o
	$(CXX) $(FLAGS) -o assign assign.o tipsy.o

assign.o : assign.cxx tipsy.h
	$(CXX) $(FLAGS) -std=c++17 -c -o assign.o assign.cxx

tipsy.o : tipsy.cxx tipsy.h	
	$(CXX) $(FLAGS) -std=c++17 -c -o tipsy.o tipsy.cxx

clean:
	rm -f assign assign.o tipsy.o

