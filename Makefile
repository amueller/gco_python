C_FLAGS = -O3

%.o: %.cpp
	g++ -c $(C_FLAGS) $< -o $@
example: graph.o maxflow.o GCoptimization.o LinkedBlockList.o  example.cpp
	g++  example.cpp graph.o maxflow.o GCoptimization.o LinkedBlockList.o -o example
