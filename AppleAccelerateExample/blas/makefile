# Makefile for Matrix Operations with Accelerate

CXX = clang++ 
CXXFLAGS = -std=c++11 -Wall -framework Accelerate

TARGET = matrix_example
SOURCES = matrix_example.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
