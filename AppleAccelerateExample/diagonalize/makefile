CXX = clang++
CXXFLAGS = -O2 -Wall -Wextra -std=c++11 -DACCELERATE_NEW_LAPACK
LDFLAGS = -framework Accelerate



# Target executable
TARGET = new_diag

# Source files
SRCS = new_diag.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile the source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the build
clean:
	rm -f $(OBJS) $(TARGET)
