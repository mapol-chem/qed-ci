CC = clang
CFLAGS = -std=c99 -fPIC -Wall -Wextra -arch arm64 -Xpreprocessor -fopenmp \
         -I$(INCDIR) -DACCELERATE_NEW_LAPACK
LDFLAGS = -L$(LIBDIR) -lomp -framework Accelerate

# Paths - make sure to find the correct path for PREFIX for your
# machine!
PREFIX := /Users/jfoley19/miniforge3/envs/p4env
INCDIR := $(PREFIX)/include
LIBDIR := $(PREFIX)/lib


OBJS = ci_solver.o orbital.o
TARGET = cfunctions.so

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) -shared -o $(TARGET) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
