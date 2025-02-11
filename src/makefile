export LDFLAGS="-L/opt/homebrew/opt/lapack/lib"
export CPPFLAGS="-I/opt/homebrew/opt/lapack/include"
CC = clang++
CFLAGS = -O3 -Xpreprocessor -fopenmp -I$(OPENBLAS_INCLUDE) -I$(OMP_INCLUDE)
LDFLAGS = -L$(OPENBLAS_LIB) -L$(OMP_LIB) -lomp -lopenblas -llapacke

# Detect macOS architecture
#UNAME_S := $(shell uname -s)
#ifeq ($(UNAME_S), Darwin)
#    ifeq ($(shell uname -m), arm64)
#        OPENBLAS_INCLUDE = opt/homebrew/opt/openblas/include
#        OPENBLAS_LIB = opt/homebrew/opt/openblas/lib
#        OMP_INCLUDE = /opt/homebrew/opt/libomp/include
#        OMP_LIB = /opt/homebrew/opt/libomp/lib
#    else
#        OPENBLAS_INCLUDE = /usr/local/include
#        OPENBLAS_LIB = /usr/local/lib
#        OMP_INCLUDE = /usr/local/opt/libomp/include
#        OMP_LIB = /usr/local/opt/libomp/lib
#    endif
#endif

OPENBLAS_INCLUDE = /opt/homebrew/opt/openblas/include
OPENBLAS_LIB = /opt/homebrew/opt/openblas/lib
OMP_INCLUDE = /opt/homebrew/Caskroom/miniforge/base/envs/p4env/lib/clang/17.0.6/include
#OMP_LIB = /opt/homebrew/opt/libomp/lib

SRCS = ci_solver.c
OBJS = $(SRCS:.c=.o)
EXEC = ci_solver

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)

