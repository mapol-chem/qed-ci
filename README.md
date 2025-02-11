# Utilizes the psi4numpy helper_CI machinery to do QED-CI up to QED-FCI.

Within the src/ directory:

- helper_cqed_rhf.py provides restricted hartree fock for the Pauli-Fierz Hamiltonian in the coherent state basis
- helper_cs_cqed_cis.py provides spin-adapted QED-CIS for Pauli-Fierz Hamiltonian in the coherent state basis
- helper_PFCI.py provides helper functions for arbitrary CI with Pauli-Fierz Hamiltonian, should be adapted to build PF Hamiltonian

## Description
Updates to helper_PFCI towards PF-CASCI calculations in different orbital bases

## Todos
The generation of determinant lists has been broken up into two methods based on the appropriate CI Level.  Keywords for the CI level can now be passed to methods for building the PF Hamiltonian. A method for computing the 1RDM from the CIS determinants based on @nhv17 's  implementation has been added.
  - [X]  generateCISDeterminants will create the CIS determinants and store associated information to attributes with "CIS" in the name, e.g. .CISdets"
  - [X] .CISexcitation_index and .CISsingdetsign for use in computing the CIS 1RDM is computed by generateCISDeterminants
  - [X] generateCASDeterminants will create the CASCI determinants and associated information to attributes with "CAS" in the name, e.g. .CASdets
  - [X] buildConstantMatrices now takes a keyword for the CI Level (e.g. "CIS" or "CAS") and will build numDet x numDet constant matrices appropriately
  - [X] generatePFHMatrix now takes a keyword for the CI level (e.g. "CIS" or "CAS") and will use the appropriate determinant list to build the CI matrix
  - [X] calc1RDMfromCIS(c_vec) will compute the 1-RDM from a given CIS vector using the CISsingdetssign information determined by the generateCISDeterminants method.  It must be supplied a CIS vector for a desired state following diagonalizing H_CIS
  - [ ] Code still needs to be added to form the natural orbitals
  - [ ] Code still needs to be added to form the CASCI Hamiltonian in a desired orbital basis
  - [ ] Unit tests for things like the trace of the 1RDMs and the 1e energy should be added sooner than later


## To run the code:
Install numba:
- pip install numba

Install intel oneapi

Compile the code with intel compiler:
- icx -fPIC -Wall -Wextra -qopenmp -c ci_solver.c orbital.c 
- icx -shared -o cfunctions.so ci_solver.o orbital.o


Replace Intel MKL with an alternative:

Apple Accelerate framework: Includes optimized BLAS and LAPACK implementations.
OpenBLAS: Open-source alternative that works on macOS.
Netlib LAPACK: Standard LAPACK library, but slower than Accelerate or OpenBLAS.
Modify #include directives:

Replace <mkl.h> and <mkl_lapacke.h> with <Accelerate/Accelerate.h> for Apple’s built-in framework.

If using OpenBLAS, include <cblas.h> and <lapacke.h>.
JJF Note: can install OpenBlas with `brew install openblas`

Modify build process:

Replace -mkl with -framework Accelerate (for Apple’s Accelerate).

If using OpenBLAS: -lopenblas -llapacke.
Ensure compatibility with Clang/GCC:

Replace any Intel-specific intrinsics.
OpenMP is supported by GCC on macOS but requires libomp: Install via brew install libomp and link using -Xpreprocessor -fopenmp -lomp.

# Note from brew install openblas:
If you need to have openjdk first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc

For compilers to find openjdk you may need to set:
  export CPPFLAGS="-I/opt/homebrew/opt/openjdk/include"
