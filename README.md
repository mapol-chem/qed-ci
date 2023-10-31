# An efficient reference implementation of QED-CASCI utilizing the psi4numpy framework.

*Note:* A working installation of the [psi4 quantum chemistry package](https://psicode.org/psi4manual/master/build_obtaining.html) is needed to interface with this implementation.

## Description

- helper_cqed_rhf.py provides restricted hartree fock for the Pauli-Fierz Hamiltonian in the coherent state basis
- helper_PFCI.py provides a Determinant class and PFHamiltonianGenerator class for performing QED-CASCI and QED-FCI with Pauli-Fierz Hamiltonian in the coherent state and photon number state bases
- tests/test_helperPFCI.py provides a test suite for different modules of the PFHamiltonianGenerator class
- examples/ provides several input examples for running QED-CASCI and QED-FCI in the coherent state and photon number basis

## Compile c shared library
- gcc -fPIC -lm -Wall -fopenmp -pedantic -c -O3 cfunctions.c
- gcc -shared -lgomp -o cfunctions.so cfunctions.o
