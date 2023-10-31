# Utilizes the psi4numpy helper_CI machinery to do QED-CI up to QED-FCI.

Within the src/ directory:

- helper_cqed_rhf.py provides restricted hartree fock for the Pauli-Fierz Hamiltonian in the coherent state basis
- helper_cs_cqed_cis.py provides spin-adapted QED-CIS for Pauli-Fierz Hamiltonian in the coherent state basis
- helper_PFCI.py provides helper functions for arbitrary CI with Pauli-Fierz Hamiltonian, should be adapted to build PF Hamiltonian

## Description
Updates to helper_PFCI towards PF-CASCI calculations in different orbital bases

## Compile c shared library
- gcc -fPIC -lm -Wall -fopenmp -pedantic -c -O3 cfunctions.c
- gcc -shared -lgomp -o cfunctions.so cfunctions.o
