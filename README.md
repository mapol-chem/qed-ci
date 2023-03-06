# Utilizes the psi4numpy helper_CI machinery to do QED-CI up to QED-FCI.

Within the src/ directory:

- helper_cqed_rhf.py provides restricted hartree fock for the Pauli-Fierz Hamiltonian in the coherent state basis
- helper_cs_cqed_cis.py provides spin-adapted QED-CIS for Pauli-Fierz Hamiltonian in the coherent state basis
- helper_CI.py provides helper functions for arbitrary CI with Pauli-Fierz Hamiltonian, should be adapted to build PF Hamiltonian
