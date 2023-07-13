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


## Compile c shared library

gcc -g -fPIC -Wall -Wextra -pedantic cfunctions.c -shared -o cfunctions.so
