# For QED-RHF:

- helper_cqed_rhf.py -> the actual cqed-rhf algorithm, where the Fock matrix is built and diagonalized, etc.  This is what should be modified to implement the level-shift CQED-RHF

- CQED_RHF.py -> a script that will run a cqed_rhf calculation to get quantities like the cqed-rhf energy, Fock matrix, orbitals, etc.  This should be used to test if the Fock matrix from the level-shift approach is truly origin invariant.  It also has a test that will pass if the CQED-RHF energy matches the expected value.

Note: to run CQED_RHF.py, you can just type
`python CQED_RHF.py` in your terminal.


