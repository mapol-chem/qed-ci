# Building 
- Open `makefile` and edit PREFIX := /Users/jfoley19/miniforge3/envs/psi4 to match the path to the conda environment in which psi4 was built.  My environment is also called psi4, but perhaps yours is called p4dev, p4env, psi4env, or something to that effect.  Also make sure you replace the username jfoley19 with yours!

- Once that is edited, save and close your makefile and then issue the commands

`make clean`

followed by

`make all`

- Try to run the code now - you can type 
`pytest -v`

to run the test suite.  

# Helper Functions

- helper_cqed_rhf.py -> cqed-rhf algorithm, where the Fock matrix is built and diagonalized, etc.  This is what should be modified to implement the level-shift CQED-RHF

- helper_PFCI.py -> helper functions to do CI on the PF Hamiltonian

# Note: To be able to import the helper functions from arbitrary locations, make sure the path to the src directory is included in your python path.  For example, in zsh with the location being $HOME/Code/qed-ci/src, the following line should be in the .zsrhc file

`export PYTHONPATH=$PYTHONPATH:$HOME/Code/qed-ci/src`

