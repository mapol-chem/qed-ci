# Helper Functions

- **helper_cqed_rhf.py ->** cqed-rhf algorithm, where the Fock matrix is built and diagonalized, etc.  This is what should be modified to implement the level-shift CQED-RHF

- **helper_PFCI.py ->** helper functions to do CI on the PF Hamiltonian

**Note:** To be able to import the helper functions from arbitrary locations, make sure the path to the src directory is included in your python path.  For example, in zsh with the location being `$HOME/Code/qed-ci/src`, the following line should be in the ``.zsrhc`` file

`export PYTHONPATH=$PYTHONPATH:$HOME/Code/qed-ci/src`

