# QED-CI
An open-source package to simulate strongly correlated molecules coupled to quantized cavity modes.

## Features

- Supports restricted references at the RHF or CQED-RHF levels, where the coherent state transformation is used for CQED-RHF
- Supports correlated levels including: QED-FCI, QED-CASCI, SA-QED-CASSCF
- Supports analytic gradients for ground- and excited- (polariton) states at the SA-QED-CASSCF level

Within the src/ directory:

- helper_cqed_rhf.py provides restricted hartree fock for the Pauli-Fierz Hamiltonian in the coherent state basis
- helper_cs_cqed_cis.py provides spin-adapted QED-CIS for Pauli-Fierz Hamiltonian in the coherent state basis
- helper_PFCI.py provides helper functions for arbitrary CI with Pauli-Fierz Hamiltonian, should be adapted to build PF Hamiltonian



## Getting Started
**0.  Clone repo** 

**1.  Install psi4** 

Using Conda (easy option):
- conda install psi4 python=3.10 -c conda-forge

or

From source (harder option):
- Follow directions [here](https://psicode.org/installs/v191/)

**2. Install numba and pytest:**
- pip install numba pytest

Note other python dependencies should be installed if you used the Conda option.  Other dependencies include numpy and scipy.

**3. Install intel oneapi**

**4. Compile the code with intel compiler:**
- icx -fPIC -Wall -Wextra -qopenmp -c ci_solver.c orbital.c 
- icx -shared -o cfunctions.so ci_solver.o orbital.o

**5. Run tests** 
- From the main repository directory (qed-ci), move into the source directory with `cd src/`
- run tests with `pytest -v`
