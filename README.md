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



## Getting started

For the validated `CQEDConfig` API, quiet/file output, and legacy migration, see
[Configuration and output](docs/configuration-and-output.md).

Clone the repository and run these commands from its root. Use the same Python
environment to build and run calculations. The Python package currently uses
`PYTHONPATH`; a packaging/API redesign is not part of this build change.

### macOS (native Apple Silicon)

Install Apple's Command Line Tools (`xcode-select --install`) and Miniforge,
then create an environment:

```sh
conda create -n qed-ci -c conda-forge python psi4 numpy scipy numba psutil pytest openblas liblapacke llvm-openmp
conda activate qed-ci
python tools/build_backend.py
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

For an existing Psi4 environment, install the missing packages there instead.
The build uses Apple Clang, LP64 OpenBLAS/LAPACKE, and LLVM OpenMP from that
environment. Intel MKL is not required on macOS. All components must match the
Python architecture; do not mix Rosetta/x86 libraries with arm64 Python.

### Linux (Intel oneAPI)

Install Intel oneAPI with the C compiler, MKL, and OpenMP runtime (the oneAPI
and HPC kits used by the existing Linux workflow). Activate your Psi4 Python
environment, then initialize oneAPI using your installation's `setvars.sh`:

```sh
conda create -n qed-ci -c conda-forge python psi4 numpy scipy numba psutil pytest
conda activate qed-ci
source /opt/intel/oneapi/setvars.sh
python tools/build_backend.py
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The Linux build uses `icx`, LP64 MKL, and Intel OpenMP. Keep the matching Intel
runtime available when running calculations. The script uses the system linker
ahead of conda's binutils. Existing manual MKL builds remain supported.

### Build checks and tests

The build creates `src/cfunctions.so` (also named `.so` on macOS for the existing
ctypes loader). It compiles in a temporary build directory, checks that the
library loads, backs up any previous library, and only then replaces it.
`build/backend/manifest.json` records compiler, source hashes, and library hash.
Use `--prefix PATH` to select dependency locations or `--compiler PATH` to select
the compiler. Rebuild after changing numerical-library environments; the macOS
library records the environment's library search path.

```sh
mkdir -p build/test-run
cd build/test-run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest ../../src/tests -v
```

Tests and calculations currently write some files in their working directory;
use a separate directory for each run. Known baseline failures and reproducible
Stage 1 audit commands are recorded in [the Stage 1 report](docs/audit/stage1/README.md).
The separate `src/cpp_casscf` CMake project and optional Psi4 DPD shim retain
their existing build instructions; this script builds the Python C backend.
