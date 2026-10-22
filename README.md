# QED-CI

QED-CI simulates strongly correlated molecules coupled to quantized cavity
modes using the Pauli-Fierz Hamiltonian. It is built on Psi4 for the molecular
mean field and ships a compiled CI/CASSCF kernel for the correlated solve.

## Implemented capabilities

- Restricted RHF and CQED-RHF references (coherent-state transformation for
  CQED-RHF). UHF, ROHF, RKS/UKS, and unrestricted references are not supported.
- QED-CIS, QED-FCI, QED-CASCI, and state-averaged QED-CASSCF.
- Coherent-state and photon-number representations with documented limitations.
- State averaging with unequal weights.
- Analytic nuclear gradients and derivative couplings for supported
  SA-QED-CASSCF calculations.
- A validated `CQEDConfig` API, scoped quiet/file output, versioned structured
  results, and conventional reports. The historical constructor and imports
  remain available for migration.

Not implemented in this stage: a `Calculator` dispatcher, cqed-scf reference
wavefunctions, a TAMM/distributed backend, and unrestricted references. The
separate C++/TAMM work under `src/cpp_casscf/` is not part of the Python package.

## Canonical examples

Start with the [canonical Python examples](examples/canonical/README.md) for
QED-CIS, QED-FCI, QED-CASCI, ground-state and state-averaged QED-CASSCF, and
analytic gradients. Each script sets up one calculation and prints its key results.

## Requirements

- Python 3.12 or 3.13. Stage 5 packaging validation used Python 3.12.13 on native
  arm64 macOS; Stage 1 also observed Python 3.13.
- Psi4 1.10 (installed from conda, not from pip).
- NumPy, SciPy, Numba (with llvmlite), and psutil.
- Native build dependencies:
  - macOS: Apple Clang, LP64 OpenBLAS/LAPACKE, and LLVM OpenMP.
  - Linux: Intel oneAPI/HPC kit with `icx`, MKL, and Intel OpenMP.
- Linux execution and Windows are not validated by Stage 5. Windows is
  unsupported.

Exact tested versions and platform notes are in
[Installation and troubleshooting](docs/installation.md).

## Install

Create or activate a compatible Psi4 environment, then install the package into
it. On macOS with Miniforge:

```sh
conda create -n qed-ci -c conda-forge python=3.12 psi4=1.10.1 numpy scipy numba psutil openblas liblapacke llvm-openmp pip setuptools wheel
conda activate qed-ci
python -m pip install --no-build-isolation --no-deps .
```

For development, replace `.` with `-e .`. If the compiler headers and libraries
are in a different prefix from the Python environment, set
`QED_BUILD_PREFIX=/path/to/prefix` before installing. The build uses
`tools/build_backend.py`, records provenance under `build/backend/`, and does not
require `PYTHONPATH`. See [installation](docs/installation.md) for Linux, source
distributions, wheel limits, and troubleshooting.

The historical source-checkout workflow still works:

```sh
python tools/build_backend.py
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

An installed or editable package takes precedence in the documented install
workflow. Do not mix a source `PYTHONPATH` with an installed wheel while
diagnosing native-loading problems.

## First calculation

```python
from qed_ci import CQEDConfig, CISolverOptions, calculate

config = CQEDConfig(
    method="casci",
    active_electrons=2,
    active_orbitals=2,
    n_roots=2,
    max_photons=0,                  # fixed-orbital calculation without cavity photons
    psi4_options={"basis": "sto-3g",
                  "e_convergence": 1e-10, "d_convergence": 1e-10},
    ci_solver=CISolverOptions(threshold=1e-9),
    quiet=True,
)
job = calculate("Li\nH 1 1.4\nsymmetry c1", config)
print(job.result.energies["final"])   # Eh
print(job.result.converged)
```

`calculate()` returns the established calculation object with a new
`result: qed_ci.CalculationResult` attribute. Large CI/RDM tensors remain on the
calculation object. See the [user guide](docs/README.md) for the result schema.

## Documentation

- [User guide index](docs/README.md)
- [Installation and troubleshooting](docs/installation.md)
- [Supported methods and limitations](docs/methods.md)
- [Configuration, output, results, and migration](docs/configuration-and-output.md)
- [Runnable examples](examples/quickstart/README.md)
- [Developer setup and testing](docs/development.md)

## Tests

```sh
python -m pytest -q
```

Packaging, documentation, and isolated-install checks live in
`tests/test_packaging.py`; see [development](docs/development.md) for the
scientific suite, native rebuilds, and known baseline failures.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
