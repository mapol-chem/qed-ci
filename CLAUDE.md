# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

QED-CI simulates strongly correlated molecules coupled to quantized cavity modes (Pauli-Fierz Hamiltonian), supporting CQED-RHF, QED-FCI, QED-CASCI, and SA-QED-CASSCF (with analytic gradients). The repo contains two largely independent codebases:

1. **`src/` (Python, primary)** — the original, actively-used package. A performance-critical CI/CASSCF backend is written in C (`ci_solver.c`/`orbital.c`) and called from Python via `ctypes`.
2. **`src/cpp_casscf/` (C++/Eigen, in progress)** — a standalone port of just the SA-QED-CASSCF orbital-optimization trust-region solvers from `helper_PFCI.py`, intended as a mechanically-portable intermediate step before a collaborator retargets this onto TAMM's distributed tensor API. It links against the *same* `ci_solver.c`/`orbital.c` sources (compiled from tracked source, not the prebuilt `cfunctions.so`).

`origin_invariant_cqed_rhf/` at the repo root holds two standalone demo scripts (not imported by `src/`).

## Python codebase (`src/`)

### Build the C backend
`cfunctions.so` is a checked-in binary artifact loaded via `ctypes.cdll.LoadLibrary` from `helper_PFCI.py`. After editing `ci_solver.c`/`orbital.c`, rebuild it (from `src/`):
```sh
icx -fPIC -Wall -Wextra -qopenmp -c ci_solver.c orbital.c
icx -shared -o cfunctions.so ci_solver.o orbital.o
```
Needs the Intel oneAPI compiler (`icx`).

### Python environment
`conda install psi4 python=3.10 -c conda-forge` then `pip install numba pytest` (numpy/scipy come with the psi4 conda install). Locally, the `p4dev` conda env has psi4 installed; a separate `pyscf` env exists but does *not* have psi4 (only relevant if it's ever used for the C++ validation scripts below, which need psi4).

### Tests
```sh
cd src && pytest -v
```
Single test: `pytest -v tests/test_helperPFCI.py::test_name`. Note: `test_lih_fci_sto3g_rdm_builds_no_cavity` loads reference `.npy` files via a hardcoded absolute path (`/home/jfoley19/UPDATED_QEDCI/qed-ci/src/tests/...`) rather than a relative one — it will fail with `FileNotFoundError` unless that exact path exists.

`PYTHONPATH` must include `src/` to import the helpers from outside this directory (see `src/README.md`).

### Architecture
- **`helper_PFCI.py`** (~17k lines) is the core of the package: `ctypes` bindings to the C backend (the `c_*`-prefixed wrapper functions near the top, e.g. `c_get_roots`, `c_build_sigma_reduced`, `c_full_transformation_macroiteration`), a `Determinant` class, and the single large `PFHamiltonianGenerator` class (from line ~1360 to the end of the file) which does essentially everything: RHF/CQED-RHF setup, CI string/graph construction, Davidson diagonalization, RDM builds, and the full SA-QED-CASSCF orbital-optimization loop (macroiteration/microiteration trust-region solvers — LSTRS, GLTR, Davidson-augmented-Hessian, BFGS quasi-Newton).
- **`helper_cqed_rhf.py`** — CQED-RHF mean-field (coherent-state Pauli-Fierz Hamiltonian), with DIIS.
- **`nuclear_grad.py`** — analytic gradient machinery for ground- and polariton-excited states.
- **`residual_minimization.py`** — `LinearRMSolver`, used by CASSCF's gradient-small Newton fallback.
- **`ortho_script.py`** — orbital orthogonalization utilities.
- `ci_solver.c`/`orbital.c` (+ `.h` headers) implement the CI Davidson solver, sigma-vector builds, RDM builds, and 4-index integral transformations; they use MKL (`cblas_*`/`LAPACKE_dsyev`) and OpenMP.
- `PFHamiltonianGenerator`'s behavior is driven by an `options_dict` (basis, SCF convergence) and a `cavity_dict`/`cavity_options` dict controlling the level of theory (`ci_level`: `fci`/`cas`), cavity coupling (`omega_value`, `lambda_vector`, `number_of_photons`), and basis choice (`coherent_state_basis` vs. photon-number basis) — see `tests/test_helperPFCI.py` for representative option combinations across FCI/CAS, with/without cavity coupling.

## C++ CASSCF port (`src/cpp_casscf/`)

**Read `cpp_casscf/README.md` first.** It is the authoritative, actively-maintained status table (which classes are ported/tested, documented deviations from the Python, validation results) — do not assume this file reflects current state; the README does.

### Build & test
```sh
cd src/cpp_casscf
cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/nvu12/miniforge3/envs/p4dev   # Eigen 3.4 lives in the p4dev conda env
cmake --build build -j
cd build && ctest --output-on-failure
```
Single test: `ctest -R <test_name>` (e.g. `ctest -R gltr_trust_region_solver`), or run the binary directly (e.g. `./build/test_bfgs_operator`).
Needs MKL + an OpenMP runtime (`ci_solver.c`/`orbital.c` are compiled from source as part of this build). Set `MKLROOT` if it isn't auto-detected, and `-DCASSCF_C_BACKEND_DIR=/path/to/qed-ci/src` if `ci_solver.c`/`orbital.c` live somewhere other than one directory up.

### Validating against real chemistry
- `validation/dump_lih_case.py` (needs the `p4dev` conda env for psi4) runs a real Python CASSCF optimization with dump hooks enabled, capturing intermediate/final state to disk.
- `tools/validate_against_python.cpp` and `tools/run_macroiteration_driver.cpp` replay those dumps through the C++ port; `validation/sweep.sh` and `validation/sweep_macroiterations.sh` run this across a fixed set of systems (the latter reads checked-in `dumps_macro_sweep_*/` fixtures — see its header comment before regenerating them).
- `validation/compare_macroiterations.sh <case-name> [dump_lih_case.py args]` runs Python and C++ (QN enabled and disabled) side by side and tabulates the energy at each macroiteration.

### Architecture
- This is a **faithful, line-cited port** of the active Python code path, not a redesign — changes should cite `helper_PFCI.py:LINE` and flag any deviation from the Python explicitly (see existing doc comments for the established style).
- `CasscfContext` (`casscf_context.hpp`) is the single struct bundling persistent cross-call orbital/integral state (mirrors Python's `self.H_spatial2`/`d_cmo`/`U_total`/`J`/`K`/`occupied_*`/etc.), passed by reference to and shared by every collaborator — there is no per-class private copy.
- `MacroiterationDriver::run()` (`macroiteration_driver.hpp`) orchestrates 4 collaborator interfaces — `CiStateAverageSolver`, `InternalOptimizationStep`, `MicroiterationOptimizationStep`, `IntegralTransformer` — each with a real `Casscf*`-prefixed implementation.
- Multiple trust-region subproblem solvers exist because the Python itself dispatches between them by problem structure (see `solver_selector.hpp` and the README's "Confirmed solver dispatch" section): `LstrsSolver`/`DavidsonDrivenLstrsSolver` (LSTRS-style beta-bisection, dense vs. Davidson-subspace), `GltrTrustRegionSolver` (Lanczos), `PcgTrustRegionSolver` (Steihaug-CG).
- `casscf_c_backend` (the CMake target compiling `ci_solver.c`/`orbital.c`) is linked directly from tracked source rather than the prebuilt `cfunctions.so`, so the C++ port can't silently drift out of sync with an out-of-date build artifact.
