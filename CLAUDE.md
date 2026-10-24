# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

QED-CI simulates strongly correlated molecules coupled to quantized cavity modes (Pauli-Fierz Hamiltonian), supporting CQED-RHF, QED-FCI, QED-CASCI, and SA-QED-CASSCF (with analytic gradients). The package lives in **`src/`** (Python). A performance-critical CI/CASSCF backend is written in C (`ci_solver.c`/`orbital.c`) and called from Python via `ctypes`.

A C++/Eigen port of the SA-QED-CASSCF trust-region solvers (`src/cpp_casscf/`) was removed on 2026-10-22; that effort continues in a separate code base. Comments in `helper_PFCI.py` and the solver modules that cite it, and the `CPP_CASSCF_VALIDATION_DIR` dump hook, are leftovers from that work.

The unused `origin_invariant_cqed_rhf/` demos were retired from the working tree;
their recovery reference is in `docs/audit/retired-origin-demos.md`. Current
standalone calculation examples live in `examples/canonical/`.

## Python codebase (`src/`)

### Build the C backend
For the repeatable Python-backend build, run `python tools/build_backend.py`
from the repository root using the calculation environment's Python. It supports
native macOS/Apple Clang + LP64 OpenBLAS and Linux/icx + MKL. See `README.md`
for dependencies and `docs/audit/stage1/README.md` for the local baseline.
The manual instructions below describe the historical **Linux/Intel** workflow.

`cfunctions.so` is a binary artifact loaded via `ctypes.cdll.LoadLibrary` from `helper_PFCI.py`. It is **not** tracked by git (`git status` shows it as untracked), so back it up before rebuilding — there is nothing to `git checkout` if the rebuild goes wrong. After editing `ci_solver.c`/`orbital.c`, rebuild it (from `src/`):
```sh
cp -a cfunctions.so cfunctions.so.bak
cp -a ci_solver.o ci_solver.o.bak; cp -a orbital.o orbital.o.bak   # see hazard 3
MKLI=$HOME/intel/oneapi/mkl/2024.1/include                          # see hazard 2
icx -fPIC -Wall -Wextra -qopenmp -I$MKLI -c ci_solver.c -o ci_solver.o.new
icx -fPIC -Wall -Wextra -qopenmp -I$MKLI -c orbital.c   -o orbital.o.new
PATH=/usr/bin:$PATH icx -shared -o cfunctions.so.new ci_solver.o.new orbital.o.new
mv ci_solver.o.new ci_solver.o && mv orbital.o.new orbital.o
mv cfunctions.so.new cfunctions.so
```
Needs the Intel oneAPI compiler (`icx`). Three hazards the naive one-liner walks into:
- **`PATH=/usr/bin:$PATH` on the link step is required inside the `p4dev` conda env.** That env ships conda's binutils, whose `ld` shadows `/usr/bin/ld` and fails with `cannot find -lgcc` / `-lgcc_s` (it searches conda's sysroot, which has no libgcc; the system copies are under `/usr/lib/gcc/x86_64-linux-gnu/*/`). Only `ld` changes — `icx` still resolves from oneAPI. The compile step needs no such override.
- **Link to a temp name and `mv` into place.** `icx -shared -o cfunctions.so` truncates the output *before* linking, so a failed link deletes the working `.so` outright.
- **`mkl.h` is not on the default include path**, so the compile line above fails at once with `fatal error: 'mkl.h' file not found` unless `-I` is given. Use the MKL that MATCHES the compiler — `~/intel/oneapi/mkl/2024.1/include` for icx 2024.1. `/opt/intel/oneapi/mkl/2025.2` also exists; using it changes the compiler and the MKL at the same time, which is exactly what makes a last-bit A/B worthless.
- **A failed COMPILE deletes the object files too**, not just a failed link. `-c` truncates its output the same way `-shared` does, and `ci_solver.o` is *tracked*, so a failed compile is a loss of version-controlled content. Compile to `-o foo.o.new` and `mv` into place, and back the objects up first.

Baseline warning counts for a clean build are 26 from `ci_solver.c` and 14 from `orbital.c` (mostly `-Wsign-compare`); anything above that is new.

- **`cfunctions.so` does not load in a bare shell.** It needs Intel's runtime (`libsvml`, `libirng`, `libimf`, `libintlc`), and without it `pytest` fails *collection* on every test with `OSError: libsvml.so: cannot open shared object file`. Prefer the minimal fix over `source setvars.sh` — the full script also puts oneAPI's MKL ahead of conda's, which changes dgemm blocking and so perturbs last-bit-sensitive results mid-A/B:
  ```sh
  export LD_LIBRARY_PATH=$HOME/intel/oneapi/compiler/2024.1/lib:$LD_LIBRARY_PATH
  ```
- **CORRECTED 2026-09-13: the tracked `ci_solver.c`/`orbital.c` DO reproduce the current `cfunctions.so`.** A rebuild from the committed sources with local icx 2024.1 and MKL 2024.1 headers gives md5 `d798b4e48c68`, bit-identical to the shipped library, with 25/14 warnings and the suite unchanged at 172/4. The note below applies only to the OLDER library (`094a7389`), which predates the 2026-09-08 rebuild — so a bitwise reference captured against `d798b4e4` DOES survive a rebuild on this machine, while one captured against `094a7389` does not.
- **The tracked `ci_solver.c`/`orbital.c` do not reproduce the `cfunctions.so` that was shipped before 2026-09-08** (`094a7389`). Rebuilding the *unmodified* sources with the local icx 2024.1 gives a different binary, and the local compiler emits 25 `ci_solver.c` warnings, not 26. The likely cause is a compiler-version gap (icx 2025.2 lives on `chemctvdv94lws`). Consequence: **any bitwise reference captured against an older `.so` cannot survive a rebuild** — re-capture it (this includes `~/mgh_repro_20260804/gate_ah/`, which is now only valid for Python-side refactors against a fixed `.so`).
- **`tests/test_trs_hard_case.py`'s two end-to-end cases are decided by last-bit noise from ANY source, and currently fail in the working tree.** They assert that a LiH deck still walks *into* a Hessian degeneracy (two eigenvalues equal to 5.7e-17); their correctness assertions pass — the energy is `-7.977855449960` either way — and it is the *coverage* assertion that fails. At least two independent things flip them, so do not attribute it to one:
  - rebuilding `cfunctions.so` (old `.so` passes, any rebuild fails, including from pristine sources);
  - the presence of `psi4_dpd_shim.so`, i.e. whether `QED_LIBTRANS` uses libtrans or falls back to `mints.mo_eri` — a clean checkout without the shim **passes** 136/2, and copying the shim in makes it fail 134/4.
  The three subproblem tests, pinned to the dumped `lstrs_hard_case_lih.npz`, still cover the hard case itself; what is lost is end-to-end reachability. The test's own docstring predicts this: "a mean field differing at 1e-14 decided whether the LiH case below walked into it or not."

### Build the psi4 DPD shim (optional, for `QED_LIBTRANS`)
`psi4_dpd_shim.so` lets `helper_PFCI.py` read psi4 libtrans's transformed integrals into numpy, which psi4 exposes no Python API for. It is only needed if `QED_LIBTRANS=1`; without it the code prints a notice and falls back to `mints.mo_eri`. From `src/`:
```sh
g++ -O2 -fPIC -shared -std=c++17 -fopenmp \
    -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/psi4 \
    psi4_dpd_shim.cpp -o psi4_dpd_shim.so
```
It links nothing — psi4's symbols are resolved at `dlopen` time out of the already-loaded `core.cpython-*.so`, which `helper_PFCI.py` first promotes to `RTLD_GLOBAL`. That makes the `.so` **ABI-coupled to the psi4 build it was compiled against**: rebuild it after changing psi4 versions or conda envs. Validate with `python validation/dpd_readback_check.py` (compares `twoeint` and the CI roots against the `mo_eri` path).

### Python environment
`conda install psi4 python=3.12 -c conda-forge` then `pip install numba pytest` (numpy/scipy come with the psi4 conda install). Locally, the `p4dev` conda env has psi4 installed; a separate `pyscf` env exists but does *not* have psi4 (only relevant if it's ever used for the C++ validation scripts below, which need psi4).

### Tests
```sh
python -m pytest -v
```
Single test: `pytest -v tests/test_helperPFCI.py::test_name`. Contract and
scientific tests are consolidated under `tests/`; the scientific suite and its
stable trust-region fixture ship in the source distribution but not the wheel.

Install the package (`python -m pip install --no-build-isolation --no-deps -e .`) for imports from any directory; the public API and documented legacy helpers no longer need `PYTHONPATH`. The historical source-checkout build (`python tools/build_backend.py`, `PYTHONPATH=src`) remains supported for compatibility. See `docs/installation.md` and `src/README.md`.

### Architecture
- **`helper_PFCI.py`** (~15k lines) is the core of the package: `ctypes` bindings to the C backend (the `c_*`-prefixed wrapper functions near the top, e.g. `c_get_roots`, `c_build_sigma_reduced`, `c_full_transformation_macroiteration`), a `Determinant` class, and the large `PFHamiltonianGenerator` class (from line ~1360 to the end of the file): RHF/CQED-RHF setup, CI string/graph construction, RDM builds, and the SA-QED-CASSCF macro/microiteration skeleton. It is meant to be the SKELETON — the solvers it dispatches to live in the modules below.
- **Solver and kernel modules**, split out of `helper_PFCI.py` (2026-09). Each takes either a `hess_x` closure or the generator itself as `ctx`; none subclasses it, and every method moved out survives on `PFHamiltonianGenerator` as a delegator so existing call sites still resolve:
  - **`otr_trust_region.py`** — OpenTrustRegion's step (`QED_OTR_TR`).
  - **`lstrs_trust_region.py`** — the LSTRS beta-bisection (`solve_lstrs_microiteration`, `solve_lstrs_internal`), which had been written inline in four places, plus `more_sorensen_dense`.
  - **`davidson_augmented_hessian.py`** — the inner augmented-Hessian Davidson the bisection calls once per beta.
  - **`trust_region_solvers.py`** — everything else that answers "given a gradient and a way to multiply by the Hessian, what step?": GLTR (both forms), AH-Krylov, the OTR step, the three Newton solvers (`minres_solve`/`scipy_minres_solve`/`linear_equation_solve`), `inner_solve_pcg`, and the shared `projection_step2`/`aug_matvec`/`build_hessian_diagonal`.
  - **`orbital_sigma.py`** — the hot path: the numba kernels `build_sigma_reduced7`/`_8`/`hessian_guess`, their MKL `dsymv` binding (`_symv_accumulate`, `_QED_G_SYMV`), and the `orbital_sigma3`/`mv2` wrappers. **A leaf module**: it does not import `helper_PFCI` at module scope and not at all on the default path, so the dependency runs one way. `helper_PFCI` imports `_symv_accumulate` back for `build_gradient`.
- **`legacy_python_ci.py`** — QUARANTINE. Retired pure-Python CI kernels (the Knowles-Handy/Olsen sigma builds, string/graph addressing, `build_H_diag`, the Python `Davidson`) and superseded solvers (`build_sigma_reduced`/`2`/`3`/`4`, `orbital_sigma`, the Steihaug-Toint `solve_pcg_*`). **Nothing imports it; do not add a caller.** Each was proved unreachable three ways — static call graph, a call counter over the full suite, and the same counter over both real CASSCF drivers — and `ci_solver.c` carries an equivalent or better implementation of nearly all of it (the module header maps them one to one).
- **`helper_cqed_rhf.py`** — CQED-RHF mean-field (coherent-state Pauli-Fierz Hamiltonian), with DIIS.
- **`nuclear_grad.py`** — analytic gradient machinery for ground- and polariton-excited states.
- **`residual_minimization.py`** — `LinearRMSolver`, used by CASSCF's gradient-small Newton fallback.
- **`ortho_script.py`** — orbital orthogonalization utilities.

  **The moves above are verbatim, not rewrites**, and are gated by `validation/lstrs_refactor_gate.sh` — six drivers x two settings, requiring the full solver TRAJECTORY (not just the converged energy) to be bit-identical. Its reference is captured from the pre-move file and lives in `~/mgh_repro_20260804/gate_ah/`; null-test it against itself before trusting a pass. Keep further work on these modules behaviour-preserving and gated the same way. Two known-real changes are deliberately NOT bundled into any move and are still open: `build_sigma_reduced8` is `build_sigma_reduced7` plus an output allocation, an `eigval` shift and a flatten (84 of ~90 code lines identical — collapsing them is a perf change needing its own timing), and `solve_gltr_trust_region` injects unseeded `np.random.randn` noise with no knob.

  **CIS is always solved by full diagonalization** (`np.linalg.eigh` on `H_PF`), as of 2026-09. The old direct path called the Python `Davidson`, which for CIS fell through to a dense `einsum` against that same already-resident `H_PF`. Verified to move no root by more than 2.3e-13 — `validation/cis_full_diagonalization_check.py`. Note the shape change for CIS callers: `CIeigs` is now `(H_dim,)` and `CIvecs` `(H_dim, H_dim)` rather than `davidson_roots`-wide.
- `ci_solver.c`/`orbital.c` (+ `.h` headers) implement the CI Davidson solver, sigma-vector builds, RDM builds, and 4-index integral transformations; they use MKL (`cblas_*`/`LAPACKE_dsyev`) and OpenMP.
- `PFHamiltonianGenerator`'s behavior is driven by an `options_dict` (basis, SCF convergence) and a `cavity_dict`/`cavity_options` dict controlling the level of theory (`ci_level`: `fci`/`cas`), cavity coupling (`omega_value`, `lambda_vector`, `number_of_photons`), and basis choice (`coherent_state_basis` vs. photon-number basis) — see `tests/test_helperPFCI.py` for representative option combinations across FCI/CAS, with/without cavity coupling.
