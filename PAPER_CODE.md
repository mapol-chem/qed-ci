# PAPER_CODE.md -- the code behind the SA-QED-CASSCF gradient paper

This branch (`paper-code`, tag `analytic-gradient-paper-v1`) is the fixed version of the code that produced every
number in the paper on analytic nuclear gradients and derivative couplings for SA-QED-CASSCF (validation on H2O,
the polariton-induced conical intersection in LiH, the S0/S1 intersection of ethylene).  It holds algorithm code
only: the job scripts, pipelines and paper sources live elsewhere.  Development continues on `cpp-casscf-port`; do
not commit experiments here.

It is written for whoever merges this code into another program (e.g. a QED-SCF code), human or agent.  Read it
with `CLAUDE.md` (build hazards, architecture), which dates from 2026-09-13: its descriptions of the code are
right for this branch, but its test counts and "current state" notes are older than this file.

## 1. Identity: how to tell you have the paper's code

Every run behind the paper recorded these md5 sums (first 12 hex digits):

| file | md5 |
|---|---|
| `src/helper_PFCI.py` | `59439471058b` |
| `src/nuclear_grad.py` | `2bc170b16102` |
| `src/mecp_optimizer.py` | `e5560b820f4a` |
| `src/cfunctions.so` (built, not tracked) | `d798b4e48c68` |
| `src/psi4_dpd_shim.so` (built, not tracked) | `dd8bef77189d` |

Both `.so` files rebuild bit-identically from the tracked sources with the recipes in section 3 (Intel icx 2024.1
with the MKL 2024.1 headers; g++ from the `p4dev` conda env).  The run records cite commit hashes from before a
history rewrite (`9815f6f`, `846916d`, `53c734c`, `94380b2`); they are not on this branch, but the code they name is
the code here (the md5s above are the check).

## 2. What is where

The calculation is a chain; each layer uses the one above it.

| layer | file | what it does |
|---|---|---|
| mean field | `helper_cqed_rhf.py` | QED-HF (CQED-RHF) in the coherent-state basis, with DIIS.  Produces `<d_HF>` and the orbitals the CASSCF starts from. |
| Hamiltonian, CI, CASSCF | `helper_PFCI.py` (`PFHamiltonianGenerator`) | Pauli-Fierz Hamiltonian in determinants x photon-number states; CI (C backend); state-averaged CASSCF macro/micro-iterations. |
| CASSCF step solvers | `lstrs_trust_region.py`, `trust_region_solvers.py`, `davidson_augmented_hessian.py`, `otr_trust_region.py`, `orbital_sigma.py`, `residual_minimization.py` | Trust-region subproblem solvers and the orbital-Hessian kernels the macro loop calls.  Split out of `helper_PFCI.py` verbatim (see `CLAUDE.md`). |
| C backend | `ci_solver.c`, `orbital.c` (+ `.h`) -> `cfunctions.so` | Davidson CI, sigma builds, RDMs, integral transformations, JK; MKL + OpenMP; loaded by `ctypes`. |
| integrals | `psi4_dpd_shim.cpp` -> `psi4_dpd_shim.so` | Reads psi4 libtrans's transformed integrals into numpy (psi4 has no Python API for this). |
| gradients, couplings | `nuclear_grad.py` (`class nuclear_grad(PFHamiltonianGenerator)`) | The Lagrangian: one coupled linear solve for the multipliers of the CASSCF orbitals (`Z`), the CI vectors of every averaged state (`z`) and the QED-HF orbitals (`kappa`), then effective densities contracted with derivative integrals. |
| crossing search | `mecp_optimizer.py` (`MECPOptimizer`) | BFGS / trust-radius MECP optimizer on the Bearpark-Robb-Schlegel effective gradient.  Pure numpy, no quantum chemistry. |

Not part of the method: `legacy_python_ci.py` (quarantined dead code, nothing imports it), `examples/` and the
loose `src/*.py` scripts (old run scripts), `cpp_casscf/` (a C++ port in progress, not used by the paper),
`validation/` (finite-difference and invariance checks; useful as references, see section 6).

Map of `nuclear_grad.py` for porting: `compute_grad` (driver, line ~2396) -> `build_Y` (CI/orbital sources of the
objective), `build_Y_hf` (QED-HF source), `solve2`/`solve3` with `matvec_product` (the coupled Z/z/kappa solve),
`project_degenerate` / `_ci_response_projection` (CI response projected out of each state's degenerate set),
`build_effective_densities_mo` -> `update_effective_densities_ao` -> `transform_effective_densities`, then
`build_total_gradient`.  `compute_df_tei_gradient` and the `*_df` helpers are the density-fitted two-electron
derivative path.  `photon_ladder_antisym` / `photon_frame_scale` give the coherent-state frame term of the coupling.

## 3. Building

From `src/`, exactly as in `CLAUDE.md` (read its hazards first -- a failed compile or link truncates the
existing `.o`/`.so`):

```sh
MKLI=$HOME/intel/oneapi/mkl/2024.1/include
icx -fPIC -Wall -Wextra -qopenmp -I$MKLI -c ci_solver.c -o ci_solver.o.new
icx -fPIC -Wall -Wextra -qopenmp -I$MKLI -c orbital.c   -o orbital.o.new
PATH=/usr/bin:$PATH icx -shared -o cfunctions.so.new ci_solver.o.new orbital.o.new
mv ci_solver.o.new ci_solver.o && mv orbital.o.new orbital.o && mv cfunctions.so.new cfunctions.so

# psi4 shim (inside the conda env that has psi4; it is ABI-coupled to that psi4 build)
g++ -O2 -fPIC -shared -std=c++17 -fopenmp \
    -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/psi4 psi4_dpd_shim.cpp -o psi4_dpd_shim.so

# runtime: cfunctions.so needs Intel's runtime libraries
export LD_LIBRARY_PATH=$HOME/intel/oneapi/compiler/2024.1/lib:$LD_LIBRARY_PATH
```

Python: psi4 (conda-forge), numpy, scipy, numba, pytest; `src/` on `PYTHONPATH`.

## 4. Using it

The paper's run scripts all follow this pattern (here: two states, gradient of each and their coupling):

```python
import numpy as np
from nuclear_grad import nuclear_grad

mol = """Li 0.0 0.0 -0.4
H  0.0 0.0  1.2
units angstrom
symmetry c1
no_reorient
nocom"""                     # no_reorient/nocom: lambda_vector is fixed in the LAB frame
options = {"basis": "6-311g", "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-10}
cavity = {
    "omega_value": 0.11342,               # Hartree
    "lambda_vector": np.array([0.0, 0.0, 0.05]),   # a.u., lab frame
    "number_of_photons": 1,               # photon states |0>..|N_p>; 0 with lambda != 0 = DSE-only model
    "coherent_state_basis": True,
    "ci_level": "cas", "nact_orbs": 9, "nact_els": 4,
    "davidson_roots": 3,
    "casscf_weight": np.array([1, 1, 1]), # state-average weights; zero weights are allowed (see below)
    "spin_adaptation": "singlet",
    "davidson_threshold": 1e-7,
}
job = nuclear_grad(mol, options, cavity)  # runs QED-HF, CI and SA-CASSCF
job.compute_grad(0, 0)                    # state 0: job.total_gradient (3N,), Hartree/bohr
g0, e0 = np.array(job.total_gradient), job.eigenvals[0]
job.compute_grad(1, 1)
g1, e1 = np.array(job.total_gradient), job.eigenvals[1]
job.compute_grad(0, 1)                    # coupling of states 0 and 1
h0 = job.h0                               # <0|dH/dR|1> = (E_1 - E_0) d^{01}: the branching vector, finite at a crossing
d = job.h_vector                          # h0 / (E_1 - E_0), bohr^-1
```

Conventions to keep when merging:
- `compute_grad(m, n, deriv_type="full")` adds the CSF (orbital-overlap) term to the coupling, which is what
  Molpro's and PySCF's full NACME include; the default omits it (ETF).  The validation tables use `"full"`; the
  MECP searches use the default, where only the direction of `h0` matters.
- Sign: `h_vector = h0 / (E_n - E_m)` with `m = state1`, `n = state2`.  Pinned by
  `tests/test_numerical_nac.py::test_nac_sign_convention`; the overall sign per pair is still the arbitrary CI phase.
- The coupling computed from the coherent-state Hamiltonian is the coherent-state-frame coupling `d_CS`.  The
  lab-frame coupling adds `(1/sqrt(2w)) (d<d>/dR) <m|(b+ - b)|n>`; set `QED_PHOTON_FRAME=1` to include it (default
  off).  This term never enters a gradient (the matrix element is antisymmetric) and vanishes from `h0` at a
  crossing.  Its nuclear half (`QED_DPHOTON_NUC`, default on) is required for the translational sum rule.
- Zero-weight states: the multiplier equations contain no weight, so a state with `W_n = 0` (e.g. a photon replica
  that is followed but not averaged) has a well-defined gradient and coupling.  This needs the fix in this branch
  (no `1/W_n` in the coupling's source term).
- `<d_HF>` and the dipole shift are frozen constants of the coherent-state Hamiltonian.  Their geometry dependence
  is carried by the QED-HF response (`kappa`), which must be kept: without it the coupling error grows from 1e-6 to
  4e-2 bohr^-1 on the validation system.  (`QED_NO_HF_RESPONSE=1` drops it, as a diagnostic only.)

MECP search (`MECPOptimizer`): pass both energies, both gradients and `h_vector` to `opt.step(...)` each iteration;
`opt.positions_ang` gives the next geometry; `opt.reject()` after a failed CASSCF.  As used in the paper:
- LiH: `MECPOptimizer(['Li','H'], pos, trust=0.3, min_trust=0.1, gap_tol=3.7e-6, seam_tol=5.0e-4, rot_axes=[[0,0,1]], cone_step=True)`.
  `rot_axes=[e]` removes only rotation about the polarization, so the orientation is a coordinate; `cone_step=True`
  is the two-condition step (both branching vectors).  `cone_step=False` is the g-only comparison.
- Ethylene: default `cone_step=False`, `trust=0.20`, and the driver re-aligns the C=C bond with the polarization
  after every step (a rigid rotation, outside the optimizer).

## 5. Environment settings

The code reads many `QED_*` environment variables; nearly all are diagnostics or performance switches whose
defaults are what the paper used.  These are the ones that matter:

| variable | default | paper runs | effect |
|---|---|---|---|
| `QED_LIBTRANS` | 1 | 1 (default) | Integrals through psi4 libtrans + the shim.  Without the shim it falls back to `mints.mo_eri`: same results to the last few bits, not bit-identical. |
| `QED_SCF_GUESS_DF` | 1 | **0** in every run whose environment was recorded | Density-fitted psi4 SCF used only as the guess.  The paper ran with it off. |
| `QED_RANDOM_SEED` | unset | 12345, 1, 2 (three seeds) | Seeds the solver's random vectors.  A bit-reproducible run needs it AND `OMP_NUM_THREADS=1` (LiH, validation); the ethylene runs used 4-8 threads. |
| `QED_GRAD_CONV` | 1e-6 | 1e-8 (validation; ethylene symmetry screen and probes) | CASSCF gradient convergence. |
| `QED_MACRO_ENERGY_CONV` | 1e-10 | 0 with the line above | 0 = stop on the gradient only (the energy test otherwise caps the reachable gradient near 1e-5). |
| `QED_MICRO_GRAD_FLOOR` | 1e-7 | 1e-9 with the lines above | Absolute floor of the inner orbital step. |
| `QED_TRUE_ENERGY_RATIO` | 0 | 2 (validation; ethylene) | Accept orbital steps on the true energy change (monotone macro energy). |
| `QED_D2H_PROJECT` | 0 | 0, and 1 for half of the ethylene FC multi-start legs (D2h geometry) | Symmetrize the orbital rotation so a D2h solution stays D2h. |
| `QED_CI_PROJ` | degen | default | Which CI directions are projected out of the response (degenerate set of each state). |
| `QED_DPHOTON_NUC` | 1 | default | Nuclear half of the coherent-state frame term (couplings only). |
| `QED_PHOTON_FRAME` | 0 | 1 for the lab-frame column of the validation table | Lab-frame coupling (see section 4). |
| `QED_LSTRS_MAX_ITER` | 50 | default | Iteration cap of the LSTRS bisection (a test hook). |

The JK kernel's `QED_JK_Q2K` and `QED_JK_T1BLOCK` are read in `orbital.c`; their defaults are the shipped ones and
were not changed.  Everything not listed was left at its default.

## 6. Tests

```sh
cd src && pytest -q --ignore=examples
```

Expected on this branch: 175 passed, 2 failed.  The two failures,
`tests/test_helperPFCI.py::test_lih_fci_sto3g_rdm_builds_{no,with}_cavity`, read reference arrays from a hard-coded
`/home/jfoley19/...` path and fail anywhere else.  `--ignore=examples` is needed because pytest otherwise collects
`examples/LiH_test.py`, an old script that no longer matches the API.

What guards what: `test_numerical_nac.py` and `test_numerical_gradient_cavity.py` (analytic vs finite difference,
cavity on, unequal weights); `test_unequal_weight_sa_vs_molpro.py` with `molpro_unequal_weight_reference.py`
(gradients and couplings against Molpro at lambda = 0); `test_trs_hard_case.py`, `test_exact_tr.py`,
`test_otr_trust_region.py` (trust-region solvers); the rest are unit tests of individual solver options.
`validation/numerical_gradient_check.py` and `validation/numerical_nac_check.py` are the finite-difference
references behind the paper's validation table, and the most direct check after a merge: run them on the merged
code and compare.
