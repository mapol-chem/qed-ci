# AGENTS.md

Guidance for an agent porting this repo's **QED-CASCI** (fixed-orbital CI in
an active space, coupled to a quantized cavity mode) to TAMM's distributed
tensor API. This file is scoped to that task specifically — for general
repo orientation (build commands, the full Python package, the *other*,
separate CASSCF-to-C++ effort), see `CLAUDE.md` at this same repo root
first.

## What "QED-CASCI" means here, and where it lives

CASCI = a single CI diagonalization within a fixed active space (no orbital
optimization), as opposed to CASSCF (the same CI solve, but with an outer
loop that also rotates the orbitals). In `PFHamiltonianGenerator`
(`src/helper_PFCI.py`), CASCI is what you get by passing `"ci_level": "cas"`
and `"casscf_optimization": False` in the `cavity_options` dict — the
orbital-optimization `while` loop (`helper_PFCI.py:2412` onward) is simply
skipped, and the CI solve done once during `__init__` *is* the final
answer (`self.CIeigs`/`self.CIvecs`).

That `__init__`-time CASCI path, in order:
1. **Active-space setup** (`helper_PFCI.py:1393-1447`, the CAS "direct"
   branch): derives `n_act_a`/`n_in_a`/`n_occupied`/`num_alpha`/`H_dim`,
   builds the CI string-graph/replacement tables (`self.table`/
   `self.table_creation`/`self.table_annihilation`/`self.b_array`/`self.Y`,
   via `get_graph`/`get_string`), and the active-space-restricted
   integral blocks (`self.gkl2`, `self.occupied_J`/`occupied_fock_core`,
   `self.E_core`). Runs once per active-space definition.
2. **Davidson diagonalization** (`helper_PFCI.py:1933-1953`): one
   `c_get_roots` call → `self.CIeigs`/`self.CIvecs`.
3. **RDM build**: `build_active_rdm`/`build_active_photon_electron_one_rdm`,
   needed for properties/gradients and (in a CASSCF run) for the next
   macroiteration's orbital gradient.

All the actual numerics (graph construction, string-driven Davidson
diagonalization, sigma-vector builds, RDM builds) are **not** Python — they
live in **`src/ci_solver.c`** (declared in `src/ci_solver.h`), called from
Python via `ctypes` (see the `c_*`-prefixed wrappers near the top of
`helper_PFCI.py`, e.g. `c_get_roots`, `c_H_diag_cas_spin`). This is your
real porting target: a plain, well-isolated C kernel, not the sprawling
Python class.

## Start from the C++ port that already exists, not from Python or C directly

**Before writing anything against `ci_solver.c` or `helper_PFCI.py`
directly, look at `src/cpp_casscf/`.** A separate, already-in-progress
effort has ported *exactly* the CASCI algorithm above to C++/Eigen, tested
against real chemistry:

- **`include/casscf/ci_setup.hpp` / `src/ci_setup.cpp` → `CasscfCiSetup`**
  — step 1 above (graph/table setup), a line-cited, tested port of
  `helper_PFCI.py:1507-1613`.
- **`include/casscf/ci_state_average_solver.hpp` / `src/ci_state_average_solver.cpp`
  → `CasscfCiStateAverageSolver`** — steps 2-3 above (Davidson solve + RDM
  build), a line-cited, tested port of `helper_PFCI.py:2424-2553` (the
  per-macroiteration CI solve inside a CASSCF run — *structurally the
  same operation* as the `__init__`-time CASCI-only solve, just called
  repeatedly instead of once; see the note on `index_Hdiag`/`self.H_diag`
  vs. `self.H_diag3` in `ci_setup.hpp`'s doc comment if you need bit-exact
  agreement with a genuinely CASSCF-free Python CASCI run specifically,
  since that one call site uses a slightly different `H_diag` buffer).
- **`include/casscf/ci_orbital_backend.hpp`** — the `extern "C"`
  declarations for every `ci_solver.c` function these two classes call
  (`get_graph`, `get_string`, `build_H_diag_cas_spin`, `get_roots`,
  `build_active_rdm`, `build_active_photon_electron_one_rdm`), transcribed
  from `ci_solver.h` directly (not from the looser ctypes `argtypes`).
  Read its header comment for the data-layout convention: every array is a
  **C-contiguous (row-major) flat buffer** — a real ABI trap if you marshal
  from an `Eigen::MatrixXd`, which is column-major by default.
- **A minimal, complete usage example**: `tests/test_ci_state_average_solver.cpp`
  shows the whole thing end to end — construct one `CasscfCiSetup`, then
  call `CasscfCiStateAverageSolver::solve()` once. That pairing *is* a full
  CASCI calculation; there's no dedicated "CASCI-only" driver tool in this
  repo yet (`tools/run_macroiteration_driver.cpp` always runs the full
  CASSCF macroiteration loop), but writing one is a short exercise from
  this same pattern.

Retargeting these two classes' *internals* onto TAMM's distributed tensor
primitives (in place of `Eigen::MatrixXd`/`Eigen::Tensor` and the direct
`extern "C"` calls into `ci_solver.c`) is very likely a smaller, more
tractable task than re-deriving the algorithm from `ci_solver.c` or
`helper_PFCI.py` cold — the shapes, call order, and every real correctness
subtlety (e.g. `index_Hdiag` staying frozen at its initial value rather
than being recomputed every call; the `use_staged_inputs` distinction
between two different call sites) are already identified, documented with
exact `helper_PFCI.py:LINE` citations, and validated against real
molecules. Read `cpp_casscf/README.md`'s `CasscfCiSetup`/
`CasscfCiStateAverageSolver` sections for the full detail behind both
classes before starting.

Build/experiment with this reference implementation first:
```sh
cd src/cpp_casscf
cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/nvu12/miniforge3/envs/p4dev
cmake --build build -j
./build/test_ci_state_average_solver
```

## What's explicitly out of scope here

This repo's C++ code (and, by extension, this porting task) does **not**
generate molecular integrals. `CasscfCiSetup`/`CasscfCiStateAverageSolver`
*consume* already-computed one-electron integrals (`H_spatial2`), the PF
dipole-coupling integrals (`d_cmo`), and active-space two-electron
integrals (`J`/`K`) — all of that comes from psi4 + `helper_cqed_rhf.py`'s
CQED-RHF on the Python side, which is not part of this port. If your TAMM
target environment has its own SCF/integrals infrastructure, that's likely
where those inputs come from there instead.

## Relationship to the CASSCF work in this same repo

`src/cpp_casscf/` is primarily a *separate* effort: a full port of the
CASSCF orbital-optimization loop (trust-region solvers: LSTRS, GLTR,
Davidson-augmented-Hessian, BFGS quasi-Newton) as an intermediate step
before *that* gets retargeted onto TAMM too. You don't need any of the
orbital-optimization machinery for CASCI — only `CasscfCiSetup`/
`CasscfCiStateAverageSolver`, which that effort built as a prerequisite
(CASSCF repeatedly calls the same CI solve CASCI calls once). Worth
skimming `cpp_casscf/README.md`'s top-level status table so you know which
pieces are unrelated to your task, and worth coordinating on `CasscfContext`
(`include/casscf/casscf_context.hpp`) and `CasscfCiConfig`/
`CasscfPhysicalConstants` (`include/casscf/ci_setup.hpp`,
`internal_optimization_step.hpp`) if your TAMM port and the CASSCF port
end up needing to share a common CI-solve interface later.

## Validating a TAMM port against ground truth

The CASSCF port's own validation methodology generalizes directly to
CASCI and is worth reusing rather than reinventing: run the real Python
driver with dump hooks enabled
(`CPP_CASSCF_VALIDATION_DIR`/`_dump_cpp_casscf_validation_case`, see
`helper_PFCI.py` and `src/cpp_casscf/validation/dump_lih_case.py`) to
capture real molecular inputs and Python's actual CASCI output to disk,
then replay those same inputs through your TAMM implementation and diff.
For a CASCI-specific (not CASSCF) reference run, set `"casscf_optimization":
False` in the `cavity_options` dict passed to `PFHamiltonianGenerator` —
`dump_lih_case.py` doesn't expose this flag yet, so you'll likely want a
small variant of it (or a direct call into `PFHamiltonianGenerator`) for
this purpose. `src/cpp_casscf/tools/validate_against_python.cpp` shows the
existing dump-replay pattern for the CASSCF solvers, if useful as a
template.

## Data-shape cheat sheet

- `n_in_a` / `n_act_orb` / `n_virtual` — inactive, active, virtual orbital
  counts; `n_occupied = n_in_a + n_act_orb`.
- `n_act_a` — active alpha electrons (`n_act_el // 2`); `num_alpha =
  C(n_act_orb, n_act_a)` — the number of alpha (and, by symmetry, beta)
  strings. `H_dim = num_alpha^2 * (N_p + 1)` (the `N_p + 1` factor is the
  photon-number-basis truncation).
- `table`/`table_creation`/`table_annihilation`/`b_array`/`Y` — the
  string-graph and single-replacement-list representation the Davidson
  sigma-vector build walks; built once per active-space definition by
  `get_graph`/`get_string` and reused for every diagonalization. Treat
  `CasscfCiSetup`'s own doc comment, not this file, as the authoritative
  description of each field.
- `constint`/`constdouble` — `get_roots`'s actual parameter-passing
  convention is two flat packed arrays (9 ints, 6 doubles — see
  `helper_PFCI.py:1912-1930` for the exact index assignment), not named
  arguments. `CasscfCiStateAverageSolver::solve()` already hides this
  packing behind normal named struct fields (`CasscfCiConfig`,
  `CasscfPhysicalConstants`) — a good reason to build on it rather than
  calling `get_roots` directly.
