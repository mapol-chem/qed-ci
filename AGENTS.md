# AGENTS.md

Guidance for an agent porting this repo's **QED-CASCI** (fixed-orbital CI in
an active space, coupled to a quantized cavity mode) to TAMM's distributed
tensor API. This file is scoped to that task specifically — for general
repo orientation (build commands, the full Python package), see `CLAUDE.md`
at this same repo root first.

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

## What's explicitly out of scope here

This porting task does **not** generate molecular integrals. A CASCI solve
*consumes* already-computed one-electron integrals (`H_spatial2`), the PF
dipole-coupling integrals (`d_cmo`), and active-space two-electron
integrals (`J`/`K`) — all of that comes from psi4 + `helper_cqed_rhf.py`'s
CQED-RHF on the Python side. If your TAMM target environment has its own
SCF/integrals infrastructure, that's likely where those inputs come from
there instead.

## Validating a TAMM port against ground truth

Run the real Python driver with a CASCI-only reference (`"casscf_optimization":
False` in the `cavity_options` dict passed to `PFHamiltonianGenerator`),
capture its inputs (`H_spatial2`, `d_cmo`, `J`/`K`, the string tables) and
its output (`self.CIeigs`/`self.CIvecs` and the RDMs) to disk, then replay
those same inputs through your TAMM implementation and diff.

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
  `ci_solver.h`, not this file, as the authoritative description of each
  field.
- `constint`/`constdouble` — `get_roots`'s actual parameter-passing
  convention is two flat packed arrays (9 ints, 6 doubles — see
  `helper_PFCI.py:1912-1930` for the exact index assignment), not named
  arguments.
