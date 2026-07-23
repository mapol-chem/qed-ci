# Integrating the updated C backend (`ci_solver.c/.h`, `orbital.c/.h`)

Notes for the collaborator maintaining a TAMM integration of the plain-C
CASCI/CASSCF backend. Covers: (1) what changed and how to pull it in, (2) the
new print/diagnostic controls, (3) how to request a spin state, and (4) how far
CASCI is from the full CASSCF.

Everything below is **backward compatible** — the signatures of `get_roots`,
`davidson_spin`, and `davidson` are unchanged. You can drop in the new
`ci_solver.c`/`ci_solver.h` (and `orbital.c`/`orbital.h`) and nothing breaks.

---

## 1. What changed, and how to update

The changes are all in `ci_solver.c`/`ci_solver.h`. They are additive:

- **Four new exported functions** (declared at the top of `ci_solver.h`):
  ```c
  int    get_ci_print_level(void);
  void   set_ci_print_level(int level);
  int    get_last_ci_iterations(void);
  double get_last_ci_root_residual(int root);
  ```
- **Internal `printf`s are now gated** by a print level. This is the only
  behavioral change, and it defaults to "print everything" (level 3 = Trace),
  so **if you do nothing, output is identical to before.** To make the solver
  quieter you call `set_ci_print_level(...)` (see §2).

To update: replace your copy of `ci_solver.c`/`ci_solver.h` with the new one and
recompile. No call sites need to change. If your build compiles the file with
`-Werror`, note the new functions use `getenv`/`atoi` (already-included
`<stdlib.h>`) and `cblas_ddot` (already used throughout the file).

There are **no changes to `orbital.c`/`orbital.h`** in this batch — pull them
only if your copy is older than the CASSCF integral-transform additions
(`full_transformation_macroiteration`, `full_transformation_internal_optimization`,
`build_sigma_reduced`).

---

## 2. Print options / diagnostics

### Print level

One integer, 0–3, matching the C++ side's `casscf::PrintLevel`:

| level | name | what the CI solver prints |
|------:|------|---------------------------|
| 0 | Silent | nothing (not even "converged") |
| 1 | Normal | one `converged` line per solve; a spin warning **only if** a root's ⟨S²⟩ deviates from the requested spin |
| 2 | Debug  | same as Normal (the per-iteration tables stay off) |
| 3 | Trace  | the full per-Davidson-iteration `ITERATION / build sigma / ROOT RESIDUAL` tables, timing, memory, and the per-solve ⟨S²⟩ table — i.e. the historical output |

Set it once (or per solve) before calling `get_roots`:

```c
#include "ci_solver.h"

set_ci_print_level(1);          /* quiet: only convergence + spin exceptions */
get_roots(/* ... */, casscf, target_spin);
```

Two ways to set it:
- **`set_ci_print_level(int)`** — explicit, wins over the environment.
- **`QED_PRINT_LEVEL` env var** — read once on the first call if you never call
  the setter. Default (unset, no setter call) is **3 (Trace)**.

If you drive the backend from a larger C++/TAMM layer that has its own logger,
call `set_ci_print_level(your_level)` at startup so the C backend matches it —
otherwise it stays at Trace and prints its own tables regardless of your logger.

### Per-solve diagnostics (read immediately after `get_roots`)

```c
get_roots(/* ... */);
int    n_iter = get_last_ci_iterations();       /* Davidson iterations taken   */
double res0   = get_last_ci_root_residual(0);   /* residual norm of root 0     */
double res1   = get_last_ci_root_residual(1);   /* ... root 1, etc.            */
```

- `get_last_ci_iterations()` — how many Davidson iterations the most recent solve
  ran (varies: a few when warm-started, up to `maxiter` when the Hamiltonian
  changed a lot).
- `get_last_ci_root_residual(root)` — that root's residual norm. `constdouble[4]`
  on return only carries the **root-averaged** residual; this getter gives the
  per-root values (returns 0.0 for an out-of-range index; capacity is 64 roots).

These are module globals set at the solve's exit, so read them right after the
synchronous `get_roots` call, before the next solve. They add no parameters and
need no allocation on your side — deliberately, so the array-size contract of
`constint`/`constdouble` did not have to change.

---

## 3. Requesting a spin state

Spin is the **last argument** of `get_roots`, `double target_spin`, and it
already exists — no new plumbing. It is the spin quantum number `S`:

| `target_spin` | meaning |
|---:|---|
| `-1.0` (any negative) | **no spin adaptation** — return the lowest `nroots` states of *any* multiplicity (this is `davidson`, the non-spin path) |
| `0.0` | singlet (target ⟨S²⟩ = 0) |
| `1.0` | triplet (target ⟨S²⟩ = 2) |
| `2.0` | quintet (target ⟨S²⟩ = 6) |
| `S`   | general: target ⟨S²⟩ = `S(S+1)` |

This is exactly the Python keyword mapping (`helper_PFCI.py`):

```python
spin_adaptation = "singlet" -> target_spin = 0.0
                  "triplet" -> target_spin = 1.0
                  "quintet" -> target_spin = 2.0
   (absent)               -> target_spin = -1.0   # "no", all states
```

`get_roots` dispatches on the sign:

```c
if (target_spin >= 0.0)  davidson_spin(... Sdiag, Sdiag_projection ... target_spin ...);
else                     davidson(...);   /* Sdiag / Sdiag_projection unused */
```

So:

- **No spin adaptation** (`target_spin < 0`): the `Sdiag` / `Sdiag_projection`
  arguments are ignored. You still pass valid pointers to satisfy the signature,
  but they need not be filled.
- **Spin-adapted** (`target_spin >= 0`): build `Sdiag` first with
  `build_S_diag(S_diag, num_alpha, nmo, N_ac, n_o_ac, n_o_in, shift)` where
  `shift = target_spin*(target_spin+1)`. The spin penalty and the ⟨S²⟩-based
  root selection inside `davidson_spin` use it. `Sdiag_projection` is only read
  by `first_order_spin_projection`, which is disabled in the active path, so a
  dummy array is fine there.

Nothing else in your call changes — same `constint`/`constdouble`/`table`/
`eigenvals`/`eigenvecs` you already pass for CASCI.

### The `constint` / `constdouble` contract (unchanged, for reference)

`constint` (length 9):
`[0]=N_ac` (active α electrons), `[1]=n_o_ac` (active orbitals),
`[2]=n_o_in` (inactive orbitals), `[3]=nmo`, `[4]=N_p` (photon-Fock truncation),
`[5]=indim`, `[6]=maxdim`, `[7]=nroots`, `[8]=maxiter` in / **0 on convergence** out.

`constdouble` (length 6):
`[0]=Enuc`, `[1]=dc` (dipole self-energy constant), `[2]=omega`,
`[3]=d_exp - d_diag`, `[4]=threshold` in / **root-averaged residual** out,
`[5]=E_core`.

---

## 4. From CASCI to CASSCF — how big a step?

CASCI = **one** CI eigensolve at fixed orbitals. CASSCF wraps an **orbital
optimization loop** around repeated CASCI solves. Concretely:

**What you already have for CASCI** (all in the C backend):
- the CI eigensolver — `get_roots` → `davidson_spin`/`davidson`;
- reduced density matrices — `build_active_rdm`, `build_active_photon_electron_one_rdm`;
- graph/string/diagonal setup — `get_graph`, `single_replacement_list`,
  `build_b_array`, `build_H_diag_cas_spin`, `build_S_diag`.

**What CASSCF adds** (an outer loop; per macroiteration: CI solve → build orbital
gradient/Hessian from the RDMs → trust-region orbital step → transform integrals
→ repeat):
1. **Orbital gradient + Hessian intermediates** (the `A`/`G` blocks) — tensor
   contractions of the RDMs with the two-electron integrals. Compute-heavy
   (dominant term ~`O(n_act⁴·nmo²)`); this is where TAMM pays off. *Not in the C
   backend* — it lives in the C++ port (`intermediates.cpp`) and the Python.
2. **Trust-region subproblem solvers** — GLTR, LSTRS, a Davidson-driven bordered
   solver, Steihaug-CG. These act on the **reduced orbital space**
   (`index_map_size`, ~`O(nmo²)`), which is *small* — dense/Eigen linear algebra,
   no TAMM needed. They can be lifted almost verbatim from the C++ port.
3. **Matrix-free orbital Hessian-vector product** — `build_sigma_reduced` in
   `orbital.c` (C), also reimplemented as `OrbitalSigmaOperator` in the C++ port.
4. **4-index integral transformations** after each orbital rotation —
   `full_transformation_macroiteration`, `full_transformation_internal_optimization`
   in `orbital.c`. Compute-heavy (`~O(nmo⁵)`); a TAMM target. *Already in the C
   backend you have.*
5. **Loop orchestration** — macroiteration / microiteration / internal-rotation
   accept-reject bookkeeping, QN dispatch. Pure control logic, small.

**The blueprint already exists.** The `cpp_casscf/` module in this repo is a
complete, tested, line-cited C++/Eigen port of all of the above (see
`cpp_casscf/README.md`). It was written specifically as the mechanically-portable
intermediate step before a TAMM retarget: the tensor-contraction kernels
(intermediates, `orbital_sigma`, the transforms) are kept in an explicit
loop-based form *on purpose*, because indexed loops map onto TAMM's tensor API
far more directly than fused BLAS/matmul chains. Those loop forms are the
intended TAMM-retarget reference.

**Rough effort split:**
- *Reuse almost as-is* (dense, small): the trust-region solvers, the loop
  orchestration, the gradient/Hessian assembly logic.
- *Retarget onto TAMM* (compute-heavy contractions): the intermediates (`A`/`G`),
  the orbital Hessian-vector product, and the integral transforms — the last of
  which you already have in `orbital.c`.
- *Unchanged*: the CI eigensolver and RDM builds you already use for CASCI.

So the step is **substantial but well-scoped and de-risked**: the new numerics
are the orbital-optimization layer, the expensive pieces are a handful of named
contraction kernels, and there is a faithful, tested reference for every one of
them in `cpp_casscf/`. Start from `cpp_casscf/README.md`'s status table and the
`MacroiterationDriver` collaborators (`CasscfCiStateAverageSolver`,
`CasscfIntegralTransformer`, `CasscfInternalOptimizationStep`,
`CasscfMicroiterationOptimizationStep`) — those four are the CASSCF outer loop.

### Still open (not yet done on our side)

A few CASSCF pieces are documented but incomplete; see the "What's still open"
section of `cpp_casscf/README.md`. The C-backend changes themselves are stable.
