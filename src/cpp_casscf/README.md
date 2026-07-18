# cpp_casscf

Eigen-based intermediate C++ port of the SA-QED-CASSCF orbital-optimization
trust-region subproblem solvers from `helper_PFCI.py`. Purpose: get a
correctness-checkable, mechanically-portable C++ layer in place before
retargeting onto TAMM's distributed tensor API.

## Status

| Component | State | Python source |
|---|---|---|
| `TrustRegionResult`, `Dimensions`, `TerminationReason` | Done | helper_PFCI.py:2319-2371, 6970-6991 |
| `HessianOperator` (dense / matrix-free / bordered) | Done | mv2 (16704), get_bfgs_mv (10792), aug_matvec (14566-14584) |
| `bordered_eigensolve` | Done | projection_step2, helper_PFCI.py:16621-16643 |
| `solve_lstrs_bisection` (shared core) | Done | shared bisection/hard-case algorithm behind both LSTRS solvers below -- see "Shared bisection core" |
| `PcgTrustRegionSolver` (Steihaug-Toint) | **Fully ported, tested** | solve_pcg_trust_region, helper_PFCI.py:16148-16264 |
| `LstrsSolver` (dense LSTRS + hard case) | **Fully ported, tested** | inline solver in internal_optimization3, helper_PFCI.py:6996-7548 |
| `DavidsonAugmentedHessianSolver` | **Fully ported, tested** (incl. subspace expansion, soft restart, hybrid preconditioning) | Davidson_augmented_hessian_solve6, helper_PFCI.py:15252-16085 |
| `gram_schmidt_orthogonalize` / `gram_schmidt_add` | Fully ported | ci_solver.c:3509-3583 |
| `inner_solve_pcg` (Jacobi-Davidson correction) | Fully ported | helper_PFCI.py:14455-14543 |
| `DavidsonDrivenLstrsSolver` | **Ported with 1 documented gap**, tested | outer bisection driver, helper_PFCI.py:11337-16085 |
| `GltrTrustRegionSolver` | **Fully ported, tested** | solve_gltr_with_operator, helper_PFCI.py:14783-15011 (solve_gltr_trust_region, 15015-15247, is a byte-for-byte duplicate modulo how it computes Hv -- both collapse into this one class via `HessianOperator`) |
| `select_trs_strategy` | **Confirmed** against real dispatch | helper_PFCI.py:11276-11292 (`if n_negative == 0: GLTR else: Davidson-driven LSTRS bisection`) |
| `should_reset_bfgs_reference` | Fully ported | condition at helper_PFCI.py:11118 |
| `QuasiNewtonPolicy` / `should_activate_qn` | Fully ported, made user-configurable per developer request | trigger at helper_PFCI.py:10417-10423, 11988-11994 |
| `build_unitary_matrix` | **Fully ported, tested** | helper_PFCI.py:6131-6164 |
| `MacroiterationDriver::run` (outer loop orchestration) | **Fully ported, tested** against mocked collaborators (see "documented gap" below) | helper_PFCI.py:2394-3060 |
| `minres_solve` (Paige/Saunders MINRES) | **Fully ported, tested** against real ill-conditioned chemistry data; used by `LstrsSolver`'s hard_case==2 | helper_PFCI.py:7547-7549 (`scipy.sparse.linalg.minres` call site) |
| `build_intermediates(_internal)`/`build_gradient(_and_hessian)`/`build_hessian_diagonal` | **Fully ported, tested** against real captured data | see "Intermediates building" section |
| `OrbitalHessianGuessProvider` (real `HessianGuessProvider`) | **Fully ported, tested** against real captured data | helper_PFCI.py:16614-16712 (`build_orbital_hessian_guess`/`hessian_guess`) |

### A naming correction from the first pass of this scaffold

Earlier iterations of this module called the small-dimension solver
"DavidsonLstrsSolver" and cited a `Davidson_augmented_hessian_solve3` as its
Python source. That function is **dead code**: it is only called from
`ah_orbital_optimization` (helper_PFCI.py:12241), which is itself never
invoked outside a commented-out line in the macroiteration driver
(helper_PFCI.py:2377-2378). The actual active solver for the small
active-inactive block is an *inline* LSTRS (Rojas/Santos/Sorensen) loop
directly inside `internal_optimization3` — that's what `LstrsSolver` ports.

The real large-dimension counterpart, used from `microiteration_optimization6`,
is `Davidson_augmented_hessian_solve6` — a genuinely different, subspace-based
algorithm (build a small guess subspace from the worst gradient/diagonal
ratios, solve the bordered eigenproblem on that subspace, check the true
matrix-free residual, expand the subspace if unconverged). That's what
`DavidsonAugmentedHessianSolver` ports -- now including the expansion loop,
not just the initial guess.

## Davidson subspace expansion loop

`DavidsonAugmentedHessianSolver` is now a full, stateful port of
Davidson_augmented_hessian_solve6 (helper_PFCI.py:15252-16085), not just its
"iteration 1" fast path. Per the developer, the active code path here is
settled -- the various "CORE BUG FIX" / "NEW HYBRID LOGIC" comments mark
retired experiments (older attempts left commented out for reference, not
signs of instability), and the two-phase preconditioning strategy (plain
diagonal Davidson preconditioning while no root has converged, a deflated
Jacobi-Davidson PCG correction via `inner_solve_pcg` once at least one has)
is a deliberate choice: it shortens the iteration count at the cost of more
expensive per-iteration correction solves.

Like the Python (which threads state through `self.sigma_total` /
`self.H_pp` / `self.H_qp` / `self.collapse_subspace_check` /
`self.idx_hessian` / `self.indim` across repeated `restart=True` calls), one
`DavidsonAugmentedHessianSolver` instance is stateful across an entire outer
beta-bisection sequence -- `DavidsonDrivenLstrsSolver` constructs exactly
one and reuses it for every bisection step, matching how the Python's
caller threads `guess_vector`/`restart` through repeated calls.

**Two real bugs were caught by testing this against a dense reference**
(both now fixed, see the git history / commit for detail if useful):

1. The Python's "converged in iteration 1" exit paths return the
   *unit-vector* basis `Q` (rebuilt unconditionally after the exit check,
   helper_PFCI.py:15521-15530), never the eigenvector-combination `Q` used
   just above it for the residual check. An initial port used the
   eigenvector-based one for what gets persisted/returned, which silently
   corrupted every subsequent `restart=True` resume (residual jumped from
   ~1e-16 to O(1) on the very next bisection step). Caught by a two-call
   regression test (`restart=false` then `restart=true` on the same simple
   problem) after the full-subspace-coverage tests didn't catch it (they
   only ever exercised the "exit within iteration 1" path once, never a
   resume).
2. The per-iteration sigma-vector build used the *bordered* operator
   (`BorderedHessianOperator`, which already folds in the `g*v0` border
   correction) where it should have used the plain reduced-space operator,
   then added the border correction again by hand right after -- silently
   double-counting the `g*v0` term. This broke the Rayleigh-Ritz variational
   guarantee (a projected Ritz value briefly went *below* the true minimum
   eigenvalue, which is mathematically impossible for a valid projection --
   that's what gave it away) specifically in the subspace *collapse* (soft
   restart) path, since that's the one place the corrupted sigma values fed
   into a subsequent eigendecomposition rather than just a residual check.

Both were only found by testing genuinely difficult cases (a small guess
subspace on a larger problem, forcing several expansion iterations and at
least one soft restart) rather than trusting the "does it converge in
iteration 1" tests alone -- worth keeping in mind if this solver is
extended further.

## Shared bisection core

Reading further confirmed that the *outer* beta-bisection loop around
`Davidson_augmented_hessian_solve6` (helper_PFCI.py:11337-16085, inside
`microiteration_optimization6`) is structurally identical to
`internal_optimization3`'s inline LSTRS loop that `LstrsSolver` ports --
same secular-equation bisection, same hard_case 1/2/3/4 resolution -- the
Python has this ~450-line algorithm written out twice, nearly verbatim. It
differs only in:

1. how the two lowest bordered-matrix eigenpairs are obtained at each trial
   alpha (dense eigh vs. Davidson subspace approximation),
2. how the interior/near-Newton ("hard_case==2") step is solved --
   internal_optimization3 calls `scipy.sparse.linalg.minres` directly on the
   dense `hessian_tilde_ai`; microiteration_optimization6 calls a custom
   `linear_equation_solve` with a MINRES fallback on a matrix-free operator.
   Both are iterative, not a direct solve -- `LstrsSolver`'s C++ port now
   matches the former exactly (`minres_solve`, see below); `DavidsonDrivenLstrsSolver`
   still substitutes plain CG for the latter (documented gap, unchanged), and
3. the quadratic-model evaluation for the hard_case==3 "quasi-optimal" test
   (dense vs. matrix-free Hessian application -- same formula either way).

Rather than port that duplication twice, `solve_lstrs_bisection()`
(`lstrs_bisection_core.hpp`) is the one shared implementation; `LstrsSolver`
and `DavidsonDrivenLstrsSolver` are now both thin wrappers that supply the
three differing pieces as callbacks. Cross-validated: on a hand-built
problem where the Davidson guess subspace is forced to cover the whole
space (so its eigenpairs are exact, not approximate), `DavidsonDrivenLstrsSolver`
reproduces `LstrsSolver`'s step to floating-point exactness.

`DavidsonDrivenLstrsSolver` has one remaining documented gap relative to the
Python (see its header doc comment for detail):

- **hard_case==2 uses plain matrix-free CG, not `linear_equation_solve`/MINRES.**
  The Python's custom `LinearRMSolver` (in `residual_minimization.py`,
  unread) with a MINRES fallback is replaced by reusing the already-tested
  `PcgTrustRegionSolver` at an effectively-unconstrained trust radius --
  mathematically the right substitute given hard_case==2 implies a
  (near-)PSD Hessian, but not a literal port.

Tested with a genuinely small guess subspace on a larger problem (forcing
real expansion iterations, not just "converges in iteration 1"), confirmed
to match `LstrsSolver`'s dense/exact reference answer.

## Confirmed solver dispatch

Traced directly in helper_PFCI.py (not inferred):

- **Small block** (active-inactive rotation, `internal_optimization3`):
  always `LstrsSolver`, dense, explicit bordered-matrix eigh per bisection
  step (dimension is small enough that this is cheap).
- **Full block** (`microiteration_optimization6`), when
  `qn_optimization == False` (helper_PFCI.py:11240-11292):
  - `n_negative == 0` (no non-positive reduced-Hessian-diagonal entries) →
    `solve_gltr_trust_region` (GLTR).
  - `n_negative > 0` → an outer beta-bisection loop, structurally the same
    shape as `LstrsSolver`'s loop, but calling
    `Davidson_augmented_hessian_solve6` in place of a direct dense `eigh` at
    each bisection step (helper_PFCI.py:11337-16085). Ported as
    `DavidsonDrivenLstrsSolver` (see "Shared bisection core" above for the
    two documented gaps).
- **Full block**, when `qn_optimization == True` (helper_PFCI.py:11114-11149):
  a *different* dispatch — GLTR against the exact, freshly-rebuilt reduced
  Hessian, or GLTR-with-operator against a running L-BFGS approximation,
  chosen by `should_reset_bfgs_reference()`.
- `qn_optimization` itself: starts off, latches on permanently once a small
  (`step_norm < 0.05`), energy-lowering step is taken (no auto-disable in the
  active code path). Per the developer, QN doesn't reliably help MCSCF
  convergence, so `QuasiNewtonPolicy::enabled` exposes this as a user-facing
  master switch (default true, matching the Python's de-facto behavior)
  rather than a purely automatic decision.

## Intermediates building (`intermediates.hpp`/`.cpp`)

The tensor-contraction-heavy code that produces the `A`/`G` blocks,
gradient, and Hessian diagonal the trust-region solvers consume is now
**fully ported and tested**:

| Function | Python source | Notes |
|---|---|---|
| `calculate_off_diagonal_photon_constant` | helper_PFCI.py:6105-6151 | direct port |
| `build_intermediates_internal` | helper_PFCI.py:5569-5747 | small block (occupied-space slices), used by `internal_optimization3` |
| `build_intermediates` | helper_PFCI.py:5748-5929 | full block, `full_space==True` only (the only value ever used); note `self.J`/`self.K` are `(n_occupied, n_occupied, nmo, nmo)`, **not** `(nmo, nmo, nmo, nmo)` -- see the header doc comment |
| `build_gradient` | helper_PFCI.py:6188-6203 | `full_space==True` branch only (the only one called, from `microiteration_optimization6`) |
| `build_gradient_and_hessian` | helper_PFCI.py:6282-6441 | `full_space==False` branch only (the only one called, from `internal_optimization3`); the `full_space==True` branch is both unreachable *and* dominated by ~O(n^6) nested-loop debug/`allclose` validation code with no bearing on the returned values, so it wasn't ported at all |
| `build_hessian_diagonal` | helper_PFCI.py:14416-14513 | direct port; reduction into `reduced_hessian_diagonal` reuses `build_index_map` |

Deliberately **not** ported: `build_intermediates2` (dead code -- every call
site is commented out, confirmed via grep) and `build_intermediates_with_blocks`
(referenced only in a commented-out line; the real `G_blocks` construction is
three reshaped slices of `build_intermediates`'s own `G` output, done inline
at the call site).

Two internal-representation surprises worth remembering if you're reading
the Python alongside this port (both documented in `intermediates.hpp`'s
doc comments):
- `self.J`/`self.K` are `(n_occupied, n_occupied, nmo, nmo)`, confirmed via
  a commented-out shape declaration at helper_PFCI.py:5480-5481 and
  `c_full_transformation_macroiteration`'s output array shapes -- easy to
  misread as the full `(nmo,nmo,nmo,nmo)` ERI tensor from the slicing
  syntax alone (an earlier pass of this reasoning did exactly that, and
  "resolved" what looked like a shape-mismatch bug in the Python before
  realizing the actual array shape made it consistent all along).
- `build_intermediates`'s active-active two-electron term in `A`
  (`"vwrt,tuvw->ru"`) uses a genuinely different index pattern than
  `build_intermediates_internal`'s analogous term (`"rtvw,tuvw->ru"`) --
  not the same formula under different variable names. Confirmed by tracing
  both term by term rather than assumed from the similar structure
  elsewhere, which is why these two functions were ported as two separate,
  literal translations rather than one shared core (unlike the LSTRS
  bisection, where the shared structure *was* verified to be exact).

Implemented as explicit nested index loops (matching each einsum's index
labels term by term) rather than chained `Eigen::Tensor::contract()`/
`shuffle()` calls, deliberately: some of the einsum strings relabel a
tensor's own axes in a way that's easy to get subtly wrong when composing
tensor-library primitives (e.g. `"tvuw"` applied to a tensor whose natural
storage order is `(t,u,v,w)` really does mean literal `D(t,v,u,w)` element
access, not a contraction over some `v`/`w` you'd have to derive the right
`shuffle()` for) -- with dimensions this small (orbital-space), explicit
loops cost nothing and are far easier to verify by eye against the source.

**Validated against real captured data**, same methodology as the
trust-region solvers (see "Validation against Python" below): the
`internal_intermediates_NNN`/`full_intermediates_NNN`/`build_gradient_NNN`/
`build_hessian_diagonal_NNN` dump categories capture every input and output
of these functions from a live run and replay them through this port. Last
full run: LiH -- 194/194 checks pass to ~1e-15-1e-17; H2O/6-31G (bigger
tensors, more active-space richness) -- 710/712, the only 2 failures being
the already-known, unrelated `internal_lstrs`/`davidson_lstrs` residuals
documented in "Sweep findings" (neither touches this code).

## `OrbitalHessianGuessProvider` (`hessian_guess.hpp`/`.cpp`)

**Fully ported and tested.** Faithful port of `hessian_guess` /
`build_orbital_hessian_guess` (helper_PFCI.py:16614-16712, the latter a
`@nb.njit`-compiled staticmethod) -- the real `HessianGuessProvider`
implementation `DavidsonAugmentedHessianSolver`/`DavidsonDrivenLstrsSolver`
were missing. Computes individual elements of the full (matrix-free)
reduced orbital Hessian at specific `index_map`-pair coordinates on demand,
rather than materializing the whole `index_map_size x index_map_size`
matrix -- this is what lets the Davidson solver build a small "guess
subspace" Hessian block cheaply. The index *selection* (ranking by
`|gradient_i / diagonal_i|`) already lived in `DavidsonAugmentedHessianSolver`
itself from an earlier session; this class only answers "what is the
Hessian element at these coordinates," exactly like the Python function it
ports.

Needs `sym_A_tilde = A_tilde_full + A_tilde_full.transpose()`
(helper_PFCI.py:15544), where `A_tilde_full` is `build_gradient`'s
`(nmo, n_occupied)` `A_tilde` embedded back into the full `(nmo, nmo)`
shape it's a slice of (virtual-orbital columns zero) -- `embed_and_symmetrize_A_tilde`
in `intermediates.hpp` does exactly that embedding, added alongside this.

**Validated against real captured data**: `hessian_guess_NNN/` dump cases
capture `U`/`sym_A_tilde`/`reduced_gradient`/`G`/the selected `idx` plus
Python's actual `guess_hessian`/`guess_gradient`. LiH: 4/4 checks (2 cases)
match to 0.0-1e-16. H2O/6-31G: 22/22 checks (11 cases) match to ~1e-15-1e-18.

## What's still open

- **The macroiteration/microiteration driver loop**
  (helper_PFCI.py:2394-3060): `MacroiterationDriver::run` is now a **fully
  ported, tested** faithful port of the loop's orchestration shape — the
  convergence latch (including the Python's never-reset `convergence` flag,
  reproduced exactly), the `n_in_a > 0` guard on `internal_step`, the
  restart branch ("RESTART MICROITERATION TO CORRECT INTERNAL ROTATION",
  helper_PFCI.py:2864-2896), and the H_spatial2/d_cmo/U_total rotation and
  accumulation. It's built entirely against the four collaborator interfaces
  (`CiStateAverageSolver`, `InternalOptimizationStep`,
  `MicroiterationOptimizationStep`, `IntegralTransformer`) declared in
  `macroiteration_driver.hpp` — none of the four has a real implementation
  yet (that needs the intermediates-building work below), so
  `test_macroiteration_driver.cpp` exercises the loop shape against mocks of
  all four.

  **Documented gap**: internal_optimization3 mutates `self.H_spatial2` /
  `self.d_cmo` / `self.U_total` directly on its own convergence
  (helper_PFCI.py:7787-7805), before control returns to the macroiteration
  loop. `MacroiterationDriver` has no channel for that — its own
  `H_spatial2`/`d_cmo`/`U_total` bookkeeping only reflects the rotations it
  applies directly (the restart-branch correction and the main
  microiteration step). A real `InternalOptimizationStep` implementation
  must share the *same* underlying storage as the driver's caller (e.g. via
  a shared context object) rather than relying on anything passed through
  this interface to carry its contribution forward — see the extended doc
  comment on `InternalOptimizationStep` in `macroiteration_driver.hpp`.

  Also deliberately not modeled: the fock_core/E_core rebuild that follows
  the JK transform in the Python (helper_PFCI.py:2990-3057, consumed by the
  next iteration's CI solve) — it depends on the J/K four-index ERI
  tensors, which this driver never holds. It belongs behind
  `IntegralTransformer::transform_macroiteration` as an implementation
  detail shared with `CiStateAverageSolver`, not threaded through
  `MacroiterationDriver::run`'s signature.
- **Real implementations of the 4 `MacroiterationDriver` collaborator
  interfaces** still don't exist, though the hard part underneath two of
  them (the intermediates-building math, see above) is now done:
  - `InternalOptimizationStep` / `MicroiterationOptimizationStep`: mostly
    plumbing at this point -- call the now-ported `build_intermediates*`/
    `build_gradient*`/`build_hessian_diagonal`, dispatch to the
    already-tested `LstrsSolver`/`GltrTrustRegionSolver`/
    `DavidsonDrivenLstrsSolver`/QN solvers, and actually resolve the
    documented `H_spatial2`/`d_cmo`/`U_total` shared-state gap above rather
    than just flagging it.
  - `CiStateAverageSolver`: needs `extern "C"` wrappers around `get_roots`/
    `build_H_diag_cas_spin` (ci_solver.c, already plain C, already
    ctypes-called from Python) plus `build_state_average_rdms`
    (helper_PFCI.py:7992+, itself a thin wrapper around `c_build_active_rdm`
    C calls) -- no new numerical work, just C++/C linkage.
  - `IntegralTransformer`: same story -- `c_full_transformation_macroiteration`/
    `c_full_transformation_internal_optimization` (orbital.c) are already
    plain C, just need wrapping.
  - `HessianGuessProvider` now has a real implementation
    (`OrbitalHessianGuessProvider`, see above) -- nothing left here for it.

`build_unitary_matrix` (`orbital_rotation.hpp`/`.cpp`, port of
helper_PFCI.py:6131-6164 -- builds `exp(R)` for the antisymmetric rotation
generator via eigendecomposition of `-R^2`) is a self-contained exception:
fully ported, wired into the build, and tested (orthogonality, det==1, exact
agreement with the closed-form 2x2 rotation, and the small-angle series
branch exercised via a genuinely decoupled virtual block rather than just a
numerically-tiny angle). It doesn't depend on the intermediates work above.

## Build

Eigen 3.4 is available via the `p4dev` conda env
(`$CONDA_PREFIX/share/eigen3/cmake` when that env is active). From this
directory:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/home/nvu12/miniforge3/envs/p4dev
cmake --build build -j
cd build && ctest --output-on-failure
```

If Eigen lives somewhere else, point `CMAKE_PREFIX_PATH` at that prefix (or
`Eigen3_DIR` directly at its `share/eigen3/cmake` directory).

## Validation against Python

The unit tests above check the ported solvers against synthetic/analytic
problems. Separately, `validation/dump_lih_case.py` + `tools/validate_against_python.cpp`
check them against **real** chemistry: a live, unmodified run of the pure-Python
`helper_PFCI.py` driver on a real molecule, with real intermediate values
captured to disk at a series of points and replayed through the
corresponding C++ port.

Two flavors of check, at different points in the pipeline:
- **Intermediates-building functions** (`build_intermediates*`/`build_gradient*`/
  `build_hessian_diagonal`, see above): dumps capture every *input* (J/K/H_spatial2/
  d_cmo/RDMs/...) and Python's actual output (A/G/gradient_tilde/hessian_tilde/
  hessian_diagonal), and this C++ port is run from scratch on those same
  inputs and compared directly -- a true from-the-ground-up reproduction,
  not just "given Python's own intermediate, does the next step match."
- **Trust-region solvers**: since the collaborator interfaces around them
  aren't wired up yet (see "What's still open"), these dumps instead
  intercept Python's already-built (Hessian, gradient, trust_radius) right
  before Python's own inline solver logic consumes it, and check that the
  ported C++ solver reproduces the same step -- decoupled from whether
  everything upstream of that point is ported, since (as of this port) it
  now actually is, but the dumps predate that and there's no need to change
  a validation methodology that already works.

**How it works**: `helper_PFCI.py` has a handful of `_dump_cpp_casscf_validation_case(...)`
calls, active only when `CPP_CASSCF_VALIDATION_DIR` is set (zero effect on
normal runs). Arrays of rank > 2 (e.g. the `J`/`K`/`G` tensors) can't go
through `np.savetxt` directly, so the dump helper flattens them (C-order)
into `<name>.txt` plus a `<name>.shape.txt` sidecar; `load_tensor4` on the
C++ side reads both back into a `Tensor4` via `TensorMap` (safe since
`Tensor4` is `RowMajor`, matching numpy's C-order exactly -- see
`tensor_types.hpp`).

- `internal_optimization3`'s `build_intermediates_internal` +
  `build_gradient_and_hessian` calls dump every input (occupied J/K/fock_core/
  d_cmo, the RDMs, `off_diagonal_constant`, `omega`, dims) plus both
  functions' outputs (`A1`/`G1`/`gradient_tilde1`/`hessian_tilde1`) as one
  `internal_intermediates_NNN/` case. Replayed by calling this port's
  `build_intermediates_internal` then `build_gradient_and_hessian` on the
  same inputs from scratch.
- `microiteration_optimization6`'s `build_intermediates` call dumps its
  inputs (`H_spatial2`/`d_cmo`/`J`/`K`/RDMs/`off_diagonal_constant`/`omega`/dims)
  and outputs (`A`/`G`) as `full_intermediates_NNN/`. Its `build_gradient`
  and `build_hessian_diagonal` calls each get their own
  `build_gradient_NNN/`/`build_hessian_diagonal_NNN/` case the same way.
- `Davidson_augmented_hessian_solve6`'s `build_orbital_hessian_guess` call
  (helper_PFCI.py:15551, `restart == False` only -- the guess subspace is
  only built once per outer bisection sequence) dumps
  `U`/`sym_A_tilde`/`reduced_gradient`/`G`/the selected `idx` plus Python's
  actual `guess_hessian`/`guess_gradient` as `hessian_guess_NNN/`. Replayed
  through `OrbitalHessianGuessProvider`.
- `internal_optimization3`'s inline LSTRS bisection (helper_PFCI.py:6996-7548)
  dumps `hessian_tilde_ai`/`gradient_tilde_ai`/`trust_radius`/`step` once the
  step is finalized. Replayed through `LstrsSolver`.
- `microiteration_optimization6`'s `n_negative == 0` dispatch to
  `solve_gltr_trust_region` (helper_PFCI.py:15045-15300ish, the
  `qn_optimization == False` path) materializes a dense Hessian by probing
  the real matrix-free `orbital_sigma3` contraction with unit vectors
  (cheap only because the test problem is tiny), and dumps that plus
  `reduced_hessian_diagonal`/`trust_radius`/`step`. Replayed through
  `GltrTrustRegionSolver`.
- `solve_gltr_trust_region` (and its near-duplicate `solve_gltr_with_operator`,
  see the QN cases below) injects random gradient noise in production
  ("`# NEW: Add microscopic noise to see hidden negative curvature`",
  helper_PFCI.py:15061/14901) via numpy's unseeded global RNG, specifically
  to steer away from exact trust-region hard cases (gradient orthogonal to
  the most-negative-curvature eigenvector). **An earlier version of this
  harness disabled that noise draw** so both sides would see an identical,
  reproducible input -- but sweeping across more molecules surfaced real
  step mismatches exactly on samples that were genuine hard cases with the
  noise turned off (see "Sweep findings" below). **Fixed**: the noise draw
  is left on (zero effect on production behavior), and the dump hook
  instead captures the actual realized post-noise gradient right after the
  draw (`self._cpp_casscf_last_noised_gradient`) and dumps *that* as
  `gradient`, not the pre-noise `reduced_gradient`. `validate_against_python`
  replays it with `GltrConfig::add_noise = false`, reproducing the literal
  vector Python operated on -- sidestepping the numpy-vs-`std::mt19937` RNG
  mismatch (never going to be bit-reproducible, see `gltr_trust_region_solver.cpp`'s
  own doc comment) without also removing the noise's hard-case-avoidance effect.
- `microiteration_optimization6`'s `n_negative > 0` branch (the outer
  beta-bisection loop around `Davidson_augmented_hessian_solve6`) dumps the
  same way (dense Hessian via unit-vector probing, plus
  `reduced_gradient`/`reduced_hessian_diagonal`/`trust_radius`/`hard_case`/`step`),
  gated on `n_negative > 0 and ||reduced_gradient|| > 1e-3` so it only fires
  for genuine Davidson-bisection solves, not the GLTR branch or the
  small-gradient Newton fallback that shares the same `step` variable.
  Replayed through `DavidsonDrivenLstrsSolver` -- see below for the
  validation strategy this uses.

**`n_negative > 0` still uses a different validation strategy than the
other two, even though `HessianGuessProvider` now has a real implementation
(`OrbitalHessianGuessProvider`, validated separately via `hessian_guess_NNN`
above).** Python's production path there uses a genuinely
subspace-approximate Davidson algorithm on a small guess subspace, so
there's still no way to make `DavidsonDrivenLstrsSolver` reproduce Python's
*exact* step here (that would need the guess subspace to be selected
identically, not just computed correctly once selected). Instead,
`validate_davidson_lstrs` follows the same methodology already established
in `test_davidson_driven_lstrs_solver.cpp` / `test_davidson_expansion_loop.cpp`:
cross-validate against `LstrsSolver`'s dense/exact answer under a guess
provider forced to cover the whole space (`DenseGuessProviderWithGradient`,
same synthetic-but-exact-coverage class as those tests, not
`OrbitalHessianGuessProvider`) -- now on the real, materialized chemistry
Hessian instead of a synthetic one. Python's actual step is also loaded and
printed for information (not pass/fail), since it isn't expected to match a
full-coverage run exactly. Swapping in `OrbitalHessianGuessProvider` with
the *actual* selected guess subspace (rather than forcing full coverage)
would let this cross-check the real Davidson expansion/subspace-selection
logic against Python's real intermediate steps, not just the final step --
a reasonable next enhancement, not done here.

- `microiteration_optimization6`'s `qn_optimization == True` dispatch
  (helper_PFCI.py:11144-11180) has two sub-branches, both dumped the same
  way (dense Hessian via unit-vector probing, plus
  `reduced_hessian_diagonal_zero`/`trust_radius`/`step`, plus the same
  post-noise-gradient capture described above):
  - `qn_gltr`: "solve step for original hessian" -- GLTR against the exact
    Hessian rebuilt at the QN reference point
    (`solve_gltr_trust_region` on `U_zero`/`A_tilde_zero`/`G_blocks_zero`,
    the *same* function as the `gltr_NNN` cases). Materialized by probing
    `orbital_sigma3` on those reference-point tensors.
  - `qn_bfgs`: "solve bfgs for updated hessian" -- GLTR-with-operator against
    a running L-BFGS approximation (`solve_gltr_with_operator` wrapping
    `get_bfgs_mv`, the *other* near-duplicate GLTR function,
    helper_PFCI.py:14850-15100ish). This one needed its own identical
    noised-gradient-capture edit since it isn't literally the same function.
    Materialized by probing `get_bfgs_mv` directly with unit vectors -- no
    need to port L-BFGS to C++ at all, it's just some `Hv` function from the
    outside once materialized.
  Both replayed through the exact same `validate_gltr` used for `gltr_NNN`.

**Not yet covered by this harness**: nothing else in the trust-region
solver dispatch -- all of `internal_optimization3` and both
`qn_optimization` states of `microiteration_optimization6` (GLTR,
Davidson-driven LSTRS, QN-exact-GLTR, QN-BFGS-GLTR) now have at least one
real captured instance.

**Running it**:

```sh
cd validation
python dump_lih_case.py --bond-length 1.6   # writes dumps_lih/{internal_lstrs,davidson_lstrs,gltr,qn_gltr,qn_bfgs}_NNN/
cd ../build
./validate_against_python ../validation/dumps_lih
```

`dump_lih_case.py` takes `--molecule lih|h2o`/`--bond-length`/`--basis`/
`--nact-orbs`/`--nact-els`/`--davidson-roots`/`--davidson-maxdim`/
`--davidson-indim`/`--omega`/`--dump-dir` if you want to sweep to a
different regime; R=1.6 (near LiH/STO-3G's equilibrium bond length) already
produces negative-curvature directions at some microiterations (3/11 at one
point) and QN activates partway through convergence (`step_norm < 0.05` and
an energy-lowering step), so no bond-stretching or parameter tuning was
actually needed to reach any of the branches on that first case. Note:
`davidson_maxdim`/`davidson_indim` are multiplied by `davidson_roots`
internally and must stay below `H_dim / davidson_roots` or the C Davidson
solver `sys.exit()`s (small active spaces with `davidson_roots > 1` need
these turned down, e.g. `--davidson-maxdim 3 --davidson-indim 2`).

`validation/sweep.sh` runs a set of these configs (LiH at 3 bond lengths,
LiH/6-31G at 2 active spaces including one with `n_in_a == 0`, a 2-root
LiH case, and H2O/6-31G at 1 and 2 roots) end to end and reports pass/fail
per config -- see its "Sweep findings" summary below for what turned up.

### Sweep findings

Sweeping past the single LiH/STO-3G case surfaced two genuine, systematic
failure modes. Both are now fixed or understood down to an intrinsic
mathematical limit; neither was a translation bug in the sense of "the C++
code doesn't implement what the Python does":

1. **`internal_lstrs` hard_case==2, ill-conditioned small block (fixed:
   `minres_solver.hpp`/`.cpp`).** All failures here were exactly the case
   where Python's `hard_case == 2` fallback fires (helper_PFCI.py:7545-7551):
   Python solves with `scipy.sparse.linalg.minres(H, -g, rtol=1e-5)`, which
   uses the Paige-Saunders MINRES algorithm's own internal (non-trivial,
   multi-quantity) stopping test -- not a simple `||residual|| / ||b|| < rtol`
   check -- and can legitimately stop well short of full convergence
   (confirmed: on the worst H2O case, condition number ~1410, scipy's
   minres reports `info=0` (converged) after 7 iterations with an actual
   relative residual of ~1.1%, nowhere near `1e-5`). `LstrsSolver`'s
   hard_case2 solve originally used `H.ldlt().solve(-g)` instead -- the
   *exact* solution -- on the stated assumption ("equivalent for the small
   dimensions this solver targets") that this wouldn't matter at small n.
   That held for LiH (internal block always 2x2, well-conditioned) but not
   for H2O's 8x8 block (condition number ~1000+), where early-stopped-MINRES
   and the exact solve are materially different vectors. **Fixed**:
   `minres_solve()` is a faithful, line-by-line port of scipy's actual
   MINRES algorithm and stopping criteria (istop codes 1-6, Acond/epsx/test1/test2,
   all of it) -- not a different iterative solver called with the same
   `rtol`, which doesn't reproduce scipy's specific early termination (Eigen's
   built-in `MINRES` was tried first and just reconverges to the exact
   answer in ~n iterations regardless of requested tolerance, since Krylov
   methods hit exact convergence in at most n iterations for n-dimensional
   systems). Unit-tested against a hardcoded real H2O case (`test_minres_solver.cpp`)
   where it reproduces Python's captured step to ~1e-10, and the exact solve
   is confirmed to differ by ~3e-3 -- i.e. this really is the case that
   mattered. After wiring it into `LstrsSolver`, full-sweep `internal_lstrs`
   failures dropped from ~9/config on H2O to 0-1/config, and the one
   remaining ~1e-6-level residual on the worst-conditioned case (cond ~1130)
   was confirmed to be ordinary floating-point accumulation over ~n Lanczos-like
   iterations (a *fresh* scipy `minres` call on that exact dumped (H,g) also
   reproduces Python's original step to `0.0`, so the C++ port is structurally
   correct -- Eigen's dense matvec vs numpy/LAPACK's just accumulate rounding
   slightly differently over the iteration).
2. **`gltr`/`qn_gltr`/`qn_bfgs` near a genuine trust-region hard case
   (fixed, with one now-understood residual).** The first sweep pass showed
   failures with a consistent signature: the Hessian's lowest eigenvalue
   negative (or ~0), and the gradient's projection onto that eigenvector
   ~1e-9-1e-17 -- i.e. numerically exactly the textbook hard-case condition.
   That's precisely what Python's "`# NEW: Add microscopic noise to see
   hidden negative curvature`" hack (both GLTR variants) exists to steer
   away from in production -- and the validation dump hook at the time
   *disabled* that noise so both sides would see an identical, reproducible
   input, which removed the very mechanism Python relies on to avoid the
   ill-posed subproblem. **Fixed**: the noise draw is left on (matching
   production exactly), and the dump hook instead captures the actual
   realized post-noise gradient and replays *that* through a noise-disabled
   C++ solve -- see the noise-handling bullet above. This eliminated the
   large majority of these failures. **One intrinsic residual remains**: on
   rare samples where the gradient norm itself is small (so
   `noise_scale = 1e-6 * ||g||` is also tiny, e.g. ~8e-9), the random noise
   draw can *by chance* still leave the perturbed gradient's projection onto
   the hard-case eigenvector at the ~1e-9 level -- i.e. even Python's own
   noise-injection heuristic doesn't guarantee escaping the hard case, only
   reduces how often it happens. Confirmed on the 2 residual failures: not a
   capture bug (the dumped gradient genuinely has a tiny `|g.v0|`), and
   Python's own step in this regime isn't fully trustworthy either (one
   observed case had `||step|| = 0.652` against a `trust_radius = 0.5` --
   even production Python's hard-case construction slightly overshoots the
   trust region here). This is a property of the underlying algorithm under
   near-exact degeneracy, not a fixable implementation gap in either language.

Sweep pass rate after both fixes: 446/451 cases across 8 configs (up from
430/450, itself up from 439/457 before any fix). All 5 remaining failures
are the two understood residuals above (floating-point accumulation on the
worst-conditioned MINRES case; genuine near-exact hard-case degeneracy
surviving the noise heuristic by chance) plus 2 `davidson_lstrs` "vs dense
LSTRS" cross-check hits that shifted for an unrelated reason: that
cross-check's *reference* solver (`LstrsSolver`) now also uses `minres_solve`
for its own hard_case==2 (previously an exact solve), so a comparison that
used to be "CG vs exact solve" is now "CG vs early-stopped-MINRES" --
`DavidsonDrivenLstrsSolver`'s hard_case==2 branch still uses plain CG (its
own pre-existing, separately documented gap, see
`davidson_driven_lstrs_solver.hpp`), unaffected by this session's changes.
See `validation/sweep.sh` to reproduce.

**Result as of last run** (LiH/STO-3G, 2 electrons in 2 active orbitals, QED-coupled,
`omega=0.1`, small `lambda`, 1 state, converged in 4 macroiterations): all
26 strict pass/fail cases (3 `internal_lstrs`, 3 `gltr`, 2 `davidson_lstrs`
vs. the dense LSTRS reference, 7 `qn_gltr`, 11 `qn_bfgs`) passed --
`internal_lstrs`/`gltr`/`qn_gltr`/`qn_bfgs` matched Python's actual step to
~1e-16-1e-21 (floating-point exactness -- including the BFGS-operator case,
confirming the materialize-then-replay approach works even when the
underlying operator has no tensor-contraction structure at all),
`davidson_lstrs` matched the dense reference to ~1e-13/1e-14 under forced
full guess coverage. The 2 informational-only comparisons (Python's actual,
subspace-approximate Davidson step vs. the dense reference) also landed at
~1e-13/1e-14 in this run -- i.e. production Davidson was already essentially
exact on a problem this small, not just "close." `validate_against_python`
is not wired into `CMakeLists.txt`'s `ctest` targets since it needs an
external, machine-specific dump directory as input rather than being
self-contained; it's built (`cmake --build build`) but run manually.

## Layout

```
include/casscf/
  types.hpp                             Dimensions, TrustRegionResult, TerminationReason
  hessian_operator.hpp                  HessianOperator, DenseHessianOperator,
                                         MatrixFreeHessianOperator, BorderedHessianOperator
  bordered_eigensolve.hpp               shared bordered-matrix eigh (projection_step2 port)
  gram_schmidt.hpp                      modified Gram-Schmidt (ci_solver.c port, implemented, tested)
  jacobi_davidson_correction.hpp        inner_solve_pcg port (implemented, tested)
  lstrs_bisection_core.hpp              shared LSTRS bisection/hard-case algorithm (implemented, tested)
  trust_region_solver.hpp               abstract TrustRegionSolver interface
  pcg_trust_region_solver.hpp           Steihaug-Toint PCG (implemented, tested)
  lstrs_solver.hpp                      dense LSTRS wrapper around the shared core (implemented, tested)
  davidson_augmented_hessian_solver.hpp full stateful subspace Davidson-LSTRS (implemented, tested)
  davidson_driven_lstrs_solver.hpp      large-dim LSTRS wrapper around the shared core (implemented, tested)
  gltr_trust_region_solver.hpp          Lanczos + secular equation (implemented, tested)
  solver_selector.hpp                   TrsStrategy dispatch (confirmed) + BFGS-reference-reset predicate
  quasi_newton_policy.hpp               user-configurable QN activation policy
  orbital_rotation.hpp                  build_unitary_matrix (implemented, tested)
  macroiteration_driver.hpp             outer loop orchestration + 4 collaborator interfaces
                                         (loop implemented and tested; collaborators not yet implemented --
                                         see "What's still open")
  minres_solver.hpp                     faithful port of scipy.sparse.linalg.minres (implemented, tested);
                                         used by LstrsSolver's hard_case==2
  tensor_types.hpp                      Tensor2/3/4 (Eigen::Tensor, RowMajor), Matrix<->Tensor2 helpers,
                                         build_index_map (shared rotation-parameter-pair enumeration)
  intermediates.hpp                     build_intermediates(_internal)/build_gradient(_and_hessian)/
                                         build_hessian_diagonal/embed_and_symmetrize_A_tilde -- the
                                         A/G/gradient/Hessian-diagonal tensor-contraction math
                                         (implemented, tested; see "Intermediates building" above)
  hessian_guess.hpp                     OrbitalHessianGuessProvider, the real HessianGuessProvider
                                         (implemented, tested; see that section above)
src/                                    corresponding .cpp files
tests/
  test_pcg_trust_region.cpp             analytic smoke tests (interior/boundary/negative-curvature)
  test_lstrs_solver.cpp                 cross-validated against PCG on the same 3 problems
  test_davidson_augmented_hessian_solver.cpp  full-subspace-coverage check against direct bordered_eigensolve
  test_gltr_trust_region_solver.cpp     cross-validated against LSTRS on the same 3 problems + zero-gradient case
  test_davidson_driven_lstrs_solver.cpp full-subspace-coverage check against dense LstrsSolver reference
  test_davidson_expansion_loop.cpp      small guess subspace on a larger problem -- forces real expansion
                                         iterations + a soft restart, cross-validated against LstrsSolver
  test_orbital_rotation.cpp             orthogonality/det==1, closed-form 2x2 case, small-angle series branch
  test_macroiteration_driver.cpp        loop shape against mocks of all 4 collaborators: convergence latch,
                                         n_in_a==0 skip, H/d_cmo/U_total bookkeeping, restart branch
  test_minres_solver.cpp                well-conditioned sanity case + a hardcoded real ill-conditioned
                                         H2O hard_case==2 Hessian, matches Python's actual captured step
tools/
  validate_against_python.cpp           replays real Python-captured solver instances (see "Validation
                                         against Python") -- standalone tool, not a ctest
validation/
  dump_lih_case.py                      runs a real LiH or H2O SA-QED-CASSCF through helper_PFCI.py with
                                         the validation dump hooks on (--molecule/--bond-length/--basis/
                                         --nact-orbs/--nact-els/--davidson-roots/--davidson-maxdim/
                                         --davidson-indim/--omega/--dump-dir)
  dumps_lih/                            captured (hessian, gradient, trust_radius, step) instances from
                                         the last dump_lih_case.py run
  sweep.sh                              runs a set of geometries/active-spaces/molecules through
                                         dump_lih_case.py + validate_against_python and reports
                                         per-config pass/fail (dumps_sweep_*/ output is gitignored,
                                         regenerate via this script -- see "Sweep findings")
```
