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
2. how the interior/near-Newton ("hard_case==2") step is solved (dense
   direct solve vs. an iterative matrix-free solve), and
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

## What's still open

- **`HessianGuessProvider`** (needed by `DavidsonAugmentedHessianSolver`, and
  transitively by `DavidsonDrivenLstrsSolver`) has
  no real implementation yet — it depends on `build_orbital_hessian_guess`
  (helper_PFCI.py:16413-16450+), which in turn depends on the
  intermediates-building tensor contractions below.
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
- **Intermediates building** (`build_intermediates`, `build_intermediates2`,
  `build_intermediates_with_blocks`, `build_gradient_and_hessian`, RDM
  contractions): the tensor-contraction-heavy code that produces the
  `A_tilde`/`G` blocks, gradient, and Hessian diagonal these solvers consume.
  Untouched. Probably the best candidate to prototype directly against TAMM
  tensors once the solver layer above is stable, since Eigen has no native
  distributed-tensor story.

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
```
