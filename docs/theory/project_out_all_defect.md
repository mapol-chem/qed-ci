# BUG: `project_out_all` breaks unequal-weight SA-CASSCF gradients

Found 2026-09-08, from the developer's own theoretical argument.  **This reverses
the earlier "the unequal-weight gradient is correct" verdict in RESULTS.md**,
which was measured only on symmetric geometries where the defect is invisible.

## The defect

`nuclear_grad.py:1447` and `:1466` (inside `solve3`'s trial/sigma helpers — the
LIVE solver; `solve2` is dead, its call at `:2255` is commented out) project each
CI response vector `z^(n)` against **every** state-averaged CI vector:

```python
temp3 = self.project_out_all(temp3, self.eigenvecs)
#self.projection(sigma[self.index_map_size:])     # <- the minimal form, disabled
```

The theory requires only `z^(n)·c^(n) = 0`.  Projecting out `c^(m)`, m≠n as well
discards a component that is **physical whenever the weights are unequal**.

## Why it is weight-dependent

Project the CI stationarity condition onto `c^(m)`, m≠n.  For a gradient
`ỹ^(n) = Hc^(n) = E_n c^(n)`, so `c^(m)·ỹ^(n) = 0` and

    (E_m - E_n) t_mn + 2<c^(m)|H~|c^(n)> = 0 ,   t_mn = c^(m)·z^(n)

`H~` is symmetric so `t_nm = -t_mn`.  These components reach the gradient only
through `D_bar^SA = ½ Σ_n W_n (z^(n)⊗c^(n) + sym)`, where the (m,n) pair
contributes

    W_n t_mn + W_m t_nm = (W_n - W_m) t_mn

**Equal weights → cancels identically → projecting is free (and legitimate:
those directions are then a genuine gauge freedom of the gradient).
Unequal weights → survives → projecting destroys a real term.**

Equivalently: under `z^(n) → z^(n)+θc^(m)`, `z^(m) → z^(m)-θc^(n)` the gradient
shifts by `θ(W_n - W_m)<c^(m)|∂H/∂x|c^(n)>`.

## Why it went unnoticed: MOLECULAR SYMMETRY HIDES IT

`H~` is totally symmetric, so `<c^(m)|H~|c^(n)>` vanishes between states of
different spatial symmetry.  Measured on H2O CAS(6,5)/6-31G, weights 5,2,1:

| geometry | `<c^(1)|H~|c^(2)>` | CI residual ‖res[1]‖, ‖res[2]‖ |
|---|---|---|
| C2v (the real molecule) | ~3e-11 — forbidden | 3.4e-08, 1.8e-08 (converged) |
| C1 (one H displaced) | **-5.13e-03** | **5.13e-03, 5.13e-03** (NOT converged) |

On C2v there is nothing for the projection to discard, so the gradients are
right.  On C1 the projection leaves the CI equation **unsatisfied**, entirely in
the c^(1)-c^(2) direction: `t` actual ~1e-16 against `t` predicted **+5.25e-02**.

## The measurement

H2O CAS(6,5)/6-31G at the **C1** geometry, vs Molpro (`noextra`), max |Δgradient|:

| weights | state | `project_out_all` (shipped) | minimal projection |
|---|---|---|---|
| 1,1,1 | 0 / 1 / 2 | 6.5e-09 / 3.3e-08 / 2.8e-08 | 2.3e-08 / 3.2e-08 / 3.1e-08 |
| **5,2,1** | 0 | **1.366e-03** | **1.66e-08** |
| **5,2,1** | 1 | **3.178e-04** | **5.59e-08** |
| **5,2,1** | 2 | **7.467e-03** | **7.14e-08** |

Equal weights: the two agree to 2e-08, exactly as `(W_n - W_m) = 0` predicts.
Unequal weights: the shipped code is wrong by **1e-3 to 7e-3**; the minimal
projection restores agreement with Molpro to ~1e-8.

## The fix

In `solve3`'s two helpers, replace `project_out_all(x, self.eigenvecs)` with the
per-state projection already written and commented out beside it
(`self.projection`, `nuclear_grad.py:867-879`), which enforces `z^(n)·c^(n)=0`
and nothing more.  **Not applied — this is the developer's call.**  It is safe
for equal weights (verified above), so it is not a regression risk there.

Optionally keep `project_out_all` for the equal-weight path only, but there is no
measured reason to: the minimal projection matches Molpro in both regimes.

## Scope — what this invalidates

* **Every unequal-weight gradient at a geometry with no usable symmetry is
  affected.**  That includes conical-intersection scans, where the symmetry that
  hides the bug is exactly what is broken.
* The earlier five-weighting H2O validation and the C2H4 validation were on
  **symmetric** geometries (C2v, D2h) and so did NOT exercise this.  They remain
  valid statements about those geometries, and say nothing about the general case.
* **Derivative coupling is untested and is the other place to look**: there
  `Y_z ≠ 0` and `project_out_all` acts on it at `nuclear_grad.py:677` as well.

## Reproduce

`tree_projall_ab/` is a copy of the tree with the two `solve3` sites switchable
via `QED_PROJ_ALL` (1 = shipped, 0 = minimal) plus residual/coupling
diagnostics; `ours/run_case_projall.py` points at it.  The working tree was
never modified.

```sh
cd ours
for W in 1,1,1 5,2,1; do for PA in 1 0; do
  QED_PROJ_ALL=$PA python run_case_projall.py --system h2o_c1 --weights $W \
      --tag v2_h2oc1_w${W//,/}_pa${PA}
done; done
```
