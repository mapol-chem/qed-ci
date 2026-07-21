# QN/BFGS + internal-step-restart review

Re-review of the quasi-Newton (BFGS-accelerated) microiteration step and the
macroiteration internal-step-norm-correction restart, checked against the
principle in `optimization_note.md` (§4 QN, §3 internal step) and `bfgs.md`
(Powell damping). Scope: does the implementation reflect the stated
principle, and is the underlying Python correct? Conclusion up front: the
C++ port is a **faithful mirror of the Python**, and the BFGS math it ports
is **correct**; but the **Python itself omits several robustness safeguards
the notes call for**. None of these are ported-vs-Python discrepancies — they
are principle-vs-Python gaps, shared by both by construction. Per the
faithful-port policy, the port was **not** changed to "fix" them; they are
documented here for a decision on whether to change the Python.

## What is correct (verified from scratch, not assumed)

- **The direct-BFGS recursive Hessian-vector product** (`get_bfgs_mv`,
  helper_PFCI.py:10904-10953 / `BfgsOperator::apply`) is the correct
  recursive form of the direct (Hessian, not inverse) BFGS update,
  Kreplin Eq. 63: `sigma += y (yᵀv) rho_y - Bs (sᵀ·sigma) rho_Bs`, where the
  `sᵀ·sigma` in the second term uses the **running** `sigma = B_k·v` (all
  prior history applied), not the original query `v`. This matches the
  standard `B_{k+1} = B_k - (B_k s sᵀ B_k)/(sᵀB_k s) + (y yᵀ)/(yᵀs)` applied
  to `v`. (`optimization_note.md`'s "final expression" writes a flat sum
  against the original `v` — that is an oversimplification in the note; the
  code is the correct running form.)
- **Powell damping** (helper_PFCI.py:11200-11217 / `BfgsOperator::update`)
  is algebraically correct. With `theta = (1-σ)·sBs/(sBs - ys)`, the damped
  `y_bar = theta·y + (1-theta)·Bs` satisfies `sᵀ·y_bar = σ·sBs` exactly
  (verified by hand), so the stored `rho_y = 1/(σ·sBs)` is exactly
  `1/(sᵀ·y_bar)` — i.e. the update term `(y_bar y_barᵀ)/(sᵀy_bar)` is right.
- **The QN wiring is faithfully ported**: activation trigger (`step_norm <
  0.05` on an accepted, energy-lowering step, §4's ε=0.05), the two-condition
  reset dispatch (reference-point refresh at 11227 vs. solve-exact-vs-solve-
  BFGS dispatch at 11262), and the fact that the QN step is **unconditionally
  accepted** (no accept/reject test in the QN branch — the trust region is
  enforced only inside the GLTR subproblem solve).
- **The internal-step-restart** (macroiteration_driver, helper_PFCI.py:
  2938-3004) faithfully implements §3: peel the internal (active-inactive)
  rotation `R_ai` off the antisymmetric part `½(U₂-U₂ᵀ)`, apply it to the
  integrals, fold it into `U_total`, then restart the microiteration from the
  complementary external (virtual-inactive / virtual-active) rotation.

## Principle-vs-Python gaps (present in BOTH Python and the faithful port)

1. **No tiny-denominator guard on the BFGS update.**
   `optimization_note.md` §4: *"Omit the BFGS update if one of the
   denominators becomes tiny."* The Python (helper_PFCI.py:11200-11221)
   divides by `ys_dot`, `sBs_dot`, and `(sBs_dot - ys_dot)` **with no
   guard** and always appends the entry. If `sBs_dot ≈ 0` (near-singular
   curvature along `s`) or `sBs_dot ≈ ys_dot`, `rho_Bs`/`theta`/`rho_y` blow
   up and inject a garbage history entry that then corrupts every subsequent
   `get_bfgs_mv`. Faithfully reproduced in `BfgsOperator::update` (no guard).

2. **`consecutive_skips` is vestigial / dead code — the skip-and-reset
   robustness mechanism is entirely non-functional.** The reset-dispatch
   condition (helper_PFCI.py:11262) includes `... or self.consecutive_skips
   >= 3` (restart BFGS after 3 consecutive skipped updates), but
   `consecutive_skips` is **only ever set to 0** (init 11012, reset 11298) —
   it is **never incremented anywhere** (confirmed by grep). So: the Python
   never actually skips an update (see gap 1), never counts skips, and the
   `>= 3` clause is dead. The variable + the `>= 3` clause are clearly
   vestiges of the §4 mechanism "omit update if denominator tiny → count
   skips → recompute the exact Hessian and restart BFGS after repeated
   skips" — designed but never wired up. The port documents `consecutive_skips`
   as a confirmed-dead local and its `>= 3` clause as dead, matching this.

3. **No CI-residual precondition before activating QN.** §4: *"Pre-condition:
   ensure that the norm of the CI residual becomes substantially lower than
   the last orbital gradient to justify the approximation g_c = 0."* The
   Python activates QN on `step_norm < 0.05` alone (helper_PFCI.py:12222) —
   it never checks the CI residual against the orbital gradient. So QN (which
   drops the CI-orbital coupling gradient `g_c`) can switch on while the CI
   residual is still large, i.e. before its justifying approximation holds.
   `current_residual` is tracked (11003, 12433) but only used in the
   total-norm convergence test, not gated into QN activation.

4. **`B₀` is the raw (possibly indefinite) exact Hessian, not the PD
   `abs(diagonal)` `B₀` `bfgs.md`'s guarantee assumes.** `bfgs.md` point 1:
   *"By forcing B₀ to be PD (abs(diagonal)) and using Damping, the resulting
   operator B is mathematically guaranteed to be Positive Definite."* But the
   active Python uses `B₀·v = orbital_sigma3(...)` — the **exact** reduced
   orbital Hessian, which near a saddle is **indefinite** (`n_negative > 0`).
   The `abs(diagonal)` B₀ that would make the guarantee hold is present but
   **commented out** (helper_PFCI.py:10929-10932). Consequence: `sᵀB₀s` can
   be `< 0`, so the Powell-damping precondition (`sBs > 0`) is not guaranteed,
   and the damped-update sign structure (`rho_Bs = 1/sBs < 0`) can flip. In
   practice this is masked because the step is computed by **GLTR**, which
   regularizes an indefinite operator with a shift λ (bfgs.md point 2
   acknowledges this) — so using the exact Hessian as `B₀` is a deliberate,
   defensible accuracy choice, but the clean PD guarantee `bfgs.md` states
   does **not** hold, and gap 1 (no denominator guard) is exactly what would
   catch the `sBs ≤ 0` cases. Faithfully reproduced (the port's `B₀` is the
   exact `OrbitalSigmaOperator`/`orbital_sigma3`).

5. **Cosmetic:** the damping print string (helper_PFCI.py:11203) says
   `< 0.2 *` but the actual `sigma = 0.1` (line 11201). Stale string only;
   the port uses `0.1` correctly (`bfgs_operator.cpp`, `damping_sigma = 0.1`).

## Internal-step-restart: one conceptual note (not a bug)

The restart splits `U₂ ≈ exp(R)` as `exp(R_int)·exp(R_ext)` (integrals get
`exp(R_int)`; microiteration restarts from `exp(R_ext)`). This is only a
first-order split — BCH leaves an `O([R_int, R_ext])` error — and the restart
warm-starts from `exp(R_ext)`, not literally "the previously obtained U
matrix" as `optimization_note.md` §3 phrases it. Neither is a correctness
bug: the microiteration re-optimizes to its own minimum regardless of the
warm-start point, so the split affects only convergence speed, and the
first-order residual is absorbed by the outer macroiteration loop. The
threshold for triggering the restart is a fixed `1e-4` (§3 says only "a
certain threshold"). The port mirrors all of this faithfully.

## Recommendation

If these are to be addressed, do it **in the Python first** (the port should
keep mirroring it), most-impactful first: (1) add the tiny-denominator guard
+ actually wire `consecutive_skips` (gaps 1+2 are one mechanism), then
(4) decide `B₀` = exact-Hessian (keep, rely on GLTR) vs. the commented-out
PD `abs(diagonal)` and make the choice explicit, then (3) the CI-residual
precondition. The port has a clean seam for each: `BfgsOperator::update`
(guard + skip signalling), `BfgsOperator`'s `B₀` source, and the QN
activation predicate.
