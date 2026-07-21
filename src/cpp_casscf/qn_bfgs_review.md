# QN/BFGS + internal-step-restart review

Re-review of the quasi-Newton (BFGS-accelerated) microiteration step and the
macroiteration internal-step-norm-correction restart, checked against **the
paper** (Kreplin et al., the authoritative source) and, secondarily, the
`optimization_note.md`/`bfgs.md` notes. Important caveat, flagged by the
developer: **`bfgs.md` is a Gemini-compiled note and is NOT authoritative** —
in particular it describes a *diagonal* / `abs(diagonal)` initial Hessian
`B_0`, which is **not** what the paper intends; the paper uses the **exact
initial orbital Hessian** `h̃_oo^0`. Where this review's earlier revision
raised concerns rooted in `bfgs.md`, those are corrected below.

Conclusion up front: the C++ port faithfully mirrors the Python, the BFGS
math is correct, and — checked against the *paper* — the Python implements
the QN algorithm as specified, with **one** genuine omission (the tiny-
denominator guard) and one piece of dead vestigial code.

## What is correct (verified against the paper)

- **The Hessian-vector product is exactly Kreplin's Eq. 63.** The paper's
  form is the flat sum
  `h̃^k q = h̃^0 q + Σ_i v1^i(v1^i·q)/d1^i - Σ_i v2^i(v2^i·q)/d2^i`
  with `v1=y`, `v2 = h̃^{k-1} s`, `d1 = v1·s`, `d2 = v2·s`. The Python
  (`get_bfgs_mv`, helper_PFCI.py:10904-10953 / `BfgsOperator::apply`) instead
  accumulates a **running** `sigma` and uses `s·sigma` in the second term
  rather than `v2·q` against the original `q`. These are **provably equal**:
  by induction the running `sigma` after `i-1` entries is `h̃^{i-1} q`, and
  since each intermediate Hessian is symmetric,
  `s_i·sigma_{i-1} = s_i·(h̃^{i-1} q) = (h̃^{i-1} s_i)·q = v2^i·q`. So the flat
  form (as written in the paper and `optimization_note.md`) and the Python's
  running form compute the identical result — the running form is not an
  approximation of the flat one. `h̃^0 q` is the **exact** initial orbital
  Hessian applied to `q` (via `orbital_sigma3` / `OrbitalSigmaOperator`),
  matching the paper's "`h̃^0 q` is the product of the initial orbital Hessian
  with a Q-space vector `q`."

- **`B_0` = the exact initial orbital Hessian is correct, per the paper.**
  The paper uses `h̃_oo^0`, the exact orbital Hessian, as the base term (not a
  diagonal). The Python does exactly this (`orbital_sigma3` at the frozen
  reference point). The `abs(diagonal)` `B_0` in `bfgs.md` — and the concern
  the previous revision of this review raised about `B_0` not being PD — do
  **not** apply: they came from the non-authoritative note, not the paper. An
  `abs(diagonal)` `B_0` is present but commented out in the Python
  (helper_PFCI.py:10929-10932); it is deliberately not used. (Developer's
  intent: keep the exact indefinite `B_0`, and additionally recompute/reset to
  the exact Hessian after BFGS iterations that encounter negative curvature —
  a planned strengthening of the paper's "recompute the Hessian and restart
  BFGS if the energy increases," not a defect in the current code.)

- **The CI-residual precondition IS implemented.** The paper: *"we ensure
  that the norm of the CI residual becomes substantially lower than the last
  orbital gradient to justify the approximation g_c = 0."* The Python does
  exactly this: when `qn_count > 0`, the CI Davidson convergence tolerance
  passed to `c_get_roots` is set to `0.1 * ||reduced_gradient||`
  (helper_PFCI.py:12402, `self.constdouble[4] = 0.1 *
  np.linalg.norm(reduced_gradient)`; `constint[8] = 10000` lifts the CI
  iteration cap so it can actually reach that tolerance). So each QN
  iteration's CI solve is driven to a residual an order of magnitude below
  the orbital gradient — precisely the precondition. Faithfully ported:
  `microiteration_optimization_step.cpp:578-581`
  (`davidson_threshold_override = 0.1 * reduced_gradient.norm()`,
  `davidson_maxiter_override = 10000` when `qn_count > 0`).
  *(This corrects the previous revision of this review, which wrongly listed
  the precondition as absent — it is present, via the CI solve tolerance, not
  a gate on QN activation.)*

- **Hessian not recomputed each microiteration; restart on density/energy
  change — both per the paper.** The initial Hessian is frozen and its change
  under `ΔR` is carried by the BFGS update (the reference point is only reset
  under the conditions below). The paper: *"Only if the averaged density
  strongly changes or the energy increases in an iteration, the orbital
  Hessian is recalculated and the BFGS method is restarted."* The Python's
  reset condition (helper_PFCI.py:11227) is `density_norm_change > 0.025`
  (density strongly changes) `or predicted_energy > 0` (energy increase proxy)
  `or qn_count == 1` (first-iteration bootstrap of the reference point) — a
  faithful implementation. Ported in `should_reset_bfgs_reference_point`.

- **Powell damping** (helper_PFCI.py:11200-11217 / `BfgsOperator::update`),
  where used, is algebraically correct: with `theta = (1-σ)·sBs/(sBs - ys)`,
  the damped `y_bar = theta·y + (1-theta)·Bs` gives `sᵀ·y_bar = σ·sBs`, so the
  stored `rho_y = 1/(σ·sBs) = 1/(sᵀy_bar)` is exact. (See the note in the gaps
  section on how this relates to the paper.)

- **The QN wiring is faithfully ported**: activation trigger
  (`step_norm < 0.05` on an accepted, energy-lowering step), the reset/dispatch
  conditions, and the fact that the QN step is unconditionally accepted (the
  trust region is enforced only inside the GLTR subproblem solve).

- **The internal-step-restart** (macroiteration_driver, helper_PFCI.py:
  2938-3004) faithfully implements the internal-rotation correction: peel the
  internal (active-inactive) rotation `R_ai` off the antisymmetric part
  `½(U₂-U₂ᵀ)`, apply it to the integrals, fold it into `U_total`, then restart
  the microiteration from the complementary external (virtual-inactive /
  virtual-active) rotation. The `exp(R_int)·exp(R_ext)` split is first-order
  (BCH), and the restart warm-starts from `exp(R_ext)` — but since the
  microiteration re-optimizes to its own minimum, this affects only
  convergence speed, not the converged answer, and the first-order residual is
  absorbed by the outer macroiteration loop. Not a correctness issue.

## The one genuine gap vs. the paper

- **The tiny-denominator guard is not implemented.** The paper is explicit:
  *"to avoid numerical instabilities, the BFGS update is omitted if one of the
  denominators becomes tiny."* The Python (helper_PFCI.py:11200-11221) divides
  by `ys_dot`, `sBs_dot`, and `(sBs_dot - ys_dot)` **with no guard** and always
  appends the history entry. If `sBs_dot ≈ 0` (near-singular curvature along
  `s`) or `sBs_dot ≈ ys_dot`, `rho_Bs`/`theta`/`rho_y` blow up and inject a
  corrupted entry that then poisons every subsequent `get_bfgs_mv`. Note that
  Powell damping does **not** cover this: it handles `ys < 0.1·sBs` (bad/
  negative curvature) by damping, but it still divides by `sBs`, so a tiny
  `sBs` is unguarded either way. Faithfully reproduced in the port
  (`BfgsOperator::update`, no guard). This is the one place the code departs
  from the paper's stated algorithm. The clean seam to add it is
  `BfgsOperator::update` (skip-and-signal) plus its caller.

## Minor / vestigial

- **`consecutive_skips` is dead code.** The reset-dispatch condition
  (helper_PFCI.py:11262) has a `... or self.consecutive_skips >= 3` clause, but
  `consecutive_skips` is only ever set to 0 (init 11012, reset 11298) and is
  **never incremented** (grep-confirmed) — so that clause never fires. It is
  the natural place to count the "omitted update" events from the gap above
  (omit-if-tiny → count skips → after several, recompute the exact Hessian and
  restart); wiring the tiny-denominator guard would naturally also wire this.
  The paper's *actual* restart condition (density/energy change) is
  independent of this and is correctly implemented. The port documents
  `consecutive_skips` as confirmed-dead, matching the Python.

- **Powell damping is an addition beyond the paper, not from it.** The paper's
  numerical-stability strategy is "omit the update if a denominator becomes
  tiny" — it does not describe Powell damping (that came from `bfgs.md`). The
  damping is a legitimate technique and is correctly implemented, but it is a
  deviation from the paper's stated method and does not substitute for the
  omit-if-tiny guard (see the gap above). Worth a deliberate decision on
  whether to keep it, given `bfgs.md` is non-authoritative.

- **Cosmetic:** the damping print string (helper_PFCI.py:11203) says `< 0.2 *`
  but the actual `sigma = 0.1`. Stale string only; the port uses `0.1`
  correctly.

## Bottom line

Measured against the paper (not `bfgs.md`), the Python is a correct
implementation of the QN algorithm — exact `B_0`, correct Eq. 63 HVP, the CI-
residual precondition, and the density/energy restart are all present and
right, and the port mirrors them faithfully. The single real deviation is the
missing "omit the update if a denominator becomes tiny" guard (with
`consecutive_skips` sitting dead as its intended companion); Powell damping is
an extra beyond the paper. If addressed, do it in the Python first and let the
port keep mirroring it.
