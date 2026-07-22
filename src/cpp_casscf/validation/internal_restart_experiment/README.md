# Internal-rotation restart-loop experiment (PAUSED — resume here)

Testing whether **repeating** the internal-rotation correction, instead of
doing it exactly once per macroiteration, reduces macroiterations. Werner's WMK
paper allows both; the code implements once, which its own diagnostic shows is
often insufficient (the internal step stays above the 1e-4 threshold for many
consecutive macroiterations).

## The knob (env-gated, default = no-op)

`helper_PFCI.py`, the restart block in the macroiteration loop (search
`RESTART MICROITERATION TO CORRECT INTERNAL ROTATION`):

    QED_INTERNAL_RESTART_MAX   default 1

`=1` reproduces the current single-correction behavior **exactly** (verified:
H2O seed 1 gives 15 macroiterations and E = -76.02974162104522, bit-identical
to the pre-change baseline — see `h2o_max1.log`). `>1` loops the correction up
to that many times per macroiteration, recomputing R each pass and restarting
the micro from the external part only.

## Result so far (H2O/6-31G CAS(4,4), seed 1)

| | macroiterations | final energy |
|---|---|---|
| MAX=1 (current) | 15 | -76.02974162 |
| MAX=5 (loop)    | **7** | **-76.03720915** |

Looping **halved the macroiterations and escaped the worse local minimum**
(-76.0297 is a known bad solution on this seed; seeds 2/3 find -76.0372
directly, and MAX=5 finds it here too). Evidence in `h2o_max5.log`.

## Within-macro norm sequence (the theory question)

Does repeating drive the internal step norm down? `awk` over `h2o_max5.log`:

    macro 2:  0.445 -> 0.381 -> 0.230 -> 0.199 -> 0.238 -> 0.242   (drops then RISES)
    macro 3:  0.112 -> 0.053 -> 0.044 -> 0.045 -> 0.047 -> 0.047   (drops then up)
    macro 4:  0.082 -> 0.067 -> 0.057 -> 0.047 -> 0.036 -> 0.026   (clean monotone)
    macro 5:  0.039 -> 0.022 -> 0.007 -> 0.0014 -> 2.5e-4 -> 4.6e-5 (geometric -> converged)
    macro 6:  1.5e-4 -> 1.0e-6
    macro 7:  1.2e-8

Confirms the theory: strongly contractive near convergence (macro 5), but
non-monotone far from it (macro 2, R~0.4) because R ~= 0.5(U - U^T) is only a
first-order truncation of log(U) and breaks for large rotations, and the
external re-optimization couples back into the internal space. The developer's
observation that the norm can *increase* after a correction is real and
expected, not a bug.

## Theory summary (why it works, why it can fail)

- The micro minimizes the **2nd-order** WMK model, which has wrong x^4
  asymptotics for inactive-active (internal) rotations, so it *underestimates*
  the internal step. The internal transform (Eq 5.18) is an **exact, all-orders**
  similarity transform of the integrals, so absorbing R_ij into the basis and
  re-expanding treats the internal-internal rotation to highest order. Fixed
  point: R_ij = 0 (internal space self-consistent with the model).
- Increase mechanisms: (1) first macroiteration, R ~ 0.5 rad, the
  0.5(U-U^T) extraction error is O(R^2) ~ 0.25 — you absorb a *wrong* rotation
  ("except sometimes in the first macroiteration, but then it is not
  important"); (2) weak contraction later, external re-optimization regenerates
  internal rotation.

## The U-accumulation question (answered from the code)

After the first micro produces U1 and the reoptimized micro produces U2, the
code uses `U_int * U2`, **not** `U1 * U2`:
- U1's internal part -> `U_int`, absorbed EXACTLY into integrals + `U_total`
  (replaces U1's underestimated internal step).
- U1's external part -> starting guess of the second micro, so it lives inside
  U2.
- U1 itself is never multiplied in. `U1 * U2` would double-count the internal
  rotation. **The current accumulation is correct.**
- The four-index transform always rebuilds J/K from the FIXED initial-basis
  `self.twoeint` by the accumulated `self.U_total` (reset to eye once, before
  the macro loop, never per-macro). The in-place internal transform of J/K is
  scratch to feed the second micro, overwritten by the four-index transform.

## NEXT STEPS when resuming

1. **A "stop on stagnation" policy will likely beat fixed MAX.** Passes 4-5 in
   macros 2-3 make the norm *worse*, so keep the best / stop when it stops
   decreasing (same idea as the spin-projection stagnation test). Add
   `QED_INTERNAL_RESTART_MODE=stagnate` alongside the fixed cap.
2. **Paired sweep on MgH+**, same harness as `../ci_cost_experiment/` (which
   just finished — see its logs): seeds 1-6 x {MAX=1, MAX=3, MAX=5, stagnate},
   OMP=1, report macroiterations / total CI iterations / orbital steps / final
   energy. Prediction: fewer macroiterations, and specifically fewer of the 9
   runaway (maxiter=100) CI solves, since each macro starts from better-
   converged internal orbitals.
3. This composes with the CI-cost knobs (`QED_CI_COUPLED_FACTOR`,
   `QED_CI_UNCOUPLED_MAXITER`) — both attack the 71% CI cost, one via solve
   count, this via macroiteration count. Worth a combined cell.
4. If it wins robustly at equal-or-better energy, port the loop into the C++
   `MacroiterationDriver::run` restart branch (currently single-correction) and
   make it the Python default.

## State

`helper_PFCI.py` is UNCOMMITTED (env-gated, default-no-op), same policy as the
QN and CI-cost experiments. Full current diff (this loop PLUS the CI-cost knobs)
backed up as `helper_PFCI_all_experiments.patch`. The CI-cost knobs alone are in
`../ci_cost_experiment/helper_PFCI_ci_knobs.patch`.
