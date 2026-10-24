# Theory and method notes

Derivations, verification methods and findings for the QED-CI / SA-QED-CASSCF
code. Split into what the code *should* do (`theory/`) and what was *measured*
(`reports/`). Every number quoted in these files is measured, with the system
and settings named; open questions are marked as open rather than smoothed over.

## theory/ — derivations and verification methods

| file | what it covers |
|---|---|
| `dphoton_frame_term.md` | The photon-frame term in derivative couplings, including the **nuclear dipole**: why `d_N` cancels inside `H_CS` but survives in `d_photon = U_CS ∂U_CS†`; the neutral and ion translational sum rules; the finite-difference method for verifying a derivative coupling and the five things that silently corrupt it; what each check must return. **Start here for anything frame- or `d_photon`-related.** |
| `numerical_nac_method.md` | The finite-difference wavefunction-overlap NAC in full: AO→MO→determinant overlap, phase and root alignment, the Cahill–Glauber displaced photon basis, Richardson extrapolation, branch-warning diagnostics. |
| `cp_casscf_projection.md` | The CP-SA-CASSCF CI-response projection. Derives `a⁽ⁿ⁾_k (E_k − E_n) = ⟨c⁽ᵏ⁾\|b⟩` and what follows: the null space is the **degenerate cluster**, not `{c⁽ⁿ⁾}` and not all roots; consistency vs uniqueness vs conditioning; why the preconditioner (not roundoff) injects the near-null direction; the clustering/orthonormalisation routine and its known chaining defect. |
| `project_out_all_equal_weight_only.md` | Why `project_out_all` is valid only at equal weights: the gauge shift is `Δgrad = θ(W_n − W_m)⟨c⁽ᵐ⁾\|dH/dx\|c⁽ⁿ⁾⟩`, so the off-diagonal CI-response components are redundant **only** when the weights are equal. |
| `project_out_all_defect.md` | The measurement: unequal-weight gradients 1.4e-03…7.5e-03 wrong with `project_out_all`, 1.7e-08…7.1e-08 right with the minimal projection, against Molpro. Includes why molecular symmetry hid it. |
| `nac_two_projectors.md` | Why the z-vector solve and the `Y_z` source term want *opposite* projections, and why that is not a contradiction. |
| `projection_explained.md` | Narrative introduction to the projection question, less formal than the above. |

## reports/ — findings from specific studies

| file | what it found |
|---|---|
| `ethylene_cavity_ci_audit.md` | Full audit of the zero-weight SA-3 ethylene FC scan: solution-branch continuity, active-space adequacy, whether the MECP scans needed re-running. Includes the orientation-drift finding (1.12° in leg 1) and the quasi-Newton optimizer result (1500 → 19 steps). |
| `ethylene_two_active_spaces.md` | CAS(4,7) vs CAS(4,9), both legs re-converged, with the accessibility margins. §4 documents that **multiplicity is a property of the FC geometry, not the active space** — and retracts the earlier claim that enlarging removes it. |
| `difference_density_and_lambda0.md` | Difference-density selection between SA solutions (17× more discriminating than the full density), why the two sound selection criteria *disagree* in CAS(4,7), and the λ=0 coupling-solve projection bug. |
| `dphoton_nuclear_open_problem.md` | Historical: the problem statement written while the `d_photon` nuclear prefactor looked wrong by 56×. **Superseded** — the analytic term was right and the finite-difference reference was broken in three places. Kept because it records the reasoning and the two wrong conclusions it corrected. |

## Companion validation scripts

In `../validation/`:

* `dphoton_translation_check.py` — rigid-translation sum rule, SCF only, no CI.
  Neutral must give `∂z/∂x = 0`; ions must give `λ_α Q/√(2ω)`. Seconds to run.
* `numerical_nac_check.py` — the finite-difference NAC driver.
  `QED_FD_PRINT_GAMMA=1` prints the displacement; `γ₊ + γ₋` must be ~0 to O(h²).
* `test_mecp_optimizer.py` — the MECP optimizer against an analytic
  linear-vibronic model with an exactly known crossing point.

## Five rules these notes keep arriving at

1. **Agreement is not validation** when both sides can share an omission. Prefer
   a rigorous invariance constraint (a sum rule) to two implementations agreeing.
2. **When one omission is found, grep every use.** Patching one failing test at a
   time is how the remaining copies survive.
3. **Never gate gradient code at a conical intersection.** Individual roots are
   noise-amplified there; gate on the state-averaged energy, a separated
   geometry, or a unit test with the SCF removed.
4. **Use an ion to fix a sign.** Neutral cancellations hide sign errors by
   construction.
5. **Distrust a plateau judged from few points**, and distrust a selection rule
   that uses the shape of the answer being measured.
