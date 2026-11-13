# Follow-up on the QED-CAS/QED-FCI 1-RDM benchmark

Work in `validation/benchmark_qed_cas_fci`. Do not modify the existing cached results; write new outputs alongside them with a clear suffix (e.g. `_v2`).

## 1. Fix the λ-matching bug
`plot_combined.py` (and any other script that filters on λ) compares `lambda_value` to `LAMBDA_MAGNITUDES` with exact float equality. Some λ vectors have magnitudes like 0.0499999… and 0.0999999… (e.g. all of HF at 0.05/0.10 and one CH4 row), so converged rows are silently dropped. Match with `np.isclose` (or round to 4 decimals) everywhere λ is used for selection or grouping. Report how many rows each figure and summary gains as a result.

## 2. Deduplicate ground states in the ratio statistics
The blue-detuned ground states nearly duplicate the resonant ground states (e.g. He2 CAS(4,2) at |λ|=0.1 gives a ratio of 0.845 in both), which inflates n.
- Panel (b) and all summary statistics: use **resonant ground states + blue-detuned bright-connected states only**. Drop the blue-detuned ground states from the ratio.
- Parity panel (a): plot the same set, so the two panels are consistent.
- Report n, median, and range per λ, and identify any point whose ratio falls outside [0.8, 1.25] (system, geometry, active space, state, λ, both errors).

## 3. Two figure variants
Make both, from the same cached data (PDF + 300-dpi PNG, same styling as now):
- **Variant A (default):** ground + blue-detuned bright-connected states only; no LP/UP anywhere.
- **Variant B:** identical to A, plus resonant LP/UP points in the parity panel only, drawn in light gray (no color by λ), behind the colored points, with a legend entry "resonant polaritons (not in ratio)". LP/UP never enter panel (b) or the statistics.

## 4. Coherent-state rerun for BeH2, LiH, and Li2
In the full run, these systems failed photon-basis convergence at 10 number states (BeH2 at every λ; Li2 above 0.01; LiH at 0.10, including the stretched geometries).
- **Diagnose first.** For BeH2, check whether the failure is truly photon truncation or whether the LP/UP assignment was flagged ambiguous, which resets the convergence check in `run_pilot.py`. Report which.
- **Rerun with a coherent-state photon basis.** Find the package's coherent-state option; check the API rather than guessing. Keep the *electronic* orbitals as canonical cavity-free RHF so the 1-RDM comparison is unchanged. The coherent-state displacement acts only on the photon, so electronic 1-RDMs should be basis-independent once converged.
- **Validate before the production rerun.** On a system that already converged in the number basis (H2 equilibrium, |λ|=0.1), confirm that the coherent-state and number-state runs agree on ground and LP/UP energies (≤1e-6 Eh) and electronic 1-RDMs (Frobenius ≤1e-6).
- **Photon-number expectation.** The current photon-number helper assumes number-state sectors, so it is not valid in the coherent-state basis. Compute it correctly or leave it blank, and say which.
- **Rerun scope.** Use the same protocol as the full run (same geometries, bases, frozen cores, active-space rule, ω tuned to the FCI bright transition, λ ∈ {0.01, 0.05, 0.10}, photon-state ladder 2/4/6/10, 1e-6 Eh criterion). Record `photon_basis="coherent_state"` in the cache and metrics.
- **Merge rule.** Merge newly converged rows into the v2 metrics, but never mix number-state and coherent-state results within the same (system, geometry, λ) group. These systems have no safe blue-detuned window, so expect them to add ground-state points only.
- **Escalate if needed.** If any group still doesn't converge, report it rather than relaxing the criterion.

## 5. Deliverables
- Updated scripts (the bug fix and the selection changes) and `results/full_v2_metrics.csv`.
- Figures for variants A and B; `results/full_v2_summary.md` (n, median, range per λ, plus the outlier list).
- A short report covering:
  - rows recovered by the λ fix;
  - the BeH2 diagnosis;
  - coherent-state validation numbers;
  - which BeH2/LiH/Li2 groups now converge;
  - updated caption statistics (molecules present, n/median/range per λ, and which systems remain omitted and why).
