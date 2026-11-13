# QED-CAS/QED-FCI 1-RDM follow-up report

All calculations and figures used the `qed-ci` conda environment. Original results and figures were verified unchanged by SHA-256. New data, logs, tables, and figures have v2 names.

## λ matching and state selection

Selection uses `np.isclose` with absolute tolerance 1e-8; grouping keys round to four decimals. The counts below isolate this fix on the original cached data, before rerun replacement. Both metrics have the same gains:

- historical resonant-only figure and summary: +0 / +7 / +7 at λ=0.01 / 0.05 / 0.10; total +14.

- historical parity (all states): +0 / +7 / +7 at λ=0.01 / 0.05 / 0.10; total +14.

- historical ratio/summary (duplicate grounds): +0 / +3 / +3 at λ=0.01 / 0.05 / 0.10; total +6.

- variant A parity; A/B ratio and summary: +0 / +3 / +3 at λ=0.01 / 0.05 / 0.10; total +6.

- variant B parity: +0 / +7 / +7 at λ=0.01 / 0.05 / 0.10; total +14.

Ratios and all summary statistics include resonant ground + blue-detuned bright-connected states only. Variant A uses exactly this set in parity. Variant B adds light-gray resonant LP/UP in parity only, behind colored points, labeled “resonant polaritons (not in ratio)”.

## BeH2 diagnosis

The original BeH2 failure is an assignment gate; its ground energy is already photon-converged at 10 states. Convergence of the actual LP/UP cannot be established from the ambiguously assigned pair. At every ladder size and coupling, one of the two selected roots has projection score far below 0.1. `run_pilot.py` resets the previous-energy comparison to None, so convergence cannot pass even when energies stabilize. The calculation requests eight QED roots; the selected pair does not contain two bright/photon-character states. The rerun retains that protocol rather than increasing the root window or weakening the gate. The λ=0.01 maximum selected-energy change below includes an unstable near-zero-character candidate; it must not be interpreted as a tracked polariton truncation error.

| λ | 6→10 ground ΔE (Eh) | 6→10 maximum selected ΔE (Eh) | Weak assignment score at 10 |
|---:|---:|---:|---:|
| 0.01 | 3.020e-14 | 1.313e-05 | 9.380e-28 |
| 0.05 | 1.350e-13 | 6.621e-11 | 2.288e-23 |
| 0.10 | 3.763e-10 | 6.379e-08 | 2.828e-26 |

## Coherent basis implementation and validation

The checked API option is `CQEDConfig(photon_basis="coherent")`. Its validation currently couples this to `reference_orbitals="qed_rhf"`; the legacy flags also forcibly pair coherent photons with QED-RHF orbitals. The orbital-guess path semicanonicalizes with the QED Fock matrix, so it cannot guarantee the requested fixed canonical orbitals.

`coherent_v2.py` therefore supplies a canonical RHF reference through a scoped reference-builder adapter, with the package’s coherent constants: d_exp=2 Tr(D d), d_c=d_exp²/2, and d_PF=-d_exp d. The actual electronic coefficients remain canonical cavity-free RHF (asserted against the bare run at 1e-10). No package source was modified and no orbital optimization was performed. Metadata explicitly records the effective canonical orbitals. The informational mean-field energy fields retain RHF values; CI energies use the displaced Hamiltonian constants.

Physical number-state character targets are transformed into the displaced oscillator before LP/UP assignment: a=b+α, α=(d_exp+λ·μ_nuc)/sqrt(2ω). Photon-number expectations are deliberately blank/null in coherent metrics/cache; displaced-sector occupation is not reported as physical photon number.

H2 equilibrium, λ=0.10, same cached resonant ω, 10 photon states:

| State | Number/coherent absolute ΔE (Eh) | Electronic 1-RDM Frobenius difference |
|---|---:|---:|
| ground | 8.8818e-16 | 5.4186e-15 |
| LP | 9.9920e-16 | 4.0131e-15 |
| UP | 2.5535e-15 | 9.3040e-15 |

All comparisons pass 1e-6; 6→10 checks also pass for both energies and RDMs. H2 orbital coefficient differences are exactly zero. Production starts only after this gate passes. An additional nonzero-displacement LiH comparison against the original number cache is in `coherent_v2_nonzero_displacement_check.csv`.

## Production results

The rerun covers all 15 BeH2/LiH/Li2 geometry/coupling groups, with the original geometries, bases, core treatment, active-space rule, resonant FCI bright-transition frequencies, λ ladder, 2/4/6/10 photon states, and 1e-6 Eh criterion. The original 24-root cavity-free fallback is retained when the first eight roots contain no bright transition; QED calculations still request eight roots.

| System | Geometry | λ | States | Result |
|---|---|---:|---:|---|
| LiH | equilibrium | 0.01 | 6 | converged |
| LiH | equilibrium | 0.05 | 10 | converged |
| LiH | equilibrium | 0.10 | 10 | photon_truncation |
| BeH2 | equilibrium | 0.01 | 10 | ambiguous_polariton_assignment |
| BeH2 | equilibrium | 0.05 | 10 | ambiguous_polariton_assignment |
| BeH2 | equilibrium | 0.10 | 10 | ambiguous_polariton_assignment |
| Li2 | equilibrium | 0.01 | 6 | converged |
| Li2 | equilibrium | 0.05 | 10 | ambiguous_polariton_assignment |
| Li2 | equilibrium | 0.10 | 10 | assignment_gate_no_valid_previous |
| LiH | LiH_2re | 0.01 | 6 | converged |
| LiH | LiH_2re | 0.05 | 10 | photon_truncation |
| LiH | LiH_2re | 0.10 | 10 | photon_truncation |
| Li2 | Li2_1p5re | 0.01 | 6 | converged |
| Li2 | Li2_1p5re | 0.05 | 10 | ambiguous_polariton_assignment |
| Li2 | Li2_1p5re | 0.10 | 10 | assignment_gate_no_valid_previous |

An assignment_gate_no_valid_previous result means the final assignment is acceptable but an ambiguous preceding rung reset the comparison; it is not evidence of a measured energy difference above tolerance. Failed assignment gates are kept distinct from actual photon-truncation failures.

5/15 groups converge in the coherent basis; 0 are newly converged relative to the original number-state run. Newly recovered groups replace the entire failed old group and add ground rows only. Already-converged number-state groups remain intact, including their parity-only polaritons; their successful coherent repeats are retained separately in coherent_v2_metrics.csv. Failed coherent groups do not replace original number-state rows. No mixed-basis group is permitted. No convergence threshold was relaxed.

## Updated caption statistics

The ratio data represent 11 molecules: BH, CH4, H2, H2O, H3+, HF, He2, HeH+, Li2, LiH, NH3. Filled symbols denote equilibrium geometries and hollow symbols stretched geometries. Panel (b) excludes blue-detuned grounds and all LP/UP; the natural-occupation metric retains its 1e-8 baseline floor.

| Metric | λ | n | Median ratio | Range |
|---|---:|---:|---:|---:|
| 1-RDM Frobenius | 0.01 | 38 | 1.00017 | 0.991835–1.007 |
| 1-RDM Frobenius | 0.05 | 29 | 0.999363 | 0.805301–1.0263 |
| 1-RDM Frobenius | 0.10 | 26 | 1.0036 | 0.393176–1.12127 |
| Natural occupations | 0.01 | 38 | 1.00063 | 0.999874–1.89938 |
| Natural occupations | 0.05 | 29 | 1.00907 | 0.997435–26.433 |
| Natural occupations | 0.10 | 26 | 1.02801 | 0.989774–94.9816 |

Complete outlier identities, both errors, remaining omitted systems/groups, and all row-count audits are in `full_v2_summary.md` and the accompanying CSV tables. Figures are `full_v2_{rdm,natural_occupations}_variant_{A,B}.{pdf,png}` (PNG at 300 dpi).

## Remaining omissions


- BeH2: no eligible ratio points; recorded statuses: not_converged_at_max.

Remaining omitted resonant-ground groups (including partially represented systems): BeH2/equilibrium/λ=0.01; BeH2/equilibrium/λ=0.05; BeH2/equilibrium/λ=0.10; Li2/Li2_1p5re/λ=0.05; Li2/Li2_1p5re/λ=0.10; Li2/equilibrium/λ=0.05; Li2/equilibrium/λ=0.10; LiH/LiH_2re/λ=0.05; LiH/LiH_2re/λ=0.10; LiH/equilibrium/λ=0.10.

The older joint ground/LP/UP convergence CSV also flags He2 and CH4, but the supplied metrics explicitly accept their ground rows and mark their LP/UP unavailable. Those accepted ground rows are preserved. Their cached 6→10 ground-energy changes are at most 6.58e-14 Eh (He2) and 1.06e-12 Eh (CH4). They are represented in the ratio statistics, not omitted; no new ground-only acceptance rule was applied to the coherent reruns.

HF blue-detuned bright-connected rows remain excluded by the original failed λ=0 state-identity check. Systems without a safe blue-detuned window gain no excited-state ratio points. CH4 has no screened detuned frequency because its spectrum is too short. Resonant LP/UP are never counted as ratios.

