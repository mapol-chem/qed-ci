# QED-CAS/QED-FCI v2 summary

Both variants use resonant ground states and blue-detuned bright-connected states in the ratio panel and all statistics. Blue-detuned ground states are excluded. Variant A has the same points in parity; Variant B adds light-gray resonant LP/UP in parity only.

| Metric | λ | n | Median | Range |
|---|---:|---:|---:|---:|
| 1-RDM Frobenius | 0.01 | 38 | 1.0001728 | 0.9918349–1.006996 |
| 1-RDM Frobenius | 0.05 | 29 | 0.99936344 | 0.80530115–1.026304 |
| 1-RDM Frobenius | 0.10 | 26 | 1.0035981 | 0.39317599–1.1212671 |
| Natural occupations | 0.01 | 38 | 1.0006289 | 0.99987419–1.8993763 |
| Natural occupations | 0.05 | 29 | 1.0090689 | 0.99743533–26.432959 |
| Natural occupations | 0.10 | 26 | 1.0280128 | 0.98977372–94.981596 |

Natural-occupation statistics retain the existing cavity-free error floor of 1e-8.

## Molecules represented

1-RDM Frobenius: 11 molecules — BH, CH4, H2, H2O, H3+, HF, He2, HeH+, Li2, LiH, NH3.

Natural occupations: 11 molecules — BH, CH4, H2, H2O, H3+, HF, He2, HeH+, Li2, LiH, NH3.

## Points outside [0.8, 1.25]

| Metric | System | Geometry | Active space | State | λ | Cavity-free error | QED error | Ratio |
|---|---|---|---|---|---:|---:|---:|---:|
| 1-RDM Frobenius | H2 | H2_2re | CAS(2,2) | bright_dressed | 0.10 | 0.101450472 | 0.0398878896 | 0.39317599 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.01 | 0.000228337228 | 0.000433698325 | 1.89937633 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.05 | 0.000228337228 | 0.00603562853 | 26.4329588 |
| Natural occupations | H2 | equilibrium | CAS(2,4) | bright_dressed | 0.05 | 0.000228034506 | 0.000472087907 | 2.07024768 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.10 | 0.000228337228 | 0.0216878344 | 94.9815964 |
| Natural occupations | H2 | equilibrium | CAS(2,4) | bright_dressed | 0.10 | 0.000228034506 | 0.00126187919 | 5.53372035 |
| Natural occupations | H2 | H2_2re | CAS(2,2) | bright_dressed | 0.10 | 0.00943931616 | 0.012825834 | 1.35876729 |
| Natural occupations | H2 | H2_2re | CAS(2,4) | bright_dressed | 0.10 | 0.000891325301 | 0.00138121812 | 1.54962293 |

## Rows recovered by tolerant λ matching

Counts below hold cached data and state selection fixed; they do not include coherent rerun additions or the removal of duplicate grounds. Each entry is the gain at λ=0.01 / 0.05 / 0.10.

- 1-RDM Frobenius, historical resonant-only figure and summary: 0 / 7 / 7 (total 14).

- 1-RDM Frobenius, historical parity (all states): 0 / 7 / 7 (total 14).

- 1-RDM Frobenius, historical ratio/summary (duplicate grounds): 0 / 3 / 3 (total 6).

- 1-RDM Frobenius, variant A parity; A/B ratio and summary: 0 / 3 / 3 (total 6).

- 1-RDM Frobenius, variant B parity: 0 / 7 / 7 (total 14).

- Natural occupations, historical resonant-only figure and summary: 0 / 7 / 7 (total 14).

- Natural occupations, historical parity (all states): 0 / 7 / 7 (total 14).

- Natural occupations, historical ratio/summary (duplicate grounds): 0 / 3 / 3 (total 6).

- Natural occupations, variant A parity; A/B ratio and summary: 0 / 3 / 3 (total 6).

- Natural occupations, variant B parity: 0 / 7 / 7 (total 14).

## Coherent rerun outcome

| System | Geometry | λ | Photon states | Outcome |
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

0 newly converged coherent groups merged as whole groups (5/15 rerun groups converged). Already-converged number-state groups are retained intact, including parity-only polaritons; successful repeat calculations remain in coherent_v2_metrics.csv. Recovered groups contribute coherent ground rows only and replace all old rows in the group. Failed groups retain their original records and statuses. No detuned calculations were added.

## Remaining omissions

- BeH2: no eligible ratio points; recorded statuses: not_converged_at_max.

Remaining omitted resonant-ground groups (including partially represented systems): BeH2/equilibrium/λ=0.01; BeH2/equilibrium/λ=0.05; BeH2/equilibrium/λ=0.10; Li2/Li2_1p5re/λ=0.05; Li2/Li2_1p5re/λ=0.10; Li2/equilibrium/λ=0.05; Li2/equilibrium/λ=0.10; LiH/LiH_2re/λ=0.05; LiH/LiH_2re/λ=0.10; LiH/equilibrium/λ=0.10.

The older joint ground/LP/UP convergence CSV also flags He2 and CH4, but the supplied metrics explicitly accept their ground rows and mark their LP/UP unavailable. Those accepted ground rows are preserved. Their cached 6→10 ground-energy changes are at most 6.58e-14 Eh (He2) and 1.06e-12 Eh (CH4). They are represented in the ratio statistics, not omitted; no new ground-only acceptance rule was applied to the coherent reruns.

HF blue-detuned bright-connected rows remain excluded by the original failed λ=0 state-identity check. Systems without a safe blue-detuned window gain no excited-state ratio points. CH4 has no screened detuned frequency because its spectrum is too short. Resonant LP/UP are never counted as ratios.
