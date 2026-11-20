# QED-CAS/QED-FCI v3 summary

## Ground-state acceptance rule (v3)

Resonant ground rows enter only the ratio/parity sets, so their acceptance no longer depends on LP/UP assignment. For every resonant (system, geometry, λ) group, number-state and coherent runs alike, the resonant ground rows are accepted iff, between the two largest photon-ladder rungs the group reached (the final rung is the one at which its QED-CAS rows were computed):

- the QED-FCI ground-state energy changes by ≤ 1e-6 Eh, and
- the QED-FCI ground-state electronic 1-RDM changes by ≤ 1e-6 (Frobenius),

with both CI solves converged. The ground root is tracked by character, not index: at each rung it is the root with the largest squared projection onto the physical photon vacuum ⊗ cavity-free ground vector (the vacuum is transformed into the displaced oscillator for coherent runs); a tracked score below 0.5 would count as a tracking failure. Separate Psi4 runs may rotate degenerate MOs, so the two rungs' RDMs are compared in one orbital frame (U = C_a⁻¹ C_b). The two rungs were recomputed by `ground_rule_v3.py` because the caches do not store CI vectors.

This rule replaces the previous special-case acceptance of He2 and CH4 ground rows. LP/UP acceptance (joint energy test and assignment gate) is unchanged, and LP/UP appear only in variant B parity. Bases are never mixed within a group; where number-state and coherent groups both pass, the number-state rows are kept.

## Per-λ statistics

| Metric | λ | n (v2) | n (v3) | Median | Range |
|---|---:|---:|---:|---:|---:|
| 1-RDM Frobenius | 0.01 | 38 | 41 | 1.0001922 | 0.9918349–1.006996 |
| 1-RDM Frobenius | 0.05 | 29 | 32 | 1.0025234 | 0.80530115–1.026304 |
| 1-RDM Frobenius | 0.10 | 26 | 29 | 1.010711 | 0.39317599–1.1212671 |
| Natural occupations | 0.01 | 38 | 41 | 1.0004644 | 0.99987419–1.8993763 |
| Natural occupations | 0.05 | 29 | 32 | 1.0073783 | 0.99743533–26.432959 |
| Natural occupations | 0.10 | 26 | 29 | 1.024062 | 0.98977372–94.981596 |

Natural-occupation statistics retain the cavity-free error floor of 1e-8. Ratios use resonant ground + blue-detuned bright-connected states only.

## Molecules represented

1-RDM Frobenius: 12 molecules — BH, BeH2, CH4, H2, H2O, H3+, HF, He2, HeH+, Li2, LiH, NH3.

Natural occupations: 12 molecules — BH, BeH2, CH4, H2, H2O, H3+, HF, He2, HeH+, Li2, LiH, NH3.

## Newly accepted groups (relative to v2)

| Group | Basis | Rungs | Ground ΔE (Eh) | Ground Δγ | Ground rows added |
|---|---|---|---:|---:|---:|
| BeH2/equilibrium/λ=0.01 | bare_number_state | 6->10 | 1.954e-14 | 5.872e-12 | 3 |
| BeH2/equilibrium/λ=0.05 | bare_number_state | 6->10 | 9.592e-14 | 2.131e-11 | 3 |
| BeH2/equilibrium/λ=0.10 | bare_number_state | 6->10 | 3.763e-10 | 7.997e-09 | 3 |

Groups accepted in v2 but failing the v3 rule: none.
Accepted groups whose basis changed from v2: none.
Number-state LP/UP rows dropped to avoid basis mixing: 0.

He2 and CH4 now pass the uniform rule on their own numbers:
- He2/equilibrium/λ=0.01: 6->10, ΔE = 8.88e-16 Eh, Δγ = 6.21e-15 → accepted.
- He2/equilibrium/λ=0.05: 6->10, ΔE = 6.22e-15 Eh, Δγ = 1.55e-14 → accepted.
- He2/equilibrium/λ=0.10: 6->10, ΔE = 5.86e-14 Eh, Δγ = 5.78e-12 → accepted.
- CH4/equilibrium/λ=0.01: 6->10, ΔE = 7.11e-15 Eh, Δγ = 3.92e-12 → accepted.
- CH4/equilibrium/λ=0.05: 6->10, ΔE = 5.68e-14 Eh, Δγ = 1.31e-11 → accepted.
- CH4/equilibrium/λ=0.10: 6->10, ΔE = 1.04e-12 Eh, Δγ = 3.26e-11 → accepted.

## Remaining omitted resonant-ground groups

| Group | Basis tested | Rungs | Ground ΔE (Eh) | Ground Δγ | Cause |
|---|---|---|---:|---:|---|
| Li2/Li2_1p5re/λ=0.05 | bare_number_state | 6->10 | 3.361e-06 | 1.460e-04 | photon_truncation_energy_and_rdm |
| Li2/Li2_1p5re/λ=0.05 | coherent_state | 6->10 | 3.361e-06 | 1.460e-04 | photon_truncation_energy_and_rdm |
| Li2/Li2_1p5re/λ=0.10 | bare_number_state | 6->10 | 2.301e-04 | 6.909e-03 | photon_truncation_energy_and_rdm |
| Li2/Li2_1p5re/λ=0.10 | coherent_state | 6->10 | 2.301e-04 | 6.909e-03 | photon_truncation_energy_and_rdm |
| Li2/equilibrium/λ=0.05 | bare_number_state | 6->10 | 3.489e-07 | 2.053e-05 | photon_truncation_rdm_only |
| Li2/equilibrium/λ=0.05 | coherent_state | 6->10 | 3.489e-07 | 2.053e-05 | photon_truncation_rdm_only |
| Li2/equilibrium/λ=0.10 | bare_number_state | 6->10 | 6.643e-05 | 2.988e-03 | photon_truncation_energy_and_rdm |
| Li2/equilibrium/λ=0.10 | coherent_state | 6->10 | 6.643e-05 | 2.988e-03 | photon_truncation_energy_and_rdm |
| LiH/LiH_2re/λ=0.05 | bare_number_state | 6->10 | 7.487e-07 | 6.580e-05 | photon_truncation_rdm_only |
| LiH/LiH_2re/λ=0.05 | coherent_state | 6->10 | 3.447e-06 | 3.789e-04 | photon_truncation_energy_and_rdm |
| LiH/LiH_2re/λ=0.10 | bare_number_state | 6->10 | 2.718e-04 | 2.385e-02 | photon_truncation_energy_and_rdm |
| LiH/LiH_2re/λ=0.10 | coherent_state | 6->10 | 6.019e-04 | 5.978e-02 | photon_truncation_energy_and_rdm |
| LiH/equilibrium/λ=0.10 | bare_number_state | 6->10 | 2.049e-06 | 6.246e-05 | photon_truncation_energy_and_rdm |
| LiH/equilibrium/λ=0.10 | coherent_state | 6->10 | 3.602e-07 | 3.342e-05 | photon_truncation_rdm_only |

## Points outside [0.8, 1.25]

| Metric | System | Geometry | Active space | State | λ | Basis | Cavity-free error | QED error | Ratio |
|---|---|---|---|---|---:|---|---:|---:|---:|
| 1-RDM Frobenius | H2 | H2_2re | CAS(2,2) | bright_dressed | 0.10 | bare_number_state | 0.101450472 | 0.0398878896 | 0.39317599 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.01 | bare_number_state | 0.000228337228 | 0.000433698325 | 1.89937633 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.05 | bare_number_state | 0.000228337228 | 0.00603562853 | 26.4329588 |
| Natural occupations | H2 | equilibrium | CAS(2,4) | bright_dressed | 0.05 | bare_number_state | 0.000228034506 | 0.000472087907 | 2.07024768 |
| Natural occupations | H2 | equilibrium | CAS(2,2) | bright_dressed | 0.10 | bare_number_state | 0.000228337228 | 0.0216878344 | 94.9815964 |
| Natural occupations | H2 | equilibrium | CAS(2,4) | bright_dressed | 0.10 | bare_number_state | 0.000228034506 | 0.00126187919 | 5.53372035 |
| Natural occupations | H2 | H2_2re | CAS(2,2) | bright_dressed | 0.10 | bare_number_state | 0.00943931616 | 0.012825834 | 1.35876729 |
| Natural occupations | H2 | H2_2re | CAS(2,4) | bright_dressed | 0.10 | bare_number_state | 0.000891325301 | 0.00138121812 | 1.54962293 |

## Consistency checks

- Tracked QED-FCI ground root is index 0 at both rungs in 60/60 groups; minimum tracked score 0.6387, largest competing score 2.14e-01.
- Recomputed ground energies reproduce the cached energies to ≤ 1.6e-13 Eh.
- QED-CAS ground rows: tracked CAS ground is root 0 in 143/147 cached rows; all cached rows sit at their group's final rung; recomputed errors match the cache to ≤ 1e-6 in 135/147 rows (exceptions below).
- Rows used in v3 whose recomputed QED-CAS error differs from the cache by > 1e-6: 5 (BH CAS(4,7), BeH2 CAS(4,8); largest 1.8e-04, relative 2.4e-03). These active spaces end inside a degenerate canonical π pair (RHF orbital-energy gap < 1e-14 Eh at the boundary), so the QED-CAS result depends on the arbitrary rotation of that pair relative to the cavity polarization and is not run-to-run reproducible. The discrepancy grows with λ and is unrelated to photon truncation; the rows are kept as cached (pre-existing protocol issue).
- Rows whose character-tracked QED-CAS ground is not root 0: 4 (LiH/LiH_2re/λ=0.10 CAS(2,2), LiH/LiH_2re/λ=0.10 CAS(2,4)); all belong to groups that fail the ground rule, so none enters the figures or statistics.
- Orbital frames differed between the two rungs in 39 recomputed groups; differences are compared after frame alignment.

Inputs: `ground_rule_v3_recompute.jsonl` (recompute), `ground_rule_v3.csv` (per-group rule table), `ground_rule_v3_cas_check.csv`. Figures: `full_v3_{rdm,natural_occupations}_variant_{A,B}.{pdf,png}` (PNG at 300 dpi).
