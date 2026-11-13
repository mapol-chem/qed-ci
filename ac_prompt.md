# Task: Benchmark 1-RDM invariance of QED-CAS vs. QED-FCI across a molecule set, and make a proposal figure

## Context
I'm writing a grant proposal section on an adiabatic connection (AC) correction for cavity QED multireference methods. AC assumes the reference (active-space) 1-RDM approximates the exact 1-RDM. I want to show, statistically across many molecules, that cavity coupling does not degrade this assumption relative to the cavity-free case.

The computations use our open-source `qed-ci` package (Psi4-based; implements QED-CASCI, QED-CASSCF, and QED-FCI for the Pauli–Fierz Hamiltonian in the length gauge under the long-wavelength approximation). Before writing anything, explore the repository to learn the actual API. Find how to:
- run (QED-)CASCI, (QED-)CASSCF, and (QED-)FCI;
- get spin-summed 1-RDMs for ground and excited states;
- set ω, the λ vector, and the photon basis size;
- choose the orbital basis (RHF vs. QED-HF);
- obtain transition dipoles and photon-number expectation values.

Do not guess function names. If something I ask for isn't supported, tell me rather than working around it silently.

## Molecule set (equilibrium geometries; use standard literature or optimized geometries and record the source)
H2, HeH+, H3+, LiH, He2, BH, BeH2, Li2, HF, H2O, NH3, CH4.
Also stretched geometries: H2 (2.0 × r_e), LiH (2.0 × r_e), Li2 (1.5 × r_e).

## Basis sets and feasibility
- For each system, estimate the QED-FCI dimension (electronic FCI determinants × photon states) for 6-311G, 6-31G, and STO-3G. Pick the largest basis whose QED-FCI dimension is under a cap (start at 2e6; make the cap a config parameter).
- A frozen core is allowed if needed, but it must be applied identically in the QED and cavity-free calculations, and recorded.
- Print a feasibility table (system, basis, frozen core, n_el, n_orb, FCI dim, QED-FCI dim) and **stop for my confirmation before launching the full set of runs.**

## Cavity settings
- For each system, run cavity-free FCI to find the first *bright* excited state, i.e. the lowest state with |transition dipole| above a threshold (default 0.05 a.u.; configurable). Set ω to that excitation energy and polarize λ along that transition dipole.
- λ magnitudes: {0.01, 0.05, 0.1} a.u.
- Photon basis: converge QED-FCI in the number of photon states (try 2, 4, 6, 10; stop when the target state energies change by less than 1e-6 Eh). Use the same photon basis for QED-CAS and QED-FCI. Report what was used.
- Use whatever photon frame (bare Fock or coherent-state) the package uses by default, apply it consistently, and record it.

## States
- Cavity-free: S0 and the bright state S_b.
- QED: ground state, lower polariton (LP), upper polariton (UP).
- Identify LP/UP robustly (e.g., by photon-number expectation and overlap/character with S_b), not by root index alone. Log the assignment criteria and flag ambiguous cases.

## Active spaces
For each system, use 2–3 active spaces from a fixed rule, e.g.:
1. minimal valence: valence electrons in valence orbitals;
2. valence + 2 virtuals;
3. valence + 4 virtuals (where not equal to the full space).

Skip any active space equal to the full orbital space. Record (n_e, n_o) for each.

## Reference type
Run CASCI in canonical cavity-free RHF orbitals for both QED and cavity-free cases (consistent with my existing figure). If QED-CASSCF/CASSCF is available and converges reliably, also run CASSCF variants, flagged separately. Report convergence failures; don't drop them silently.

## Metrics (all 1-RDMs spin-summed, electronic part only)
For each (system, geometry, basis, state, active space, λ), and for the matching cavity-free case:
1. Frobenius norm of (γ_CAS − γ_FCI), with both expressed in the same canonical RHF orbital basis. Embed the active-space 1-RDM into the full orbital space with core occupation 2 and virtual occupation 0.
2. Max absolute error in sorted natural occupation numbers.
3. Sanity values: trace (should equal N_el), energies, photon-number expectation.

## Validation checks (run these first, report results)
- QED at λ = 0 reproduces cavity-free results (energies and 1-RDMs to ~1e-8).
- CAS with all orbitals active reproduces FCI.
- Each 1-RDM trace equals the number of electrons.
- Photon-basis convergence as above.

## Outputs
- Code: a config-driven run script (YAML or Python dict for systems, bases, λ, active-space rules) and a *separate* plotting script that reads cached results.
- Results: cache each calculation (JSON or CSV with full metadata: geometry source, basis, frozen core, ω, λ vector, photon basis, active space, reference type, state label, all metrics, and timings) so that rerunning the plotting doesn't recompute.
- A summary table (CSV and a pretty-printed markdown version).
- A README describing how to rerun.

## Figure (proposal-ready)
- Two panels, ~13 cm total width; vector PDF plus 300-dpi PNG.
- (a) Parity plot: x = cavity-free CAS 1-RDM Frobenius error, y = QED-CAS error, log–log if the spread warrants it, with a y = x line. Marker shape = state type (ground vs. LP/UP-vs-S_b); color = λ (colorblind-safe sequential palette). Stretched geometries get an outlined/hollow variant of the marker.
- (b) Distribution of the error ratio (QED/cavity-free) vs. λ: strip plot over a light box plot, with a horizontal line at 1.
- Also generate the same two panels for the natural-occupation metric, as a separate file, so I can choose.
- Clean styling: no titles inside panels, panel labels (a)/(b), readable at print size (8–9 pt fonts), minimal gridlines.
- Write a draft caption (2–3 sentences) in a text file summarizing what the figure shows, with key statistics (e.g., median ratio and range per λ).

## Working style
- Start with H2 and LiH end to end (all validation checks, metrics, and a draft figure) before scaling to the full set.
- Report anything surprising: ratios far from 1, failed state assignments, convergence problems, large CASCI vs. CASSCF differences. Don't paper over them.
