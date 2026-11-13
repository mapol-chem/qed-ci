# QED-CAS / QED-FCI 1-RDM benchmark

This directory contains the H₂/LiH pilot, the twelve-molecule full run, the
dimension-only basis screen, and the cached-data plotting workflow. It uses the existing
`PFHamiltonianGenerator` engine through `qed_ci.calculate`.

## API findings from this repository

- `CQEDConfig(method="fci")` selects FCI; `method="casci"` selects fixed
  orbitals; `method="casscf"` requests orbital optimization. Set
  `reference_orbitals="canonical"` and `photon_basis="number"` together to
  retain canonical cavity-free RHF orbitals and a bare number-state basis.
- `lambda_vector` is in atomic units. `omega` is in Hartree. `max_photons` is
  the highest occupation, so a basis with *N* states uses `max_photons=N-1`.
- `PFHamiltonianGenerator.electronic_dipole_array[i,j,:]` contains the
  electronic transition dipole in e·bohr. The pilot identifies the first
  bright state from the smallest excited root above the configurable dipole
  threshold.
- The public `CalculationResult` does not include 1-RDMs. The pilot calls the
  existing C binding `helper_PFCI.c_build_one_rdm` for a requested root and
  pads the returned core+active block with virtual zeros. Its trace is checked
  against the full electron count.
- There is no direct photon-number expectation helper. The pilot computes
  `sum_p p * ||C_p||²` from the CI vector's contiguous photon-sector blocks in
  the number-state basis and records that implementation explicitly.
- Frozen core is represented by a CASCI calculation with all remaining
  orbitals active and `n_act_el = n_el - 2*n_core`. Use the same active
  electron/orbital restriction for cavity-free and QED references.

## Pilot contents

`config.py` holds the input geometries, basis priority, cap, coupling values,
photon-state trial list, active-space rules, and stretched geometries.
`feasibility.py` estimates the determinant and QED-FCI dimensions using
Psi4's installed basis definitions. It launches no SCF/CI calculations and
writes `results/feasibility.csv` and `results/feasibility.md`.

`run_pilot.py` runs H₂ and LiH at the configured equilibrium and stretched
geometries. It performs the all-active CASCI/FCI and λ=0 checks, cavity-free
bright-state selection, active-space RDM comparisons, λ sweeps, and photon
basis convergence. Each completed calculation is appended to
`results/pilot_cache.jsonl`, including the electronic RDMs, energies,
metadata, timings, and dipoles. `validate_checks.py` rechecks λ=0 off
resonance, audits cached RDM traces, and derives the convergence summary.

The geometry coordinates are in Angstrom. Equilibrium structures use the
recorded experimental or optimized geometry sources in `geometry_sources.csv`;
the H₂ and LiH pilot bond lengths are 0.7414 Å and 1.595 Å from NIST's CCCBDB.
The stretched structures scale the corresponding equilibrium coordinates by
the factors in `config.py`.

## Rerun commands

From the repository root:

```sh
conda run -n qed-ci python validation/benchmark_qed_cas_fci/feasibility.py
conda run -n qed-ci python validation/benchmark_qed_cas_fci/run_pilot.py
conda run -n qed-ci python validation/benchmark_qed_cas_fci/validate_checks.py
MPLCONFIGDIR=/private/tmp/mpl-qedci conda run -n p4dev \
  python validation/benchmark_qed_cas_fci/plot_results.py
```

The plotting script reads only cached CSV results and does not recalculate.
It writes the two panel figures in PDF and 300 dpi PNG form, the per-coupling
summary tables, and draft captions. The natural-occupation ratio plot uses a
log scale because a few small cavity-free reference errors produce large
ratios; those values are retained and reported.

The full run covers the twelve equilibrium molecules and the three configured
stretched geometries. Run it with `run_pilot.py --full --confirm-full`; review
the explicit confirmation and verified geometry-source gate before starting.
The completed results are in `results/full_cache.jsonl` and the associated
`full_*.csv` files. The two full-set proposal figures are in `figures/`.

Full-run validation results need qualification. All 15 cached 1-RDM traces
match their electron counts within 1.5e-12. The off-resonance λ=0 checks pass
for 25/30 ground/bright comparisons; the five bright-state failures involve
excited-state root ambiguity or near-degenerate states. The all-active
CASCI/FCI comparisons pass for all ground states and for 9/15 selected bright
states. Six bright-state comparisons fail: four have equal energies to
numerical precision but different 1-RDMs, consistent with root rotations in
degenerate manifolds; two Li₂ comparisons also show energy differences of
0.0118 and 0.0053 Eh. Treat these excited-state validation failures as
unresolved.

Of 45 molecule/geometry/coupling photon-basis groups, 29 converge by 10
photon states at the 1e-6 Eh target; 16 do not. The assignment score threshold
is 0.1, and ambiguous FCI polariton assignments are flagged in the cache and
convergence CSV. The summary and plotted statistics use only converged rows;
the `assignment_ambiguous` column also excludes flagged LP/UP rows. The full
run did not persist QED-CAS LP/UP assignment scores, so that side's root
confidence cannot be audited independently from the cache. A few
minimal active spaces have no cavity-free bright root and therefore have no
LP/UP metric. QED-CASSCF/CASSCF variants were not run in this benchmark.

## Spectrum-guided blue detuning screen

`screen_offresonant.py` computes expanded cavity-free spectra for the selected
basis and frozen-core definition; `select_offresonant.py` screens candidate
blue shifts. The scan is conservative at the largest coupling, |λ|=0.1: it
requires at least three estimated one-photon coupling widths from each
polarized transition line and three second-order two-photon coupling widths
from each included singlet gap. It selects the smallest blue shift satisfying
both conditions and constrains 2ω to the computed spectral range. This is a
screening heuristic; QED-FCI photon expectation and bare-S1 overlap must
still verify that the resulting state is bright-connected.

The spectra are in `results/offres_cavity_free_spectra.jsonl`; proposed
frequencies and screen status are in `results/offres_frequency_screen.csv`.
CH₄ remains short of the required two-photon spectral range even with 130
roots, so no frequency is proposed for it. Systems marked
`no_3x_safe_window` have a best candidate but no frequency passing the
conservative detuning screen at |λ|=0.1; treat those separately.

Off-resonant calculations were run for the six `safe_window` geometries at
all three coupling strengths. All 18 QED-FCI photon-basis sequences converged
by 10 states. At |λ|=0.1, tracked FCI bright-state vacuum overlaps ranged
from 0.888 to 0.999, with photon expectations from 8e-4 to 0.117. The
off-resonant λ=0 identity check passed 11/12 comparisons; the HF bright-root
1-RDM differs by 0.938 despite matching energy, so HF bright-connected
ratios are excluded pending resolution of that root ambiguity.

For the seven remaining active-space bright comparisons per coupling, median
Frobenius error ratios were 1.000, 0.998, and 0.987 at |λ|=0.01, 0.05, and
0.10. The corresponding natural-occupation medians were 1.003, 1.094, and
1.359; larger ratios are driven mainly by small cavity-free H₂ errors.
`plot_combined.py` writes figures that put blue-detuned bright-connected
points beside resonant ground/LP/UP points in the parity panel. Its ratio
panel includes resonant and blue-detuned ground states plus only the
blue-detuned bright-connected states.

## v2 follow-up

Run in the `qed-ci` conda environment from this directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n qed-ci python run_coherent_v2.py
MPLCONFIGDIR=/tmp/qed-mpl-v2 conda run --no-capture-output -n qed-ci python finalize_v2.py
conda run -n qed-ci python -m pytest -q test_v2.py
```

The coherent runner validates H2 first and resumes completed production
geometry/coupling groups from its **v2** CSVs. `--validate-only` runs the gate
without production. The benchmark-local `coherent_v2.py` adapter keeps
canonical cavity-free RHF orbitals while using the package's coherent photon
Hamiltonian; the current public config couples these choices otherwise.
Photon-number expectations are blank for coherent calculations. The saved
LP/UP assignment projections use physical number-state targets transformed
into the displaced basis.

`finalize_v2.py` merges only complete converged groups, creates
`results/full_v2_metrics.csv`, the summary/report/outlier/recovery tables,
and both A/B PDF and 300-dpi PNG variants for Frobenius and natural-occupation
errors. It checks original result/figure hashes against
`results/original_checksums_v2.json`. `plot_combined.py` and
`plot_results.py --prefix full` call this same finalizer.
Both panels in A, panel (b) in B, and every statistic use only resonant
ground plus blue-detuned bright-connected states. B additionally shows gray
resonant LP/UP in parity. Existing number-state caches are never rewritten by
these v2 entry points. See `results/full_v2_report.md` for the completed
rerun's convergence limitations and `results/full_v2_summary.md` for the
full outlier list and caption statistics.
