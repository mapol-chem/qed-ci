# CI-cost experiment

Where the time in a real SA-QED-CASSCF run actually goes, and a paired sweep of
the two constants that control it.

## 1. The profile (measured, not estimated)

From the six `v0` MgH+ CAS(8,12)/cc-pVDZ runs in
`../qn_experiment_scratch/mgh_det_logs/` — real 1–2 hour runs at `OMP=1`, using
the timing instrumentation `helper_PFCI.py` already emits. Single-run figures
are `s1_v0.log` (5690 s total); Davidson bucket figures are all six seeds
pooled (17,060 s of Davidson).

| | time | % of run |
|---|---|---|
| **CI step** | 4046 s | **71.1%** |
| — CI `build sigma` | 2082 s | 36.6% |
| — `building RDM` (×213, 3.15 s each) | 671 s | 11.8% |
| — unaccounted inside CI step | 1293 s | 22.7% |
| `internal optimization` | 624 s | 11.0% |
| **entire orbital side** (see below) | **3.03 s** | **0.053%** |

"Entire orbital side" is the sum of `build orbital sigma for Q space` (836
calls, 0.086 ms each, **0.07 s total**), `build orbital sigma for w`,
`build hessian diagonal`, `build gradient`, `build intermediates`,
`second order integral transformation`, `full JK transformation`,
`transpose matrix G`, and LSTRS/GLTR `expand` + `residual`.

**This is the single most important number in this directory.** Optimizing the
orbital-side trust-region machinery — reducing LSTRS/GLTR iterations, speeding
up orbital Hessian-vector products, vectorizing the intermediates — cannot move
total runtime, because all of it together is 0.05%. The orbital HVP count *is*
large (836), but each one costs 0.086 ms.

### Davidson time by CI context

Three live call sites use different settings (`microiteration_optimization5` is
dead code — no callers — and is excluded):

| site | maxiter | threshold |
|---|---|---|
| macroiteration, helper_PFCI.py:2560 | 100 | 1e-6 |
| `internal_optimization3`, :7781 | 5 | 1e-5 |
| `microiteration_optimization6` uncoupled, :12466 | 5 | 1e-9 |
| `microiteration_optimization6` **coupled** (`qn_count > 0`), :12469 | 10000 | `0.1*‖g_orb‖` |

Bucketing all 1363 Davidson calls by observed iteration count:

| bucket | calls | total | % of Davidson | s/call | mean iters | peak subspace |
|---|---|---|---|---|---|---|
| **coupled QN solves** | 833 | 13,469 s | **79.0%** | 16.2 | 25.9 | 64 |
| uncoupled / internal (Kreplin) | 521 | 1,344 s | 7.9% | 2.6 | ≤5 | 13 |
| macro solves hitting the 100 cap | 9 | 2,247 s | 13.2% | **249.7** | 100 | 366 |

The Kreplin uncoupled-CI strategy is working exactly as intended — 521 solves
for 1,344 s, ~6× cheaper per call than a coupled one. The cost has simply
migrated into the QN phase.

A caution on an earlier misreading: ~30% of Davidson calls print
`"Maximum iteration reaches"`, which looks alarming but is **by design** for
the capped uncoupled solves (91% of them "hit maxiter" because maxiter is 5).
Only the 9 macro-cap solves are genuine non-convergence.

### The coupled threshold is measurably load-bearing

The coupled threshold is `0.1*‖reduced_gradient‖`, so it tightens as the
optimization converges. Bucketing the 833 coupled solves by gradient norm:

| ‖g‖ quartile | mean iters | mean s |
|---|---|---|
| largest | 23.9 | 13.7 |
| 2nd | 23.1 | 12.4 |
| 3rd | 25.9 | 16.8 |
| smallest | 29.9 | 18.8 |

+25% iterations, +37% time from the loosest to the tightest end.

## 2. The two knobs

Both are env-gated in `helper_PFCI.py` and **default to a no-op**:

| env var | default | what it is |
|---|---|---|
| `QED_CI_COUPLED_FACTOR` | `0.1` | the `0.1` in `0.1*‖g_orb‖` |
| `QED_CI_UNCOUPLED_MAXITER` | `5` | the Kreplin uncoupled CI cap |

Kreplin/Werner state the precondition qualitatively — the CI residual must
become "substantially lower than the last orbital gradient" to justify
`g_c = 0` — so `0.1` is a choice, not a derived value. That is what makes it
worth sweeping.

Verified default-no-op before launching: lih/beh2 × seeds 1,2 reproduce the
pre-change baseline bit-for-bit (7/6 macroiterations, energies
`-7.9970194128908085`, `-7.997019412893538`, `-15.786118657852942`,
`-15.786118657852988`).

### A preliminary signal worth knowing

Smoke test on H2O/6-31G CAS(4,4), seed 1:

| variant | macroiterations | final energy |
|---|---|---|
| base | 15 | -76.02974162 |
| `QED_CI_UNCOUPLED_MAXITER=10` | **10** | **-76.03720915** |
| `QED_CI_COUPLED_FACTOR=0.5` | 15 | -76.02974162 |

Raising the uncoupled cap did not merely cut macroiterations — it **escaped the
worse local solution**. Seed 1's `-76.0297` is a known outlier (seeds 2 and 3
find `-76.0372` from the same code); with a cap of 10 this seed finds the good
solution too. That is a solution-quality effect, not just a speed effect, and
it is exactly why the tabulator refuses to report wall time without energy.

## 3. The sweep

`./ci_sweep.sh` — MgH+ CAS(8,12)/cc-pVDZ, seeds 1–6 × {base, f0.3, f0.5,
unc10, unc20} = 30 cells, paired by seed (identical `QED_RANDOM_ORBITAL_SEED`
and `--seed`, so every variant sees the same starting guess and noise stream).
Concurrency capped at 12 (`MAXJOBS`); each run peaks around 4 GB.

`OMP_NUM_THREADS=1` is mandatory, not a performance choice: from a
bitwise-identical pinned guess, threaded CI-solver noise amplifies to 4.5e-6 by
macroiteration 13 — enough to flip a trust-radius accept/reject or the
`step_norm < 0.05` QN trigger. See `../qn_experiment_scratch/orbguess_findings.md`.

Cells already carrying a `CASSCF converged` line are skipped, so the script is
re-runnable to fill gaps after a failure.

### Reading the results

```sh
python tabulate_ci_sweep.py          # '...' = still running, NOT a result
pgrep -fc "run_mgh_case.py --seed"   # 0 means everything finished
```

Nothing is interpretable while `...` markers remain: a partially written log
reads as a *fast-converging* variant, which is exactly backwards. The tabulator
excludes in-progress runs from every summary and only forms the paired summary
over seeds where **all** variants finished.

**Wall time is the primary metric here** (unlike the QN experiment, where it
was macroiteration count) — but the tabulator never reports it alone. Its
`worst dE vs best` column is the guard: anything above ~1e-6 means the variant
landed on a *different, worse* solution, and a wall-time "win" there is not a
win. Check that column before believing any speedup.

## 4. Status / what to do with the result

`src/helper_PFCI.py` is **uncommitted** while this runs, same policy as the QN
experiment: that file is the C++ port's reference and should stay identical to
production unless a change earns its place. The diff is backed up here as
`helper_PFCI_ci_knobs.patch` so it cannot be lost. Decide once the sweep
finishes:

- if a variant robustly wins on wall time **at equal-or-better final energy**,
  it is a real change worth making the default (and worth carrying into the C++
  port's `CasscfCiStateAverageSolver`);
- if the sweep comes back flat, revert, and record it here as a negative result
  the way `../qn_experiment_scratch/RESUME.md` records the QN one.

Either way the profile in section 1 stands on its own and should outlive the
sweep — it is the reason to stop optimizing the orbital side.
