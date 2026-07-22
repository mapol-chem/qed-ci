# QN/BFGS variant experiment — CLOSED 2026-07-22, negative result

Nothing here is running and nothing here is pending. This directory is kept
as the evidence trail for a question that was asked and answered.

**The write-up lives in `cpp_casscf/README.md`** — section "The QN/BFGS
convergence-variant experiment: a negative result", under "What's still open".
Read that, not this file, for the conclusions.

## One-paragraph summary

Does the Python's QN/BFGS exact-Hessian reset policy leave convergence on the
table? No. Five variants, swept paired-by-seed on small systems (150
runs/variant) and then on MgH+ CAS(8,12)/cc-pVDZ at OMP=1 (6 seeds x 6
variants, 36/36 converged to the same solution): all indistinguishable from
default except v7 (`QED_QN_REJECT`), which is decisively **worse** — 16.33 vs
12.50 mean macroiterations, +39% wall, 954 rejections. v3/v4 never fire on
this system (2 Powell-damping events across all six v0 seeds). v6's small-system
reset halving does not generalize (342 vs v0's 345 on MgH+). v2 really does cut
resets 59% but buys no wall time — **the useful finding: exact-Hessian rebuilds
are not the bottleneck, Davidson is.**

## What happened to the code

The `helper_PFCI.py` toggles were **reverted**. The Python here is the C++
port's reference and has to stay identical to production. The full diff is
preserved as `helper_PFCI_qn_experiment.patch` in this directory.

**Kept** (still uncommitted, gated, default-no-op): only the
`QED_RANDOM_ORBITAL_SEED` random-orbital-guess block, which is hunk 1 of that
patch. It is the only reproducer for the hard >10-macroiteration regime,
`run_mgh_case.py` depends on it, and it carries the MO sign-pinning fix without
which the whole experiment is irreproducible. To restore the full experiment:
`git apply helper_PFCI_qn_experiment.patch` (it applies over a clean
`helper_PFCI.py`; the kept hunk is its first).

## Artifacts

| | |
|---|---|
| `mgh_det_logs/` | 36 MgH+ runs, `s<seed>_<variant>.log` |
| `tabulate_mgh_det.py` | the results table (macroiterations, resets, energies) |
| `results_paired_v0_v4.json`, `results_v6.json`, `results_v7.json` | small-system sweeps |
| `qn_sweep.py`, `mgh_det*.sh` | sweep drivers |
| `orbguess_findings.md`, `orbguess_run/` | the pinned-orbital determinism test |
| `superseded_v6_or_semantics/` | pre-rewrite v6 runs, **not** comparable to the rest |

Useful per-log counters: `solve step for original hessian` (exact-Hessian
resets), `qwrqwrq`/`erewrw` (energy-raising vs -lowering QN steps), `QNREJECT`,
`Curvature Violation` (Powell damping).

## Two methodology lessons worth not rediscovering

- **`OMP_NUM_THREADS=1` is mandatory** for comparisons like this. From a
  bitwise-identical pinned guess, threaded CI-solver noise starts at 4.3e-14 and
  amplifies to 4.5e-6 by macroiteration 13 — enough to flip a trust-radius
  accept/reject or the `step_norm < 0.05` QN trigger. See `orbguess_findings.md`.
- **Never compare macroiteration counts alone.** A variant that stops looser
  looks faster. Every cell must be reported with its final energy.

One caveat on the data, checked rather than assumed: `helper_PFCI.py` was edited
at 23:50 on 07-21 while 12 v2/v3/v4 cells were in flight. That rewrite only
converted **v6's** OR into a three-way choice — v2 was already a replacement
branch — and v2's reset-reduction regime is the same either side of the boundary
(seeds 1-4: 0.39-0.61x v0; seeds 5-6: 0.29-0.63x). The cells are comparable.
