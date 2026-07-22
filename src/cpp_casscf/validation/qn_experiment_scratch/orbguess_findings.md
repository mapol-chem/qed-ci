# Pinned-initial-orbital determinism test — result

Run: `orbguess_case.py`, h2o/6-31G CAS(4,4) SA-CASSCF, seed 1, in `orbguess_run/`.
`--mode make` (QED_RANDOM_ORBITAL_SEED=1, OMP=1) wrote the rotated starting
orbitals to `orbital2.out`; that file was copied to `orbital.out` (kept as
`pinned_guess.backup`, md5 verified unchanged after all reps) and replayed with
`--mode use` three times at OMP_NUM_THREADS=1 and three times at OMP=3.

The pinned start is hard: 19 macroiterations to converge.

## Pinning the guess does not restore bitwise determinism at OMP>1

Metric is the md5 of the full 19-entry `new CI energy` trace, not just the final
energy, so a run that diverges and recovers is still counted as divergent.

| OMP | reps | trace md5s | final-energy spread |
|-----|------|------------|---------------------|
| 1   | 3    | all identical | 0 (bitwise) |
| 3   | 3    | 3 distinct | ~5e-13 |

## The noise amplifies ~8 orders of magnitude mid-trajectory

Max pairwise deviation across the three OMP=3 reps, by macroiteration:

```
macro  0: 4.3e-14   <- threaded CI-solver reductions, from a bitwise-identical guess
macro  2: 3.0e-07
macro 13: 4.5e-06   <- peak
macro 16: 9.0e-11
macro 18: 5.0e-13   <- convergence squeezes it back
```

All six runs still took 19 macroiterations and reached the same solution, so for
this well-conditioned case the divergence was cosmetic.

## What this does and does not license

Confirmed: the guess is a real, separate source of jitter, and pinning it removes
that source — the earlier unpinned h2o measurement (8 vs 11 macroiterations, 7.5e-3
apart, see `mgh_det.sh`'s header) is far larger than anything seen here.

NOT confirmed: that OMP>1 is now safe for variant comparison. A 4.5e-6
mid-trajectory spread is orders of magnitude more than enough to flip a discrete
branch — trust-radius accept/reject, the `step_norm < 0.05` QN activation trigger,
the consecutive-Powell-damping counters the v3/v6 variants are built on — in a case
that sits near one of those thresholds. This case did not sit near one; MgH+
CAS(8,12) is in the experiment precisely because it does. Sample is also one system,
one seed, three reps.

**The OMP_NUM_THREADS=1 requirement for the variant sweeps stands.** Pinning the
guess is a complementary control, not a replacement.
