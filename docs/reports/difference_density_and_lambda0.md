# New analysis: difference-density selection, and the λ=0 coupling-solve bug

Follow-up to the main report. Three results, all from runs completed after it.

---

## 1. The density-overlap metric was blunt — confirmed, and it changes the (4,7) verdict

The criticism is right. `Tr(D_A S D_B S)` over 16 electrons is dominated by the
core and σ framework, which are identical in every solution. Using the
**difference density** (root 2 − root 0) instead, on the same saved CAS(4,7)
λ=0.1 data:

| solution | ⟨r²⟩ | full-density overlap | difference-density overlap |
|---|---|---|---|
| A forward chain (7.794 eV) | 108.93 | 0.995204 | **0.925536** |
| B reverse chain (7.260 eV) | 110.68 | 0.997168 | **0.952377** |
| basin 3 (7.201 eV) | 110.91 | 0.996786 | **0.934622** |
| from SA-2 orbitals (7.663 eV) | 108.42 | 0.994792 | **0.910254** |

Spread widens from 0.25% to **4.2%** — 17× more discriminating.

### 1.1 The predicted two-cluster structure is NOT there

The expectation was that the difference density would separate the ⟨r²⟩/valence
clusters (108.4–108.9 / 7.66–7.79 vs 110.7–110.9 / 7.20–7.26). It does not.
Pairwise difference-density overlaps:

```
           A fwd    B rev   basin3  fromSA2
A fwd      1.0000   0.9889   0.9685   0.9823
B rev      0.9889   1.0000   0.9764   0.9796
basin3     0.9685   0.9764   1.0000   0.9456
fromSA2    0.9823   0.9796   0.9456   1.0000
```

Within-cluster pairs score 0.9823 (A–fromSA2) and 0.9764 (B–basin3), while the
*highest* overlap of all, 0.9889, is the cross-cluster pair A–B. No two-group
structure; the four solutions are mutually similar at 0.946–0.989.

### 1.2 What it does show: the two sound criteria disagree

The difference-density overlap ranks the solutions, and the ranking contradicts
the variational principle:

| rank by V-state character | solution | valence (eV) | SA energy (Eh) |
|---|---|---|---|
| 1 | B reverse chain | 7.260 | −77.9152046 |
| 2 | basin 3 | 7.201 | −77.9094651 |
| 3 | A forward chain | 7.794 | **−77.9156877 (lowest)** |
| 4 | from SA-2 | 7.663 | −77.9074748 |

**The most V-state-like solution is not the variational winner, and the
variational winner ranks third.** In CAS(4,7) the two defensible criteria —
best state character, lowest state-averaged energy — select different solutions
differing by **0.53 eV** in the V-state energy.

This is a sharper statement than "irreducible ambiguity": it is not that we
cannot choose, it is that the two principled ways of choosing disagree, which is
what "none of the solutions is complete" looks like quantitatively. A ±0.3 eV
tolerance around a chosen value understates it; the choice itself is between two
criteria 0.53 eV apart.

---

## 2. The λ=0 coupling solve: a projection bug, now confirmed and fixed

Diagnosis confirmed in full. At exact degeneracy (H − E_n) has a null direction
along c⁽ᵐ⁾, and the CI-response projection must remove it. Testing both settings
on the λ=0 MECP geometry, CAS(4,7), `compute_grad(0,1)`:

| | GMRES restarts | Max Iterations | wall time | \|h₀\| |
|---|---|---|---|---|
| `QED_PROJ_ALL=0` (current default) | **2002**, stalled at 1.3220e-07 | **reached** | **821 s** | 1.354530e-06 |
| `QED_PROJ_ALL=1` | **3** | no | **29 s** | **7.809573e-02** |

Three consequences:

- **The solve should not be skipped.** With the correct projection it converges
  in 3 restarts, 28× faster.
- **h at exact degeneracy is finite**, as expected analytically. The single-vector
  projection returned |h₀| = 1.35e-06 — an unconverged solve masquerading as a
  vanishing coupling. The correct value is 7.81e-02, five orders larger.
- **The λ=0 seam norm IS well defined.** An earlier claim that it is not is
  withdrawn: the branching direction was never undefined by degeneracy, it was
  simply never computed. This also explains the 6.6× seam discrepancy at λ=0
  (3.28e-03 vs the reported 4.997e-04) — that run used a garbage h.

It also explains the λ=0 per-step slowdown (~3.5× vs λ=0.4) that the main report
leaves unexplained. An earlier attempt to measure it counted
`Convergence at Inner Iter` — the inner loop — and concluded λ=0 did *less*
work; the relevant counter is `Iter (Restart)`, which the production launcher
filters out of the logs.

---

## 3. Why "project all" works here, and why it is NOT the general rule

The question is the right one, and the answer is that **"all" is correct here
only by coincidence of the 2-root setup.**

The null space of (H − E_n) is spanned by c⁽ⁿ⁾ **and by every c⁽ᵐ⁾ degenerate
with it**. Non-degenerate states are *not* null directions:
(H − E_n)c⁽ᵏ⁾ = (E_k − E_n)c⁽ᵏ⁾ ≠ 0. Projecting those out removes a legitimate
component of the response — over-projection.

Measured degeneracies in the two calculation types:

**MECP runs** (`davidson_roots=2`, equal weights) — the two roots *are* the
degenerate pair:

| λ | gap (eV) | gap (Eh) |
|---|---|---|
| 0.00 | 2.28e-05 | 8.38e-07 |
| 0.10 | 1.03e-05 | 3.78e-07 |
| 0.20 | 5.14e-05 | 1.89e-06 |
| 0.30 | 5.25e-05 | 1.93e-06 |
| 0.40 | 1.50e-05 | 5.52e-07 |

**FC runs** (`davidson_roots=3`, weights (1,0,1)) — the three roots are far apart:

| λ | E₁−E₀ (eV) | E₂−E₀ (eV) |
|---|---|---|
| 0.10 | 3.269 | 7.794 |
| 0.20 | 3.208 | 8.011 |
| 0.30 | 3.173 | 8.815 |
| 0.40 | 3.165 | 9.558 |

So:

- In the **MECP** calculation, "all averaged CI vectors" **equals** the degenerate
  subspace. `project_out_all` is correct there — which is why it fixes λ=0.
- In the **FC** calculation, "all" is strictly larger than the degenerate subspace
  (nothing is degenerate), so `project_out_all` **over-projects**. That is
  exactly the documented unequal-weight failure recorded at
  `nuclear_grad.py:38–104`: errors of 1.4e-03 to 7.5e-03 with `project_out_all`
  where the minimal projection gives 6.5e-09.

**With three states, two degenerate, "project all" is wrong.** The correct rule
is to project onto the complement of the degenerate subspace containing state n:

    null space = span{ c⁽ᵏ⁾ : |E_k − E_n| < tol }   (which always includes n)

and to leave non-degenerate roots alone. Neither existing setting implements
this: `QED_PROJ_ALL=0` under-projects at degeneracy (the λ=0 stall),
`QED_PROJ_ALL=1` over-projects when states are separated (the unequal-weight
errors). They happen to be individually correct in the two regimes this project
uses, which is why the bug stayed hidden.

**Recommendation:** replace the boolean with a degeneracy-aware projection using
an energy tolerance, so the correct behaviour is automatic rather than depending
on an environment variable matching the deck. `QED_PROJ_ALL=1` should **not** be
made the default — it would reintroduce the unequal-weight error at every FC
point.

There is a second, separate site: the GMRES trial/sigma projection at
`nuclear_grad.py:830/840/891` is hardcoded to the single-vector form with
`project_out_all` commented out above each, and is not covered by the knob at
all. It sits inside a `if norm(residual) < 1e-3` random-guess branch, so it may
not be reached on every path, but it should follow the same rule.

---

## 4. What should be re-run

- **λ=0 MECP, both active spaces**, with the correct projection. Now ~29 s per
  coupling solve rather than 821. The energetic effect on the published number
  was 0.07 meV, so this is for correctness of the reported seam and gap, not a
  change to the table.
- Nothing else. The λ≥0.1 MECPs have gaps of 3.8e-07 to 1.9e-06 Eh, so they are
  in the same degenerate regime and were solved with the same under-projection —
  worth re-running for consistency, but their CI energies shifted <0.5 meV under
  re-tightening, which bounds the effect.
