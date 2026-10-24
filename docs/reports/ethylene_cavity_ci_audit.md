# Ethylene Cavity CI Audit

Testing the zero-weight SA-3 Franck–Condon scan for solution-branch continuity
and active-space adequacy, and deciding whether the MECP scans need re-running.

aug-cc-pVDZ · CAS(4,7) / (4,9) / (4,10) · ω = 0.1212727168509417 Eh · λ = 0 → 0.4 a.u.

This records what was tested after two rounds of review. Every claim below is
backed by a run, and where my earlier attribution was wrong it is marked as
withdrawn rather than quietly replaced. Three of the reviewer's points hold, one
mechanism is refuted, and one prescription produces a mixed result.

| Concern | Verdict |
|---|---|
| λ = 0.1 is the anomalous point, not λ = 0.2 | **Holds** |
| Mechanism: it "collapsed onto the SA-2 stationary point" | **Refuted** |
| Geometry effect of −1.0 eV is implausibly large | **Holds — my error** |
| The scan is not one surface end to end | **Holds** |
| Enlarging the active space fixes the instability | **Holds** |
| Test: penalty becomes smooth *and* small (~0.09 eV) | **Smooth, not small** |
| 9.56 eV should survive a bigger active space | **Survives** |
| Drop the SS-vs-SA-2 bias analysis | **Agreed, +1 reason** |

---

## 1. The λ = 0.1 solution is real, but the branch structure is too

**Claim.** The λ = 0.1 zero-weight run converged onto the SA-2 stationary point —
ground-state orbitals, valence state too high.

**Test.** Seed the zero-weight SA-3 optimization directly *from* the SA-2 λ = 0.1
orbitals and see whether it stays.

**Result.** It moves **16 macroiterations away** and lands on a fourth, distinct
solution at SA = −77.9074748040. SA-2 is therefore *not* a stationary point of the
zero-weight SA-3 functional, and the −0.010 eV bias at λ = 0.1 is a coincidence of
energies, not an identity of solutions. → **mechanism refuted**

A second test traced the branch at **fixed geometry** — holding the λ = 0.1 FC
structure and chaining the coupling up from zero, so only λ varies:

| coupling λ | SA energy (Eh) | valence (eV) | root 2, 1-photon |
|---|---|---|---|
| 0.00 | −77.9755399544 | 6.933 | 0.0000 |
| 0.02 | −77.9725388978 | 7.000 | 0.0001 |
| 0.04 | −77.9641442971 | 7.166 | 0.0006 |
| 0.06 | −77.9514909579 | 7.371 | 0.0014 |
| 0.08 | −77.9352677157 | 7.581 | 0.0025 |
| **0.10** | **−77.9156876523** | **7.794** | **0.0039** |

Smooth and monotonic in every observable, and the endpoint is exactly the
forward-chain solution. So the λ = 0.1 point is on the continuous λ → 0 branch *and*
is the lowest solution at λ = 0.1. It is not spurious.

But that does not make the scan continuous, because there are two branches. Branch B
traced downward from λ = 0.1 against branch A traced upward from λ = 0 (root 2's
photon fraction is a clean branch label):

| coupling λ | A: SA / r₂(1ph) | B: SA / r₂(1ph) | A − B |
|---|---|---|---|
| 0.02 | −77.9725389 / 0.0001 | −77.9725389 / 0.0001 | one branch |
| 0.04 | −77.9641443 / 0.0006 | −77.9641443 / 0.0006 | **B merges into A** |
| 0.06 | −77.9514910 / 0.0014 | −77.9482431 / 0.0451 | A lower 3.2 mEh |
| 0.08 | −77.9352677 / 0.0025 | −77.9334425 / 0.0477 | A lower 1.8 mEh |
| 0.10 | −77.9156877 / 0.0039 | −77.9152046 / 0.0463 | A lower 0.48 mEh |
| 0.20 | −77.7704169 / 0.0132 | −77.7796815 / 0.0391 | **B lower 9.3 mEh** |

> Branch B bifurcates off A near λ ≈ 0.04–0.05 and **crosses it before λ = 0.2**.
> The CAS(4,7) forward chain sits on A at λ ≤ 0.1 and on B at λ ≥ 0.2, switching
> silently. The reported surface was not one surface.

This is why re-seeding λ = 0.1 from λ = 0.2 "fixes" the increments — it forces branch
B, which is where λ ≥ 0.2 lives. But in CAS(4,7) that choice is uncomfortable: branch
B is *not* the lowest solution at λ = 0.1, and it does not exist below λ ≈ 0.05, so
the λ = 0 point can only be on A. Patching moves the switch rather than removing it.
λ = 0.1 has four distinct solutions in this space.

---

## 2. CORRECTION — the geometry explanation was wrong (withdrawn)

I previously attributed a **−1.028 eV** shift in the valence excitation to geometry
relaxation between the λ = 0.1 and λ = 0.2 FC structures. The reviewer's arithmetic —
π→π* sensitivity of roughly 5 eV/Å would need ~0.2 Å — was correct, and my
attribution was not.

All five FC geometries are strictly planar (pyramidalization 0.000°, H–C–C–H
dihedrals exactly 0/180°, max |z| ≈ 3e-7 Å), with R_CC contracting smoothly:

```
R_CC (Ang):  1.35749  1.35091  1.34664  1.34276  1.33546
dR_CC(0.1->0.2) = -0.0043 Ang      RMSD = 0.022 Ang
```

Evaluating the λ = 0.1 geometry at λ = 0.2 coupling *on the correct branch* gives
8.029 eV against 8.011 eV at λ = 0.2's own geometry. The geometry effect is
**0.018 eV**. The 1.03 eV was a branch difference my fixed-geometry trace had walked
into. The manuscript's "FC geometry insensitive to the cavity" holds, and the
geometry optimizations are not implicated.

---

## 3. The active-space test

Four chains run by continuation: CAS(4,9) and CAS(4,10), each with the zero-weight
SA-3 chain and the SA-2 chain, across λ = 0.1–0.4, plus λ = 0 anchors.

### 3a. The bifurcation disappears

Root 2's one-photon fraction — the branch label:

| λ | (4,7) | (4,9) | (4,10) |
|---|---|---|---|
| 0.10 | **0.0039** (branch A) | 0.0472 | 0.0471 |
| 0.20 | 0.0387 | 0.0395 | 0.0395 |
| 0.30 | 0.0298 | 0.0304 | 0.0320 |
| 0.40 | 0.0229 | 0.0240 | — |

The A/B split at λ = 0.1 does not exist in either enlarged space.

### 3b. The kink disappears, and 9.56 eV survives

Zero-weight SA-3 valence excitation (one photon, W = (½, 0, ½)):

| λ | (4,7) eV | (4,9) eV | (4,10) eV |
|---|---|---|---|
| 0.00 | 6.911 | 6.991 | 7.011 |
| 0.10 | **7.794** | 7.378 | 7.437 |
| 0.20 | 8.011 | 8.124 | 8.180 |
| 0.30 | 8.815 | 8.931 | 8.943 |
| 0.40 | **9.558** | **9.609** | **9.565** |

Step-to-step increments — the kink test:

| space | 0 → 0.1 | 0.1 → 0.2 | 0.2 → 0.3 | 0.3 → 0.4 |
|---|---|---|---|---|
| (4,7) | +0.883 | **+0.217** | +0.804 | +0.743 |
| (4,9) | +0.387 | +0.747 | +0.807 | +0.678 |
| (4,10) | +0.427 | +0.743 | +0.762 | +0.622 |

> The λ = 0.4 headline moves 9.558 → 9.609 → 9.565 eV — **stable to 0.05 eV** — even
> though the λ = 0.1 point moves by 0.42 eV. The point that was wrong was the one on
> the wrong branch; the endpoint was never in doubt.

The zero-weight SA-3 average also improves by a near-constant offset, the signature
of a stable active-space composition across λ:

```
(4,9)  - (4,7) = -1.11, -1.10, -1.14, -1.11  x 1e-2 Eh
(4,10) - (4,7) = -1.51, -1.51, -1.48, -1.50  x 1e-2 Eh
```

### 3c. The penalty: smooth, but not small

Orbital-averaging penalty = SA-2 root 0 − zero-weight SA-3 root 0 (eV). Starred
values required repairing the SA-2 branch (see 3d).

| λ | (4,7) | (4,9) | (4,10) |
|---|---|---|---|
| 0.10 | **−0.010** | −0.380* | −0.327 |
| 0.20 | **−0.525** | −0.379* | −0.326 |
| 0.30 | −0.500 | −0.342 | −0.328 |
| 0.40 | −0.463 | −0.332 | −0.340 |

The 0.5 eV *jump* vanishes in both enlarged spaces — it was the active space, as
predicted. But the penalty settles near a **constant ~0.33 eV**, not the ~0.09 eV the
pass criterion asked for. Being constant, it does not distort the λ-dependence; it is
the standing cost of averaging the ground state with a diffuse V state at finite
coupling. → **partial pass**

### 3d. A new diagnostic, and two more branch failures

CAS(4,7) ⊂ CAS(4,9), so the larger space cannot *raise* the state average. The (4,9)
SA-2 chain violated this at λ = 0.2 by **+2.20e-03 Eh**. Backward re-seeding found
solutions **1.05e-02** (λ = 0.2) and **6.89e-03 Eh** (λ = 0.1) lower — two branch
failures caught by arithmetic alone, no extra chemistry. Worth adopting as a routine
check.

Note which leg keeps failing: **SA-2, in every active space**. The zero-weight SA-3
chains passed the consistency check at every λ in both spaces. That is the decisive
argument for dropping the penalty column — not that it is uninteresting, but that one
of its two legs will not hold a branch.

### 3e. Residual multiplicity in (4,9)

CAS(4,9) at λ = 0.1 from four different starting points:

| route | SA (Eh) | valence (eV) | r₂(1ph) | vs chain |
|---|---|---|---|---|
| forward chain | −77.9267672692 | 7.378 | 0.0472 | — |
| reverse (from λ = 0.2) | −77.9267672692 | 7.378 | 0.0472 | **+4.8e-11** |
| random seed 1 | −77.9267672678 | 7.378 | 0.0472 | +1.4e-09 |
| random seed 2 | −77.9277182155 | 7.366 | 0.0471 | −9.5e-04 |

Multiplicity is not eliminated — a random scramble finds a solution 0.95 mEh below
the chain. But the comparison that matters is the spread *in the observable*:
CAS(4,7) at λ = 0.1 spans **0.593 eV** across four solutions; CAS(4,9) spans
**0.012 eV**. And the specific pathology — forward and reverse chains disagreeing —
is gone: they now agree to 4.8e-11, where in (4,7) they landed on different branches
differing by 0.53 eV. There is no branch to switch between, so no scan can silently
switch.

---

## 4. Where the CAS(4,7) deficiency actually lives

CAS(4,7) with aug-cc-pVTZ is the literature standard for ethylene conical
intersections. That standing is not in question, and the audit supports it. Four
random starting orbital sets at each of four settings:

| setting | solutions | spread |
|---|---|---|
| CI geometry, λ = 0, no cavity *(literature case)* | **1** | **0.00e+00** |
| FC geometry, λ = 0, no cavity | 2 | 7.1 mEh; canonical finds the lower |
| FC geometry, λ = 0.1, cavity | **4** | 8.2 mEh, **0.593 eV** in the observable |
| CI geometry, λ = 0.4, cavity | 2 | 5.5 mEh — see below |

> At the no-cavity conical intersection every random scramble returns
> −77.9272572988 with **zero spread**. The failure is specific to the Franck–Condon
> geometry *with the cavity on* — a regime the literature never tested, and where the
> diffuse V-state orbital must also carry the photon coupling.

### 4a. The MECP scans are not contaminated

At the λ = 0.4 CI geometry the two CAS(4,7) solutions are:

| solution | SA-2 average (Eh) | gap (eV) |
|---|---|---|
| canonical + seeds 2, 3 | **−77.3051342** | **0.0007** |
| seed 1 | −77.2996787 | 0.7142 |

The correct solution is lower in the SA-2 average — which is what SA-2 minimizes —
*and* is the only degenerate one. The MECP algorithm drives the gap to zero, so it
cannot converge to the other. Confirming check: the recomputed E₀ = −77.30514648
matches the −77.30513428 used in the summary table to 1.2e-05 Eh.

### 4b. The geometries do not transfer

The CAS(4,7)-optimized MECP geometry evaluated in larger active spaces:

| geometry | (4,7) gap | (4,9) gap | (4,10) gap |
|---|---|---|---|
| λ = 0.4 CI | 0.0007 eV | 0.0581 eV | 0.0770 eV |
| λ = 0 CI | 0.0000 eV | 0.1149 eV | 0.1878 eV |

The degeneracy is lifted, so a consistent (4,9) margin column would require genuinely
re-optimizing every intersection.

---

## 5. Cost

One CASSCF plus two analytic gradients at the λ = 0.4 MECP geometry, 8 threads. The
MECP driver needs both state gradients per step.

| space | CASSCF | grad(0) | grad(1) | per step | 500 steps |
|---|---|---|---|---|---|
| (4,7) | 23.3 s | 65.3 s | 70.8 s | 159.5 s | 22.2 h |
| (4,9) | 19.7 s | 65.5 s | 69.6 s | **154.7 s** | **21.5 h** |

**The larger active space costs nothing extra** — it is marginally faster here,
because the CASSCF converges in fewer macroiterations, offsetting the larger CI. The
gradient dominates at ~85% of each step and is insensitive to `nact` at this size. So
the cost objection applies to the scan length, not to the active space. Against the
original ~5 days per 500-step leg, this is roughly 5× faster; a full 1500-step λ point
is ~2.8 days. A complete five-λ re-do remains ~6 weeks of wall time.

Caveat on that figure: it is measured at a converged geometry with a canonical guess
and no orbital reuse. A production scan seeds each step from the previous geometry's
orbitals, which should cut the ~20 s CASSCF portion but leave the ~136 s of gradient
untouched — so a realistic per-step figure is 140–155 s.

---

## 6. Recommendation

1. **Keep CAS(4,7) for production.** It is validated at the no-cavity intersection
   (unique from every random start), the MECP scans are on the correct branch, and the
   physics is active-space-independent at the level of the increments — (4,7) with the
   branch fixed gives +0.349, +0.751, +0.804, +0.743 against (4,9)'s +0.387, +0.747,
   +0.807, +0.678.

2. **Re-run one single point.** The λ = 0.1 FC zero-weight point, seeded from λ = 0.2,
   so the FC scan sits on one branch. Minutes, not days.

3. **Cite the (4,9)/(4,10) chains as the active-space validation.** They are already
   complete — all four chains, every λ, plus the variational-consistency check. This is
   a stronger referee answer than a partial re-do, because a mixed (4,9)-FC / (4,7)-CI
   margin would be inconsistent in a way the present table is not.

4. **Drop the SS-vs-SA-2 bias analysis**, for the original reason plus one more: the
   SA-2 leg will not hold a branch in any active space.

5. **If a referee demands a consistent (4,9) margin**, seed the MECP from the converged
   (4,7) geometry (gap already 0.058 eV at λ = 0.4 rather than from a guess) and do
   λ = 0 and λ = 0.4 only — roughly two days, not six weeks.

The compromise being accepted, stated plainly: the λ = 0.1 point in the recommended
(4,7) sequence is branch B, which is *not* the lowest solution at λ = 0.1 and does not
connect to λ = 0. Documented, that is honest; undocumented, it is the same silent
switch this audit started from. The (4,9) results are what justify the choice, since
there the single continuous branch agrees with branch B.

---

## 7. Open items

- **The margin column mixes active spaces.** The CI energies are (4,7) values; the FC
  valence excitations quoted for (4,9)/(4,10) are not paired with (4,9)/(4,10)
  intersections. Only the (4,7) margins are internally consistent today.

- **The (4,9) chain is not the global minimum** at λ = 0.1, and only two random seeds
  were run there. A production (4,9) scan would want more seeds per λ and the same
  variational-consistency check against (4,10).

- **Geometry chains were not generated by continuation.** The FC optimizations
  pre-date this audit; each λ's structure was optimized independently under SA-2
  orbitals. The resulting off-stationarity on the zero-weight surface is |g| = 0.0122
  at λ = 0.4, roughly 3 meV of relaxation energy — second order, as assessed.

- **The physical story is unaffected.** The intersection remains energetically
  reachable at every λ; the margin grows from 2.13 to 3.43 eV because the DSE
  penalizes the spatially diffuse V state at the planar geometry more than the compact
  intersection. Both stated caveats stand: sensitivity of the V-state DSE to the
  description of ⟨x²⟩, and the fixed-orientation assumption.

---

### Reference: the summary table as it stands (CAS(4,7), continuation chain)

| λ | S0(FC)→CI | S0(FC)→valence | margin | reachable? |
|---|---|---|---|---|
| 0.00 | 4.78 | 6.91 | 2.13 | yes |
| 0.10 | 4.94 | 7.79 | 2.85 | yes |
| 0.20 | 4.93 | 8.01 | 3.08 | yes |
| 0.30 | 5.43 | 8.81 | 3.38 | yes |
| 0.40 | 6.13 | 9.56 | 3.43 | yes |

With the λ = 0.1 branch repaired per recommendation 2, the λ = 0.1 row becomes
cost 4.66, valence 7.26, margin 2.60, and the valence increments become
+0.349, +0.751, +0.804, +0.743.

---

Runs and logs: `~/mgh_repro_20260804/fc_zeroweight_ab/` — `old_chain/` (five-λ
continuation), `probe/nact*_sa*.log` (enlarged spaces), `probe/branch*_trace.log`
(branch traces), `probe/T_n*.log` (timings).
