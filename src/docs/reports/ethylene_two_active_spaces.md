# Ethylene polaritonic CI — CAS(4,7) vs CAS(4,9), both legs re-converged

All numbers below are from runs completed in this session. aug-cc-pVDZ,
ω = 0.1212727168509417 Eh, λ along x, C–C direction fixed on x.

---

## 1. Headline: the published accessibility table survives

Re-optimizing every CAS(4,7) MECP with a quasi-Newton optimizer, per-step
re-alignment and tighter gaps moves the CI energies by **under half a meV**:

| λ | published CI (Eh) | re-tightened (Eh) | shift |
|---|---|---|---|
| 0.00 | −77.92725732 | −77.92725995 | −0.07 meV |
| 0.10 | −77.87728073 | −77.87727935 | +0.04 meV |
| 0.20 | −77.74675256 | −77.74675280 | −0.01 meV |
| 0.30 | −77.55093527 | −77.55095184 | −0.45 meV |
| 0.40 | −77.30513428 | −77.30514254 | −0.22 meV |

Against margins of 2–3.5 eV, the cost column stands as published.

## 2. Both columns, equally converged

Same five starting geometries, same thresholds, same per-step re-alignment.

**CAS(4,7)**

| λ | steps | mean (Eh) | gap (eV) | f_seam | f_seam_mw | R_CC (Å) | status |
|---|---|---|---|---|---|---|---|
| 0.00 | 5 | −77.92725995 | 2.28e-05 | 4.77e-04 | 3.46e-04 | 1.40893 | converged |
| 0.10 | 2 | −77.87727935 | 1.03e-05 | 1.94e-04 | 5.16e-05 | 1.40387 | converged |
| 0.20 | 4 | −77.74675280 | 5.14e-05 | 2.21e-04 | 1.23e-04 | 1.38375 | converged |
| 0.30 | 6 | −77.55095184 | 5.25e-05 | 2.30e-04 | 1.57e-04 | 1.35510 | converged |
| 0.40 | 15 | −77.30514254 | 1.50e-05 | 7.30e-04 | 5.36e-05 | 1.33113 | **plateaued** |

**CAS(4,9)**

| λ | steps | mean (Eh) | gap (eV) | f_seam | f_seam_mw | R_CC (Å) | status |
|---|---|---|---|---|---|---|---|
| 0.00 | 13 | −77.93592243 | 1.21e-05 | 4.18e-04 | 3.79e-04 | 1.41037 | converged |
| 0.10 | 16 | −77.88848050 | 8.56e-06 | 3.78e-04 | 2.53e-04 | 1.40382 | converged |
| 0.20 | 7 | −77.75832760 | 6.51e-05 | 3.81e-04 | 2.16e-04 | 1.38304 | converged |
| 0.30 | 14 | −77.56285364 | 6.66e-05 | 3.66e-04 | 1.91e-04 | 1.35566 | converged |
| 0.40 | 13 | −77.31778350 | 4.54e-05 | 3.10e-04 | 9.19e-05 | 1.33212 | converged |

CI energies: `ci_energies_n7.json`, `ci_energies_n9.json`, `report_data.json`.

**Note on the (4,9) starting geometries.** They are the converged (4,7) MECPs,
which are *not* degenerate in (4,9) — the gap there opens to 0.058–0.119 eV. So
these were genuine re-optimizations, not single-point evaluations. The
(4,9)−(4,7) CI offset is a near-constant ~11–13 mEh.

## 3. Accessibility margins

| λ | margin (4,7) | margin (4,9) |
|---|---|---|
| 0.00 | 2.13 | 2.22 |
| 0.10 | 2.85 | 2.65 |
| 0.20 | 3.08 | 3.15 |
| 0.30 | 3.38 | 3.46 |
| 0.40 | 3.43 | 3.50 |

Both monotonic; the intersection stays energetically reachable at every λ.

**Caveat on the FC leg.** These margins use the FC values from a continuation
chain. The seed scan of §4.1 shows that chain is **not** on the lowest solution
at λ=0.1 or λ=0.2 in (4,9). Re-selecting by the §4.6 procedure (density overlap,
then variational) gives the FC sequence 6.991 → 7.366 → 8.110 → 8.931 → 9.609 eV
(increments +0.375, +0.744, +0.821, +0.678), which moves the λ=0.1 and λ=0.2
margins by ~0.01 eV each — negligible, but the selected values are the ones to
quote. In (4,7) the branch choice is worth up to 0.59 eV and cannot be resolved
(see §4.3).

## 4. Active-space question — REVISED

An earlier version of this report argued that CAS(4,9) should be preferred
because CAS(4,7) has multiple SA solutions at the FC geometry. **That argument
does not survive testing and is withdrawn.** CAS(4,9) has multiple solutions at
FC as well. The correct distinction is different, and sharper.

### 4.1 Multiplicity is a property of the FC geometry, not of the active space

Identical protocol both spaces: zero-weight SA-3 at the FC geometry, six random
orbital scrambles per λ.

| λ | (4,7) solutions | (4,7) valence span | (4,9) solutions | (4,9) valence span |
|---|---|---|---|---|
| 0.10 | 4 | 0.593 eV | 2 | 0.011 eV |
| 0.20 | 4 | — | 4 | 2.525 eV |
| 0.30 | 2 | — | 4 | 3.234 eV |
| 0.40 | 4 | — | 3 | 3.156 eV |

Planar ethylene is D2h, so the diffuse orbitals fall in near-degenerate symmetry
blocks and which one occupies a weakly-occupied active slot is barely
determined. Enlarging the active space does not remove that.

### 4.2 (4,7) is chemically converged — the occupations say so

Active natural occupations at the FC geometry, zero-weight SA-3:

| λ | space | occupations |
|---|---|---|
| 0.1 | (4,7) | 1.9798 1.4432 0.4722 0.0527 0.0280 0.0180 0.0060 |
| 0.1 | (4,9) | 1.9742 1.4431 0.4717 0.0508 0.0285 0.0176 0.0067 **0.0057 0.0017** |
| 0.4 | (4,7) | 1.9559 1.3464 0.4532 0.1506 0.0496 0.0408 0.0035 |
| 0.4 | (4,9) | 1.9511 1.3458 0.4507 0.1480 0.0499 0.0432 0.0046 **0.0041 0.0026** |

The added orbitals carry 0.002–0.006, far below the conventional 0.02 threshold,
and (4,7)'s seven track the larger space's first seven closely. So (4,7) is not
truncating correlation, and an occupation-number argument does **not** justify
enlarging. (4,7)/aug-cc-pVTZ is the literature standard for ethylene conical
intersections and this audit supports it: at the no-cavity CI geometry, four
random scrambles return one solution with **zero** spread.

### 4.3 The real distinction: in (4,9) the multiplicity is resolvable, in (4,7) it is not

Selection must not use the energy trend — that is circular, since the trend is
the quantity being measured. The sound test is the **state character**: the
per-state 1-RDM transformed to the AO basis, which is independent of the MO
basis and therefore comparable between solutions whose active orbitals are
internally rotated. Two observables, neither consulting an energy: ⟨r²⟩ (the V
state is diffuse) and the density overlap Tr(D_A S D_B S)/norms against the
*cavity-free* states.

**CAS(4,9), λ=0.2 — the test discriminates:**

| solution | SA (Eh) | valence | root2 \|⟨r²⟩\| | overlap vs cavity-free valence |
|---|---|---|---|---|
| s2 | −77.79270870 | 8.110 | 101.466 | **0.992119** |
| s6 | −77.79169990 | 8.090 | 101.348 | **0.992110** |
| s1 | −77.78357460 | 7.943 | 101.484 | **0.991929** |
| s5 | −77.73953160 | 10.469 | **81.896** | **0.927829** |

s5's root 2 is compact (81.9, barely above its own ground state at 81.1) and its
overlap drops to 0.928 — it is not the V state. The three valid solutions agree
to **0.167 eV**. Cavity-free reference: ⟨r²⟩ = 121.3, so the cavity contracts
the diffuse orbital, as the DSE should.

**CAS(4,7), λ=0.1 — the test does NOT discriminate:**

| solution | SA (Eh) | valence | root2 \|⟨r²⟩\| | overlap vs cavity-free valence |
|---|---|---|---|---|
| forward chain A | −77.9156877 | 7.794 | 108.933 | 0.995204 |
| reverse chain B | −77.9152046 | 7.260 | 110.680 | 0.997168 |
| random basin 3 | −77.9094651 | 7.201 | 110.911 | 0.996786 |
| from SA-2 orbitals | −77.9074748 | 7.663 | 108.418 | 0.994792 |

All four are the **same electronic state** — overlap 0.9948–0.9972, ⟨r²⟩
108.4–110.9, all clearly the diffuse V state — and they nevertheless disagree by
**0.593 eV** in the V-state energy. There is nothing wrong with any of them to
detect, so no diagnostic can select among them. The only remaining selector is
the variational principle, and it picks the forward chain (valence 7.794 eV) —
the solution furthest from the smooth trend.

**Statement that survives:** in both spaces the FC geometry supports several SA
solutions. In (4,9) the spurious ones are identifiable by density overlap and
⟨r²⟩, and the physically valid ones agree to ≤0.17 eV. In (4,7) all four
solutions are the same state and still disagree by 0.59 eV: the ambiguity is
irreducible and can only be reported, not resolved. Any (4,7) FC V-state energy
should therefore carry a systematic of roughly ±0.3 eV from branch choice.

### 4.4 Supporting evidence for (4,9)

- **Forward/reverse chain agreement.** In (4,7) a forward chain from λ=0 and a
  reverse chain from λ=0.2 land 0.53 eV apart at λ=0.1, and the scan switches
  branches between λ=0.1 and 0.2 with no indication — two branches that
  bifurcate near λ≈0.04–0.05 and cross before λ=0.2 (A lower by 3.2/1.8/0.48
  mEh at λ=0.06/0.08/0.10, then B lower by 9.3 mEh at λ=0.20). In (4,9) forward
  and reverse agree to 4.8e-11.
- **λ=0.4 V-state energy is active-space stable:** 9.558 (4,7) / 9.609 (4,9) /
  9.565 (4,10) eV.
- **Variational consistency** (a subspace cannot raise the state average): the
  zero-weight SA-3 average improves by a uniform offset, the signature of a
  stable active-space composition — (4,9)−(4,7) = −1.11/−1.10/−1.14/−1.11 ×10⁻²
  Eh, (4,10)−(4,7) = −1.51/−1.51/−1.48/−1.50 ×10⁻² Eh. The same check caught two
  bad SA-2 branch points, repaired by backward re-seeding (1.05e-02 and 6.89e-03
  Eh lower).
- **CI geometries are clean.** At the converged (4,9) MECPs, λ=0.0 and λ=0.1 give
  one solution each from a canonical guess plus two random scrambles, agreeing to
  3e-11 Eh. Multiplicity is an FC-geometry phenomenon; at the intersection the
  molecule is twisted and pyramidalised, symmetry is broken and the two states
  are held degenerate by the seam condition, so the description is pinned.

### 4.5 Orbital-averaging penalty — recommend dropping the column

SA-2 root0 minus zero-weight SA-3 root0 (eV). The 0.5 eV *jump* vanishes in the
larger spaces but the magnitude settles near a constant ~0.33 eV, not the
~0.09 eV a "small and smooth" criterion would want:

| λ | (4,7) | (4,9) | (4,10) |
|---|---|---|---|
| 0.10 | −0.010 | −0.380 | −0.327 |
| 0.20 | −0.525 | −0.379 | −0.326 |
| 0.30 | −0.500 | −0.342 | −0.328 |
| 0.40 | −0.463 | −0.332 | −0.340 |

The SA-2 leg would not hold a branch in any active space, which is the decisive
reason to drop this column rather than report it.

### 4.6 Recommended procedure for any FC number

1. Scan random orbital seeds at every λ (six was enough to expose four solutions).
2. Identify the state by **density overlap** against the cavity-free (or
   previous-λ) reference, plus ⟨r²⟩. Do **not** select on the energy trend.
3. Apply the variational principle within the set that passes step 2.
4. Report the residual spread among valid solutions as a systematic.

## 5. λ=0 is a special case: the derivative-coupling equations are singular

At λ=0 the two states are degenerate to 1.19e-06 eV and there is no cavity, so
the CP-SA-CASSCF equations for the derivative coupling are singular. GMRES
stalls: residual reaches 1.3412e-07 at restart 119 and is unchanged at restart
479 (531 restarts total), and the production run reports `Max Iterations
Reached` once at λ=0 against zero times at λ=0.4.

Consequence: `h` is unconverged, so `u_h` is not a defined direction, so
`|f_seam| = |σ − (σ·u_g)u_g − (σ·u_h)u_h|` is **not a well-defined number** at
λ=0. Two codes disagree 6.6× at the identical geometry (4.997e-04 vs 3.28e-03)
and neither is more correct. The λ=0 seam should not be used as a convergence
criterion; converge on the mean-energy gradient with only `u_g` projected out.

The energetic consequence is nil — re-optimizing λ=0 moved the energy 0.07 meV
and R_CC by 0.0003 Å.

## 6. Orientation drift (affects the published geometries)

`project_out_transrot` removes only *infinitesimal* rotations, so finite
rotation accumulates. In the published λ=0.4 run the C–C axis drifted to
**1.12°** off the polarization axis within leg 1, resetting at each manual
rerun (leg 2: 0.39°, leg 3: 0.099°).

Calibration, measured directly by re-aligning the λ=0.4 start and recomputing:
**0.0987° ⇔ ΔE₀ = 1.46e-05 Eh and Δgap = 2.90e-04 eV**, i.e. ~3× the 1e-04 eV
gap tolerance. Leg 1's 1.12° is worth ~3e-03 eV, ~30× tolerance.

The published geometries carry 0.03–0.22° of residual tilt. Energy effect
~1e-05 Eh, so no conclusion is threatened, but it is larger than the quoted
convergence thresholds. Re-imposing the direction every step fixes it (tilt
0.00e+00 throughout), and the residual per step falls quadratically with the
step size.

## 7. Performance (why this was affordable)

| change | effect | verification |
|---|---|---|
| `ao_tei_deriv1` hoisted out of the Cartesian loop | gradient 68.1 → 25.7 s (2.65×) | bitwise identical, 18/18 components, CASSCF removed from the test |
| quasi-Newton MECP (exact Newton on the gap + BFGS in the seam) | 1500 → 19 steps at λ=0.4 (79×) | reproduces production numbers at both start geometries |

λ=0.4 from the published leg-1 start: gap 6.910e-04 → 5.534e-06 eV (125×),
f_seam_mw 4.145e-04 → 6.765e-05 (6.1×), in 19 steps vs 1500. A λ point is now
~30 min rather than ~5 days; all five CAS(4,9) MECPs took ~25 min in parallel.

## 7a. The MECP algorithm

Same Bearpark–Robb–Schlegel (BRS) effective gradient as the original driver; the
step construction is different. All quantities in Hartree/Bohr, all vectors in
mass-weighted Cartesians q = M^(1/2) x.

**Branching plane.** With σ = (∇E_p + ∇E_q)/2, g = ∇E_q − ∇E_p, and h the
derivative coupling:

    u_g = g / |g|
    u_h = normalise( h − (h·u_g) u_g )

**The step is SPLIT, not summed.** The two directions have different local
structure, so they get different treatments:

- *Along u_g the gap is LINEAR*, with slope |g|. Its exact Newton (root-finding)
  step is therefore known in closed form and needs no Hessian:

      s_gap = −(ΔE / |g|) · u_g

- *Inside the seam the mean energy is QUADRATIC*, so BFGS belongs there, applied
  only in the orthogonal complement Q of span{u_g, u_h}:

      g_seam = Q σ
      s_seam = −B⁻¹ g_seam              (B updated by BFGS with Powell damping)

- Total step s = s_gap + s_seam, then scaled to the trust radius. The two live
  in orthogonal subspaces and cannot fight each other; the BFGS history is
  projected with Q before storage, so no gap motion enters the Hessian.

**Why not hand the summed BRS gradient to BFGS** (which is the textbook recipe,
and what the original driver effectively does via steepest descent): the step
along u_g then becomes 2ΔE/B, where B is curvature accumulated from the *seam* —
unrelated to |g|. With the usual 0.5 diagonal guess that gives 4ΔE where 6.7ΔE
is required at |g| ≈ 0.15, a 1.7× under-correction. Measured consequence on the
λ=0.4 deck: the gap grew monotonically 0.00068 → 0.0024 → 0.0117 → 0.0130 →
0.0137 eV over five steps. Splitting the step fixed it.

**Trust radius.** Steered by |g_seam| in mass-weighted coordinates — the only
part BFGS models, since the BRS gradient is not the gradient of any scalar.
Expand ×1.25 on improvement, halve otherwise, bounded [1e-3, 1.0] Bohr. There is
no reject-and-retry: a poor step shrinks the radius for the next one instead of
being recomputed, because each evaluation costs three gradients.

**Translation/rotation.** Removed with the *symmetric* mass-weighted projector.
Note that in plain Cartesians the projector for displacements is
M^(−1/2) P M^(1/2) while for gradients it is the transpose M^(1/2) P M^(−1/2);
a single non-symmetric matrix cannot serve both, and using the displacement
projector on a gradient is silently wrong (still idempotent, still kills
translations, but mixes modes). Working mass-weighted throughout leaves one
symmetric projector and one metric.

**Fixed bond direction.** Re-imposed every step by a rigid rotation putting the
C–C vector on +x. A rigid rotation lies in the projector's null space, so the
BFGS history is unaffected; what it removes is the second-order drift off the
constraint that accumulates when finite steps are taken along tangent
directions (θ ≈ |dx|²/L, dying quadratically as the optimizer converges).

**Coupling vector.** `h_vector` = h₀/(E_q − E_p) rather than h₀ itself. h₀ = d^pq·ΔE
vanishes as the gap closes, which leaves the second branching direction
undefined exactly at convergence; the gap-divided form stays finite. A
previous-step u_h is carried forward as a fallback (as in Maeda–Morokuma's
updated branching plane).

**Convergence.** |ΔE| < 3.7e-06 Eh *and* |f_seam| < 5e-04, where f_seam is
computed with the *Cartesian* branching-plane projection to match the original
driver — the mass-weighted seam norm differs by ~4× at the same geometry
(7.28e-04 vs 1.76e-04 at the λ=0.4 MECP), so testing on it would silently move
the goalpost. A plateau stop is also available: f_seam_mw improving <5% over N
steps compared with the previous N, with the gap already inside tolerance.

**Validation.** On an anisotropic two-state linear-vibronic model with an exactly
known crossing point (12 internal dof, force-constant spread 10.7×), 73 steps
versus 9336 for the steepest descent it replaces — the latter given its best
step size out of eight at each start. On the real λ=0.4 deck, 19 steps versus the
published 1500, better on every measure.

Implementation: `src/mecp_optimizer.py` (numpy only, no quantum chemistry),
driver `src/scan_conical_qn.py`, test `src/validation/test_mecp_optimizer.py`.

## 8. Corrections to earlier claims in this analysis

- A "−1.028 eV geometry effect" between λ=0.1 and 0.2 was **wrong**; the
  geometries differ by ΔR_CC = −0.0043 Å and the true geometry effect is
  **0.018 eV**. The 1.03 eV was a branch difference.
- "`seam_tol = 5e-04` is unreachable / no (4,7) run ever met it" was **wrong** —
  four of five published geometries already met it (4.997e-04, 1.653e-04,
  1.824e-04, 4.979e-04); only λ=0.4 (7.278e-04) did not. λ=0.4 was always the
  only point in trouble.
- A claimed seam "floor" at ~7.3e-04 was **retracted** — f_seam_mw later fell to
  8.24e-05 on the same deck.
- "CAS(4,9) removes the multiplicity (spread 0.01 eV)" was **wrong** — it rested
  on two random seeds at one λ. Six seeds at four λ give 2–4 solutions with
  valence spreads up to 3.2 eV. §4 is rewritten accordingly.
- An "energy-trend" filter was used at one point to select among solutions.
  That is **circular** — it uses the shape of the answer being measured to pick
  the answer. It also fails to discriminate in (4,7), where all four solutions
  pass it with a 0.59 eV spread. Replaced by the density-overlap test of §4.3.
- At λ=0, two explanations offered for a 6.6× seam discrepancy (per-step
  re-alignment, then the projector) were both **wrong**; with no cavity the
  Hamiltonian is rotationally invariant, so neither can change an energy or a
  gradient norm. The actual cause is §5.

## 9. Open items

- The λ=0 MECP should be re-run with the coupling solve skipped (it cannot
  converge and is not needed with no cavity). Not yet done. Its energetic effect
  is ~0.07 meV, so this is bookkeeping rather than a correction.
- The (4,9) chain is not the global SA solution at λ=0.1 (a random scramble
  found one 0.95 mEh lower, same character, valence differing 0.012 eV); only
  two random seeds were tried there.
- (4,9) MECPs start from (4,7) geometries, so they find the nearest (4,9)
  intersection, not necessarily the global one. Whether the seam has more than
  one minimum is **untested** — it needs displaced or independently constructed
  starting guesses, and it is the open item most able to move a margin.
- The seed scan covers the FC leg only. The CI geometries were probed at λ=0.0
  and λ=0.1 (one solution each, three starts); λ=0.2–0.4 were probed from a
  canonical guess only.
- Six seeds per λ is a weak probe: the (4,7) FC case needed six to reveal four
  solutions, so a solution with a small basin can still be hiding.
- λ=0 runs ~3.5× slower per step than λ=0.4; attributable to the stalling
  coupling solve, but not profiled directly.
