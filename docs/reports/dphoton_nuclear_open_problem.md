# Open problem: the prefactor of the nuclear half of d_photon

Question for Fable. The physics of the nuclear term is settled (and I was wrong
twice on the way); what is not settled is the prefactor of my implementation,
which disagrees with a finite-difference reference by a factor of ~56 with the
opposite sign.

---

## 1. Settled: the term belongs, and the code cannot tell us the convention

`U_CS = exp(z(b† − b))` with `z = −⟨d⟩/√(2ω)` and `⟨d⟩ = ⟨d_e⟩ + d_N`.

Inside `H_CS` the nuclear dipole cancels, because the coupling and DSE contain
only the difference

```
    d̂ − ⟨d⟩ = (d̂_e + d_N) − (⟨d_e⟩ + d_N) = d̂_e − ⟨d_e⟩ ,
```

so the Hamiltonian is **identical under either choice of z** and cannot be used
to infer the convention. Our code stores `d_exp = d_exp_el` (helper_cqed_rhf.py
:413, explicitly "ONLY electronic contributions") and never constructs `U_CS`
explicitly at all — so there is no ground truth about the displacement anywhere
in the code. It is a choice, and the physically correct choice is the full ⟨d⟩.

**Two of my earlier claims are therefore withdrawn:** that the code "uses an
electronic-only z" (the stored variable cannot distinguish the conventions), and
that the good agreement without the term validated its absence (two calculations
sharing one omission).

## 2. Settled: the term is missing from both sides symmetrically

`d_photon = U_CS ∂_R U_CS†` sees `z` directly, not the difference, so

```
    d_photon^{mn}_{A,α} = (G^{mn}/√(2ω)) [ ∂⟨d_e⟩/∂R_{A,α} + λ_α Z_A ]
```

with `G^{mn} = ⟨Ψ^m|(b† − b)|Ψ^n⟩`, antisymmetric, so `G^{mm} = 0` and the term
is absent from every gradient — it appears only in derivative couplings.

A/B on the lab-frame coupling (h2o_cav, weights 5,2,1, states 1–2), with the
nuclear piece switched independently on the analytic side and in the
finite-difference displacement:

| analytic nuclear | FD displacement | discrepancy |
|---|---|---|
| on | off | 8.122e-01 |
| off | on | 8.123e-01 |
| on | on | 5.646e-03 |
| off | off | 3.67e-07 |

`on/off ≈ off/on` to four digits — unambiguously the *same* term appearing on one
side or the other. That is the symmetry that confirms the term is real.

**Independent confirmation from a sum rule.** For a NEUTRAL molecule `H_PF` is
translation invariant, so `Σ_A h^{mn}_{A,α} = 0` for every state pair. Measured
on LiH, λ along z:

| frame term | nuclear piece | Σ_A h (z component) |
|---|---|---|
| off | — | 5.6e-17 |
| on | off | **−3.64e-02  (violated)** |
| on | on | 1.1e-16 |

So the nuclear piece is required for translational invariance. (For an ion the
sum should be `λ_α Q G^{mn}/√(2ω)`; not yet tested.)

## 3. THE PROBLEM: the two sides respond to it completely differently

With the nuclear piece added to **both** sides the discrepancy falls 144× (0.812
→ 5.65e-03) but does not close — it is still ~2800× above the 2e-06 tolerance
and far above the ~1e-04 plain-h finite-difference error. Comparing how each side
responded:

```
analytic:  0.0025272598  ->  0.0026264089     change  +9.91e-05
FD:       +0.0025268930  ->  -0.0030189716    change  -5.55e-03   (sign flip)
```

**Factor ~56 in magnitude and opposite in sign.** Not a clean prefactor slip;
the two routes are structurally different.

### How each side is built

**Analytic** (`nuclear_grad.py`). A single scale factor

```
    photon_frame_scale(m,n) = (E_n − E_m) · G^{mn} / √(2ω)
```

is added to `scale_factor` at two sites — `Y_hf = 4 · scale_factor · d_hf`
(build_Y_hf) and `one_rdm_pe_eff_ao += scale_factor · one_rdm_hf` — so the
ELECTRONIC half arrives through the orbital response and the density contracted
with the AO dipole derivatives `d_derivs`. The NUCLEAR half I added directly:

```
    photon_nuclear_gradient[3A+α] = photon_frame_scale(m,n) · Z_A · λ_α
```

added straight into `total_gradient`. The reported coupling is then `h₀/ΔE`.

**Finite difference** (`validation/numerical_nac_check.py`). The displaced
photon-basis overlap uses the Cahill–Glauber displacement matrix

```
    α_ref = ⟨d⟩_ref / √(2ω) ,   α_dis = ⟨d⟩_dis / √(2ω)
    factor = D(α_dis − α_ref) ,   ⟨m|D(γ)|n⟩ with ⟨0|D|1⟩ = −γ + O(γ³)
```

i.e. the displacement enters at FIRST order. Adding `d_N` changes `α_dis − α_ref`
by `λ_α Z_A h / √(2ω)` for a displacement h of atom A along α.

### Why the FD response is large, and plausibly right

For a single-atom displacement `Δd_N = λ_α Z_A h` is comparable in magnitude to
`Δ⟨d_e⟩` and opposite in sign — the same near-cancellation the translational sum
rule expresses. (Measured in the checker's own docstring: `d_exp` goes
−0.007361 → −0.008198 over h = 1e-3 Å.) So adding `d_N` largely cancels the
electronic displacement and can flip the sign of `α_dis − α_ref`, which is
exactly what the FD did. The analytic side moved by almost nothing in
comparison.

### The question

Matching the two requires differentiating the displacement-matrix element
`⟨m|D(α)|n⟩` with respect to `R_{A,α}` to first order and identifying the
coefficient that multiplies `λ_α Z_A`, then comparing it against
`photon_frame_scale · Z_A · λ_α` as coded. Specifically:

1. Is `(E_n − E_m) G^{mn}/√(2ω)` the correct prefactor for the nuclear half, or
   does the nuclear piece enter with a different factor from the electronic one?
   The electronic half reaches the gradient through the orbital response and the
   density (so it is multiplied by whatever `Y_hf`'s factor of 4 and the
   `one_rdm_pe_eff_ao` contraction supply), while the nuclear half is added
   directly — the factor-of-4 asymmetry between the two sites is a candidate.
2. Should the nuclear half be added to `h₀` (which is then divided by ΔE), or
   directly to the coupling? The analytic term carries an explicit `(E_n − E_m)`
   and the reported coupling divides by ΔE, so there is a risk of multiplying
   and dividing by the same gap — or of failing to.
3. What sets the sign? `G^{mn}` is antisymmetric and the FD carries a
   `sign`/phase-alignment convention; the observed sign flip may be a
   convention mismatch rather than an error in either.

The FD gives a direct numerical target: the nuclear piece must change the
1–2 coupling by **−5.55e-03**, from +0.0025268930 to −0.0030189716.

## 4. Status

| item | status |
|---|---|
| nuclear dipole is in `U_CS`, not in `H_CS` | settled (the difference cancels it) |
| the code cannot reveal the convention | settled (`H_CS` is convention-blind; `U_CS` never built) |
| the term is real and missing from both sides | settled (on/off ≈ off/on to 4 digits; sum rule) |
| translational sum rule requires it | measured: −3.64e-02 without, 1.1e-16 with |
| **the prefactor of my implementation** | **WRONG — 56× too small, opposite sign** |
| whether the FD's `d_nuc` insertion is itself correct | plausible but unverified |

Knobs, both default OFF pending resolution: `QED_DPHOTON_NUC` (analytic),
`QED_FD_DNUC` (finite-difference displacement; default 1 in the checker, set 0
to restore the old form).

Without the term the suite is 169 passed / 7 failed (4 pre-existing and
documented, 3 lab-frame tests that this term flips); with it, those 3 fail.
