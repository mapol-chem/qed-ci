# The photon-frame term in derivative couplings, and how to verify it numerically

Covers three things: the theory of `d_photon` including the nuclear dipole, the
method used to verify a derivative coupling by finite difference, and what each
check must return and why. Numbers quoted are measured, with the system named.

---

## 1. Two frames, and why they differ

The Pauli-Fierz Hamiltonian in the dipole gauge is transformed to the
coherent-state basis by

```
    U_CS = exp[ z (b† − b) ] ,        z = − ⟨d⟩ / √(2ω) ,
    ⟨d⟩  = λ·⟨μ⟩ = λ·⟨μ_e⟩ + λ·μ_N  ≡  ⟨d_e⟩ + d_N ,
    H_CS = U_CS H_PF U_CS† .
```

`U_CS` displaces the field to the equilibrium set by the molecular dipole, so
the photon vacuum in the new basis is the *displaced* vacuum. Two consequences
pull in opposite directions and both matter:

**Inside `H_CS` the nuclear dipole cancels.** The coupling and dipole
self-energy terms contain only the difference

```
    d̂ − ⟨d⟩ = (d̂_e + d_N) − (⟨d_e⟩ + d_N) = d̂_e − ⟨d_e⟩ ,
```

so the bilinear term becomes `−√(ω/2)(d̂_e − ⟨d_e⟩)(b† + b)` and the DSE
`½(d̂_e − ⟨d_e⟩)²`. Every *energy* is therefore independent of `d_N`, and a code
needs only the electronic expectation value to build the Hamiltonian. Ours
stores exactly that (`helper_cqed_rhf.py`: "⟨μ⟩ contains ONLY electronic
contributions").

**This makes `H_CS` convention-blind.** `d̂ − ⟨d⟩` is identical whether `z` was
built from `⟨d_e⟩` or from the full `⟨d⟩`, so the Hamiltonian — and hence the
stored variable — cannot tell you which convention is in force. Inferring the
convention from the stored `d_exp` is a mistake (one made during this work).

**But `z` is geometry dependent, and `U_CS` does not cancel anything.** The
derivative coupling is a property of the *states*, and the states in the two
frames are related by `U_CS`, which changes from geometry to geometry:

```
    d_lab^{mn} = ⟨Ψ^m_lab | ∂_R | Ψ^n_lab⟩
               = d_CS^{mn} + ⟨Ψ^m_CS | U_CS ∂_R U_CS† | Ψ^n_CS⟩
               = d_CS^{mn} + (∂z/∂R) ⟨Ψ^m_CS | (b† − b) | Ψ^n_CS⟩
               ≡ d_CS^{mn} + d_photon^{mn} .
```

Since `∂_R U_CS† = −(∂z/∂R)(b† − b) U_CS†` for the real displacement `z`, the
frame term is exactly

```
    d_photon^{mn}_{A,α} = (G^{mn} / √(2ω)) [ ∂⟨d_e⟩/∂R_{A,α} + λ_α Z_A ] ,     (1)

    G^{mn} = ⟨Ψ^m | (b† − b) | Ψ^n⟩ .
```

The nuclear term `λ_α Z_A` is there because `d_N` sits in `z`, and `d_photon`
sees `z` directly rather than the difference. It cancelled out of the
Hamiltonian *because* it is in the transformation; that is the same fact, not a
contradiction.

Properties worth noting:

* `G^{mn}` is **antisymmetric**, so `G^{mm} = 0` and this term is absent from
  every gradient. It appears only in derivative couplings.
* The nuclear part needs **no orbital response**. `κ·f` covers the electronic
  half only; `λ_α Z_A` is a geometry-independent constant per atom.
* The term is not small. On h2o_cav (λ = 0.05 along x, ω = 0.1213, weights
  5,2,1, pair 0–1) `d_CS = +0.0109` and `d_lab = −0.888`: the frame choice
  changes the coupling by more than the coupling itself.

## 2. The translational sum rule

For a **neutral** molecule `H_PF` is invariant under a rigid translation, so
every derivative coupling must satisfy

```
    Σ_A d_photon^{mn}_{A,α} = 0 .                                            (2)
```

This follows from (1) with no further input: under a rigid shift the electrons
follow the nuclei, so

```
    Σ_A ∂⟨d_e⟩/∂R_{A,α} = − λ_α N_e ,        Σ_A λ_α Z_A = + λ_α Σ_A Z_A ,
```

and for a neutral molecule `Σ_A Z_A = N_e`, so the two cancel exactly. For an
ion of total charge `Q` the sum does **not** vanish:

```
    Σ_A d_photon^{mn}_{A,α} = λ_α Q G^{mn} / √(2ω) .                         (3)
```

Physically a moving net charge *does* couple to the field: (3) is the
Coulomb-gauge `Q Â·P̂/M` coupling in its length-gauge disguise. This is the
sharpest available constraint on the term, because it fixes both the **sign**
and the **prefactor** of the nuclear half relative to the electronic one with no
freedom left — once the sum vanishes for a neutral molecule and equals (3) for
an ion, there is nothing left to choose.

## 3. Verifying a derivative coupling by finite difference

The analytic coupling is checked against a finite-difference overlap of the
wavefunctions themselves. Given two geometries `x` and `x ± h`:

```
    d^{mn} ≈ [ ⟨Ψ^m(x) | Ψ^n(x+h) ⟩ − ⟨Ψ^m(x) | Ψ^n(x−h) ⟩ ] / 2h
```

with the overlap built as: AO overlap across the two basis-set centres → MO
overlap `Cᵀ S C'` → determinant overlap factorised into α and β string
overlaps → contraction with the two CI vectors,

```
    ⟨Ψ^m|Ψ^n'⟩ = Σ c^m_{ipab} P_{pq} D_{aA} D_{bB} c^n'_{jqAB} .
```

Three requirements, each of which silently corrupts the result if missed:

**(a) Phase and root alignment.** The displaced states must be matched to the
reference by maximum overlap and phase-fixed, because a Davidson returns
eigenvectors with arbitrary sign and can swap near-degenerate roots between
geometries. A swap is the failure mode that makes a numerical coupling look
plausible but wrong.

**(b) The photon basis moves.** In the coherent-state basis the photon states
are themselves displaced, by `α = ⟨d⟩/√(2ω)`, and `α` depends on geometry. The
overlap of two displaced photon bases is a Cahill-Glauber displacement matrix

```
    ⟨m| D(γ) |n⟩ ,        γ = α(x±h) − α(x) ,
```

whose leading off-diagonal is `⟨0|D|1⟩ = −γ + O(γ³)`. It therefore enters at
**first** order: omitting it corrupts the coupling at leading order rather than
mildly. Including it yields `d_lab`; treating the photon states as
geometry-independent yields `d_CS`. Comparing the two legs against each other
measures `d_photon` with no analytic input at all.

**(c) `⟨d⟩` must be built identically everywhere.** `α` must use the full
`⟨d_e⟩ + d_N` at the reference *and* at every displaced geometry. If the nuclear
part is present at one and absent at another, `γ` acquires a constant offset
`γ₀ = d_N/√(2ω)`, and since `D(γ₀ ± δ) = D(γ₀) D(±δ)` the finite difference
returns

```
    ⟨Ψ^m| D(γ₀) |∂_x Ψ^n⟩ + γ′ ⟨Ψ^m| D(γ₀)(b† − b) |Ψ^n⟩ + …
```

instead of `d_CS + γ′G^{mn}`. The first term is the CS-frame coupling with a
displacement operator sandwiched in, perturbed by `O(γ₀ · d_CS)`. That
contribution is **not proportional to `G^{mn}`**, so it carries either sign and
hits weakly photon-coupled pairs proportionally hardest — a diagnostic
fingerprint, and the reason a residual can be large on a small-`G` pair while
another pair looks nearly fine.

The practical invariant: **`γ₊ + γ₋` must be ≈ 0 to O(h²)**. Print it.
(`QED_FD_PRINT_GAMMA=1`; measured −4.9e-08 once `⟨d⟩` was consistent.)

**(d) Richardson extrapolation.** `(4f(h/2) − f(h))/3` removes the O(h²) error
and takes plain-`h` agreement of ~1e-04 down to ~1e-07. Worth doing before
attributing a 1e-04 residual to physics.

**(e) Solution-branch hygiene.** Compare `1 − |S_ii|` at `+h` and `−h`. A large
asymmetry means the two displaced calculations converged to different solution
branches, which Richardson cannot repair — it will appear as a residual that
barely improves with `h`.

## 4. What each check must return, and why

| check | system | required value | why |
|---|---|---|---|
| frame term off | any | FD ≡ `d_CS` | photon states held fixed |
| frame term on | any | FD ≡ `d_lab` | displaced-basis overlap included |
| frame relation | any | `d_lab − d_CS = (∂z/∂R) G^{mn}` | pure theory; no analytic gradient used |
| sum rule, neutral | H₂O | `Σ_A d_photon = 0` | electrons follow; `Σ Z_A = N_e` |
| rigid translation, neutral | H₂O | `∂z/∂x = 0` | same statement, SCF only, no CI |
| rigid translation, ion | H₃O⁺, OH⁻ | `∂z/∂x = λ_α Q/√(2ω)` | fixes the SIGN of `d_N` vs `⟨d_e⟩` |

The last row is the one that cannot be faked. A sign error in `d_N` passes every
neutral test, because there everything cancels; it cannot produce **opposite**
answers of equal magnitude for a cation and an anion.

### Measured

Rigid translation, SCF only (`validation/dphoton_translation_check.py`),
λ = 0.05 along x, ω = 0.1213, 6-31G, h = 1e-3 Å:

| system | Q | `∂z/∂x` measured | required | residual |
|---|---|---|---|---|
| H₂O | 0 | −7.46e-14 | 0 | 7.5e-14 |
| H₃O⁺ | +1 | +1.01525106e-01 | +1.01525106e-01 | 3.9e-13 |
| OH⁻ | −1 | −1.01525106e-01 | −1.01525106e-01 | 1.1e-10 |
| Li⁺ | +1 | +1.01525106e-01 | +1.01525106e-01 | 7.5e-11 |

Control with the nuclear part omitted: worst residual **1.12**, all four fail.
Note the electronic-only `∂z/∂x` is **−1.015 for every molecular case** — a
rigid translation moves the electrons by the same amount regardless of the
molecule — so dropping `d_N` does not perturb the displacement slightly, it
replaces it with something of order unity.

Analytic sum rule, LiH, λ along z: frame term off 5.6e-17; on **without** the
nuclear half **−3.64e-02** (violated); on with it 1.1e-16.

Analytic vs finite difference, h2o_cav weights 5,2,1, lab frame: worst pair
1.49e-05 against a 1e-03 tolerance, with the small-`G` pair best at 1.56e-07.

## 5. Practical guidance

1. Build `⟨d⟩ = ⟨d_e⟩ + d_N` in **one** place and use it everywhere. Three
   separate sites needed it here (the displaced-point builder, a separately
   written reference builder, and `∂z/∂R` in the frame decomposition), and
   patching them one failing test at a time is how a subtle inconsistency
   survives. Grep every use.
2. Trust an **invariance constraint** over an agreement between two
   implementations. Agreement is not validation when both can share an
   omission — the analytic and FD couplings here agreed to 3.67e-07 while both
   were missing the same term.
3. Check `γ₊ + γ₋ ≈ 0` before interpreting any residual.
4. Use the **ion** test whenever a sign is in question. Neutral cancellations
   hide sign errors by construction.
5. Do not gate gradient code at a conical intersection: individual roots are
   noise-amplified there (repeat runs move each root 2–3e-08 in opposite
   directions with their sum stable to 2–4e-12). Gate on the state-averaged
   energy, at a separated geometry, or with a unit test that removes the SCF.

## 6. Status and open questions

Settled: the form (1), the sum rules (2)–(3), the prefactor (fixed by the sum
rule), and the sign (fixed by the ion pair).

Open:
* `QED_PHOTON_FRAME` is **default OFF**, so the analytic coupling reported by
  default is `d_CS`, the working-frame one. Which frame should be reported is a
  physics decision, not a numerical one, and it is unresolved.
* The ion sum rule (3) has been verified for `∂z/∂x`; the full
  `Σ_A d_photon = λ_α Q G^{mn}/√(2ω)` including `G^{mn}` has not been checked on
  an ion with well-converged roots.
* Whether the displaced-basis convention should be shared between the two states
  of a pair (`--photon-basis-derivative` is off by default) is a separate
  question from the one resolved here.
