# Numerical derivative couplings by finite-difference wavefunction overlap

Method note for `numerical_nac_check.py` and `tests/test_numerical_nac.py`.
Written 2026-09-08.

**Results at a glance** (H2O, CAS(6,5)/6-31G, C1 geometry, SA-3):

| what | result |
|---|---|
| cavity off, analytic vs FD overlap | 1.6e-05 (pair 0-1), 1.9e-04 (pair 1-2, whose \|NAC\| ~ 2) |
| the same analytic value vs Molpro | ~5e-07 — so analytic, Molpro and FD all agree |
| finite-difference convergence | error ∝ h², ratios 4.00 / 4.00 over h = 4e-3 → 1e-3 Å |
| cavity on (ω = 0.121, λ = 0.05, 1 photon), unequal weights 5,2,1 | 1.48e-05, h² ratios 4.00 / 3.99 |
| QED-HF orbital response (Σ κ_ai f_ai) dropped | 1.6e-02 — **1100× worse, so the term is required** |
| photon-basis-derivative factor added | ~8.8e-01 — **breaks agreement; off by default (§7)** |

## 1. Why this is not a numerical gradient

A gradient can be checked by differencing energies. A derivative coupling
cannot: it is a property of the *wavefunctions*, not of the spectrum,

$$d_{ij}(x) \;=\; \langle \Psi_i(x) \,|\, \partial_x \Psi_j(x)\rangle .$$

So a numerical reference has to evaluate wavefunction overlaps between two
different geometries, which is the whole of the difficulty. The estimator is
an ordinary central difference of that overlap:

$$d_{ij} \;\approx\; \frac{\langle\Psi_i(x)|\Psi_j(x+h)\rangle \;-\;
                            \langle\Psi_i(x)|\Psi_j(x-h)\rangle}{2h}$$

with the bra held fixed at the reference geometry. Both displaced kets are
expressed in *their own* orbital and determinant bases; nothing is assumed to
be common between the three calculations except the active-space definition.

## 2. Evaluating the cross-geometry overlap

Four layers, outermost first.

**(a) AO overlap across two basis-set centres.** The basis functions move with
the nuclei, so this is a genuine two-centre integral between two different
basis sets. psi4 provides it directly:

```python
S_ao = MintsHelper(bs_x).ao_overlap(bs_x, bs_xp)     # (nbf, nbf), NOT symmetric
```

This was the one piece that decided whether the method was practical, so it is
worth verifying rather than assuming: `ao_overlap(b1, b2)` differs from
`ao_overlap(b1, b1)` (5.4e-05 for a 1e-4 Å displacement), and the same-geometry
call has unit diagonal.

**(b) MO overlap.** With the converged CASSCF orbitals at each geometry,

$$S^{\text{MO}} = C(x)^{\dagger}\, S^{\text{AO}}(x,x')\, C(x') .$$

Use `opt_C`, the orbitals *after* orbital optimisation — not `C`.

**(c) Determinant overlap.** For two Slater determinants built from different
orbital sets, the overlap is the determinant of the occupied block of
`S_MO`, and alpha and beta factorise:

$$\langle D_I(x) | D_J(x')\rangle
  = \det S^{\text{MO}}[\alpha_I, \alpha_J] \cdot
    \det S^{\text{MO}}[\beta_I, \beta_J] .$$

The occupied lists include the inactive core, which also rotates between
geometries — dropping it is wrong.

**(d) CI overlap, without ever forming N_det² determinants.** Because the alpha
and beta strings are independent, the same string-overlap matrix
`D[a,b] = det S_MO[occ_a, occ_b]` serves both spins, and the CAS overlap is one
contraction:

$$\langle\Psi_i|\Psi_j\rangle
  = \sum_{ab,AB} c^{i}_{ab}\, D_{aA}\, D_{bB}\, c'^{j}_{AB}$$

```python
np.einsum("iab,aA,bB,jAB->ij", c_ref, D, D, c_dis, optimize=True)
```

with the CI vector reshaped to `(n_alpha, n_beta)`. For H2O CAS(6,5) this is
**100 small determinants rather than 10 000**, and the cost is negligible next
to the CASSCF runs themselves.

With a cavity the CI vector carries a photon index as well. The layout is
**photon-major** — `helper_PFCI` reshapes a CI vector as `(np1, num_det)` at
`helper_PFCI.py:5098` — so `num_det` then splits into `(n_alpha, n_beta)` and
the contraction gains one index:

```python
np.einsum("ipab,pq,aA,bB,jqAB->ij", c_ref, P, D, D, c_dis, optimize=True)
```

where `P` is the photon overlap. **`P` is the identity by default**, which is
the substantive question §7 is about — not a simplification made for
convenience.

*Determinant ordering.* The direct (Davidson) path does not populate
`self.CASdets` — the string tables live in the C backend — but the ordering is
reproducible from `generateCASCIDeterminants`: `for alpha in combinations(...):
for beta in combinations(...)`, with the inactive core prepended to each
string. Hence index `= i_alpha * n_beta + i_beta`, and the reshape above is
valid. Confirmed by shape: CAS(6,5) gives `eigenvecs` of shape `(roots, 100)`
with `100 = 10 x 10`.

## 3. Two corrections that are not optional

Both of these turn the result into noise if skipped, and neither is detectable
by eye in the final number.

**Phase.** A CI eigenvector's overall sign is arbitrary and is chosen
independently by the eigensolver at every geometry. Each displaced state is
therefore phase-aligned to the reference by requiring
$\langle\Psi_k(x)|\Psi_k(x')\rangle > 0$ before the difference is taken.

**State ordering.** States are matched to the reference by **maximum overlap**,
not by energy order. This is the correct identification in general, and it is
what lets the method work where the energy ordering *swaps* under displacement
— the failure mode that makes numerical **per-state gradients** unreliable near
a conical intersection (see `numerical_gradient_check.py`, which mitigates the
same problem differently, by offering a state-average leg that is invariant
under relabelling). A swap that was corrected is reported rather than hidden,
because a swap that happens at one displaced point and not its partner is a
sign the step size is too large.

## 4. Which coupling this produces

Because the AO basis moves with the nuclei, the overlap difference contains the
basis-set-derivative contribution, so this returns the **full** coupling —
*with* the CSF term, not the ETF form. In the identifications established
elsewhere in this tree:

| this method | Molpro | pyscf `nac_method` | ours |
|---|---|---|---|
| FD overlap | `SA-MC NACME` | `kernel(state=(i,j))` | `NAC (with CSF)` from `compute_grad(i, j, deriv_type="full")` |

A coupling also carries the arbitrary *relative* phase of the two CI vectors,
so it compares up to an overall sign per state pair. A global flip is
convention, not disagreement.

## 5. Validation, and why the scaling test is the one that matters

Three checks, in increasing order of how much they prove.

1. **It reproduces the analytic coupling.** At h = 1e-3 Å, worst residual
   1.6e-05 on pair 0-1 and 1.9e-04 on pair 1-2 (whose coupling is of order 2).
2. **The analytic value is independently Molpro's**, to ~5e-07 — so analytic,
   Molpro and finite-difference all agree.
3. **The error scales as h².** This is the real evidence:

   | h (Å) | worst \|analytic − numerical\| | ratio |
   |---|---|---|
   | 4e-3 | 2.590e-04 | |
   | 2e-3 | 6.479e-05 | 4.00 |
   | 1e-3 | 1.619e-05 | 4.00 |

   Agreement at a *single* step size can be luck, or two errors cancelling.
   Central differences carry an O(h²) truncation error, so observing exactly
   4× per halving shows the numerical estimate is *converging to* the analytic
   value rather than coincidentally near it. It is a test in its own right
   (`test_finite_difference_error_scales_as_delta_squared`).

### 5b. A separate, positive check: the QED-HF orbital response

Agreement between two sides shows they are *consistent*; it does not show that
a particular term inside one of them is carrying its weight. The
$\sum_{ai}\kappa_{ai}f_{ai}$ term — the CP-HF response of the QED-HF orbitals,
with its
$\mathrm{tr}[\mathbf{X}^{\text{HF}}(\mathbf{C}^{\text{HF}}\mathbf{S}
\mathbf{C}^{\text{HF}\dagger}-\mathbf{1})]$ constraint — is made falsifiable
by `QED_NO_HF_RESPONSE=1`, which discards the solved CP-HF response and leaves
everything else intact.

This works as a test because the finite-difference reference contains the
CQED-HF response **by construction**: every displaced point re-converges
CQED-HF from scratch. So if the analytic side dropped that response, the two
must diverge. Measured (h2o_cav, weights 5,2,1, h = 1e-3 Å):

| | worst \|analytic − numerical\| |
|---|---|
| $\kappa\!\cdot\!f$ included (shipped) | 1.476e-05 |
| $\kappa\!\cdot\!f$ dropped | 1.624e-02 — **1100× worse** |

The term is therefore necessary and correct, not merely present. The CP-HF
solve only runs when `lambda_vector` is nonzero, so this is a cavity-only
statement.

## 6. Error budget

Two competing terms set the usable window for h:

* **truncation**, O(h²), which wants h small — the table above;
* **convergence noise**, amplified by 1/(2h), which wants h large. A central
  difference at h = 1e-3 Å ≈ 1.9e-3 bohr amplifies noise by ~260×; the
  numerical *gradient* tool, at h = 1e-4 Å, amplifies by ~2600× and is
  correspondingly more sensitive to how tightly the CASSCF is converged.

This is why both numerical tools run with the microiteration convergence
backstops lowered (`QED_MICRO_GRAD_FLOOR`, `QED_MICRO_ENERGY_FLOOR`,
`QED_MACRO_ENERGY_CONV`). At the shipped defaults the per-state numerical
*gradient* comparison degrades from ~1e-07 to ~5e-06, and to 3.6e-05 for the
highest state — an order of magnitude, on identical physics.

If the observed error stops falling as h², one of the two terms above has taken
over; that is the diagnostic, not a reason to widen a tolerance.

## 7. Cavity on: the photon overlap, and a measured answer

The coherent-state photon states are displaced number states whose displacement
is built from $\langle d_{\text{HF}}\rangle$ at **that calculation's** reference
geometry, so two geometries do not share a photon basis. Completing the square
in $\omega b^\dagger b - \sqrt{\omega/2}\,\langle d\rangle (b + b^\dagger)$ puts
the shift at $\alpha = \langle d_{\text{HF}}\rangle/\sqrt{2\omega}$, and only the
difference $\gamma = \alpha(x') - \alpha(x)$ enters the overlap, through
displacement-operator matrix elements between Fock states:

$$\langle m|D(\gamma)|n\rangle = \sqrt{\tfrac{n!}{m!}}\,\gamma^{m-n}
  e^{-\gamma^2/2} L_n^{m-n}(\gamma^2), \quad m \ge n$$

(implemented in `displacement_matrix`, verified **exactly** — 1.1e-16 — against
`expm(gamma*(a.T - a))`).

**This term is not small, and it must not be added.** Both halves of that
sentence are measurements.

*Not small:* for `h2o_cav` at h = 1e-3 Å, `d_exp` moves −0.007361 → −0.008198,
so $\gamma = -1.70\times10^{-3}$ and $\gamma/(2h) = -0.45$. Since the leading
off-diagonal element is $\langle 0|D|1\rangle = -\gamma + O(\gamma^3)$, the factor
would contribute at **first** order — O(0.5) on a coupling of order 1.

*Must not be added:* measured against the analytic coupling (h2o_cav, weights
5,2,1, h = 1e-3 Å):

| photon treatment in the overlap | worst \|analytic − numerical\| |
|---|---|
| geometry-independent (default) | **1.48e-05** |
| + displaced-basis overlap, sign +1 | 8.77e-01 |
| + displaced-basis overlap, sign −1 | 8.99e-01 |

and the agreeing leg converges as h²: 2.356e-04 / 5.896e-05 / 1.476e-05 at
h = 4e-3 / 2e-3 / 1e-3 Å, ratios 4.00 and 3.99 — the same quality as
cavity-off. So the analytic implementation and this overlap share a convention,
and the numerical cavity-on coupling is a real check rather than a tautology.

**What the extra term would be, physically.** The wavefunction has *two*
geometry-dependent factors,

$$|\Psi(x)\rangle = \sum_{I,n} c_{I,n}(x)\,|D_I(x)\rangle \otimes |n\rangle_{\alpha(x)},$$

the determinants (MOs move — handled by §2) and the photon basis (the origin
moves). Only the second is at issue.

**A double-counting reading against $\kappa\!\cdot\!f$ does not survive the
magnitudes.** It is tempting — the origin's motion is driven by
$\langle d_{\text{HF}}\rangle$, a functional of the CQED-HF orbitals whose
response the Lagrangian already carries — and it is structurally the same shape
as the `Y_z` trap in `nuclear_grad.py`. But the two terms are not comparable in
size:

| term | worth, in the same comparison |
|---|---|
| dropping $\kappa\!\cdot\!f$ entirely (`QED_NO_HF_RESPONSE=1`) | 1.6e-02 |
| adding the photon-basis factor | ~8.8e-01 |

The photon factor is ~55x the *entire* $\kappa\!\cdot\!f$ contribution. A term
that duplicated $\kappa\!\cdot\!f$ would be of its order. So this is very
probably not double counting, and an earlier draft of this note saying it was
is corrected here.

**The likelier reading is a FRAME distinction.** The code works entirely in the
coherent-state-transformed frame: its Hamiltonian is $U^\dagger H U$ and its
wavefunction is $U^\dagger\Psi$. Taking overlaps of those transformed objects
against undisplaced Fock states yields the **working-frame** coupling — and the
analytic coupling, derived from the same transformed Hamiltonian, is the
working-frame one as well. That is why they agree, and why the cavity-on
comparison is a genuine check rather than a tautology. Including the
displaced-basis factor moves the numerical estimate toward the **lab-frame**
coupling, a different quantity.

On that reading neither is wrong, but they are not interchangeable: a dynamics
propagation must use the coupling in the same frame as the Hamiltonian driving
it. Which frame is wanted is a question about the intended use, not something
this comparison can settle.

**What is and is not established.** Established: the two sides share a
convention, to 1.48e-05 with clean h^2 convergence, and $\kappa\!\cdot\!f$ is
required within it (1100x, §5 of `test_numerical_nac.py`). NOT established:
which convention the physics calls for. Agreement cannot distinguish "both are
the working-frame coupling, correctly" from "both omit the same lab-frame
term". `--photon-basis-derivative` exists so the alternative can be examined
without rewriting anything.

## 8. Usage

```sh
python validation/numerical_nac_check.py --system h2o_c1 --weights 5,2,1 \
       --components 2 --delta 1e-3
```

`--components N` restricts the run to the first N of the 3N Cartesian
components, which is the runtime dial; `--weights` takes one weight per root.
Systems are defined at the top of the file.

The suite test (`tests/test_numerical_nac.py`, 3 tests, 24 s) runs equal and
unequal weights plus the h² scaling check. Null-tested: under
`QED_PROJ_ALL=1`, which restores the all-states CI projection, the
**unequal-weight** leg fails and the equal-weight leg passes — so this catches
the projection defect without reference to any external code, which matters
because the Molpro comparisons expire with the licence on 2026/09/20.

## 9. The open question, stated plainly

Everything above is either derived or measured except one thing, and it is the
thing worth arguing about.

**Established.** The analytic coupling and this finite-difference reference
agree — cavity off and cavity on, equal and unequal weights — to ~1e-05 with
clean h² convergence, and cavity-off they both agree with Molpro to ~5e-07.
Within that shared convention the QED-HF orbital response is required (1100×).

**Not established.** Whether that shared convention is the coupling the physics
calls for. The code works in the coherent-state-transformed frame; both sides
therefore produce what is naturally called the *working-frame* coupling.
Including the displaced-basis photon factor would move the numerical estimate
toward a *lab-frame* coupling — a different quantity, larger by O(0.5) here.
Agreement between two implementations that share a frame cannot distinguish

  * both are the working-frame coupling, correctly, and that is what is
    wanted; from
  * both omit the same lab-frame term.

Deciding that is a question about the intended use — in particular, a dynamics
propagation must use the coupling in the same frame as the Hamiltonian that
drives it — and not something a consistency check can settle.
`--photon-basis-derivative` switches the alternative on without rewriting
anything, so the comparison is one command away if it is wanted.
