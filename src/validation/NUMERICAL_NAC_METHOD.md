# Numerical derivative couplings by finite-difference wavefunction overlap

Method note for `numerical_nac_check.py` and `tests/test_numerical_nac.py`.
Written 2026-09-08.

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

**Why, most likely — and this is the developer's call, not established here.**
The motion of the coherent-state origin is plausibly already carried by the
QED-HF orbital response in the Lagrangian: the $\sum_{ai}\kappa_{ai}f_{ai}$ term
with its $\mathrm{tr}[\mathbf{X}^{\text{HF}}(\mathbf{C}^{\text{HF}}\mathbf{S}
\mathbf{C}^{\text{HF}\dagger}-\mathbf{1})]$ constraint. If so, a separate
photon-basis-derivative term double counts it. That is structurally the *same
trap* as `Y_z` in `nuclear_grad.py` — a legitimate Lagrangian object that must
nevertheless not be added on top of a contribution already present, and which
also announced itself as a large, clean, wrong contribution rather than a
subtle one.

What this does **not** settle: whether the shared convention is the physically
complete derivative coupling. The agreement shows the two sides are consistent;
it cannot by itself distinguish "both correct" from "both missing the same
term". Deciding that is a question about the Lagrangian, and
`--photon-basis-derivative` exists so the alternative can be re-examined
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
