# The two CI projections in `nuclear_grad.py`: what they are and why they differ

Written 2026-09-08. Companion to `NAC_two_projectors.md` (which carries the
measurement tables). This file is the *explanation*, and it also records two
wrong explanations I gave along the way, because both were plausible and someone
will reach for them again.

---

## 0. The problem in one paragraph

`nuclear_grad.py` applies a projection to CI-space vectors in three places. Two
are inside `solve3` (the z-vector linear solve): `my_preconditioner_wrapper`
(:1511) projects the trial vector, `my_matvec_wrapper` (:1536) projects sigma.
The third is in `build_total_gradient` (:751), on `Y_z`, the CI block of the
solve's right-hand side. Historically all three used `project_out_all`, which
orthogonalises a vector against **every** state-averaged CI vector. The
alternative, `self.projection` (:867), removes only each state's **own**
component (`z^(n)·c^(n) = 0`). The question is which projection belongs where.
The answer is **different in the two places**, which is why one knob could not
express it: `QED_PROJ_ALL=0` (minimal) for the solve, `QED_PROJ_ALL_YZ=1`
(project_out_all) for `Y_z`.

---

## 1. "Gauge", without the word

There is one concrete fact: **`(H - E_n)` annihilates `c^(n)`.** So the CI block
of the response matrix `A` has `c^(n)` in its null space -- `A` is *singular*
along that direction. Two consequences:

* nothing in the equations determines `z^(n)·c^(n)`; every value solves them;
* if that direction is left in the solver's search space, an iterative method is
  being asked to solve a singular system, and it wanders along the null vector.

So the `z^(n)·c^(n)` component is removed for **solvability**, not physics. It is
weight-independent, and *both* projectors already do it. It is never the issue.

---

## 2. Three sectors, three separate equations

Decompose any CI-space vector for state `n` into
  (i) along `c^(n)`;  (ii) along `c^(m)`, m != n;  (iii) orthogonal to the CI space.

Project the CI stationarity condition onto each. The three projections are
mutually orthogonal and give three equations that do not mix:

| project onto | resulting equation | determines | how |
|---|---|---|---|
| `c^(n)` | the `z` terms drop, since `(H-E_n)c^(n)=0` | the multiplier `x_n` | trivially |
| `c^(m)`, m != n | `(E_m-E_n) t_mn + 2<c^(m)|H~|c^(n)> = 0` | `t_mn = c^(m)·z^(n)` | **closed form** |
| orthogonal complement | the actual linear system | the rest of `z` | **iterative solve** |

The middle row is the one that matters: `t_mn` is given by a *formula*. It never
needed the iterative solver.

---

## 3. Which sector is `Y_z` in? Sector (ii), by construction

`nuclear_grad.py:744`:

```python
Y_z[state1] = -(E[state2] - Enuc)/w[state1] * eigenvecs[state2]
```

The `state1` row is proportional to `c^(state2)`. The CI vectors are orthonormal,
so `c^(state1)·Y_z[state1] = 0` **exactly**. `Y_z` therefore lies entirely in
sector (ii) -- not approximately, by construction. `project_out_all` removes
sector (ii), so it removes all of `Y_z`; the minimal projection removes only
sector (i), which `Y_z` does not have, so it leaves `Y_z` untouched.

For a *gradient* (`state1 == state2`) the `if` above never fires, `Y_z` stays
zero, and both projectors are no-ops. **That is why this hid for so long.**

---

## 4. What keeping `Y_z` actually does: an exact double count

MEASURED, H2O CAS(6,5)/6-31G at a C1 geometry, equal weights. Two independent
tests, both to ~1e-08 on all three state pairs:

1. **ETF branch** (`deriv_type=None`, no CSF/transition-density terms added):
   keeping `Y_z` gives **exactly +-2 x** the projected result. (The sign is the
   arbitrary relative CI phase, which differs run to run; ratios came out
   -2.0000 for two pairs and +2.0000 for the third.)
2. **Full branch** (`deriv_type="full"`): `(kept - projected)` equals the **ETF
   core** -- i.e. keeping `Y_z` adds exactly one extra copy of it.

So the derivative coupling's CI-space contribution is *already* produced in full
by the rest of the machinery, and `Y_z` re-injects the same thing through the
z-vector. **`project_out_all` on `Y_z` is not deleting a source term; it is
removing a second copy of one.**

Note what is NOT duplicated: the CSF/overlap term. Test 2 shows the extra copy is
the ETF core alone. See section 7.

### The sign experiment: equal and opposite, confirmed

`QED_YZ_SIGN` (diagnostic, default +1) flips the sign of the CI source, so the
question "is this just a sign error?" can be answered by measurement rather than
argument.  H2O C1, equal weights, ETF branch, NAC*dE row 0 x:

| configuration | value |
|---|---|
| `Y_z` projected (shipped) | **0.1477103158**  <- correct |
| `Y_z` kept, sign +1 | -0.2954206358  = 2x |
| `Y_z` kept, sign -1 | **-0.0000000030**  ~ 0 |

**Flipping the sign gives exactly zero, to 3e-09.**  So the explicit route and
the z route are equal and opposite in MAGNITUDE -- not merely similar.  The
code's sign makes them add; the other makes them cancel.

**It is not a sign fix.**  The correct answer is C = 0.1477, and neither kept
sign produces it: `+` gives 2C, `-` gives 0.  Only removing the term gives C.
Exactly one of the two routes may be present, and no choice of sign rescues
keeping both.

### An open question this raises -- in the Lagrangian, not the code

Following the theory section, `y~^(n) = (1/W_n) d(c^m+ H c^n)/dc^(n)
= (E_m/W_n) c^(m)`, so the RHS source is `-y~`.  The code's
`Y_z[n] = -(E_m - E_nuc)/W_n c^(m)` matches that; the sign looks right.

But then `dL/dx` should be the total derivative of `c^m+ H c^n`, which is
identically zero at any solution -- and the configuration that returns zero is
the FLIPPED sign, not the shipped one.  One of these must hold, and this
document does not settle which:

* the objective is meant to be differentiated with `c^m` and `c^n` HELD FIXED,
  yielding `h^mn` rather than zero -- in which case including `y~` as a CI
  source is what double counts; or
* the sign relation between `y~` and the code's `Y_z` is not as traced above.

That is a question about the construction, not the implementation.

**Not in doubt:** the shipped configuration reproduces Molpro to ~5e-07, pyscf
to ~1e-06, and the finite-difference wavefunction overlap to 1.6e-05 with clean
h^2 convergence.  The answer is right; the bookkeeping account of why is what
remains unsettled.

### Explanations of this that were WRONG, in order


* **"It annihilates the source term, so the shipped code cannot compute a
  coupling at all."** Backwards. It removes a duplicate. I reached this from a
  correct numerical demo (`||project_out_all(Y_z)|| = 1.03e-15`) plus a wrong
  interpretation, and from having a single knob so the A/B could not separate
  the three sites.
* **"It is not a clean double count -- the ratios are 1.8-6.6, so the spurious
  source must propagate through the orbital-CI coupling."** Also wrong. I
  measured the *ratio* in the full branch, where the additive CSF term disguises
  the doubling: `(2*core + extra)/(core + extra)` varies per component with the
  size of `extra`. Taking the *difference* instead of the ratio, and testing the
  ETF branch where `extra = 0`, shows an exact factor of two.
* **"The equal-and-opposite account was wrong -- it conflated the z-vector
  multiplier with the CI response."**  An OVER-correction, retracted in turn by
  the sign experiment above: flipping the sign cancels the two contributions to
  3e-09, so they are equal and opposite in magnitude exactly as that account
  said.  Three revisions of one explanation is fair warning that the bookkeeping
  here is easy to get wrong by argument alone.  **The measurements never
  moved.**

**Method lesson: when a quantity is a sum of an affected and an unaffected part,
compare DIFFERENCES, not ratios, and find the branch where the unaffected part
is switched off.**

---

## 4b. WHY it is a duplicate: `H^{mn}` is identically zero

Section 4 proves the duplication numerically. This is the reason, and it is
checkable on paper. (Due to the developer, 2026-09-08 -- it is a better
explanation than the "already added by the transition-density terms" story I
gave first, which was never more than an inference about code paths.)

Start from the CI stationarity condition of the Lagrangian,

    y~^(n) + (H - E_n) z^(n) - x_n c^(n) + 2 H~ c^(n) = 0 ,

with the NAC source `y~^(n)_{I,M} = (1/W_n) dH^{mn}/dc_{I,M}` at `c = c^(n)`.
Since `H^{mn} = <c^(m)|H|c^(n)>`,

    dH^{mn}/dc^(n) = H c^(m) = E_m c^(m) ,

so `y~^(n)` is proportional to `c^(m)` -- which is exactly `Y_z` as built at
`nuclear_grad.py:744` (the `(E - Enuc)` there is the electronic energy, the
`1/w` is the `1/W_n`). So `Y_z` is a legitimate Lagrangian source term, not a
mistake. The mistake is what it is combined with.

**The key fact: at any SA-CASSCF solution both states are eigenvectors of the
same `H`, so `H^{mn} = 0` IDENTICALLY, at every geometry.** Its TOTAL derivative
is therefore identically zero, and a Lagrangian whose objective is `H^{mn}` is a
Lagrangian for the zero function: `dL/dx = 0`. **`dL/dx` is NOT the NAC.**

The NAC / branching vector is the PARTIAL derivative
`<c^(m)|(dH/dx)|c^(n)>` -- integrals differentiated, CI vectors held fixed.

Differentiating the identically-zero quantity gives the whole story:

    0 = d/dx <c^(m)|H|c^(n)>
      = <c^(m)|(dH/dx)|c^(n)>  +  (E_m - E_n) t_mn      [using <dc^(m)|c^(n)> = -t_mn]

which is the same equation as section 2's middle row. Read it as a statement
about ROUTES rather than about `t`:

* **explicit route**: `<c^(m)|(dH/dx)|c^(n)>`, transition density x derivative
  integrals;
* **z route**: `y~^(n) ∝ c^(m)` drives `z` into sector (ii), yielding `t_mn`,
  contributing `(E_m - E_n) t_mn`.

The equation says the two are **equal in magnitude**. Including both counts the
same physics twice -- and this PREDICTS the exact factor of 2 measured in
section 4. It is not a coincidence; it is this relation.

The measurement confirms both directions. Section 4's test 2 --
`(kept - projected) == the ETF core` to 1e-08 -- says the **z route alone
reproduces the correct branching vector exactly**:

| route | result |
|---|---|
| explicit only (`Y_z` projected) | variant D, correct |
| z route only (`kept - projected`) | variant D, correct to 1e-08 |
| both (`Y_z` kept) | 2 x variant D, wrong |

So the two are genuinely interchangeable: removing the explicit term OR removing
the z source both give the right answer. (In the textbook relation above they
are equal and OPPOSITE and would cancel to zero; the explicit minus sign on
`Y_z` at :744 flips it, so they add instead. Hence 2x rather than 0.)
**Removing `Y_z` is the right choice here** -- it matches pyscf and Molpro and
leaves every existing sign convention untouched.

### What the z-vector is still for

The ORBITAL response. SA-CASSCF orbitals are optimised for the average, so they
are not variational for any individual state, and `dC/dx` genuinely enters
`<c^(m)|(dH/dx)|c^(n)>` through the MO coefficients; that is what the CP-MCSCF
solve delivers. What it must not ALSO do is re-supply the `c^(m)`-direction CI
response of the two target states, which the explicit term already carries.

---

## 5. Why the solve needs the *minimal* projection instead

Different question. Here the object is the *solution* `z`, and the issue is
sector (ii)'s physical content.

In-space components reach the gradient only through
`D_bar^SA = 1/2 * sum_n W_n (z^(n) (x) c^(n) + sym)`, where the (m,n) pair
contributes `W_n t_mn + W_m t_nm = (W_n - W_m) t_mn`, using `t_nm = -t_mn`.

* **Equal weights:** cancels identically -> sector (ii) of `z` contributes
  nothing -> projecting it out is harmless.
* **Unequal weights:** survives -> projecting it out deletes real physics.

Equivalently: under `z^(n) -> z^(n) + theta c^(m)`, `z^(m) -> z^(m) - theta c^(n)`
the gradient shifts by `theta (W_n - W_m) <c^(m)|dH/dx|c^(n)>`. Note this uses
`dH/dx`, which has no eigenvalue property -- the cancellation that makes sector
(ii) look redundant relies on `H c^(n) = E_n c^(n)` and does not survive
differentiation.

These sector-(ii) components of `z` are not the same thing as `Y_z`. They are
generated *during* the solve, through the orbital-CI coupling blocks of `A`, and
they must be allowed to appear. Hence: do not project them out of `x`.

---

## 6. So: RHS vs LHS is the wrong framing

It is not "full projection on the right-hand side, partial on the left". The two
projectors answer different questions:

* projector on trial + sigma = **which components `x` is allowed to have**
  (together they make the effective operator `PAP`);
* projector on `b` = **what drives the solve**.

| sector | in `b` (`Y_z`) | in `x` (trial/sigma) | why |
|---|---|---|---|
| (i) along `c^(n)` | remove | **remove** | `A` is singular there; required for solvability |
| (ii) along `c^(m)` | **remove** | **keep** | in `b` it is a duplicate (section 4); in `x` it is physical at unequal weights (section 5) |
| (iii) orthogonal | keep | keep | this is the actual solve |

There is no inconsistency in `b` and `x` living in different subspaces. You only
need `x`'s space to contain `b`'s, which it does.

This also predicts what was measured: at equal weights the `solve3` choice is a
no-op (5.7e-06 either way, section 5's cancellation), while the `Y_z` choice
matters at **every** weighting (section 4's duplication is weight-independent).

---

## 7. The CSF/overlap term is a separate axis -- you cannot trade one for the other

A natural question: if I stop projecting `Y_z`, do I still need the CSF and
overlap terms computed separately? **Yes, and dropping the projection does not
help.** They are independent:

* `Y_z` projection controls whether the ETF core is counted **once or twice**.
* the CSF/overlap term (added only under `deriv_type="full"`) is what converts
  the **ETF** form into the **with-CSF** form.

Test 2 of section 4 proves they are orthogonal: the extra copy that keeping
`Y_z` produces is the ETF core *exactly*, containing none of the CSF term. So
"keep `Y_z` and skip the CSF term" gives you twice the ETF core and no CSF
contribution -- neither of the two standard quantities.

The verified label map (H2O C1 to ~1e-06 vs pyscf AND Molpro; LiH CAS(2,2)/STO-3G
SA-2 to 1e-07/1e-08 vs the developer's own `test_nac.py`):

| our call | printed block | pyscf | meaning |
|---|---|---|---|
| `compute_grad(i,j)` | `NAC * dE` | D | `use_etfs+mult_ediff` = `<i|dH/dR|j>`, MECI branching vector |
| `compute_grad(i,j)` | `NAC` | B | `use_etfs=True` (ETF, no CSF) |
| `...deriv_type="full"` | `NAC * dE` | C | `mult_ediff=True` (with CSF) |
| `...deriv_type="full"` | `NAC` | A | `kernel(state=(i,j))` = **Molpro SA-MC NACME** |

A third wrong claim of mine, for the record: **`h - csf` is NOT the ETF form.**
It differs from pyscf's B by 38%. I inferred it from the translational sum rule
(both sum to zero over atoms) -- but two quantities can both be translationally
invariant and still differ. The sum rule identifies the CSF term; it does not
identify the ETF construction. The ETF form is the **bare** `compute_grad(i,j)`.

---

## 8. Status

Defaults now: `QED_PROJ_ALL=0`, `QED_PROJ_ALL_YZ=1`. Verified:

* unequal-weight **gradients** vs Molpro: worst **7.390e-08** (was 1.4e-03..7.5e-03);
* equal-weight **couplings** vs pyscf and Molpro: **5.7e-06 / 7.4e-06 / 7.0e-07**;
* unequal-weight **couplings** vs Molpro: **3.15e-07 / 3.49e-07 / 1.03e-07**
  (was 9.3e-02 / 1.2e-01 / 2.5e-01 -- the shipped code was 9-25% wrong);
* all four NAC variants on LiH CAS(2,2) vs `test_nac.py`: **1e-07 / 1e-08**;
* suite 134 passed / 4 failed (the 4 are pre-existing and unrelated).

### Still inference, not measurement

That the "first copy" of the ETF core comes from the effective transition
densities built on the orbital side. The *duplication* is proven to 1e-08; the
attribution of which code path produces the surviving copy is not.
