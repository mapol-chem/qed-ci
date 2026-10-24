# Why an all-states CI projection is an equal-weight-only construction

Step-by-step derivation of where the `(W_n - W_m)` factor comes from, in the
notation of the SA-QED-CASSCF Lagrangian.

Companion to `BUG_project_out_all.md` (the measurement) in the same directory.
Written 2026-09-08.

Notation: `z^(n)` is the CI Lagrange multiplier for state n, `c^(n)` the CI
vector, `W_n` the state-average weight, `t_mn = c^(m) . z^(n)`.

---

## Step 1: the rule you already have, restated as a test

`z^(n) . c^(n) = 0` holds because that component is *undetermined*. The test is:
add `alpha c^(n)` to `z^(n)` and see if anything changes.

    alpha * W_n * c^(n)+ (H - E_n) c^(n) = alpha * W_n * (E_n - E_n) = 0

Nothing changes, so the component is free, so you fix it by convention.

---

## Step 2: apply the same test to the off-diagonal

Now add `theta c^(m)` to `z^(n)`, with m != n:

    theta * W_n * c^(m)+ (H - E_n) c^(n) = theta * W_n * (E_n - E_n) * c^(m)+c^(n) = 0

Also zero. **This is the trap.** It looks like the same redundancy, so it looks
safe to project those components out too -- which is what BAGEL does for the
equal-weight case, and what `solve3` inherited.

---

## Step 3: why that is the wrong test for a Lagrange multiplier

At the solution the constraint is satisfied, so that whole term is zero *no
matter what* `z` is. Its value was never the point. `z` earns its keep in the
**derivative**. Differentiating the constraint term with respect to the nuclear
coordinate:

    dL/dx  contains  sum_n W_n * z^(n)+ (dH/dx) c^(n)

(the `dE_n/dx` piece drops precisely *because* `z^(n).c^(n) = 0` -- the Step 1
convention earning its keep a second time).

So the real test is: does adding `theta c^(m)` change THIS?

    theta * W_n * c^(m)+ (dH/dx) c^(n)

`c^(m)` and `c^(n)` are eigenvectors of **H**, not of **dH/dx**. So this is NOT
zero. Step 2's cancellation relied on `H c^(n) = E_n c^(n)`; `dH/dx` has no such
property.

---

## Step 4: where (W_n - W_m) comes from

The components do not move one at a time. The natural direction is the
antisymmetric pair:

    z^(n) -> z^(n) + theta c^(m)
    z^(m) -> z^(m) - theta c^(n)

Adding both contributions, and using that `dH/dx` is symmetric:

    theta W_n c^(m)+ (dH/dx) c^(n)  -  theta W_m c^(n)+ (dH/dx) c^(m)
      = theta (W_n - W_m) <c^(m)| dH/dx |c^(n)>

    ================================================================
     Delta(gradient) = theta (W_n - W_m) <c^(m)| dH/dx |c^(n)>
    ================================================================

* **Equal weights:** zero -> genuinely redundant -> projecting is free.
  This is why BAGEL can do it.
* **Unequal weights:** non-zero -> not redundant -> projecting deletes a real
  contribution to the gradient.

That is the entire origin of the factor.

---

## Step 5: why that pairing, physically

The state-averaged density is `D^SA = sum_n W_n |c^(n)><c^(n)|`. Rotate two
states inside the averaged space (`c^(n) -> c^(n) + theta c^(m)`,
`c^(m) -> c^(m) - theta c^(n)`):

    D^SA -> D^SA + theta (W_n - W_m) ( |c^(m)><c^(n)| + h.c. ) + O(theta^2)

* **Equal weights: you are averaging over a SUBSPACE.** Only the span matters, so
  rotations inside it are meaningless -- and that meaninglessness propagates to
  the multipliers.
* **Unequal weights: you are averaging over SPECIFIC STATES.** Which vector is
  which now matters, so those rotations are physical.

Same `(W_n - W_m)`, reached independently of Step 4.

---

## Step 6: so what fixes the component, if it is not free?

Your own CI stationarity condition does. Take `c^(m)+` of it, m != n:

    y~^(n) + (H - E_n) z^(n) - x_n c^(n) + 2 H~ c^(n) = 0

term by term:

* `c^(m)+ y~^(n) = 0` -- for the gradient `y~^(n) = H c^(n) = E_n c^(n)`, which is
  orthogonal to `c^(m)`
* `c^(m)+ (H - E_n) z^(n) = (E_m - E_n) t_mn`
* `- x_n c^(m)+ c^(n) = 0`
* `+ 2 <c^(m)| H~ |c^(n)>`

giving

    (E_m - E_n) t_mn + 2 <c^(m)| H~ |c^(n)> = 0

    t_mn = -2 <c^(m)| H~ |c^(n)> / (E_m - E_n)

**The equation determines it.** Forcing it to zero means not solving the equation
you wrote down.

Consistency check: `H~` is symmetric, so this formula gives `t_nm = -t_mn` -- the
antisymmetric pairing assumed in Step 4 is exactly the direction the equations
actually populate.

---

## Step 7: what the numbers showed

H2O CAS(6,5)/6-31G, weights 5,2,1, vs Molpro (`noextra`):

| geometry | `<c^(1)|H~|c^(2)>` | CI residual | t_12 predicted | t_12 actual | grad error |
|---|---|---|---|---|---|
| C2v (real molecule) | ~3e-11, symmetry-forbidden | 3.4e-08 converged | ~0 | ~0 | ~1e-8 (fine) |
| C1 (one H displaced) | **-5.13e-03** | **5.13e-03 UNCONVERGED** | **+5.25e-02** | ~1e-16 | **1e-3 .. 7e-3** |

On C2v, `t_mn = 0` is the *correct* answer, so the projection costs nothing --
which is why every symmetric test case passed. On C1 the projection leaves the CI
equation unsatisfied, entirely in the `c^(1)-c^(2)` direction.

Switching the two `solve3` sites to the minimal projection:

| weights | shipped (`project_out_all`) | minimal projection |
|---|---|---|
| 1,1,1 | 6.5e-09 / 3.3e-08 / 2.8e-08 | 2.3e-08 / 3.2e-08 / 3.1e-08 |
| **5,2,1** | **1.37e-03 / 3.18e-04 / 7.47e-03** | **1.7e-08 / 5.6e-08 / 7.1e-08** |

---

## Step 8: on "nothing in the original work says to project"

That is right, and it is consistent with everything above. The theory gives
exactly one convention, `z^(n).c^(n) = 0`, for exactly one undetermined
direction. The all-states projection is an **extra** assumption imported from an
equal-weight implementation. There it is harmless (Step 4) and mildly convenient
-- it removes a null direction that would otherwise make the linear system
singular. Outside that setting it is an unstated approximation.

Since `(W_n - W_m) = 0` makes the extra projection a no-op anyway, **the minimal
projection is correct in both regimes** -- which is what the measurement shows:
equal weights move by 2e-08, unequal weights go from 7e-03 wrong to 7e-08 right.

---

## Where this lives in the code

* `nuclear_grad.py:1447` and `:1466` -- the two live `project_out_all` calls,
  inside `solve3`'s trial and sigma helpers. **`solve3` is the live solver**
  (called at `:2256`); `solve2` is dead (its call at `:2255` is commented out)
  and uses the opposite projector, which is easy to misread.
* `nuclear_grad.py:867-879` -- `self.projection`, the minimal form
  (`z^(n) . c^(n) = 0` only), already written and commented out beside both call
  sites.
* `nuclear_grad.py:677` -- a third `project_out_all`, on `Y_z`. For a gradient
  `Y_z = 0` so it is a no-op, but for **derivative coupling** `Y_z != 0` and this
  one does act. That path is untested.
