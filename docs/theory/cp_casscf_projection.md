# The CP-SA-CASSCF CI-response projection: statement, algebra, and the general rule

Rewritten after two rounds of testing that overturned parts of the first version.
Corrections to my earlier claims are marked **[WAS WRONG]**.

---

## 1. The equation and its three separate properties

The coupled-perturbed SA-CASSCF system couples the orbital response κ and the CI
response z⁽ⁿ⁾:

```
    [ A_oo  A_oc ] [ κ ]      [ b_o ]
    [             ] [   ]  =  [     ]
    [ A_co  A_cc ] [ z ]      [ b_c ]
```

with the CI–CI block acting as `(A_cc z)⁽ⁿ⁾ = (H − E_n) z⁽ⁿ⁾`. Solved by
right-preconditioned GMRES with a diagonal preconditioner `M⁻¹ = 1/denom`.

Expanding in the CI eigenbasis, `z⁽ⁿ⁾ = Σ_k a⁽ⁿ⁾_k c⁽ᵏ⁾`, the equation decouples
into one scalar row per k:

```
    a⁽ⁿ⁾_k (E_k − E_n) = ⟨c⁽ᵏ⁾|b⟩ .                                      (∗)
```

Everything below follows from (∗). Three properties must be kept apart, because
they fail independently:

| property | question | answer |
|---|---|---|
| **consistency** | does a solution exist? | yes if ⟨c⁽ᵏ⁾\|b⟩ = 0 wherever E_k = E_n |
| **uniqueness** | is it unique? | no — z + (anything null) also solves it; this is gauge |
| **conditioning** | can a Krylov solver find it? | only if near-null directions are kept out of the search space |

**[WAS WRONG]** My first version treated the stall as a consistency/solvability
problem. It is not. In this code the RHS is already consistent (§3), and GMRES
still fails. Existence of a solution says nothing about whether an iterative
method converges to it.

## 2. The null space

From (∗), `(H − E_n)` annihilates c⁽ᵏ⁾ exactly when E_k = E_n:

```
    N_n = span{ c⁽ᵏ⁾ : E_k = E_n }   (always contains c⁽ⁿ⁾)
```

and nothing else — for a separated root, `(H − E_n)c⁽ᵏ⁾ = (E_k − E_n)c⁽ᵏ⁾ ≠ 0`.

So neither legacy option is right in general:

* `projection` removes only c⁽ⁿ⁾ → at a degeneracy c⁽ᵐ⁾ stays in the search space.
* `project_out_all` removes every root → deletes determined components a⁽ⁿ⁾_k
  whenever states are separated. Measured in the toy (§5, case 1): it returns
  a_2 = 0 where the exact answer is +1.333, and fails to converge.

## 3. Why ⟨c⁽ᵏ⁾|b⟩ = 0 here, and why that is not a physics result

In this code `Y_z` — the CI part of b — is passed through `project_out_all` at
`nuclear_grad.py:820` whenever `QED_PROJ_ALL_YZ=1` (the default). So
⟨c⁽ᵏ⁾|b⟩ = 0 for **all** k by construction, and measurement confirms it:
`‖P_N b‖ = 0.000000e+00` exactly at the λ=0 MECP.

**[WAS WRONG]** I presented that zero as confirming the Lagrangian's invariance
under rotations in the degenerate manifold. That was circular: the zero is
manufactured by the code's own projection of `Y_z`, not an independent check.

Consequence worth noting: with `QED_PROJ_ALL_YZ=0` — the setting that is correct
at unequal weights — b **does** retain components along c⁽ᵏ⁾ for k ≠ n. The
general case therefore matters, and §5 tests it.

## 4. Why the iteration stalls anyway: the preconditioner

Let S = N⊥ and P the projector onto it. Right-preconditioned GMRES builds

```
    v_{k+1} ∝ orth( A M⁻¹ v_k , {v_0 … v_k} )
```

If v₀ ∈ S and both M⁻¹ and A map S → S, every Krylov vector stays in S by
induction, and **the recurrence itself is unchanged** — the projections are what
make that induction hold, not an extra step.

`M⁻¹` is **diagonal**, and a diagonal matrix does not commute with P. Measured in
the toy:

```
    v ∈ S :          ⟨c₁|v⟩       =  5.2e-18
    M⁻¹ v :          ⟨c₁|M⁻¹ v⟩   = -4.7e-02      ← not zero
```

16 orders above machine epsilon, so this is **deterministic, not roundoff**. `A`
then scales that component by `(E₁ − E₀) = 8e-7`. The Krylov space therefore
acquires, on every iteration, a direction on which the operator acts ~10⁶ times
more weakly than on everything else — an injected condition number of ~10⁶.
GMRES converges at a rate set by conditioning, so it crawls.

Measured, λ=0 ethylene MECP, root gap 8.4e-07 Eh:

| projection | restarts | outcome |
|---|---|---|
| `single` (c⁽ⁿ⁾ only) | 377+ (2002 in an earlier run) | frozen at ~1.3e-07, `Max Iterations` |
| `degen` (degenerate cluster) | **3** | converged, 30 s, \|h₀\| = 7.7e-02 |

Isolated toy reproduction: 12 restarts vs 2. The toy confirms the *mechanism*;
the real case is far worse in magnitude, presumably from dimension and from 2000
restarts of accumulation.

## 5. The general rule, from two tests that both surprised me

### Case 1 — b has a component along a NON-degenerate c⁽ᵏ⁾

Deliberate, and the situation on the unequal-weight-correct path. Toy: E₂ = 0.30,
⟨c₂|b⟩ = 0.4, so (∗) gives the exact `a₂ = 0.4/0.30 = +1.3333`.

| projection | a₂ obtained | restarts |
|---|---|---|
| `degen` (c₀,c₁,c₃) | **+1.333333** correct | 2 |
| `all` (c₀…c₃) | 0.000000 — deleted | 200, res 4.2e-01 |

**[WAS WRONG]** I first read this as `degen` failing, because my printed
reference carried a spurious minus sign (the formula `−⟨c|b⟩/(E_k−E_n)` is for
`A z = −b`, while the toy solves `A x = +b`). `degen` was correct throughout;
`all` is the one that deletes a real component *and* fails to converge.

### Case 2 — a root outside the computed set is degenerate with state n

Your scenario: average over states 1…n, project b against those n, but state n+1
is degenerate with n and was never computed, so ⟨c⁽ⁿ⁺¹⁾|b⟩ ≠ 0 and we cannot
project what we do not have.

| behaviour | restarts | final residual |
|---|---|---|
| projects c₃ (b keeps ⟨c₃\|b⟩ = 0.05) | 200 | **5.28e-02** — floors |
| blind to c₃ (does not project it) | 95 | 9.8e-11 — converges |

Opposite to my prediction, and the reason is the rule below. Projecting a
direction makes any b-component along it **unreachable**: no Krylov vector can
cancel it, so the residual floors at ≈|⟨c₃|b⟩|. Not projecting lets the solver
reach it, and because E₃ − E_n = 3e-7 is nonzero *exactly*, (∗) makes the
component formally determined — but enormous: `a₃ = 0.05/3e-7 ≈ 1.7e+05`.

So "converged" here is not reassuring. A hidden near-degenerate root produces
either a residual floor (if projected) or a huge response component (if not), and
**both are detectable**:

* residual stalls at a nonzero floor → suspect a b-component in a projected
  direction, i.e. an inconsistent system;
* a single `|a⁽ⁿ⁾_k|` far larger than the rest → suspect an uncomputed
  near-degenerate partner.

### The rule

```
    a direction is EXACTLY null (E_k = E_n):
        project it from BOTH b and the iteration.
        (legitimate: that component is gauge, fixed to 0 by the SA constraints)

    a direction is NON-null (E_k != E_n):
        project it from NEITHER.  Solve for it: a_k = <c_k|b>/(E_k - E_n).

    b and the iteration must project the SAME set.
        projecting more from b than from the iteration is safe but changes the
        problem (those components become 0 in the solution);
        projecting more from the iteration than from b makes the system
        inconsistent and floors the residual.
```

The current code violates the last clause in a benign direction: b is
`project_out_all`'d (all components zeroed) while the iteration projects only the
degenerate cluster. Benign because the extra zeroed components are then simply
not driven — but it is exactly why `QED_PROJ_ALL_YZ=1` deletes the real Y_z
contribution at unequal weights. With `QED_PROJ_ALL_YZ=0` the two are consistent
under `degen`, which is the combination that is correct in general.

## 6. The implemented routine

For each row n:

1. **Cluster the roots** by single linkage on the sorted energies, tolerance τ
   (`QED_DEGEN_TOL`, default 1e-5 Eh), so every member of a cluster is projected
   against the *same* subspace. A per-row pairwise test fails because
   near-degeneracy is not transitive: with E = 0, 6e-6, 1.2e-5 it gives row 0 the
   subspace {0,1}, row 1 {0,1,2}, row 2 {1,2} — three different "null spaces" for
   one operator.
2. **Orthonormalise each cluster by SVD.** Sequential Gram-Schmidt is a projector
   only on an orthonormal set, and degenerate Davidson roots need not come back
   orthogonal: with ⟨c₀|c₁⟩ = 0.3 the sequential form leaves |⟨row|c₀⟩| = 0.265
   instead of 0 — it fails silently.
3. **Project** the preconditioned vector and the sigma vector onto the complement
   of that cluster. Both are required: M⁻¹ breaks S-invariance deterministically
   (§4), and A leaks under roundoff.

All computed roots participate, including zero-weight ones — which matters for
the zero-weight SA-3 setup, where root 1 carries weight 0 but is computed and so
is available to the clustering.

Unit tests (`probe/test_proj_general.py`), all ~1e-16: two separate degenerate
pairs (with the other block correctly left at 1.55), a three-fold degeneracy, the
chained case, the non-orthonormal case.

## 7. Known defect: τ cannot be chosen well in general

Single linkage **chains**. Measured with τ = 1e-5:

| spectrum | clusters | span |
|---|---|---|
| 0, 2e-7, 1.0 | 2 ✓ | 2.0e-07 |
| 0, 8e-6, 1.6e-5, 2.4e-5, 3.2e-5, 4.0e-5 | **1** | **4.0e-05 = 4τ** |
| 0, 1e-7, 5e-5, 5.00001e-5 | 2 ✓ | 1e-07, 1e-10 |

A ladder of sub-τ steps merges into one cluster of unbounded span — that is
`project_out_all` by the back door, with exactly the over-projection error τ was
meant to prevent. Tightening τ splits genuine clusters; loosening merges more.
No absolute tolerance avoids both.

Harmless for the present decks, where the regimes sit three orders either side of
τ (MECP root gaps 3.8e-07…1.9e-06 Eh; FC separations 0.118 Eh and up), but not a
general routine.

## 8. What would remove the tolerance

**Deflation instead of projection.** Exactly-null directions are gauge and must be
projected. Near-null directions are *determined* and should be solved exactly,
not deleted — case 2 shows deleting them floors the residual. Augmenting the
Krylov space with the known CI eigenvectors (deflated/augmented GMRES) handles
small denominators through the augmented subspace, with no threshold anywhere.

**Diagnostics that need no threshold**, and both come free:

* print |E_k − E_n| and |⟨c⁽ᵏ⁾|b⟩| for every root pair, so the conditioning is
  visible rather than decided by τ;
* print the largest |a⁽ⁿ⁾_k| in the converged solution — a value orders above the
  rest signals an uncomputed near-degenerate partner (case 2);
* compute one root beyond the averaged set and report its gap to the highest
  averaged root. This is the only way to *detect* case 2 rather than infer it,
  and it is cheap insurance.

A note on case 2 beyond the solver: if state n+1 is degenerate with state n and
only 1…n are averaged, the SA-CASSCF functional itself is ill-posed — the orbital
optimisation splits a degenerate pair across the averaged and non-averaged sets
and breaks the degeneracy arbitrarily. The response equations are then the second
problem, not the first.

## 9. Status

| item | status |
|---|---|
| null space = the degenerate cluster | derived from (∗); 3 restarts vs stall |
| a⁽ⁿ⁾_k = 0 in the null space | gauge + solvability |
| ⟨c⁽ᵏ⁾\|b⟩ = 0 in this code | **by construction** (project_out_all on Y_z), not physics |
| projection on preconditioner and sigma | both required; M⁻¹ breaks S deterministically (4.7e-02) |
| projection on b | required only for *consistency* with the iteration's projector |
| non-degenerate components must NOT be projected | measured: `all` gives 0 vs exact +1.333 |
| hidden degenerate root outside the set | detectable by residual floor or a huge a⁽ⁿ⁾_k; **not handled** |
| single-linkage chaining | **known defect**, demonstrated, not fixed |
| deflation | proposed, not implemented |
