# The two projectors are DIFFERENT questions with OPPOSITE answers

Settled 2026-09-08 against **two independent references that agree with each
other to ~5e-07**: pyscf's `nac_method` and Molpro's `SA-MC NACME`.

**This document CORRECTS an earlier draft of mine** which claimed
`project_out_all` "annihilates the derivative-coupling source term" and was
therefore a second bug. That claim was WRONG. It came from putting all three
call sites behind one knob and reading a single A/B. The projector on `Y_z` is
*required*; removing it is what breaks the coupling.

## The result

`nuclear_grad.py` has three live projections. They are now two separate knobs
because the evidence points opposite ways:

| site | knob | correct setting | what decides it |
|---|---|---|---|
| `solve3`'s preconditioner + matvec wrappers (the z-vector SOLVE) | `QED_PROJ_ALL` | **0 = minimal** (`z^(n).c^(n)=0` only) | unequal-weight gradients AND couplings |
| `build_total_gradient`'s `Y_z` (the CI block of the z-vector RHS) | `QED_PROJ_ALL_YZ` | **1 = project_out_all** | derivative coupling at ANY weighting |

### Equal-weight coupling: the `Y_z` projector is what matters

Relative error of our `h/egap` vs pyscf variant A (= Molpro):

| solve3 / Y_z | pair 0-1 | pair 0-2 | pair 1-2 |
|---|---|---|---|
| minimal / minimal | 2.36e+00 | 1.95e+00 | 1.03e+00 |
| **minimal / projall (NEW DEFAULT)** | **5.74e-06** | **7.41e-06** | **7.02e-07** |
| projall / minimal | DIVERGED | | |
| projall / projall (old shipped) | 5.72e-06 | 7.42e-06 | 6.59e-07 |

At equal weights the solve3 choice is irrelevant (`W_n - W_m = 0`), exactly as
the gradient theory predicts. **`Y_z` must be projected against all states.**

### Unequal-weight coupling: the solve3 projector is what matters

Weights 5,2,1, relative error vs Molpro `SA-MC NACME`:

| solve3 / Y_z | pair 0-1 | pair 0-2 | pair 1-2 |
|---|---|---|---|
| **minimal / projall (NEW DEFAULT)** | **3.15e-07** | **3.49e-07** | **1.03e-07** |
| projall / projall (old shipped) | 9.34e-02 | 1.24e-01 | 2.46e-01 |

**So the shipped code was wrong by 9-25% on unequal-weight derivative
couplings** -- the same defect as the gradient, reaching the coupling through
the same two solve3 sites.

### Gradients are unaffected by the split

`Y_z` is identically zero when `state1 == state2`, so `QED_PROJ_ALL_YZ` cannot
touch a gradient. Re-verified after the split: worst discrepancy vs Molpro
**7.390e-08** across both weightings.

## Why this is not a contradiction

Rotations WITHIN the state-averaged CI space are not part of `z` at all -- they
are carried by the separate off-diagonal multipliers fixed by
`(E_m - E_n) t_mn + 2<c^(m)|H~|c^(n)> = 0`. `Y_z` is built ENTIRELY from those
in-space directions (`Y_z[state1]` is proportional to `eigenvecs[state2]`), so
projecting them out there is not deleting a source term -- it is declining to
double-count one. The z-vector SOLVE is the different question: there the
minimal projection is what the theory licenses, and projecting more discards a
physical `(W_n - W_m)` component whenever the weights are unequal.

## Conventions -- now pinned, and they were the whole confusion

pyscf's `nac_method` labels four variants; `test_nac.py` in
`/home/nvu12/jobs/pyscf/n2/` is the developer's own verification deck:

| pyscf call | meaning | atom sum |
|---|---|---|
| `kernel(state=(i,j))` | NAC **with** CSF, **not** scaled by ediff | nonzero |
| `+ use_etfs=True` | NAC **without** CSF (ETFs) | **zero** |
| `+ mult_ediff=True` | NAC x ediff | |
| `use_etfs + mult_ediff` | `<i|dH/dR|j>` -- the MECI branching-plane vector | |

* **Molpro's `SA-MC NACME` == pyscf variant A** (with CSF, unscaled). Verified
  directly: pair 0-2 pyscf `0.021214622 / -0.184709252 / 0.131018611` vs Molpro
  `0.021214072 / -0.184708838 / 0.131018415`.
* **Our `compute_grad(i,j,deriv_type="full")` "h vector" contains the CSF term**
  and our `h - csf` is the ETF form -- confirmed by the translational sum rule,
  which is the code-independent discriminator: `h - csf` sums to **1e-10** over
  atoms, `h` does not.
* Our NAC = `h/egap`; our `h` itself is the x-ediff form. Note the second block
  our code prints under a coupling is labelled "Total gradient" but is actually
  `h0/(E1-E2)` (`nuclear_grad.py`) -- a mislabelled print, worth fixing.
* NACs carry the arbitrary relative CI phase, so they compare up to an overall
  sign per pair. A global sign flip is not a discrepancy.

## Reproduce

```sh
# references
python pyscf/h2o_c1_nac.py                       # pyscf, all four variants
#   molpro/h2o_c1_nacm.com -> out/h2o_c1_nacm.out (both weightings)

# ours, 2x2 over the two knobs
for PA in 0 1; do for YZ in 0 1; do
  QED_PROJ_ALL=$PA QED_PROJ_ALL_YZ=$YZ \
    ./run_ours_nacm.sh h2o_c1 1,1,1 nacm2_h2oc1_w111_pa${PA}yz${YZ}
done; done
```
