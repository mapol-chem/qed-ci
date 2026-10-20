# Runnable examples

First follow [installation](../../docs/installation.md). Run from a separate
working directory; the paths below refer to your checkout. No `PYTHONPATH` is
needed after installation.

```sh
python /path/to/qed-ci/examples/quickstart/methods.py cis --quiet
python /path/to/qed-ci/examples/quickstart/methods.py fci --output lih.out
python /path/to/qed-ci/examples/quickstart/methods.py casci
python /path/to/qed-ci/examples/quickstart/methods.py casscf
python /path/to/qed-ci/examples/quickstart/methods.py sa --quiet
python /path/to/qed-ci/examples/quickstart/gradient.py
```

All use modest LiH/STO-3G with a cavity and two requested roots. The methods are:

- `cis` — dense QED-CIS.
- `fci` — full CI in the reference orbital space.
- `casci` — one fixed-orbital active-space diagonalization (2 electrons in
  2 orbitals).
- `casscf` — state-averaged orbital optimization for the same CAS(2,2), with
  ground-state weights `(1, 0)`.
- `sa` — state-averaged CASSCF with equal weights `(0.5, 0.5)`.
- `gradient.py` — equal-weight SA-QED-CASSCF plus an analytic ground-state
  nuclear gradient. This is the most expensive example; CASSCF and response
  calculations cost more than the fixed-orbital examples. Larger FCI spaces grow
  combinatorially.

`--quiet` suppresses routine calculation output while the small application
summary remains under the caller's control. `--output lih.out` writes the
conventional report to a file (replacing an existing file) with no duplicate
console report. No orbital exports are requested. Psi4 can still write
`timer.dat` at process exit; use a disposable run directory.

## Reading the results

- `result.energies['final']` contains the requested final energies in Eh,
  including optimized energies for CASSCF.
- `result.converged` is `True`, `False`, or `None` (unknown). Check it before
  interpreting the last energies as converged; a capped run returns its last
  available CI energies.
- `result.energies['state_average']` is the weighted final-state average in Eh
  for state-averaged calculations.
- `result.summary()` is detached and JSON-compatible. Prefer it over parsing the
  human-readable report.
- The gradient example stores the analytic gradient under
  `result.properties['gradient_state_0']['value']`, shape `(2, 3)`, in Eh/bohr.
  Derivative couplings use units of 1/bohr.

The examples do not promise portable timings or last printed digits. See the
[option and result reference](../../docs/configuration-and-output.md) and the
[method restrictions](../../docs/methods.md).
