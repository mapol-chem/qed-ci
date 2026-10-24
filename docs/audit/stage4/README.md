# Stage 4 validation

Scope: additive structured results and normal-verbosity reports. No numerical
algorithms, convergence defaults, C backend, C++/TAMM code, packaging, or reference
provider integration changed. `timer.dat` already had user changes and is preserved.

The local environment has changed since Stage 1: the same `p4dev` path now provides
Python 3.12.13, Psi4 1.10.1, and NumPy 2.4.6. The existing backend SHA256 is
`63deb238c8a641e91195a72de1870a8a0b451decdf692763f29bc35eabc4a4ef`.
It was not rebuilt. Comparisons use the original saved arrays and tolerances;
they do not assert a bit-identical environment.

## Scientific output-consumer audit

- `test_unequal_weight_sa_vs_molpro.py` formerly extracted printed gradient/NAC
  matrices. Its subprocess now emits the JSON-compatible property summary and
  its parent reads those arrays, preserving the same numerical assertions.
- `test_output.py`'s gradient append assertion now checks the conventional property
  heading. This is a formatting assertion, independent of gradient comparisons.
- `test_trs_hard_case.py` reads standalone driver `ROOT0` / `ROOTn` / `AVGE`
  records and uses trace diagnostics to verify solver-branch coverage. Those
  interfaces are retained. In particular, its initial-root assertions have not
  been silently changed to optimized-root assertions.
- `test_numerical_gradient_cavity.py` and `test_numerical_nac.py` consume standalone
  finite-difference validation tables (`worst`, `WORST OVERALL`, pair/frame/branch
  checks). The validation scripts compute comparisons from arrays, not engine
  report text; their output contracts remain unchanged.
- `validation/lstrs_refactor_gate.sh` excludes the bounded Stage 4 presentation
  block before its existing trajectory comparison; driver records and numerical
  trace lines are retained. Its Linux/taskset chemistry gate was not executed on
  this Mac; the filter is checked separately. No solver refactor is part of this stage.

## Reproduction

Run from a scratch directory with the repository's `src` on `PYTHONPATH`, using
`/Users/jfoley19/miniforge3/envs/p4dev/bin/python`, with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`:

```sh
python -B -m pytest /path/to/qed-ci/tests/test_config.py /path/to/qed-ci/tests/test_output.py /path/to/qed-ci/tests/test_results.py -q
python -B /path/to/qed-ci/tools/validate_stage23.py --output /path/to/qed-ci/build/stage4/replay-new
python -B -m pytest /path/to/qed-ci/src/tests -q
```

The existing replay tool now additionally verifies result/legacy energy equality,
including that CASSCF initial and optimized roots are distinct. All eight Stage 1
inputs are replayed through both entry paths. Array shapes and finiteness are
checked, energies use `1e-8` Eh, and saved RDMs/weights use `1e-7` absolute tolerance.
CI-vector signs/degenerate rotations are not elementwise compared.

Focused coverage includes normal/debug/trace, quiet and file-only routing,
append/exclusive modes, gradient/NAC append and owned property snapshots,
subsequent scoped calls, exception restoration, normalized unequal weights,
energy/gradient macro stopping, forced CI and macro nonconvergence, unknown
canonical-reference diagnostics, and generated versus stale export files.
Timings and full logs are not snapshot-tested.

Machine-readable counts, comparison maxima, failures and provenance are recorded
in `results.json`. Raw logs and XML remain under ignored `build/stage4/`.

## Limitations

Canonical Psi4 SCF diagnostics and dense CI residuals/iterations are explicitly
unavailable. Package release version is not yet defined. Native individual CI
residual storage covers at most 64 spin-adapted roots; unadapted Davidson
provides only the mean residual. Microiteration diagnostics and all
free-text legacy notices are not converted into structured events; debug/trace
retain those logs. Results do not serialize CI/RDM tensors. File inventories
cover package exports and reports, not Psi4 scratch or developer dump trees.
Output/Psi4/CWD routing remains process-global; independent concurrent calculations
need separate processes. Direct imports and Psi4 process-finalization output are
outside constructor scopes. Linux was not executed. Stages 5 and 6 remain deferred.

## Recorded outcome

- **54/54 focused tests passed** on the final diagnostic implementation.
- **16/16 baseline replays passed**, maximum energy difference
  `6.36e-13` Eh and maximum RDM/weight difference `2.93e-9`.
- **Full scientific suite: 174 passed, 3 failed** (177 tests). Two failures are
  the preserved hardcoded Linux fixture path and absent
  `total_energy_from_rdms` attribute. The additional
  `test_finite_difference_error_scales_as_delta_squared` failure occurred because
  a calculation printed `GLTR: Tridiagonal solver failed` and never reached its
  comparison table. Its **unchanged isolated rerun passed**. This observed
  intermittent solver failure remains recorded; it has not been repaired or
  hidden by changing tolerances. The run is not claimed fully green.
- The migrated unequal-weight module also passed **20/20** in an isolated run.
- Shell syntax, trajectory-report filtering, and `git diff --check` passed.

The full run included 53 focused tests as well (227 passed / 3 failed combined).
After the per-root residual availability review, the focused set was extended
and rerun (54 passed), along with all 16 baseline replays. Only diagnostics changed
in that correction; numerical algorithms and thresholds remained untouched.
