# Stage 1: configuration, output, and validation baseline

Scope: baseline capture before the proposed input/output refactor. Stage 2 is
not authorized and has not started. The only runtime source changes in this
stage are the separately approved native-build prerequisites: a BLAS portability
header, equivalent LP64 types/thread controls, and a safe macOS/Linux memory query.
No solver defaults, input parsing, output policy, or convergence tests are changed.

## Local results (2026-10-19)

Native arm64 build succeeded with Apple Clang 21, OpenBLAS 0.3.30 and LLVM OpenMP,
using the existing `p4dev` environment (Python 3.13.13, Psi4 1.10). The compiler
reported 25 warnings in `ci_solver.c` and 14 in `orbital.c`; warnings are retained
in `build/backend/build.log`. Linux/Intel execution has not been tested on this Mac.

| Check | Observed result |
|---|---|
| Existing Python suite | 175 passed, 2 failed; 177 collected; 409.78 seconds |
| Native backend checks | 4 passed: both LAPACK solver paths/layout, memory query, failed-build preservation |
| Explicit method/cavity captures | 16/16 completed; eight cases repeated twice; all captured arrays finite |
| Repeat energy differences | CIS/FCI/CASCI exactly repeated; largest CASSCF difference 2.43e-12 Eh or less |
| Final build / final capture | Library hashes match |

The two existing-suite failures are left intact:

1. `test_lih_fci_sto3g_rdm_builds_no_cavity` cannot read its hardcoded
   `/home/jfoley19/UPDATED_QEDCI/.../LiH_sto3g_fci_d1.npy` fixture path.
2. `test_lih_fci_sto3g_rdm_builds_with_cavity` requests
   `total_energy_from_rdms`, which is absent from `PFHamiltonianGenerator`.

Neither failure reached a failed numerical comparison. This is not a claim of a
fully passing suite. The other scientific checks, including the cavity gradient
and derivative-coupling tests, passed in this run. The full suite used the first
successful native library; final captures and four native checks used the final
library after a whitespace-only source cleanup and stable macOS install-name fix.
Both library hashes are retained in `results.json`; no numerical code changed
between those builds. Unseeded solver behavior can still affect trajectories.

## Reproduce

From the repository root, in the calculation environment:

```sh
python tools/build_backend.py
python tools/audit_stage1.py
python tools/capture_stage1.py --output build/stage1/baselines
python -m pytest tests/test_backend_build.py -q
```

The recorder refuses to reuse case directories. To repeat the complete capture,
choose a new output directory. It launches each calculation in a fresh process,
clears `QED_*` and `CPP_CASSCF_*` overrides, and pins BLAS/OpenMP/Psi4 to one thread.
The eight LiH/STO-3G cases span CIS, FCI, CASCI and equal-weight SA-CASSCF, each
with coupling off/on, two requested roots, and two repetitions. CAS uses two
active electrons in two orbitals. Each input dictionary, stdout/stderr, emitted
file, numerical array and result summary is retained. These captures establish
local behavior, not an independent proof of scientific correctness. Existing
scientific tests supply independent reference assertions and gradient coverage.
`baselines.json` preserves the exact inputs, result shapes, energies and measured
repeat differences; `reference/*.npz` preserves the first repetition's arrays.
CI-vector signs and degenerate-state rotations are arbitrary: raw elementwise
differences are recorded as diagnostics, not universal pass/fail tolerances.

Full test baseline (run from a scratch directory with `src` on `PYTHONPATH`):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest /path/to/qed-ci/src/tests -q --junitxml=tests.xml
```

Local observations and failures are in `results.json`. Large raw artifacts live
under ignored `build/`; compact numerical summaries are retained alongside this
report. Preserve the raw directory when comparing later refactors. Build provenance
includes the library/source hashes; exact trajectories are environment-specific.

## Inventory and its limits

`inventory.json` is generated without importing the package. It covers every
top-level `src/*.py` and `src/*.c` source, with source hashes and line references:

- all 28 `parseCavityOptions` keys, full parsing expressions, and default branches;
- direct environment reads, defaults as written, and Python import/call timing;
- dictionary reads for locally consumed options (Psi4's complete schema is external);
- top-level driver command-line options and public CI/gradient method signatures;
- Python output, warnings, exits, file reads/writes and C stdio/exit sites;
- test functions and Markdown documentation paths.

C comments are stripped before scanning. C caching requires inspection of the
listed surrounding source; an environment read does not necessarily occur every
call. Legacy Python kernels are explicitly tagged as quarantined. Driver files
are included, so counts are source sites rather than reachable production calls.
Indirect/dynamic option access and output through external libraries cannot be
exhaustively inferred from this static inventory; raw run logs complement it.

## Cavity input defaults and interactions

| Input | Initial default | Interpretation / issue |
|---|---|---|
| `omega_value` | 0 | Frequency; units and finite/physical validation need a public contract. |
| `lambda_vector` | `[0,0,0]` | Three coupling components; no shape/finite validation here. |
| `number_of_photons` | 1 | Maximum photon number, not number of sectors. |
| `ci_level` | `cis` | Main dispatch accepts lowercase; uppercase handling is inconsistent. |
| `casscf_optimization` | `True` | Meaningful for CAS; method and optimization flags overlap. |
| `ignore_coupling` | `False` | Changes multiple Hamiltonian terms; not assumed equivalent to zero lambda. |
| `natural_orbitals` | `False` | Runs an additional CIS calculation to construct starting orbitals. |
| `canonical_mos` | `False` | Either this or photon-number flag forces both true and coherent false. |
| `photon_number_basis` | `False` | Coupled to orbital choice, not independently validated. |
| `coherent_state_basis` | conditional `True` | Defaults false if canonical/photon-number requested; can be overwritten. |
| `ignore_dse_terms` | `False` | Forces photon-number true/coherent false after prior reconciliation; does not also force canonical true. |
| `rdm_weights` | `[1,1]` | Read only for natural orbitals; distinct from CASSCF averaging weights. |
| `full_diagonalization` | `False` | CIS uses dense diagonalization regardless. |
| `test_mode` | `False` | Enables dense allocation paths; not merely extra assertions. |
| `break_degeneracy` | `False` | Parsed attribute has no subsequent use in this Python module; C ABI still has arguments with this name. |
| `davidson_roots` | 3 | Also controls state averaging; CIS returns the whole spectrum. |
| `davidson_guess` | `unit guess` | String selector; needs supported-value documentation. |
| `davidson_threshold` | `1e-5` | CI residual threshold; separate from CASSCF convergence. |
| `davidson_indim` | 4 | Initial-subspace control; actual dimensions depend on roots/backend logic. |
| `davidson_maxdim` | 20 | Restart/subspace control; validate against initial dimension and problem size. |
| `davidson_maxiter` | 100 | CI iteration budget; other CI budgets exist in environment variables. |
| `use_orbital_guess` | `False` | Reads fixed `orbital.out`. |
| `save_orbital` | `False` | Does not control the unconditional `orbital2.out` write. |
| `rdm_root` | 0 | Zero-based root selection; should be range checked. |
| `nact_orbs` | 0 | Read for CAS; otherwise reset to zero. |
| `nact_els` | 0 | Read for CAS; missing active space is not rejected early. |
| `spin_adaptation` | `no` | CAS/FCI: singlet/triplet/quintet/no; invalid values print and exit. |
| `casscf_weight` | ones for all roots | Normalize; short arrays zero-pad; nonpositive total or too many entries print and exit. Individual negative/nonfinite weights are not rejected here. |

Keys are lowercased at construction. Unknown keys are ignored. The resolved
configuration cannot be inferred from one default column alone: ordering and
conditional overwrites matter. The inventory retains the exact source expressions.

## Environment and output findings

Environment settings fall into scientific convergence controls (macro/micro/CI),
algorithm choices (trust region, quasi-Newton, spin treatment), integral and BLAS
optimizations, orbital initialization, profiling, and validation dumps. Many Python
settings are frozen at import. C includes cached settings and runtime reads. A later
configuration refactor must record precedence and preserve defaults before moving
controls; merely assigning environment values around an existing object is unsafe.

The C solver already supports `set_ci_print_level` and `QED_PRINT_LEVEL`, defaulting
to trace (3), but this does not control Python, all C diagnostics, or Psi4 output.
Output includes useful convergence tables alongside raw arrays, internal timing and
debug labels. Some validation scripts parse this text: change their data interface
before reformatting it. There is no unified quiet/file flag at the Python entry point.

File classes to separate later: human report, orbital guesses/checkpoints, optional
property exports, developer dumps, integral spill/scratch files, and Psi4 runtime
files. Fixed orbital filenames and the unconditional `orbital2.out` write prevent
safe shared-working-directory use. Captures retain actual emitted file manifests.

Two additional measured interface issues: CASSCF's `CIeigs` retains the initial
CI solve while `eigenvals` holds optimized state energies; both must be retained
until a replacement result API defines ownership. The dense CIS path allocates
two photon sectors (`generatePFHMatrix`), even for `number_of_photons=0`; the
zero-coupling capture consequently includes duplicate photon-sector energies.
Neither behavior was changed. On macOS the existing `orbital_sigma.py` no-MKL
fallback disables `QED_G_SYMV` and uses its NumPy/Numba matrix-vector path;
this is an existing supported fallback, not a newly ported MKL optimization.

## Tests and documentation findings

The source suite has 13 modules and 137 test function definitions; parametrization
produces 177 collected tests in the local environment. Coverage includes reference
energies, RDM identities, solver subproblems, cavity-on finite-difference gradients,
unequal-weight Molpro comparisons, and derivative couplings. C++ tests and validation
scripts are separate coverage, not included in those counts.

Known portability issues include an absolute `/home/jfoley19/...` RDM fixture path,
heavy imports for simple unit tests, subprocess dependence on import-time settings,
stdout-based result parsing, and floating-point-sensitive branch-coverage assertions.
Do not weaken those assertions during baseline collection. Record local failures
with evidence and distinguish a known structural issue from a new numerical mismatch.

The README now documents the approved macOS/Linux build prerequisite. Remaining
documentation work is deferred: a public options reference, units and result shapes,
method-specific minimal examples, output controls, migration guidance, and a supported
feature matrix. Existing theory and measured-report notes should be preserved.

## Stage boundary

No `CQEDConfig`, Calculator, reference-provider replacement, output redesign, test
expectation fixes, or solver tuning is implemented. Stage 2 requires user approval.
Linux retains Intel oneAPI; this Mac cannot validate Linux execution. The separate
C++ CMake build and optional DPD shim are not ported by this prerequisite change.
