# Developer setup and testing

This page covers the Python package and its ctypes backend.

## Environment and editable install

Use a compatible Psi4 environment from [installation](installation.md), then:

```sh
git clone https://github.com/FoleyLab/qed-ci.git
cd qed-ci
python -m pip install --no-build-isolation --no-deps -e .
```

The editable install compiles `src/qed_ci/_native/cfunctions.so`. `src/` is added
to the import path by the editable install, so the public `qed_ci` API and the
documented legacy modules (`helper_PFCI`, `helper_cqed_rhf`, `nuclear_grad`)
import from any working directory without `PYTHONPATH`. If native dependencies
live in another prefix:

```sh
QED_BUILD_PREFIX=/path/to/openblas-or-mkl-prefix \
    python -m pip install --no-build-isolation --no-deps -e .
```

Rerun the editable install after changing C sources and restart Python; a running
process does not remap a replaced shared library.

## Native rebuilds

The same helper used by packaging is the supported low-level build:

```sh
python tools/build_backend.py                 # legacy src/cfunctions.so target
python tools/build_backend.py --output /tmp/cfunctions.so --prefix /path/to/env
python tools/build_backend.py --dry-run       # print commands only
```

macOS uses `/usr/bin/clang` (override with `--compiler`/`CC`), LP64
OpenBLAS/LAPACKE, and LLVM OpenMP. Linux uses `icx`, MKL, and Intel OpenMP; source
oneAPI `setvars.sh` first. The helper never writes an object file or shared library
in place. It compiles and load-checks in a temporary directory, backs up an
existing destination, then replaces it atomically after success. The manifest in
`build/backend/manifest.json` records the compiler, commands, dependency prefix,
source hashes, and library hash.

`src/cfunctions.so` is a build artifact and is not tracked. The source-checkout
fallback in `qed_ci.backend` finds it only when `src/ci_solver.c` is also present;
installed and editable packages prefer `_native`.

## Test suites

From the repository root:

```sh
python -m pytest -q                         # contract and scientific tests
python tools/check_docs.py                  # links and documented options
python -m pytest tests/test_packaging.py -q # metadata, sdist/wheel, installs, examples
```

The top-level `tests/` directory contains both focused interface tests and the
scientific numerical correctness suite. The latter deliberately ships in the
source distribution: QED electronic-structure methods do not yet have a single
widely accepted reference package, so the reference comparisons and regression
fixtures are part of the reproducibility record. Tests are not installed into
the runtime package or included in wheels. The focused contract tests are
`tests/test_config.py`, `tests/test_output.py`, and `tests/test_results.py`.
Stage 1 baseline replay is:

```sh
python tools/validate_stage23.py --output build/stage23/replay-new
```

Use one thread for reproducible comparisons:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Known historical baseline failures are recorded in
[Stage 1](audit/stage1/README.md) and [Stage 4](audit/stage4/README.md). The two
tests for an unfinished public FCI/CAS RDM contract and their `.npy` references
were retired when the suites were consolidated. One finite-difference scaling
test has shown an intermittent GLTR tridiagonal-solver failure that passed an
unchanged isolated rerun. Do not repair scientific failures by weakening
assertions or changing tolerances as part of packaging work.

## Deferred RDM API

QED-CI still builds internal density matrices needed by its numerical kernels,
but it does not yet promise a stable public RDM result contract. Future work
should define per-root one- and two-particle RDM access, orbital/spin indexing
and normalization conventions, photon-sector handling, energy reconstruction,
serialization boundaries, focused regression tests, and canonical examples.
Until that contract exists, large RDM tensors remain on calculation objects and
are not part of `CalculationResult`.

Packaging tests create isolated temporary virtual environments with
`--system-site-packages` to reuse Psi4/SciPy and install the wheel or editable
package there. They skip only when native build dependencies cannot be found; set
`QED_BUILD_PREFIX` in that case. They do not modify the user's active environment.

## Documentation

The user guide is plain Markdown with a single authoritative option reference:

- `docs/installation.md` — environment, installation, distribution limits,
  troubleshooting.
- `docs/methods.md` — supported methods, state averaging, gradients, limitations.
- `docs/configuration-and-output.md` — options/defaults/units/validation,
  migration, output, results, reports.
- `examples/quickstart/README.md` — runnable examples and expected fields.
- `tools/check_docs.py` — internal link and documented-option consistency check.

When adding or renaming a `CQEDConfig`, `CISolverOptions`, or `CASSCFOptions` field,
update `docs/configuration-and-output.md`; the docs check reads the dataclass
fields from `src/qed_ci/config.py` and fails if one is undocumented. Add runnable
coverage when the option changes user-visible behavior.

## Adding an option or result field

For a new user option:

1. Add the field and validation in `src/qed_ci/config.py`.
2. Map it through `to_legacy_options()` and, if it has a legacy dictionary name,
   `from_legacy()`; record conflicts in `diagnostics` instead of silently ignoring
   them.
3. Apply it in `helper_PFCI.py` or the relevant solver module. Do not change
   existing numerical defaults unless the change is separately authorized and
   validated.
4. Document defaults, units, allowed values, and validation in the authoritative
   option reference.
5. Add focused tests (lightweight validation tests must not import Psi4).

For a new result field, extend the versioned snapshot in
`src/qed_ci/results.py` (`build_result`) and, when appropriate,
`CalculationResult.summary()`. Keep large CI/RDM tensors on the calculation
object, use zero-based state indices, document units, and make unavailable values
`None` rather than fabricating them. Update the schema table in
`docs/configuration-and-output.md`. Presentation belongs in `qed_ci/report.py`;
scientific consumers should use `result.summary()`, not parse reports.

## Packaging and CI guidance

Lightweight jobs can run `tools/check_docs.py`, the metadata/docs tests, and the
focused configuration tests without Psi4 or a native toolchain. Jobs that import
the engine, run calculations, build the wheel/sdist, or execute the packaging
install tests require Psi4, the platform compiler, and BLAS/OpenMP. Keep those
separate so a packaging-only failure is not hidden behind a missing scientific
environment. Linux native execution remains unverified on the macOS development
machine; do not claim a Linux pass without running it there.
