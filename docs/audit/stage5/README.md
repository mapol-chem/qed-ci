# Stage 5: packaging and documentation validation

Scope: conventional Python packaging, installation, user documentation, runnable
examples, and packaging/install/documentation checks. Stage 6 was not started.
No scientific algorithm, numerical default, convergence threshold, supported
reference type, Calculator dispatcher, cqed-scf reference provider, or C++/TAMM
code was changed.

The machine-readable record is [results.json](results.json). Raw logs and build
artifacts are under the ignored `build/stage5/` directory.

## Supported workflows and what was tested

Stage 5 used the existing native arm64 macOS setup: Python 3.12.13, Psi4 1.10.1,
NumPy 2.4.6, SciPy 1.17.1, Numba 0.66.0, Apple Clang, LP64 OpenBLAS/LAPACKE, and
LLVM OpenMP. The validation Python environment did not contain the LAPACKE
headers, so native compilation used the supported `QED_BUILD_PREFIX` override to
select a local prefix containing OpenBLAS/LAPACKE and LLVM OpenMP. The package was
still installed and executed with Python 3.12.

Tested successfully:

1. **Wheel build and install.** `pip wheel --no-build-isolation --no-deps`
   produced `qed_ci-0.1.0-py3-none-macosx_15_0_arm64.whl`. The tag is platform
   specific (`Root-Is-Purelib: false`), and the wheel contains
   `qed_ci/_native/cfunctions.so`, not an `any` wheel.
2. **Wheel execution outside the checkout.** A temporary venv with
   `--system-site-packages` installed the wheel. From an unrelated working
   directory, `qed_ci`, `helper_PFCI`, `helper_cqed_rhf`, and `nuclear_grad`
   imported, the backend resolved to the installed `_native/cfunctions.so`, and a
   LiH/STO-3G CASCI calculation returned the expected energies.
3. **Editable install.** `pip install --no-build-isolation --no-deps -e .`
   compiled into `src/qed_ci/_native/cfunctions.so`; an unrelated working
   directory imported the source package and ran the same small CASCI calculation.
4. **Source distribution.** The sdist contains selected C/Python sources, the
   native build helper, user docs, examples, and packaging tests. It excludes
   prebuilt libraries, objects, fixtures, run outputs, the generated egg-info, and
   the C++/TAMM tree. Installing the sdist in a clean venv built the native
   library and ran the same calculation.
5. **Historical source checkout.** The existing `python tools/build_backend.py`
   plus `PYTHONPATH=src` path remains documented and supported for compatibility.
   It is not the preferred installation path.

Linux behavior was inspected only. No suitable Linux execution environment was
available here, so Linux compilation and runtime behavior remain **unverified**.
Windows remains unsupported.

## Documentation and examples

Added or revised:

- Root `README.md`: implemented capabilities, limitations, requirements,
  installation, first calculation, and a documentation map.
- `docs/README.md`: task-oriented user-guide index.
- `docs/installation.md`: validated environment, macOS/Linux setup, native prefix
  selection, wheel/sdist/editable installs, source-checkout compatibility,
  distribution limits, and build/install/convergence/input troubleshooting.
- `docs/methods.md`: CIS/FCI/CASCI/CASSCF capabilities, reference/orbital/basis
  restrictions, active spaces and roots, state averaging, analytic gradient/NAC
  restrictions, and future-integration boundaries.
- `docs/configuration-and-output.md`: now the single authoritative option
  reference, with units, allowed values, validation, precedence, legacy
  migration, output/file behavior, and the version-1 result/report contract.
- `docs/development.md`: editable setup, native rebuilds, testing, packaging
  checks, CI guidance, and how to add options/result fields.
- `examples/quickstart/README.md`: quiet/file-output commands, CASCI versus
  CASSCF, state averaging, gradient use, expected fields/units, and cost notes.
- `src/README.md` and `CLAUDE.md`: removed the stale claim that `PYTHONPATH` is
  always required, while retaining the compatibility workflow.

The examples use the public `calculate()`/`CQEDConfig` API and
`result.summary()`/structured properties. All example entry points were run:
`methods.py cis|fci|casci|casscf|sa` and `gradient.py` returned status 0. The
gradient example's `result.properties['gradient_state_0']['value']` is in Eh/bohr.

Documentation tooling remains plain Markdown; no large framework was introduced.
`tools/check_docs.py` verifies relative links and every public configuration field
against the authoritative option reference. It checked 36 links and 48 fields.

## Packaging implementation notes

- `pyproject.toml` is the source of package metadata, dependencies, optional dev
  tools, package/module lists, and pytest configuration. `pytest.ini` was removed
  to avoid duplicate configuration.
- `setup.py` is limited to the ctypes backend build bridge and wheel/sdist
  behavior. It reuses `tools/build_backend.py`, creates the editable `_native`
  destination, tags wheels `py3-none-<platform>`, and removes generated egg-info
  from the sdist release tree.
- `MANIFEST.in` controls sdist contents and excludes compiled objects, fixtures,
  run outputs, build trees, and the separate C++ port.
- Native discovery is package-relative through `qed_ci.backend`; the historical
  `src/cfunctions.so` fallback remains for source checkouts.
- Dependencies are declared with lower bounds; Psi4 is deliberately not a pip
  dependency. Only the exact macOS versions above were exercised for packaging.

## Tests and recorded outcomes

- **Focused tests: 65 passed, 0 failed** (`build/stage5/focused.xml`), including
  the new packaging, isolated wheel/editable install, docs, and example checks.
- **Scientific suite: 175 passed, 2 failed, 177 collected**
  (`build/stage5/scientific.xml`). The failures are the two unchanged Stage 1
  baseline issues: a hardcoded Linux fixture path and the absent
  `total_energy_from_rdms` attribute. The intermittent GLTR finite-difference
  scaling failure recorded in Stage 4 passed on its rerun here. No new failure is
  attributed to Stage 5, and no assertion or tolerance was weakened.
- **Baseline replay: 16/16 passed**, maximum energy difference `3.38e-12` Eh and
  maximum saved RDM/weight difference `8.73e-9`, within the established `1e-8`
  Eh and `1e-7` tolerances (`build/stage5/replay/`).
- `tools/check_docs.py`, `git diff --check`, and root-level pytest collection
  (242 tests) passed.
- Tracked run artifacts (`output.dat`, `orbital2.out`, and ancillary logs) that
  the scientific suite rewrites were restored. The pre-existing user modification
  to `timer.dat` was preserved; subsequent validation-run timer blocks were removed.

## Compatibility and remaining limitations

- Legacy `helper_PFCI`/`helper_cqed_rhf`/`nuclear_grad` imports and the
  three-positional-argument constructor remain supported. The new package does not
  require `PYTHONPATH`.
- Wheels are local platform/runtime artifacts: they do not bundle BLAS, OpenMP,
  MKL, compiler runtimes, or Psi4, and the macOS install name/search path refers to
  the build environment. Do not distribute a wheel to machines without matching
  libraries; rebuild from source there.
- Linux native execution is not validated. No manylinux or universal macOS wheel
  is claimed.
- Dependency lower bounds are declarations, not a validated version matrix.
- Output, Psi4, and working-directory routing remain process-global; Psi4 may
  write `timer.dat` at process finalization outside the calculation scope.
- Stage 6, a Calculator dispatcher, cqed-scf reference-wavefunction integration,
  and TAMM work remain out of scope and have not begun.
