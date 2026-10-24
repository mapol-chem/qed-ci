# Installation and troubleshooting

QED-CI installs into an existing compatible scientific Python environment. The
Python package installation compiles the ctypes CI backend from the bundled C
sources. It does **not** install Psi4, a C compiler, BLAS/LAPACK, or OpenMP. Use
the same environment for building and running unless you deliberately select a
separate native dependency prefix.

## Validated environment

Stage 5 packaging validation used native arm64 macOS with:

| Component | Validated version |
|---|---|
| Python | 3.12.13 |
| Psi4 | 1.10.1 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Numba | 0.66.0 |
| psutil | 7.2.2 |
| Native backend | Apple Clang, LP64 OpenBLAS/LAPACKE, LLVM OpenMP |

Stage 1 also recorded a Python 3.13/Psi4 1.10 run. Metadata admits Python
3.12–3.13; only 3.12 was revalidated for packaging in Stage 5. Dependency lower
bounds in `pyproject.toml` are declarations, not a tested version matrix. NumPy,
SciPy, Numba, llvmlite, and psutil should come from the scientific environment
rather than being force-upgraded by QED-CI. No broader platform or version matrix
is claimed. Archive `conda list --explicit` and `python -m pip freeze` for a
reproducible environment.

## Environment setup

### macOS (native arm64)

Install Xcode Command Line Tools (`xcode-select --install`) and Miniforge, then:

```sh
conda create -n qed-ci -c conda-forge python=3.12 psi4=1.10.1 \
    numpy scipy numba psutil openblas liblapacke llvm-openmp \
    pip setuptools wheel
conda activate qed-ci
```

The build uses Apple Clang, LP64 OpenBLAS/LAPACKE, and LLVM OpenMP. All components
must match Python's architecture; do not mix Rosetta/x86 libraries with an arm64
Python. Intel MKL is not used on macOS.

### Linux (Intel oneAPI)

Install the Intel oneAPI/HPC kit with `icx`, MKL, and Intel OpenMP, create an
equivalent Psi4 environment, and initialize your installed oneAPI:

```sh
conda create -n qed-ci -c conda-forge python=3.12 psi4 \
    numpy scipy numba psutil pip setuptools wheel
conda activate qed-ci
source /opt/intel/oneapi/setvars.sh
```

Adjust the `setvars.sh` path for your installation. `MKLROOT/include/mkl.h` and
`icx` must exist. The helper selects LP64 MKL and puts the system linker ahead of
conda's binutils. Keep the matching Intel runtime libraries available at execution
time, including in cluster jobs. Linux commands were inspected but not executed
in Stage 5; Linux execution remains unverified. Windows is unsupported.

### Compiler and library prefixes

The build defaults to the Python environment's prefix (`sys.base_prefix`, so a
virtual environment normally points at its base scientific environment). If the
compiler headers and libraries live elsewhere, select that prefix explicitly:

```sh
export QED_BUILD_PREFIX=/path/to/conda/env/with/openblas-or-mkl
```

The low-level helper also accepts `--prefix PATH`, `--compiler PATH`,
`--output PATH`, and `--dry-run`:

```sh
python tools/build_backend.py --help
```

Do not mix a Python environment with a native prefix of a different architecture.
On macOS the built library records the selected prefix's library search path; keep
that environment available when running the library.

## Install the Python package

From the checkout or an unpacked source distribution:

```sh
python -m pip install --no-build-isolation --no-deps .
```

For development and live source edits:

```sh
python -m pip install --no-build-isolation --no-deps -e .
```

`--no-build-isolation` avoids downloading build tools and makes the native
dependency environment explicit. `--no-deps` preserves your existing scientific
environment; ensure NumPy, SciPy, Numba/llvmlite, and psutil are present. Psi4 is
supplied by conda and is intentionally not a pip dependency. If you omit
`--no-deps`, pip may resolve the declared Python dependencies but still cannot
supply Psi4. The build requires setuptools >= 77 and wheel.

A venv created with `--system-site-packages` can reuse a scientific environment;
the native build still defaults to the base prefix. Set `QED_BUILD_PREFIX` if that
is not where the native libraries live.

After installation the public package works from any working directory without
manual `PYTHONPATH`:

```python
from qed_ci import CQEDConfig, calculate

job = calculate(
    "Li\nH 1 1.4\nsymmetry c1",
    CQEDConfig(method="fci", n_roots=2, max_photons=0, quiet=True,
               psi4_options={"basis": "sto-3g"}),
)
print(job.result.energies["final"])  # Eh
```

The public API loads the engine lazily. The documented legacy imports
`helper_PFCI`, `helper_cqed_rhf`, and `nuclear_grad` are also installed as
top-level modules, along with their internal support modules. Standalone scan
drivers, quarantined `legacy_python_ci`, and the optional Psi4 DPD
shim are not distributed. Use `CQEDConfig(method="cis", ...)` rather than the
historical `helper_cs_cqed_cis` module, which is absent from this checkout.

## Source-checkout compatibility

The pre-packaging workflow remains supported for local development:

```sh
python tools/build_backend.py                 # writes src/cfunctions.so
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

This is a compatibility path, not the recommended installation. If both a source
checkout and an installed package are importable, Python's search order decides
which one wins; a leftover `PYTHONPATH` can hide a just-installed wheel or its
native library. Unset `PYTHONPATH` when testing an installation, and rerun the
editable install after rebuilding its `_native` library.

## Native builds, wheels, and distribution limits

Installation builds `qed_ci/_native/cfunctions.so` relative to the package (the
suffix is `.so` on macOS as well). Discovery is package-relative and never depends
on the caller's working directory. A source checkout can still use the historical
`src/cfunctions.so`; installed and editable packages prefer `_native`.

For an editable native rebuild, rerun the editable installation and restart
Python after replacing a loaded library. The build helper compiles in a temporary
directory, load-checks the result, backs up an existing library, and only then
replaces it. Provenance is written to `build/backend/manifest.json`.

Build a wheel or source distribution without replacing your environment:

```sh
python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m build --sdist --no-isolation
```

The second command requires the optional `build` frontend (`pip install build`).
Source distributions include the selected C/Python sources, user documentation,
runnable examples, the native build helper, and packaging checks. They never
include prebuilt libraries, object files, numerical fixtures, run outputs, or
machine-specific build manifests.

Wheels are tagged `py3-none-<platform>`, never `py3-none-any`: ctypes does not
impose a CPython extension ABI, but the library is platform- and runtime-specific.
They are **local environment wheels**, not portable binary releases. They do not
bundle BLAS, OpenMP, MKL, Psi4, or the compiler runtime. macOS native search paths
refer to the build environment. Do not relocate that environment or distribute
the wheel to machines without matching runtime libraries; rebuild from source
there instead. No manylinux certification, universal macOS wheel, or runtime
repair is claimed.

## Troubleshooting

### Build or installation

- **Missing compiler/header/library.** The helper names the missing path. Verify
  the selected prefix (`QED_BUILD_PREFIX`, `--prefix`) and that its architecture
  matches Python. On Linux, source `setvars.sh` so `MKLROOT` and `icx` resolve.
- **A failed build deleted or changed the working library.** It should not: the
  helper compiles to temporary outputs and replaces the installed library only
  after a successful compile and load check. Keep the reported `build/backend`
  logs and manifest when reporting a failure.
- **`Cannot load QED-CI backend`.** Activate the environment used to build the
  library, or reinstall from source in the current environment. On Linux also
  ensure the matching Intel compiler runtime is on the loader path. On macOS the
  prefix's `lib` directory must still exist.
- **Missing `psi4`.** Activate the scientific environment; pip does not install
  Psi4.
- **An old module or library is being imported.** Check `qed_ci.__file__` and
  `qed_ci.backend.backend_path()`. Unset `PYTHONPATH`, remove stale editable
  installs, and reinstall. An editable `_native` library takes precedence over a
  historical `src/cfunctions.so`.
- **Native rebuild appears ineffective.** Rerun `pip install -e .` and restart
  Python. A process that has already loaded a replaced shared library keeps the
  old mapping.

### Input and convergence

- **Input validation errors.** Unknown keys, unsupported methods/references,
  invalid active spaces, bad weights, and impossible root counts fail before CI
  work. See the [authoritative configuration reference](configuration-and-output.md).
- **Nonconvergence.** Inspect `result.convergence` and `result.warnings`; use
  `verbosity="debug"` or `"trace"` with an output file for engine diagnostics.
  Review the active space, root count, CI residual threshold, iteration budgets,
  and CASSCF thresholds. A returned energy alone does not establish convergence.
  Do not mask failures by loosening scientific assertions.
- **Unexpected console/file behavior.** Quiet output and file redirection cover
  package Python/C/Psi4 output inside the calculation scope, but Psi4's
  process-finalization `timer.dat` can still appear in the process exit directory.
  Use a disposable working directory or an explicit `scratch_directory`.
- **Concurrent jobs overwrite each other.** Output, Psi4, and working-directory
  routing are process-global. Run independent calculations in separate processes
  and give each its own report/scratch paths.
