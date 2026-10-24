"""Locate the ctypes backend independently of the caller's working directory."""
import ctypes
from pathlib import Path


def backend_path():
    package = Path(__file__).resolve().parent
    installed = package / '_native' / 'cfunctions.so'
    if installed.is_file():
        return installed
    # Compatibility with the documented pre-packaging source build.
    source = package.parent / 'cfunctions.so'
    if source.is_file() and (package.parent / 'ci_solver.c').is_file():
        return source
    raise FileNotFoundError(
        'QED-CI native backend is missing. Activate the scientific environment and '
        'run python -m pip install --no-build-isolation --no-deps . '
        '(or -e . for development) from the QED-CI source directory. If the '
        'compiler libraries live in another prefix, set QED_BUILD_PREFIX first. '
        'See docs/installation.md for native dependencies.')


def load_backend():
    path = backend_path()
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise OSError(
            f'Cannot load QED-CI backend {path}: {exc}. Activate the environment '
            'used to build it (OpenBLAS/LLVM OpenMP on macOS; oneAPI MKL/Intel '
            'OpenMP on Linux), or reinstall from source in this environment.'
        ) from exc
