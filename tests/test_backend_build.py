"""Native backend ABI checks; require tools/build_backend.py, but not Psi4."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / 'src/cfunctions.so'


@pytest.mark.parametrize('qr', ['0', '1'])
def test_lapack_row_major_eigenpairs(qr):
    # A fresh process is necessary: the eigensolver selection is cached in C.
    code = '''
import ctypes, sys
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
lib.symmetric_eigenvalue_problem.argtypes = [array, ctypes.c_int, array]
lib.symmetric_eigenvalue_problem.restype = None
original = np.array([[2., 1., -0.5], [1., 3., 0.25], [-0.5, 0.25, -1.]])
vectors = original.copy()
values = np.empty(3)
lib.symmetric_eigenvalue_problem(vectors, 3, values)
np.testing.assert_allclose(values, np.linalg.eigvalsh(original), atol=1e-12)
np.testing.assert_allclose(original @ vectors, vectors * values, atol=1e-12)
np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)
'''
    env = dict(os.environ, QED_DSYEV=qr, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable, '-c', code, str(LIBRARY)], env=env, check=True, timeout=30)


def test_memory_query_returns_kib_without_proc_on_macos():
    lib = ctypes.CDLL(str(LIBRARY))
    lib.set_ci_print_level.argtypes = [ctypes.c_int]
    lib.set_ci_print_level.restype = None
    lib.getMemory2.argtypes = [ctypes.POINTER(ctypes.c_ulong)] * 4
    lib.getMemory2.restype = None
    lib.set_ci_print_level(0)
    values = [ctypes.c_ulong(999) for _ in range(4)]
    lib.getMemory2(*[ctypes.byref(value) for value in values])
    if sys.platform == 'darwin' or Path('/proc/self/status').exists():
        assert values[0].value > 0
        assert values[1].value >= values[0].value
        assert values[2].value >= values[0].value
    if sys.platform == 'darwin':
        assert values[3].value == 0  # unavailable, not fabricated


def test_failed_compilation_preserves_installed_library():
    before = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    # A real executable that fails on every invocation, including compilation.
    compiler = shutil.which('false')
    assert compiler is not None
    result = subprocess.run([sys.executable, str(LIBRARY.parents[1] / 'tools/build_backend.py'),
                             '--compiler', compiler], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest() == before
