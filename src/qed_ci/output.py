"""Scoped output routing for legacy Python, C stdio and Psi4.

The engine uses process-global state. Scopes serialize package calculations and
restore descriptors, streams and Psi4 settings even after exceptions. Other
application threads writing stdout during a scope are also redirected; use
separate processes for concurrent calculations. Text formatting is unchanged.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import threading

_LOCK = threading.RLock()
_LOCAL = threading.local()
_LIBC = ctypes.CDLL(None)
_LIBC.fflush.argtypes = [ctypes.c_void_p]
_LIBC.fflush.restype = ctypes.c_int


def flush():
    sys.stdout.flush()
    sys.stderr.flush()
    _LIBC.fflush(None)


@contextmanager
def calculation_output(config, *, append=False, backend=None):
    with _LOCK:
        # calculate() opens a scope before lazy import. The constructor reuses it.
        active = getattr(_LOCAL, 'config', None)
        if active is config:
            with native_level(config, backend):
                yield
            return
        _LOCAL.config = config
        prior_psi = None
        psi = None
        saved = []
        stream = None
        psi_log = None
        try:
            # Open output before changing process state; path failures are early.
            if config.output_file:
                stream = open(config.output_file, 'a' if append else config.output_mode, buffering=1)
            elif config.quiet:
                stream = open(os.devnull, 'w')
            flush()
            if stream is not None:
                saved = [os.dup(1), os.dup(2)]
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)
            # Psi4 has its own file manager. Capture separately to avoid two
            # independent writers truncating/overwriting a shared report file.
            psi = sys.modules.get('psi4')
            if psi is not None and (stream is not None or not config.legacy):
                prior_psi = psi.core.get_output_file()
                if config.quiet and not config.output_file:
                    psi.core.set_output_file(os.devnull, True)
                else:
                    fd, psi_log = tempfile.mkstemp(prefix='qed-ci-psi4-', suffix='.out')
                    os.close(fd)
                    psi.core.set_output_file(psi_log, False)
            with native_level(config, backend):
                if stream is None:
                    yield
                else:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        yield
        finally:
            try:
                flush()
                if prior_psi is not None:
                    # Append is essential: restoring an existing caller file
                    # must not truncate its earlier content.
                    psi.core.set_output_file(prior_psi, True)
                if psi_log is not None:
                    text = Path(psi_log).read_text(errors='replace')
                    if text:
                        destination = stream if stream is not None else sys.stdout
                        destination.write('\n--- Psi4 engine output ---\n' + text)
                        destination.flush()
            finally:
                for fd, original in zip((1, 2), saved):
                    os.dup2(original, fd)
                    os.close(original)
                if stream is not None:
                    stream.close()
                if psi_log is not None:
                    os.unlink(psi_log)
                _LOCAL.config = active


@contextmanager
def native_level(config, backend):
    if backend is None:
        yield
        return
    backend.get_ci_print_level.restype = ctypes.c_int
    backend.set_ci_print_level.argtypes = [ctypes.c_int]
    backend.set_ci_print_level.restype = None
    prior = backend.get_ci_print_level()
    level = {'normal': 1, 'debug': 2, 'trace': 3}[config.verbosity]
    if config.quiet and not config.output_file:
        level = 0
    # An existing legacy environment setting retains precedence over the
    # historical trace default; explicit new configs control the level.
    if config.legacy and config.verbosity == 'trace' and 'QED_PRINT_LEVEL' in os.environ:
        level = int(os.environ['QED_PRINT_LEVEL'])
    backend.set_ci_print_level(level)
    try:
        yield
    finally:
        backend.set_ci_print_level(prior)


@contextmanager
def workspace(owner):
    """Isolate unsolicited engine files; retain scratch while owner is alive."""
    config = owner.config
    if config.legacy and config.scratch_directory is None:
        yield
        return
    if not hasattr(owner, '_working_directory'):
        if config.scratch_directory:
            Path(config.scratch_directory).mkdir(parents=True, exist_ok=True)
            owner._working_directory = config.scratch_directory
        else:
            owner._scratch_handle = tempfile.TemporaryDirectory(prefix='qed-ci-')
            owner._working_directory = owner._scratch_handle.name
    previous = os.getcwd()
    os.chdir(owner._working_directory)
    try:
        yield
    finally:
        os.chdir(previous)
