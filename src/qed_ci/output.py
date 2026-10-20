"""Scoped output routing for legacy Python, C stdio and Psi4.

The engine uses process-global state. Scopes serialize package calculations and
restore descriptors, streams and Psi4 settings even after exceptions. Other
application threads writing stdout during a scope are also redirected; use
separate processes for concurrent calculations.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import ctypes
import os
import re
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


def progress(message):
    """Write a compact event through the report destination, even inside the engine."""
    stream = getattr(_LOCAL, 'progress_stream', None)
    if stream is not None:
        print('  ' + message, file=stream, flush=True)


@contextmanager
def engine_output(config, backend=None, diagnostics=None):
    """Hide routine legacy text while retaining warning/error lines.

    Normal engine text is spooled to temporary files, never accumulated in
    memory. Native verbosity stays at Normal so spin warnings are computed.
    Scientific results still come directly from engine arrays, not this text.
    """
    if config.verbosity != 'normal':
        yield
        return
    psi = sys.modules.get('psi4')
    previous = psi.core.get_output_file() if psi is not None else None
    saved = []
    level = backend.get_ci_print_level() if backend is not None else None
    previous_progress = getattr(_LOCAL, 'progress_stream', None)
    # A numerical table's "error" column is not an error diagnostic.
    marker = re.compile(r'\b(warning|failed|failure|fallback)\b|\berror\s*[:!]|^error\b', re.I)
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8', errors='replace') as sink, \
            tempfile.NamedTemporaryFile(mode='w+', suffix='.out', encoding='utf-8',
                                        errors='replace') as psi_sink:
        try:
            flush()
            saved = [os.dup(1), os.dup(2)]
            _LOCAL.progress_stream = os.fdopen(os.dup(1), 'w', buffering=1)
            for fd in (1, 2):
                os.dup2(sink.fileno(), fd)
            if psi is not None:
                psi.core.set_output_file(psi_sink.name, False)
            if backend is not None:
                backend.set_ci_print_level(1)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            flush()
            if backend is not None:
                backend.set_ci_print_level(level)
            if previous is not None:
                psi.core.set_output_file(previous, True)
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)
            stream = getattr(_LOCAL, 'progress_stream', None)
            _LOCAL.progress_stream = previous_progress
            if stream is not None and stream is not previous_progress:
                stream.close()
            seen = set()
            for captured in (sink, psi_sink):
                captured.seek(0)
                for line in captured:
                    message = line.strip()
                    if marker.search(message) and message not in seen:
                        seen.add(message)
                        if diagnostics is not None and message not in diagnostics:
                            diagnostics.append(message)
                        print('  Engine diagnostic: ' + message, flush=True)
