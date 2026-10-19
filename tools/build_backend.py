#!/usr/bin/env python3
"""Build the Python ctypes backend with Apple Clang/OpenBLAS or Intel icx/MKL.

Run with the Python interpreter from the calculation environment. Existing
objects are never touched; the library is replaced only after a load check.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path(sys.prefix),
                        help='dependency prefix (default: this Python environment)')
    parser.add_argument('--compiler', help='C compiler executable')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    system = platform.system()
    prefix = args.prefix.resolve()
    env = os.environ.copy()
    if system == 'Darwin':
        compiler = args.compiler or '/usr/bin/clang'
        required = ['include/cblas.h', 'include/lapacke.h', 'include/omp.h',
                    'lib/libopenblas.dylib', 'lib/libomp.dylib']
        missing = [str(prefix / p) for p in required if not (prefix / p).exists()]
        if missing:
            parser.error('Missing dependencies: ' + ', '.join(missing) +
                         '\nInstall conda-forge openblas liblapacke llvm-openmp in this environment.')
        flags = ['-DQED_USE_OPENBLAS', '-I' + str(prefix / 'include'),
                 '-Xpreprocessor', '-fopenmp']
        link = ['-dynamiclib', '-Wl,-install_name,@rpath/cfunctions.so', '-L' + str(prefix / 'lib'),
                '-Wl,-rpath,' + str(prefix / 'lib'), '-lopenblas', '-lomp']
        backend = 'OpenBLAS LP64'
    elif system == 'Linux':
        compiler = args.compiler or 'icx'
        mklroot = os.environ.get('MKLROOT')
        if not mklroot or not (Path(mklroot) / 'include/mkl.h').exists():
            parser.error('Source Intel oneAPI setvars.sh first (MKLROOT must contain include/mkl.h).')
        flags = ['-qopenmp', '-I' + str(Path(mklroot) / 'include')]
        link = ['-shared', '-qmkl=parallel', '-qopenmp']
        # Avoid conda binutils shadowing the system linker (see CLAUDE.md).
        compiler = shutil.which(compiler)
        env['PATH'] = '/usr/bin:' + env.get('PATH', '')
        backend = 'Intel MKL LP64 (Intel OpenMP)'
    else:
        parser.error('Supported builds: macOS with OpenBLAS; Linux with Intel oneAPI.')
    if not compiler or not shutil.which(compiler):
        parser.error('Compiler not found; install Xcode Command Line Tools or source oneAPI setvars.sh.')
    compiler = shutil.which(compiler)
    base = [compiler, '-O2', '-fPIC', '-std=gnu11', '-Wall', '-Wextra']
    build_dir = ROOT / 'build' / 'backend'
    build_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix='compile-', dir=build_dir) as tmp:
        tmp = Path(tmp)
        objects = []
        for name in ('ci_solver', 'orbital'):
            obj = tmp / (name + '.o')
            objects.append(str(obj))
            commands.append(base + flags + ['-c', str(ROOT / 'src' / (name + '.c')), '-o', str(obj)])
        library = tmp / 'cfunctions.so'
        commands.append([compiler] + objects + link + ['-o', str(library)])
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, env=env)
        if args.dry_run:
            return
        # Check in a child: do not keep a dlopen handle to a replaced library.
        subprocess.run([sys.executable, '-c',
                        'import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); '
                        '[getattr(lib,n) for n in ("get_roots","get_graph",'
                        '"getMemory2","full_transformation_macroiteration")]',
                        str(library)], check=True, env=env)
        target = ROOT / 'src' / 'cfunctions.so'
        if target.exists():
            stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
            shutil.copy2(target, build_dir / ('cfunctions-' + stamp + '.so.bak'))
        os.replace(library, target)
        sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT / 'src' / n for n in
                             ('ci_solver.c', 'ci_solver.h', 'orbital.c', 'orbital.h', 'qed_blas.h')]}
        manifest = dict(platform=platform.platform(), architecture=platform.machine(),
                        python=sys.version, prefix=str(prefix), backend=backend,
                        compiler=subprocess.check_output([compiler, '--version'], text=True),
                        commands=commands, source_sha256=sources,
                        library_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        (build_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Built and load-checked ' + str(target))
        print('Build provenance: ' + str(build_dir / 'manifest.json'))


if __name__ == '__main__':
    main()
