#!/usr/bin/env python3
"""Capture small, explicit legacy inputs and their outputs in isolated processes.

This is a baseline recorder, not a replacement calculation API. Each case gets
its own working directory, raw stdout/stderr, numerical arrays and metadata.
No solver settings or return values are modified. Run after build_backend.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MOLECULE = '0 1\nLi\nH 1 1.4\nsymmetry c1\nno_reorient\nno_com\nunits angstrom\n'
SCF = dict(basis='sto-3g', scf_type='pk', e_convergence=1e-10, d_convergence=1e-10)


def cases():
    result = {}
    for method in ('cis', 'fci', 'casci', 'casscf'):
        for cavity in (False, True):
            opts = dict(ci_level='cas' if method in ('casci', 'casscf') else method,
                        casscf_optimization=method == 'casscf',
                        omega_value=0.12 if cavity else 0.0,
                        lambda_vector=[0.0, 0.0, 0.05 if cavity else 0.0],
                        number_of_photons=1 if cavity else 0,
                        davidson_roots=2, davidson_threshold=1e-9,
                        coherent_state_basis=True, rdm_root=0)
            if method in ('casci', 'casscf'):
                opts.update(nact_els=2, nact_orbs=2, casscf_weight=[1, 1])
            result[f'{method}_' + ('cavity' if cavity else 'zero')] = opts
    return result


def child(case):
    import numpy as np
    import psi4
    import scipy
    import numba
    from helper_PFCI import PFHamiltonianGenerator
    psi4.set_num_threads(1)
    obj = PFHamiltonianGenerator(MOLECULE, SCF.copy(), case)
    names = ('CIeigs', 'CIvecs', 'eigenvals', 'eigenvecs', 'one_electron_rdm',
             'two_electron_rdm', 'D_tu_avg', 'D_tuvw_avg', 'Dpe_tu_avg', 'C',
             'opt_C', 'weight')
    arrays = {name: np.asarray(getattr(obj, name)) for name in names if hasattr(obj, name)}
    np.savez('arrays.npz', **arrays)
    summary = dict(versions=dict(python=sys.version, psi4=psi4.__version__, numpy=np.__version__,
                                 scipy=scipy.__version__, numba=numba.__version__),
                   shapes={k: list(v.shape) for k, v in arrays.items()},
                   energies={k: arrays[k].tolist() for k in ('CIeigs', 'eigenvals') if k in arrays},
                   resolved={k: getattr(obj, k) for k in ('ci_level', 'casscf_optimization',
                             'canonical_mos', 'photon_number_basis', 'coherent_state_basis',
                             'N_p', 'n_act_orb', 'n_act_el')})
    summary['finite_arrays'] = all(bool(np.isfinite(v).all()) for v in arrays.values())
    Path('result.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/stage1/baselines')
    parser.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    if args.child:
        child(json.loads(args.child.read_text())['cavity_options'])
        return
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QED_', 'CPP_CASSCF_'))}
    env.update(PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    manifest = dict(source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    library_sha256=hashlib.sha256((ROOT / 'src/cfunctions.so').read_bytes()).hexdigest(),
                    environment={k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    note='QED_* and CPP_CASSCF_* removed from child environment; fresh process per case. '
                         'CIeigs and eigenvals both retained: legacy objects may keep initial and optimized energies separately.',
                    runs=[])
    for name, opts in cases().items():
        for repeat in range(args.repeat):
            directory = out / f'{name}_{repeat+1}'
            directory.mkdir(exist_ok=False)
            input_path = directory / 'input.json'
            input_path.write_text(json.dumps(dict(molecule=MOLECULE, psi4_options=SCF, cavity_options=opts), indent=2) + '\n')
            started = time.monotonic()
            with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                try:
                    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(input_path)],
                                       cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=300)
                    status = p.returncode
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            manifest['runs'].append(dict(case=name, repeat=repeat+1, status=status,
                seconds=time.monotonic()-started, files=[dict(name=p.name, bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(directory.iterdir()) if p.is_file()]))
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'{directory.name}: {status}', flush=True)
    if any(r['status'] != 0 for r in manifest['runs']):
        sys.exit(1)


if __name__ == '__main__':
    main()
