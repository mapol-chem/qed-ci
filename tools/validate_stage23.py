#!/usr/bin/env python3
"""Replay Stage 1 inputs through both supported entry paths against saved energies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def child(name, api):
    import numpy as np
    from qed_ci import CQEDConfig, calculate
    data=json.loads((ROOT/'docs/audit/stage1/baselines.json').read_text())[name]
    inputs=data['input']
    if api=='legacy':
        from helper_PFCI import PFHamiltonianGenerator
        opts=dict(inputs['cavity_options'], quiet=True)
        obj=PFHamiltonianGenerator(inputs['molecule'],inputs['psi4_options'],opts)
    else:
        config=CQEDConfig.from_legacy(inputs['psi4_options'],inputs['cavity_options'])
        # New API explicitly exposes the fixed two-sector CIS implementation.
        config=config.copy_with(legacy=False,_legacy_options=None,quiet=True,verbosity='trace',
                                max_photons=1 if config.method=='cis' else config.max_photons)
        obj=calculate(inputs['molecule'],config)
    result = obj.result.summary()
    np.testing.assert_array_equal(result['energies']['initial_ci'], obj.CIeigs)
    np.testing.assert_array_equal(result['energies']['final'], getattr(obj, 'eigenvals', obj.CIeigs))
    if inputs['cavity_options'].get('casscf_optimization'):
        assert result['energies']['final_source'] == 'eigenvals'
        assert not np.allclose(result['energies']['initial_ci'], result['energies']['final'], rtol=0, atol=1e-6)
    differences={}
    for attr,expected in data['result']['energies'].items():
        actual=np.asarray(getattr(obj,attr))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
        differences[attr]=float(np.max(np.abs(actual-expected)))
    rdm_differences={}
    with np.load(ROOT/'docs/audit/stage1'/data['reference_arrays']) as reference:
        for key in reference.files:
            actual=np.asarray(getattr(obj,key))
            assert actual.shape==reference[key].shape,(key,actual.shape,reference[key].shape)
            assert np.isfinite(actual).all(),key
            if key in {'D_tu_avg','D_tuvw_avg','Dpe_tu_avg','one_electron_rdm','two_electron_rdm','weight'}:
                np.testing.assert_allclose(actual,reference[key],rtol=0,atol=1e-7)
                rdm_differences[key]=float(np.max(np.abs(actual-reference[key])))
    Path('comparison.json').write_text(json.dumps(dict(case=name,api=api,energy_max_abs=differences,
                                                        rdm_max_abs=rdm_differences,
                                                        resolved_options=obj.resolved_options, result=result),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',nargs=2,metavar=('CASE','API'))
    p.add_argument('--output',type=Path,default=ROOT/'build/stage23/replay')
    args=p.parse_args()
    if args.child:
        child(*args.child)
        return
    data=json.loads((ROOT/'docs/audit/stage1/baselines.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.startswith(('QED_','CPP_CASSCF_'))}
    env.update(PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    runs=[]
    for name in data:
        for api in ('legacy','config'):
            directory=args.output.resolve()/f'{name}_{api}'
            directory.mkdir(parents=True,exist_ok=False)
            with (directory/'stdout.log').open('w') as out,(directory/'stderr.log').open('w') as err:
                try:
                    r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child',name,api],
                                     cwd=directory,env=env,stdout=out,stderr=err,timeout=300)
                    status=r.returncode
                except subprocess.TimeoutExpired:
                    status='timeout'
            runs.append(dict(case=name,api=api,status=status))
            (args.output/'summary.json').write_text(json.dumps(runs,indent=2)+'\n')
            print(name,api,status,flush=True)
    if any(r['status']!=0 for r in runs):
        sys.exit(1)


if __name__=='__main__':
    main()
