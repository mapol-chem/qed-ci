"""Descriptor-level output, restoration, and real calculation integration."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qed_ci import CQEDConfig
from qed_ci.output import calculation_output


SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'src'


def source_environment(**updates):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get('PYTHONPATH'))))
    environment.update(updates)
    return environment


def emit():
    print('python-output',flush=True)
    os.write(2,b'stderr-output\n')
    libc=ctypes.CDLL(None)
    libc.puts(b'native-output')


def test_quiet_covers_python_and_native_descriptors(capfd):
    with calculation_output(CQEDConfig(quiet=True)):
        emit()
    captured=capfd.readouterr()
    assert captured.out==captured.err==''
    print('restored')
    assert capfd.readouterr().out=='restored\n'


def test_file_only_and_exception_restoration(tmp_path,capfd):
    path=tmp_path/'job.out'
    with pytest.raises(RuntimeError):
        with calculation_output(CQEDConfig(output_file=str(path),quiet=True)):
            emit()
            raise RuntimeError('test')
    assert capfd.readouterr().out==''
    text=path.read_text()
    assert all(word in text for word in ('python-output','stderr-output','native-output'))
    print('restored')
    assert capfd.readouterr().out=='restored\n'


def test_append_and_exclusive_modes(tmp_path):
    path=tmp_path/'job.out'
    path.write_text('existing\n')
    with calculation_output(CQEDConfig(output_file=str(path),output_mode='a')):
        print('next')
    assert path.read_text()=='existing\nnext\n'
    with pytest.raises(FileExistsError):
        with calculation_output(CQEDConfig(output_file=str(path),output_mode='x')):
            pass
    assert path.read_text().startswith('existing')


def test_real_silent_file_and_restoration(tmp_path):
    # Child includes Psi4 imports in the captured descriptors, avoiding pytest's
    # own capture mechanics hiding a native stdout leak.
    code='''
import json, pathlib, numpy as np
from qed_ci import CQEDConfig, CISolverOptions, calculate
base=CQEDConfig(method='casci', active_electrons=2, active_orbitals=2,
    n_roots=2, max_photons=0, psi4_options={'basis':'sto-3g','e_convergence':1e-10,'d_convergence':1e-10},
    ci_solver=CISolverOptions(threshold=1e-9),quiet=True)
a=calculate('Li\\nH 1 1.4\\nsymmetry c1',base)
import psi4
prior=pathlib.Path('caller.out').resolve()
psi4.core.set_output_file(str(prior),False)
psi4.core.print_out('before\\n')
b=calculate('Li\\nH 1 1.4\\nsymmetry c1',base.copy_with(output_file='job.out'))
assert psi4.core.get_output_file()==str(prior)
psi4.core.print_out('after\\n')
psi4.core.set_output_file('stdout',True)
assert 'before' in prior.read_text() and 'after' in prior.read_text()
np.testing.assert_allclose(a.CIeigs,b.CIeigs,rtol=0,atol=1e-10)
assert 'weight' in pathlib.Path('job.out').read_text()
assert not pathlib.Path('orbital2.out').exists()
assert not pathlib.Path('orbital.out').exists()
pathlib.Path('done.json').write_text(json.dumps(a.CIeigs.tolist()))
'''
    env=source_environment(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr+r.stdout
    assert r.stdout==r.stderr==''
    assert (tmp_path/'done.json').exists()


def test_gradient_appends_to_same_report(tmp_path):
    code='''
from pathlib import Path
from qed_ci import CQEDConfig, CISolverOptions
from qed_ci.output import calculation_output
c=CQEDConfig(method='casscf',active_electrons=2,active_orbitals=2,n_roots=2,
    max_photons=0,psi4_options={'basis':'sto-3g','e_convergence':1e-10,'d_convergence':1e-10},
    ci_solver=CISolverOptions(threshold=1e-9),quiet=True,output_file='gradient.out')
with calculation_output(c):
    from nuclear_grad import nuclear_grad
    obj=nuclear_grad('Li\\nH 1 1.4\\nsymmetry c1',config=c)
before=Path('gradient.out').read_text()
obj.compute_grad(0,0)
after=Path('gradient.out').read_text()
assert after.startswith(before)
assert 'gradient_state_0 (Eh/bohr)' in after[len(before):]
assert not Path('orbital2.out').exists()
'''
    env=source_environment(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=180)
    assert r.returncode==0,r.stdout+r.stderr+(tmp_path/'gradient.out').read_text()[-1500:]
    assert r.stdout==r.stderr==''


def test_normal_engine_retains_native_warnings_and_live_progress(tmp_path, capfd):
    from qed_ci.output import engine_output, progress
    path = tmp_path / 'report.out'
    config = CQEDConfig(output_file=str(path))
    diagnostics = []

    class Backend:
        level = 3
        def get_ci_print_level(self):
            return self.level
        def set_ci_print_level(self, value):
            self.level = value

    backend = Backend()
    with calculation_output(config):
        with pytest.raises(RuntimeError), engine_output(config, backend, diagnostics):
            assert backend.level == 1  # Native spin warnings must remain enabled.
            print('large routine diagnostic hidden')
            print('state      eigenvalues      total energies       error')
            ctypes.CDLL(None).puts(b'*** WARNING: CI root 0 spin contamination')
            progress('CASSCF macro 0')
            # Read before leaving the engine: this must be live, not replayed.
            assert 'CASSCF macro 0' in path.read_text()
            raise RuntimeError('deliberate')
    assert backend.level == 3
    assert diagnostics == ['*** WARNING: CI root 0 spin contamination']
    text = path.read_text()
    assert 'spin contamination' in text
    assert 'large routine' not in text
    assert 'total energies' not in text
    assert capfd.readouterr().out == ''
