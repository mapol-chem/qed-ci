"""Scientific result contracts and focused report assertions, without log snapshots."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qed_ci import CalculationResult


def test_summary_unknown_and_detached():
    result = CalculationResult({}, {}, {}, {}, {'ci_final': {'converged': None}}, {})
    assert result.converged is None
    summary = result.summary()
    assert summary['status'] == 'unknown'
    summary['requested']['changed'] = True
    assert result.requested == {}
    json.dumps(summary, allow_nan=False)
    result.convergence['ci_final']['converged'] = False
    assert result.converged is False


BASE = '''
import json, pathlib, numpy as np
from qed_ci import CQEDConfig, CISolverOptions, CASSCFOptions, calculate
geom='Li\\nH 1 1.4\\nsymmetry c1\\nno_reorient\\nno_com'
config=CQEDConfig(method='casscf', active_electrons=2, active_orbitals=2,
    n_roots=2, max_photons=0, psi4_options={'basis':'sto-3g','scf_type':'pk',
    'e_convergence':1e-10,'d_convergence':1e-10},ci_solver=CISolverOptions(threshold=1e-9))
'''


def run(code, tmp_path, **env):
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.update(env)
    process = subprocess.run([sys.executable, '-B', '-c', BASE + code], cwd=tmp_path,
                             env=environment, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout[-3000:] + process.stderr[-3000:]
    return process


@pytest.mark.parametrize('verbosity', ['normal', 'debug', 'trace'])
def test_report_and_scientific_contract(tmp_path, verbosity):
    proc = run(f'''
obj=calculate(geom,config.copy_with(verbosity={verbosity!r},output_file='job.out',state_weights=(3.,1.)))
r=obj.result.summary()
json.dumps(r,allow_nan=False)
np.testing.assert_array_equal(r['energies']['initial_ci'],obj.CIeigs)
np.testing.assert_array_equal(r['energies']['final'],obj.eigenvals)
assert not np.allclose(obj.CIeigs,obj.eigenvals,rtol=0,atol=1e-6)
assert r['energies']['state_average']==np.dot(obj.weight,obj.eigenvals)
assert r['effective']['state_weights']==[.75,.25]
assert r['convergence']['casscf']['converged']
assert r['convergence']['ci_final']['residuals'] is None
assert r['convergence']['ci_final']['mean_residual'] >= 0
assert r['effective']['engine']['H_dim']==obj.CIvecs.shape[-1]
assert 'CIvecs' not in r and 'D_tuvw_avg' not in r
text=pathlib.Path('job.out').read_text()
for section in ('Geometry (bohr)','Convergence','Final state energies','Excitation (Eh)','Normalized state weights'):
    assert section in text,section
assert '      0' in text and '      1' in text
assert f'{{obj.eigenvals[0]:.12f}}' in text
if {verbosity!r}=='normal':
    assert 'gfhgy' not in text and 'Psi4 engine output' not in text
else:
    assert 'Macroiteration' in text
with obj.output_context():
    print('subsequent property')
assert pathlib.Path('job.out').read_text().endswith('subsequent property\\n')
''', tmp_path)
    assert proc.stdout == proc.stderr == ''


@pytest.mark.parametrize('criterion', ['energy', 'gradient'])
def test_actual_macro_stopping_criterion(tmp_path, criterion):
    settings = 'energy_threshold=1.,gradient_threshold=0.' if criterion == 'energy' else 'energy_threshold=0.,gradient_threshold=1.'
    run(f'''
obj=calculate(geom,config.copy_with(quiet=True,casscf=CASSCFOptions({settings})))
d=obj.result.convergence['casscf']
assert d['converged'] and d['reason']=={criterion!r},d
''', tmp_path)


def test_partial_macro_result(tmp_path):
    run('''
obj=calculate(geom,config.copy_with(quiet=True))
r=obj.result.summary()
assert r['converged'] is False
assert r['convergence']['casscf']['reason']=='iteration_limit'
assert r['convergence']['casscf']['iterations']==1
assert r['warnings']
np.testing.assert_array_equal(r['energies']['final'],obj.eigenvals)
''', tmp_path, QED_MACRO_MAXITER='1')


def test_partial_ci_result(tmp_path):
    run('''
obj=calculate(geom,config.copy_with(method='fci',active_electrons=None,active_orbitals=None,
    quiet=True,ci_solver=CISolverOptions(threshold=1e-16,max_iterations=1,initial_subspace=2)))
r=obj.result.summary()
assert r['converged'] is False,r
assert r['convergence']['ci_final']['reason']=='iteration_limit'
assert r['convergence']['ci_final']['iterations']==1
''', tmp_path)


def test_properties_preserve_prior_states(tmp_path):
    proc = run('''
from qed_ci.output import calculation_output
config=config.copy_with(quiet=True,output_file='grad.out')
with calculation_output(config):
    from nuclear_grad import nuclear_grad
    obj=nuclear_grad(geom,config=config)
obj.compute_grad(0,0)
old=obj.result.properties['gradient_state_0']['value'].copy()
obj.compute_grad(1,1)
np.testing.assert_array_equal(old,obj.result.properties['gradient_state_0']['value'])
np.testing.assert_array_equal(obj.total_gradient.reshape(-1,3),obj.result.properties['gradient_state_1']['value'])
obj.compute_grad(0,1,deriv_type='full')
np.testing.assert_array_equal(obj.h_vector.reshape(-1,3),obj.result.properties['nac_0_1_full']['value'])
assert obj.result.properties['nac_0_1_full']['units']=='1/bohr'
json.dumps(obj.result.summary(),allow_nan=False)
assert 'gradient_state_1 (Eh/bohr)' in pathlib.Path('grad.out').read_text()
''', tmp_path)
    assert proc.stdout == proc.stderr == ''


def test_engine_exception_restoration(tmp_path):
    proc = run('''
obj=calculate(geom,config.copy_with(quiet=True))
import helper_PFCI, psi4
from qed_ci.output import engine_output
prior=psi4.core.get_output_file()
level=helper_PFCI.cfunctions.get_ci_print_level()
cwd=pathlib.Path.cwd()
try:
    with obj.output_context(), engine_output(config,helper_PFCI.cfunctions):
        print('hidden')
        raise RuntimeError('deliberate')
except RuntimeError:
    pass
assert pathlib.Path.cwd()==cwd
assert psi4.core.get_output_file()==prior
assert helper_PFCI.cfunctions.get_ci_print_level()==level
print('restored')
''', tmp_path)
    assert proc.stdout == 'restored\n' and proc.stderr == ''


def test_generated_files_and_unknown_reference_status(tmp_path):
    run('''
obj=calculate(geom,config.copy_with(method='fci',active_electrons=None,active_orbitals=None,
    reference_orbitals='canonical',photon_basis='number',quiet=True,
    initial_orbital_output_file='initial.txt'))
r=obj.result.summary()
assert r['convergence']['scf']['converged'] is None
assert r['converged'] is None
assert r['files']['initial_orbital_output_file']==str(pathlib.Path('initial.txt').resolve())
assert pathlib.Path(r['files']['initial_orbital_output_file']).exists()
assert r['effective']['engine']['n_act_el']==4
''', tmp_path)


def test_stale_export_not_claimed_on_iteration_cap(tmp_path):
    run('''
pathlib.Path('optimized.txt').write_text('old export')
obj=calculate(geom,config.copy_with(quiet=True,orbital_output_file='optimized.txt'))
assert obj.result.converged is False
assert 'orbital_output_file' not in obj.result.files
assert pathlib.Path('optimized.txt').read_text()=='old export'
''', tmp_path, QED_MACRO_MAXITER='1')


def test_spin_residuals_are_from_the_current_solve(tmp_path):
    run('''
obj=calculate(geom,config.copy_with(spin='singlet',quiet=True))
d=obj.result.convergence['ci_final']
assert len(d['residuals'])==2
np.testing.assert_allclose(np.mean(d['residuals']),d['mean_residual'],rtol=1e-12,atol=1e-25)
plain=calculate(geom,config.copy_with(quiet=True))
assert plain.result.convergence['ci_final']['residuals'] is None
''', tmp_path)


def test_failure_has_partial_report_and_preserves_exception(tmp_path):
    proc = run('''
from helper_PFCI import PFHamiltonianGenerator
from qed_ci.output import progress
original = PFHamiltonianGenerator._run_calculation
report_path = pathlib.Path('failed.out').resolve()
def fail(self, *args):
    progress('Starting initial CI solve')
    assert 'Starting initial CI solve' in report_path.read_text()
    raise RuntimeError('deliberate engine failure')
PFHamiltonianGenerator._run_calculation = fail
try:
    calculate(geom, config.copy_with(output_file=str(pathlib.Path('failed.out').resolve())))
except RuntimeError as exc:
    assert str(exc) == 'deliberate engine failure'
else:
    raise AssertionError('engine exception swallowed')
finally:
    PFHamiltonianGenerator._run_calculation = original
text = pathlib.Path('failed.out').read_text()
assert 'Starting QED-CI calculation' in text
assert 'Calculation failed: RuntimeError: deliberate engine failure' in text
assert 'Final state energies' not in text
''', tmp_path)
    # Direct legacy imports can print before the calculation scope.
    assert 'Calculation failed' not in proc.stdout


def test_native_warning_survives_quiet_in_results(tmp_path):
    proc = run('''
import os
# Load inside a quiet scope before installing the diagnostic injection.
from qed_ci.output import calculation_output
with calculation_output(config.copy_with(quiet=True)):
    from helper_PFCI import PFHamiltonianGenerator
original = PFHamiltonianGenerator._run_calculation
def with_diagnostic(self, *args):
    os.write(2, b'WARNING: injected native scientific diagnostic\\n')
    import psi4
    psi4.core.print_out('WARNING: injected Psi4 diagnostic\\n')
    return original(self, *args)
PFHamiltonianGenerator._run_calculation = with_diagnostic
obj = calculate(geom, config.copy_with(method='casci', quiet=True))
assert 'WARNING: injected native scientific diagnostic' in obj.result.warnings
assert 'WARNING: injected Psi4 diagnostic' in obj.result.warnings
''', tmp_path)
    assert proc.stdout == proc.stderr == ''
