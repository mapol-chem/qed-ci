"""Configuration contract tests: deliberately independent of Psi4/native code."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qed_ci import CQEDConfig, CISolverOptions, CASSCFOptions


SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'src'


def source_environment():
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(SOURCE_ROOT), environment.get('PYTHONPATH'))))
    return environment


def test_configuration_import_is_lightweight():
    subprocess.run([sys.executable, '-c',
        'import sys; from qed_ci import CQEDConfig; CQEDConfig(); '
        'assert "psi4" not in sys.modules; assert "helper_PFCI" not in sys.modules'],
        env=source_environment(), check=True)


@pytest.mark.parametrize('kwargs', [
    {'method':'typo'}, {'reference':'uhf'}, {'reference':'rks'}, {'multiplicity':3},
    {'omega':float('nan')}, {'omega':-1}, {'lambda_vector':[0,1]},
    {'lambda_vector':[0,0,float('inf')]}, {'max_photons':True}, {'n_roots':0},
    {'rdm_root':3}, {'state_weights':[1,-1,1]}, {'state_weights':[0,0,0]},
    {'state_weights':[1,1]}, {'method':'casci'},
    {'state_weights':[1e308,1e308,1e308]},
    {'method':'fci','max_photons':0,'ci_solver':CISolverOptions(algorithm='dense')},
    {'method':'casci','active_electrons':3,'active_orbitals':2},
    {'photon_basis':'number'}, {'ci_solver':CISolverOptions(algorithm='davidson')},
    {'method':'casscf','active_electrons':2,'active_orbitals':2,'ci_solver':CISolverOptions(algorithm='dense')},
    {'psi4_options':{'REFERENCE':'UHF'}}, {'quiet':'yes'}, {'output_mode':'bad'},
    {'casscf':CASSCFOptions(energy_threshold=1e-9)},
])
def test_invalid_inputs_fail_before_engine_import(kwargs):
    with pytest.raises((ValueError,TypeError)):
        CQEDConfig(**kwargs)


def test_copy_and_serialization_are_defensive():
    options={'basis':'sto-3g','nested':[1]}
    c=CQEDConfig(psi4_options=options)
    options['nested'].append(2)
    assert c.psi4_options['nested']==[1]
    d=c.copy_with(quiet=True)
    d.psi4_options['nested'].append(3)
    assert c.psi4_options['nested']==[1]
    assert json.loads(json.dumps(c.to_dict()))['reference']=='rhf'
    assert CQEDConfig.from_dict(json.loads(json.dumps(c.to_dict()))).to_dict()==c.to_dict()


def test_legacy_defaults_and_explicit_casci():
    c=CQEDConfig.from_legacy({}, {'ci_level':'cas','nact_els':2,'nact_orbs':2})
    assert c.method=='casscf'
    assert c.verbosity=='trace'
    d=CQEDConfig.from_legacy({}, {'ci_level':'CAS','casscf_optimization':False,'nact_els':2,'nact_orbs':2})
    assert d.method=='casci'


def test_legacy_conflicts_are_reported_and_weights_padded():
    c=CQEDConfig.from_legacy({}, {'canonical_mos':True,'coherent_state_basis':True,'casscf_weight':[2]})
    assert c.photon_basis=='number'
    assert c.state_weights==(2,0,0)
    assert len(c.diagnostics)==2


def test_unknown_legacy_key_suggests_correction():
    with pytest.raises(ValueError,match='davidson_roots'):
        CQEDConfig.from_legacy({}, {'davidson_root':2})


def test_stage1_inputs_convert_without_changing_method():
    data=json.loads((Path(__file__).resolve().parents[1]/'docs/audit/stage1/baselines.json').read_text())
    for name,entry in data.items():
        c=CQEDConfig.from_legacy(entry['input']['psi4_options'],entry['input']['cavity_options'])
        assert c.method==name.split('_')[0]
        c.validate_dimensions(6,2,2)


def test_dimension_validation():
    c=CQEDConfig(method='casci',active_electrons=4,active_orbitals=4,n_roots=2)
    with pytest.raises(ValueError,match='Active space'):
        c.validate_dimensions(3,2,2)
    with pytest.raises(ValueError,match='closed-shell'):
        c.validate_dimensions(6,2,1)


def test_iteration_and_convergence_options():
    with pytest.raises(ValueError): CISolverOptions(initial_subspace=5,max_subspace=4)
    with pytest.raises(ValueError): CASSCFOptions(energy_threshold=0,gradient_threshold=0)
    with pytest.raises(TypeError): CASSCFOptions(coupled_ci_iterations=1.5)


def test_explicit_environment_and_default_precedence(monkeypatch):
    from types import SimpleNamespace
    from qed_ci.config import CASSCF_CONTROLS
    from qed_ci.runtime import solver_settings
    module=SimpleNamespace(**{'_'+env:default for env,default in CASSCF_CONTROLS.values()})
    monkeypatch.setitem(sys.modules,'helper_PFCI',module)
    monkeypatch.setenv('QED_GRAD_CONV','2e-7')
    owner=SimpleNamespace(config=CQEDConfig(method='casscf',active_electrons=2,active_orbitals=2,
                    casscf=CASSCFOptions(gradient_threshold=1e-9)))
    with pytest.raises(RuntimeError):
        with solver_settings(owner):
            assert module._QED_GRAD_CONV==1e-9
            assert owner.resolved_options['effective_casscf']['gradient_threshold']['source']=='config'
            raise RuntimeError('test rollback')
    assert module._QED_GRAD_CONV==1e-6
    owner.config=owner.config.copy_with(casscf=CASSCFOptions())
    with solver_settings(owner):
        assert module._QED_GRAD_CONV==2e-7
    assert module._QED_GRAD_CONV==1e-6


def test_legacy_dse_basis_order_is_preserved():
    c=CQEDConfig.from_legacy({}, {'ignore_dse_terms':True})
    assert c.photon_basis=='number' and c.reference_orbitals=='qed_rhf'
    raw=c.to_legacy_options()
    assert not raw['photon_number_basis'] and not raw['canonical_mos']
    assert raw['ignore_dse_terms'] # the established parser applies this last


def test_reports_cannot_overwrite_orbital_files(tmp_path):
    with pytest.raises(ValueError,match='distinct'):
        CQEDConfig(method='casscf',active_electrons=2,active_orbitals=2,
                   output_file=tmp_path/'same',orbital_output_file=tmp_path/'same')
