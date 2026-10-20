"""Scoped adapters for the existing engine; no numerical algorithms live here."""
from contextlib import contextmanager
import os
import sys
from .config import CASSCF_CONTROLS


@contextmanager
def solver_settings(owner):
    """Explicit config > current environment > established default.

    Only the registered CASSCF controls are migrated. Other experimental/import-
    time controls retain their old semantics and are recorded, not reinterpreted.
    """
    module = sys.modules['helper_PFCI']
    previous = {}
    resolved = {}
    for field, (env, default) in CASSCF_CONTROLS.items():
        value = getattr(owner.config.casscf, field)
        source = 'config'
        if value is None:
            if env in os.environ:
                value = int(os.environ[env]) if field.endswith('iterations') else float(os.environ[env])
                source = 'environment'
            else:
                # Preserve programmatic overrides of established module state.
                value = getattr(module, '_' + env, default)
                source = 'module/default'
        from .config import finite, integer
        if field.endswith('iterations'):
            integer(value, field, 1)
        else:
            finite(value, field)
        resolved[field] = {'value': value, 'source': source, 'environment_name': env}
    if owner.config.method == 'casscf' and resolved['energy_threshold']['value'] == 0 and resolved['gradient_threshold']['value'] == 0:
        raise ValueError('Both effective CASSCF convergence criteria are disabled')
    dimensions = getattr(owner, 'resolved_options', {}).get('dimensions')
    owner.resolved_options = owner.config.to_dict()
    if dimensions is not None:
        owner.resolved_options['dimensions'] = dimensions
    owner.resolved_options['effective_casscf'] = resolved
    owner.resolved_options['legacy_environment'] = {k:v for k,v in os.environ.items() if k.startswith(('QED_', 'CPP_CASSCF_'))}
    owner.resolved_options['import_time_controls'] = {
        name: {k:v for k,v in vars(m).items() if k.startswith('_QED_') and isinstance(v,(str,int,float,bool,type(None)))}
        for name,m in list(sys.modules.items()) if m is not None and name in {
            'helper_PFCI','helper_cqed_rhf','orbital_sigma','nuclear_grad','trust_region_solvers',
            'lstrs_trust_region','davidson_augmented_hessian'}}
    try:
        for field, (env, default) in CASSCF_CONTROLS.items():
            key = '_' + env
            previous[key] = getattr(module, key)
            setattr(module, key, resolved[field]['value'])
        yield
    finally:
        for key,value in previous.items():
            setattr(module,key,value)


@contextmanager
def calculation_scope(owner, backend, *, append=False):
    from .output import calculation_output, workspace
    with calculation_output(owner.config, append=append, backend=backend):
        with workspace(owner), solver_settings(owner):
            yield
