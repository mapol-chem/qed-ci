"""Version 1 scientific results, independent of presentation and engine imports."""
from dataclasses import dataclass, field
from copy import deepcopy
import math


def json_value(value):
    """JSON-safe values; nonfinite diagnostics are explicitly unavailable."""
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class CalculationResult:
    """Small scientific snapshot. Energies are Eh, states zero-based.

    Large CI/RDM tensors remain on the calculation; no copies or implicit tensor
    serialization. Property arrays are owned snapshots, so later calls cannot
    overwrite earlier results. ``summary`` returns detached JSON-compatible data.
    """
    requested: dict
    effective: dict
    geometry: dict
    energies: dict
    convergence: dict
    provenance: dict
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)
    schema_version: int = 1

    @property
    def converged(self):
        statuses = [v['converged'] for k, v in self.convergence.items()
                    if k != 'ci_initial' and v.get('applicable', True)]
        if False in statuses:
            return False
        return True if all(v is True for v in statuses) else None

    def summary(self):
        from dataclasses import asdict
        data = asdict(self)
        data['converged'] = self.converged
        data['status'] = {True: 'converged', False: 'unconverged', None: 'unknown'}[self.converged]
        data['units'] = {'energies': 'Eh', 'geometry': 'bohr', 'timings': 's'}
        data['state_index_base'] = 0
        return json_value(data)


def ci_diagnostics(backend, constint, constdouble, target_spin=None):
    """Capture immediately after get_roots, before another solve changes C state."""
    import ctypes
    backend.get_last_ci_iterations.restype = ctypes.c_int
    backend.get_last_ci_root_residual.argtypes = [ctypes.c_int]
    backend.get_last_ci_root_residual.restype = ctypes.c_double
    # Only davidson_spin publishes these getters. The unadapted davidson
    # path leaves their global storage untouched, potentially from another job.
    residuals = ([backend.get_last_ci_root_residual(i) if i < 64 else None
                  for i in range(int(constint[7]))]
                 if target_spin is not None and target_spin >= 0 else None)
    converged = int(constint[8]) == 0
    return dict(converged=converged, reason='residual_threshold' if converged else 'iteration_limit',
                iterations=backend.get_last_ci_iterations(),
                mean_residual=float(constdouble[4]), residuals=residuals)


def build_result(owner, seconds):
    import sys
    import hashlib
    from pathlib import Path
    import numpy as np
    import psi4
    initial = np.asarray(owner.CIeigs)
    optimized = getattr(owner, 'eigenvals', None)
    final = initial if optimized is None else np.asarray(optimized)
    weights = getattr(owner, 'weight', None)
    macro = deepcopy(owner._macro_diagnostics)
    if owner.config.method == 'casscf' and not macro['applicable']:
        macro['reason'] = 'orbital_optimization_skipped'
    convergence = {'scf': deepcopy(getattr(owner, '_scf_diagnostics', dict(converged=None, reason='unavailable'))),
                   'ci_initial': deepcopy(owner._initial_ci_diagnostics),
                   'ci_final': deepcopy(owner._final_ci_diagnostics), 'casscf': macro}
    effective = deepcopy(owner.resolved_options)
    effective['engine'] = {k: json_value(getattr(owner, k, None)) for k in
                           ('ci_level', 'casscf_optimization', 'canonical_mos', 'coherent_state_basis',
                            'photon_number_basis', 'N_p', 'n_act_orb', 'n_act_el', 'n_in_a', 'H_dim', 'target_spin')}
    effective['engine']['H_dim'] = int(owner.CIvecs.shape[-1])
    effective['engine']['photon_sectors'] = 2 if owner.ci_level == 'cis' or owner.full_diagonalization else owner.N_p + 1
    if owner.ci_level == 'fci':
        effective['engine']['n_act_el'] = 2 * owner.n_act_a
    effective['state_weights'] = json_value(weights)
    effective['method'] = ('casscf' if macro['applicable'] else
                           'casci' if owner.ci_level == 'cas' else owner.ci_level)
    files = dict(owner._generated_files)
    if owner.config.output_file:
        files['output_file'] = owner.config.output_file
    backend = Path(sys.modules['helper_PFCI'].lib_path)
    from . import __version__
    warnings = list(owner.diagnostics)
    if owner.config.method == 'casscf' and not macro['applicable']:
        warnings.append('Orbital optimization was skipped by the effective engine route.')
    for name, diagnostic in convergence.items():
        if diagnostic.get('converged') is False:
            warnings.append(f'{name}: {diagnostic["reason"]}; this stage did not converge')
    result = CalculationResult(
        requested=owner.config.to_dict(), effective=effective, geometry=owner._result_geometry,
        energies=dict(reference=float(getattr(owner, 'cqed_reference_energy', owner.rhf_reference_energy)),
                      rhf_reference=float(owner.rhf_reference_energy), initial_ci=initial.tolist(),
                      final=final.tolist(), final_source='eigenvals' if optimized is not None else 'CIeigs',
                      excitation=(final-final[0]).tolist(),
                      state_average=float(np.dot(weights, final[:len(weights)]))
                      if weights is not None and owner.ci_level != 'cis' else None),
        convergence=convergence,
        provenance=dict(package='qed-ci', package_version=__version__, python=sys.version.split()[0],
                        numpy=np.__version__, psi4=psi4.__version__, backend=str(backend),
                        backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest()),
        warnings=warnings, timings={'calculation': seconds}, files=files)
    if hasattr(owner, 'dipole_array'):
        result.properties['dipole'] = dict(units='e bohr', value=owner.dipole_array.copy(),
            axes=['bra_state', 'ket_state', 'xyz'],
            energy_source='eigenvals' if macro.get('converged') else 'CIeigs')
    return result
