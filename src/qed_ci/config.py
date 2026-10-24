"""Validated options; no NumPy, Psi4 or native library imports.

Legacy conversion preserves valid historical defaults and records diagnostics
for conflicting/ignored options. New inputs use explicit method names.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from difflib import get_close_matches
import math
from pathlib import Path


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        # Accept integral NumPy scalars without importing NumPy.
        import numbers
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f'{name} must be an integer')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')


def finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        import numbers
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f'{name} must be a real number')
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')


def weights(values, name, length=None):
    result = tuple(values)
    if not result or (length is not None and len(result) != length):
        raise ValueError(f'{name} must have {length or "at least one"} entries')
    for x in result:
        finite(x, name)
    if not math.isfinite(sum(result)) or sum(result) <= 1e-14:
        raise ValueError(f'{name} must have a finite positive total weight')
    # Leave normalization to the established NumPy parser, preserving arithmetic.
    return result


@dataclass(frozen=True)
class CISolverOptions:
    algorithm: str = 'auto'
    threshold: float = 1e-5
    initial_subspace: int = 4
    max_subspace: int = 20
    max_iterations: int = 100
    guess: str = 'unit guess'

    def __post_init__(self):
        if self.algorithm not in {'auto', 'dense', 'davidson'}:
            raise ValueError('CI algorithm must be auto, dense, or davidson')
        finite(self.threshold, 'CI threshold')
        if self.threshold == 0:
            raise ValueError('CI threshold must be positive')
        for key in ('initial_subspace', 'max_subspace', 'max_iterations'):
            integer(getattr(self, key), key, 1)
        if self.max_subspace < self.initial_subspace:
            raise ValueError('max_subspace must be >= initial_subspace')
        if self.guess != 'unit guess':
            raise ValueError('Only unit guess is implemented by the current CI input path')


# None means inherit the existing effective setting, not overwrite it.
CASSCF_CONTROLS = {
    'energy_threshold': ('QED_MACRO_ENERGY_CONV', 1e-10),
    'gradient_threshold': ('QED_GRAD_CONV', 1e-6),
    'micro_gradient_floor': ('QED_MICRO_GRAD_FLOOR', 1e-7),
    'micro_energy_floor': ('QED_MICRO_ENERGY_FLOOR', 1e-10),
    'micro_gradient_scale': ('QED_MICRO_GRAD_SCALE', 0.1),
    'micro_energy_scale': ('QED_MICRO_ENERGY_SCALE', 0.01),
    'uncoupled_ci_iterations': ('QED_CI_UNCOUPLED_MAXITER', 5),
    'coupled_ci_iterations': ('QED_CI_COUPLED_MAXITER', 5),
    'coupled_ci_factor': ('QED_CI_COUPLED_FACTOR', 0.1),
}


@dataclass(frozen=True)
class CASSCFOptions:
    energy_threshold: float | None = None
    gradient_threshold: float | None = None
    micro_gradient_floor: float | None = None
    micro_energy_floor: float | None = None
    micro_gradient_scale: float | None = None
    micro_energy_scale: float | None = None
    uncoupled_ci_iterations: int | None = None
    coupled_ci_iterations: int | None = None
    coupled_ci_factor: float | None = None

    def __post_init__(self):
        for key in CASSCF_CONTROLS:
            value = getattr(self, key)
            if value is not None:
                if key.endswith('iterations'):
                    integer(value, key, 1)
                else:
                    finite(value, key)
        if self.energy_threshold == 0 and self.gradient_threshold == 0:
            raise ValueError('At least one CASSCF convergence criterion must be enabled')


@dataclass(frozen=True)
class CQEDConfig:
    method: str = 'cis'
    lambda_vector: tuple = (0., 0., 0.)
    omega: float = 0.
    psi4_options: dict = field(default_factory=dict)
    reference: str = 'rhf'
    charge: int | None = None
    multiplicity: int = 1
    photon_basis: str = 'coherent'
    reference_orbitals: str = 'qed_rhf'
    max_photons: int = 1
    active_electrons: int | None = None
    active_orbitals: int | None = None
    n_roots: int = 3
    state_weights: tuple | None = None
    spin: str = 'no'
    rdm_root: int = 0
    ci_solver: CISolverOptions = field(default_factory=CISolverOptions)
    casscf: CASSCFOptions = field(default_factory=CASSCFOptions)
    natural_orbitals: bool = False
    natural_orbital_weights: tuple = (1., 1.)
    ignore_coupling: bool = False
    ignore_dse_terms: bool = False
    test_mode: bool = False
    orbital_guess_file: str | None = None
    orbital_output_file: str | None = None
    natural_orbital_output_file: str | None = None
    occupation_output_file: str | None = None
    initial_orbital_output_file: str | None = None
    quiet: bool = False
    verbosity: str = 'normal'
    output_file: str | None = None
    output_mode: str = 'w'
    scratch_directory: str | None = None
    # Internal compatibility metadata is retained in resolved settings.
    legacy: bool = field(default=False, repr=False)
    diagnostics: tuple = field(default=(), repr=False)
    _legacy_options: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'method', self.method.strip().lower())
        object.__setattr__(self, 'diagnostics', tuple(self.diagnostics))
        object.__setattr__(self, '_legacy_options', deepcopy(self._legacy_options))
        object.__setattr__(self, 'reference', self.reference.strip().lower())
        if self.method not in {'cis', 'fci', 'casci', 'casscf'}:
            raise ValueError('method must be cis, fci, casci, or casscf')
        if self.reference != 'rhf':
            raise ValueError('QED-CI currently supports only reference="rhf"; UHF/KS correlation is not implemented')
        if self.charge is not None:
            integer(self.charge, 'charge', -1000000)
        integer(self.multiplicity, 'multiplicity', 1)
        if self.multiplicity != 1:
            raise ValueError('The current restricted reference requires multiplicity=1; target excited-state spin uses spin=')
        finite(self.omega, 'omega')
        coupling = tuple(self.lambda_vector)
        if len(coupling) != 3:
            raise ValueError('lambda_vector must have three components')
        for x in coupling:
            finite(x, 'lambda_vector component', -math.inf)
        object.__setattr__(self, 'lambda_vector', coupling)
        object.__setattr__(self, 'psi4_options', deepcopy(dict(self.psi4_options)))
        for key in self.psi4_options:
            if str(key).lower() == 'reference' and str(self.psi4_options[key]).lower() != 'rhf':
                raise ValueError('psi4_options reference conflicts with the supported RHF reference')
        integer(self.max_photons, 'max_photons')
        integer(self.n_roots, 'n_roots', 1)
        integer(self.rdm_root, 'rdm_root')
        if self.rdm_root >= self.n_roots and not (self.legacy and self.method == 'cis'):
            raise ValueError('rdm_root must be less than n_roots')
        if self.photon_basis not in {'coherent', 'number'}:
            raise ValueError('photon_basis must be coherent or number')
        if self.reference_orbitals not in {'qed_rhf', 'canonical'}:
            raise ValueError('reference_orbitals must be qed_rhf or canonical')
        if ((self.reference_orbitals == 'canonical') != (self.photon_basis == 'number')) and not (
                self.legacy and self.ignore_dse_terms and self.reference_orbitals == 'qed_rhf' and self.photon_basis == 'number'):
            raise ValueError('Current supported pairing: qed_rhf/coherent or canonical/number')
        if self.spin not in {'no', 'singlet', 'triplet', 'quintet'}:
            raise ValueError('spin must be no, singlet, triplet, or quintet')
        if self.method in {'casci', 'casscf'}:
            integer(self.active_electrons, 'active_electrons', 2)
            integer(self.active_orbitals, 'active_orbitals', 1)
            if self.active_electrons % 2 or self.active_electrons > 2 * self.active_orbitals:
                raise ValueError('Active electrons must be even and fit in the active orbitals')
        elif not self.legacy and (self.active_electrons is not None or self.active_orbitals is not None):
            raise ValueError('An active space applies only to CASCI/CASSCF')
        if not isinstance(self.ci_solver, CISolverOptions) or not isinstance(self.casscf, CASSCFOptions):
            raise TypeError('ci_solver and casscf require their respective options objects')
        if not self.legacy:
            if self.method == 'cis' and (self.max_photons != 1 or self.ci_solver.algorithm == 'davidson'):
                raise ValueError('Current CIS implementation requires max_photons=1 and a dense/auto solver')
            if self.ci_solver.algorithm == 'dense' and self.max_photons != 1:
                raise ValueError('The current dense Hamiltonian builder requires max_photons=1')
            if self.method == 'cis' and self.spin != 'no':
                raise ValueError('CIS spin selection is not implemented by this input path')
            if self.method == 'casscf' and self.ci_solver.algorithm == 'dense':
                raise ValueError('Dense diagonalization skips orbital optimization; use method="casci"')
            if self.method != 'casscf' and any(v is not None for v in asdict(self.casscf).values()):
                raise ValueError('CASSCF controls apply only to method="casscf"')
            if self.ignore_dse_terms and self.photon_basis != 'number':
                raise ValueError('ignore_dse_terms currently requires the number photon basis')
            if self.method != 'casscf' and any((self.orbital_output_file,self.natural_orbital_output_file,self.occupation_output_file)):
                raise ValueError('Optimized/natural orbital exports currently require CASSCF; use initial_orbital_output_file for starting orbitals')
        if self.state_weights is not None:
            object.__setattr__(self, 'state_weights', weights(self.state_weights, 'state_weights', self.n_roots))
        object.__setattr__(self, 'natural_orbital_weights', weights(self.natural_orbital_weights, 'natural_orbital_weights'))
        for name in ('quiet', 'natural_orbitals', 'ignore_coupling', 'ignore_dse_terms', 'test_mode'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be a bool')
        if self.natural_orbitals and self.orbital_guess_file:
            raise ValueError('Choose natural orbitals or an orbital guess file, not both')
        if self.verbosity not in {'normal', 'debug', 'trace'}:
            raise ValueError('verbosity must be normal, debug, or trace')
        if self.output_mode not in {'w', 'a', 'x'}:
            raise ValueError('output_mode must be w (replace), a (append), or x (exclusive)')
        for name in ('orbital_guess_file', 'orbital_output_file', 'natural_orbital_output_file',
                     'occupation_output_file', 'initial_orbital_output_file', 'output_file', 'scratch_directory'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(Path(value).expanduser().resolve()))
        artifact_paths = [getattr(self, name) for name in ('orbital_guess_file','orbital_output_file',
                          'natural_orbital_output_file','occupation_output_file','initial_orbital_output_file')]
        if self.output_file is not None and self.output_file in artifact_paths:
            raise ValueError('The report file must be distinct from orbital input/output files')
        if self.orbital_guess_file and not Path(self.orbital_guess_file).is_file():
            raise ValueError(f'Orbital guess file does not exist: {self.orbital_guess_file}')

    def copy_with(self, **updates):
        return replace(self, **updates)

    @classmethod
    def from_dict(cls, values):
        """Read a JSON-compatible options mapping, validating nested groups."""
        values = deepcopy(dict(values))
        if isinstance(values.get('ci_solver'), dict):
            values['ci_solver'] = CISolverOptions(**values['ci_solver'])
        if isinstance(values.get('casscf'), dict):
            values['casscf'] = CASSCFOptions(**values['casscf'])
        return cls(**values)

    def to_dict(self):
        result = asdict(self)
        result.pop('_legacy_options')
        def plain(value):
            if isinstance(value, dict):
                return {k:plain(v) for k,v in value.items()}
            if isinstance(value, (tuple,list)):
                return [plain(v) for v in value]
            if hasattr(value, 'tolist'):
                return plain(value.tolist())
            return value
        return plain(result)

    def to_legacy_options(self):
        if self._legacy_options is not None:
            result = deepcopy(self._legacy_options)
        else:
            result = {}
        result.update(ci_level='cas' if self.method in {'casci', 'casscf'} else self.method,
            casscf_optimization=self.method == 'casscf', omega_value=self.omega,
            lambda_vector=list(self.lambda_vector), number_of_photons=self.max_photons,
            davidson_roots=self.n_roots, rdm_root=self.rdm_root, spin_adaptation=self.spin,
            canonical_mos=self.reference_orbitals == 'canonical', photon_number_basis=self.photon_basis == 'number',
            coherent_state_basis=self.photon_basis == 'coherent', ignore_coupling=self.ignore_coupling,
            ignore_dse_terms=self.ignore_dse_terms, natural_orbitals=self.natural_orbitals,
            rdm_weights=list(self.natural_orbital_weights), test_mode=self.test_mode,
            full_diagonalization=self.ci_solver.algorithm == 'dense', davidson_guess=self.ci_solver.guess,
            davidson_threshold=self.ci_solver.threshold, davidson_indim=self.ci_solver.initial_subspace,
            davidson_maxdim=self.ci_solver.max_subspace, davidson_maxiter=self.ci_solver.max_iterations,
            use_orbital_guess=self.orbital_guess_file is not None,
            save_orbital=any((self.orbital_output_file, self.natural_orbital_output_file, self.occupation_output_file)))
        if self.active_electrons is not None:
            result.update(nact_els=self.active_electrons, nact_orbs=self.active_orbitals)
        if self.state_weights is not None:
            result['casscf_weight'] = list(self.state_weights)
        else:
            result.pop('casscf_weight', None)
        if self.legacy and self.ignore_dse_terms and self.reference_orbitals == 'qed_rhf':
            # The original parser forces photon-number AFTER reconciling the
            # canonical flag. Reproduce that order, rather than re-running the
            # already resolved flags through a different precedence path.
            result.update(canonical_mos=False, photon_number_basis=False, coherent_state_basis=True)
        return result

    def validate_dimensions(self, nmo, nalpha, nbeta):
        if nalpha != nbeta:
            raise ValueError('Only closed-shell restricted references are supported')
        if self.method in {'casci', 'casscf'}:
            inactive = nalpha - self.active_electrons // 2
            if inactive < 0 or inactive + self.active_orbitals > nmo:
                raise ValueError('Active space is incompatible with molecular electrons/orbitals')
            dim = math.comb(self.active_orbitals, self.active_electrons // 2)**2 * (self.max_photons + 1)
        elif self.method == 'fci':
            dim = math.comb(nmo, nalpha)**2 * (self.max_photons + 1)
        else:
            dim = 2 * (1 + 2 * nalpha * (nmo - nalpha))
        if self.n_roots > dim:
            raise ValueError(f'n_roots={self.n_roots} exceeds CI dimension {dim}')
        if self.rdm_root >= dim:
            raise ValueError(f'rdm_root={self.rdm_root} exceeds CI dimension {dim}')
        if self.method != 'cis' and self.ci_solver.algorithm != 'dense' and self.n_roots == dim:
            raise ValueError('Iterative CI needs room beyond the requested roots; request fewer roots')
        return {'ci_dimension': dim, 'requested_root_vector_bytes': dim * self.n_roots * 8,
                'note': 'Vector storage only; not a total memory estimate'}

    @classmethod
    def from_legacy(cls, psi4_options, cavity_options):
        original = {str(k).lower(): v for k, v in cavity_options.items()}
        known = {'omega_value','lambda_vector','number_of_photons','ci_level','casscf_optimization',
                 'ignore_coupling','natural_orbitals','canonical_mos','photon_number_basis',
                 'coherent_state_basis','ignore_dse_terms','rdm_weights','full_diagonalization',
                 'test_mode','break_degeneracy','davidson_roots','davidson_guess','davidson_threshold',
                 'davidson_indim','davidson_maxdim','davidson_maxiter','use_orbital_guess','save_orbital',
                 'rdm_root','nact_orbs','nact_els','spin_adaptation','casscf_weight'}
        output_keys = {'quiet','verbosity','output_file','output_mode','scratch_directory',
                       'orbital_guess_file','orbital_output_file','initial_orbital_output_file',
                       'natural_orbital_output_file','occupation_output_file'}
        unknown = set(original) - known - output_keys
        if unknown:
            key = sorted(unknown)[0]
            hint = get_close_matches(key, known | output_keys, n=1)
            raise ValueError(f'Unknown cavity option {key!r}' + (f'; did you mean {hint[0]!r}?' if hint else ''))
        c = original
        diagnostics = []
        for key in ('casscf_optimization','ignore_coupling','natural_orbitals','canonical_mos',
                    'photon_number_basis','coherent_state_basis','ignore_dse_terms',
                    'full_diagonalization','test_mode','break_degeneracy','use_orbital_guess','save_orbital'):
            if key in c and not isinstance(c[key], bool):
                raise TypeError(f'{key} must be a bool')
        method = str(c.get('ci_level', 'cis')).lower()
        optimize = c.get('casscf_optimization', True)
        if method == 'cas':
            method = 'casscf' if optimize else 'casci'
        if method not in {'casci','casscf'} and c.get('casscf_optimization') is True:
            diagnostics.append('casscf_optimization is ignored outside CAS calculations')
        if method == 'casscf' and c.get('full_diagonalization',False):
            diagnostics.append('Legacy full_diagonalization skips CASSCF optimization; only fixed-orbital CI is performed')
        canonical = bool(c.get('canonical_mos', False) or c.get('photon_number_basis', False))
        coherent = c.get('coherent_state_basis', not canonical) and not canonical
        number = canonical
        if c.get('ignore_dse_terms', False):
            number, coherent = True, False
        for key, effective in [('canonical_mos',canonical), ('photon_number_basis',number), ('coherent_state_basis',coherent)]:
            if key in c and c[key] != effective:
                diagnostics.append(f'{key}={c[key]} was overridden to {effective} by legacy basis precedence')
        if not coherent and not number:
            raise ValueError('Legacy input disables both photon representations')
        roots = c.get('davidson_roots', 3)
        integer(roots, 'davidson_roots', 1)
        w = tuple(c.get('casscf_weight', [1.] * roots))
        if len(w) > roots:
            raise ValueError('casscf_weight has more entries than davidson_roots')
        if len(w) < roots:
            diagnostics.append('casscf_weight was padded with zero weights')
            w += (0.,) * (roots-len(w))
        if method == 'cis' and c.get('number_of_photons',1) != 1:
            diagnostics.append('Legacy dense CIS still uses two photon sectors regardless of number_of_photons')
        if c.get('break_degeneracy',False):
            diagnostics.append('break_degeneracy has no effect in the current Python dispatch')
        guess = c.get('davidson_guess','unit guess')
        if guess != 'unit guess':
            diagnostics.append('davidson_guess is unused by the current backend; effective guess is unit guess')
        output = {k:c[k] for k in output_keys if k in c}
        if c.get('use_orbital_guess',False): output.setdefault('orbital_guess_file','orbital.out')
        if c.get('save_orbital',False):
            output.setdefault('orbital_output_file','orbital.out')
            output.setdefault('natural_orbital_output_file','natural_orbital.out')
            output.setdefault('occupation_output_file','occupation_number.out')
        # Preserve legacy verbosity; new configs default to normal.
        output.setdefault('verbosity','trace')
        return cls(method=method, psi4_options=psi4_options, legacy=True, diagnostics=tuple(diagnostics),
            _legacy_options={k:v for k,v in c.items() if k in known},
            omega=c.get('omega_value',0), lambda_vector=c.get('lambda_vector',(0,0,0)),
            max_photons=c.get('number_of_photons',1), n_roots=roots, state_weights=w,
            reference_orbitals='canonical' if canonical else 'qed_rhf', photon_basis='coherent' if coherent else 'number',
            active_electrons=c.get('nact_els',0) if method in {'casci','casscf'} else None,
            active_orbitals=c.get('nact_orbs',0) if method in {'casci','casscf'} else None,
            spin=c.get('spin_adaptation','no'), rdm_root=c.get('rdm_root',0),
            natural_orbitals=c.get('natural_orbitals',False), natural_orbital_weights=c.get('rdm_weights',(1,1)),
            ignore_coupling=c.get('ignore_coupling',False), ignore_dse_terms=c.get('ignore_dse_terms',False),
            test_mode=c.get('test_mode',False),
            ci_solver=CISolverOptions(algorithm='dense' if c.get('full_diagonalization',False) else 'auto',
                threshold=c.get('davidson_threshold',1e-5), initial_subspace=c.get('davidson_indim',4),
                max_subspace=c.get('davidson_maxdim',20), max_iterations=c.get('davidson_maxiter',100), guess='unit guess'),
            **output)
