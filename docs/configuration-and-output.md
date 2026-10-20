# Configuration, results and output (Stages 2–5)

This is the authoritative option reference for the validated `CQEDConfig` API and
the version-1 result schema. The numerical methods and existing result attributes
are unchanged. It does not add a Calculator, replace the mean-field provider, or
replace the established calculation object. Install the package as described in
[installation](installation.md); no manual `PYTHONPATH` is needed. Configuration
imports do not import Psi4, NumPy, or the native library. The engine loads when
requested.

## New entry point

```python
from qed_ci import CQEDConfig, CISolverOptions, CASSCFOptions, calculate

config = CQEDConfig(
    method="casscf",             # cis, fci, casci, casscf
    reference="rhf",            # only restricted HF references are supported
    psi4_options={"basis": "sto-3g", "scf_type": "pk",
                  "e_convergence": 1e-10, "d_convergence": 1e-10},
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.12,                  # atomic units
    max_photons=1,              # occupations 0 through 1
    active_electrons=2,
    active_orbitals=2,
    n_roots=2,
    state_weights=(1.0, 1.0),   # relative weights, normalized by the engine
    ci_solver=CISolverOptions(threshold=1e-9),
    output_file="lih.out",      # no duplicate stdout
)
calculation = calculate("Li\nH 1 1.4\nsymmetry c1", config)
print(calculation.eigenvals)    # optimized CASSCF energies
print(calculation.resolved_options)
```

`PFHamiltonianGenerator(molecule, config=config)` and
`PFHamiltonianGenerator(molecule, config)` also work. Do not mix a config with
the old dictionaries. `calculate()` additionally routes output during the initial
lazy import of the engine. Treat configurations as immutable; use
`config.copy_with(quiet=True)` for changes. Mappings are defensively copied when
the config is constructed/copied. `to_dict()` and `CQEDConfig.from_dict(mapping)`
support JSON-compatible configuration records, including nested solver options.
Do not mutate nested mappings while a calculation is running.

## Options and validation

| Group | Options and defaults |
|---|---|
| Method/reference | `method="cis"`, `reference="rhf"`, `charge=None` (take geometry), `multiplicity=1` |
| Quantum chemistry | `psi4_options={}` passes standard options to Psi4. A conflicting `REFERENCE` is rejected. |
| Cavity | `omega=0.0`, `lambda_vector=(0,0,0)`, `max_photons=1` |
| Representation | `reference_orbitals="qed_rhf"`, `photon_basis="coherent"`; alternative pairing is `"canonical"`/`"number"` |
| Active space | `active_electrons`, `active_orbitals` required for CASCI/CASSCF; even active electron count |
| States | `n_roots=3`, `state_weights=None` (equal weights), `spin="no"`, `rdm_root=0` |
| Starting orbitals | `natural_orbitals=False`, `natural_orbital_weights=(1,1)`, or `orbital_guess_file=...` |
| Expert physics | `ignore_coupling=False`, `ignore_dse_terms=False`; DSE omission requires number basis in the new API |
| Developer | `test_mode=False` retains the established additional dense allocations |
| Output | `quiet=False`, `verbosity="normal"`, `output_file=None`, `output_mode="w"` |
| Files | `scratch_directory=None`, `initial_orbital_output_file=None`, `orbital_output_file=None`, `natural_orbital_output_file=None`, `occupation_output_file=None` |

Units, allowed values, and precedence:

- `omega` is the cavity frequency in atomic units (Eh); `lambda_vector` is a
  three-component coupling in atomic units. Neither is converted from SI.
- `active_electrons` and `active_orbitals` are integer counts. Active electrons
  must be even. `n_roots`, `rdm_root`, and `max_photons` are integer counts;
  `max_photons` is the maximum occupation, so `N_p + 1` photon sectors.
- `state_weights` are dimensionless relative weights, normalized by the engine.
  `CASSCFOptions` thresholds/floors and `CISolverOptions.threshold` retain their
  established units: energies in Eh, orbital-gradient quantities in atomic units,
  and the CI residual in the engine's normalized residual convention.
- Allowed values are checked for `method` (`cis`, `fci`, `casci`, `casscf`),
  `reference` (`rhf` only), `photon_basis` (`coherent`, `number`),
  `reference_orbitals` (`qed_rhf`, `canonical`), `spin` (`no`, `singlet`,
  `triplet`, `quintet`), `verbosity` (`normal`, `debug`, `trace`), and
  `output_mode` (`w`, `a`, `x`).
- Option precedence is: explicit `CQEDConfig` value, then the current
  `QED_*` environment setting for the nine registered CASSCF controls, then the
  established module/default value. Legacy dictionaries are converted first, so
  their historical defaults and flags remain the baseline. Conflicting or ignored
  legacy flags are recorded in `diagnostics` rather than silently reapplied.
- Unknown keys are rejected by the new config and by the legacy adapter (with a
  spelling suggestion). Type errors, nonfinite values, invalid weights, impossible
  root bounds, and contradictory options are rejected before CI work.

UHF/KS references are rejected until the correlation kernels support them. The
reference multiplicity is separate from selecting singlet/triplet/quintet excited
states using `spin`. The current dense CIS implementation always has two photon
sectors; the new API therefore requires `max_photons=1`. CASSCF requires the
iterative solver. The optional dense CASCI/FCI Hamiltonian builder likewise
requires `max_photons=1`. For CASSCF, requesting dense diagonalization would skip
orbital optimization and is rejected. `natural_orbitals` and an external orbital
guess are alternatives.
Optimized/natural orbital exports currently require CASSCF; starting orbitals can
be exported for every method using `initial_orbital_output_file`.

Types, finite values, option conflicts, weight lengths/signs, and supported methods
are checked before SCF. Geometry charge/multiplicity is checked before SCF too.
Electron/orbital capacity and root bounds are checked once the reference dimensions
exist, before CI allocation. Orbital guesses are checked for shape and finite
values before use. `resolved_options["dimensions"]` records the CI dimension and
requested root-vector storage; this is **not** a total memory prediction.

`CISolverOptions` has `algorithm="auto"` (`dense`/`davidson` alternatives),
`threshold=1e-5`, `initial_subspace=4`, `max_subspace=20`,
`max_iterations=100`, `guess="unit guess"`. Subspace controls retain the existing
backend interpretation and clamping. Alternative guess labels have no implementation
in the current Python dispatch and are rejected in new configs.

## Per-calculation CASSCF controls

All fields of `CASSCFOptions` default to `None`, meaning inherit. The nine controls
below now resolve for each calculation, instead of at module import:

| Field | Legacy environment name | Default |
|---|---|---|
| `energy_threshold` | `QED_MACRO_ENERGY_CONV` | `1e-10` |
| `gradient_threshold` | `QED_GRAD_CONV` | `1e-6` |
| `micro_gradient_floor` | `QED_MICRO_GRAD_FLOOR` | `1e-7` |
| `micro_energy_floor` | `QED_MICRO_ENERGY_FLOOR` | `1e-10` |
| `micro_gradient_scale` | `QED_MICRO_GRAD_SCALE` | `0.1` |
| `micro_energy_scale` | `QED_MICRO_ENERGY_SCALE` | `0.01` |
| `uncoupled_ci_iterations` | `QED_CI_UNCOUPLED_MAXITER` | `5` |
| `coupled_ci_iterations` | `QED_CI_COUPLED_MAXITER` | `5` |
| `coupled_ci_factor` | `QED_CI_COUPLED_FACTOR` | `0.1` |

Precedence: explicit config, current environment, established module/default
value. Settings are applied only inside a calculation scope and restored even
on exceptions. `resolved_options["effective_casscf"]` records values and origins.
The existing energy **or** gradient convergence semantics are preserved; zero
disables that macro criterion, but both cannot be disabled simultaneously.

Other experimental settings retain their existing import-time/runtime/cache
semantics. They are not silently migrated or overridden. Their environment
snapshot and available module values are recorded under `legacy_environment` and
`import_time_controls`. Use a fresh process when changing those controls, as before.
This staged boundary avoids changing unrelated solver experiments during I/O work.

## Legacy compatibility

The three-positional-argument constructor continues to work. Its two dictionaries
go through `CQEDConfig.from_legacy(psi4_options, cavity_options)`. Method spelling
is normalized and valid historical defaults are retained, including CAS's default
orbital optimization, short-weight zero-padding, and the old basis-flag precedence.
Conflicts/ignored settings are recorded in `calculation.diagnostics` and printed
inside the selected output scope. Unknown cavity keys now raise with a spelling
suggestion rather than being silently ignored. Invalid inputs raise exceptions.

`quiet`, `verbosity`, `output_file`, `output_mode`, `scratch_directory`, and all
explicit file paths can also be placed in the legacy cavity dictionary:

```python
cavity_options["quiet"] = True
cavity_options["output_file"] = "calculation.out"
calculation = PFHamiltonianGenerator(molecule, psi4_options, cavity_options)
```

Legacy calls retain trace verbosity and fixed orbital exports unless redirected
explicitly, preserving existing scripts. In particular, they still write
`orbital2.out`; use the new config for the clean file policy. The legacy adapter
diagnoses dense CIS photon truncation and dense-CASSCF inconsistencies without
changing the existing numerical route. Private compatibility fields on CQEDConfig
are not intended as user-controlled escape hatches.

## Output and file policy

| Setting | Behavior |
|---|---|
| Defaults | Conventional report on stdout; detailed engine diagnostics at debug/trace |
| `quiet=True` | No routine Python/C/Psi4 stdout or stderr during the calculation |
| `output_file="job.out"` | All calculation output goes to the file, with no console duplication |
| `quiet=True, output_file="job.out"` | Silent console, report retained |
| `output_mode="w"` | Replace the report when starting a calculation (default) |
| `output_mode="a"` | Append a new calculation |
| `output_mode="x"` | Fail if the report already exists |

Exceptions are still raised to the caller. Python and native descriptors are
routed together. Psi4 output is captured separately and included as a labelled
section at the end of the scope, avoiding multiple writers overwriting a shared
file. Normal output uses the Stage 4 report; debug/trace also retain engine diagnostics.

Every scope restores stdout/stderr, Psi4's output destination (without truncating
the caller's file), C verbosity, working directory, and scoped solver settings.
The engine and descriptor redirection use process-global state: package scopes
are serialized, but unrelated threads' prints can also be redirected. Run concurrent
calculations in separate processes. This is an explicit transitional limitation.

New-config calculations keep unsolicited engine files in a private temporary
working directory retained while the result object is alive. Supply a dedicated
`scratch_directory` to retain it explicitly. Export/guess/report paths are resolved
relative to the caller's directory when the config is created. No orbital export
is requested by default. Do not use the report path as an orbital input/output
path. Scratch isolation covers execution; Psi4's process-finalization `timer.dat`
can still appear in the process's exit directory, outside a calculation scope.
The package does not replace Psi4's global shutdown hooks.

## Gradients and additional property calls

The existing gradient subclass accepts the same config:

```python
from nuclear_grad import nuclear_grad
g = nuclear_grad(molecule, config=config)  # config.method must be casscf
g.compute_grad(0, 0)                     # automatically appends to the same report
```

State indices are checked before gradient work. For other legacy helper/property
methods, use `with calculation.output_context(): ...` to apply the same routing
and settings. Direct imports occur before a constructor can know its options;
use `calculate()` for silent lazy engine initialization, or wrap a gradient import
and construction in `qed_ci.output.calculation_output(config)`.

Result attributes retain their meanings: `CIeigs` contains the initial CI solve;
CASSCF's optimized energies are in `eigenvals`. Both are exposed explicitly by the Stage 4 result API described below.

## Verification

```sh
python -m pytest tests/test_config.py tests/test_output.py tests/test_results.py -q
python tools/validate_stage23.py --output build/stage23/replay-new
python tools/check_docs.py
python -m pytest tests/test_packaging.py -q
```

The replay checks both APIs against all eight saved Stage 1 cases: energy tolerance
`1e-8` Eh, RDM/weight tolerance `1e-7`, unchanged array shapes and finite values.
Each case runs in its own process with one thread. These tolerances allow the
already measured unseeded CASSCF variation; CI-vector signs are not compared.
The original scientific suite remains the gradient/reference correctness check.

Stages 2–3 validation: 41 new tests passed; the original suite remained at 175
passed with the same two Stage 1 failures. All 16 replay runs passed, with maximum
energy difference `8.76e-13` Eh and maximum saved RDM/weight difference `6.17e-9`.
See [the Stage 2–3 validation record](audit/stage23/results.json). Stage 4
extended the focused set to 54 tests and replayed all 16 cases with a maximum
energy difference of `6.36e-13` Eh; see
[the Stage 4 record](audit/stage4/README.md). Stage 5 packaging/install results
are in [the Stage 5 record](audit/stage5/README.md).

## Stage 4: structured scientific results

`calculate()` and the legacy constructor still return the existing calculation
object. Its new `result` attribute is a `qed_ci.CalculationResult`. No numerical
algorithm, convergence default, CI/RDM attribute, or gradient return value changes.

```python
import json
calculation = calculate(molecule, config)
r = calculation.result
print(r.energies['final'])
print(r.converged)                 # True, False, or None (unknown)
record = r.summary()
json.dumps(record, allow_nan=False)
```

Schema version **1** uses zero-based state indices throughout. Treat the result
as read-only; subsequent `compute_grad` calls add properties and timings to it.
`summary()` returns a detached JSON-compatible dictionary. Unavailable values are
`None` / JSON `null`; nonfinite diagnostic values serialize as null. Timings are
seconds. The schema is independent of report formatting.

| Field | Contract |
|---|---|
| `requested` | Configuration as supplied through the validated/legacy adapter; includes output settings and adapter diagnostics. |
| `effective` | Resolved configuration, scoped CASSCF controls with their origins, environment snapshot, and `engine` flags/dimensions. `method` records the route actually taken; `state_weights` holds normalized weights. |
| `geometry` | `symbols` (natom), `coordinates` (natom, 3) in bohr in the Psi4 input frame, charge and multiplicity. |
| `energies.reference`, `rhf_reference` | Selected orbital-reference energy and canonical RHF energy, in Eh. |
| `energies.initial_ci` | Snapshot of `CIeigs`, shape (number of returned states,). Dense paths retain their full spectrum here. |
| `energies.final`, `final_source` | Snapshot of optimized `eigenvals` when present; otherwise `CIeigs`. On an iteration cap these are the last available CI energies, **not converged energies**. |
| `energies.excitation` | `final[i] - final[0]`, in Eh, same shape as final energies. |
| `energies.state_average` | Dot product of normalized weights with the corresponding final roots, in Eh; null when not applicable. It does not reuse a potentially stale optimizer `avg_energy` attribute. |
| `convergence` | Separate `scf`, `ci_initial`, `ci_final`, and `casscf` records, with `converged`, `reason`, and available iteration counts, residuals and thresholds. |
| `properties` | Computed dipole matrix and subsequently requested gradients/NACs, with explicit units and state labels. No unrequested gradient calculations are performed. |
| `warnings`, `timings`, `provenance`, `files` | Adapter notices, captured Python warnings and nonconvergence notices; elapsed calculation/property times; package version, Python/NumPy/Psi4 versions, and backend path/SHA256; paths actually written by package export operations and the report path. |

`r.converged` combines the applicable SCF, **final** CI, and orbital-optimization
statuses: any false means false, otherwise any unknown means unknown. The initial
CI status is separately retained and warned about, but does not invalidate a
subsequently converged final solve. Exceptions still propagate; a failed
constructor does not return a fabricated result.

The CQED-RHF loop records its actual energy-and-density stopping test. For a
canonical reference, detailed Psi4 convergence diagnostics are not exposed by
this adapter and are explicitly unknown. Davidson diagnostics are captured
immediately after each relevant native solve, using its packed return flag,
mean residual, and existing iteration/residual getters. Per-root residuals are unavailable for unadapted Davidson, which does not
publish the getter storage; stale values from another solve are never reused.
Individual spin-adapted residuals beyond the backend's 64-root diagnostic capacity are null. Dense diagonalization
records successful eigensolver completion, with iteration counts and residuals
unavailable. No additional numerical solve is performed to manufacture diagnostics.

CASSCF records each macroiteration's state-average energy, energy change, and
lagged gradient norm (the existing criterion uses the previous macro's gradient).
The stopping reason distinguishes `energy`, `gradient`, `energy_and_gradient`,
`iteration_limit`,
and skipped/not-requested optimization. Iterations count macro loop entries,
including entry zero; `trigger_iteration` identifies the criterion that set the
existing stopping flag (including a flag latched on an earlier entry). History labels retain the engine's zero-based numbering.
This instrumentation preserves the existing energy **or** gradient stopping logic.
A skipped optimization is reported as such, rather than claimed converged.

The result never owns or serializes CI vectors or RDM tensors. These remain on
the calculation under their existing names/layouts. Small energy/geometry lists
and property arrays are copied. Gradients are (natom, 3) in Eh/bohr under
`gradient_state_<i>`; NACs are (natom, 3) in 1/bohr under
`nac_<i>_<j>_<full|etf>`, with the existing phase and derivative conventions.
Each property survives subsequent calls for other states. Dipoles are
(nroot, nroot, 3) in e bohr, indexed by bra, ket, xyz; their `energy_source`
identifies initial versus optimized states. A capped CASSCF can still have only
initial-state dipoles, which are labelled accordingly. Repeating the same property
request replaces that property's entry. RDM and CI-vector ownership is unchanged.

## Stage 4: report and verbosity

Normal verbosity now emits a compact report with `==> section <==` headings,
aligned tables, explicit units and zero-based state labels. It includes the
molecule/reference/cavity/CI setup, convergence diagnostics and macro history,
final requested-root energies and excitations, state-average energy, available
dipoles, warnings, total timing and generated files. Additional gradient calls
append a labelled property table to the same destination. The headings and `@`
energy markers follow CQED-SCF's formatting conventions without importing its
output module or adopting its global verbosity state.

Normal mode suppresses the legacy engine's array dumps, internal timing chatter,
raw Psi4 summary and debug labels inside the existing serialized output scope.
It writes a starting summary before the engine runs, live CQED-RHF iteration
rows, CI stage notices, and CASSCF macroiteration rows as they become available.
Canonical Psi4 SCF and native CI solves currently provide stage notices rather
than per-iteration progress. If the engine raises an exception, a failure record
is written to the same destination and the original exception is re-raised;
no successful final report is produced. Quiet operation still suppresses console
output, and file-only output retains these records.
Debug retains Python engine diagnostics and native level 2; trace retains those
plus native level 3 iteration detail. Both also include the structured report.
Legacy calls still default to trace. To get the conventional report from legacy
dictionaries, set `verbosity='normal'`. No scientific consumer should parse these
human-readable reports; use `result.summary()` instead. Standalone validation
scripts' own `ROOT`, `AVGE`, comparison and `WORST OVERALL` interfaces are unchanged.

Quiet mode, file-only output, append/exclusive modes, gradient append behavior,
and exception restoration continue to use the Stage 3 scope. No new output
configuration or logger is introduced. Python warnings raised inside the main
calculation are retained in the result. In normal mode, engine output is spooled
to temporary files and lines marked warning, error, failed, failure, or fallback
are retained in `result.warnings` and printed after the engine scope, including
on failure. Native Normal verbosity remains enabled so spin-contamination warnings
are computed. Routine text is discarded and temporary logs are removed. This is
a compatibility bridge for free-text diagnostics, not a structured scientific
data parser; unmarked or multiline diagnostic context is not reconstructed.
Debug/trace retain their existing engine logs. Direct imports
can still print before a configuration exists; `calculate()` also suppresses
import chatter at normal verbosity. Process-global routing and Psi4 shutdown
file limitations described above still apply. The files field inventories package
exports, not arbitrary Psi4 scratch files or developer dump trees.
