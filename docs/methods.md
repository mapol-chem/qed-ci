# Supported methods and limitations

The option names and defaults are defined once in
[configuration and output](configuration-and-output.md). This page explains what
each method does, when to use it, and which restrictions are enforced before CI
work starts.

## Method map

| `method` | Meaning | Orbital optimization | Typical use |
|---|---|---|---|
| `cis` | QED-CIS. The current implementation always diagonalizes the dense CIS Hamiltonian. | No | Fast qualitative excited states, small basis sets. |
| `fci` | QED-FCI in the reference orbital space. | No | Benchmark correlated energies in a small orbital space. |
| `casci` | One CI diagonalization in a fixed active space. | No | Correlated energies/state properties without orbital relaxation. |
| `casscf` | State-averaged QED-CASSCF. Repeats the active-space CI solve inside an orbital-optimization macroiteration. | Yes | Orbital-relaxed energies, state averages, and analytic gradients. |

`method` is case-insensitive in the legacy adapter. The new `CQEDConfig` API uses
the lower-case spellings above. `calculate()` and the legacy
`PFHamiltonianGenerator` constructor return the same established calculation
object; `CQEDConfig` is an input layer, not a replacement algorithm.

## Reference and basis choices

- Only closed-shell restricted references are implemented: RHF and CQED-RHF.
  UHF/ROHF, RKS/UKS, and unrestricted references are rejected.
- The molecular multiplicity must be 1. Excited-state spin selection is separate:
  `spin` may be `no`, `singlet`, `triplet`, or `quintet` where the CI path
  supports it; CIS in the new API uses `spin="no"`.
- The supported representation pairing is `reference_orbitals="qed_rhf"` with
  `photon_basis="coherent"`, or `reference_orbitals="canonical"` with
  `photon_basis="number"`. Other combinations are rejected.
- `ignore_dse_terms=True` requires the photon-number basis in the new API.
- `omega` is the cavity frequency in atomic units (Eh) and `lambda_vector` is the
  three-component coupling in atomic units. Zero coupling is allowed; it is not
  the same switch as `ignore_coupling`.

## Active spaces and roots

- `casci` and `casscf` require `active_electrons` and `active_orbitals`. The
  active electron count must be even and fit in the active orbitals.
- State indices in results are zero-based. `n_roots` is the number of CI roots
  requested, and `rdm_root` selects the root used by RDM-consuming operations.
- `state_weights` is optional. When supplied it must have exactly `n_roots`
  entries, all finite and non-negative, with a positive finite total. The engine
  normalizes the weights. Unequal weights are supported for state-averaged
  CASSCF and are tested against saved references.
- CASCI is a single fixed-orbital diagonalization; it is not a state-averaged
  orbital optimization. Use CASSCF when the active-space orbitals themselves
  should relax for a weighted state average.

Representation and solver limits that are easy to miss:

- New `CQEDConfig` CIS calculations require `max_photons=1`; the current dense CIS
  implementation always uses two photon sectors.
- The optional dense CASCI/FCI Hamiltonian builder requires `max_photons=1`.
- CASSCF requires the iterative CI solver. Requesting dense diagonalization with
  `method="casscf"` is rejected because it would skip orbital optimization.
- Iterative CI needs more Hilbert-space dimension than the number of requested
  roots. Requesting every state of a small space is rejected.
- `natural_orbitals` and `orbital_guess_file` are alternatives, not additive.
  Optimized/natural orbital exports are available only for CASSCF; starting
  orbitals can be exported for every method.

## State averaging

State-averaged CASSCF is selected with `method="casscf"` and weights:

```python
from qed_ci import CQEDConfig

config = CQEDConfig(
    method="casscf",
    active_electrons=2,
    active_orbitals=2,
    n_roots=2,
    state_weights=(0.5, 0.5),
)
```

The result records normalized weights under `effective["state_weights"]`,
per-root final energies under `energies["final"]`, and the weighted average under
`energies["state_average"]`. Equal weights are the default when
`state_weights=None`. The legacy `casscf_weight` key is migrated with its
historical zero-padding behavior; conflicts and ignored settings are reported in
`calculation.diagnostics` and in the result warnings.

## Analytic gradients and derivative couplings

Analytic response is available only for a CASSCF calculation. The supported
public entry point is the existing `nuclear_grad` subclass, which accepts the
same `CQEDConfig`:

```python
from qed_ci import CQEDConfig
from qed_ci.output import calculation_output
from nuclear_grad import nuclear_grad

config = CQEDConfig(method="casscf", active_electrons=2, active_orbitals=2,
                    n_roots=2, state_weights=(0.5, 0.5))
with calculation_output(config):
    job = nuclear_grad("Li\nH 1 1.4\nsymmetry c1", config=config)
job.compute_grad(0, 0)                 # same state -> nuclear gradient
job.compute_grad(0, 1, deriv_type="full")  # different states -> derivative coupling
```

Restrictions enforced by the current API:

- `method` must be `casscf`; CIS/FCI/CASCI do not have analytic gradients.
- Both state indices must be valid zero-based roots below `n_roots`.
- `compute_grad(i, i)` fills `properties["gradient_state_i"]`, in Eh/bohr.
- `compute_grad(i, j)` with `i != j` fills a derivative-coupling property in
  1/bohr. `deriv_type=None` uses the existing ETF form; `"full"` includes the
  CSF/overlap term. The phase and derivative conventions are the established
  engine conventions, not a new convention introduced by `CQEDConfig`.
- A result from a capped or nonconverged CASSCF run is still returned, but its
  energies are the last available CI energies, not converged energies. Check
  `result.converged` and the convergence records before using a gradient.
- Gradients are more expensive than fixed-orbital CASCI and use the same
  process-global output routing. Run independent jobs in separate processes when
  concurrent console/file isolation matters.

## Explicitly out of scope in this stage

- A `Calculator` dispatcher or a replacement for `PFHamiltonianGenerator`.
- cqed-scf reference wavefunctions or cross-package calculator interoperability.
- Additional reference types, functionals, unrestricted references, or TAMM
  distributed tensors.
- Changes to scientific algorithms, numerical defaults, convergence thresholds,
  or supported reference types.
