# Canonical QED-CI examples

These standalone Python scripts follow the cqed-scf canonical-example style:
geometry, Psi4 options, `CQEDConfig`, calculation, and a few explicit result
prints. There are no command-line arguments or shared example helpers. Edit a
script's configuration directly to explore it.

The single-point examples use LiH at 1.4 angstrom in STO-3G, with a z-polarized cavity
coupling of 0.05 atomic units, frequency 0.12 Eh, and photon occupations 0 and 1.
The geometry is held fixed relative to the cavity polarization. These small
systems illustrate the interface; the basis, active space, and photon cutoff
are not convergence recommendations for production calculations.

| Script | Calculation and key results |
|---|---|
| [qed_cis_energy.py](qed_cis_energy.py) | QED-CIS total and excitation energies for the first two states |
| [qed_fci_energy.py](qed_fci_energy.py) | QED-FCI in the full molecular orbital space, within the chosen photon cutoff |
| [lih_polariton_surface_scan.py](lih_polariton_surface_scan.py) | Three LiH QED-FCI polaritonic surfaces over a bond-length scan, with archival JSON and plot-ready CSV output |
| [mghplus_casscf_orbital_restart.py](mghplus_casscf_orbital_restart.py) | MgH+ SA-QED-CASSCF stretch with converged orbital checkpoints passed to each neighboring geometry |
| [qed_casci_energy.py](qed_casci_energy.py) | QED-CASCI(2,2) with fixed CQED-RHF orbitals |
| [qed_casscf_energy.py](qed_casscf_energy.py) | Ground-state QED-CASSCF(2,2); two roots calculated, weights `(1, 0)` for orbital optimization |
| [sa_qed_casscf_energy.py](sa_qed_casscf_energy.py) | Equal-weight two-state SA-QED-CASSCF(2,2), individual and weighted energies |
| [sa_qed_casscf_energy_gradient.py](sa_qed_casscf_energy_gradient.py) | SA-QED-CASSCF energies and the analytic **state 0** gradient at the SA-optimized orbitals, not the weighted-average gradient |
| [sa_qed_casscf_coupling_vectors.py](sa_qed_casscf_coupling_vectors.py) | Equal-weight states 0/1: full derivative coupling and gradient difference `grad(E1) - grad(E0)` |
| [lih_polaritonic_mecp.py](lih_polaritonic_mecp.py) | Iterative crossing search for excited singlet roots 1/2, using the existing `MECPOptimizer`; see the distinct setup below |

Install QED-CI in a compatible Psi4 environment using the
[installation guide](../../docs/installation.md), then run, for example:

```sh
python examples/canonical/qed_casci_energy.py
python examples/canonical/sa_qed_casscf_energy_gradient.py
```

An installed package works from any directory; use the script's absolute path
when running elsewhere. The scripts use `quiet=True` so their explicit result
prints are easy to read. Uncomment `output_file` for a calculation report, or
set `quiet=False` for normal progress and reporting. The gradient example uses
the supported `nuclear_grad` subclass and scopes its import to silence engine
initialization. It does not introduce a new Calculator interface.

Results come from `calculation.result`, not stdout parsing. Energies are in Eh;
gradients are in Eh/bohr, with atom rows in geometry order and x/y/z columns.
CASSCF examples use the final optimized energies rather than the initial CI
roots. Inspect the printed convergence status and warnings before using results.
CIS computes a full spectrum internally; the example prints only two states.
Psi4 can still create `timer.dat` at process exit; run in a scratch directory if
you want to keep example folders free of run artifacts.

The LiH surface scan is deliberately more substantial than the single-point
examples. It uses QED-FCI/6-311G at ten bond lengths from 1.0 to 3.0 angstrom,
with three singlet roots, a z-polarized coupling of 0.05 atomic units and cavity
frequency 0.12086 Eh. Run it from the directory where you want its output:

```sh
python /path/to/qed-ci/examples/canonical/lih_polariton_surface_scan.py
```

It writes `lih_qed_fci_scan.json` and `lih_qed_fci_scan.csv` after every
completed geometry. The JSON contains the full configuration and versioned
structured result for each point; the CSV has one row per geometry and state
for direct plotting. Root indices are energy ordered and this compact example
does not perform overlap-based state tracking through crossings.

The coupling-vector example uses `deriv_type="full"`, including the CSF/overlap
contribution, in the engine's working photon frame. Its derivative coupling has
units bohr^-1 and an arbitrary overall sign from the relative CI phases. The
gradient difference is in Eh/bohr and has no factor of one half. Both gradients
come from the same equal-weight SA calculation; saved result properties retain
their values across subsequent derivative calls. This is a finite-gap example,
not an evaluation of the singular derivative coupling at an exact crossing.
The MECP driver's ETF coupling convention is distinct from this full coupling.

See the [configuration and results reference](../../docs/configuration-and-output.md)
for options and restrictions, and the [quickstart scripts](../quickstart/README.md)
for command-line selection of methods.

## MgH+ orbital restart scan

Run the self-contained example in a directory where you want its output:

```sh
python /path/to/qed-ci/examples/canonical/mghplus_casscf_orbital_restart.py
```

The example uses MgH+ (charge +1, singlet, 12 electrons), cc-pVDZ, CAS(4,4),
three equally weighted singlet roots, and 20 bond lengths from 1.0 to 3.5 angstrom.
The cavity frequency and coupling are illustrative. Change these settings,
the active space, weights and grid to match your own calculation; this is a
restart demonstration, not a reproduction of a published potential curve.
CAS(8,12) is more expensive and can be selected by changing the two active-space
counts. The inactive count is derived as `(12 - active_electrons) // 2`.

The first geometry starts from CQED-RHF orbitals. Each subsequent geometry
sets `orbital_guess_file` to the preceding converged point's
`orbital_output_file`. Every point has a separate checkpoint and detailed log
inside a newly created `mghplus_restart_*` directory. `energies.csv` contains
the final optimized energies; `scan.json` retains configurations and structured
results, including an unconverged result if the scan stops. Only converged
points go into the CSV or supply the next guess. For a new invocation starting
from an existing checkpoint, set `previous_orbitals` to its absolute path and
edit `bond_lengths` to begin at the next geometry. Keep the basis, atom order,
orientation, active space and calculation settings compatible with that file.

The example also requires Matplotlib (`python -m pip install matplotlib`).
After the scan completes, it writes `surfaces.png` alongside the CSV: all
computed roots on a shared total-energy scale, with a marker at each geometry
and straight segments between neighbors. No smoothing is applied, so jumps
remain visible. Open the saved image to inspect the surfaces. The plotting
block reads `energies.csv` and can be rerun separately by setting `run_directory`
to an existing output folder and `n_roots` to the number of computed roots.

**Both active-space specifications are required.** `active_electrons` and
`active_orbitals` configure QED-CI; `restricted_docc` and `active` in
`psi4_options` configure the existing Psi4-based orthogonalizer. For this
12-electron molecule CAS(4,4) needs `[4]` / `[4]`, while CAS(8,12) needs
`[2]` / `[12]`. These settings are passed to Psi4 by the calculation; no separate
`psi4.set_options` call is needed with this example. Use C1 symmetry and zero
frozen orbital counts. The restart file contains the full MO coefficient
matrix in text format, not a Psi4 wavefunction file or CI-vector checkpoint.

The orthogonalizer uses the **new geometry's overlap matrix**: it orthogonalizes
the inactive orbitals, projects them out of the active orbitals, then projects
both spaces out of the virtual orbitals. It orthogonalizes each space and
semicanonicalizes within the spaces using the new reference Fock matrix.
The loaded orbitals therefore need not be identical column by column to the
starting orbitals actually used. It does not perform a cross-basis projection;
use the same basis and small geometry steps.

Orbital continuation helps retain an orbital solution, but the roots remain
energy ordered. It does not guarantee smooth curves or follow an electronic
state by character through a crossing. Inspect state character and convergence,
keep the state-average weights and root count fixed, and try smaller steps and
scans in both directions if discontinuities remain. A different root entering
the averaged set can also change the optimized solution. For a character-following
curve, additional state tracking is needed; this script does not implement it.

### Compatibility with the older dictionary interface

The restart check in `ortho_script.py` must count frozen orbitals by summing
the entries of Psi4's `Dimension`. Older code used `Dimension.n()`, which counts
symmetry blocks and rejects even `[0]` frozen spaces with Psi4 1.10. The fix
accompanying this example uses `sum(nfdoccpi) + sum(nfuoccpi)`; installations
with the older check need this correction before the example can restart.

The refactored code still accepts `"use_orbital_guess": True` and
`"save_orbital": True` in the legacy cavity dictionary. They default to reading
and writing `orbital.out`, respectively. Both can be enabled together: the input
is read during setup and the optimized orbitals are written on CASSCF convergence.
That overwrites the prior checkpoint. Explicit `orbital_guess_file` and
`orbital_output_file` paths can also be supplied in the legacy dictionary.
The new `CQEDConfig` interface uses the paths directly and requests no orbital
exports by default. Optimized orbital export is a CASSCF feature; a CASCI run
does not produce a new optimized checkpoint.

The coherent-state dipole expectation is taken from a fresh CQED-RHF calculation
at **each geometry**, then kept fixed during that geometry's CASSCF solve. Loading
orbitals updates `d_cmo = C.T @ d_ao @ C` and does not replace that reference
expectation with the dipole of the loaded orbitals. It does not carry the prior
geometry's dipole constant along the scan.

The older suggestion to uncomment a random orbital rotation is unnecessary in
the current code: `QED_RANDOM_ORBITAL_SEED` already enables a seeded random
rotation in `parseArrays`. This is a separate tool for exploring initial guesses,
not state tracking. Leave it unset for ordinary continuation. An external
orbital guess is loaded after that rotation and supersedes it.

## Validation

The MgH+ restart example was run on native macOS with Psi4 1.10 on
2026-11-11. All three coupled CAS(4,4)/cc-pVDZ points (1.60, 1.65 and 1.70
angstrom) reported convergence, and the latter two exercised the orbital
restart path. The default grid was subsequently extended to 20 points from
1.0 to 3.5 angstrom; that full grid has not been validated here. The plotting
block was checked using the saved three-point results. This validates the
demonstration, not a production active space or a longer stretch.
The orthogonalizer regression checks orthonormality at
the new geometry with and without semicanonicalization, and verifies that
nonzero frozen occupied or virtual counts are still rejected.

All six scripts were run from separate scratch directories on native macOS in
the local `p4dev` environment on 2026-10-21. Every script completed successfully
and reported convergence; the gradient example produced a finite 2-by-3 state 0
gradient. The source-distribution contract check confirmed that all six scripts
are included and that the documentation links resolve after extraction. No
scientific engine code was changed for these examples.

The coupling-vector example was added and run on 2026-10-22. Equal weights,
convergence, finite 2-by-3 vectors, the gradient-difference definition, and
`(E1-E0) * d_01 = h0` for the full-coupling convention were checked. Its two-state
gap is approximately 0.119943885 Eh for the supplied geometry; this example is
not intended to be a resonant polaritonic crossing geometry.

## LiH polaritonic crossing search

This example has a distinct, larger setup: SA-QED-CASSCF(4e,9o)/6-311G,
three singlet roots with equal weights `(1, 1, 1)` (normalized internally), z-polarized coupling
`(0, 0, 0.05)`, and a fixed frequency of **3.086 eV**, converted to Eh in the
script. It targets the first and second excited singlet roots (indices 1 and 2).
The supplied reference crossing is approximately **1.857 angstrom and 90 degrees
to z**. The initial guess is deliberately displaced to 1.88 angstrom and
85 degrees; the reference coordinates are not imposed as constraints.

The script calls the repository's existing `mecp_optimizer.MECPOptimizer` with
its existing two-vector cone step. Only translations and rotation about z are
projected out; changing the angle to the polarization remains possible. One
SA-CASSCF calculation supplies both state gradients and the ETF coupling at each
geometry. The full-CSF coupling in the single-point vector example is not used
as a substitute. The derivative coupling is gap-divided and is singular at an
exact degeneracy; this example starts off the crossing and stops at a finite
gap tolerance. It refuses to divide by a numerically zero gap.

Run from a scratch directory using the installed package and the script's path.
Reinstall after updating this checkout so `mecp_optimizer` is installed too.
It requires no ASE and has no machine-specific import paths. This search is
more expensive than the single-point examples. It prints the evaluated bond
length, angle, mean energy, gap and projected seam-gradient norm. It writes
`lih_mecp.xyz` only on convergence; exhausting the step budget or electronic
nonconvergence raises an error instead of claiming a successful search.
`lih_mecp_electronic.out` records the electronic progress and properties, and
`lih_mecp_orbitals.txt` supplies the previous converged orbital guess for the
next geometry, following the existing scan driver's continuation approach.
The first step starts fresh and replaces those two files; subsequent steps
append to the electronic log and update the orbitals.

The root indices are energy ordered; this compact example does not implement
overlap-based state tracking. A small gap and optimizer convergence alone are
not a complete characterization of polaritonic state character or conical
topology. Inspect nearby displacements and state character before interpreting
a different parameter set as the same crossing. The explicit weights here
average all three computed roots while the optimizer targets only roots 1/2.
This differs from the two-root average in the single-point vector example.
With the same other settings, averaging only the target pair with `(0, 0.5, 0.5)`
instead moves the optimized crossing to approximately 1.851126 angstrom.

On 2026-10-22 the three-root example converged in five evaluated geometries
from 1.88 angstrom / 85 degrees to **1.857129 angstrom / 89.999885 degrees**,
with a final gap of **5.79e-7 eV**. These are observed values, not hard-coded
targets or a promise of bitwise reproducibility. The existing optimizer source
was not changed. Electronic convergence, a finite-gap stopping criterion and
an explicit iteration limit are checked by the script.
Four converged checks at bond displacements of +/-0.002 angstrom and angular
displacements of +/-0.2 degrees reopened the gap in both directions.
The [validation record](../../docs/audit/canonical-lih-mecp.json) includes the
search trace, final coordinates and the comparison of averaging choices.
