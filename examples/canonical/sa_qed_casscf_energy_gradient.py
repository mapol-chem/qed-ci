"""State-averaged QED-CASSCF energy and gradient for LiH/STO-3G in a single cavity mode."""

from qed_ci import CQEDConfig, CISolverOptions
from qed_ci.output import calculation_output


# ---------------------------------------------------------
# Geometry (angstrom; keep its orientation relative to the cavity)
# ---------------------------------------------------------

geometry = """
0 1
Li 0.0 0.0 0.0
H  0.0 0.0 1.4
units angstrom
symmetry c1
no_reorient
no_com
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4_options = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    method="casscf",
    reference="rhf",
    psi4_options=psi4_options,
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.12,  # Eh
    max_photons=1,  # Photon occupations 0 and 1.
    n_roots=2,
    active_electrons=2,
    active_orbitals=2,  # CAS(2,2); the Li core stays doubly occupied.
    state_weights=(0.5, 0.5),  # Orbital optimization weights.
    ci_solver=CISolverOptions(threshold=1e-9),
    quiet=True,  # Only the explicit result prints below reach the console.
    # output_file="calculation.out",  # Optional detailed report, even when quiet.
)


# ---------------------------------------------------------
# Run calculation and print key results
# ---------------------------------------------------------

# Gradients currently use the supported gradient subclass. Scope its import
# as well as the calculation to keep engine initialization quiet.
with calculation_output(config):
    from nuclear_grad import nuclear_grad

    calculation = nuclear_grad(geometry, config=config)
    calculation.compute_grad(0, 0)  # State 0 gradient at the SA-optimized orbitals.

result = calculation.result
print("\nState-averaged QED-CASSCF energy and gradient")
print("=" * 45)
print(f"Converged : {result.converged}")
print(f"Reference : {result.energies['reference']:.12f} Eh")
for state, energy in enumerate(result.energies["final"][:config.n_roots]):
    excitation = result.energies["excitation"][state]
    print(f"State {state}   : {energy:.12f} Eh   excitation {excitation:.8f} Eh")
print(f"Weighted energy : {result.energies['state_average']:.12f} Eh")
print("State 0 gradient / Eh bohr^-1 (rows: Li, H; columns: x, y, z):")
print(result.properties["gradient_state_0"]["value"])
for warning in result.warnings:
    print(f"Warning: {warning}")
