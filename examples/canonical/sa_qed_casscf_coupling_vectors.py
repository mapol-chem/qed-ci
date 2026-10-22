"""Derivative coupling and gradient difference of the lowest two SA-QED-CASSCF states."""

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
    if calculation.result.converged is not True:
        raise RuntimeError("SA-CASSCF must converge before evaluating the vectors.")
    calculation.compute_grad(0, 0)
    calculation.compute_grad(1, 1)
    calculation.compute_grad(0, 1, deriv_type="full")  # Include the CSF/overlap term.

result = calculation.result
gradient_0 = result.properties["gradient_state_0"]["value"]
gradient_1 = result.properties["gradient_state_1"]["value"]
gradient_difference = gradient_1 - gradient_0  # grad(E1 - E0), no factor of 1/2.
derivative_coupling = result.properties["nac_0_1_full"]["value"]
energy_gap = result.energies["final"][1] - result.energies["final"][0]

print("\nEqual-weight SA-QED-CASSCF: states 0 and 1")
print("========================================")
print(f"Converged : {result.converged}")
print(f"State 0   : {result.energies['final'][0]:.12f} Eh")
print(f"State 1   : {result.energies['final'][1]:.12f} Eh")
print(f"Gap E1-E0 : {energy_gap:.12f} Eh")
print("Rows: Li, H; columns: x, y, z")
print("Gradient difference grad(E1)-grad(E0) / Eh bohr^-1:")
print(gradient_difference)
print("Derivative coupling d_01 (full, with CSF) / bohr^-1:")
print(derivative_coupling)
# The NAC sign depends on the relative phases of the two CI states.
# This finite-gap example does not evaluate the singular NAC at a degeneracy.
for warning in result.warnings:
    print(f"Warning: {warning}")
