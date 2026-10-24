"""Ground-state QED-CASSCF for LiH/STO-3G in a single cavity mode."""

from qed_ci import CQEDConfig, CISolverOptions, calculate


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
    state_weights=(1.0, 0.0),  # Orbital optimization weights.
    ci_solver=CISolverOptions(threshold=1e-9),
    quiet=True,  # Only the explicit result prints below reach the console.
    # output_file="calculation.out",  # Optional detailed report, even when quiet.
)


# ---------------------------------------------------------
# Run calculation and print key results
# ---------------------------------------------------------

calculation = calculate(geometry, config)

result = calculation.result
print("\nGround-state QED-CASSCF")
print("=" * 23)
print(f"Converged : {result.converged}")
print(f"Reference : {result.energies['reference']:.12f} Eh")
for state, energy in enumerate(result.energies["final"][:config.n_roots]):
    excitation = result.energies["excitation"][state]
    print(f"State {state}   : {energy:.12f} Eh   excitation {excitation:.8f} Eh")
print(f"Weighted energy : {result.energies['state_average']:.12f} Eh")
for warning in result.warnings:
    print(f"Warning: {warning}")
