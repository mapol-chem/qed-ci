"""LiH polaritonic crossing search using the repository's existing MECPOptimizer.

Target: excited singlet roots 1/2; reported crossing near 1.857 A and 90 degrees.
This is an iterative, more expensive example than the single-point examples.
"""

import numpy as np

from qed_ci import CQEDConfig, CISolverOptions, CASSCFOptions
from qed_ci.output import calculation_output
from mecp_optimizer import MECPOptimizer, HARTREE2EV


# ---------------------------------------------------------
# Starting geometry: perturb the reported crossing in bond length and angle
# ---------------------------------------------------------

symbols = ["Li", "H"]
bond_length = 1.88  # Angstrom; reported crossing approximately 1.857.
theta = np.radians(85.0)  # Angle from z; reported crossing at 90 degrees.
bond = bond_length * np.array([np.sin(theta), 0.0, np.cos(theta)])
positions = np.array([[0.0, 0.0, 0.0], bond])


# ---------------------------------------------------------
# Calculation configuration (fixed cavity frequency throughout the search)
# ---------------------------------------------------------

config = CQEDConfig(
    method="casscf",
    psi4_options={"basis": "6-311g", "scf_type": "pk",
                  # Required by the existing orbital-guess orthogonalizer.
                  "restricted_docc": [0], "active": [9],
                  "e_convergence": 1e-10, "d_convergence": 1e-10},
    lambda_vector=(0.0, 0.0, 0.05),
    omega=3.086 / HARTREE2EV,  # Convert the specified eV value to Eh.
    max_photons=1,
    active_electrons=4,
    active_orbitals=9,
    spin="singlet",
    n_roots=3,
    # Equal averaging of all three roots reproduces the reported 1.857 A
    # crossing. The optimizer still targets only excited singlet roots 1/2.
    state_weights=(1.0, 1.0, 1.0),
    ci_solver=CISolverOptions(threshold=1e-9, initial_subspace=21,
                             max_subspace=31, max_iterations=100),
    casscf=CASSCFOptions(energy_threshold=1e-11, gradient_threshold=1e-7),
    orbital_output_file="lih_mecp_orbitals.txt",
    output_file="lih_mecp_electronic.out",  # Detailed progress if a step is slow.
    quiet=True,
)


# ---------------------------------------------------------
# Existing MECP search algorithm; no optimizer is implemented in this script
# ---------------------------------------------------------

optimizer = MECPOptimizer(
    symbols, positions,
    rot_axes=[(0.0, 0.0, 1.0)],  # Keep rotations that change the angle to z.
    cone_step=True,
    trust=0.10,  # Bohr, as required by MECPOptimizer.
    gap_tol=1e-6,  # Eh.
    seam_tol=5e-4,  # Eh/bohr, optimizer's Cartesian seam criterion.
    verbose=False,
)
max_steps = 30

print("LiH SA-QED-CASSCF(4,9)/6-311G polaritonic crossing: states 1 and 2")
print("Step     R / A    theta / deg     Mean E / Eh      Gap / eV     Seam norm", flush=True)
for step in range(max_steps):
    if step > 0:
        # Reuse only orbitals written by the preceding converged calculation.
        config = config.copy_with(orbital_guess_file=config.orbital_output_file,
                                  output_mode="a")
    positions = optimizer.positions_ang.copy()
    geometry = "0 1\n" + "\n".join(
        f"{symbol} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}"
        for symbol, xyz in zip(symbols, positions)
    ) + "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com"

    with calculation_output(config):
        from nuclear_grad import nuclear_grad

        calculation = nuclear_grad(geometry, config=config)
        result = calculation.result
        if result.converged is not True:
            raise RuntimeError(f"SA-CASSCF did not converge at step {step}: {result.convergence}")
        energy_1, energy_2 = result.energies["final"][1:3]
        if abs(energy_2 - energy_1) < 1e-12:
            raise RuntimeError("Numerical degeneracy: do not divide by the gap to form a NAC. "
                               "Start slightly away from the crossing.")
        calculation.compute_grad(1, 1)
        calculation.compute_grad(2, 2)
        # The existing MECP driver uses the ETF (no CSF) coupling direction.
        # This differs from the full NAC in the single-point vector example.
        calculation.compute_grad(1, 2)

    gradient_1 = result.properties["gradient_state_1"]["value"]
    gradient_2 = result.properties["gradient_state_2"]["value"]
    coupling = result.properties["nac_1_2_etf"]["value"]
    if not all(np.isfinite(value).all() for value in (gradient_1, gradient_2, coupling)):
        raise RuntimeError("Nonfinite derivative; refusing an optimizer step.")

    # step() reports convergence at the evaluated geometry and moves only if
    # not converged. Report the saved positions, not the unevaluated next guess.
    info = optimizer.step(energy_1, energy_2, gradient_1, gradient_2, coupling)
    bond = positions[1] - positions[0]
    distance = np.linalg.norm(bond)
    angle = np.degrees(np.arccos(np.clip(bond[2] / distance, -1.0, 1.0)))
    print(f"{step:4d}  {distance:9.6f}  {angle:11.6f}  "
          f"{0.5*(energy_1+energy_2):15.10f}  {info['gap_eV']:12.5e}  "
          f"{info['seam_norm']:10.3e}", flush=True)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    if info["converged"]:
        print("MECP converged (energy-gap and projected seam-gradient criteria).")
        with open("lih_mecp.xyz", "w") as output:
            output.write(f"2\nSA-QED-CASSCF roots 1/2; gap {info['gap_eV']:.8e} eV\n")
            for symbol, xyz in zip(symbols, positions):
                output.write(f"{symbol} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}\n")
        print("Saved evaluated crossing geometry to lih_mecp.xyz")
        break
else:
    raise RuntimeError(f"MECP did not converge within {max_steps} steps; no converged geometry saved.")
