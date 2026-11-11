"""MgH+ bond stretch with converged CASSCF orbitals passed between geometries.

Run with the installed QED-CI package in a Psi4 environment. Edit the settings
below to match your calculation. CAS(4,4) is a small restart demonstration,
not the active space or cavity parameters of a particular published curve.
"""

import csv
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Save plots on workstations and headless compute nodes.
import matplotlib.pyplot as plt
import psi4

from qed_ci import CQEDConfig, CISolverOptions, CASSCFOptions, calculate


# Scan in order, starting from a geometry with a satisfactory orbital solution.
# Use smaller steps around difficult regions; reverse this list to scan inward.
bond_lengths = np.linspace(1.0, 3.5, 20)  # Angstrom
active_electrons, active_orbitals = 4, 4
n_roots = 3
state_weights = (1.0, 1.0, 1.0)  # Equal-weight SA; keep fixed along the scan.

# MgH+ has 12 electrons: inactive doubly occupied orbitals = (12 - Nact)/2.
# For CAS(8,12), change the counts above to 8, 12; restricted_docc becomes [2].
inactive_orbitals = (12 - active_electrons) // 2
psi4_options = {
    "basis": "cc-pvdz",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    # These DETCI spaces are ALSO needed by the restart orthogonalizer.
    "restricted_docc": [inactive_orbitals],
    "active": [active_orbitals],
    "frozen_docc": [0],
    "frozen_uocc": [0],
}

psi4.set_num_threads(1)
psi4.set_memory("2 GB")
base = CQEDConfig(
    method="casscf",
    psi4_options=psi4_options,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals,
    n_roots=n_roots,
    state_weights=state_weights,
    spin="singlet",
    lambda_vector=(0.0, 0.0, 0.05),  # Atomic units; set to zero for no coupling.
    omega=0.12086,  # Eh; illustrative, not fitted to MgH+.
    max_photons=1,
    reference_orbitals="qed_rhf",
    photon_basis="coherent",
    ci_solver=CISolverOptions(threshold=1e-9, initial_subspace=8,
                             max_subspace=30, max_iterations=200),
    casscf=CASSCFOptions(energy_threshold=1e-10, gradient_threshold=1e-6),
    quiet=True,
    verbosity="debug",
)

# A new directory per invocation prevents a stale checkpoint from being reused.
run_directory = Path(mkdtemp(prefix="mghplus_restart_", dir=Path.cwd()))
print(f"Outputs: {run_directory}", flush=True)
previous_orbitals = None  # Or an absolute path to a compatible converged file.
records = []

with (run_directory / "energies.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["R_angstrom", "state", "energy_Eh"])
    stream.flush()
    for point, distance in enumerate(bond_lengths):
        geometry = f"""
1 1
Mg 0.0 0.0 0.0
H  0.0 0.0 {distance:.12f}
units angstrom
symmetry c1
no_reorient
no_com
"""
        orbitals = run_directory / f"point_{point:03d}_orbitals.txt"
        config = base.copy_with(
            orbital_guess_file=previous_orbitals,
            orbital_output_file=orbitals,
            output_file=run_directory / f"point_{point:03d}.out",
        )
        print(f"R = {distance:.4f} A; guess = {previous_orbitals or 'CQED-RHF'}",
              flush=True)
        calculation = calculate(geometry, config)
        result = calculation.result
        records.append({"R_angstrom": distance, "config": config.to_dict(),
                        "result": result.summary()})
        (run_directory / "scan.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.converged is not True:
            raise RuntimeError(f"Unconverged point at R={distance}; see {config.output_file}. "
                               "Stopping before passing its orbitals to the next point.")
        energies = np.asarray(result.energies["final"])[:n_roots]
        if not np.isfinite(energies).all() or not orbitals.is_file():
            raise RuntimeError("Missing converged orbitals or nonfinite energies; stopping.")
        for state, energy in enumerate(energies):
            writer.writerow([distance, state, float(energy)])
        stream.flush()
        for warning in result.warnings:
            print(f"Warning: {warning}", flush=True)
        print("  Final energies / Eh:", " ".join(f"{e:.10f}" for e in energies), flush=True)
        previous_orbitals = str(orbitals)  # Set only after a successful point.

# Roots are energy ordered. Orbital reuse is not overlap-based state tracking.

# Plot the saved energies so the figure can also be regenerated from the CSV.
# Mark every calculated point and connect neighbors without smoothing, which
# could hide a discontinuity. All surfaces share the same total-energy scale.
data = np.genfromtxt(run_directory / "energies.csv", delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for state in range(n_roots):
    surface = data[data["state"] == state]
    ax.plot(surface["R_angstrom"], surface["energy_Eh"], "o-",
            markersize=4, label=f"State {state}")
ax.set_xlabel("Mg–H distance (Å)")
ax.set_ylabel("Total energy (Eh)")
ax.set_title("MgH⁺ SA-QED-CASSCF potential energy surfaces")
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.grid(alpha=0.3)
ax.legend()
plot_file = run_directory / "surfaces.png"
fig.savefig(plot_file, dpi=200)
plt.close(fig)
print(f"Saved surfaces: {plot_file}", flush=True)
