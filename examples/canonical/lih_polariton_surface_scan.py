"""Scan three LiH QED-FCI polaritonic surfaces with a 6-311G basis.

The JSON file is the archival record, including the full structured result at
each geometry.  The CSV file has one row per geometry and state for plotting.
Both files are refreshed after every completed geometry so an interrupted scan
still leaves usable partial results.
"""

import csv
import json
from pathlib import Path

import numpy as np

from qed_ci import CQEDConfig, CISolverOptions, calculate


JSON_OUTPUT = Path("lih_qed_fci_scan.json")
CSV_OUTPUT = Path("lih_qed_fci_scan.csv")
BOND_LENGTHS_ANGSTROM = np.linspace(1.0, 3.0, 10)

PSI4_OPTIONS = {
    "basis": "6-311g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}

CONFIG = CQEDConfig(
    method="fci",
    reference="rhf",
    psi4_options=PSI4_OPTIONS,
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.12086,  # Eh
    max_photons=1,
    n_roots=3,
    spin="singlet",
    ci_solver=CISolverOptions(threshold=1e-7, initial_subspace=8,
                              max_subspace=24, max_iterations=100),
    quiet=True,
)


def geometry(bond_length):
    """Return an oriented LiH geometry at ``bond_length`` in angstrom."""
    return f"""
0 1
Li 0.0 0.0 0.0
H  0.0 0.0 {bond_length:.10f}
units angstrom
symmetry c1
no_reorient
no_com
"""


def write_outputs(records, rows):
    archive = {
        "schema_version": 1,
        "description": "LiH QED-FCI polariton surface scan",
        "scan_coordinate": {"name": "Li-H bond length", "units": "angstrom"},
        "configuration": CONFIG.to_dict(),
        "points": records,
    }
    JSON_OUTPUT.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "bond_length_angstrom", "state", "energy_Eh", "excitation_energy_Eh",
        "converged", "status", "warning_count",
    ]
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    records = []
    rows = []
    for bond_length in BOND_LENGTHS_ANGSTROM:
        calculation = calculate(geometry(float(bond_length)), CONFIG)
        result = calculation.result.summary()
        records.append({"bond_length_angstrom": float(bond_length), "result": result})

        for state, energy in enumerate(result["energies"]["final"][:CONFIG.n_roots]):
            rows.append({
                "bond_length_angstrom": float(bond_length),
                "state": state,
                "energy_Eh": energy,
                "excitation_energy_Eh": result["energies"]["excitation"][state],
                "converged": result["converged"],
                "status": result["status"],
                "warning_count": len(result["warnings"]),
            })

        write_outputs(records, rows)
        print(f"R = {bond_length:.6f} angstrom: {result['status']}")

    print(f"Wrote archival results to {JSON_OUTPUT}")
    print(f"Wrote plot-ready results to {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
