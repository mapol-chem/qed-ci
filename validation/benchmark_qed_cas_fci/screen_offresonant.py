"""Compute wider cavity-free singlet spectra for off-resonant mode selection."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from config import BRIGHT_DIPOLE_THRESHOLD, RESULTS, STRETCHED, SYSTEMS
from run_pilot import geom_variant, run_calc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=24,
                        help="number of cavity-free CI roots to request (default: 24)")
    parser.add_argument("--system", action="append", choices=tuple(SYSTEMS),
                        help="limit rerun to named systems; may be repeated")
    args = parser.parse_args()
    if args.roots < 2:
        parser.error("--roots must be at least 2")

    with (RESULTS / "feasibility.csv").open() as f:
        feasibility = list(csv.DictReader(f))
    cases = [("equilibrium", system, 1.0) for system in SYSTEMS]
    cases.extend((name, spec["system"], spec["scale"]) for name, spec in STRETCHED.items())
    if args.system:
        cases = [case for case in cases if case[1] in args.system]
    output = RESULTS / "offres_cavity_free_spectra.jsonl"
    existing = {}
    if args.system and output.exists():
        existing = {(r["system"], r["geometry"]): r for r in
                    map(json.loads, output.read_text().splitlines())}
    for geometry, system, scale in cases:
            selected = next(r for r in feasibility
                            if r["system"] == system and r["selected"] == "True")
            basis = selected["basis"]
            n_e, n_o = int(selected["n_el"]), int(selected["n_orb"])
            n_core = int(selected["frozen_core"])
            method = "casci" if n_core else "fci"
            owner, elapsed = run_calc(geom_variant(system, scale), basis, method,
                                      n_e, n_o, 0.0, (0, 0, 0), 1,
                                      nroots=args.roots)
            record = dict(system=system, geometry=geometry, basis=basis,
                method=method, active_electrons=n_e, active_orbitals=n_o,
                frozen_core_orbitals=n_core, roots_requested=args.roots,
                root_count=len(owner.CIeigs), bright_dipole_threshold=BRIGHT_DIPOLE_THRESHOLD,
                energies=np.asarray(owner.CIeigs).tolist(),
                dipoles=np.asarray(owner.electronic_dipole_array).tolist(),
                timing_s=elapsed, ci_converged=owner.result.convergence["ci_final"].get("converged"),
                ci_diagnostics=owner.result.convergence["ci_final"])
            existing[(system, geometry)] = record
            print(f"{system} {geometry}: {record['root_count']} roots, {elapsed:.1f} s", flush=True)
    with output.open("w") as f:
        for record in existing.values():
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
