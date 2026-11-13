"""Re-run lambda=0 checks away from degeneracies and audit cached traces."""
import csv
import json
import argparse
from pathlib import Path

import numpy as np

from config import RESULTS, lambda_key
from run_pilot import geom_variant, match_qed_state, rdm, root_vector, run_calc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("pilot", "full"), default="pilot")
    prefix = parser.parse_args().prefix
    path = RESULTS / f"{prefix}_validation.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    rows = [r for r in rows if r["check"] not in {
        "lambda0_qed_vs_cavity_free", "lambda0_qed_vs_cavity_free_off_resonance",
        "all_cached_1rdm_traces"}]
    # Pilot lambda=0 checks are repeated away from resonance to avoid arbitrary
    # rotations in exactly degenerate electronic/photon subspaces. Full jobs
    # already contain the same check for every configured geometry.
    if prefix == "pilot":
        for system in ("H2", "LiH"):
            basis = "6-311g"
            ne, no = ((2, 6) if system == "H2" else (4, 16))
            for geometry, scale in (("equilibrium", 1.0), ("stretched_2re", 2.0)):
                mol = geom_variant(system, scale)
                bare, _ = run_calc(mol, basis, "fci", ne, no, 0.0, (0, 0, 0), 1)
                norms = np.linalg.norm(np.asarray(bare.electronic_dipole_array)[0], axis=1)
                bright = next(j for j in range(1, len(norms)) if norms[j] > 0.05)
                test_omega = 0.321
                q0, _ = run_calc(mol, basis, "fci", ne, no, test_omega, (0, 0, 0), 2)
                for label, bare_root in (("S0", 0), ("Sb", bright)):
                    qr = match_qed_state(q0, root_vector(bare, bare_root), 0)
                    ed = float(abs(q0.CIeigs[qr] - bare.CIeigs[bare_root]))
                    rd = float(np.linalg.norm(rdm(q0, qr) - rdm(bare, bare_root)))
                    rows.append(dict(system=system, geometry=geometry,
                        check="lambda0_qed_vs_cavity_free_off_resonance", state=label,
                        energy_abs_diff=ed, rdm_frobenius=rd,
                        trace=float(np.trace(rdm(q0, qr))), passed=(ed < 1e-8 and rd < 1e-8)))
    # The cache contains each root's full electronic 1-RDM. Audit all of them.
    cache_path = RESULTS / f"{prefix}_cache.jsonl"
    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    source_path = Path(__file__).resolve().parent / "geometry_sources.csv"
    with source_path.open() as f:
        source_rows = list(csv.DictReader(f))
    for rec in records:
        total_e = int(rec.get("n_el_total", rec["n_el"]))
        active_e = int(rec.get("active_electrons", total_e))
        active_o = int(rec.get("active_orbitals", rec["nmo"]))
        rec.setdefault("n_el_total", total_e)
        rec.setdefault("n_orb_active", active_o)
        rec.setdefault("frozen_core_orbitals", max(0, (total_e - active_e) // 2))
        rec.setdefault("photon_basis", "bare_number_state")
        rec.setdefault("reference_type", "CASCI" if "casci" in str(rec.get("method", "")).lower() else "FCI")
        rec.setdefault("active_space", f"CAS({active_e},{active_o})")
        rec.setdefault("state_label", rec.get("state", "unspecified"))
        source_key = rec["system"] if rec.get("geometry") == "equilibrium" else rec.get("geometry")
        if source_key == "stretched_2re":
            source_key = f"{rec['system']}_2re"
        source_row = next((r for r in source_rows if r["system"] == source_key), None)
        if source_row:
            rec["geometry_source"] = f"{source_row['geometry_definition']}; {source_row['source']}"
        for root in rec["roots"]:
            gamma = np.asarray(root["rdm"])
            root.setdefault("trace", float(np.trace(gamma)))
            root.setdefault("natural_occupations", np.sort(np.linalg.eigvalsh(gamma))[::-1].tolist())
    with cache_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    by_system_geometry = {}
    for rec in records:
        key = (rec["system"], rec["geometry"])
        error = max(abs(float(np.trace(np.asarray(x["rdm"]))) - rec["n_el"])
                    for x in rec["roots"])
        by_system_geometry[key] = max(by_system_geometry.get(key, 0.0), error)
    for (system, geometry), error in sorted(by_system_geometry.items()):
        rows.append(dict(system=system, geometry=geometry, check="all_cached_1rdm_traces",
            state="all", energy_abs_diff="", rdm_frobenius="", trace_error_max=error,
            passed=error < 1e-8))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

    # Derive sequential basis changes from the saved root assignment and energy list.
    conv = [r for r in records if r.get("state") == "photon_convergence"]
    grouped = {}
    for rec in conv:
        grouped.setdefault((rec["system"], rec["geometry"], lambda_key(np.linalg.norm(rec["lambda_vector"]))), []).append(rec)
    conv_rows = []
    for (system, geometry, magnitude), group in sorted(grouped.items()):
        lam = group[0]["lambda_vector"]
        group.sort(key=lambda r: r["photon_states"])
        prev = None
        first_converged = ""
        for rec in group:
            lp, up = rec["lp_root"], rec["up_root"]
            ambiguous = bool(rec.get("assignment_ambiguous", False)) or min(
                float(rec.get("lp_assignment_score", 1.0)),
                float(rec.get("up_assignment_score", 1.0))) < 0.1
            cur = (float(rec["energies"][0]), float(rec["energies"][lp]), float(rec["energies"][up]))
            delta = "" if prev is None or ambiguous else max(abs(a-b) for a,b in zip(cur, prev))
            converged = delta != "" and delta < 1e-6
            if converged and not first_converged:
                first_converged = rec["photon_states"]
            conv_rows.append(dict(system=system, geometry=geometry, lambda_vector=list(lam),
                lambda_magnitude=float(np.linalg.norm(lam)), photon_states=rec["photon_states"],
                ground_energy=cur[0], lp_energy=cur[1], up_energy=cur[2],
                lp_root=lp, up_root=up, max_target_energy_change=delta,
                assignment_ambiguous=ambiguous,
                converged_from_previous=converged))
            prev = None if ambiguous else cur
        for row in conv_rows[-len(group):]:
            row["first_converged_photon_states"] = first_converged
            row["converged_by_max_basis"] = bool(first_converged)
    if conv_rows:
        with (RESULTS / f"{prefix}_photon_convergence.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(conv_rows[0]))
            writer.writeheader()
            writer.writerows(conv_rows)
        status_map = {}
        for row in conv_rows:
            key = (row["system"], row["geometry"], lambda_key(row["lambda_magnitude"]))
            if row["first_converged_photon_states"]:
                status_map[key] = "converged"
            else:
                status_map[key] = "not_converged_at_max"
        metric_path = RESULTS / f"{prefix}_metrics.csv"
        metric_rows = list(csv.DictReader(metric_path.open()))
        assignments = {}
        for rec in records:
            state = rec.get("state")
            if state not in {"photon_convergence", "polariton_roots"}:
                continue
            lam_value = lambda_key(np.linalg.norm(rec.get("lambda_vector", [0, 0, 0])))
            key = (rec["system"], rec["geometry"], lam_value,
                   int(rec.get("active_electrons", rec["n_el"])),
                   int(rec.get("active_orbitals", rec["nmo"])))
            lp_score = float(rec.get("lp_assignment_score", 1.0))
            up_score = float(rec.get("up_assignment_score", 1.0))
            assignments[key] = dict(lp_assignment_score=lp_score,
                up_assignment_score=up_score,
                assignment_ambiguous=bool(rec.get("assignment_ambiguous", False))
                    or min(lp_score, up_score) < 0.1)
        for row in metric_rows:
            lam_value = float(row["lambda_value"] or 0)
            if lam_value:
                key = (row["system"], row["geometry"], lambda_key(lam_value))
                row["status"] = status_map.get(key, "unknown")
                if row["state"] in {"LP", "UP"}:
                    assignment_key = (*key, int(row["active_electrons"]), int(row["active_orbitals"]))
                    row.update(assignments.get(assignment_key, dict(
                        lp_assignment_score="", up_assignment_score="",
                        assignment_ambiguous=True)))
                else:
                    row.update(lp_assignment_score="", up_assignment_score="",
                               assignment_ambiguous=False)
        with metric_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
                key for row in metric_rows for key in row)))
            writer.writeheader()
            writer.writerows(metric_rows)
    failed = [r for r in rows if r.get("passed") in (False, "False")]
    print(f"Validation checks: {len(rows)-len(failed)}/{len(rows)} passed")
    for r in failed:
        print("FAILED", r)
    print(f"Photon convergence groups: {len(grouped)}")


if __name__ == "__main__":
    main()
