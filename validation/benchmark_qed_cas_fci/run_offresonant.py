"""Run off-resonant QED-CAS/QED-FCI checks for spectrally screened cases."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from config import (BRIGHT_DIPOLE_THRESHOLD, LAMBDA_MAGNITUDES,
                    PHOTON_STATE_COUNTS, RESULTS, SYSTEMS, STRETCHED)
from run_pilot import (cache_calc, geom_variant, match_qed_state, photon_number,
                       rdm, rdm_metrics, root_vector, run_calc, ENERGY_TOL)

ROOTS = 8
CHARACTER_MIN = 0.5


def case_list():
    cases = [("equilibrium", name, 1.0) for name in SYSTEMS]
    cases.extend((name, spec["system"], spec["scale"]) for name, spec in STRETCHED.items())
    return cases


def find_bright(owner):
    norms = np.linalg.norm(np.asarray(owner.electronic_dipole_array)[0], axis=1)
    return next((i for i in range(1, len(norms))
                 if norms[i] > BRIGHT_DIPOLE_THRESHOLD), None)


def overlap_score(owner, root, bare_vector):
    target = np.zeros(int(owner.num_det) * (int(owner.N_p) + 1))
    target[:int(owner.num_det)] = bare_vector
    return float(abs(np.dot(np.asarray(owner.CIvecs[root]), target)) ** 2)


def bare_vector(owner, root):
    return root_vector(owner, root)[:int(owner.num_det)]


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (RESULTS / "feasibility.csv").open() as f:
        feasibility = list(csv.DictReader(f))
    with (RESULTS / "offres_frequency_screen.csv").open() as f:
        screen = list(csv.DictReader(f))
    safe_cases = {(r["system"], r["geometry"]): r for r in screen
                  if r["status"] == "safe_window"}
    full_metrics = list(csv.DictReader((RESULTS / "full_metrics.csv").open()))
    bare_cache = [json.loads(line) for line in
                  (RESULTS / "full_cache.jsonl").read_text().splitlines()]
    cache_path = RESULTS / "offres_cache.jsonl"
    cache_path.write_text("")
    metrics_rows, convergence_rows, validation_rows = [], [], []

    for geometry, system, scale in case_list():
        selection = safe_cases.get((system, geometry))
        if selection is None:
            continue
        spec = SYSTEMS[system]
        fr = next(r for r in feasibility if r["system"] == system and r["selected"] == "True")
        basis, n_e, n_o = fr["basis"], int(fr["n_el"]), int(fr["n_orb"])
        n_core = int(fr["frozen_core"])
        method = "casci" if n_core else "fci"
        omega = float(selection["omega_candidate_eh"])
        mol = geom_variant(system, scale)

        # Canonical cavity-free full-space spectrum and bright root.
        bare = next(r for r in bare_cache if r.get("system") == system
                    and r.get("geometry") == geometry
                    and r.get("state") == "cavity_free_spectrum"
                    and r.get("method") in ("fci", "casci")
                    and int(r.get("active_orbitals", 0)) == n_o)
        fci0, tfci0 = run_calc(mol, basis, method, n_e, n_o, 0.0,
                               (0, 0, 0), 1, nroots=24)
        bright = find_bright(fci0)
        if bright is None:
            validation_rows.append(dict(system=system, geometry=geometry,
                check="offres_bright_root", passed=False, status="no_bright_root"))
            continue
        bare_ground, bare_bright = root_vector(fci0, 0), root_vector(fci0, bright)

        # Same-frequency lambda=0 identity check in the smallest photon basis.
        q0, tq0 = run_calc(mol, basis, method, n_e, n_o, omega,
                           (0, 0, 0), 2, nroots=max(ROOTS, bright + 1))
        for label, root, vector in (("ground", 0, bare_ground),
                                    ("bright_dressed", bright, bare_bright)):
            qr = match_qed_state(q0, vector, 0)
            de = float(abs(q0.CIeigs[qr] - fci0.CIeigs[root]))
            dr = float(np.linalg.norm(rdm(q0, qr) - rdm(fci0, root)))
            validation_rows.append(dict(system=system, geometry=geometry,
                check="offres_lambda0_identity", state=label, omega=omega,
                energy_abs_diff=de, rdm_frobenius=dr,
                passed=de < 1e-8 and dr < 1e-8))

        # Cache the cavity-free states used for root tracking and RDM baselines.
        cache_path.open("a").write(json.dumps(cache_calc(fci0, dict(
            system=system, geometry=geometry, method=method,
            lambda_vector=[0, 0, 0], omega=0.0, state="cavity_free_reference",
            basis=basis, active_electrons=n_e, active_orbitals=n_o), tfci0)) + "\n")

        # Active spaces mirror the original fixed valence/+2/+4 rule.
        active_spaces = []
        for row in full_metrics:
            if (row["system"] == system and row["geometry"] == geometry
                    and row["status"] == "cavity_free" and row["state"] == "S0"):
                pair = (int(row["active_electrons"]), int(row["active_orbitals"]))
                if pair[1] < n_o and pair not in active_spaces:
                    active_spaces.append(pair)
        cas_bare = {}
        for ae, ao in active_spaces:
            c0, tc0 = run_calc(mol, basis, "casci", ae, ao, 0.0,
                               (0, 0, 0), 1)
            cb = find_bright(c0)
            cas_bare[(ae, ao)] = (c0, cb, tc0)

        for magnitude in LAMBDA_MAGNITUDES:
            lam_vec = tuple(float(x) for x in (0.0, 0.0, magnitude))
            previous = None
            chosen_nstates = PHOTON_STATE_COUNTS[-1]
            converged = False
            final_fci = None
            final_roots = None
            for nstates in PHOTON_STATE_COUNTS:
                qfci, t_fci = run_calc(mol, basis, method, n_e, n_o,
                    omega, lam_vec, nstates, nroots=max(ROOTS, bright + 1))
                rg = match_qed_state(qfci, bare_ground, 0)
                rb = match_qed_state(qfci, bare_bright, 0)
                current = (float(qfci.CIeigs[rg]), float(qfci.CIeigs[rb]))
                sg = overlap_score(qfci, rg, bare_ground)
                sb = overlap_score(qfci, rb, bare_bright)
                convergence_rows.append(dict(system=system, geometry=geometry,
                    lambda_value=magnitude, omega=omega, photon_states=nstates,
                    ground_root=rg, bright_root=rb, ground_energy=current[0],
                    bright_energy=current[1], ground_overlap=sg, bright_overlap=sb,
                    ground_photon_number=photon_number(qfci, rg),
                    bright_photon_number=photon_number(qfci, rb),
                    converged_from_previous=(previous is not None and
                        max(abs(current[i] - previous[i]) for i in range(2)) < ENERGY_TOL)))
                cache_path.open("a").write(json.dumps(cache_calc(qfci, dict(
                    system=system, geometry=geometry, method="fci",
                    lambda_vector=list(lam_vec), omega=omega,
                    state="offres_photon_convergence", basis=basis,
                    active_electrons=n_e, active_orbitals=n_o,
                    ground_root=rg, bright_root=rb,
                    ground_overlap=sg, bright_overlap=sb,
                    screen_status=selection["status"]), t_fci)) + "\n")
                final_fci, final_roots = qfci, (rg, rb)
                if previous is not None and max(abs(current[i] - previous[i]) for i in range(2)) < ENERGY_TOL:
                    chosen_nstates, converged = nstates, True
                    break
                previous = current

            rg, rb = final_roots
            fg, fb = rdm(fci0, 0), rdm(fci0, bright)
            fci_ground_overlap = overlap_score(final_fci, rg, bare_ground)
            fci_bright_overlap = overlap_score(final_fci, rb, bare_bright)
            for ae, ao in active_spaces:
                cas0, cb, tc0 = cas_bare[(ae, ao)]
                cground = rdm(cas0, 0)
                cf_g, cn_g = rdm_metrics(cground, fg)
                qcas, t_cas = run_calc(mol, basis, "casci", ae, ao,
                    omega, lam_vec, chosen_nstates,
                    nroots=max(ROOTS, (cb + 1) if cb is not None else ROOTS))
                cas_ground_vector = bare_vector(cas0, 0)
                qcg = match_qed_state(qcas, cas_ground_vector, 0)
                qfg = rdm(final_fci, rg)
                qg = rdm(qcas, qcg)
                qfrob_g, qn_g = rdm_metrics(qg, qfg)
                metrics_rows.append(dict(system=system, geometry=geometry,
                    basis=basis, active_electrons=ae, active_orbitals=ao,
                    active_space=f"CAS({ae},{ao})", state="ground",
                    lambda_value=magnitude, omega=omega,
                    cavity_free_frobenius=cf_g, qed_frobenius=qfrob_g,
                    ratio=qfrob_g / cf_g if cf_g > 0 else "",
                    cavity_free_natural_occupation_error=cn_g,
                    qed_natural_occupation_error=qn_g,
                    natural_occupation_ratio=qn_g / cn_g if cn_g > 0 else "",
                    energy=float(qcas.CIeigs[qcg]),
                    photon_number=photon_number(qcas, qcg),
                    trace=float(np.trace(qg)), photon_states=chosen_nstates,
                    timing_s=t_cas, status="converged" if converged else "not_converged_at_max",
                    fci_bright_overlap=fci_bright_overlap,
                    cas_bright_overlap="", bright_dominated=""))
                if cb is None:
                    for key in ("bright_dressed",):
                        metrics_rows.append(dict(system=system, geometry=geometry,
                            basis=basis, active_electrons=ae, active_orbitals=ao,
                            active_space=f"CAS({ae},{ao})", state=key,
                            lambda_value=magnitude, omega=omega,
                            photon_states=chosen_nstates,
                            status="bright_state_not_in_active_space"))
                    continue
                cas_bright_vector = bare_vector(cas0, cb)
                qcb = match_qed_state(qcas, cas_bright_vector, 0)
                qbright = rdm(qcas, qcb)
                cf_b, cn_b = rdm_metrics(rdm(cas0, cb), fb)
                qfrob_b, qn_b = rdm_metrics(qbright, rdm(final_fci, rb))
                sb_cas = overlap_score(qcas, qcb, cas_bright_vector)
                bright_dominated = (fci_bright_overlap >= CHARACTER_MIN
                                    and sb_cas >= CHARACTER_MIN
                                    and photon_number(final_fci, rb) < 0.5
                                    and photon_number(qcas, qcb) < 0.5)
                metrics_rows.append(dict(system=system, geometry=geometry,
                    basis=basis, active_electrons=ae, active_orbitals=ao,
                    active_space=f"CAS({ae},{ao})", state="bright_dressed",
                    lambda_value=magnitude, omega=omega,
                    cavity_free_frobenius=cf_b, qed_frobenius=qfrob_b,
                    ratio=qfrob_b / cf_b if cf_b > 0 else "",
                    cavity_free_natural_occupation_error=cn_b,
                    qed_natural_occupation_error=qn_b,
                    natural_occupation_ratio=qn_b / cn_b if cn_b > 0 else "",
                    energy=float(qcas.CIeigs[qcb]),
                    photon_number=photon_number(qcas, qcb),
                    trace=float(np.trace(qbright)), photon_states=chosen_nstates,
                    timing_s=t_cas, status="converged" if converged else "not_converged_at_max",
                    fci_bright_overlap=fci_bright_overlap,
                    cas_bright_overlap=sb_cas,
                    bright_dominated=bright_dominated))
                cache_path.open("a").write(json.dumps(cache_calc(qcas, dict(
                    system=system, geometry=geometry, method="casci",
                    lambda_vector=list(lam_vec), omega=omega,
                    state="offres_roots", basis=basis,
                    active_electrons=ae, active_orbitals=ao,
                    ground_root=qcg, bright_root=qcb,
                    ground_overlap=overlap_score(qcas, qcg, cas_ground_vector),
                    bright_overlap=sb_cas,
                    screen_status=selection["status"]), t_cas)) + "\n")

    identity = {(r["system"], r["geometry"], r["state"]): bool(r["passed"])
                for r in validation_rows if r.get("check") == "offres_lambda0_identity"}
    for row in metrics_rows:
        state = "bright_dressed" if row["state"] == "bright_dressed" else "ground"
        passed = identity.get((row["system"], row["geometry"], state), "")
        row["lambda0_state_identity_pass"] = passed
        if state == "bright_dressed" and passed is False:
            row["bright_dominated"] = False
    write_csv(RESULTS / "offres_metrics.csv", metrics_rows)
    write_csv(RESULTS / "offres_convergence.csv", convergence_rows)
    write_csv(RESULTS / "offres_validation.csv", validation_rows)
    print(f"Wrote off-resonant outputs to {RESULTS}")
    print(f"λ=0 checks passed: {sum(r.get('passed') is True for r in validation_rows)}"
          f"/{sum(r.get('check') == 'offres_lambda0_identity' for r in validation_rows)}")


if __name__ == "__main__":
    main()
