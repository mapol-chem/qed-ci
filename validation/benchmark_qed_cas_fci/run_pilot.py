"""Run the H2/LiH pilot, including their stretched geometries."""
import csv
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from helper_PFCI import c_build_one_rdm
from qed_ci import CQEDConfig, CISolverOptions, calculate

from config import (BRIGHT_DIPOLE_THRESHOLD, FIGURES, LAMBDA_MAGNITUDES,
                    PHOTON_STATE_COUNTS, RESULTS, SYSTEMS, STRETCHED)

ROOTS = 8
ENERGY_TOL = 1.0e-6
RUN_PREFIX = "pilot"


def geom_variant(name, scale=1.0):
    spec = SYSTEMS[name]
    lines = spec["geometry"].splitlines()
    atom_indices = [i for i, line in enumerate(lines)
                    if len(line.split()) == 4 and line.split()[0].isalpha()]
    if scale != 1.0:
        x0 = np.array([float(x) for x in lines[atom_indices[0]].split()[1:]])
        for i in atom_indices[1:]:
            fields = lines[i].split()
            xyz = np.array([float(x) for x in fields[1:]])
            fields[1:] = [f"{x:.10f}" for x in (x0 + scale * (xyz - x0))]
            lines[i] = " ".join(fields)
    return "\n".join(lines)


def run_calc(mol, basis, method, active_e, active_o, omega, lam, nstates, nroots=ROOTS):
    # The public max_photons setting is the highest occupation; nstates counts
    # the number-state basis functions, including vacuum.
    ci_dim = (math.comb(active_o, active_e // 2) ** 2 if method == "casci"
              else None)
    algorithm = "davidson"
    if ci_dim is not None:
        # The dense builder requires at least the vacuum and one-photon state;
        # use that smallest supported basis for the one-determinant CAS case.
        if ci_dim == 1 and nstates == 1:
            nstates = 2
        max_photons = nstates - 1
        ci_dim *= nstates
        nroots = min(nroots, ci_dim - 1)
    else:
        max_photons = nstates - 1
    cfg = CQEDConfig(
        method=method, psi4_options={"basis": basis, "scf_type": "pk",
                                     "e_convergence": 1e-10, "d_convergence": 1e-10},
        reference_orbitals="canonical", photon_basis="number",
        max_photons=max_photons, active_electrons=active_e if method == "casci" else None,
        active_orbitals=active_o if method == "casci" else None,
        lambda_vector=tuple(float(x) for x in lam), omega=float(omega),
        n_roots=max(1, nroots),
        ci_solver=CISolverOptions(algorithm=algorithm, threshold=1e-9,
                                  initial_subspace=10, max_subspace=max(48, nroots + 8),
                                  max_iterations=200),
        quiet=True, scratch_directory=str(RESULTS / "scratch"))
    started = time.perf_counter()
    owner = calculate(mol, cfg)
    return owner, time.perf_counter() - started


def root_vector(owner, root):
    # Davidson stores one eigenvector per row; all pilot calculations request
    # the iterative solver so this convention is unambiguous.
    return np.asarray(owner.CIvecs[root], dtype=float)


def photon_number(owner, root):
    vector = root_vector(owner, root)
    n_det = int(owner.num_det)
    expectation = 0.0
    for p in range(int(owner.N_p) + 1):
        block = vector[p * n_det:(p + 1) * n_det]
        expectation += p * float(np.dot(block, block))
    return expectation


def rdm(owner, root):
    nocc = int(owner.n_occupied)
    data = np.zeros(nocc * nocc)
    c_build_one_rdm(owner.CIvecs, owner.CIvecs, data, owner.table,
                    owner.n_act_a, owner.n_act_orb, owner.n_in_a,
                    int(owner.N_p) + 1, root, root, False)
    gamma = np.zeros((int(owner.nmo), int(owner.nmo)))
    gamma[:nocc, :nocc] = data.reshape(nocc, nocc)
    return gamma


def occupations(gamma):
    return np.sort(np.linalg.eigvalsh(gamma))[::-1]


def rdm_metrics(gamma, reference):
    delta = gamma - reference
    return (float(np.linalg.norm(delta)),
            float(np.max(np.abs(occupations(gamma) - occupations(reference)))))


def match_qed_state(owner, bare_vector, photon_sector=0):
    n_det = int(owner.num_det)
    target = np.zeros(n_det * (int(owner.N_p) + 1))
    target[photon_sector * n_det:(photon_sector + 1) * n_det] = bare_vector
    overlaps = np.abs(np.asarray(owner.CIvecs) @ target)
    return int(np.argmax(overlaps))


def polariton_roots(owner, bare_ground, bare_bright):
    n_det = int(owner.num_det)
    vectors = np.asarray(owner.CIvecs)
    ground_char = np.abs(vectors[:, :n_det] @ bare_ground) ** 2
    bright_char = np.abs(vectors[:, :n_det] @ bare_bright) ** 2
    photon_ground_char = np.zeros(len(vectors))
    if owner.N_p >= 1:
        photon_ground_char = np.abs(vectors[:, n_det:2 * n_det] @ bare_ground) ** 2
    score = bright_char + photon_ground_char
    score[0] = -1.0
    chosen = np.argsort(score)[-2:]
    chosen = chosen[np.argsort(np.asarray(owner.CIeigs)[chosen])]
    return int(chosen[0]), int(chosen[1]), score


def cache_calc(owner, meta, seconds):
    nstates = int(owner.N_p) + 1
    roots = []
    for root, energy in enumerate(np.asarray(owner.CIeigs)):
        gamma = rdm(owner, root)
        roots.append({"root": root, "energy": float(energy),
                      "photon_number": photon_number(owner, root),
                      "trace": float(np.trace(gamma)),
                      "natural_occupations": occupations(gamma).tolist(),
                      "rdm": gamma.tolist()})
    n_el_total = int(2 * owner.ndocc)
    active_e = meta.get("active_electrons", n_el_total)
    active_o = meta.get("active_orbitals", int(owner.nmo))
    system = meta.get("system", "")
    geometry = meta.get("geometry", "equilibrium")
    source_file = Path(__file__).resolve().parent / "geometry_sources.csv"
    with source_file.open() as f:
        source_rows = list(csv.DictReader(f))
    source_key = system if geometry == "equilibrium" else geometry
    if geometry == "stretched_2re":
        source_key = f"{system}_2re"
    source_row = next((r for r in source_rows if r["system"] == source_key), None)
    geometry_source = (f"{source_row['geometry_definition']}; {source_row['source']}"
                       if source_row else "geometry source not found")
    route = "CASCI" if "casci" in str(meta.get("method", "")).lower() else "FCI"
    return {**meta, "reference_type": route,
            "state_label": meta.get("state", "unspecified"),
            "geometry_source": meta.get("geometry_source", geometry_source),
            "frozen_core_orbitals": max(0, (n_el_total - int(active_e)) // 2),
            "active_space": f"CAS({active_e},{active_o})",
            "n_el_total": n_el_total, "n_orb_active": int(active_o),
            "photon_basis": "bare_number_state", "timing_s": seconds, "photon_states": nstates,
            "energies": np.asarray(owner.CIeigs).tolist(), "roots": roots,
            "dipoles": np.asarray(owner.electronic_dipole_array).tolist(),
            "nmo": int(owner.nmo), "n_el": n_el_total,
            "ci_dimension": int(owner.CIvecs.shape[-1]),
            "converged": owner.result.converged,
            "ci_converged": owner.result.convergence["ci_final"].get("converged"),
            "ci_diagnostics": owner.result.convergence["ci_final"],
            "warnings": owner.result.warnings}


def save_cache(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{RUN_PREFIX}_cache.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def selected_feasibility(system):
    with (RESULTS / "feasibility.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["system"] == system and r["selected"] == "True")
    return (row["basis"], int(row["n_el"]), int(row["n_orb"]),
            int(row["frozen_core"]), int(row["n_el_total"]))


def full_geometry_cases():
    cases = [("equilibrium", system, 1.0) for system in SYSTEMS]
    for key, spec in STRETCHED.items():
        cases.append((key, spec["system"], spec["scale"]))
    return cases


def main():
    global RUN_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run all configured systems/geometries")
    parser.add_argument("--confirm-full", action="store_true", help="explicitly authorize starting the full run")
    args = parser.parse_args()
    RUN_PREFIX = "full" if args.full else "pilot"
    if args.full and not args.confirm_full:
        parser.error("--full requires --confirm-full after review of feasibility.md")
    if args.full:
        with (Path(__file__).resolve().parent / "geometry_sources.csv").open() as f:
            source_rows = list(csv.DictReader(f))
        unverified = [r["system"] for r in source_rows
                      if r["system"] in SYSTEMS and not r["verification_status"].startswith("verified")]
        if unverified:
            raise RuntimeError("Full run blocked until geometry sources are verified: " + ", ".join(unverified))
        case_list = full_geometry_cases()
    else:
        case_list = [("equilibrium", "H2", 1.0), ("stretched_2re", "H2", 2.0),
                     ("equilibrium", "LiH", 1.0), ("stretched_2re", "LiH", 2.0)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    validation_rows = []
    convergence_rows = []
    for geometry_name, system, scale in case_list:
        spec = SYSTEMS[system]
        mol = geom_variant(system, scale)
        basis, nelec, norb_total, frozen_core, total_e = selected_feasibility(system)
        reference_method = "casci" if frozen_core else "fci"
        # First get the cavity-free FCI spectrum and its lowest bright root.
        fci0, t_fci0 = run_calc(mol, basis, reference_method, nelec, norb_total,
                                0.0, (0, 0, 0), 1)
        norms = np.linalg.norm(np.asarray(fci0.electronic_dipole_array)[0], axis=1)
        bright_candidates = [j for j in range(1, len(norms))
                             if norms[j] > BRIGHT_DIPOLE_THRESHOLD]
        if not bright_candidates and len(norms) < 24:
            fci0, t_fci0 = run_calc(mol, basis, reference_method, nelec, norb_total,
                                    0.0, (0, 0, 0), 1, nroots=24)
            norms = np.linalg.norm(np.asarray(fci0.electronic_dipole_array)[0], axis=1)
            bright_candidates = [j for j in range(1, len(norms))
                                 if norms[j] > BRIGHT_DIPOLE_THRESHOLD]
        if not bright_candidates:
            failure = dict(system=system, geometry=geometry_name, state="bright_state_not_found",
                           roots_examined=len(norms), threshold=BRIGHT_DIPOLE_THRESHOLD,
                           transition_dipole_norms=norms.tolist(), status="no_bright_root")
            save_cache(failure)
            print(f"SKIP {system}/{geometry_name}: no bright root above threshold among {len(norms)} roots")
            continue
        bright = bright_candidates[0]
        omega = float(fci0.CIeigs[bright] - fci0.CIeigs[0])
        mu = np.asarray(fci0.electronic_dipole_array)[0, bright]
        direction = mu / np.linalg.norm(mu)
        ground_vec = root_vector(fci0, 0)
        bright_vec = root_vector(fci0, bright)
        fci0_record = cache_calc(fci0, dict(system=system, geometry=geometry_name,
            method="fci", lambda_vector=[0, 0, 0], omega=0.0, state="cavity_free_spectrum",
            basis=basis, active_electrons=nelec, active_orbitals=norb_total,
            geometry_source="NIST CCCBDB equilibrium geometry; stretched by stated factor"), t_fci0)
        save_cache(fci0_record)

        # The all-active CASCI route is the package's frozen-core-capable FCI
        # route; here no core is frozen for H2 and LiH.
        allcas, t_allcas = run_calc(mol, basis, "casci", nelec, norb_total,
                                    0.0, (0, 0, 0), 1, nroots=bright + 1)
        for label, root in (("S0", 0), ("Sb", bright)):
            rcas, rfci = rdm(allcas, root), rdm(fci0, root)
            validation_rows.append(dict(system=system, geometry=geometry_name,
                check="all_active_casci_vs_fci", state=label,
                energy_abs_diff=float(abs(allcas.CIeigs[root] - fci0.CIeigs[root])),
                rdm_frobenius=float(np.linalg.norm(rcas - rfci)),
                trace=float(np.trace(rcas)), passed=(abs(allcas.CIeigs[root] - fci0.CIeigs[root]) < 1e-8
                    and np.linalg.norm(rcas-rfci) < 1e-8)))
        save_cache(cache_calc(allcas, dict(system=system, geometry=geometry_name,
            method="casci_all_active_validation", lambda_vector=[0, 0, 0], omega=0.0,
            state="cavity_free_spectrum", basis=basis, active_electrons=nelec,
            active_orbitals=norb_total, geometry_source="NIST CCCBDB; geometry variant per metadata"), t_allcas))

        # Lambda=0 must recover the electronic root in the photon vacuum.
        # Keep this lambda=0 identity check away from the resonance used in
        # the coupling study; exact degeneracy permits arbitrary root mixing.
        omega_test = 0.321
        q0, t_q0 = run_calc(mol, basis, reference_method, nelec, norb_total,
                            omega_test, (0, 0, 0), 2, nroots=max(ROOTS, bright + 1))
        for label, bare_root, bare_vec in (("S0", 0, ground_vec), ("Sb", bright, bright_vec)):
            qroot = match_qed_state(q0, bare_vec, 0)
            gdiff = float(np.linalg.norm(rdm(q0, qroot) - rdm(fci0, bare_root)))
            ediff = float(abs(q0.CIeigs[qroot] - fci0.CIeigs[bare_root]))
            validation_rows.append(dict(system=system, geometry=geometry_name,
                check="lambda0_qed_vs_cavity_free_off_resonance", state=label,
                energy_abs_diff=ediff, rdm_frobenius=gdiff,
                passed=(ediff < 1e-8 and gdiff < 1e-8)))
        save_cache(cache_calc(q0, dict(system=system, geometry=geometry_name,
            method="fci", lambda_vector=[0, 0, 0], omega=omega_test, state="lambda0_validation",
            basis=basis, active_electrons=nelec, active_orbitals=norb_total,
            geometry_source="NIST CCCBDB; geometry variant per metadata"), t_q0))

        # Per molecule active spaces: valence, valence+2, valence+4, omitting
        # any space that already spans all orbitals.
        val_e = int(spec["valence_electrons"])
        val_o = int(spec["valence_orbitals"])
        active_spaces = [(val_e, val_o), (val_e, val_o + 2), (val_e, val_o + 4)]
        active_spaces = list(dict.fromkeys((e, o) for e, o in active_spaces
                                           if o < norb_total and e <= nelec))
        cas_bare = {}
        for ae, ao in active_spaces:
            cas0, tc0 = run_calc(mol, basis, "casci", ae, ao, 0.0, (0, 0, 0), 1)
            nb = int(np.asarray(cas0.electronic_dipole_array).shape[1])
            bnorm = np.linalg.norm(np.asarray(cas0.electronic_dipole_array)[0], axis=1)
            bright_cas_candidates = [j for j in range(1, min(nb, len(bnorm)))
                                     if bnorm[j] > BRIGHT_DIPOLE_THRESHOLD]
            bc = bright_cas_candidates[0] if bright_cas_candidates else None
            cas_bare[(ae, ao)] = (cas0, bc)
            comparisons = [("S0", 0, 0)]
            if bc is not None:
                comparisons.append(("Sb", bc, bright))
            for label, rcas, rfci in comparisons:
                gcas = rdm(cas0, rcas)
                gfci = rdm(fci0, rfci)
                frob, nocc = rdm_metrics(gcas, gfci)
                metrics_rows.append(dict(system=system, geometry=geometry_name,
                    basis=basis, active_electrons=ae, active_orbitals=ao,
                    active_space=f"CAS({ae},{ao})", state=label, lambda_value=0.0,
                    cavity_free_frobenius=frob, qed_frobenius="", ratio="",
                    cavity_free_natural_occupation_error=nocc,
                    qed_natural_occupation_error="", natural_occupation_ratio="",
                    energy=float(cas0.CIeigs[rcas]), photon_number=0.0,
                    trace=float(np.trace(gcas)), photon_states=1, timing_s=tc0,
                    status="cavity_free"))
            save_cache(cache_calc(cas0, dict(system=system, geometry=geometry_name,
                method="casci", lambda_vector=[0, 0, 0], omega=0.0,
                active_electrons=ae, active_orbitals=ao, state="cavity_free_spectrum",
                basis=basis), tc0))

        # FCI photon truncation study for all couplings. The same converged
        # basis size is subsequently used for every CASCI space.
        for magnitude in LAMBDA_MAGNITUDES:
            lam = direction * magnitude
            converged_states = None
            previous = None
            chosen_nstates = PHOTON_STATE_COUNTS[-1]
            photon_converged = False
            for nstates in PHOTON_STATE_COUNTS:
                qfci, tfci = run_calc(mol, basis, reference_method, nelec, norb_total,
                                      omega, lam, nstates)
                lp, up, scores = polariton_roots(qfci, ground_vec, bright_vec)
                target = {"ground": 0, "LP": lp, "UP": up}
                energies = {key: float(qfci.CIeigs[idx]) for key, idx in target.items()}
                assignment_ambiguous = min(float(scores[lp]), float(scores[up])) < 0.1
                convergence_rows.append(dict(system=system, geometry=geometry_name,
                    lambda_value=float(magnitude), photon_states=nstates,
                    ground_energy=energies["ground"], lp_energy=energies["LP"],
                    up_energy=energies["UP"], lp_root=lp, up_root=up,
                    lp_assignment_score=float(scores[lp]), up_assignment_score=float(scores[up]),
                    assignment_ambiguous=assignment_ambiguous,
                    lp_photon_number=photon_number(qfci, lp),
                    up_photon_number=photon_number(qfci, up),
                    assignment_criterion="top two projections onto bare bright/0-photon and ground/1-photon character"))
                save_cache(cache_calc(qfci, dict(system=system, geometry=geometry_name,
                    method="fci", lambda_vector=lam.tolist(), omega=omega,
                    active_electrons=nelec, active_orbitals=norb_total,
                    state="photon_convergence", basis=basis, lp_root=lp, up_root=up,
                    lp_assignment_score=float(scores[lp]), up_assignment_score=float(scores[up]),
                    assignment_ambiguous=assignment_ambiguous,
                    assignment_criterion="two largest bright-vacuum + ground-one-photon projection scores"), tfci))
                if (not assignment_ambiguous and previous is not None
                        and max(abs(energies[k] - previous[k]) for k in energies) < ENERGY_TOL):
                    converged_states = target
                    chosen_nstates = nstates
                    photon_converged = True
                    break
                previous = None if assignment_ambiguous else energies
                converged_states = target
            for ae, ao in active_spaces:
                cas0, bc = cas_bare[(ae, ao)]
                qcas, tcas = run_calc(mol, basis, "casci", ae, ao,
                                      omega, lam, chosen_nstates)
                if bc is None:
                    gc, gf = rdm(qcas, 0), rdm(qfci, converged_states["ground"])
                    qed_frob, qed_no = rdm_metrics(gc, gf)
                    cavity_frob, cavity_no = rdm_metrics(rdm(cas0, 0), rdm(fci0, 0))
                    metrics_rows.append(dict(system=system, geometry=geometry_name,
                        basis=basis, active_electrons=ae, active_orbitals=ao,
                        active_space=f"CAS({ae},{ao})", state="ground",
                        lambda_value=float(magnitude), cavity_free_frobenius=cavity_frob,
                        qed_frobenius=qed_frob,
                        ratio=(qed_frob/cavity_frob if cavity_frob > 0 else ""),
                        cavity_free_natural_occupation_error=cavity_no,
                        qed_natural_occupation_error=qed_no,
                        natural_occupation_ratio=(qed_no/cavity_no if cavity_no > 0 else ""),
                        energy=float(qcas.CIeigs[0]), photon_number=photon_number(qcas, 0),
                        trace=float(np.trace(gc)), photon_states=chosen_nstates,
                        timing_s=tcas, status="converged" if photon_converged else "not_converged_at_max"))
                    for state in ("LP", "UP"):
                        metrics_rows.append(dict(system=system, geometry=geometry_name,
                            basis=basis, active_electrons=ae, active_orbitals=ao,
                            active_space=f"CAS({ae},{ao})", state=state,
                            lambda_value=float(magnitude), cavity_free_frobenius="",
                            qed_frobenius="", ratio="",
                            cavity_free_natural_occupation_error="",
                            qed_natural_occupation_error="", natural_occupation_ratio="",
                            energy="", photon_number="", trace="", photon_states=chosen_nstates,
                            timing_s=tcas, status="bright_state_not_in_active_space"))
                    save_cache(cache_calc(qcas, dict(system=system, geometry=geometry_name,
                        method="casci", lambda_vector=lam.tolist(), omega=omega,
                        active_electrons=ae, active_orbitals=ao, state="bright_state_not_in_active_space",
                        basis=basis, assignment_criterion="not assigned: active space has no cavity-free bright root"), tcas))
                    continue
                gvec, bvec = root_vector(cas0, 0), root_vector(cas0, bc)
                clp, cup, cscores = polariton_roots(qcas, gvec, bvec)
                for label, casroot, fciroot, casbare, fcibare in (
                    ("ground", 0, converged_states["ground"], 0, 0),
                    ("LP", clp, converged_states["LP"], bc, bright),
                    ("UP", cup, converged_states["UP"], bc, bright)):
                    gc, gf = rdm(qcas, casroot), rdm(qfci, fciroot)
                    qed_frob, qed_no = rdm_metrics(gc, gf)
                    cavity_frob, cavity_no = rdm_metrics(rdm(cas0, casbare), rdm(fci0, fcibare))
                    metrics_rows.append(dict(system=system, geometry=geometry_name,
                        basis=basis, active_electrons=ae, active_orbitals=ao,
                        active_space=f"CAS({ae},{ao})", state=label,
                        lambda_value=float(magnitude), cavity_free_frobenius=cavity_frob,
                        qed_frobenius=qed_frob,
                        ratio=(qed_frob/cavity_frob if cavity_frob > 0 else ""),
                        cavity_free_natural_occupation_error=cavity_no,
                        qed_natural_occupation_error=qed_no,
                        natural_occupation_ratio=(qed_no/cavity_no if cavity_no > 0 else ""),
                        energy=float(qcas.CIeigs[casroot]), photon_number=photon_number(qcas, casroot),
                        trace=float(np.trace(gc)), photon_states=chosen_nstates,
                        timing_s=tcas, status="converged" if photon_converged else "not_converged_at_max",
                        assignment_score=(float(cscores[casroot]) if label in ("LP", "UP") else ""),
                        assignment_ambiguous=(bool(cscores[casroot] < 0.1) if label in ("LP", "UP") else False)))
                save_cache(cache_calc(qcas, dict(system=system, geometry=geometry_name,
                    method="casci", lambda_vector=lam.tolist(), omega=omega,
                    active_electrons=ae, active_orbitals=ao, state="polariton_roots",
                    basis=basis, lp_root=clp, up_root=cup,
                    lp_assignment_score=float(cscores[clp]), up_assignment_score=float(cscores[cup]),
                    assignment_ambiguous=bool(min(cscores[clp], cscores[cup]) < 0.1),
                    assignment_criterion="two largest bright-vacuum + ground-one-photon projection scores"), tcas))

    for file_name, rows in ((f"{RUN_PREFIX}_metrics.csv", metrics_rows),
                            (f"{RUN_PREFIX}_validation.csv", validation_rows),
                            (f"{RUN_PREFIX}_photon_convergence.csv", convergence_rows)):
        if rows:
            with (RESULTS / file_name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Wrote {RUN_PREFIX} outputs to {RESULTS}")
    print(f"Validation checks passed: {sum(bool(r['passed']) for r in validation_rows)}/{len(validation_rows)}")


if __name__ == "__main__":
    main()
