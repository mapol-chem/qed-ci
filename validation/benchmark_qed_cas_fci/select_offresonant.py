"""Select blue-detuned frequencies from the expanded cavity-free spectra."""
import csv
import json
from pathlib import Path

import numpy as np

from config import BRIGHT_DIPOLE_THRESHOLD, LAMBDA_MAGNITUDES, RESULTS

MAX_LAMBDA = max(LAMBDA_MAGNITUDES)
SAFETY_FACTOR = 3.0
LINE_DIPOLE_CUTOFF = BRIGHT_DIPOLE_THRESHOLD
GRID_STEP_EH = 0.001
MAX_BLUE_SHIFT_FRACTION = 3.0


def choose(record):
    energies = np.asarray(record["energies"], dtype=float)
    dipoles = np.asarray(record["dipoles"], dtype=float)
    root_mu = np.linalg.norm(dipoles[0], axis=1)
    bright = next((i for i in range(1, len(energies))
                   if root_mu[i] > BRIGHT_DIPOLE_THRESHOLD), None)
    if bright is None:
        return dict(status="no_bright_root", roots_used=len(energies))
    omega_trans = float(energies[bright] - energies[0])
    polarization = dipoles[0, bright] / root_mu[bright]
    projected = np.einsum("ijk,k->ij", dipoles, polarization)

    pairs = [(i, j) for i in range(len(energies)) for j in range(i + 1, len(energies))]
    pair_i_all = np.asarray([i for i, _ in pairs], dtype=int)
    pair_j_all = np.asarray([j for _, j in pairs], dtype=int)
    gaps_all = energies[pair_j_all] - energies[pair_i_all]
    products_all = projected[pair_i_all, :] * projected[:, pair_j_all].T
    mu_all = np.abs(projected[pair_i_all, pair_j_all])
    gaps, mu = gaps_all[mu_all >= LINE_DIPOLE_CUTOFF], mu_all[mu_all >= LINE_DIPOLE_CUTOFF]
    # Search the blue side through a full transition-energy interval. The
    # first candidate clear of both line sets at 3x the estimated coupling is
    # preferred, retaining the largest useful bright-state dressing.
    low = omega_trans + GRID_STEP_EH
    # Keep the two-photon target 2*omega inside the calculated excitation
    # energy range, so a safe result is based on observed states.
    high = min(omega_trans * (1.0 + MAX_BLUE_SHIFT_FRACTION),
               float(energies[-1] - energies[0]) / 2.0)
    if high <= low:
        return dict(status="insufficient_two_photon_spectral_coverage",
            roots_used=len(energies), bright_root=bright,
            bright_transition_eh=omega_trans, omega_candidate_eh=float("nan"),
            blue_shift_percent=float("nan"), one_photon_margin_eh="",
            two_photon_margin_eh="", nearest_one_photon_gap_eh="",
            nearest_one_photon_dipole="", nearest_two_photon_gap_eh="",
            lambda_screen=MAX_LAMBDA, safety_factor=SAFETY_FACTOR)
    candidates = np.arange(low, high + GRID_STEP_EH / 2, GRID_STEP_EH)
    safe = []
    best = None
    for omega in candidates:
        g = MAX_LAMBDA * np.sqrt(omega / 2.0) * mu
        margins_1 = np.abs(omega - gaps) - SAFETY_FACTOR * g
        margin_1 = float(np.min(margins_1)) if len(margins_1) else float("inf")

        den = omega - (energies[None, :] - energies[pair_i_all, None])
        den[np.abs(den) < 1e-10] = np.copysign(1e-10, den[np.abs(den) < 1e-10] + 1e-30)
        amp_2 = np.sum(products_all / den, axis=1)
        v_2 = np.sqrt(2.0) * MAX_LAMBDA**2 * (omega / 2.0) * np.abs(amp_2)
        margins_2 = np.abs(2.0 * omega - gaps_all) - SAFETY_FACTOR * v_2
        margin_2 = float(np.min(margins_2[v_2 > 1e-10])) if np.any(v_2 > 1e-10) else float("inf")
        score = min(margin_1, margin_2)
        detail = (float(omega), margin_1, margin_2)
        if best is None or score > best[0]:
            best = (score, detail)
        if margin_1 > 0.0 and margin_2 > 0.0:
            safe.append(detail)
            break

    if safe:
        omega, margin_1, margin_2 = safe[0]
        status = "safe_window"
    else:
        omega, margin_1, margin_2 = best[1]
        status = "no_3x_safe_window"
    nearest_1 = int(np.argmin(np.abs(gaps - omega))) if len(gaps) else None
    nearest_2 = int(np.argmin(np.abs(gaps_all / 2.0 - omega))) if len(gaps_all) else None
    return dict(status=status, roots_used=len(energies), bright_root=bright,
        bright_transition_eh=omega_trans, omega_candidate_eh=omega,
        blue_shift_percent=100.0 * (omega / omega_trans - 1.0),
        one_photon_margin_eh=margin_1, two_photon_margin_eh=margin_2,
        nearest_one_photon_gap_eh=float(gaps[nearest_1]) if nearest_1 is not None else "",
        nearest_one_photon_dipole=float(mu[nearest_1]) if nearest_1 is not None else "",
        nearest_two_photon_gap_eh=float(gaps_all[nearest_2]) if nearest_2 is not None else "",
        lambda_screen=MAX_LAMBDA, safety_factor=SAFETY_FACTOR)


def main():
    spectra = [json.loads(line) for line in
               (RESULTS / "offres_cavity_free_spectra.jsonl").read_text().splitlines()]
    rows = [{k: record[k] for k in ("system", "geometry", "basis", "active_electrons",
                                      "active_orbitals", "frozen_core_orbitals")}
            | choose(record) for record in spectra]
    output = RESULTS / "offres_frequency_screen.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['system']:5s} {r['geometry']:14s} {r['status']:20s} "
              f"ωtrans={r.get('bright_transition_eh', float('nan')):.5f} Eh "
              f"ωc={r.get('omega_candidate_eh', float('nan')):.5f} Eh "
              f"blue={r.get('blue_shift_percent', float('nan')):.1f}%")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
