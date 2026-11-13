"""Dimension-only basis screen; no SCF or CI calculation is started."""
import csv
import math
import sys

import psi4

from config import (BASIS_PRIORITY, DIMENSION_CAP, PHOTON_STATES_FOR_SCREEN,
                    RESULTS, SYSTEMS)


def binom(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def dimension(nelec, norb):
    na = (nelec + 1) // 2
    nb = nelec // 2
    return binom(norb, na) * binom(norb, nb)


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else DIMENSION_CAP
    psi4.core.be_quiet()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in SYSTEMS.items():
        mol = psi4.geometry(spec["geometry"])
        nelec = sum(int(mol.Z(i)) for i in range(mol.natom())) - spec["charge"]
        core = spec["core_orbitals"]
        for basis in BASIS_PRIORITY:
            try:
                bset = psi4.core.BasisSet.build(mol, "BASIS", basis.upper(), puream=0)
                norb = int(bset.nbf())
            except Exception as exc:
                rows.append(dict(system=name, basis=basis, frozen_core="not applied", n_el=nelec,
                                 n_el_total=nelec,
                                 n_orb="", fci_dim="", qed_fci_dim="", selected=False,
                                 note=f"basis unavailable: {exc}"))
                continue
            frozen = 0
            nel_active, norb_active = nelec, norb
            fci_dim = dimension(nel_active, norb_active)
            if fci_dim * PHOTON_STATES_FOR_SCREEN > cap and core:
                frozen = core
                nel_active -= 2 * frozen
                norb_active -= frozen
                fci_dim = dimension(nel_active, norb_active)
            qed_dim = fci_dim * PHOTON_STATES_FOR_SCREEN
            rows.append(dict(system=name, basis=basis, frozen_core=frozen, n_el=nel_active,
                             n_el_total=nelec,
                             n_orb=norb_active, fci_dim=fci_dim, qed_fci_dim=qed_dim,
                             selected=False, note=""))
    for name in SYSTEMS:
        candidates = [r for r in rows if r["system"] == name and isinstance(r["qed_fci_dim"], int)
                      and r["qed_fci_dim"] <= cap]
        if candidates:
            selected = min(candidates, key=lambda r: BASIS_PRIORITY.index(r["basis"]))
            selected["selected"] = True
        else:
            print(f"WARNING: {name}: no candidate fits cap {cap:,}", file=sys.stderr)
    csv_path = RESULTS / "feasibility.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = RESULTS / "feasibility.md"
    with md_path.open("w") as f:
        f.write(f"# QED-FCI feasibility screen\n\nCap: **{cap:,}**; screen assumes {PHOTON_STATES_FOR_SCREEN} photon states. When needed, frozen core means CASCI with all remaining orbitals active and the listed core orbitals doubly occupied.\n\n")
        f.write("| System | Basis | Frozen core (orbitals) | n_el(active) | n_el(total) | n_orb(active) | FCI dim | QED-FCI dim | Selected |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for r in rows:
            f.write("| {system} | {basis} | {frozen_core} | {n_el} | {n_el_total} | {n_orb} | {fci_dim} | {qed_fci_dim} | {selected} |\n".format(**r))
    print(md_path.read_text())
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
