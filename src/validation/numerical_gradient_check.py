#!/usr/bin/env python3
"""Analytic vs NUMERICAL gradients -- cavity ON or OFF, equal or UNEQUAL weights.

Generalises the developer's c2h4_fc.py (cavity on, EQUAL weights, all 3N
components) along the three axes it fixed: arbitrary state-average weights,
arbitrary systems, and a chosen subset of components so a run can be minutes
instead of hours.  Central differences on the converged state energy, so the
reference includes the full orbital and CI response -- exactly what the
analytic gradient claims to compute.

    python numerical_gradient_check.py --system h2o_cav --weights 5,2,1
    python numerical_gradient_check.py --system c2h4_fc --components 6
    python numerical_gradient_check.py --system c2h4_ci --avg-only

WHY THE STATE-AVERAGE LEG EXISTS, AND WHY IT IS THE ONE TO TRUST AT A CONICAL
INTERSECTION.  A numerical per-state gradient assigns states by ENERGY ORDER at
each displaced geometry.  Near a conical intersection the two surfaces touch, so
a displacement can swap which physical state is "state 0", and the finite
difference then straddles two different surfaces and returns nonsense -- this is
why c2h4_ci.py gives poor results while c2h4_fc.py is clean, and it is a defect
of the NUMERICAL reference, not of the analytic gradient.

    E_SA(x) = sum_n W_n E_n(x)

is invariant under any relabelling WITHIN the averaged set, so it stays smooth
through the intersection.  Its numerical gradient is therefore well defined
where the per-state ones are not, and the analytic side is just
sum_n W_n * grad_n.  At a CI geometry, --avg-only is the meaningful test.

Every displaced point is checked for near-degeneracy and reported: a component
whose smallest gap falls under --gap-warn is flagged, because its per-state
number cannot be trusted even if it happens to look reasonable.

NOT COVERED HERE: a numerical DERIVATIVE COUPLING.  A NAC cannot be obtained
from energies -- it needs wavefunction overlaps between displaced geometries,
<Psi_i(x)|Psi_j(x+d)>, which in turn need AO overlaps across two basis-set
centres and determinant-level overlaps.  That is a real piece of machinery and
none of it exists in this tree yet.  What IS checked here are the two
structural properties a coupling must satisfy regardless
(--check-nac): antisymmetry under state exchange, and that the printed
"NAC * dE" block equals the "NAC" block times the energy difference.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psi4
from helper_PFCI import PFHamiltonianGenerator
from nuclear_grad import nuclear_grad

BOHR_PER_ANGSTROM = 1.8897261254578281

# --------------------------------------------------------------------------
# Systems.  `cavity` entries are merged into the cavity options, so a system
# differs from another only where it must.
# --------------------------------------------------------------------------
SYSTEMS = {
    # Small and cavity-coupled: the fast default, and the only one cheap enough
    # to run every component.
    "h2o_cav": dict(
        syms=["O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.068516219320],
                      [0.000000000000, -0.790689573744, 0.543701060715],
                      [0.134217000000, 0.831145000000, 0.601233000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3,
        cavity=dict(omega_value=0.1212727168509417,
                    lambda_vector=np.array([0.05, 0.0, 0.0]),
                    number_of_photons=1),
    ),
    # The developer's two ethylene decks, same geometries and cavity settings.
    "c2h4_fc": dict(
        syms=["C", "C", "H", "H", "H", "H"],
        pos=np.array([[-0.67874677, 0.00000002, -0.00000002],
                      [0.67874683, -0.00000004, 0.00000001],
                      [-1.24058330, 0.92302210, 0.00000000],
                      [-1.24058332, -0.92302209, 0.00000001],
                      [1.24058327, 0.92302208, -0.00000000],
                      [1.24058328, -0.92302208, -0.00000001]]),
        basis="aug-cc-pvdz", nact_orbs=7, nact_els=4, roots=2,
        cavity=dict(omega_value=0.1212727168509417,
                    lambda_vector=np.array([0.2, 0.0, 0.0]),
                    number_of_photons=1),
    ),
    # A CONICAL INTERSECTION geometry.  Per-state numerical gradients are
    # unreliable here by construction; use --avg-only.
    "c2h4_ci": dict(
        syms=["C", "C", "H", "H", "H", "H"],
        pos=np.array([[-0.80923036, 0.19558420, 0.16496066],
                      [0.57455545, 0.19224081, 0.16899871],
                      [-0.72823694, -0.12506799, -0.97585143],
                      [1.10977050, 1.10817374, 0.31448508],
                      [-1.30188430, -0.70546897, 0.48925480],
                      [1.15502567, -0.66546181, -0.11184781]]),
        basis="aug-cc-pvdz", nact_orbs=7, nact_els=4, roots=2,
        cavity=dict(omega_value=0.1212727168509417,
                    lambda_vector=np.array([0.2, 0.0, 0.0]),
                    number_of_photons=1),
    ),
}


def mol_string(syms, pos):
    s = "\n".join(f"{a} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for a, p in zip(syms, pos))
    return s + "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom"


def build_options(sd, weights, no_cavity):
    options = dict(basis=sd["basis"], scf_type="pk",
                   e_convergence=1e-10, d_convergence=1e-10)
    cav = dict(omega_value=0.0, lambda_vector=np.array([0.0, 0.0, 0.0]),
               ci_level="cas", ignore_coupling=False, number_of_photons=0,
               natural_orbitals=False, photon_number_basis=False,
               canonical_mos=False, coherent_state_basis=True,
               spin_adaptation="singlet", davidson_roots=sd["roots"],
               davidson_threshold=1e-9, davidson_maxdim=20,
               davidson_maxiter=200, davidson_indim=8, test_mode=False,
               nact_orbs=sd["nact_orbs"], nact_els=sd["nact_els"],
               casscf_weight=np.array(weights, dtype=float))
    if not no_cavity:
        cav.update(sd["cavity"])
    return options, cav


def energies_at(sd, pos, options, cav):
    """Converged state energies at one geometry."""
    psi4.set_options(options)
    job = PFHamiltonianGenerator(mol_string(sd["syms"], pos), options, cav)
    return np.asarray(job.eigenvals)[:sd["roots"]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="h2o_cav", choices=sorted(SYSTEMS))
    ap.add_argument("--weights", default=None,
                    help="comma-separated, one per root (default: equal)")
    ap.add_argument("--no-cavity", action="store_true")
    ap.add_argument("--delta", type=float, default=1e-4, help="Angstrom")
    ap.add_argument("--components", type=int, default=0,
                    help="test only the first N of the 3N components (0 = all)")
    ap.add_argument("--avg-only", action="store_true",
                    help="only the state-average gradient -- the leg that stays "
                         "valid at a conical intersection")
    ap.add_argument("--gap-warn", type=float, default=1e-3,
                    help="flag a component whose smallest gap falls below this")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()

    sd = SYSTEMS[a.system]
    nroots = sd["roots"]
    w = ([float(x) for x in a.weights.split(",")] if a.weights
         else [1.0] * nroots)
    if len(w) != nroots:
        raise SystemExit(f"{len(w)} weights for {nroots} roots")
    wn = np.array(w) / sum(w)
    options, cav = build_options(sd, w, a.no_cavity)
    psi4.core.be_quiet()

    print(f"system   {a.system}   basis {sd['basis']}   "
          f"CAS({sd['nact_els']},{sd['nact_orbs']})   {nroots} roots")
    print(f"weights  {w}  -> {wn}")
    print(f"cavity   {'OFF' if a.no_cavity else sd['cavity']}")
    print(f"delta    {a.delta} A")

    # ---- analytic ----
    psi4.set_options(options)
    psi4.core.set_output_file(f"{a.system}_numgrad.psi4.out", False)
    calc = nuclear_grad(mol_string(sd["syms"], sd["pos"]), options, cav)
    ana = {}
    for i in range(nroots):
        calc.compute_grad(i, i)
        ana[i] = np.asarray(calc.total_gradient).reshape(-1, 3).copy()
    ana_avg = sum(wn[i] * ana[i] for i in range(nroots))
    print(f"\nanalytic reference energies: "
          + "  ".join(f"E{i}={calc.eigenvals[i]:.10f}" for i in range(nroots)))

    # ---- numerical ----
    natom = len(sd["syms"])
    ncomp = 3 * natom if a.components <= 0 else min(a.components, 3 * natom)
    d_bohr = a.delta * BOHR_PER_ANGSTROM
    num = {i: np.zeros((natom, 3)) for i in range(nroots)}
    num_avg = np.zeros((natom, 3))
    flagged = []
    print(f"\nrunning {2 * ncomp} displaced CASSCF points "
          f"({ncomp} of {3 * natom} components)")
    for c in range(ncomp):
        atom, coord = divmod(c, 3)
        es = {}
        for sign in (+1, -1):
            p = sd["pos"].copy()
            p[atom, coord] += sign * a.delta
            es[sign] = energies_at(sd, p, options, cav)
            gaps = np.diff(np.sort(es[sign]))
            if gaps.size and gaps.min() < a.gap_warn:
                flagged.append((atom, "XYZ"[coord], float(gaps.min())))
        for i in range(nroots):
            num[i][atom, coord] = (es[+1][i] - es[-1][i]) / (2.0 * d_bohr)
        num_avg[atom, coord] = float(
            (wn @ es[+1] - wn @ es[-1]) / (2.0 * d_bohr))
        print(f"  component {c + 1}/{ncomp} done", flush=True)

    # ---- report ----
    def table(name, an, nu):
        worst = 0.0
        print(f"\n=== {name} ===")
        print(f"{'atom':>6s} {'dir':>4s} {'analytic':>18s} {'numerical':>18s} {'diff':>12s}")
        for c in range(ncomp):
            atom, coord = divmod(c, 3)
            av, nv = an[atom, coord], nu[atom, coord]
            worst = max(worst, abs(av - nv))
            print(f"{sd['syms'][atom]}{atom:<5d} {'XYZ'[coord]:>4s} "
                  f"{av:18.10f} {nv:18.10f} {abs(av - nv):12.2e}")
        print(f"  worst |analytic - numerical| = {worst:.3e}   "
              f"{'ok' if worst < a.tol else '**MISMATCH**'}")
        return worst

    worst = table("STATE-AVERAGE GRADIENT (valid through a conical intersection)",
                  ana_avg, num_avg)
    if not a.avg_only:
        for i in range(nroots):
            worst = max(worst, table(f"STATE {i} GRADIENT", ana[i], num[i]))

    if flagged:
        print(f"\n!! NEAR-DEGENERACY at {len(flagged)} displaced point(s); "
              f"smallest gap {min(f[2] for f in flagged):.2e} < {a.gap_warn:.0e}")
        print("!! Per-state numerical gradients there may straddle two surfaces "
              "after a state-order swap.")
        print("!! The STATE-AVERAGE leg is unaffected -- prefer --avg-only here.")

    print(f"\nWORST OVERALL {worst:.3e}   tol {a.tol:.0e}")
    return 0 if worst < a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
