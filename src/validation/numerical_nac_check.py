#!/usr/bin/env python3
"""NUMERICAL derivative couplings by finite-difference wavefunction overlap.

A NAC cannot be obtained from energies, so unlike a numerical gradient this
needs the wavefunctions themselves:

    d_ij = <Psi_i | d Psi_j / dx>
         ~ [ <Psi_i(x)|Psi_j(x+h)> - <Psi_i(x)|Psi_j(x-h)> ] / (2h)

Everything below exists to evaluate that overlap between two DIFFERENT
geometries, which needs three pieces:

  1. AO overlap across two basis-set centres, S_AO(x, x') -- psi4 gives this
     directly as MintsHelper(b1).ao_overlap(b1, b2).
  2. MO overlap, S_MO = C(x)^T S_AO C(x'), from the CASSCF orbitals at each
     geometry (opt_C).
  3. Determinant overlaps.  <D_I(x)|D_J(x')> factorises into an alpha and a
     beta determinant of the occupied blocks of S_MO, and the alpha and beta
     strings are independent, so the CAS overlap is one einsum over the
     string-overlap matrices rather than N_det^2 determinant evaluations.

BECAUSE THE AO BASIS MOVES WITH THE NUCLEI, this returns the FULL coupling --
the one INCLUDING the CSF contribution, i.e. Molpro's "SA-MC NACME" and pyscf
nac_method's kernel(state=(i,j)).  That is our
compute_grad(i, j, deriv_type="full") "NAC (with CSF)" block.

TWO THINGS MUST BE FIXED OR THE FINITE DIFFERENCE IS MEANINGLESS, and both are
done here rather than assumed:

  * PHASE.  A CI vector's overall sign is arbitrary and is chosen independently
    at every geometry, so each displaced state is phase-aligned to the
    reference by requiring <Psi_k(x)|Psi_k(x')> > 0.
  * STATE ORDERING.  States are matched to the reference by MAXIMUM OVERLAP,
    not by energy order.  This is the right thing anyway and it is what makes
    the method survive near a conical intersection, where the energy ordering
    swaps under displacement -- the failure mode that makes numerical
    PER-STATE GRADIENTS unreliable there (see numerical_gradient_check.py).
    A swap that was corrected is reported, not hidden.

CAVITY-OFF ONLY, and that is a real limit rather than an oversight.  In the
coherent-state basis the photon states are built on <d_HF> frozen at the
reference geometry, so at a displaced geometry the photon basis is a DIFFERENT
basis and <p(x)|p'(x')> is not delta_pp'.  Handling that needs the photon
overlap as well, which is not attempted here.  With omega = 0 and lambda = 0
there is one photon block and the question does not arise.

    python numerical_nac_check.py --system h2o_c1 --weights 5,2,1 --components 3
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psi4
from helper_PFCI import PFHamiltonianGenerator
from nuclear_grad import nuclear_grad

BOHR_PER_ANGSTROM = 1.8897261254578281

SYSTEMS = {
    "h2o_c1": dict(
        syms=["O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.068516219320],
                      [0.000000000000, -0.790689573744, 0.543701060715],
                      [0.134217000000, 0.831145000000, 0.601233000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3),
    "h2o_c1b": dict(
        syms=["O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.068516219320],
                      [0.000000000000, -0.850000000000, 0.600000000000],
                      [0.134217000000, 0.831145000000, 0.601233000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3),
    "ch2o_c1": dict(
        syms=["C", "O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.522500000000],
                      [0.000000000000, 0.000000000000, 0.677500000000],
                      [0.150000000000, 0.937000000000, -1.062500000000],
                      [0.000000000000, -0.937000000000, -1.112500000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3),
}


def mol_string(syms, pos):
    s = "\n".join(f"{a} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
                  for a, p in zip(syms, pos))
    return s + "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom"


def build_options(sd, weights):
    options = dict(basis=sd["basis"], scf_type="pk",
                   e_convergence=1e-10, d_convergence=1e-10)
    cav = dict(omega_value=0.0, lambda_vector=np.array([0.0, 0.0, 0.0]),
               ci_level="cas", ignore_coupling=False, number_of_photons=0,
               natural_orbitals=False, photon_number_basis=False,
               canonical_mos=False, coherent_state_basis=True,
               spin_adaptation="singlet", davidson_roots=sd["roots"],
               davidson_threshold=1e-10, davidson_maxdim=20,
               davidson_maxiter=200, davidson_indim=8, test_mode=False,
               nact_orbs=sd["nact_orbs"], nact_els=sd["nact_els"],
               casscf_weight=np.array(weights, dtype=float))
    return options, cav


def strings(sd, ndocc):
    """Alpha (== beta) occupation lists, in generateCASCIDeterminants' order.

    That routine loops `for alpha in combinations(...): for beta in ...`, so
    the determinant index is i_alpha * n_beta + i_beta and the CI vector
    reshapes to (n_alpha, n_beta).
    """
    half = sd["nact_els"] // 2
    n_in = ndocc - half
    core = list(range(n_in))
    return [core + [o + n_in for o in c]
            for c in combinations(range(sd["nact_orbs"]), half)]


def run_point(sd, pos, options, cav):
    psi4.set_options(options)
    job = PFHamiltonianGenerator(mol_string(sd["syms"], pos), options, cav)
    mol = psi4.geometry(mol_string(sd["syms"], pos))
    bs = psi4.core.BasisSet.build(mol, "ORBITAL", options["basis"])
    return dict(C=np.asarray(job.opt_C).copy(),
                vecs=np.asarray(job.eigenvecs).copy(),
                ndocc=job.ndocc, bs=bs)


def wf_overlap(sd, ref, dis):
    """<Psi_i(ref) | Psi_j(dis)> for every (i, j), as a (nroots, nroots) array."""
    S_ao = np.asarray(psi4.core.MintsHelper(ref["bs"]).ao_overlap(ref["bs"], dis["bs"]))
    S_mo = ref["C"].T @ S_ao @ dis["C"]
    occ = strings(sd, ref["ndocc"])
    n = len(occ)
    # string-overlap matrix: det of the occupied block, alpha and beta alike
    D = np.empty((n, n))
    for a, oa in enumerate(occ):
        for b, ob in enumerate(occ):
            D[a, b] = np.linalg.det(S_mo[np.ix_(oa, ob)])
    nr = sd["roots"]
    cr = ref["vecs"][:nr].reshape(nr, n, n)
    cd = dis["vecs"][:nr].reshape(nr, n, n)
    # <Psi_i|Psi_j> = sum c_i[a,b] Dalpha[a,A] Dbeta[b,B] c_j[A,B]
    return np.einsum("iab,aA,bB,jAB->ij", cr, D, D, cd, optimize=True)


def align(S):
    """Match displaced states to reference by MAX OVERLAP and fix their phase.

    Returns (permutation, signs, swapped) where column `perm[k]` of the
    displaced set is the partner of reference state k.
    """
    nr = S.shape[0]
    perm, used = np.zeros(nr, dtype=int), set()
    for i in range(nr):
        order = np.argsort(-np.abs(S[i]))
        perm[i] = next(j for j in order if j not in used)
        used.add(perm[i])
    signs = np.sign(S[np.arange(nr), perm])
    signs[signs == 0] = 1.0
    return perm, signs, bool(np.any(perm != np.arange(nr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="h2o_c1", choices=sorted(SYSTEMS))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--delta", type=float, default=1e-3, help="Angstrom")
    ap.add_argument("--components", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()

    sd = SYSTEMS[a.system]
    nr = sd["roots"]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * nr
    options, cav = build_options(sd, w)
    psi4.core.be_quiet()
    psi4.core.set_output_file(f"{a.system}_numnac.psi4.out", False)

    print(f"system {a.system}  basis {sd['basis']}  "
          f"CAS({sd['nact_els']},{sd['nact_orbs']})  {nr} roots  weights {w}")
    print(f"delta {a.delta} A   (cavity OFF -- see the module docstring)")

    # analytic couplings: the "NAC (with CSF)" block
    psi4.set_options(options)
    calc = nuclear_grad(mol_string(sd["syms"], sd["pos"]), options, cav)
    pairs = [(i, j) for i in range(nr) for j in range(i + 1, nr)]
    ana = {}
    for (i, j) in pairs:
        calc.compute_grad(i, j, deriv_type="full")
        ana[(i, j)] = np.asarray(calc.h_vector).reshape(-1, 3).copy()

    ref = run_point(sd, sd["pos"], options, cav)
    natom = len(sd["syms"])
    ncomp = 3 * natom if a.components <= 0 else min(a.components, 3 * natom)
    d_bohr = a.delta * BOHR_PER_ANGSTROM
    num = {p: np.zeros((natom, 3)) for p in pairs}
    swaps = []

    for c in range(ncomp):
        atom, coord = divmod(c, 3)
        Sd = {}
        for sign in (+1, -1):
            p = sd["pos"].copy()
            p[atom, coord] += sign * a.delta
            dis = run_point(sd, p, options, cav)
            S = wf_overlap(sd, ref, dis)
            perm, sgn, swapped = align(S)
            if swapped:
                swaps.append((atom, "XYZ"[coord], sign, perm.tolist()))
            Sd[sign] = S[:, perm] * sgn[None, :]
        for (i, j) in pairs:
            num[(i, j)][atom, coord] = (Sd[+1][i, j] - Sd[-1][i, j]) / (2.0 * d_bohr)
        print(f"  component {c + 1}/{ncomp} done", flush=True)

    worst = 0.0
    for (i, j) in pairs:
        an, nu = ana[(i, j)], num[(i, j)]
        # a coupling carries the arbitrary relative CI phase -> compare up to sign
        d = min(np.abs(s * an[:ncomp // 3 + 1] - nu[:ncomp // 3 + 1]).max()
                for s in (1, -1))
        best = min(((np.abs(s * an - nu), s) for s in (1, -1)),
                   key=lambda t: t[0].flat[:ncomp].max())
        dd = best[0].flat[:ncomp].max()
        worst = max(worst, dd)
        print(f"\n=== NAC {i}-{j}   (phase {best[1]:+d}) ===")
        print(f"{'atom':>6s} {'dir':>4s} {'analytic':>18s} {'numerical':>18s} {'diff':>12s}")
        for cc in range(ncomp):
            at, co = divmod(cc, 3)
            print(f"{sd['syms'][at]}{at:<5d} {'XYZ'[co]:>4s} "
                  f"{best[1]*an[at,co]:18.10f} {nu[at,co]:18.10f} "
                  f"{abs(best[1]*an[at,co]-nu[at,co]):12.2e}")
        print(f"  worst {dd:.3e}  {'ok' if dd < a.tol else '**MISMATCH**'}")

    if swaps:
        print(f"\n!! {len(swaps)} displaced point(s) needed a STATE REORDERING; "
              "matched by overlap, not energy:")
        for s in swaps[:6]:
            print(f"     atom {s[0]} {s[1]} sign {s[2]:+d} -> perm {s[3]}")
    print(f"\nWORST OVERALL {worst:.3e}   tol {a.tol:.0e}")
    return 0 if worst < a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
