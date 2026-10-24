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

CAVITY ON IS SUPPORTED, and the photon question turned out to have a MEASURED
answer rather than an assumed one.

The coherent-state origin genuinely moves with geometry: for h2o_cav at a
1e-3 A displacement, d_exp goes -0.007361 -> -0.008198, so the displacement
difference is gamma = -1.70e-03 and gamma/(2h) = -0.45.  A photon-basis
derivative term is therefore NOT small -- it is O(0.5) on a coupling of
order 1.

Measured against the analytic coupling (h2o_cav, weights 5,2,1, h = 1e-3 A):

    photon states treated as geometry-independent   1.48e-05   <-- agrees
    + displaced-basis overlap, sign +1              8.77e-01
    + displaced-basis overlap, sign -1              8.99e-01

and the agreeing leg converges as h^2 (2.356e-04 / 5.896e-05 / 1.476e-05 at
h = 4e-3 / 2e-3 / 1e-3 A; ratios 4.00 and 3.99), which is the same quality as
the cavity-off case.  So the analytic implementation and this overlap use the
SAME convention, and adding the displaced-basis factor on top breaks it.

WHY, most likely: a FRAME distinction, not double counting.  The code works
entirely in the coherent-state-transformed frame -- Hamiltonian U+HU,
wavefunction U+Psi -- so overlaps of those objects against undisplaced Fock
states give the WORKING-FRAME coupling, and the analytic coupling, from the
same transformed Hamiltonian, is the working-frame one too.  Including the
displaced-basis factor moves the estimate toward the LAB-FRAME coupling, which
is a different quantity.  Neither is wrong; they are not interchangeable.

A double-counting reading against the QED-HF response (sum_ai kappa_ai f_ai)
was considered and does NOT fit the magnitudes: dropping that term entirely
moves the comparison by 1.6e-02, while adding the photon factor moves it by
~8.8e-01, ~55x larger.  A term duplicating kappa.f would be of its order.

`--photon-basis-derivative` switches the displaced-basis factor on for anyone
who wants to re-examine that; it is OFF by default because the measurement
above says so.

    python numerical_nac_check.py --system h2o_c1 --weights 5,2,1 --components 3

FULL METHOD NOTE, including the error budget, the h^2 validation and what a
cavity-on extension would require: docs/theory/numerical_nac_method.md
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
from scipy.special import eval_genlaguerre, factorial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import psi4
from helper_PFCI import PFHamiltonianGenerator
from nuclear_grad import nuclear_grad

BOHR_PER_ANGSTROM = 1.8897261254578281
_NO_PHOTON = [False]

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
    # CAVITY ON: the case the photon-overlap factor exists for.
    "h2o_cav": dict(
        syms=["O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.068516219320],
                      [0.000000000000, -0.790689573744, 0.543701060715],
                      [0.134217000000, 0.831145000000, 0.601233000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3,
        cavity=dict(omega_value=0.1212727168509417,
                    lambda_vector=np.array([0.05, 0.0, 0.0]),
                    number_of_photons=1)),
    # THE HARD CASE: ethylene distorted to C1, CAS(4,7)/cc-pVDZ, with omega on
    # its own bare S0->S1 gap (0.331337 Ha) and lambda = 0.2 along the C=C axis
    # -- the developer's own c2h4_fc.py value and direction.  That transition is
    # pi->pi* with a large transition dipole along x, so unlike formaldehyde
    # (whose low transition dipoles are all <= 0.021, giving no mixing whatever
    # omega is) this one really does hybridise.  6 atoms, 48 basis functions,
    # 441 determinants x 2 photon states.
    "c2h4_res": dict(
        syms=["C", "C", "H", "H", "H", "H"],
        pos=np.array([[-0.67874677, 0.00000002, -0.00000002],
                      [0.67874683, -0.00000004, 0.00000001],
                      [-1.19058330, 0.92302210, 0.13000000],
                      [-1.24058332, -0.92302209, 0.00000001],
                      [1.24058327, 0.92302208, -0.00000000],
                      [1.24058328, -0.92302208, -0.00000001]]),
        basis="cc-pvdz", nact_orbs=7, nact_els=4, roots=3,
        cavity=dict(omega_value=0.331336628349,
                    lambda_vector=np.array([0.2, 0.0, 0.0]),
                    number_of_photons=1)),
    # RESONANT + STRONGLY COUPLED, and bigger than water: formaldehyde with
    # omega set to its own bare S0->S1 gap at this geometry (0.151415 Ha) and
    # lambda = 0.1, four times the h2o_cav value.  On resonance |1,0> and
    # |0,1> are degenerate and mix into two polaritons, so the states carry
    # maximal photon character -- which is where the frame term is largest and
    # where overlap-based state matching is most likely to be needed.
    "ch2o_res": dict(
        syms=["C", "O", "H", "H"],
        pos=np.array([[0.000000000000, 0.000000000000, -0.522500000000],
                      [0.000000000000, 0.000000000000, 0.677500000000],
                      [0.150000000000, 0.937000000000, -1.062500000000],
                      [0.000000000000, -0.937000000000, -1.112500000000]]),
        basis="6-31g", nact_orbs=5, nact_els=6, roots=3,
        cavity=dict(omega_value=0.151415261575,
                    lambda_vector=np.array([0.1, 0.0, 0.0]),
                    number_of_photons=1)),
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
    cav.update(sd.get("cavity", {}))
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


def displacement_matrix(gamma, nmax):
    """<m|D(gamma)|n> for REAL gamma, m,n = 0..nmax.

    Cahill-Glauber; verified exactly (1.1e-16) against expm(gamma*(a^dag - a)).
    The leading off-diagonal is <0|D|1> = -gamma + O(gamma^3), i.e. it enters at
    FIRST order in the displacement -- which is why omitting this factor
    corrupts a finite-difference coupling at leading order rather than mildly.
    """
    P = np.zeros((nmax + 1, nmax + 1))
    g2 = gamma * gamma
    for m in range(nmax + 1):
        for n in range(nmax + 1):
            if m >= n:
                P[m, n] = (np.sqrt(factorial(n) / factorial(m)) * gamma ** (m - n)
                           * np.exp(-g2 / 2) * eval_genlaguerre(n, m - n, g2))
            else:
                P[m, n] = (np.sqrt(factorial(m) / factorial(n)) * (-gamma) ** (n - m)
                           * np.exp(-g2 / 2) * eval_genlaguerre(m, n - m, g2))
    return P


def photon_overlap(ref, dis, sign=+1.0):
    """<n; alpha(x) | n'; alpha(x')> between two coherent-state photon bases.

    In the coherent-state basis the photon operators are displaced so the
    bilinear term reads -sqrt(w/2)(d - <d>)(b + b^dag).  Completing the square
    in  w b^dag b - sqrt(w/2) <d> (b + b^dag)  puts the shift at

        alpha = <d_HF> / sqrt(2 w)

    and <d_HF> is frozen at each calculation's OWN reference geometry, so two
    geometries carry different alpha and the photon bases do not coincide.
    Only the DIFFERENCE enters:  gamma = alpha(x') - alpha(x).

    `sign` exists because the overall sign of the code's coherent-state
    transformation is a convention that has to be taken from the source rather
    than assumed; it is determined empirically and checked (see the method
    note).  Note the diagonal of D is EVEN in gamma, so a sign error is
    invisible there and only shows up off-diagonal -- do not "verify" this by
    looking at <n|D|n>.
    """
    if ref["np1"] == 1:
        return np.eye(1)
    # NUCLEAR DIPOLE IN THE DISPLACEMENT.  U_CS is built from the FULL dipole,
    # z = -(<d_e> + d_N)/sqrt(2w).  d_N cancels inside H_CS because the coupling
    # and DSE contain only the DIFFERENCE d - <d> = d_e - <d_e>, which is why the
    # code stores the electronic d_exp -- but the TRANSFORMATION still carries
    # d_N, and the finite-difference displacement must too.  d_N is geometry
    # dependent (lam . mu_nuc changes when a nucleus moves), so omitting it
    # leaves out exactly  lam_a Z_A / sqrt(2w)  per displaced coordinate, which
    # is the nuclear half of d_photon.  QED_FD_DNUC=0 restores the old form.
    _fd_dnuc = os.environ.get("QED_FD_DNUC", "1") not in ("0", "", "false")
    _dr = ref.get("d_nuc", 0.0) if _fd_dnuc else 0.0
    _dd = dis.get("d_nuc", 0.0) if _fd_dnuc else 0.0
    alpha_r = (ref["d_exp"] + _dr) / np.sqrt(2.0 * ref["omega"])
    alpha_d = (dis["d_exp"] + _dd) / np.sqrt(2.0 * dis["omega"])
    _gamma = sign * (alpha_d - alpha_r)
    if os.environ.get("QED_FD_PRINT_GAMMA"):
        print("GAMMA %+.10e   (alpha_r %+.6e  alpha_d %+.6e  d_nuc_r %+.6e "
              "d_nuc_d %+.6e)" % (_gamma, alpha_r, alpha_d, _dr, _dd), flush=True)
    return displacement_matrix(_gamma, ref["np1"] - 1)


def run_point(sd, pos, options, cav):
    psi4.set_options(options)
    job = PFHamiltonianGenerator(mol_string(sd["syms"], pos), options, cav)
    mol = psi4.geometry(mol_string(sd["syms"], pos))
    bs = psi4.core.BasisSet.build(mol, "ORBITAL", options["basis"])
    return dict(C=np.asarray(job.opt_C).copy(),
                vecs=np.asarray(job.eigenvecs).copy(),
                ndocc=job.ndocc, bs=bs, np1=job.N_p + 1,
                d_exp=float(getattr(job, "d_exp", 0.0)),
                d_nuc=float(np.dot(np.asarray(cav["lambda_vector"]),
                                   np.asarray(getattr(job, "nuclear_dipole_moment",
                                                      np.zeros(3))))),
                omega=float(getattr(job, "omega", 0.0)) or 1.0)


def wf_overlap(sd, ref, dis, photon_sign=+1.0):
    """<Psi_i(ref) | Psi_j(dis)> for every (i, j), as a (nroots, nroots) array."""
    if _NO_PHOTON[0]:
        ref = dict(ref); dis = dict(dis)
    S_ao = np.asarray(psi4.core.MintsHelper(ref["bs"]).ao_overlap(ref["bs"], dis["bs"]))
    S_mo = ref["C"].T @ S_ao @ dis["C"]
    occ = strings(sd, ref["ndocc"])
    n = len(occ)
    # string-overlap matrix: det of the occupied block, alpha and beta alike
    D = np.empty((n, n))
    for a, oa in enumerate(occ):
        for b, ob in enumerate(occ):
            D[a, b] = np.linalg.det(S_mo[np.ix_(oa, ob)])
    nr, np1 = sd["roots"], ref["np1"]
    # PHOTON-MAJOR: helper_PFCI reshapes a CI vector as (np1, num_det); see
    # helper_PFCI.py:5098.  num_det then splits (n_alpha, n_beta).
    cr = ref["vecs"][:nr].reshape(nr, np1, n, n)
    cd = dis["vecs"][:nr].reshape(nr, np1, n, n)
    P = (np.eye(np1) if _NO_PHOTON[0]
         else photon_overlap(ref, dis, sign=photon_sign))
    # <Psi_i|Psi_j> = sum c_i[p,a,b] P[p,q] Dalpha[a,A] Dbeta[b,B] c_j[q,A,B]
    return np.einsum("ipab,pq,aA,bB,jqAB->ij", cr, P, D, D, cd, optimize=True)


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


def sign_check(a):
    """Pin the SIGN of the NAC, which every other comparison here cannot see.

    A derivative coupling carries the arbitrary phase of the CI vectors, so any
    comparison between two independent calculations must allow an overall sign
    -- and every other check in this file does.  That makes them blind to a
    systematic sign error, which is exactly how an inverted energy denominator
    survived: d^{mn}_CI = h^{mn}/(E^n - E^m), and the code had
    E[state1] - E[state2].

    The fix is to remove the freedom: use the SAME nuclear_grad object both for
    the analytic coupling and as the bra of the finite-difference overlap, so
    both routes refer to one set of CI vectors and the signs are comparable.
    """
    sd = SYSTEMS[a.system]
    nr = sd["roots"]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * nr
    options, cav = build_options(sd, w)
    if a.photons > 0:
        cav["number_of_photons"] = a.photons
    psi4.core.be_quiet()
    psi4.core.set_output_file(f"{a.system}_sign.psi4.out", False)
    psi4.set_options(options)
    geom = mol_string(sd["syms"], sd["pos"])
    calc = nuclear_grad(geom, options, cav)

    ref = dict(C=np.asarray(calc.opt_C).copy(),
               vecs=np.asarray(calc.eigenvecs).copy(),
               ndocc=calc.ndocc, np1=calc.N_p + 1,
               d_exp=float(getattr(calc, "d_exp", 0.0)),
               # d_nuc MUST be present here too.  There are two dict builders --
               # this one for the reference point and run_point() for the
               # displaced ones -- and adding d_nuc to only one leaves
               # ref.get("d_nuc", 0.0) = 0 while dis carries the full value.
               # That is a CONSTANT offset gamma_0 = d_N/sqrt(2w) in the
               # displacement, so gamma_+ and gamma_- stop being equal and
               # opposite and D(gamma_0 +- delta) = D(gamma_0) D(+-delta) leaves
               # a spurious O(gamma_0 * d_CS) term that is NOT proportional to
               # G^{mn} -- which is why it hit the small-G pair hardest.
               d_nuc=float(np.dot(np.asarray(cav["lambda_vector"]),
                                  np.asarray(getattr(calc, "nuclear_dipole_moment",
                                                     np.zeros(3))))),
               omega=float(getattr(calc, "omega", 0.0)) or 1.0,
               bs=psi4.core.BasisSet.build(psi4.geometry(geom), "ORBITAL",
                                           options["basis"]))
    def fd_at(step):
        """Central-difference coupling matrix at displacement `step` (Angstrom).

        Also runs the BRANCH CHECK described below and reports any failure.
        """
        hb = step * BOHR_PER_ANGSTROM
        S, dev = {}, {}
        for sgn in (+1, -1):
            p = sd["pos"].copy()
            p[0, 0] += sgn * step
            Sm = wf_overlap(sd, ref, run_point(sd, p, options, cav),
                            photon_sign=a.photon_sign)
            perm, sg, _ = align(Sm)
            S[sgn] = Sm[:, perm] * sg[None, :]
            dev[sgn] = [1.0 - abs(Sm[i, perm[i]]) for i in range(nr)]
        # BRANCH CHECK.  For a wavefunction varying smoothly with x,
        # 1 - |<Psi_i(x)|Psi_i(x+-h)>| is O(h^2) and therefore EQUAL at +h and
        # -h.  A large asymmetry means one displaced CASSCF converged to a
        # DIFFERENT SOLUTION, and the finite difference then straddles two
        # branches -- Richardson cannot repair that, because it removes
        # truncation, not a wrong solution.
        #
        # This is not hypothetical: on c2h4_res (resonant, lambda = 0.2, state 1
        # 73% photon) the -h branch diverges for h >= 2e-3, and the comparison
        # reads 7.1e-02 instead of the 5.9e-06 obtained at h = 1e-3, where both
        # sides are clean.  Without this check that looks like a defect in the
        # analytic coupling.
        for i in range(nr):
            hi, lo = max(dev[+1][i], dev[-1][i]), min(dev[+1][i], dev[-1][i])
            if lo > 0 and hi / lo > 1.5:
                print(f"  !! BRANCH WARNING state {i}, h={step:.1e}: "
                      f"1-|S_ii| is {dev[+1][i]:.2e} at +h but {dev[-1][i]:.2e} "
                      f"at -h (x{hi/lo:.1f}).  Expected equal for a smooth "
                      f"wavefunction -- one displaced CASSCF is on a different "
                      f"solution.  REDUCE h.", flush=True)
        return (S[+1] - S[-1]) / (2.0 * hb)

    fd_h = fd_at(a.delta)
    if a.richardson:
        # Central differences carry an O(h^2) error, so (4 f(h/2) - f(h))/3
        # cancels it and leaves O(h^4).  The h^2 behaviour was verified
        # separately (ratios 4.00 per halving), which is what licenses this.
        fd_h2 = fd_at(a.delta / 2.0)
        fd = (4.0 * fd_h2 - fd_h) / 3.0
    else:
        fd = fd_h

    hdr = "  Richardson" if a.richardson else " finite diff"
    print(f"{'pair':>6s} {'analytic':>16s} {hdr:>16s} {'signed diff':>13s}"
          + ("   plain h" if a.richardson else ""))
    worst = 0.0
    for i in range(nr):
        for j in range(i + 1, nr):
            calc.compute_grad(i, j, deriv_type="full")
            an = np.asarray(calc.h_vector).reshape(-1, 3)[0, 0]
            worst = max(worst, abs(an - fd[i, j]))
            extra = f"   {abs(an - fd_h[i,j]):.2e}" if a.richardson else ""
            print(f"{i}-{j:<4d} {an:16.10f} {fd[i,j]:16.10f} "
                  f"{abs(an-fd[i,j]):13.2e}{extra}")
    print(f"\nWORST OVERALL {worst:.3e}   tol {a.tol:.0e}   (NO sign freedom)")
    return 0 if worst < a.tol else 1


def frame_decomposition(a):
    """Check d_lab - d_CS against theory, with no reference to the analytic code.

    The coherent-state transformation U = exp[z (b+ - b)] with
    z = <d_HF>/sqrt(2 w) depends on geometry, so the working-frame ("CS") and
    lab-frame states differ and so do their derivative couplings:

        d_lab = d_CS + (dz/dx) <Psi_m|(b+ - b)|Psi_n>

    The right-hand side is computable from the CI vectors and from d(<d_HF>)/dx
    alone, so this validates the displaced-basis photon overlap against theory
    rather than against our own analytic gradient -- the two FD legs differ by
    exactly this, or the photon implementation is wrong.
    """
    sd = SYSTEMS[a.system]
    nr = sd["roots"]
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * nr
    options, cav = build_options(sd, w)
    if a.photons > 0:
        cav["number_of_photons"] = a.photons
    psi4.core.be_quiet()
    psi4.core.set_output_file(f"{a.system}_frame.psi4.out", False)
    hb = a.delta * BOHR_PER_ANGSTROM
    pts = {}
    for lab, dx in (("0", 0.0), ("+", a.delta), ("-", -a.delta)):
        p = sd["pos"].copy()
        p[0, 0] += dx
        pts[lab] = run_point(sd, p, options, cav)
    ref, np1 = pts["0"], pts["0"]["np1"]
    n = len(strings(sd, ref["ndocc"]))
    # dz/dx must use the FULL dipole, <d> = <d_e> + d_N, for the same reason the
    # displacement does: U_CS is built from z = -<d>/sqrt(2w).  This is the THIRD
    # site that needs d_nuc (the other two are run_point's dict and the reference
    # dict); with the electronic part alone the predicted RHS came out -0.899
    # against a measured d_lab - d_CS of -0.087 on pair 0-1.
    _fdn = os.environ.get("QED_FD_DNUC", "1") not in ("0", "", "false")
    _dnp = pts["+"].get("d_nuc", 0.0) if _fdn else 0.0
    _dnm = pts["-"].get("d_nuc", 0.0) if _fdn else 0.0
    dz = (((pts["+"]["d_exp"] + _dnp) - (pts["-"]["d_exp"] + _dnm)) / (2.0 * hb)
          / np.sqrt(2.0 * ref["omega"]))
    B = np.zeros((np1, np1))
    for q in range(np1):
        if q + 1 < np1:
            B[q + 1, q] += np.sqrt(q + 1)          # <p|b^dag|q>
        if q - 1 >= 0:
            B[q - 1, q] -= np.sqrt(q)              # -<p|b|q>
    c = ref["vecs"][:nr].reshape(nr, np1, n, n)
    Bmn = np.einsum("mpab,pq,nqab->mn", c, B, c)
    pred = dz * Bmn

    def leg(with_photon):
        _NO_PHOTON[0] = not with_photon
        S = {}
        for sgn in ("+", "-"):
            Sm = wf_overlap(sd, ref, pts[sgn], photon_sign=a.photon_sign)
            perm, sg, _ = align(Sm)
            S[sgn] = Sm[:, perm] * sg[None, :]
        return (S["+"] - S["-"]) / (2.0 * hb)

    d_cs, d_lab = leg(False), leg(True)
    print(f"dz/dx = {dz:+.8f}   (z = <d_HF>/sqrt(2w), w = {ref['omega']:.6f})")
    print(f"{'pair':>6s} {'d_CS':>13s} {'d_lab':>13s} {'d_lab-d_CS':>13s} "
          f"{'predicted':>13s} {'diff':>11s}")
    worst = 0.0
    for i in range(nr):
        for j in range(i + 1, nr):
            dd = d_lab[i, j] - d_cs[i, j]
            worst = max(worst, abs(dd - pred[i, j]))
            print(f"{i}-{j:<4d} {d_cs[i,j]:13.8f} {d_lab[i,j]:13.8f} "
                  f"{dd:13.8f} {pred[i,j]:13.8f} {abs(dd-pred[i,j]):11.2e}")
    print(f"\nWORST OVERALL {worst:.3e}   tol {a.tol:.0e}")
    return 0 if worst < a.tol else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="h2o_c1", choices=sorted(SYSTEMS))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--delta", type=float, default=1e-3, help="Angstrom")
    ap.add_argument("--components", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--photons", type=int, default=0,
                    help="override number_of_photons (N_p); 0 = the system default. "
                         "Use it to check that the frame term CONVERGES with N_p "
                         "rather than tracking the truncation.")
    ap.add_argument("--richardson", action="store_true",
                    help="with --sign-check: also evaluate at h/2 and Richardson-"
                         "extrapolate, (4 f(h/2) - f(h))/3, removing the O(h^2) "
                         "truncation that otherwise limits the comparison")
    ap.add_argument("--sign-check", action="store_true",
                    help="compare the analytic NAC to the finite difference "
                         "WITHOUT allowing a sign flip, by sharing the reference "
                         "CI vectors between the two routes")
    ap.add_argument("--frame-decomposition", action="store_true",
                    help="report d_CS, d_lab and the predicted difference "
                         "(dz/dx)<m|b+ - b|n>, z = <d_HF>/sqrt(2w), instead of "
                         "comparing to the analytic coupling")
    ap.add_argument("--photon-sign", type=float, default=1.0,
                    help="sign convention of the coherent-state displacement")
    ap.add_argument("--photon-basis-derivative", action="store_true",
                    help="ALSO include the displaced-coherent-state photon "
                         "overlap.  OFF by default -- see the module docstring; "
                         "switching it on injects a spurious O(0.5) term.")
    a = ap.parse_args()

    _NO_PHOTON[0] = not a.photon_basis_derivative
    if a.sign_check:
        return sign_check(a)
    if a.frame_decomposition:
        return frame_decomposition(a)
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
            S = wf_overlap(sd, ref, dis, photon_sign=a.photon_sign)
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
