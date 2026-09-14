"""
Analytic test of MECPOptimizer against the steepest descent it replaces.

A two-state linear-vibronic model with an EXACTLY KNOWN crossing point, built on
a real ethylene geometry so the translation/rotation projection is exercised.

The model is defined in MASS-WEIGHTED coordinates q = M^1/2 x, where there is a
single symmetric projector P onto the internal subspace. Cartesian gradients are
recovered as g_x = M^1/2 g_q on the way out, since a gradient transforms
oppositely to a displacement -- the distinction that a first version of the
optimizer got wrong, and that this test exists to catch.

    d      = P (q - q0)
    dE(d)  = sqrt((g.d)^2 + (h.d)^2)       seam is  g.d = h.d = 0
    Emean  = 0.5 (d - d0)^T K (d - d0)     K diagonal, deliberately anisotropic
    E_p    = Emean - dE/2,  E_q = Emean + dE/2

The exact MECP minimises Emean over the internal subspace subject to
g.d = h.d = 0. With anisotropic K that is the K-weighted projection of d0, not
the orthogonal one, so it is solved here as an equality-constrained QP in an
orthonormal internal basis rather than by a closed form that is easy to get
wrong.
"""

import sys, numpy as np
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src")
from mecp_optimizer import (MECPOptimizer, transrot_projector_mw,
                            ATOMIC_MASSES, ANG2BOHR, HARTREE2EV)

rng = np.random.default_rng(7)
SYMBOLS = ['H', 'H', 'H', 'H', 'C', 'C']
X0_ANG = np.array([
    [-0.85931728, -0.07489331, -0.97619746],
    [ 1.08642379,  1.06216978,  0.32443478],
    [-1.19732277, -0.62760333,  0.62856868],
    [ 1.11016023, -0.68940206, -0.17995881],
    [-0.73555057,  0.16544841,  0.12559011],
    [ 0.59560663,  0.16428049,  0.12756271]])
X0 = X0_ANG * ANG2BOHR
N = len(SYMBOLS); NDIM = 3 * N
MASSES = np.array([ATOMIC_MASSES[s] for s in SYMBOLS])
W = np.repeat(np.sqrt(MASSES), 3)               # M^1/2, flattened
P = transrot_projector_mw(X0, MASSES)
assert np.abs(P - P.T).max() < 1e-12, "projector must be symmetric"
assert np.abs(P @ P - P).max() < 1e-12, "projector must be idempotent"
Q0 = X0.ravel() * W                             # reference in mass-weighted coords


def _internal_unit(v):
    v = P @ v
    return v / np.linalg.norm(v)


GV = _internal_unit(rng.standard_normal(NDIM))
HV = rng.standard_normal(NDIM)
HV = _internal_unit(HV - (HV @ GV) * GV)
KDIAG = np.exp(rng.uniform(np.log(0.02), np.log(0.30), NDIM))
D0 = P @ (rng.standard_normal(NDIM) * 1.2)

# --- exact MECP by equality-constrained QP in an orthonormal internal basis ---
_w, _V = np.linalg.eigh(P)
U = _V[:, _w > 0.5]
Ku = U.T @ (KDIAG[:, None] * U)
bu = U.T @ (KDIAG * D0)
A = np.array([GV @ U, HV @ U])
KKT = np.block([[Ku, A.T], [A, np.zeros((2, 2))]])
sol = np.linalg.solve(KKT, np.concatenate([bu, np.zeros(2)]))
D_MECP = U @ sol[:U.shape[1]]
Q_MECP = Q0 + D_MECP
X_MECP = Q_MECP / W
assert abs(GV @ D_MECP) < 1e-10 and abs(HV @ D_MECP) < 1e-10, "MECP not on the seam"


def model(x_flat):
    """e_p, e_q, grad_p, grad_q, h at a CARTESIAN geometry (Bohr)."""
    d = P @ (x_flat * W - Q0)
    a, b = GV @ d, HV @ d
    dE = np.hypot(a, b)
    dd = d - D0
    emean = 0.5 * float(dd @ (KDIAG * dd))
    gmean_q = P @ (KDIAG * dd)                       # gradient wrt q
    gdiff_q = GV.copy() if dE < 1e-14 else (a * GV + b * HV) / dE
    e_p, e_q = emean - 0.5 * dE, emean + 0.5 * dE
    grad_p = (gmean_q - 0.5 * gdiff_q) * W           # back to Cartesian gradients
    grad_q = (gmean_q + 0.5 * gdiff_q) * W
    h = (dE * HV) * W
    return e_p, e_q, grad_p.reshape(-1, 3), grad_q.reshape(-1, 3), h.reshape(-1, 3)


GAP_TOL, SEAM_TOL = 3.7e-6, 5.0e-4


def _seam_and_gap(x_flat):
    e_p, e_q, gp, gq, h = model(x_flat)
    g = (gq - gp).ravel() / W                        # to mass-weighted gradients
    sigma = 0.5 * (gp + gq).ravel() / W
    hq = h.ravel() / W
    u_g = g / np.linalg.norm(g)
    hp = hq - (hq @ u_g) * u_g
    u_h = hp / np.linalg.norm(hp) if np.linalg.norm(hp) > 1e-12 else np.zeros_like(u_g)
    f_seam = P @ (sigma - (sigma @ u_g) * u_g - (sigma @ u_h) * u_h)
    return e_q - e_p, f_seam, u_g, u_h, e_p, e_q, gp, gq, h


def run_steepest_descent(x_start, alpha, max_steps=6000):
    """The algorithm currently in scan_conical_*.py: f_gap built from the gap in
    eV while f_seam is in Hartree/Bohr, a fixed multiple of the total force as
    the step, and a smaller step near the seam."""
    x = x_start.copy()
    for step in range(1, max_steps + 1):
        dE, f_seam, u_g, u_h, *_ = _seam_and_gap(x)
        if abs(dE) < GAP_TOL and np.linalg.norm(f_seam * W) < SEAM_TOL:
            return step, x, True
        gap_eV = dE * HARTREE2EV
        total = P @ (-(gap_eV * u_g + f_seam))
        eff = min(alpha, 0.05) if abs(gap_eV) < 0.01 else alpha
        x = x + (eff * total / W)
        if not np.all(np.isfinite(x)):
            return step, x, False
    return max_steps, x, False


def run_quasi_newton(x_start, max_steps=6000):
    opt = MECPOptimizer(SYMBOLS, x_start.reshape(-1, 3) / ANG2BOHR,
                        gap_tol=GAP_TOL, seam_tol=SEAM_TOL, verbose=False)
    for step in range(1, max_steps + 1):
        xf = opt.positions_ang.ravel() * ANG2BOHR
        e_p, e_q, gp, gq, h = model(xf)
        info = opt.step(e_p, e_q, gp, gq, h)
        if info["converged"]:
            return step, xf, True, info
    return max_steps, opt.positions_ang.ravel() * ANG2BOHR, False, info


print("=" * 74)
print("MECP optimizer test: anisotropic 2-state model, exact reference MECP")
print("=" * 74)
print(f"  atoms {N}   internal dof {NDIM - 6}   force-constant spread "
      f"{KDIAG.max()/KDIAG.min():.1f}x")
def _exact_plane_seam(x_flat):
    """Seam gradient using the KNOWN branching plane span{GV, HV}.

    At the seam the driver's h0 = d^{pq} * dE vanishes, so the u_h derived from
    the model is undefined exactly there; using the exact plane gives an honest
    reference for the sanity check.
    """
    dd = P @ (x_flat * W - Q0) - D0
    gm = P @ (KDIAG * dd)
    return np.linalg.norm(gm - (gm @ GV) * GV - (gm @ HV) * HV)


dE0 = _seam_and_gap(X_MECP)[0]
print(f"  sanity at the exact MECP:  gap {abs(dE0):.2e} Eh   "
      f"|f_seam| {_exact_plane_seam(X_MECP):.2e}   (both must be ~0)")
print()
hdr = (f"  {'start |dq|':>11s} {'steepest descent':>18s} {'quasi-Newton':>14s} "
       f"{'speedup':>8s} {'err_SD':>9s} {'err_QN':>9s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))
tot_sd = tot_qn = 0
for scale in (0.10, 0.25, 0.50, 1.00):
    x_start = (Q_MECP + P @ (rng.standard_normal(NDIM) * scale)) / W
    best = None
    for a in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        n, xx, ok = run_steepest_descent(x_start, a)
        if ok and (best is None or n < best[0]):
            best = (n, xx, ok, a)
    n_sd, x_sd, ok_sd, a_sd = best if best else (6000, x_start, False, float('nan'))
    n_qn, x_qn, ok_qn, info = run_quasi_newton(x_start)
    # compare the INTERNAL coordinate d = P(q - Q0). The model depends on x only
    # through d, so it is invariant to rigid-body motion and either optimizer may
    # drift freely in translation/rotation at zero energy cost; |x - x_exact|
    # would report that drift rather than any error in the located MECP.
    d_ref = P @ (X_MECP * W - Q0)
    e_sd = np.linalg.norm(P @ (x_sd * W - Q0) - d_ref)
    e_qn = np.linalg.norm(P @ (x_qn * W - Q0) - d_ref)
    tot_sd += n_sd; tot_qn += n_qn
    print(f"  {scale:11.2f} {str(n_sd) + ('' if ok_sd else '*'):>18s} "
          f"{str(n_qn) + ('' if ok_qn else '*'):>14s} {n_sd/max(n_qn,1):7.1f}x "
          f"{e_sd:9.2e} {e_qn:9.2e}   (SD alpha {a_sd:g})")
print("  " + "-" * (len(hdr) - 2))
print(f"  {'total':>11s} {tot_sd:18d} {tot_qn:14d} {tot_sd/max(tot_qn,1):7.1f}x")
print("\n  * = hit the step limit.  err = |d - d_exact| (internal coordinate,")
print("  mass-weighted Bohr). Either optimizer may drift rigidly at no energy")
print("  cost, so a plain |x - x_exact| would measure drift instead of error.")
print(f"  convergence: |gap| < {GAP_TOL:.1e} Eh (1e-4 eV) and |f_seam| < {SEAM_TOL:.0e}"
      "  -- the production thresholds.")
