"""
MECP scan driven by the quasi-Newton optimizer instead of fixed-step descent.

Same I/O, restart, CSV and orbital-reuse behaviour as scan_conical_*.py; the only
changes are the three that matter:

  1. the geometry update is MECPOptimizer.step(), not
         pos + effective_step * total_force
  2. the coupling passed in is `job.h_vector` (the gap-divided derivative
     coupling) rather than `job.h0`. h0 = d^{pq} * (E_q - E_p) VANISHES as the
     gap closes, which leaves the second branching direction undefined exactly
     at convergence; h_vector stays finite there.
  3. a CASSCF failure calls opt.reject(), which restores the previous geometry,
     halves the trust radius and clears the BFGS history, instead of only
     halving a global step size.

Usage:
    MECP_LAM=0.4 MECP_START=/path/to/start.xyz MECP_PREFIX=mecp_qn_040 \\
        python validation/drivers/ethylene_mecp_scan.py
Environment:
    MECP_LAM       coupling strength, a.u.                  (default 0.4)
    MECP_START     starting geometry (xyz)                   (required)
    MECP_PREFIX    output prefix                             (default mecp_qn)
    MECP_MAX_STEPS step cap                                  (default 200)
    MECP_TRUST     initial trust radius, Bohr                (default 0.30)
    MECP_NACT      active orbitals                           (default 7)
Run from an installed or editable QED-CI checkout. ASE is an optional dependency.
"""

import os, csv, shutil
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

import psi4
from nuclear_grad import *                      # noqa: F401,F403  (nuclear_grad)
from mecp_optimizer import MECPOptimizer, HARTREE2EV

LAM        = float(os.environ.get("MECP_LAM", "0.4"))
START      = os.environ["MECP_START"]
PREFIX     = os.environ.get("MECP_PREFIX", "mecp_qn")
MAX_STEPS  = int(os.environ.get("MECP_MAX_STEPS", "200"))
TRUST      = float(os.environ.get("MECP_TRUST", "0.30"))
NACT       = int(os.environ.get("MECP_NACT", "7"))
# Convergence thresholds are knobs because the shipped seam_tol of 5e-04 is
# UNREACHABLE on the locked-orientation ethylene deck: the Cartesian seam
# plateaus at 5.7-7.0e-04, which is why the original scans needed three manual
# 500-step legs and had to be stopped by eye. f_seam_mw is the metric with room
# (it fell 6.1x where the Cartesian norm barely moved, the residual being
# carbon-localised), so PLATEAU termination on seam_mw is the default stop.
GAP_TOL    = float(os.environ.get("MECP_GAP_TOL",  "3.7e-6"))
SEAM_TOL   = float(os.environ.get("MECP_SEAM_TOL", "5.0e-4"))
PLATEAU_N  = int(os.environ.get("MECP_PLATEAU_N", "6"))
# MECP_ALIGN_ATOMS: re-impose the fixed BOND DIRECTION after every step by
# rigidly rotating the molecule so that the vector between these two atoms lies
# on +x, the cavity polarisation axis. The original drivers align once at the
# start and then rely on project_out_transrot, which removes only INFINITESIMAL
# rotations -- so finite rotation accumulates. Measured on the published
# lambda=0.4 geometries: 0.03-0.22 deg of tilt, and a 0.0987 deg tilt alone
# shifts E0 by 1.46e-05 Eh and the gap by 2.90e-04 eV, i.e. ~3x the 1e-04 eV gap
# tolerance. A rigid rotation lies in the null space of the projector, so this
# does not disturb the BFGS history; it only enforces the intended constraint.
# Set MECP_ALIGN_ATOMS="" to disable.
ALIGN = os.environ.get("MECP_ALIGN_ATOMS", "4,5").strip()
ALIGN_IJ = tuple(int(x) for x in ALIGN.split(",")) if ALIGN else None
STATE_P, STATE_Q = 0, 1

options_dict = {
    'basis': 'aug-cc-pvdz', 'scf_type': 'pk', 'soscf': 'true',
    'level_shift': 0.1, 'e_convergence': 1e-8, 'd_convergence': 1e-8,
    'restricted_docc': [6], 'active': [NACT], 'num_roots': 1,
}
cavity_options = {
    'omega_value': 0.1212727168509417,
    'lambda_vector': np.array([LAM, 0.0, 0.0]),
    'ignore_coupling': False, 'number_of_photons': 1 if LAM > 0 else 0,
    'ci_level': 'cas', 'davidson_roots': 2, 'nact_orbs': NACT, 'nact_els': 4,
    'coherent_state_basis': True, 'spin_adaptation': "singlet",
    'davidson_threshold': 1e-7, 'davidson_maxdim': 21,
    'davidson_maxiter': 100, 'davidson_indim': 11,
    'save_orbital': True, 'use_orbital_guess': False,
}

def realign(positions):
    """Rigidly rotate so positions[j]-positions[i] lies on +x."""
    if ALIGN_IJ is None:
        return positions
    i, j = ALIGN_IJ
    p = np.asarray(positions, dtype=float)
    v = p[j] - p[i]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return p
    a = v / n
    b = np.array([1.0, 0.0, 0.0])
    cr = np.cross(a, b)
    sn = np.linalg.norm(cr)
    cs = float(a @ b)
    if sn < 1e-14:
        return p if cs > 0 else -p
    K = np.array([[0, -cr[2], cr[1]], [cr[2], 0, -cr[0]], [-cr[1], cr[0], 0]])
    Rm = np.eye(3) + K + K @ K * (1.0 - cs) / (sn * sn)
    com = p.mean(0)
    return (Rm @ (p - com).T).T + com


atoms = read(START)
symbols = atoms.get_chemical_symbols()
geom_dir, data_dir = f'{PREFIX}_geometries', f'{PREFIX}_casscf_data'
os.makedirs(geom_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
csv_file, traj_file = f'{PREFIX}_vectors.csv', f'{PREFIX}_trajectory.traj'

with open(csv_file, 'w', newline='') as f:
    wr = csv.writer(f)
    hdr = ['step', 'R_CC', 'gap_eV', 'seam_norm', 'seam_norm_mw', 'g_eff_norm',
           'trust', 'step_norm', 'cos_phi', 'e_p', 'e_q']
    for i, s in enumerate(symbols):
        hdr += [f'gx_{s}{i}', f'gy_{s}{i}', f'gz_{s}{i}']
    for i, s in enumerate(symbols):
        hdr += [f'hx_{s}{i}', f'hy_{s}{i}', f'hz_{s}{i}']
    wr.writerow(hdr)
traj = Trajectory(traj_file, 'a')

opt = MECPOptimizer(symbols, atoms.get_positions(), trust=TRUST,
                    gap_tol=GAP_TOL, seam_tol=SEAM_TOL)
mw_hist = []

print("=" * 68, flush=True)
print(f"Quasi-Newton MECP search   states {STATE_P}/{STATE_Q}   lambda = {LAM}")
print(f"  CAS(4,{NACT})   start {START}")
print(f"  initial trust {TRUST} Bohr   max steps {MAX_STEPS}")
print("=" * 68, flush=True)

first = True
for step in range(MAX_STEPS):
    cavity_options['use_orbital_guess'] = (not first) and os.path.exists('orbital.out')
    if first and os.path.exists('orbital.out'):
        os.remove('orbital.out')

    pos = realign(opt.positions_ang)
    opt.x = pos * 1.8897261254578281      # keep the optimizer on the aligned frame
    atoms.set_positions(pos)
    mol_str = "\n".join(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}"
                        for s, p in zip(symbols, pos))
    mol_str += "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom"

    job = nuclear_grad(mol_str, options_dict, cavity_options)

    job.compute_grad(STATE_P, STATE_P)
    if not getattr(job, 'casscf_converged', True):
        print(f"Step {step}: SA-CASSCF failed on state {STATE_P}; rejecting step.", flush=True)
        opt.reject(); first = False
        continue
    grad_p = np.array(job.total_gradient).reshape(-1, 3)
    e_p = float(job.eigenvals[STATE_P])

    job.compute_grad(STATE_Q, STATE_Q)
    if not getattr(job, 'casscf_converged', True):
        print(f"Step {step}: SA-CASSCF failed on state {STATE_Q}; rejecting step.", flush=True)
        opt.reject(); first = False
        continue
    grad_q = np.array(job.total_gradient).reshape(-1, 3)
    e_q = float(job.eigenvals[STATE_Q])

    job.compute_grad(STATE_P, STATE_Q)
    # h_vector = h0 / (E_q - E_p); finite as the gap closes, unlike h0 itself.
    h_src = getattr(job, 'h_vector', None)
    if h_src is None:
        h_src = job.h0
        print("  NOTE: h_vector unavailable, falling back to h0", flush=True)
    h = np.array(h_src).reshape(-1, 3)

    r_cc = atoms.get_distance(4, 5)
    info = opt.step(e_p, e_q, grad_p, grad_q, h)

    comment = (f"Step={step} Gap={info['gap_eV']:.8f}eV R_CC={r_cc:.5f}A "
               f"seam={info['seam_norm']:.6e} cos_phi={info['cos_phi']:.4f}")
    write(f'{geom_dir}/step_{step:04d}.xyz', atoms, comment=comment)
    atoms.info.update(step=step, gap_eV=info['gap_eV'], seam=info['seam_norm'])
    traj.write(atoms)
    for fn in ['orbital.out', 'occupation_number.out', 'natural_orbital.out']:
        if os.path.exists(fn):
            b, e = os.path.splitext(fn)
            shutil.copy(fn, os.path.join(data_dir, f"{b}_step_{step:04d}{e}"))
    with open(csv_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            step, r_cc, info['gap_eV'], info['seam_norm'], info['seam_norm_mw'],
            info['g_eff_norm'], info['trust'], info['step_norm_bohr'],
            info['cos_phi'], e_p, e_q,
            *(grad_q - grad_p).ravel(), *h.ravel()])

    print(f"Step {step:4d}: Gap={info['gap_eV']:12.8f} eV | R_CC={r_cc:.5f} A | "
          f"seam={info['seam_norm']:.4e} | seam_mw={info['seam_norm_mw']:.4e} | "
          f"|g_eff|={info['g_eff_norm']:.4e} | trust={info['trust']:.3f} | "
          f"|dx|={info['step_norm_bohr']:.4f}", flush=True)

    # PLATEAU STOP -- tests the recent TREND, not an all-time best.
    #
    # A previous version counted steps since seam_mw last beat its best ever.
    # That fails: the starting geometry's seam_mw is often the lowest value seen
    # for a while, because the first few steps move the geometry substantially
    # while BFGS has no curvature yet. One early excursion then locks the
    # detector out permanently. Observed at lambda=0.1, CAS(4,9): it stopped at
    # step 6 with seam_mw 2.65e-03 and still descending, purely because step 0
    # had read 1.92e-03.
    #
    # Instead: compare the best of the last N steps against the best of the N
    # before that, and stop only when that improves by less than 5%. Requires
    # 2N steps of history, so it cannot fire during the initial transient.
    mw_hist.append(info['seam_norm_mw'])
    plateaued = False
    if len(mw_hist) >= 2 * PLATEAU_N and abs(info['gap_hartree']) < GAP_TOL:
        recent = min(mw_hist[-PLATEAU_N:])
        earlier = min(mw_hist[-2 * PLATEAU_N:-PLATEAU_N])
        plateaued = recent >= 0.95 * earlier
    if plateaued and not info['converged']:
        print("=" * 68)
        print(f"MECP PLATEAUED at step {step}: seam_mw {recent:.4e} vs {earlier:.4e} "
              f"over the previous {PLATEAU_N} steps (<5% improvement); gap inside tol")
        print(f"  R_CC = {r_cc:.5f} A   Gap = {info['gap_eV']:.8f} eV")
        print(f"  seam = {info['seam_norm']:.6e}   seam_mw = {info['seam_norm_mw']:.6e}")
        print("=" * 68, flush=True)
        write(f'{PREFIX}_converged.xyz', atoms, comment=comment)
        break

    if info['converged']:
        print("=" * 68)
        print(f"MECP CONVERGED at step {step}")
        print(f"  R_CC = {r_cc:.5f} A   Gap = {info['gap_eV']:.8f} eV")
        print(f"  seam = {info['seam_norm']:.6e}   seam_mw = {info['seam_norm_mw']:.6e}")
        print(f"  trust shrinks {opt.n_shrink}   rejects {opt.n_reject}")
        print("=" * 68, flush=True)
        write(f'{PREFIX}_converged.xyz', atoms, comment=comment)
        break
    first = False

traj.close()
write(f'{PREFIX}_final.xyz', atoms)
print(f"\nFinal geometry -> {PREFIX}_final.xyz", flush=True)
