#!/usr/bin/env python
# ============================================================================
# QED_SCF_GUESS_DF correctness gate -- WITH THE CAVITY ON.
#
# WHAT IS ACTUALLY AT RISK.  In the coherent-state basis the CASSCF Hamiltonian
# freezes three quantities at the CQED-RHF reference:
#     <d_HF> = d_exp_el       helper_cqed_rhf.py:577   (converged CQED-RHF rho)
#     d_c    = 1/2 <d_HF>^2   helper_cqed_rhf.py:580
#     d_cmo  = d_el_mo        helper_cqed_rhf.py:630   (converged CQED-RHF C)
# They reach the CASSCF as helper_PFCI.py:6318-6320 and are used as CONSTANTS
# (constdouble[3] = self.d_exp - d_diag, :4773).  The orbital optimiser rotates
# orbitals *inside* that Hamiltonian; it never re-optimises its origin.  So a
# density-fitted CQED-RHF does not give a worse guess to the same problem -- it
# solves a DIFFERENT one, and no amount of CASSCF convergence recovers it.
#
# QED_SCF_GUESS_DF density-fits only psi4's SCF, whose sole downstream product
# is the guess `C = wfn.Ca()`.  This script tests that claim where it can fail:
# at lambda != 0, where all three quantities are non-zero.  (An earlier
# lambda = 0 comparison agreed to 2.8e-12 and PROVED NOTHING: d_exp_el is
# identically zero there, so the constants under test do not exist.)
#
# DESIGN: both legs in ONE process, differing only in the module global, so the
# comparison cannot drift on psi4 version, threads, geometry or scratch state
# (same reasoning as validation/dpd_stream_check.py).
#
# PRE-REGISTERED GATE (standing rule 5).  These are CONVERGED SCF quantities,
# guess-independent in exact arithmetic, so the tolerance is the SCF's own
# convergence, not machine epsilon:
#   G1  |d_exp(df-guess) - d_exp(control)| < 1e-9
#   G2  |d_c  (df-guess) - d_c  (control)| < 1e-9
#   G3  max|D(df-guess) - D(control)| < SCF_TOL, D = the converged CQED-RHF AO
#       DENSITY MATRIX.  This, not d_cmo, is the basis-independent statement
#       that the two legs converged to the same reference.  The tolerance is
#       the SCF's own (d_convergence 1e-10), not machine epsilon -- see below.
#   G5  max|eig(d_cmo, df-guess) - eig(d_cmo, control)| < 1e-8.  The two legs
#       hold orbital sets spanning the same space, so their d_cmo are
#       ORTHOGONALLY SIMILAR and must share a spectrum -- the basis-independent
#       way to say "the same dipole operator" without asking which orbitals are
#       degenerate.
#   G4  |CQED-RHF energy difference| < 1e-9
#   R1  REPORT psi4's own SCF energy from each leg.  It SHOULD differ (it is
#       the density-fitted RHF energy); a leg where it does NOT differ means
#       the knob did not engage and the whole run is void.
#   G6  max|W| < SCF_TOL outside the degenerate blocks, W = C_ctrl^T S C_df.  The
#       sharpest statement of the lot: it says the two orbital sets differ ONLY
#       within sets of degenerate orbitals (where the SCF fixes nothing) and by
#       phase.  Measured 5e-13 at lambda_z = 0.2, with six degenerate pairs.
#   R2  REPORT max|d_cmo difference| as a DIAGNOSTIC, not a gate.
#
# WHY d_cmo IS NOT GATED (this was a gate in the first draft and it failed at
# 7e-2 while G1/G2/G4 passed at 1e-13).  d_cmo is COVARIANT with the orbitals,
# not frozen with the reference: helper_PFCI.py:5948-5949 rotates it with the
# orbital step (d_cmo <- U2^T d_cmo U2) and :6384/:6428/:6457 rebuild it as
# C^T d_ao C from the CURRENT C.  It is the fixed AO operator d_ao expressed in
# whatever MO basis is current, so it inherits every freedom C has -- phase, and
# rotation within a degenerate set (MgH+/cc-pVDZ has a degenerate pi pair, and
# the observed difference sits in exactly one such column).  Requiring it to be
# invariant under a change of GUESS would be requiring the guess to be unique,
# which it is not.  d_ao itself is a one-electron AO integral and never depends
# on the SCF at all.  The frozen, genuinely at-risk quantities are the SCALARS
# <d_HF> and d_c, which G1/G2 gate directly.
# (helper_PFCI.py:6310 self.d_hf IS a frozen deepcopy of d_cmo -- but it is
# written once and never read anywhere in the file.)
#
# usage:  python validation/scf_guess_df_check.py [LAMBDA_Z] [GUESS_TYPE]
# ============================================================================
import os
import sys

import numpy as np
import psi4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAMBDA_Z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
GUESS = sys.argv[2] if len(sys.argv) > 2 else "df"

# The knob is read at import time; drive it directly so both legs share one
# interpreter.  Import with it OFF so the mutual-exclusion check cannot fire on
# an inherited QED_SCF_REUSE_JK from the caller's environment.
os.environ["QED_SCF_GUESS_DF"] = "0"
os.environ.pop("QED_SCF_REUSE_JK", None)
import helper_cqed_rhf as hcr  # noqa: E402

MOL = """
Mg
H 1 1.7
symmetry c1
1 1
"""
OPTS = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "save_jk": False,
}

KEYS = {
    "d_exp": "COHERENT STATE EXPECTATION VALUE OF d",
    "d_c": "COHERENT STATE DIPOLE ENERGY",
    "E_cqed": "CQED-RHF ENERGY",
    "E_psi4": "RHF ENERGY",
}


def leg(tag, guess):
    hcr._QED_SCF_GUESS_DF = guess
    print(f"\n{'='*70}\n=== leg {tag}: _QED_SCF_GUESS_DF = {guess!r}\n{'='*70}",
          flush=True)
    d = hcr.cqed_rhf([0.0, 0.0, LAMBDA_Z], MOL, dict(OPTS))
    out = {k: d[v] for k, v in KEYS.items()}
    out["d_cmo"] = np.asarray(d["1-E DIPOLE MATRIX MO"])
    out["D"] = np.asarray(d["CQED-RHF DENSITY MATRIX"])
    out["eps"] = np.asarray(d["CQED-RHF EPS"])
    out["C"] = np.asarray(d["CQED-RHF C"])
    out["S"] = np.asarray(
        psi4.core.MintsHelper(d["PSI4 WFN"].basisset()).ao_overlap())
    return out


print(f"lambda_z = {LAMBDA_Z}, guess scf_type = {GUESS!r}")
ctrl = leg("control (pk throughout)", None)
test = leg(f"df-guess ({GUESS})", GUESS)

print(f"\n{'='*70}\nRESULT\n{'='*70}")
fails = []

# R1 first: if psi4's SCF energy is identical the knob never engaged and every
# other line below is a tautology.
dE_psi4 = abs(test["E_psi4"] - ctrl["E_psi4"])
print(f"R1  psi4 RHF energy   control {ctrl['E_psi4']:.12f}   "
      f"df-guess {test['E_psi4']:.12f}   diff {dE_psi4:.3e}")
if dE_psi4 == 0.0:
    fails.append("R1 VOID: psi4's SCF energy is identical -- the knob did not "
                 "engage, so the agreement below tests nothing")

for tag, key, tol in (("G1", "d_exp", 1e-9), ("G2", "d_c", 1e-9),
                      ("G4", "E_cqed", 1e-9)):
    diff = abs(test[key] - ctrl[key])
    ok = diff < tol
    print(f"{tag}  {key:8s} control {ctrl[key]:.14f}   "
          f"df-guess {test[key]:.14f}   diff {diff:.3e}  "
          f"[{'PASS' if ok else 'FAIL'} < {tol:g}]")
    if not ok:
        fails.append(f"{tag} {key} differs by {diff:.3e}")

# ONE tolerance for the two structural checks (G3, G6), and it is the SCF's own
# rather than machine epsilon: both compare quantities converged to
# d_convergence = 1e-10 above, so 100x that.  The lambda dependence is the
# reason it cannot be tighter, and it runs the OPPOSITE way to intuition:
#     lambda_z    0        0.05      0.2
#     G3 (D)      3.2e-09  1.9e-11   1.6e-12
#     G6 (W)      1.1e-09  4.8e-12   5.0e-13
# At strong coupling the CQED-RHF does real work from the guess and converges it
# away entirely; at lambda ~ 0 it has nothing to do, exits after one iteration,
# and the DF guess's own residual survives.  The energy is second order in that
# residual and agrees at 1e-13 either way (G4), as do <d_HF> and d_c (G1/G2).
SCF_TOL = 1e-8
D_TOL = SCF_TOL
dD = np.abs(test["D"] - ctrl["D"]).max()
ok = dD < D_TOL
print(f"G3  D (AO)   max|diff| {dD:.3e}   "
      f"[{'PASS' if ok else 'FAIL'} < {D_TOL:g}]")
if not ok:
    fails.append(f"G3 the converged CQED-RHF AO density differs by {dD:.3e} -- "
                 f"the two legs did NOT reach the same reference")

# G5 is the basis-INDEPENDENT statement about d_cmo, and it is the one worth
# gating.  Both legs hold full square orbital sets spanning the same AO space,
# so C_df = C_ctrl W with W orthogonal, hence d_cmo_df = W^T d_cmo_ctrl W: the
# two matrices are ORTHOGONALLY SIMILAR and their SPECTRA must coincide.  That
# tests "the same operator, expressed in some orthonormal basis" without any
# appeal to which orbitals happen to be degenerate -- which the first draft got
# wrong twice (it demanded d_cmo itself be invariant, then tried to excuse the
# difference by hunting for exactly-degenerate partners, and neither is right).
de = np.abs(np.sort(np.linalg.eigvalsh(test["d_cmo"]))
            - np.sort(np.linalg.eigvalsh(ctrl["d_cmo"]))).max()
ok = de < 1e-8
print(f"G5  eig(d_cmo) max|diff| {de:.3e}   [{'PASS' if ok else 'FAIL'} < 1e-08]"
      f"  -- d_cmo is the same operator in a rotated basis")
if not ok:
    fails.append(f"G5 the spectrum of d_cmo differs by {de:.3e} -- the two legs "
                 f"do NOT represent the same dipole operator")

# G6 says EXACTLY how the two orbital sets differ, and is the sharpest of the
# lot.  W = C_ctrl^T S C_df is the orthogonal matrix relating them.  If the legs
# converged to the same reference, W can be non-trivial only INSIDE sets of
# degenerate orbitals (where the SCF fixes nothing) and on the diagonal (phase);
# any weight outside those blocks is two genuinely different orbital sets.
eps = ctrl["eps"]
W = ctrl["C"].T @ ctrl["S"] @ test["C"]
blocks, cur = [], [0]
for j in range(1, len(eps)):
    if abs(eps[j] - eps[j - 1]) < 1e-7:
        cur.append(j)
    else:
        blocks.append(cur)
        cur = [j]
blocks.append(cur)
mask = np.ones_like(W, dtype=bool)
for b in blocks:
    mask[np.ix_(b, b)] = False
w_off = np.abs(W[mask]).max()
ok = w_off < SCF_TOL
print(f"G6  W=C^T S C  max|W| outside degenerate blocks {w_off:.3e}   "
      f"[{'PASS' if ok else 'FAIL'} < {SCF_TOL:g}]  -- the orbital sets differ ONLY "
      f"within\n    degenerate sets {[b for b in blocks if len(b) > 1]} and by phase")
if not ok:
    fails.append(f"G6 the orbital sets differ by {w_off:.3e} OUTSIDE any "
                 f"degenerate block -- they are not the same reference")

# R2: diagnostic only.  d_cmo itself is covariant with C (helper_PFCI.py:5948
# rotates it, :6384 rebuilds it), so a raw difference here is a rotation of the
# guess and is EXPECTED to be large -- 0.49 at lambda_z = 0.2.  G5 and G6 are
# what constrain it.  (Do not try to explain it away with a per-column sign
# taken from the diagonal overlap: that is meaningless inside a degenerate pair
# that has genuinely mixed, and it leaves a spurious residual behind.)
dmat = np.abs(test["d_cmo"] - ctrl["d_cmo"])
worst = int(np.argmax(dmat.max(axis=0)))
print(f"R2  d_cmo    max|diff| {dmat.max():.3e}  (worst column {worst}, "
      f"eps {eps[worst]:.10f})  -- diagnostic, not gated")

print()
if fails:
    print("FAIL\n  " + "\n  ".join(fails))
    sys.exit(1)
print("PASS: the density-fitted guess leaves <d_HF>, d_c and the converged "
      "CQED-RHF reference exact.\n      (d_cmo may differ by a rotation within "
      "a degenerate set -- see R2 -- which is a\n       freedom of the guess, "
      "not of the Hamiltonian.)")
