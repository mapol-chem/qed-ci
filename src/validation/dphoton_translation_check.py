"""
Rigid-translation test of the nuclear half of d_photon -- SCF ONLY, no CI.

The quantity that must be checked is the displacement derivative

    dz/dx  =  [ d<d_e>/dx + lam_a * sum_A Z_A ] / sqrt(2w)

under a RIGID translation of every atom along the polarisation axis.  It is
fixed in closed form by the total charge:

    dz/dx  =  lam_a * Q / sqrt(2w) ,      Q = sum_A Z_A - N_e

so it must VANISH for a neutral molecule and change SIGN between a cation and
an anion.  The ion pair is the only test here that fixes the sign of d_N
relative to <d_e> WITHOUT relying on the neutral cancellation.

d_photon^{mn} = (dz/dx) G^{mn}, and G^{mn} depends on the CI vectors, but dz/dx
does NOT -- it needs only the CQED-RHF density and the nuclear geometry.  An
earlier version of this check ran a 3-root FCI per point and spent all its time
converging eigenvectors that never entered the answer; this one calls cqed_rhf
directly.  Seconds instead of tens of minutes.

    python dphoton_translation_check.py
    QED_FD_DNUC=0 python dphoton_translation_check.py     # control: must FAIL
"""
import argparse, os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psi4
from helper_cqed_rhf import cqed_rhf

ANG2BOHR = 1.8897261254578281
_FD_DNUC = os.environ.get("QED_FD_DNUC", "1") not in ("0", "", "false")
OMEGA, LAM = 0.1212727168509417, 0.05          # lambda along x

CASES = {
    "h2o":  dict(chg=0,  syms=["O", "H", "H"],
                 pos=np.array([[0.0, 0.0, -0.0685162193],
                               [0.0, -0.7906895737, 0.5437010607],
                               [0.134217, 0.831145, 0.601233]])),
    "h3o+": dict(chg=1,  syms=["O", "H", "H", "H"],
                 pos=np.array([[0.0, 0.0, 0.0], [0.0, -0.938, 0.320],
                               [0.812, 0.469, 0.320], [-0.812, 0.469, 0.320]])),
    "oh-":  dict(chg=-1, syms=["O", "H"],
                 pos=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.970]])),
    "li+":  dict(chg=1,  syms=["Li"], pos=np.array([[0.0, 0.0, 0.0]])),
}


def mol_string(sd, pos):
    body = "\n".join(f"{a} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
                     for a, p in zip(sd["syms"], pos))
    return (f"{sd['chg']} 1\n" + body +
            "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom")


def d_at(sd, pos, basis):
    opts = {"basis": basis, "scf_type": "pk", "e_convergence": 1e-11,
            "d_convergence": 1e-11, "reference": "rhf"}
    res = cqed_rhf(np.array([LAM, 0.0, 0.0]), mol_string(sd, pos), opts)
    d_el = float(res["COHERENT STATE EXPECTATION VALUE OF d"])
    mu_n = np.asarray(res["NUCLEAR DIPOLE MOMENT"])
    d_nuc = float(np.dot(np.array([LAM, 0.0, 0.0]), mu_n)) if _FD_DNUC else 0.0
    return d_el, d_nuc


def run(name, h, basis):
    sd = CASES[name]
    shift = np.zeros_like(sd["pos"]); shift[:, 0] = h      # ALL atoms, along x
    ep, np_ = d_at(sd, sd["pos"] + shift, basis)
    em, nm = d_at(sd, sd["pos"] - shift, basis)
    hb = h * ANG2BOHR
    dz = ((ep + np_) - (em + nm)) / (2.0 * hb) / np.sqrt(2.0 * OMEGA)
    dz_el = (ep - em) / (2.0 * hb) / np.sqrt(2.0 * OMEGA)
    exact = LAM * sd["chg"] / np.sqrt(2.0 * OMEGA)
    res = abs(dz - exact)
    print(f"  {name:6s} Q={sd['chg']:+d}   dz/dx = {dz:+.8e}   required "
          f"{exact:+.8e}   residual {res:.2e}   (electronic only {dz_el:+.6e})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--cases", default="h2o,h3o+,oh-,li+")
    a = ap.parse_args()
    psi4.core.be_quiet(); psi4.core.set_output_file("dphoton_trans.psi4.out", False)
    print(f"RIGID TRANSLATION along x, h = {a.h} Ang, basis {a.basis}, lam = {LAM}")
    print(f"QED_FD_DNUC = {'1 (nuclear dipole INCLUDED)' if _FD_DNUC else '0 (OMITTED -- control, must fail)'}\n")
    worst = max(run(n.strip(), a.h, a.basis) for n in a.cases.split(","))
    print(f"\nWORST residual {worst:.3e}   -> {'PASS' if worst < 1e-6 else 'FAIL'}")
    sys.exit(0 if worst < 1e-6 else 1)
