"""
Runs a real SA-QED-CASSCF optimization (LiH or H2O, small active space)
through the pure-Python helper_PFCI.py driver, with the validation dump
hooks (see CPP_CASSCF_VALIDATION_DIR handling in helper_PFCI.py) turned on.
Each real trust-region solve at any of the instrumented solver call sites
(internal_optimization3's LSTRS, microiteration_optimization6's GLTR /
Davidson-driven-LSTRS / QN-GLTR / QN-BFGS dispatch) gets written out as a
(hessian, gradient, trust_radius, step) instance under the dump directory,
one directory per solve.

These dumps are then replayed through the corresponding ported C++ solvers
(cpp_casscf/tools/validate_against_python.cpp) to check for an exact step
match -- validating the ported solver logic against real chemistry, without
requiring the (still-unported) intermediates-building code to be ported.

Run from this directory: python dump_lih_case.py [--molecule lih|h2o] [--bond-length R]
[--h2o-oh-bond-length R] [--h2o-hoh-angle DEG] [--omega W] [--lambda-vector LX LY LZ]
[--n-photons N] [--dump-dir NAME] ...
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--molecule", choices=["lih", "h2o"], default="lih")
parser.add_argument("--bond-length", type=float, default=1.6, help="Li-H bond length in Angstrom (lih only)")
parser.add_argument("--basis", default="sto-3g")
parser.add_argument("--nact-orbs", type=int, default=2)
parser.add_argument("--nact-els", type=int, default=2)
parser.add_argument("--davidson-roots", type=int, default=1)
parser.add_argument("--davidson-maxdim", type=int, default=8, help="multiplied by davidson-roots; must stay below H_dim/roots or the C solver sys.exit()s")
parser.add_argument("--davidson-indim", type=int, default=6, help="multiplied by davidson-roots; must stay below H_dim/roots or the C solver sys.exit()s")
parser.add_argument("--omega", type=float, default=0.1)
parser.add_argument("--lambda-vector", type=float, nargs=3, default=(0.0, 0.0, 0.01), metavar=("LX", "LY", "LZ"),
                     help="cavity coupling (lambda) vector, lab-frame Cartesian, atomic units. Default (0,0,0.01) "
                          "matches this script's previous hardcoded value.")
parser.add_argument("--n-photons", type=int, default=1, help="number_of_photons (Fock-space truncation)")
parser.add_argument("--h2o-oh-bond-length", type=float, default=1.0,
                     help="O-H bond length in Angstrom (h2o only). Default 1.0 matches this script's previous "
                          "hardcoded geometry.")
parser.add_argument("--h2o-hoh-angle", type=float, default=104.5,
                     help="H-O-H bond angle in degrees (h2o only). Default 104.5 matches this script's previous "
                          "hardcoded geometry.")
parser.add_argument("--dump-dir", default="dumps_lih", help="subdirectory (under this script's directory) to write dumps into")
parser.add_argument("--random-seed", type=int, default=None,
                     help="TEMPORARY, for controlled comparison against the cpp_casscf port only -- seeds "
                          "numpy's global RNG before the CASSCF run starts, so the two random draws "
                          "helper_PFCI.py makes (GLTR's trust-region-perturbation noise and "
                          "linear_equation_solve's small-gradient initial probe) are reproducible run to "
                          "run. Does NOT make Python match the C++ port's own specific draws (different RNG "
                          "algorithm entirely, and unseeded by default on both sides otherwise) -- only makes "
                          "each side internally deterministic so repeated runs/debugging are comparable. "
                          "Omit (default) for normal, unseeded production behavior.")
args = parser.parse_args()

_here = os.path.dirname(os.path.abspath(__file__))
_dump_dir = os.path.join(_here, args.dump_dir)
os.environ["CPP_CASSCF_VALIDATION_DIR"] = _dump_dir

sys.path.insert(0, os.path.join(_here, "..", ".."))  # qed-ci/src

import numpy as np

if args.random_seed is not None:
    np.random.seed(args.random_seed)

import psi4
from helper_PFCI import PFHamiltonianGenerator

if args.molecule == "lih":
    mol_str = f"""
0 1
Li 0.0 0.0 0.0
H  0.0 0.0 {args.bond_length}
symmetry c1
no_reorient
nocom
"""
else:
    # Parametric water geometry: O on the z axis, both H atoms placed
    # symmetrically in the yz plane (the same C2-axis-along-z, molecule-in-
    # the-yz-plane convention the previous hardcoded geometry used) -- this
    # orientation matters physically here, not just the internal bond
    # length/angle, since --lambda-vector's default points along z (the QED
    # coupling depends on the molecular dipole's orientation relative to the
    # lab-fixed cavity polarization). O is placed at the origin rather than
    # the previous hardcoded geometry's slightly-offset z (-0.068516...):
    # with `nocom`/`no_reorient` set (no auto-recentering), any Cartesian
    # dipole computed here is the CI dipole of a *neutral* molecule, which is
    # origin-independent, and psi4's own electronic-structure energies are
    # manifestly translation-invariant regardless -- so this offset was never
    # physically load-bearing, and --h2o-oh-bond-length 1.0 --h2o-hoh-angle
    # 104.5 (the defaults) reproduce the exact same physics as the old
    # hardcoded string, just rigidly translated along z.
    _half_angle = np.radians(args.h2o_hoh_angle / 2.0)
    _r = args.h2o_oh_bond_length
    _h_y = _r * np.sin(_half_angle)
    _h_z = _r * np.cos(_half_angle)
    mol_str = f"""
0 1
   O            0.000000000000     0.000000000000     0.000000000000
   H            0.000000000000    {-_h_y:.12f}    {_h_z:.12f}
   H            0.000000000000     {_h_y:.12f}    {_h_z:.12f}
symmetry c1
no_reorient
nocom
"""

options_dict = {
    "basis": args.basis,
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}

cavity_options = {
    "omega_value": args.omega,
    "lambda_vector": np.array(args.lambda_vector),
    "ci_level": "cas",
    "ignore_coupling": False,
    "number_of_photons": args.n_photons,
    "natural_orbitals": False,
    "photon_number_basis": False,
    "canonical_mos": False,
    "coherent_state_basis": True,
    "spin_adaptation": "singlet",
    "davidson_roots": args.davidson_roots,
    "davidson_threshold": 1e-7,
    "davidson_maxdim": args.davidson_maxdim,
    "davidson_maxiter": 100,
    "davidson_indim": args.davidson_indim,
    "test_mode": False,
    "nact_orbs": args.nact_orbs,
    "nact_els": args.nact_els,
}

psi4.set_options(options_dict)
psi4.core.set_output_file(os.path.join(_here, "dump_lih.out"), False)
psi4.geometry(mol_str)

print(f"Dumping validation cases to {_dump_dir}")
if args.molecule == "lih":
    print(f"molecule={args.molecule} R={args.bond_length} basis={args.basis} "
          f"active=({args.nact_els},{args.nact_orbs}) roots={args.davidson_roots}")
else:
    print(f"molecule={args.molecule} OH={args.h2o_oh_bond_length} HOH_angle={args.h2o_hoh_angle} "
          f"basis={args.basis} active=({args.nact_els},{args.nact_orbs}) roots={args.davidson_roots}")
print(f"omega={args.omega} lambda_vector={list(args.lambda_vector)} n_photons={args.n_photons}")
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print("Done. CASSCF converged:", getattr(pf, "casscf_converged", None))
