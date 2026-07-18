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

Run from this directory: python dump_lih_case.py [--molecule lih|h2o] [--bond-length R] [--dump-dir NAME] ...
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
parser.add_argument("--dump-dir", default="dumps_lih", help="subdirectory (under this script's directory) to write dumps into")
args = parser.parse_args()

_here = os.path.dirname(os.path.abspath(__file__))
_dump_dir = os.path.join(_here, args.dump_dir)
os.environ["CPP_CASSCF_VALIDATION_DIR"] = _dump_dir

sys.path.insert(0, os.path.join(_here, "..", ".."))  # qed-ci/src

import numpy as np
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
    # Same geometry as examples/h2o_grad.py.
    mol_str = """
0 1
   O            0.000000000000     0.000000000000    -0.068516219320
   H            0.000000000000    -0.790689573744     0.543701060715
   H            0.000000000000     0.790689573744     0.543701060715
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
    "lambda_vector": np.array([0.0, 0.0, 0.01]),
    "ci_level": "cas",
    "ignore_coupling": False,
    "number_of_photons": 1,
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
print(f"molecule={args.molecule} R={args.bond_length} basis={args.basis} "
      f"active=({args.nact_els},{args.nact_orbs}) roots={args.davidson_roots}")
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print("Done. CASSCF converged:", getattr(pf, "casscf_converged", None))
