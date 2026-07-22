"""General small-system SA-CASSCF runner for QN/BFGS convergence experiments.

Parametrized molecule / basis / active space / roots so cheap systems (that
still take many macroiterations under a random orbital guess) can be swept
quickly. QN experiment toggles are env-gated inside helper_PFCI.py:
    QED_DISABLE_QN, QED_ENERGY_TRIGGER_ACTUAL, QED_HESSIAN_RESET_NEG_CURV,
    QED_RANDOM_ORBITAL_SEED

Prints helper_PFCI.py's "Macroiteration ..." lines; count them for the
macroiteration count. Run from qed-ci/src or with PYTHONPATH including it.
"""
import argparse
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..")))

import psi4  # noqa: E402
from helper_PFCI import PFHamiltonianGenerator  # noqa: E402

# name -> (charge, mult, geometry-with-**R**)
MOLECULES = {
    "lih":  (0, 1, "0 1\nLi\nH 1 **R**\n"),
    "hf":   (0, 1, "0 1\nH\nF 1 **R**\n"),
    "h2o":  (0, 1, "0 1\nO\nH 1 **R**\nH 1 **R** 2 104.5\n"),
    "beh2": (0, 1, "0 1\nBe\nH 1 **R**\nH 1 **R** 2 180.0\n"),
    "mgh":  (1, 1, "1 1\nMg\nH 1 **R**\n"),
}

p = argparse.ArgumentParser()
p.add_argument("--mol", default="lih", choices=list(MOLECULES))
p.add_argument("--basis", default="cc-pvdz")
p.add_argument("--r", type=float, default=1.6)
p.add_argument("--nact-orbs", type=int, default=4)
p.add_argument("--nact-els", type=int, default=2)
p.add_argument("--roots", type=int, default=1)
p.add_argument("--maxdim", type=int, default=8)
p.add_argument("--indim", type=int, default=6)
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

psi4.set_memory("4 GB")
psi4.core.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

charge, mult, geom = MOLECULES[args.mol]
mol_str = geom.replace("**R**", str(args.r)) + "symmetry c1\n"

options_dict = {"basis": args.basis, "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-10}
cavity_options = {
    "omega_value": 0.0, "lambda_vector": [0, 0, 0.0], "ci_level": "cas",
    "ignore_coupling": False, "number_of_photons": 0, "natural_orbitals": False,
    "photon_number_basis": False, "canonical_mos": False, "coherent_state_basis": True,
    "davidson_roots": args.roots, "davidson_threshold": 1e-6, "davidson_maxdim": args.maxdim,
    "spin_adaptation": "singlet", "davidson_maxiter": 100, "davidson_indim": args.indim,
    "test_mode": False, "nact_orbs": args.nact_orbs, "nact_els": args.nact_els,
}

psi4.core.set_output_file(os.path.join(_here, f"qn_{args.mol}.out"), False)
psi4.geometry(mol_str)
print(f"{args.mol} {args.basis} R={args.r} CAS({args.nact_els},{args.nact_orbs}) "
      f"roots={args.roots} seed={args.seed}", flush=True)
_t0 = time.time()
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print(f"Done. CASSCF converged: {getattr(pf, 'casscf_converged', None)}  wall={time.time()-_t0:.1f}s", flush=True)
