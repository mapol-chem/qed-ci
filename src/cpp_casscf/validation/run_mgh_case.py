"""Standalone MgH+ CAS(8,12) SA-QED-CASSCF driver for QN/BFGS convergence
experiments (user-provided system that takes >10 macroiterations).

Run from qed-ci/src (or with PYTHONPATH including it):
    python cpp_casscf/validation/run_mgh_case.py [--seed N]

Prints the per-macroiteration "Macroiteration ..." lines helper_PFCI.py
already emits; count them / read the final one for the macroiteration count.
"""
import argparse
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _src)

import psi4  # noqa: E402
from helper_PFCI import PFHamiltonianGenerator  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=float, default=1.0, help="Mg-H bond length (Angstrom)")
parser.add_argument("--seed", type=int, default=None,
                    help="np.random seed (for the orbital-guess randomness the QN path is sensitive to)")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

psi4.set_memory("8 GB")
# Respect OMP_NUM_THREADS (run_qn_case.py already does). This matters for
# more than speed: the threaded reductions in ci_solver.c are NOT
# bitwise-reproducible, and on this system a >1 thread count makes repeated
# runs of an IDENTICAL seeded configuration diverge -- measured on h2o/6-31G,
# OMP=1 reproduces exactly while OMP=3 gave 8 vs 11 macroiterations and even
# converged to a different solution. Any controlled variant comparison must
# therefore run single-threaded.
psi4.core.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

mol_str = f"""
1 1
Mg
H 1 {args.r}
symmetry c1
"""

options_dict = {
    "basis": "cc-pvdz",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}

cavity_options = {
    "omega_value": 0.000,
    "lambda_vector": [0, 0, 0.00],
    "ci_level": "cas",
    "ignore_coupling": False,
    "number_of_photons": 0,
    "natural_orbitals": False,
    "photon_number_basis": False,
    "canonical_mos": False,
    "coherent_state_basis": True,
    "davidson_roots": 3,
    "davidson_threshold": 1e-6,
    "davidson_maxdim": 200,
    "spin_adaptation": "singlet",
    "davidson_maxiter": 100,
    "davidson_indim": 140,
    "test_mode": False,
    "nact_orbs": 12,
    "nact_els": 8,
}

psi4.core.set_output_file(os.path.join(_here, "mgh.out"), False)
psi4.geometry(mol_str)

print(f"MgH+ CAS(8,12) cc-pVDZ R={args.r} seed={args.seed}", flush=True)
_t0 = time.time()
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
_t1 = time.time()
print(f"Done. CASSCF converged: {getattr(pf, 'casscf_converged', None)}  wall={_t1 - _t0:.1f}s", flush=True)
