"""Determinism test using a PINNED initial orbital set (developer's suggestion).

Runs h2o/6-31G CAS(4,4) SA-CASSCF either
  --mode make : with a seeded random orbital rotation, which helper_PFCI.py
                writes to orbital2.out as a side effect (the rotated STARTING
                orbitals, before any optimization), or
  --mode use  : with use_orbital_guess=True, reading orbital.out.

Copy orbital2.out -> orbital.out between the two to get a HARD (random,
far-from-converged) start that is bitwise identical on every repeat, which
removes the threaded-SCF jitter that a --mode make run still carries.

orbital.out / orbital2.out are FIXED filenames written into the CWD, so runs
sharing a directory clobber each other -- always run these SEQUENTIALLY, in a
scratch directory, never concurrently.
"""
import argparse
import os
import sys
import time

import numpy as np

SRC = "/home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src"
sys.path.insert(0, SRC)

import psi4  # noqa: E402
from helper_PFCI import PFHamiltonianGenerator  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["make", "use"], required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tag", default="run")
args = p.parse_args()

np.random.seed(args.seed)

psi4.set_memory("4 GB")
psi4.core.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

mol_str = "0 1\nO\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n"

# psi4 needs the orbital spaces spelled out for the use_orbital_guess path
# (ortho_script.py reads RESTRICTED_DOCC / ACTIVE). Water has 10 electrons;
# CAS(4,4) leaves (10-4)/2 == 3 doubly-occupied inactive orbitals.
options_dict = {
    "basis": "6-31g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "restricted_docc": [3],
    "active": [4],
}

cavity_options = {
    "omega_value": 0.0, "lambda_vector": [0, 0, 0.0], "ci_level": "cas",
    "ignore_coupling": False, "number_of_photons": 0, "natural_orbitals": False,
    "photon_number_basis": False, "canonical_mos": False, "coherent_state_basis": True,
    "davidson_roots": 1, "davidson_threshold": 1e-6, "davidson_maxdim": 8,
    "spin_adaptation": "singlet", "davidson_maxiter": 100, "davidson_indim": 6,
    "test_mode": False, "nact_orbs": 4, "nact_els": 4,
    "use_orbital_guess": args.mode == "use",
    "save_orbital": False,
}

psi4.core.set_output_file(f"psi4_{args.tag}.out", False)
psi4.geometry(mol_str)
print(f"mode={args.mode} seed={args.seed} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
_t0 = time.time()
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print(f"Done. converged={getattr(pf,'casscf_converged',None)} wall={time.time()-_t0:.1f}s", flush=True)
