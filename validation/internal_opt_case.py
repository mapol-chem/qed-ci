"""A small real run that actually EXERCISES ``internal_optimization3``.

WHY THIS EXISTS.  ``validation/h2e_disk_case.py`` (LiH CAS(4,6)) and the H4
dimer decks never reach the internal optimizer at all: it is called only under

    if macroiteration > 0 and self.n_in_a > 0:        # helper_PFCI.py:5790

and both of those put EVERY electron in the active space, so ``n_in_a == 0``.
Any change to ``internal_optimization3`` -- the extraction of its LSTRS
beta-bisection, the hard-case fallback at that site, ``QED_INTERNAL_EXACT_TRS``
-- is therefore INVISIBLE to the existing gate, which would report a clean pass
while validating nothing.  That is the failure mode this file removes.

LiH/6-31G CAS(2,6) leaves one inactive orbital, reaches the internal optimizer
5 times, and finishes in a few seconds.

usage (all optional, defaults as shown):
    LAMBDA_Z=0.05 OMEGA=0.12 NACT_ELS=2 NACT_ORBS=6 MAXDIM=8 INDIM=4 \\
        python validation/internal_opt_case.py

Everything it prints on the trajectory is deterministic; the ``ROOT0`` line is
the summary. Note the Davidson dimensions are deliberately small: CAS(2,6) has
few determinants and the shipping defaults (maxdim 20 / indim 8) abort with
"subspace size is too large".
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import psi4                                          # noqa: E402
from helper_PFCI import PFHamiltonianGenerator       # noqa: E402

np.random.seed(0)
psi4.set_memory("2 GB")
psi4.core.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
psi4.core.be_quiet()

mol_str = """
0 1
Li
H 1 1.6
symmetry c1
"""
options_dict = {
    "basis": "6-31G",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}
cavity_options = {
    "omega_value": float(os.environ.get("OMEGA", "0.12")),
    "lambda_vector": [0.0, 0.0, float(os.environ.get("LAMBDA_Z", "0.05"))],
    "ci_level": "cas",
    "ignore_coupling": False,
    "number_of_photons": 1,
    "natural_orbitals": False,
    "photon_number_basis": False,
    "canonical_mos": False,
    "coherent_state_basis": True,
    "davidson_roots": 2,
    "davidson_threshold": 1e-8,
    "davidson_maxdim": int(os.environ.get("MAXDIM", "8")),
    "davidson_maxiter": 100,
    "davidson_indim": int(os.environ.get("INDIM", "4")),
    "spin_adaptation": "singlet",
    "test_mode": False,
    "nact_orbs": int(os.environ.get("NACT_ORBS", "6")),
    "nact_els": int(os.environ.get("NACT_ELS", "2")),
}

psi4.geometry(mol_str)
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
for _i, _e in enumerate(np.atleast_1d(pf.CIeigs)[:cavity_options["davidson_roots"]]):
    print("ROOT%d %.12f" % (_i, _e), flush=True)
