import numpy as np
import json
import sys
sys.path.append("/home/jfoley19/Code/qed-ci/src/")
#import sys
#sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf6/qed-ci/src")
import psi4
from helper_PFCI import PFHamiltonianGenerator

options_dict = {"basis": "6-31g", "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-10}
cavity_options = {"omega_value": 0.219,
    "lambda_vector": [0, 0, 0.05],
    "ci_level": "cas",
    "ignore_coupling": False,
    "number_of_photons": 1,
    "natural_orbitals": False,
    "photon_number_basis": False,
    "canonical_mos": False,
    "coherent_state_basis": True,
    "davidson_roots": 1,
    "davidson_threshold": 1e-8,
    "davidson_maxdim": 8,
    #"spin_adaptation": "singlet", 
    #"use_orbital_guess": True,
    #"save_orbital": True,
    "davidson_maxiter": 100,
    "davidson_indim": 8,
    "test_mode": False,
    "nact_orbs": 2,
    "nact_els": 2
    }  # Fixed nact_els to 4

mol_tmpl = """
-1 1
O 0.000 0.000 -0.450
H 0.000 0.000  0.450   
symmetry c1
nocom 
no_reorient
"""

r_val = 1.0
mol_str = mol_tmpl.replace('**R**', str(r_val))
psi4.set_options({'restricted_docc': [4],'active': [2],'num_roots':1})

psi4.set_options(options_dict)
psi4.core.set_output_file(f'oh_2_2_1.out', False)
test_pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)

