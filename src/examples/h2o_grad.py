import numpy as np
import sys
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf7/qed-ci/src/")
#sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_112123/qed-ci/src/")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
from nuclear_grad import *
np.set_printoptions(threshold=sys.maxsize)

mol_str = """
0 1
   O            0.000000000000     0.000000000000    -0.068516219320   
   H            0.000000000000    -0.790689573744     0.543701060715   
   H            0.000000000000     0.790689573744     0.543701060715   
symmetry c1
no_reorient
nocom
"""

options_dict = {'basis': '6-31g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

cavity_options = {
    #'omega_value' : 0.12086,
    'omega_value' : 0.10000,
    'lambda_vector' : np.array([0.0, 0.0, 0.05]),
    'ci_level' : 'cas',
    #'casscf_optimization': False,
    'ignore_coupling' : False,
    'number_of_photons' : 1,
    'natural_orbitals' : False,
    'photon_number_basis' : False,
    'canonical_mos' : False,
    'coherent_state_basis' : True,
    'spin_adaptation': "singlet",
    'davidson_roots' : 4,
    'davidson_threshold' : 1e-7,
    'davidson_maxdim': 8,
    'davidson_maxiter':100,
    'davidson_indim':6,
    'test_mode': False,
    'nact_orbs' : 5,
    'nact_els' : 6 
}

psi4.set_options(options_dict)
psi4.core.set_output_file('h2o_1.0.out', False)

#H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
H2_PF = nuclear_grad(mol_str, options_dict, cavity_options)
state = 3
H2_PF.compute_grad(state)

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)
