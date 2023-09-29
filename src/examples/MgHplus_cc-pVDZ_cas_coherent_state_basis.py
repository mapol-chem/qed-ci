import numpy as np
import sys
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
np.set_printoptions(threshold=sys.maxsize)

# options for mgf
mol_str = """
Mg
H 1 2.2
symmetry c1
1 1
"""

options_dict = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}


#mol = psi4.geometry(mol_str)

cavity_options = {
    'omega_value' : 0.0,
    'lambda_vector' : np.array([0, 0, 0.0]),
    'ci_level' : 'cas',
    'ignore_coupling' : False,
    'number_of_photons' : 1,
    'natural_orbitals' : False,
    'davidson_roots' : 6,
    'davidson_maxdim':10,
    'nact_orbs' : 10,
    'nact_els' : 8,
    'full_diagonalization' : False,
    
}

H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
