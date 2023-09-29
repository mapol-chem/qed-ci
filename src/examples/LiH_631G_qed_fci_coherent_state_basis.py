import psi4
from helper_PFCI import PFHamiltonianGenerator
import numpy as np
mol_str = """
Li
H 1 1.4
symmetry c1
"""

options_dict = {
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
}

cavity_dict = {
        'omega_value' : 0.12086,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'davidson_roots' : 10,
        'number_of_photons' : 10,
        'photon_number_basis' : False,
        'canonical_mos' : False,
        'coherent_state_basis' : True
}

test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)
~                                                                                                                                                                                                                     
~                                                                                                                                                                                                                     
~                                                                                                                                                                                                                     
~                                      
