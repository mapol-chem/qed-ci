import psi4
from helper_PFCI import PFHamiltonianGenerator 
import numpy as np
mol_str = """
Li
H 1 1.4
symmetry c1
"""

options_dict = {
        "basis": "sto-3g",
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
        'photon_number_basis' : True,
        'canonical_mos' : True,
        'coherent_state_basis' : False
}

# Np = 10 results
_expected_eigs = np.array([-7.8749559054562726,
        -7.756654252046298,
        -7.747674819146274,
        -7.729087280855876,
        -7.701685049223775,
        -7.7016850492237685,
        -7.678038607940438])

test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

assert np.allclose(test_pf.CIeigs[:7], _expected_eigs[:7])
