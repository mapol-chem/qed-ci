import psi4
from helper_PFCI import PFHamiltonianGenerator
import numpy as np

file_string = "LiH_r_1.4_6311g_fci"

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
        'davidson_roots' : 5,
        'number_of_photons' : 10,
        'davidson_threshold' : 1e-9,
        'photon_number_basis' : True,
        'canonical_mos' : True,
        'coherent_state_basis' : False
}

test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

# save eigenvalues and dipole elements to npy files 
#C_string = file_string + "_Eigenvectors"
#E_string = file_string + "_Energies"
#Mu_string = file_string + "_Dipoles"

#np.save(C_string, test_pf.CIvecs)
#np.save(E_string, test_pf.CIeigs)
#np.save(Mu_string, test_pf.dipole_array)

