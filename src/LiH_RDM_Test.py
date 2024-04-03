import psi4
from helper_PFCI import PFHamiltonianGenerator
import numpy as np

file_string = "LiH_sto3g_qedfci_om_12086_lam_05"

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
        'omega_value' : 0.,
        'lambda_vector' : np.array([0, 0, 0.0]),
        'ci_level' : 'fci',
        'davidson_roots' : 2,
        'number_of_photons' : 0,
        'rdm_root' : 0,
        'photon_number_basis' : True,
        'canonical_mos' : True,
        'coherent_state_basis' : False
}



#cavity_dict = {
#        'omega_value' : 0.12086,
#        'lambda_vector' : np.array([0, 0, 0.05]),
#        'ci_level' : 'fci',
#        'davidson_roots' : 2,
#        'number_of_photons' : 10,
#        'photon_number_basis' : True,
#        'canonical_mos' : True,
#        'coherent_state_basis' : False
#}



test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

print(test_pf.two_electron_rdm[:20])

# save rdms to npy files 
#d1_string = file_string + "_d1"
#d2_string = file_string + "_d2"


#np.save(d1_string, test_pf.one_rdm)
#np.save(d2_string, test_pf.two_rdm)



#print(test_pf.total_energy_from_rdms)
#print(test_pf.CIeigs[0])


