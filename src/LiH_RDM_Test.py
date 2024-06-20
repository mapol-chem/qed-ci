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
        'coherent_state_basis' : False, 
        'compute_properties' : True,
        'check_rdms' : True,
}


test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

print(" Printing 1-RDM")
print(test_pf.one_electron_rdm)
print(" Printing 2-RDM")
print(test_pf.two_electron_rdm)

print(" Total Energy from Trace( 2K 2D) for RDM from root ", cavity_dict["rdm_root"])
print(test_pf.total_energy_from_rdms)
print(" Corresponding CI Energy Eigenvalues ")
print(test_pf.CIeigs[0])


