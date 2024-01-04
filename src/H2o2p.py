import psi4
from helper_PFCI import PFHamiltonianGenerator
import numpy as np

mol_str = """
2 1
o 
h 1 1.0
h 1 1.0 2 104.5
symmetry c1
no_reorient
nocom
"""

options_dict = {'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

psi4.set_options(options_dict)
psi4.core.set_output_file('h2o_0.6.out', False)

#H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)



cavity_dict = {
        'omega_value' : 0.,
        'lambda_vector' : np.array([0, 0, 0.0]),
        'ci_level' : 'fci',
        'number_of_photons' : 0,
        "full_diagonalization" : True
}

test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

test_pf.compute_1_and_2_electron_energy(0)
test_pf.compute_1_and_2_electron_energy(1)
test_pf.compute_1_and_2_electron_energy(2)
test_pf.compute_1_and_2_electron_energy(3)
test_pf.compute_1_and_2_electron_energy(4)
