import numpy as np
import sys
sys.path.append("../../../../src")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
np.set_printoptions(threshold=sys.maxsize)

mol_str = """
0 1
O
H 1 0.6
H 1 0.6 2 104.5  
symmetry c1
"""

options_dict = {'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

cavity_options = {
    'omega_value' : 0.0,
    'lambda_vector' : np.array([0, 0, 0.0]),
    'ci_level' : 'fci',
    'ignore_coupling' : True,
    'number_of_photons' : 1,
    'natural_orbitals' : False,
    'davidson_roots' : 6,
    'davidson_maxdim':10,
}

psi4.set_options(options_dict)
psi4.core.set_output_file('h2o_0.6.out', False)

H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)


# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# now compute cqed-rhf to get transformation vectors with cavity
##cqed_rhf_dict = cqed_rhf(lambda_vector, mol_str, options_dict)
##rhf_reference_energy = cqed_rhf_dict["RHF ENERGY"]
##cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
##C = cqed_rhf_dict["CQED-RHF C"]



# collect rhf wfn object as dictionary
##wfn_dict = psi4.core.Wavefunction.to_file(wfn)

# update wfn_dict with orbitals from CQED-RHF
##wfn_dict["matrix"]["Ca"] = C
##wfn_dict["matrix"]["Cb"] = C
# update wfn object
##wfn = psi4.core.Wavefunction.from_file(wfn_dict)
#psi4.set_options({'restricted_docc': [3],'active': [4],'num_roots':2})

fci_energy = psi4.energy('fci',ref_wfn=wfn)
print(np.allclose(-74.14620410799654,fci_energy,1e-8,1e-8))

