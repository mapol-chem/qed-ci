import numpy as np
import sys
sys.path.append("/home/nvu12/software/qed_ci_052223/qed-ci/src")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
np.set_printoptions(threshold=sys.maxsize)

mol_str = """
    0 1
    Li
    H 1 0.6
    symmetry c1
"""
def basisspec_psi4_yo__anonymous775(mol, role):
    basstrings = {}
    mol.set_basis_all_atoms("6-311++G(d,p)", role=role)
    mol.set_basis_by_symbol("Li", "libasis", role=role)
    basstrings['libasis'] = """
spherical
****
Li     0
S   6   1.00
    900.4600000              0.00228704       
    134.4330000              0.0176350        
     30.4365000              0.0873434        
      8.6263900              0.2809770        
      2.4833200              0.6587410        
      0.3031790              0.1187120        
SP   3   1.00
      4.8689000              0.0933293              0.0327661        
      0.8569240              0.9430450              0.1597920        
      0.2432270             -0.00279827             0.8856670        
SP   1   1.00
      0.0635070              1.0000000              1.0000000        
SP   1   1.00
      0.0243683              1.0000000              1.0000000        
D   1   1.00
      0.2000000              1.0000000        
SP   1   1.00
      0.0074000              1.0000000              1.0000000
****      
"""
    return basstrings
psi4.qcdb.libmintsbasisset.basishorde['ANONYMOUS775'] = basisspec_psi4_yo__anonymous775
options_dict = {'basis': 'anonymous775',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

cavity_options = {
    'omega_value' : 0.12,
    'lambda_vector' : np.array([0, 0, 0.015]),
    'ci_level' : 'cas',
    'ignore_coupling' : False,
    'natural_orbitals' : False,
    'davidson_roots' : 6,
    'davidson_maxdim':10,
    'nact_orbs' : 8,
    'nact_els' : 4,
    'full_diagonalization' : True,
    # specify the number of photons - 2 means |0> , |1>, |2> will be in the basis
    'number_of_photons' : 5
}

psi4.set_options(options_dict)
psi4.core.set_output_file('lih_0.6.out', False)

H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print(np.shape(H2_PF.H_PF))
for i in range(10):
 print('state', i, H2_PF.CIeigs[i])

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
######psi4.set_options({'restricted_docc': [2],'active': [8],'num_roots':2})
######fci_energy = psi4.energy('fci',ref_wfn=wfn)
######print(np.allclose(H2_PF.cis_e[0],fci_energy,1e-8,1e-8))

