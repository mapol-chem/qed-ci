from oo_cqed_rhf import CQEDRHFCalculator
import numpy as np
import psi4
import sys
sys.path.append("/home/jfoley19/Code/qed-ci/src/")
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
from nuclear_grad import *
np.set_printoptions(threshold=sys.maxsize)
psi4.core.be_quiet()



## geometry template
mol_tmpl = """
0 1
Li 0.0    0.0   0.0
H  0.0    0.0  **R**
symmetry c1
no_reorient
nocom
"""



# lambda vector along z
lambda_vector = np.array([0, 0, 0.05])


# psi4 options
psi4_options = {
    "basis": "6-31g",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-7, # controls rhf and cqed-rhf convergence
    "d_convergence": 1e-5, 
}

options_dict = psi4_options

# controls casscf options
cavity_options = {
    'omega_value' : 0.12086,
    'lambda_vector' : np.array([0.0, 0.0, 0.05]),
    'ci_level' : 'cas',
    'number_of_photons' : 1,
    'photon_number_basis' : False,
    'canonical_mos' : False,
    'coherent_state_basis' : True,
    'spin_adaptation': "singlet",
    'davidson_roots' : 3,
    'davidson_threshold' : 1e-7,
    'davidson_maxdim': 8,
    'davidson_maxiter':100,
    'davidson_indim':6,
    'nact_orbs' : 4,
    'nact_els' : 4
}

# get a gradient for each state
n_states = cavity_options["davidson_roots"]

# conversion factor
BOHR_TO_ANGSTROM = 0.52917721092

# central geometry in Angstroms
r_center = 1.0

# instantiate cqed-rhf gradient object
mol_str = mol_tmpl.replace("**R**", f"{r_center:.6f}")
calc = CQEDRHFCalculator(lambda_vector, mol_str, psi4_options)

# get analytic gradient at center bond length, this gets full cartesian matrix
en1, cqed_rhf_grad1, g = calc.calc_force_and_energy(mol_str, use_psi4_scf_grad=False)
# get analytic gradient at center, this gets full cartesian matrix
en2, cqed_rhf_grad2, g = calc.calc_force_and_energy(mol_str, use_psi4_scf_grad=True)

## instantiate qed-cas gradient object
CASG = nuclear_grad(mol_str, options_dict, cavity_options)

# Initialize array for QED-CASSCF gradient elements for the z-coordinate of H atom
cqed_cas_analytic_grad = np.zeros((n_states,2,3))
for i in range(n_states):
    CASG.compute_grad(i)
    # get gradient for state i
    cqed_cas_analytic_grad[i,:,:] = CASG.total_gradient.reshape(2,3)


### set up numerical gradient for only z component of H
# displacement in angstroms
h_ang = 0.002

# displacement in Bohr
h_bohr = h_ang / BOHR_TO_ANGSTROM

# array of bond lengths in Angstrom to go into the geometries
r_vals = np.array([r_center - 2 * h_ang, r_center - h_ang, r_center + h_ang, r_center + 2 * h_ang])

# coefficients to multply the energies at each bond length by
coeffs = np.array([-1, 8, -8, 1]) / (12 * h_bohr)

# only ground state gradient for cqed-rhf by definition
CQED_RHF_E_Array = np.zeros(4)

# all n_states for qed-casscf
CQED_CASSCF_E_Array = np.zeros((n_states,4))

for idx, r in enumerate(r_vals):
    mol_str = mol_tmpl.replace("**R**", f"{r:.6f}")
    calc.molecule_string = mol_str
    calc.calc_cqed_rhf_energy()
    CQED_RHF_E_Array[idx] = calc.cqed_rhf_energy
    CAS = PFHamiltonianGenerator(mol_str, psi4_options, cavity_options)
    for i in range(n_states):
        CQED_CASSCF_E_Array[i, idx] = CAS.CASSCFeigs[i]

cqed_rhf_numerical_grad = np.dot(CQED_RHF_E_Array, coeffs)
cqed_cas_numerical_grad = np.dot(CQED_CASSCF_E_Array, coeffs)

print("Analytical CQED-RHF Gradient Without Density Fitting:\n")
print(cqed_rhf_grad1)

print("Analytical CQED-RHF Gradient With Density Fitting:\n")
print(cqed_rhf_grad1)

print("Numerical Gradient Element at CQED-RHF Level:\n")
print(cqed_rhf_numerical_grad)

### differences between different cqed-rhf gradient approximations to z-component of H atom
error_g1g2 = cqed_rhf_grad1[0,2] - cqed_rhf_grad2[0,2]
error_gng1 = cqed_rhf_grad1[0,2] - cqed_rhf_numerical_grad
error_gng2 = cqed_rhf_grad2[0,2] - cqed_rhf_numerical_grad

print(F"Error between g1 and g2: {error_g1g2:.12e}")
print(F"Error between gn and g1: {error_gng1:.12e}")
print(F"Error between gn and g2: {error_gng2:.12e}")

for i in range(n_states):
    print(F"Analytical CQED-CASSCF Gradient for state {i}")
    print(cqed_cas_analytic_grad[i,:,:])
    state_error = cqed_cas_analytic_grad[i,0,2] - cqed_cas_numerical_grad[i]
    print(F"Numerical Gradient Element z for atom H")
    print(cqed_cas_numerical_grad[i])
    print("Error between Analytic and Numerical")
    print(state_error)

