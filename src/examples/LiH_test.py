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

# conversion factor from Bohr to Angstrom
BOHR_TO_ANGSTROM = 0.52917721092

# delta value for geometry displacements in Angstroms
delta_ang = 1e-4

# delta value for geometry displacements in Bohr
delta_au = delta_ang / BOHR_TO_ANGSTROM

## initial geometry
original_mol_string = """
0 1
Li 0.0    0.0   0.0
H  0.0    0.0   1.2
no_reorient
nocom
symmetry c1
"""

# get basic info about the atom
mol = psi4.geometry(original_mol_string)
num_atoms = mol.natom()

# lambda vector 
lambda_vector = np.array([0, 0, 0.05])

# psi4 options
options_dict = {
    "basis": "6-31g",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-7, # controls rhf and cqed-rhf convergence
    "d_convergence": 1e-5, 
}

### STEP 1: Get CQED-RHF Gradient of the ground state analytically and with finite differences
calc = CQEDRHFCalculator(lambda_vector, original_mol_string, options_dict)

# store mol_string as an attribute of calc
calc.molecule_string = original_mol_string

# get analytic cqed-rhf gradient 
en, cqed_rhf_analytical_grad, g = calc.calc_force_and_energy(original_mol_string, use_psi4_scf_grad=False)

# get numerical cqed-rhf gradient - not returned by this function, but stored in attribute self.numerical_energy_gradient
calc.compute_numerical_gradient()
# store numerical gradient 
cqed_rhf_numerical_grad = calc.numerical_energy_gradient
# restore calc.molecule_string to be safe
calc.molecule_string = original_mol_string

### STEP 2: Get CQED-RHF Gradient of all selected states analytically and with finite differences

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

## instantiate qed-cas gradient object
CASG = nuclear_grad(original_mol_string, options_dict, cavity_options)

# store number of states 
n_states = cavity_options["davidson_roots"]

# Initialize array for QED-CASSCF gradient elements for the z-coordinate of H atom
cqed_cas_analytic_grad = np.zeros((n_states,num_atoms,3))
for i in range(n_states):
    CASG.compute_grad(i)
    # get gradient for state i
    cqed_cas_analytic_grad[i,:,:] = CASG.total_gradient.reshape(2,3)


### set up numerical QED-CASSCF gradient
cqed_cas_numeric_grad = np.zeros((n_states, num_atoms, 3))

# loop over atoms and coords and do displacements along each
for i in range(num_atoms):
    for j in range(3):
        _displacement = np.zeros((num_atoms, 3))

        # displacement for atom i along coord j
        _displacement[i, j] = delta_ang

        # forward-displaced molecule string
        _mol_string_f = calc.modify_geometry_string(original_mol_string, _displacement)
        
        # backward-displaced molecule string
        _mol_string_b = calc.modify_geometry_string(original_mol_string, -1 * _displacement)

        # CASSCF calculation at forward-displaced geometry
        CAS_F = PFHamiltonianGenerator(_mol_string_f, options_dict, cavity_options)

        # CASSCF calculation at backward-displaced geometry
        CAS_B = PFHamiltonianGenerator(_mol_string_b, options_dict, cavity_options)

        # loop over states and compute gradient element for each one
        for k in range(n_states):
            cqed_cas_numeric_grad[k, i, j] = (CAS_F.CASSCFeigs[k] - CAS_B.CASSCFeigs[k]) / (2 * delta_ang / BOHR_TO_ANGSTROM)



print("Analytical CQED-RHF Gradient:\n")
print(cqed_rhf_analytical_grad)

print("Numerical CQED-RHF Gradient:\n")
print(cqed_cas_numeric_grad)

cqed_rhf_grad_norm = np.linalg.norm(cqed_rhf_analytical_grad-cqed_rhf_numerical_grad)

cqed_cas_norms = np.zeros(n_states)
for i in range(n_states):
    print(F"Analytical CQED-CASSCF Grad for State {i}:\n")
    print(cqed_cas_analytic_grad[i,:,:])

    print(F"Numeric CQED-CASSCF Grad for State {i}:\n")
    print(cqed_cas_numeric_grad[i,:,:])

    cqed_cas_norms[i] = np.linalg.norm(cqed_cas_analytic_grad[i,:,:] - cqed_cas_numeric_grad[i,:,:])


print(F"Norm of CQED-RHF Error is {cqed_rhf_grad_norm}")
for i in range(n_states):
    print(F"Norm of CQED-CASSCF Error for state {i} is {cqed_cas_norms[i]}")

    if np.isclose(cqed_cas_norms[i], cqed_rhf_grad_norm):
        print("This error is acceptable")
    else:
        print("This error is not acceptable")
