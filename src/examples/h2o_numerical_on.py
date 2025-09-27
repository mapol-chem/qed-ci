import numpy as np
import sys
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf6/qed-ci/src/")
#sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_112123/qed-ci/src/")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
from nuclear_grad import *
np.set_printoptions(threshold=sys.maxsize)

#mol_str = """
#0 1
#li
#h 1 1.0
#symmetry c1
#no_reorient
#nocom
#"""
#
#options_dict = {'basis': 'sto-3g',
#                  'scf_type': 'pk',
#                  'e_convergence': 1e-10,
#                  'd_convergence': 1e-10
#                  }
#
#mol = psi4.geometry(mol_str)
#
#cavity_options = {
#    #'omega_value' : 0.12086,
#    'omega_value' : 0.00000,
#    'lambda_vector' : np.array([0.0, 0.0, 0.00]),
#    'ci_level' : 'cas',
#    #'casscf_optimization': False,
#    'ignore_coupling' : False,
#    'number_of_photons' : 0,
#    'natural_orbitals' : False,
#    'photon_number_basis' : False,
#    'canonical_mos' : False,
#    'coherent_state_basis' : True,
#    'spin_adaptation': "singlet",
#    'davidson_roots' : 2,
#    'davidson_threshold' : 1e-5,
#    'davidson_maxdim': 6,
#    'davidson_maxiter':100,
#    'davidson_indim':4,
#    'test_mode': False,
#    'nact_orbs' : 4,
#    'nact_els' : 2 
#}
#
#psi4.set_options(options_dict)
#psi4.core.set_output_file('lih_2.3.out', False)

#H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
#H2_PF = nuclear_grad(mol_str, options_dict, cavity_options)
#H2_PF.compute_grad()

# First compute SCF energy using Psi4
#scf_e, wfn = psi4.energy('SCF', return_wfn=True)
def modify_geometry_string(geometry_string, displacement_array):
    """
    Extracts Cartesian coordinates from a Psi4 geometry string, applies a
    transformation function to the coordinates, and returns a new geometry string.

    Args:
        geometry_string (str): A Psi4 molecular geometry string.
        transformation_function (callable): A function that takes a NumPy
            array of Cartesian coordinates (N x 3) as input and returns a
            NumPy array of the same shape with the transformed coordinates.

    Returns:
        str: A new Psi4 molecular geometry string with the transformed coordinates.
    """
    lines = geometry_string.strip().split('\n')
    atom_data = []
    symmetry = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("symmetry"):
            symmetry = line
            continue
        parts = line.split()
        if len(parts) == 4:
            atom = parts[0]
            try:
                x, y, z = map(float, parts[1:])
                atom_data.append([atom, x, y, z])
            except ValueError:
                # Handle cases where the line might not be atom coordinates
                pass

    if not atom_data:
        return ""

    coordinates = np.array([[data[1], data[2], data[3]] for data in atom_data])

    # Apply the transformation function
    transformed_coordinates = displacement_array  + coordinates

    new_geometry_lines = []
    for i, data in enumerate(atom_data):
        atom = data[0]
        new_geometry_lines.append(f"{atom} {transformed_coordinates[i, 0]:.8f} {transformed_coordinates[i, 1]:.8f} {transformed_coordinates[i, 2]:.8f}")

    new_geometry_string = "\n".join(new_geometry_lines)
    if symmetry:
        new_geometry_string += f"\n{symmetry}"
    return f"""{new_geometry_string}"""


def run_psi4_calculation(geometry_string, displacement_array, basis_set='sto-3g', method='scf'):
    """
    Runs a Psi4 calculation with the given geometry string, basis set, and method.

    Args:
        geometry_string (str): A Psi4 molecular geometry string.
        displacement_array (np.ndarray): An array of displacements to apply to the coordinates.
        basis_set (str): The basis set to use for the calculation, defaults to 'sto-3g'.
        method (str): The quantum chemistry method to use for the calculation, defaults to 'scf'.
        

    Returns:
        dict: A dictionary containing the results of the Psi4 calculation.
    """
    # Modify the geometry string with the displacement array
    modified_geometry_string = modify_geometry_string(geometry_string, displacement_array)
    # Add charge and multiplicity if not present
    full_geometry_string = f"""
    0 1
    {modified_geometry_string}
    no_reorient
    nocom
    """
    print("modify",modified_geometry_string)
   

    options_dict = {'basis': '6-31g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                   }

    
    cavity_options = {
        #'omega_value' : 0.12086,
        'omega_value' : 0.10000,
        'lambda_vector' : np.array([0.0, 0.0, 0.05]),
        'ci_level' : 'cas',
        #'casscf_optimization': False,
        'ignore_coupling' : False,
        'number_of_photons' : 1,
        'natural_orbitals' : False,
        'photon_number_basis' : False,
        'canonical_mos' : False,
        'coherent_state_basis' : True,
        'spin_adaptation': "singlet",
        'davidson_roots' : 4,
        'davidson_threshold' : 1e-5,
        'davidson_maxdim': 10,
        'davidson_maxiter':100,
        'davidson_indim':4,
        'test_mode': False,
        'nact_orbs' : 5,
        'nact_els' : 6 
    }
    
    psi4.set_options(options_dict)
    #psi4.core.set_output_file('lih_2.3.out', False)
    
    
    #psi4.core.clean()                 # wipes wave-functions, basis sets, etc.

    mol = psi4.geometry(modified_geometry_string)  # new molecule string
    psi4.core.set_active_molecule(mol)
    # Run the Psi4 calculation
    try:
        H2_PF = PFHamiltonianGenerator(full_geometry_string, options_dict, cavity_options)
        energy = H2_PF.eigenvals[3]
    except Exception as e:
        print(f"Error during Psi4 calculation: {e}")
        return None

    return energy


def compute_psi4_numerical_gradient(geometry_string, displacement_unit=0.01, basis_set='sto-3g', method='scf'):
    """
    NEEDS COMPLETING: Computes the numerical gradient for a given geometry string using Psi4.

    Args:
        geometry_string (str): A Psi4 molecular geometry string.
        displacement_unit (float): The unit of displacement for numerical gradient calculation, defaults to 0.01.
        basis_set (str): The basis set to use for the calculation, defaults to 'sto-3g'.
        method (str): The quantum chemistry method to use for the calculation, defaults to 'scf'.

    Returns:
        np.ndarray: The numerical gradient as a NumPy array.
    """
    # Set up Psi4 options
    psi4.set_options({
        'basis': basis_set,
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
    })

    # Get the number of atoms from the geometry string
    num_atoms = len(geometry_string.strip().split('\n')) - 1

    # Initialize the numerical gradient array
    numerical_gradient = np.zeros((num_atoms, 3))

    # loop over all atoms
    for i in range(num_atoms):

        # loop over all three dimensions
        for j in range(3):
            # Create a displacement unit vector
            displacement_array = np.zeros((num_atoms, 3))
            displacement_array[i, j] = displacement_unit

            # print the displacement unit vector
            print("displacement", displacement_array)

            # Insert code to compute psi4 energy at forward displacement
            e_f = run_psi4_calculation(geometry_string, displacement_array, basis_set, method)
            print("finish e forward")
            # Insert code to compute psi4 energy at backwards displacement
            displacement_array *= -1
            e_b = run_psi4_calculation(geometry_string, displacement_array, basis_set, method)
            print("finish e backward")

            # Insert code to compute finite difference along this displacement
            grad_element = (e_f - e_b) / (2 * 1.88973 * displacement_unit)

            # Insert code to store this to the appropriate gradient element
            numerical_gradient[i, j] = grad_element


    # return the gradient
    return numerical_gradient



starting_string = """
     O            0.000000000000     0.000000000000    -0.068516219320   
     H            0.000000000000    -0.790689573744     0.543701060715   
     H            0.000000000000     0.790689573744     0.543701060715   
symmetry c1
"""

starting_displacement = np.array([[0, 0.0, 0.0], [0, 0, 0.0], [0, 0, 0.0]])

# Run the Psi4 calculation
#energy = run_psi4_calculation(starting_string, starting_displacement, basis_set='sto-3g', method='scf')

# test the numerical gradient
numerical_gradient = compute_psi4_numerical_gradient(starting_string, displacement_unit=0.0001, basis_set='6-31g', method='scf')
print(numerical_gradient)
print(np.linalg.norm(numerical_gradient))
