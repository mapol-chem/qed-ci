# set path to the directory where helper_PFCI.py is located
import sys
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf4/qed-ci/src/")

# import helper libraries
import numpy as np
import psi4
from helper_PFCI import PFHamiltonianGenerator
import json 

# set precision for printing numpy arrays
np.set_printoptions(threshold=sys.maxsize)

# create file prefix for output json file
file_string = "LiH_r_scan_6311g_qedfci"

# create template for molecular geometry
mol_str = """
Li
H 1 **R**
symmetry c1
"""

# specify basic input parameters for the electronic structure calculation
options_dict = {
        "basis": "6-311g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
}

# specify input parameters that are unique to the cavity quantum electrodynamics calculation
cavity_options = {
    'omega_value' : 0.12086,
    'lambda_vector' : np.array([0, 0, 0.05]),
    'ci_level' : 'cas',
    'ignore_coupling' : False,
    'number_of_photons' : 1,
    'natural_orbitals' : False,
    'photon_number_basis' : False,
    'canonical_mos' : False,
    'coherent_state_basis' : True,
    'davidson_roots' : 7,
    'davidson_threshold' : 1e-5,
    'davidson_maxdim':40,
    'spin_adaptation': "singlet",
    #'casscf_weight':np.array([1,1,1]),
    'davidson_maxiter':100,
    'davidson_indim':8,
    'test_mode': False,
    'nact_orbs' : 5,
    'nact_els' : 4
}

# create new dictionary to store calculation data
calculation_data = {

    "molecular_geometry" : {
        "z-matrix" : [],
    },
    "return_results" : {

        "energies" : [],
        "dipoles" : [],
        "transition_dipoles" : [],
    }

}
# store cavity parameters in dictionary
for keys, values in cavity_dict.items():
    calculation_data[keys] = values
# store standard electronic structure data in dictionary
for keys, values in options_dict.items():
    calculation_data[keys] = values

# going to loop over r values and compute QED-CASSCF energy for each one
# assign number of r values
N_r_values = 10

# create a list of r values between 1.0 and 3.0 angstroms
r_values = np.linspace(1.0, 3.0, N_r_values)

test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
)

# Sample data structure representing TD-DFT calculation results
data = {
    "basis_set": "6-31G*",
    "functional": "B3LYP",
    "convergence_threshold": 1e-6,
    "points": [
        {
            "geometry": [
                {"atom": "H", "x": 0.0, "y": 0.0, "z": 0.0},
                {"atom": "O", "x": 0.0, "y": 0.0, "z": 0.96},
                {"atom": "H", "x": 0.0, "y": 0.75, "z": -0.48}
            ],
            "ground_state_energy": -76.345,
            "excited_state_energies": [-76.234, -76.200],
            "ground_state_dipole_moment": [0.0, 0.0, 1.85],
            "excited_state_dipole_moments": [
                [0.1, 0.0, 1.9],
                [0.0, 0.2, 1.8]
            ],
            "transition_dipole_moments": [
                [0.05, 0.0, 0.1],
                [0.02, 0.1, 0.08]
            ]
        },
        {
            "geometry": [
                {"atom": "H", "x": 0.0, "y": 0.0, "z": 0.0},
                {"atom": "O", "x": 0.0, "y": 0.0, "z": 1.0},
                {"atom": "H", "x": 0.0, "y": 0.8, "z": -0.5}
            ],
            "ground_state_energy": -76.342,
            "excited_state_energies": [-76.230, -76.198],
            "ground_state_dipole_moment": [0.0, 0.0, 1.83],
            "excited_state_dipole_moments": [
                [0.1, 0.0, 1.88],
                [0.0, 0.2, 1.78]
            ],
            "transition_dipole_moments": [
                [0.05, 0.0, 0.09],
                [0.02, 0.1, 0.07]
            ]
        }
    ]
}

# Function to write data to JSON file
def write_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Writing data to a JSON file
output_filename = "td_dft_pes_data.json"
write_to_json(data, output_filename)

print(f"Data successfully written to {output_filename}")

