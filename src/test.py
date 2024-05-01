from vibronic_helper import Vibronic
import numpy as np

mol_str = """
Li
H 1 0.74
symmetry c1
"""

options = {
"number_of_photons" : 0,
"number_of_electronic_states" : 10,
"omega" : 0,
"basis" : "6-31G",
"lambda_vector" : np.array([0, 0, 0]),
"target_root" : 0,
"mass_A" : 1,
"mass_B" : 1,
"qed_type" : "qed-ci",
"molecule_template" :
"""
Li
H 1 **R**
symmetry c1
""",
"guess_bondlength" : 0.74,
}
X = Vibronic(options)

X.optimize_geometry_full_nr()
                         
