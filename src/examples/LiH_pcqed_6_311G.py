import sys
sys.path.append("/home/jfoley19/vibronic/qed-ci/src/")
from morse import Morse
from vibronic_helper import Vibronic
import numpy as np

mol_str = """
Li
H 1 1.4
symmetry c1
"""

options = {
"only_singlets" : False,
"number_of_photons" : 10,
"number_of_electronic_states" : 250,
"omega" :  0.12086,
"basis" : "6-311G",
"lambda_vector" : np.array([0, 0, 0.02]),
"target_root" : 0,
"mass_A" : 1.,
"mass_B" : 6.941,
"qed_type" : "pcqed",
"ci_level" : "fci",
"molecule_template" :
"""
Li
H 1 **R**
symmetry c1
""",
"guess_bondlength" : 1.4,
}
X = Vibronic(options)

En = X.compute_qed_energy()
root = options["target_root"]
print(F'PCQED Energy for root {root} is {En}')

# See cell D50 here: https://docs.google.com/spreadsheets/d/1VlQ13mRHjHrfu2XhhD6aMdwFkatnwVciVjIusqkRbPY/edit?usp=sharing
_expected_energy = -8.0117857647

assert np.isclose(En, _expected_energy)
