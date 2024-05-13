from morse import Morse
from vibronic_helper import Vibronic
import numpy as np
import sys
sys.path.append("/home/jfoley19/vibronic/qed-ci/src/")

mol_str = """
H
F 1 0.91783
symmetry c1
"""

options = {
"only_singlets" : True,
"number_of_photons" : 0,
"number_of_electronic_states" : 35,
"omega" : 0,
"basis" : "6-311++G**",
"lambda_vector" : np.array([0, 0, 0]),
"target_root" : 0,
"mass_A" : 1.0,
"mass_B" : 19.0,
"qed_type" : "qed-ci",
"ci_level" : "cas",
"nact_orbs" : 16,
"nact_els" : 6,
"molecule_template" :
"""
H
F 1 **R**
symmetry c1
""",
"guess_bondlength" : 0.917831,
}
X = Vibronic(options)


X.compute_potential_scan(r_min=0.5, r_max=3.5, N_points=300, filename="hf_pes_35_states_only_singlets_300_points.npy")

