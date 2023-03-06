"""
Simple demonstration of CQED-RHF method on the water molecule
coupled to a strong photon field with comparison to results from 
code in the hilbert package described in [DePrince:2021:094112] and available
at https://github.com/edeprince3/hilbert

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, and helper_CQED_RHF <==
import numpy as np
import psi4
from helper_cqed_rhf import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2


# options for H2O
h2o_options_dict = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}


# molecule string for H2O
h2o_string = """

0 1
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
no_reorient
symmetry c1
"""


# create a shifted string of h2o coordinates here!
h2o_string_shifted = """

0 1
 #   O      0.000000000000   0.000000000000  -0.068516219320
 #   H      0.000000000000  -0.790689573744   0.543701060715
 #   H      0.000000000000   0.790689573744   0.543701060715
    O      0.000000000000   0.000000000000  100000.93148378068
    H      0.000000000000  -0.790689573744  100001.543701060715
    H      0.000000000000   0.790689573744  100001.543701060715
no_reorient
symmetry c1
"""


# energy for H2O from hilbert package described in [DePrince:2021:094112]
expected_h2o_e = -76.016355284146

# electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
lam_h2o = np.array([0.0, 0.0, 0.05])


# run cqed_rhf on H2O at origin
h2o_dict_origin = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)

# run cqed_rhf on H2O shifted 100 units along z
h2o_dict_shifted = cqed_rhf(lam_h2o, h2o_string_shifted, h2o_options_dict)


F_origin = h2o_dict_origin["CQED-RHF FOCK MATRIX"]
F_shifted = h2o_dict_shifted["CQED-RHF FOCK MATRIX"]

# need another call to cqed_rhf with shifted h2o coordinates
# catch the output in a dictionary called h2o_dict_shifted
# get F_shifted = h2o_dict_shifted["CQED-RHF FOCK MATRIX"]

assert np.allclose(F_origin, F_shifted, 1e-12, 1e-12)

# parse dictionary for ordinary RHF and CQED-RHF energy
h2o_cqed_rhf_e = h2o_dict_origin["CQED-RHF ENERGY"]
h2o_rhf_e = h2o_dict_origin["RHF ENERGY"]


print("\n    RHF Energy:                %.8f" % h2o_rhf_e)
print("    CQED-RHF Energy:           %.8f" % h2o_cqed_rhf_e)
print("    Reference CQED-RHF Energy: %.8f\n" % expected_h2o_e)

psi4.compare_values(h2o_cqed_rhf_e, expected_h2o_e, 8, "H2O CQED-RHF E")
