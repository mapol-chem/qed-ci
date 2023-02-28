"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
import pytest
import sys

 
# setting path
sys.path.append('../../src')
 
# importing
from src.helper_cqed_rhf import *


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


