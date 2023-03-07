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
from helper_cqed_rhf import *


def test_h2o_rhf():
    # options for H2O
    h2o_options_dict = {
        "basis": "cc-pVDZ",
        "save_jk": True,
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }
    
    # electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
    lam_h2o = np.array([0.0, 0.0, 0.05])
    
    
    
    # molecule string for H2O
    h2o_string = """
    
    0 1
        O      0.000000000000   0.000000000000  -0.068516219320
        H      0.000000000000  -0.790689573744   0.543701060715
        H      0.000000000000   0.790689573744   0.543701060715
    no_reorient
    symmetry c1
    """
    #energy from hilbert(scf_type = cd,cholesky_tolerance 1e-12)
    expected_h2o_e = -76.016355284146
    
    h2o_dict_origin = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)
    h2o_cqed_rhf_e = h2o_dict_origin["CQED-RHF ENERGY"]
    assert psi4.compare_values(h2o_cqed_rhf_e,expected_h2o_e)

def test_mgf_rhf():
    # options for mgf
    mgf_options_dict = {
        "basis": "cc-pVDZ",
        "save_jk": True,
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }
    
    # electric field for mgf - polarized along z-axis with mangitude 0.05 atomic units
    lam_mgf = np.array([0.0, 0.0, 0.05])
    
    
    
    # molecule string for H2O
    mgf_string = """
    
    1 1
    Mg 
    F    1  1.0
    no_reorient
    symmetry c1
    """
    #energy from hilbert(scf_type = cd,cholesky_tolerance 1e-12)
    expected_mgf_e = -297.621331340683
    
    mgf_dict_origin = cqed_rhf(lam_mgf, mgf_string, mgf_options_dict)
    mgf_cqed_rhf_e = mgf_dict_origin["CQED-RHF ENERGY"]
    assert psi4.compare_values(mgf_cqed_rhf_e,expected_mgf_e)


