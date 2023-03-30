import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
import numpy as np
import pytest
import sys

np.set_printoptions(threshold=sys.maxsize)

def test_h2o_qed_rhf():
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

def test_mgf_qed_rhf():
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


def test_build_1rdm():

    mol_str = """
    0 1
    O
    H 1 1.0
    H 1 1.0 2 104.0
    symmetry c1
    """

    options_dict = {
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cis',
        'ignore_coupling' : False,
        'number_of_photons' : 1,
        'natural_orbitals' : False,
        'nact_orbs' : 0,
        'nact_els' : 0
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)
    psi4.core.set_output_file("output.dat", False)


    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)

    test_pf.calc1RDMfromCIS(wavefunctions[:, 14])

    # test traces of different blocks
    expected_trace_Dij = 9.0
    expected_trace_Dab = 1.0
    expected_trace_D1 = 10.0
    expected_trace_D1_spatial = 10.0

    trace_Dij = np.trace(test_pf.Dij)
    trace_Dab = np.trace(test_pf.Dab)
    trace_D1 = np.trace(test_pf.D1)
    trace_D1_spatial = np.trace(test_pf.D1_spatial)

    assert np.isclose(expected_trace_D1, trace_D1)
    assert np.isclose(expected_trace_D1_spatial, trace_D1_spatial)
    assert np.isclose(expected_trace_Dab, trace_Dab)
    assert np.isclose(expected_trace_Dij, trace_Dij)

    # test trace of K1 D1 against <C|1H|C>
    e1_test_rdm = np.einsum("pq,pq->", test_pf.Hspin, test_pf.D1)

    # 1H @ C
    temp = np.einsum("pq,q->p", test_pf.H_1E, wavefunctions[:, 14])
    e1_test_wfn = np.dot(wavefunctions[:, 14].T, temp)

    expected_1e_energy = e1_test_wfn - test_pf.Enuc

    assert np.isclose(e1_test_rdm, expected_1e_energy)

def test_build_1rdm_with_Davidson():

    mol_str = """
    0 1
    O
    H 1 1.0
    H 1 1.0 2 104.0
    symmetry c1
    """

    options_dict = {
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cis',
        'davidson_roots' : 20
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)
    psi4.core.set_output_file("output.dat", False)


    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)

    _tmp_14 = test_pf.cis_c[:,14]

    # note that this Davidson vector is likely shorter than the real eigenvector
    _lt = len(_tmp_14)
    _davidson_vec_14 = np.zeros(len(test_pf.H_PF[0,:]))
    _davidson_vec_14[:_lt] = _tmp_14

    
    test_pf.calc1RDMfromCIS(_davidson_vec_14)

    # test traces of different blocks
    expected_trace_Dij = 9.0
    expected_trace_Dab = 1.0
    expected_trace_D1 = 10.0
    expected_trace_D1_spatial = 10.0

    trace_Dij = np.trace(test_pf.Dij)
    trace_Dab = np.trace(test_pf.Dab)
    trace_D1 = np.trace(test_pf.D1)
    trace_D1_spatial = np.trace(test_pf.D1_spatial)

    assert np.isclose(expected_trace_D1, trace_D1)
    assert np.isclose(expected_trace_D1_spatial, trace_D1_spatial)
    assert np.isclose(expected_trace_Dab, trace_Dab)
    assert np.isclose(expected_trace_Dij, trace_Dij)

    # test trace of K1 D1 against <C|1H|C>
    e1_test_rdm = np.einsum("pq,pq->", test_pf.Hspin, test_pf.D1)

    # 1H @ C
    temp = np.einsum("pq,q->p", test_pf.H_1E, _davidson_vec_14)
    e1_test_wfn = np.dot(_davidson_vec_14.T, temp)

    expected_1e_energy = e1_test_wfn - test_pf.Enuc

    assert np.isclose(e1_test_rdm, expected_1e_energy)




