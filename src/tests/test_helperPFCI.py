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

def test_h2o_qed_fci_no_cavity():

    options_dict = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'fci',
        'number_of_photons' : 1,
        'davidson_roots' : 4,
        'davidson_threshold' : 1e-8
    }

    # molecule string for H2O
    h2o_string = """
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """

    test_pf = PFHamiltonianGenerator(
        h2o_string,
        options_dict,
        cavity_dict
    )
    expected_g   = -75.0129801827
    excpected_e1 = -74.7364625844

    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_e1 = test_pf.CIeigs[2] # <== first excited state

    assert np.isclose(actual_g, expected_g)
    assert np.isclose(actual_e1, excpected_e1)

def test_mghp_qed_cis_dipole_calculations_no_cavity():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cis',
        'davidson_roots' : 8,
        'davidson_threshold' : 1e-8,
        'full_diagonalization' : True,
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    expected_e4 = -199.69011028328705

    # psi4's cis tdm between gs and first singlet excited state
    expected_mu_04 = np.array([-6.93218490e-16, -1.77990759e-15,  2.33258251e+00])
    
    # psi4's rhf dipole moment
    expected_mu_rhf = np.array([1.23668023e-15,  3.26291298e-15, -1.50279429e+00])

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    actual_e4 = test_pf.CIeigs[4]

    actual_mu_04 = test_pf.compute_dipole_moment(0, 4)
    actual_mu_g = test_pf.compute_dipole_moment(0, 0) + test_pf.mu_nuc
    assert np.isclose(actual_e4, expected_e4)
    assert np.allclose( np.abs(actual_mu_04), np.abs(expected_mu_04))
    assert np.allclose( actual_mu_g, expected_mu_rhf)


def test_mghp_qed_cis_no_cavity():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.0,
        'lambda_vector' : np.array([0, 0, 0]),
        'ci_level' : 'cis',
        'davidson_roots' : 8,
        'davidson_threshold' : 1e-8
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_eg = -199.8639591041915
    
    expected_mghp_e1 = -199.6901102832973

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_e0 = test_pf.CIeigs[0] # <== ground state
    actual_e1 = test_pf.CIeigs[4] # <== root 5 is first singlet excited state

    assert np.isclose(actual_e0, expected_mghp_eg)
    assert np.isclose(actual_e1, expected_mghp_e1)

def test_mghp_qed_cis_with_cavity():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0125]),
        'ci_level' : 'cis',
        'davidson_roots' : 8,
        'davidson_threshold' : 1e-8
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_g_e = -199.86358254419457
    expected_mghp_lp_e = -199.69776087489558
    expected_mghp_up_e = -199.68066502792058


    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_lp = test_pf.CIeigs[2] # <== root 3 is LP
    actual_up = test_pf.CIeigs[5] # <== root 6 is UP

    assert np.isclose(actual_g, expected_mghp_g_e)
    assert np.isclose(actual_lp, expected_mghp_lp_e)
    assert np.isclose(actual_up, expected_mghp_up_e)


def test_mghp_qed_cis_with_cavity_canonical_mo():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0125]),
        'ci_level' : 'cis',
        'full_diagonalization' : True,
        'canonical_mos' : True
    }

    mol = psi4.geometry(mol_str)

    psi4.set_options(options_dict)

    #energy from psi4numpy
    expected_mghp_g_e = -199.86358254419457
    expected_mghp_lp_e = -199.69776087489558
    expected_mghp_up_e = -199.68066502792058

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    #e_fci, wavefunctions = np.linalg.eigh(test_pf.H_PF)
    actual_g = test_pf.CIeigs[0] # <== ground state
    actual_lp = test_pf.CIeigs[2] # <== root 3 is LP
    actual_up = test_pf.CIeigs[5] # <== root 6 is UP

    assert np.isclose(actual_g, expected_mghp_g_e )
    assert np.isclose(actual_lp, expected_mghp_lp_e)
    assert np.isclose(actual_up, expected_mghp_up_e)


def test_mghp_qed_cis_no_cavity_compare_mos():
    # options for mgf
    mol_str = """
    Mg
    H 1 2.2
    symmetry c1
    1 1
    """

    options_dict = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cmo_cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0]),
        'ci_level' : 'cis',
        'full_diagonalization' : True,
        'canonical_mos' : True
    }

    qedmo_cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.0]),
        'ci_level' : 'cis',
        'full_diagonalization' : True
    }

    qedmo = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        qedmo_cavity_dict
    )
    cmo = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cmo_cavity_dict
    )

    assert np.allclose(qedmo.CIeigs[:50], cmo.CIeigs[:50], 1e-8)

def test_h2o_qed_fci_compare_mos():

    options_dict = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cmo_cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'number_of_photons' : 6,
        'full_diagonalization' : True,
        'canonical_mos' : True
    }

    qedmo_cavity_dict = {
        'omega_value' : 4.75 / psi4.constants.Hartree_energy_in_eV,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'number_of_photons' : 6,
        'full_diagonalization' : True
    }

    # molecule string for H2O
    h2o_string = """
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """

    qedmo = PFHamiltonianGenerator(
        h2o_string,
        options_dict,
        qedmo_cavity_dict
    )
    cmo = PFHamiltonianGenerator(
        h2o_string,
        options_dict,
        cmo_cavity_dict
    )

    assert np.allclose(qedmo.CIeigs[:5], cmo.CIeigs[:5], 1e-8)

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
        'davidson_roots' : 4,
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

    _tmp_g = test_pf.CIvecs[:,0]


    # note that this Davidson vector is likely shorter than the real eigenvector
    _lt = len(_tmp_g)
    _davidson_vec_g = np.zeros(len(test_pf.H_PF[0,:]))
    _davidson_vec_g[:_lt] = _tmp_g

    
    test_pf.calc1RDMfromCIS(_davidson_vec_g)
    davidson_rdm = np.copy(test_pf.D1_spatial)


    full_vec_g = wavefunctions[:, 0]
    test_pf.calc1RDMfromCIS(full_vec_g)
    full_rdm = np.copy(test_pf.D1_spatial)


    assert np.allclose(davidson_rdm, full_rdm)




