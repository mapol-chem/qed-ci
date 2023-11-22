import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_cqed_rhf import cqed_rhf
import numpy as np
import pytest
import sys

np.set_printoptions(threshold=sys.maxsize)

def test_lih_dipole_matrix_elements_no_cavity():

    mol_str = """
    Li
    H 1 1.4
    symmetry c1
    """

    options_dict = {
            "basis": "6-31g",
            "scf_type": "pk",
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
    }

    cavity_dict = {
            'omega_value' : 0.,
            'lambda_vector' : np.array([0, 0, 0.0]),
            'ci_level' : 'fci',
            'davidson_roots' : 5,
            'number_of_photons' : 0,
            'photon_number_basis' : True,
            'canonical_mos' : True,
            'coherent_state_basis' : False
    }

    test_pf = PFHamiltonianGenerator(
            mol_str,
            options_dict,
            cavity_dict
    )

    #### MAPPINT BETWEEN OUR ROOTS AND PSI4s ROOTS for LiH FCI/6-31G at r = 1.4
    #0       0.000001288442      -7.988942118138  -> psi4 root 0
    #1       0.000001822213      -7.876497218749  -> triplet
    #2       0.000003108562      -7.860341938452  -> psi4 root 1
    #3       0.000001848376      -7.834899121669  -> triplet
    #4       0.000006119714      -7.834899121661  -> triplet 
    #5       0.000002800391      -7.821446193352  -> psi4 root 2
    #6       0.000002800391      -7.821446193352  -> psi4 root 3 
    #7       0.000005218866      -7.766938322161  -> triplet
    #8       0.000000888709      -7.728582918194  -> psi4 root 4 

    # nuclear dipole moment from psi4 
    _expected_mu_nuc = np.array([0.0, 0.0, 1.3164164])
   
    # electronic dipole moment of root 0 from psi4
    _expected_mu_el_00  = np.array([-0.000000000000,       0.000000000000,      -3.3766326])

    # electronic dipole moment of root 1 from psi4 (equivalent to our root 2)
    _expected_mu_el_22  = np.array([-0.000000000000,       0.000000000000,      0.6518255])

    # electronic dipole moment of root 2 from psi4 (equivalent to our root 5)
    _expected_mu_el_55  = np.array([-0.000000000000,       0.000000000000,      -1.3064145])

    # electronic tdm 0->1 from psi4 (equivalent to our 0->2)
    _expected_mu_el_02  = np.array([-0.000000000000,       0.000000000000,      -0.9416200])

    # electronic tdm 0->2 from psi4 (equivalent to our 0->5)
    _expected_mu_el_05  = np.array([1.4288007,       0.5863472,      0.])

    # add nuclear contribution to get total dipole moments
    _expected_mu_00 = _expected_mu_el_00 + _expected_mu_nuc
    _expected_mu_22 = _expected_mu_el_22 + _expected_mu_nuc
    _expected_mu_55 = _expected_mu_el_55 + _expected_mu_nuc

    # test permanent dipole moments of different states - includes electronic and nuclear
    assert np.allclose(test_pf.dipole_array[0,0,:], _expected_mu_00)
    assert np.allclose(test_pf.dipole_array[2,2,:], _expected_mu_22)
    assert np.allclose(test_pf.dipole_array[5,5,:], _expected_mu_55)
    # test the electronic dipole moments of different states
    assert np.allclose(test_pf.electronic_dipole_array[0,0,:], _expected_mu_el_00)
    assert np.allclose(test_pf.electronic_dipole_array[2,2,:], _expected_mu_el_22)
    assert np.allclose(test_pf.electronic_dipole_array[5,5,:], _expected_mu_el_55)
    # test the transition dipole moments from ground-state, by definition only includes electronic contribution
    assert np.allclose(test_pf.electronic_dipole_array[0,2,:], _expected_mu_el_02)
    assert np.allclose(test_pf.electronic_dipole_array[0,5,:], _expected_mu_el_05)
    # test against dipole array to make sure we didn't add nuclear term to off-diagonal parts (tdms)
    assert np.allclose(test_pf.dipole_array[0,2,:], _expected_mu_el_02)
    assert np.allclose(test_pf.dipole_array[0,5,:], _expected_mu_el_05)

    



def test_lih_dipole_matrix_elements_with_cavity():

    mol_str = """
    Li
    H 1 1.4
    symmetry c1
    """

    options_dict = {
            "basis": "6-31g",
            "scf_type": "pk",
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
    }
    
    cavity_dict = {
        'omega_value' : 0.12086,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'davidson_roots' : 5,
        'number_of_photons' : 10,
        'photon_number_basis' : True,
        'canonical_mos' : True,
        'coherent_state_basis' : False
        
    }


    test_pf = PFHamiltonianGenerator(
            mol_str,
            options_dict,
            cavity_dict
    )


    _expected_mu_11 = np.array([-0.000000000000,       0.000000000000,      -3.378315615821])
    _expected_mu_12 = np.array([0.000000000000,       0.000000000000,       -0.561039417045])
    _expected_mu_13 = np.array([-0.000000000000,      -0.000000000000,       0.000000000000])
    _expected_mu_14 = np.array([0.000000000000,      0.000000000000,       0.700426875294])
    _expected_mu_22 = np.array([0.000000000000,       0.000000000000,      -2.133477501518])

    assert np.allclose(test_pf.dipole_array[0,0,:], _expected_mu_11)
    assert np.allclose(test_pf.dipole_array[0,1,:], _expected_mu_12)
    assert np.allclose(test_pf.dipole_array[0,2,:], _expected_mu_13)
    assert np.allclose(test_pf.dipole_array[0,3,:], _expected_mu_14)
    assert np.allclose(test_pf.dipole_array[1,1,:], _expected_mu_22)



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

def test_mghp_qed_cis_tdm_no_cavity():
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

    #energy from psi4numpy
    expected_mu_04 = np.array([-6.93218490e-16, -1.77990759e-15,  2.33258251e+00])
    

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    pass



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

    print(actual_g - expected_mghp_g_e)
    print(actual_lp - expected_mghp_lp_e)
    print(actual_up - expected_mghp_up_e)
    pass

def test_lih_direct_qed_fci_with_cavity():
    """ Test LiH using direct FCI with cavity in coherent state basis compared to 
        full diagonalization 
    """

    mol_str = """
    Li
    H 1 1.5
    symmetry c1
    """

    options_dict = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.12086,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'davidson_roots' : 10,
        'number_of_photons' : 1
    }
    
    # this is with Np = 6 for full diagonalization
    # _expected_eigs = np.array([[-7.878802702,  -7.7614561393, -7.75645532,   -7.7361601158,
    #                            -7.7083528272, -7.7083528272, -7.6868798814, -7.6868798814,
    #                            -7.6427875594, -7.6372558418]])

    # Np = 1 results
    _expected_eigs = np.array([-7.878792424,  -7.7601800101, -7.7547250121])

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    assert np.allclose(test_pf.CIeigs[:2], _expected_eigs[:2])

def test_lih_number_basis_direct_qed_fci_with_cavity():
    """ Test LiH using direct FCI with cavity in photon number basis compared to 
        full diagonalization 
            "model": {
        "method": "qed-fci",
        "orbital_basis": "sto-3g",
        "photon_basis": "number_basis",
        "number_of_photon_states": 10,
        "lambda": [
            0.0,
            0.0,
            0.05
        ],
        "omega": 0.12086
    """

    mol_str = """
    Li
    H 1 1.4
    symmetry c1
    """

    options_dict = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.12086,
        'lambda_vector' : np.array([0, 0, 0.05]),
        'ci_level' : 'fci',
        'davidson_roots' : 10,
        'number_of_photons' : 10,
        'photon_number_basis' : True,
        'canonical_mos' : True,
        'coherent_state_basis' : False
    }
    
    # this is with Np = 10 for full diagonalization in photon number basis
    #  "return_result": [
    #    -7.8749559054562726,
    #    -7.756654252046298,
    #    -7.747674819146274,
    #    -7.729087280855876,
    #    -7.701685049223775,
    #    -7.7016850492237685,
    #    -7.678038607940438
    #],

    # Np = 10 results
    _expected_eigs = np.array([-7.8749559054562726,
        -7.756654252046298,
        -7.747674819146274,
        -7.729087280855876,
        -7.701685049223775,
        -7.7016850492237685,
        -7.678038607940438])

    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    assert np.allclose(test_pf.CIeigs[:4], _expected_eigs[:4])

def test_lih_direct_qed_cas_with_cavity():
    """ Test LiH using direct CASCI with cavity compared to 
        full diagonalization 
    """

    mol_str = """
    Li
    H 1 1.5
    symmetry c1
    """

    options_dict = {
        "basis": "6-311G**",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    cavity_dict = {
        'omega_value' : 0.12086,
        'lambda_vector' : np.array([0, 0, 0.01]),
        'ci_level' : 'cas',
        'nact_orbs' : 6, 
        'nact_els' : 4,
        'davidson_roots' : 10,
        'number_of_photons' : 1
    }
    
    _expected_eigs = np.array([-7.9848177366, -7.8640432702, -7.8443027321, -7.8351500625, -7.7967681454,
            -7.7967681454, -7.790848489,  -7.790848489,  -7.7563223937, -7.7437841386])


    test_pf = PFHamiltonianGenerator(
        mol_str,
        options_dict,
        cavity_dict
    )

    assert np.allclose(test_pf.CIeigs[:10], _expected_eigs)

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




