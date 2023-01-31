"""
Helper function for "prism of QED-CISS methods" in the coherent state basis

References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], [McTague:2021:ChemRxiv], [Yang:2021:064107] 

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-01-15"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import scipy.linalg as la
import time
from helper_cqed_rhf import cqed_rhf


def cs_cqed_cis(lambda_vector, omega_val, molecule_string, psi4_options_dict):
    """Computes the QED-RHF energy and density

    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]

    omega_val : complex float
        the complex energy associated with the photon, see Eq. (3) in [McTague:2021:ChemRxiv]

    molecule_string : string
        specifies the molecular geometry

    psi4_options_dict : dictionary
        specifies the psi4 options to be used in running requisite psi4 calculations

    Returns
    -------
    cqed_ciss_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'RHF-PF ENERGY' -> result of RHF calculation with PF Hamiltonian
            'CISS-PF ENERGY' -> eigenvalues from the CISS-PF Hamiltonian
            'CIS-PF ENERGY' -> eigenvalues from the CIS-PF Hamiltonian
            'CISS-JC ENERGY' -> eigenvalues from the CISS-JC Hamiltonian
            'CIS-JC ENERGY' -> eigenvalues from the CIS-JC Hamiltonian

    Example
    -------
    >>> cqed_cis_dictionary = cs_cqed_cis([0., 0., 1e-2], 0.2-0.001j, '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)

    """

    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    # scf_e, wfn = psi4.energy('scf', return_wfn=True)

    # run cqed_rhf method
    cqed_rhf_dict = cqed_rhf(lambda_vector, molecule_string, psi4_options_dict)

    # grab necessary quantities from cqed_rhf_dict
    scf_e = cqed_rhf_dict["RHF ENERGY"]
    cqed_scf_e = cqed_rhf_dict["CQED-RHF ENERGY"]
    wfn = cqed_rhf_dict["PSI4 WFN"]
    C = cqed_rhf_dict["CQED-RHF C"]
    D = cqed_rhf_dict["CQED-RHF DENSITY MATRIX"]
    eps = cqed_rhf_dict["CQED-RHF EPS"]
    Fc = cqed_rhf_dict["CANONICAL FOCK MATRIX"]
    Fdse = cqed_rhf_dict["DSE FOCK MATRIX"]

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Grab data from wavfunction
    # number of doubly occupied orbitals
    ndocc = wfn.nalpha()

    # total number of orbitals
    nmo = wfn.nmo()

    # number of virtual orbitals
    nvirt = nmo - ndocc

    # need to update the Co and Cv core matrix objects so we can
    # utlize psi4s fast integral transformation!

    # collect rhf wfn object as dictionary
    wfn_dict = psi4.core.Wavefunction.to_file(wfn)

    # update wfn_dict with orbitals from CQED-RHF
    wfn_dict["matrix"]["Ca"] = C
    wfn_dict["matrix"]["Cb"] = C
    # update wfn object
    wfn = psi4.core.Wavefunction.from_file(wfn_dict)

    # occupied orbitals as psi4 objects but they correspond to CQED-RHF orbitals
    Co = wfn.Ca_subset("AO", "OCC")

    # virtual orbitals same way
    Cv = wfn.Ca_subset("AO", "VIR")

    # 2 electron integrals in CQED-RHF basis
    ovov = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))

    # build the (oo|vv) integrals:
    oovv = np.asarray(mints.mo_eri(Co, Co, Cv, Cv))

    # strip out occupied orbital energies, eps_o spans 0..ndocc-1
    eps_o = eps[:ndocc]

    # strip out virtual orbital energies, eps_v spans 0..nvirt-1
    eps_v = eps[ndocc:]

    # dipole arrays in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # electronic dipole expectation value with CQED-RHF density
    mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
    mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
    mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

    # get electronic dipole expectation value
    mu_exp_el = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    # \lambda \cdot < \mu > where < \mu > contains only
    # electronic terms 
    l_dot_mu_exp = np.dot(lambda_vector, mu_exp_el)

    # transform dipole array to CQED-RHF basis
    mu_cmo_x = np.dot(C.T, mu_ao_x).dot(C)
    mu_cmo_y = np.dot(C.T, mu_ao_y).dot(C)
    mu_cmo_z = np.dot(C.T, mu_ao_z).dot(C)

    # transform canonical Fock matrix to the CQED-RHF basis
    Fc_cmo = np.dot(C.T, Fc).dot(C)
    Fc_cmo_alt = np.einsum("pu, uv, vq->pq", C.T, Fc, C)
    assert np.allclose(Fc_cmo, Fc_cmo_alt)

    # transform DSE contribution to Fock matrix to the CQED-RHF basis
    Fdse_cmo = np.dot(C.T, Fdse).dot(C)
    Fdse_cmo_alt = np.einsum("pu, uv, vq->pq", C.T, Fdse, C)
    assert np.allclose(Fdse_cmo, Fdse_cmo_alt)

    # \lambda \cdot \mu_{el}
    # e.g. line 4 Eq. (18) in [McTague:2021:ChemRxiv]
    l_dot_mu_el = lambda_vector[0] * mu_cmo_x
    l_dot_mu_el += lambda_vector[1] * mu_cmo_y
    l_dot_mu_el += lambda_vector[2] * mu_cmo_z

    # compute Koch's definition of the Fock matrix 

    # Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
    d_c = 0.5 * l_dot_mu_exp**2

    # check to see if d_c what we have from CQED-RHF calculation
    assert np.isclose(d_c, cqed_rhf_dict["DIPOLE ENERGY"])

    # build g matrix and its adjoint
    g = np.zeros((1,ndocc * nvirt), dtype=complex)
    g_dag = np.zeros((ndocc * nvirt, 1), dtype=complex)
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            A = a + ndocc
            ia = i * nvirt + a 
            g[0,ia] = (
                -np.sqrt(omega_val) * l_dot_mu_el[i, A]
            )

    # Now compute the adjoint of g
    g_dag = np.conj(g).T

    # get the A + \Delta matrix and the G matrix, since both 
    # involive <S | H | S> terms where |S> represents singly-excited determinants
    A_matrix = np.zeros((ndocc * nvirt, ndocc * nvirt), dtype=complex)
    D_matrix = np.zeros((ndocc * nvirt, ndocc * nvirt), dtype=complex)
    G = np.zeros((ndocc * nvirt, ndocc * nvirt), dtype=complex)
    Omega = np.zeros((ndocc * nvirt, ndocc * nvirt), dtype=complex)
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            A = a + ndocc
            ia = i * nvirt + a
            
            for j in range(0, ndocc):
                for b in range(0, nvirt):
                    B = b + ndocc
                    jb = j * nvirt + b
                    
                    # ERI contribution to A 
                    A_matrix[ia, jb] += (2.0 * ovov[i, a, j, b] - oovv[i, j, a, b])

                    # Canonical Fock matrix contribution to A
                    A_matrix[ia, jb] += Fc_cmo[A, B] * (i == j)
                    A_matrix[ia, jb] -= Fc_cmo[i, j] * (a == b)
                    
                    # 2-electron dipole contribution to \Delta
                    D_matrix[ia, jb] += 2.0 * l_dot_mu_el[i, A] * l_dot_mu_el[j, B]
                    D_matrix[ia, jb] -= l_dot_mu_el[i, j] * l_dot_mu_el[A, B]

                    # DSE Fock matrix contribution to \Delta
                    D_matrix[ia, jb] += Fdse_cmo[A, B] * (i == j)
                    D_matrix[ia, jb] -= Fdse_cmo[i, j] * (a == b)
                    
                    # bilinear coupling contributions to G
                    # off-diagonal terms (plus occasional diagonal terms)
                    G[ia, jb] += np.sqrt(omega_val / 2) * l_dot_mu_el[i, j] * (a == b)
                    G[ia, jb] -= np.sqrt(omega_val / 2) * l_dot_mu_el[A, B] * (i == j)
                    
                    # diagonal contributions \Omega matrix
                    Omega[ia, jb] += omega_val * (a == b) * (i == j)
                        

    # define the offsets
    R0_offset = 0
    S0_offset = 1
    R1_offset = ndocc * nvirt + 1
    S1_offset = ndocc * nvirt + 2

    # CIS DSE Hamiltonian
    H_CIS_DSE = np.zeros((ndocc * nvirt, ndocc * nvirt), dtype=float)

    # CISS Hamiltonians
    H_CISS_PF = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    H_CISS_JC = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)

    # Just the DSE contribution to the CISS-PF Hamiltonian
    H_CISS_DSE = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2))
    # Just the bilinear coupling contribution to the CISS-PF Hamiltonian
    H_CISS_BLC = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2))

    # build the supermatrix
    # g coupling
    # PF
    H_CISS_PF[R0_offset:S0_offset, S1_offset:] = g
    H_CISS_PF[S0_offset:R1_offset, R1_offset:S1_offset] = g_dag
    H_CISS_PF[R1_offset:S1_offset, S0_offset:R1_offset] = g
    H_CISS_PF[S1_offset:,          R0_offset:S0_offset] = g_dag

    # add the g terms to the BLC matrix
    H_CISS_BLC += np.real(H_CISS_PF)

    # JC
    H_CISS_JC[R0_offset:S0_offset, S1_offset:] = g
    H_CISS_JC[S0_offset:R1_offset, R1_offset:S1_offset] = g_dag
    H_CISS_JC[R1_offset:S1_offset, S0_offset:R1_offset] = g
    H_CISS_JC[S1_offset:,          R0_offset:S0_offset] = g_dag

    # A + \Delta for CIS_DSE
    H_CIS_DSE = A_matrix + D_matrix

    # A + \Delta for PF
    H_CISS_PF[S0_offset:R1_offset, S0_offset:R1_offset] = A_matrix + D_matrix

    # \Delta for H_CISS_DSE
    H_CISS_DSE[S0_offset:R1_offset, S0_offset:R1_offset] = D_matrix

    # A for JC
    H_CISS_JC[S0_offset:R1_offset, S0_offset:R1_offset] = A_matrix

    # omega
    # PF
    H_CISS_PF[R1_offset, R1_offset] = omega_val
    # JC
    H_CISS_JC[R1_offset, R1_offset] = omega_val

    # A + \Delta + \Omega for PF
    H_CISS_PF[S1_offset:, S1_offset:] = A_matrix + D_matrix + Omega

    # \Delta for H_CISS_DSE
    H_CISS_DSE[S1_offset:, S1_offset:] = D_matrix

    # A + \Omega for JC
    H_CISS_JC[S1_offset:, S1_offset:] = A_matrix + Omega

    # G coupling
    # PF
    H_CISS_PF[S1_offset:,S0_offset:R1_offset] = G
    H_CISS_PF[S0_offset:R1_offset, S1_offset:] = G

    # H_CISS_BLC
    H_CISS_BLC[S1_offset:,S0_offset:R1_offset] = G
    H_CISS_BLC[S0_offset:R1_offset, S1_offset:] = G

    # JC
    H_CISS_JC[S1_offset:,S0_offset:R1_offset] = G
    H_CISS_JC[S0_offset:R1_offset, S1_offset:] = G

    # CIS Hamiltonians
    H_CIS_PF = np.zeros((ndocc * nvirt + 1, ndocc * nvirt + 1), dtype=complex)
    H_CIS_JC = np.zeros((ndocc * nvirt + 1, ndocc * nvirt + 1), dtype=complex)

    # define the CIS offsets
    CIS_S0_offset = 0
    CIS_R1_offset = ndocc * nvirt


    # build the supermatrix
    # g coupling
    # PF
    H_CIS_PF[CIS_R1_offset:, CIS_S0_offset:CIS_R1_offset] = g
    H_CIS_PF[CIS_S0_offset:CIS_R1_offset, CIS_R1_offset:] = g_dag
    # JC
    H_CIS_JC[CIS_R1_offset:, CIS_S0_offset:CIS_R1_offset] = g
    H_CIS_JC[CIS_S0_offset:CIS_R1_offset, CIS_R1_offset:] = g_dag

    # A + \Delta for PF
    H_CIS_PF[CIS_S0_offset:CIS_R1_offset, CIS_S0_offset:CIS_R1_offset] = A_matrix + D_matrix
    # A  for JF
    H_CIS_JC[CIS_S0_offset:CIS_R1_offset, CIS_S0_offset:CIS_R1_offset] = A_matrix

    # omega
    # PF
    H_CIS_PF[CIS_R1_offset, CIS_R1_offset] = omega_val
    # JC
    H_CIS_JC[CIS_R1_offset, CIS_R1_offset] = omega_val

    # diagonalize different versions of the QED-CISS matrix
    E_CIS_DSE, C_CIS_DSE = np.linalg.eigh(H_CIS_DSE)

    E_CISS_PF, C_CISS_PF = np.linalg.eigh(H_CISS_PF)
    E_CISS_JC, C_CISS_JC = np.linalg.eigh(H_CISS_JC)

    E_CIS_PF, C_CIS_PF = np.linalg.eigh(H_CIS_PF)
    E_CIS_JC, C_CIS_JC = np.linalg.eigh(H_CIS_JC)


    cqed_cis_dict = {
        "H CIS-DSE": H_CIS_DSE,
        "H CISS-DSE" : H_CISS_DSE,
        "H CISS-BLC" : H_CISS_BLC,
        "RHF ENERGY": scf_e,
        "CQED-RHF ENERGY": cqed_scf_e,
        "CISS-PF ENERGY": E_CISS_PF,
        "CISS-JC ENERGY": E_CISS_JC,
        "CIS-PF ENERGY": E_CIS_PF,
        "CIS-JC ENERGY": E_CIS_JC,
        "CIS-DSE ENERGY": E_CIS_DSE,
        "C VECTOR": C_CISS_PF, 
        "R1 OFFSET": R1_offset 
    }
    print(ndocc * nvirt)
    return cqed_cis_dict
