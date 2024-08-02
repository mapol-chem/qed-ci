"""
Helper function for CQED_RHF
References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 
    Pulay's DIIS has been implemented in this version.
"""

__authors__ = ["Nam Vu", "Jon McTague", "Jonathan Foley"]
__credits__ = ["Nam Vu", "Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
from curses import def_shell_mode
import psi4
import numpy as np
import time


def b_coefficient(error_vectors):
    b_mat = np.zeros((len(error_vectors) + 1, len(error_vectors) + 1))
    b_mat[-1, :] = -1
    b_mat[:, -1] = -1
    b_mat[-1, -1] = 0
    rhs = np.zeros((len(error_vectors) + 1, 1))
    rhs[-1, -1] = -1
    for i in range(len(error_vectors)):
        for j in range(i + 1):
            b_mat[i, j] = np.dot(error_vectors[i].transpose(), error_vectors[j])
            b_mat[j, i] = b_mat[i, j]
    *diis_coeff, _ = np.linalg.solve(b_mat, rhs)
    return diis_coeff


class Subspace(list):
    def append(self, item):
        list.append(self, item)
        if len(self) > dimSubspace:
            del self[0]


def cqed_rhf(lambda_vector, molecule_string, psi4_options_dict, canonical_basis=False):
    """Computes the QED-RHF energy and density
    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]
    molecule_string : string
        specifies the molecular geometry
    options_dict : dictionary
        specifies the psi4 options to be used in running the canonical RHF
    Returns
    -------
    cqed_rhf_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:ChemRxiv]
            'CQED-RHF C' -> orbitals resulting from CQED-RHF calculation
            'CQED-RHF DENSITY MATRIX' -> density matrix resulting from CQED-RHF calculation
            'CQED-RHF EPS'  -> orbital energies from CQED-RHF calculation
            'PSI4 WFN' -> wavefunction object from psi4 canonical RHF calcluation
            'CQED-RHF DIPOLE MOMENT' -> total dipole moment from CQED-RHF calculation (1x3 numpy array)
            'NUCLEAR DIPOLE MOMENT' -> nuclear dipole moment (1x3 numpy array)
            'NUCLEAR REPULSION ENERGY' -> Total nuclear repulsion energy
    Example
    -------
    >>> cqed_rhf_dictionary = cqed_rhf([0., 0., 1e-2], '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)
    """
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    psi4_rhf_energy, wfn = psi4.energy("scf", return_wfn=True)

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Grab data from wavfunction
    # number of doubly occupied orbitals
    ndocc = wfn.nalpha()

    # grab all transformation vectors and store to a numpy array
    C = np.asarray(wfn.Ca())

    # use canonical RHF orbitals for guess CQED-RHF orbitals
    Cocc = C[:, :ndocc]

    # form guess density
    D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

    # Integrals required for CQED-RHF
    # Ordinary integrals first
    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())
    I = np.asarray(mints.ao_eri())

    # Extra terms for Pauli-Fierz Hamiltonian
    # electronic dipole integrals in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # \lambda \cdot \mu_el (see within the sum of line 3 of Eq. (9) in [McTague:2021:ChemRxiv])
    d_el_ao = lambda_vector[0] * mu_ao_x
    d_el_ao += lambda_vector[1] * mu_ao_y
    d_el_ao += lambda_vector[2] * mu_ao_z

    # transform to the RHF MO basis
    d_el_mo = np.dot(C.T, d_el_ao).dot(C)

    # compute electronic dipole expectation value with
    # canonincal RHF density
    mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
    mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
    mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

    # get electronic dipole expectation value
    mu_exp_el = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    # get nuclear dipole moment
    mu_nuc = np.array(
        [mol.nuclear_dipole()[0], mol.nuclear_dipole()[1], mol.nuclear_dipole()[2]]
    )
    rhf_dipole_moment = mu_exp_el + mu_nuc
    # We need to carry around the electric field dotted into the nuclear dipole moment

    # \lambda_vecto \cdot < \mu > where <\mu> contains ONLY electronic contributions
    d_exp_el = np.dot(lambda_vector, mu_exp_el)

    # \lambda \cdot \mu_nuc
    d_nuc = np.dot(lambda_vector, mu_nuc)

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_ao = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_ao -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_ao -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_ao -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_ao -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_ao -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    # 1-e dipole terms scaled <\mu>_e for coherent-state basis
    d1_coherent_state_ao = -1 * d_exp_el * d_el_ao

    # 1-e dipole term scaled by \mu_nuc for photon number basis
    d1_number_state_ao = d_nuc * d_el_ao

    # 1/2 <d_e>^2 for coherent-state basis
    d_c_coherent_state = 0.5 * d_exp_el**2

    # 1/2 d_N^2 for photon number basis
    d_c_number_state = 0.5 * d_nuc**2

    # ordinary H_core
    H_0 = T + V

    # Add Pauli-Fierz terms to H_core
    # Eq. (11) in [McTague:2021:ChemRxiv]
    H = H_0 + Q_ao

    # Overlap for DIIS
    S = mints.ao_overlap()
    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.0e-16)
    A = np.asarray(A)

    print("\nStart SCF iterations:\n")
    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    E_1el_crhf = np.einsum("pq,pq->", H_0 + H_0, D)
    E_1el = np.einsum("pq,pq->", H + H, D)
    print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
    print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
    print("Nuclear repulsion energy          = %4.16f" % Enuc)
    print("1/2 <d_e>^2                       = %4.16f" % d_c_coherent_state)
    print("1/2 d_N^2                         = %4.16f" % d_c_number_state)

    # start storing constant and ao quantities to dictionary
    cqed_rhf_dict = {
        "PSI4 WFN": wfn,
        "RHF ENERGY": psi4_rhf_energy,
        "RHF C": C,
        "RHF DENSITY MATRIX": D,
        "RHF DIPOLE MOMENT": rhf_dipole_moment,
        "NUCLEAR DIPOLE MOMENT": mu_nuc,
        "NUCLEAR REPULSION ENERGY": Enuc,
        "DIPOLE AO X": mu_ao_x,
        "DIPOLE AO Y": mu_ao_y,
        "DIPOLE AO Z": mu_ao_z,
        "QUADRUPOLE AO XX": Q_ao_xx,
        "QUADRUPOLE AO YY": Q_ao_yy,
        "QUADRUPOLE AO ZZ": Q_ao_zz,
        # the cross-terms are not scaled by 2 here
        "QUADRUPOLE AO XY": Q_ao_xy,
        "QUADRUPOLE AO XZ": Q_ao_xz,
        "QUADRUPOLE AO YZ": Q_ao_zz,
        "1-E KINETIC MATRIX AO": T,
        "1-E POTENTIAL MATRIX AO": V,
        "1-E DIPOLE MATRIX AO": d_el_ao,
        "1-E DIPOLE MATRIX MO": d_el_mo,
        "1-E QUADRUPOLE MATRIX AO": Q_ao,
        "NUMBER STATE NUCLEAR DIPOLE TERM": d_nuc,
        "NUMBER STATE NUCLEAR DIPOLE ENERGY": d_c_number_state,
        "NUMBER STATE 1-E SCALED DIPOLE MATRIX AO": d1_number_state_ao,
        "CQED-RHF ENERGY": None,
        "CQED-RHF ONE-ENERGY": None,
        "CQED-RHF C": None,
        "CQED-RHF FOCK MATRIX": None,
        "CQED-RHF DENSITY MATRIX": None,
        "CQED-RHF EPS": None,
        "CQED-RHF ELECTRONIC DIPOLE MOMENT": None,
        "CQED-RHF DIPOLE MOMENT": None,
        "COHERENT STATE 1-E SCALED DIPOLE MATRIX AO": None,
        "COHERENT STATE EXPECTATION VALUE OF d": None,
        "COHERENT STATE DIPOLE ENERGY": None,
    }

    if canonical_basis:
        return cqed_rhf_dict

    else:
        # Set convergence criteria from psi4_options_dict
        if "e_convergence" in psi4_options_dict:
            E_conv = psi4_options_dict["e_convergence"]
        else:
            E_conv = 1.0e-7
        if "d_convergence" in psi4_options_dict:
            D_conv = psi4_options_dict["d_convergence"]
        else:
            D_conv = 1.0e-5

        t = time.time()
        global dimSubspace
        dimSubspace = 8
        error_list = Subspace()
        fock_list = Subspace()
        # maxiter
        maxiter = 500
        for SCF_ITER in range(1, maxiter + 1):
            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
            J = np.einsum("pqrs,rs->pq", I, D)
            K = np.einsum("prqs,rs->pq", I, D)

            # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
            # M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
            N = np.einsum("pr,qs,rs->pq", d_el_ao, d_el_ao, D)

            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
            # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
            F = H + 2 * J - K - N
            # save current Fock matrix into memory
            fock_list.append(F)
            diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum(
                "ij,jk,kl->il", S, D, F
            )
            # turn diis_e into column vector and then save to memory
            error_vector = diis_e.reshape(diis_e.shape[0] * diis_e.shape[0], 1)
            error_list.append(error_vector)

            diis_e = A.dot(diis_e).dot(A)
            dRMS = np.mean(diis_e**2) ** 0.5

            # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
            # Pauli-Fierz terms Eq. 13 of [McTague:2021:ChemRxiv]
            SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc

            print(
                "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
                % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS)
            )
            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            if SCF_ITER >= 2:
                diis_coeff = b_coefficient(error_list)
                F = np.zeros_like(D)
                for i in range(len(fock_list)):
                    F += diis_coeff[i] * fock_list[i]

            # Diagonalize Fock matrix: [Szabo:1996] pp. 145
            Fp = A.dot(F).dot(A)  # Eqn. 3.177
            e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178

            # if optional flag to use canonical basis is set to True, break before updating
            # any scf quantities

            C = A.dot(C2)  # Back transform, Eqn. 3.174
            Cocc = C[:, :ndocc]
            D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

            # update electronic dipole expectation value
            mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
            mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
            mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

            # get electronic dipole expectation value
            mu_exp_el = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

            # dot field vector into <\mu>_e
            d_exp_el = np.dot(lambda_vector, mu_exp_el)

            # Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
            d_c_coherent_state = 0.5 * d_exp_el**2

            d1_coherent_state_ao = -1 * d_exp_el * d_el_ao

            # update Core Hamiltonian
            H = H_0 + Q_ao

            if SCF_ITER == maxiter:
                psi4.core.clean()
                raise Exception("Maximum number of SCF cycles exceeded.")

        print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
        print("QED-RHF   energy: %.8f hartree" % SCF_E)
        print("Psi4  SCF energy: %.8f hartree" % psi4_rhf_energy)

        # compute various energetic contributions
        SCF_1E = np.einsum("pq,pq->", 2 * H_0, D)
        SCF_2E_J = np.einsum("pq,pq->", 2 * J, D)
        SCF_2E_K = np.einsum("pq,pq->", -1 * K, D)
        PF_1E_Q = np.einsum("pq,pq->", 2 * Q_ao, D)
        PF_2E_N = np.einsum("pq,pq->", -1 * N, D)

        # sum these together and see if equal to SCF_E - Enuc - d_c
        PF_E_el = SCF_1E + SCF_2E_J + SCF_2E_K + PF_1E_Q + PF_2E_N
        # does this agree with the final SCF energy when you subtract off nuclear contribut
        assert np.isclose(SCF_E - Enuc, PF_E_el, 1e-9)

        # transform \lambda \cdot \mu to CMO basis
        d_el_mo = np.dot(C.T, d_el_ao).dot(C)

        # update the entries of the dictionary now that the cqed-rhf iterations have converged
        cqed_rhf_dict["CQED-RHF ENERGY"] = SCF_E
        cqed_rhf_dict["CQED-RHF ONE-ENERGY"] = SCF_1E
        cqed_rhf_dict["CQED-RHF C"] = C
        cqed_rhf_dict["CQED-RHF FOCK MATRIX"] = F
        cqed_rhf_dict["CQED-RHF DENSITY MATRIX"] = D
        cqed_rhf_dict["CQED-RHF EPS"] = e
        cqed_rhf_dict["CQED-RHF ELECTRONIC DIPOLE MOMENT"] =  mu_exp_el
        cqed_rhf_dict["CQED-RHF DIPOLE MOMENT"] = mu_exp_el + mu_nuc
        cqed_rhf_dict["COHERENT STATE 1-E SCALED DIPOLE MATRIX AO"] = d1_coherent_state_ao
        cqed_rhf_dict["COHERENT STATE EXPECTATION VALUE OF d"] = d_exp_el
        cqed_rhf_dict["COHERENT STATE DIPOLE ENERGY"] = d_c_coherent_state
        cqed_rhf_dict["1-E DIPOLE MATRIX MO"] = d_el_mo
        return cqed_rhf_dict
