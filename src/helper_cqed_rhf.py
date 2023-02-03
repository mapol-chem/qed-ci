"""
Helper function for CQED_RHF
References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 
    JJF Note: This implementation utilizes only electronic dipole contributions 
    and ignore superflous nuclear dipole terms!

TO-DO: Implement the level-shifted version that Daniel presented where the Fock matrix has the form
\begin{multline}
    F_{QED-HF} = F_{HF} + \frac{1}{2} \lambda \cdot q \cdot \lambda 

    -   <\mu_{nuc}>/N_e \cdot \lambda}  d

    + 1/2 (<\mu_{nuc}>/N_e \cdot \lambda )^2 S

    - d P d

    + (<\mu_{nuc}>/N_e \cdot \lambda ) (d P S + S P d)

    - (<\mu_{nuc} / N_e \cdot \lambda)^2 SPS

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import time


def cqed_rhf(lambda_vector, molecule_string, psi4_options_dict):
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
    l_dot_mu_el = lambda_vector[0] * mu_ao_x
    l_dot_mu_el += lambda_vector[1] * mu_ao_y
    l_dot_mu_el += lambda_vector[2] * mu_ao_z

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
    # We need to carry around the electric field dotted into the nuclear dipole moment

    # \lambda_vecto \cdot < \mu > where <\mu> contains ONLY electronic contributions
    l_dot_mu_exp = np.dot(lambda_vector, mu_exp_el)

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_PF = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_PF -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_PF -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    # Pauli-Fierz 1-e dipole terms scaled <\mu>_e
    d_PF = -1 * l_dot_mu_exp * l_dot_mu_el

    # Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
    d_c = 0.5 * l_dot_mu_exp**2

    # ordinary H_core
    H_0 = T + V

    # Add Pauli-Fierz terms to H_core
    # Eq. (11) in [McTague:2021:ChemRxiv]
    H = H_0 + Q_PF 

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
    print("Dipole energy                     = %4.16f" % d_c)

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

    # maxiter
    maxiter = 500
    for SCF_ITER in range(1, maxiter + 1):

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        J = np.einsum("pqrs,rs->pq", I, D)
        K = np.einsum("prqs,rs->pq", I, D)

        # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
        #M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
        N = np.einsum("pr,qs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
        F = H + 2 * J - K - N

        diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
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

        # Diagonalize Fock matrix: [Szabo:1996] pp. 145
        Fp = A.dot(F).dot(A)  # Eqn. 3.177
        e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178
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
        l_dot_mu_exp = np.dot(lambda_vector, mu_exp_el)

        # Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
        d_c = 0.5 * l_dot_mu_exp**2

        # update Core Hamiltonian
        H = H_0 + Q_PF 

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
    PF_1E_Q = np.einsum("pq,pq->", 2 * Q_PF, D)
    PF_2E_N = np.einsum("pq,pq->", -1 * N, D)

    # sum these together and see if equal to SCF_E - Enuc - d_c
    PF_E_el = SCF_1E + SCF_2E_J + SCF_2E_K + PF_1E_Q + PF_2E_N
    # does this agree with the final SCF energy when you subtract off nuclear contribut
    assert np.isclose(SCF_E - Enuc, PF_E_el, 1e-9)

    cqed_rhf_dict = {
        "RHF ENERGY": psi4_rhf_energy,
        "CQED-RHF ENERGY": SCF_E,
        "CQED-RHF C": C,
        "CQED-RHF DENSITY MATRIX": D,
        "CQED-RHF EPS": e,
        "PSI4 WFN": wfn,
        "CQED-RHF DIPOLE MOMENT": mu_exp_el + mu_nuc,
        "NUCLEAR DIPOLE MOMENT": mu_nuc,
        "DIPOLE ENERGY": d_c,
        "NUCLEAR REPULSION ENERGY": Enuc,
    }
    return cqed_rhf_dict