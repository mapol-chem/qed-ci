"""
Helper function for CQED_RHF
References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 
    JJF Note: This implementation utilizes only electronic dipole contributions 
    and ignore superflous nuclear dipole terms!
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


def koch_cqed_rhf(lambda_vector, molecule_string, psi4_options_dict):
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

    # transform to MO basis
    d_MO = np.dot(C.T, l_dot_mu_el.dot(C))
    d_MO_occ = d_MO[:,:ndocc]
    d_MO_vir = d_MO[:,ndocc:]
    d_MOV_ov = d_MO[ndocc:, :ndocc]

    # ordinary H_core
    H_0 = T + V

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

        # Eq. (30) in Koch and co, 10.1103/PhysRevX.10.041043 
        M = np.einsum("pa, aq->pq", d_MO_vir, d_MO_vir.T)
        N = np.einsum("pi, iq->pq", d_MO_occ, d_MO_occ.T)
       
        # Get inverse arrays
        #d_MO = np.dot(C.T, l_dot_mu_el.dot(C))
        ICT = np.linalg.inv(C.T)
        IC = np.linalg.inv(C)
        # back transform 
        M_ao = np.dot(ICT, M.dot(IC))
        N_ao = np.dot(ICT, N.dot(IC))


        # Eq (31) in Koch and co, 10.1103/PhysRevX.10.041043
        E_DSE = np.einsum("ai->", d_MOV_ov)
        E_DSE *= E_DSE
        

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        # plus DSE terms from Eq. (30) in Koch and co
        F = H_0 + 2 * J - K + M_ao - N_ao

        # build canonical Fock operator for computing the SCF energy
        Fcan = H_0 + 2 * J - K

        diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
        diis_e = A.dot(diis_e).dot(A)
        dRMS = np.mean(diis_e**2) ** 0.5


        # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
        # Pauli-Fierz terms Eq. 13 of [McTague:2021:ChemRxiv]
        SCF_E = np.einsum("pq,pq->", Fcan + H_0, D) + Enuc + E_DSE

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

        # transform to MO basis
        d_MO = np.dot(C.T, l_dot_mu_el.dot(C))
        d_MO_occ = d_MO[:,:ndocc]
        d_MO_vir = d_MO[:,ndocc:]
        d_MOV_ov = d_MO[ndocc:, :ndocc]

        if SCF_ITER == maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
    print("QED-RHF   energy: %.8f hartree" % SCF_E)
    print("Psi4  SCF energy: %.8f hartree" % psi4_rhf_energy)



    cqed_rhf_dict = {
        "RHF ENERGY": psi4_rhf_energy,
        "CQED-RHF ENERGY": SCF_E,
        "CQED-RHF C": C,
        "CQED-RHF DENSITY MATRIX": D,
        "CQED-RHF EPS": e,
        "PSI4 WFN": wfn,
        "NUCLEAR REPULSION ENERGY": Enuc,
        "PF 1-E DIPOLE MATRIX MO" : d_MO,
        "DIPOLE SELF ENERGY" : E_DSE,
        "1-E KINETIC MATRIX AO" : T,
        "1-E POTENTIAL MATRIX AO" : V
    }
    return cqed_rhf_dict
