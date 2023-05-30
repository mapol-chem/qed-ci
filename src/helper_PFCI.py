"""
Helper Classes for Cavity Quantum Electrodynamics Configuration Interaction methods.
Adapted from a helper class for Configuration Interaction by the Psi4Numpy Developers, specifically
Tianyuan Zhang, Jeffrey B. Schriber, and Daniel G. A. Smith.

References:
- Equations from [Szabo:1996], [Foley:2022], [Koch:2020]
"""

__authors__ = "Nam Vu", "Jonathan J. Foley IV"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2023, The Psi4NumPy Developers, Foley Lab, Mapol Project"
__license__ = "GNU-GPL-3"
__date__ = "2023-01-21"

import psi4
import sys
from helper_cqed_rhf import cqed_rhf
from itertools import combinations
import time


def compute_excitation_level(ket, ndocc):
    level = 0
    homo = ndocc - 1
    for i in range(0, len(ket)):
        if ket[i] > homo:
            level += 1
    return level


def spin_idx_to_spat_idx_and_spin(P):
    """function to take the numeric label of a spin orbital
    and return the spatial index and the spin index separately.
    Starts counting from 0:

    Arguments
    ---------
    P : int
        spin orbital label

    Returns
    -------
    [p, spin] : numpy array of ints
        p is the spatial orbital index and spin is the spin index.
        spin = 1  -> alpha
        spin = -1 -> beta

    Example
    -------
    >>> spin_idx_to_spat_idx_and_spin(0)
    >>> [0, 1]
    >>> spin_idx_to_spat_idx_and_spin(3)
    >>> [1, -1]

    """
    spin = 1
    if P % 2 == 0:
        p = P / 2
        spin = 1
    else:
        p = (P - 1) / 2
        spin = -1
    return np.array([p, spin], dtype=int)


def map_spatial_to_spin(tei_spatial, I, J, K, L):
    """function to take two electron integrals in the spatial orbital basis
    in chemist notation along with 4 indices I, J, K, L and return
    the corresponding two electron integral in the spin orbital basis
    in phycisit notation, <IJ||KL>

    """
    # Phys to Chem: <IJ||KL> -> [IK|JL] - [IL|JK]
    i_s = spin_idx_to_spat_idx_and_spin(I)
    k_s = spin_idx_to_spat_idx_and_spin(K)
    j_s = spin_idx_to_spat_idx_and_spin(J)
    l_s = spin_idx_to_spat_idx_and_spin(L)

    # print(i_s[1])
    # (ik|jl)
    spat_ikjl = (
        tei_spatial[i_s[0], k_s[0], j_s[0], l_s[0]]
        * (i_s[1] == k_s[1])
        * (j_s[1] == l_s[1])
    )

    # (il|jk)
    spat_iljk = (
        tei_spatial[i_s[0], l_s[0], j_s[0], k_s[0]]
        * (i_s[1] == l_s[1])
        * (j_s[1] == k_s[1])
    )

    return spat_ikjl - spat_iljk


def map_spatial_dipole_to_spin(mu, I, J, K, L):
    """function to take the dipole matrix (a 1-electron matrix)
    and return the product of dipole matrix elements such that it matches
    the <IJ||KL> convention.

    """
    # Phys to Chem: <IJ||KL> -> [IK|JL] - [IL|JK]
    i_s = spin_idx_to_spat_idx_and_spin(I)
    k_s = spin_idx_to_spat_idx_and_spin(K)
    j_s = spin_idx_to_spat_idx_and_spin(J)
    l_s = spin_idx_to_spat_idx_and_spin(L)

    # print(i_s[1])
    # (ik|jl)
    spat_ikjl = (
        mu[i_s[0], k_s[0]]
        * mu[j_s[0], l_s[0]]
        * (i_s[1] == k_s[1])
        * (j_s[1] == l_s[1])
    )

    # (il|jk)
    spat_iljk = (
        mu[i_s[0], l_s[0]]
        * mu[j_s[0], k_s[0]]
        * (i_s[1] == l_s[1])
        * (j_s[1] == k_s[1])
    )

    return spat_ikjl - spat_iljk


class Determinant:
    """
    A class for a bit-Determinant.
    """

    def __init__(
        self, alphaObtBits=0, betaObtBits=0, alphaObtList=None, betaObtList=None
    ):
        """
        Constructor for the Determinant
        """

        if alphaObtBits == 0 and alphaObtList != None:
            alphaObtBits = Determinant.obtIndexList2ObtBits(alphaObtList)
        if betaObtBits == 0 and betaObtList != None:
            betaObtBits = Determinant.obtIndexList2ObtBits(betaObtList)
        self.alphaObtBits = alphaObtBits
        self.betaObtBits = betaObtBits

    def getNumOrbitals(self):
        """
        Return the number of orbitals (alpha, beta) in this determinant
        """

        return Determinant.countNumOrbitalsInBits(
            self.alphaObtBits
        ), Determinant.countNumOrbitalsInBits(self.betaObtBits)

    def getOrbitalIndexLists(self):
        """
        Return lists of orbital index
        """

        return Determinant.obtBits2ObtIndexList(
            self.alphaObtBits
        ), Determinant.obtBits2ObtIndexList(self.betaObtBits)

    def getOrbitalMixedIndexList(self):
        """
        Return lists of orbital in mixed spin index alternating alpha and beta
        """

        return Determinant.obtBits2ObtMixSpinIndexList(
            self.alphaObtBits, self.betaObtBits
        )

    @staticmethod
    def countNumOrbitalsInBits(bits):
        """
        Return the number of orbitals in this bits
        """

        count = 0
        while bits != 0:
            if bits & 1 == 1:
                count += 1
            bits >>= 1
        return count

    @staticmethod
    def countNumOrbitalsInBitsUpTo4(bits):
        """
        Return the number of orbitals in this bits
        """

        count = 0
        while bits != 0 and count < 4:
            if bits & 1 == 1:
                count += 1
            bits >>= 1
        return count

    @staticmethod
    def obtBits2ObtIndexList(bits):
        """
        Return the corresponding list of orbital numbers from orbital bits
        """

        i = 0
        obts = []
        while bits != 0:
            if bits & 1 == 1:
                obts.append(i)
            bits >>= 1
            i += 1
        return obts

    @staticmethod
    def mixIndexList(alphaList, betaList):
        """
        Mix the alpha and beta orbital index list to one mixed list
        """

        return [elem * 2 for elem in alphaList] + [elem * 2 + 1 for elem in betaList]

    @staticmethod
    def obtBits2ObtMixSpinIndexList(alphaBits, betaBits):
        """
        Return the corresponding list of orbital numbers of orbital bits
        """

        alphaList, betaList = Determinant.obtBits2ObtIndexList(
            alphaBits
        ), Determinant.obtBits2ObtIndexList(betaBits)
        return Determinant.mixIndexList(alphaList, betaList)

    @staticmethod
    def obtIndexList2ObtBits(obtList):
        """
        Return the corresponding orbital bits of list from orbital numbers
        """

        if len(obtList) == 0:
            return 0
        obtList = sorted(obtList, reverse=True)
        iPre = obtList[0]
        bits = 1
        for i in obtList:
            bits <<= iPre - i
            bits |= 1
            iPre = i
        bits <<= iPre
        return bits

    @staticmethod
    def getOrbitalPositions(bits, orbitalIndexList):
        """
        Return the position of orbital in determinant
        """

        count = 0
        index = 0
        positions = []
        for i in orbitalIndexList:
            while index < i:
                if bits & 1 == 1:
                    count += 1
                bits >>= 1
                index += 1
            positions.append(count)
            continue
        return positions

    def getOrbitalPositionLists(self, alphaIndexList, betaIndexList):
        """
        Return the positions of indexes in lists
        """

        return Determinant.getOrbitalPositions(
            self.alphaObtBits, alphaIndexList
        ), Determinant.getOrbitalPositions(self.betaObtBits, betaIndexList)

    def addAlphaOrbital(self, orbitalIndex):
        """
        Add an alpha orbital to the determinant
        """

        self.alphaObtBits |= 1 << orbitalIndex

    def addBetaOrbital(self, orbitalIndex):
        """
        Add an beta orbital to the determinant
        """

        self.betaObtBits |= 1 << orbitalIndex

    def removeAlphaOrbital(self, orbitalIndex):
        """
        Remove an alpha orbital from the determinant
        """

        self.alphaObtBits &= ~(1 << orbitalIndex)

    def removeBetaOrbital(self, orbitalIndex):
        """
        Remove an beta orbital from the determinant
        """

        self.betaObtBits &= ~(1 << orbitalIndex)

    def numberOfCommonOrbitals(self, another):
        """
        Return the number of common orbitals between this determinant and another determinant
        """

        return Determinant.countNumOrbitalsInBits(
            self.alphaObtBits & another.alphaObtBits
        ), Determinant.countNumOrbitalsInBits(self.betaObtBits & another.betaObtBits)

    def getCommonOrbitalsInLists(self, another):
        """Return common orbitals between this determinant and another determinant in lists"""
        return Determinant.obtBits2ObtIndexList(
            self.alphaObtBits & another.alphaObtBits
        ), Determinant.obtBits2ObtIndexList(self.betaObtBits & another.betaObtBits)

    def getCommonOrbitalsInMixedSpinIndexList(self, another):
        alphaList, betaList = self.getCommonOrbitalsInLists(another)
        return Determinant.mixIndexList(alphaList, betaList)

    def numberOfDiffOrbitals(self, another):
        """
        Return the number of different alpha and beta orbitals between this determinant and another determinant
        """

        diffAlpha, diffBeta = Determinant.countNumOrbitalsInBits(
            self.alphaObtBits ^ another.alphaObtBits
        ), Determinant.countNumOrbitalsInBits(self.betaObtBits ^ another.betaObtBits)
        return diffAlpha / 2, diffBeta / 2

    def numberOfTotalDiffOrbitals(self, another):
        """
        Return the number of different orbitals between this determinant and another determinant
        """

        diffAlpha, diffBeta = self.numberOfDiffOrbitals(another)
        return diffAlpha + diffBeta

    def diff2OrLessOrbitals(self, another):
        """
        Return true if two determinants differ 2 or less orbitals
        """

        diffAlpha, diffBeta = Determinant.countNumOrbitalsInBitsUpTo4(
            self.alphaObtBits ^ another.alphaObtBits
        ), Determinant.countNumOrbitalsInBitsUpTo4(
            self.betaObtBits ^ another.betaObtBits
        )
        return (diffAlpha + diffBeta) <= 4

    @staticmethod
    def uniqueOrbitalsInBits(bits1, bits2):
        """
        Return the unique bits in two different bits
        """

        common = bits1 & bits2
        return bits1 ^ common, bits2 ^ common

    @staticmethod
    def uniqueOrbitalsInLists(bits1, bits2):
        """
        Return the unique bits in two different bits
        """

        bits1, bits2 = Determinant.uniqueOrbitalsInBits(bits1, bits2)
        return Determinant.obtBits2ObtIndexList(
            bits1
        ), Determinant.obtBits2ObtIndexList(bits2)

    def getUniqueOrbitalsInLists(self, another):
        """
        Return the unique orbital lists in two different determinants
        """

        alphaList1, alphaList2 = Determinant.uniqueOrbitalsInLists(
            self.alphaObtBits, another.alphaObtBits
        )
        betaList1, betaList2 = Determinant.uniqueOrbitalsInLists(
            self.betaObtBits, another.betaObtBits
        )
        return (alphaList1, betaList1), (alphaList2, betaList2)

    def getUnoccupiedOrbitalsInLists(self, nmo):
        """
        Return the unoccupied orbitals in the determinants
        """

        alphaBits = ~self.alphaObtBits
        betaBits = ~self.betaObtBits
        alphaObts = []
        betaObts = []
        for i in range(nmo):
            if alphaBits & 1 == 1:
                alphaObts.append(i)
            alphaBits >>= 1
            if betaBits & 1 == 1:
                betaObts.append(i)
            betaBits >>= 1
        return alphaObts, betaObts

    def getSignToMoveOrbitalsToTheFront(self, alphaIndexList, betaIndexList):
        """
        Return the final sign if move listed orbitals to the front
        """

        sign = 1
        alphaPositions, betaPositions = self.getOrbitalPositionLists(
            alphaIndexList, betaIndexList
        )
        for i in range(len(alphaPositions)):
            if (alphaPositions[i] - i) % 2 == 1:
                sign = -sign
        for i in range(len(betaPositions)):
            if (betaPositions[i] - i) % 2 == 1:
                sign = -sign
        return sign

    def getUniqueOrbitalsInListsPlusSign(self, another):
        """
        Return the unique orbital lists in two different determinants and the sign of the maximum coincidence determinants
        """

        alphaList1, alphaList2 = Determinant.uniqueOrbitalsInLists(
            self.alphaObtBits, another.alphaObtBits
        )
        betaList1, betaList2 = Determinant.uniqueOrbitalsInLists(
            self.betaObtBits, another.betaObtBits
        )
        sign1, sign2 = self.getSignToMoveOrbitalsToTheFront(
            alphaList1, betaList1
        ), another.getSignToMoveOrbitalsToTheFront(alphaList2, betaList2)
        return (alphaList1, betaList1), (alphaList2, betaList2), sign1 * sign2

    def getUniqueOrbitalsInMixIndexListsPlusSign(self, another):
        """
        Return the unique orbital lists in two different determinants and the sign of the maximum coincidence determinants
        """

        (
            (alphaList1, betaList1),
            (alphaList2, betaList2),
            sign,
        ) = self.getUniqueOrbitalsInListsPlusSign(another)
        return (
            Determinant.mixIndexList(alphaList1, betaList1),
            Determinant.mixIndexList(alphaList2, betaList2),
            sign,
        )

    def toIntTuple(self):
        """
        Return a int tuple
        """

        return (self.alphaObtBits, self.betaObtBits)

    @staticmethod
    def createFromIntTuple(intTuple):
        return Determinant(alphaObtBits=intTuple[0], betaObtBits=intTuple[1])

    def generateSingleExcitationsOfDet(self, nmo):
        """
        Generate all the single excitations of determinant in a list
        """

        alphaO, betaO = self.getOrbitalIndexLists()
        alphaU, betaU = self.getUnoccupiedOrbitalsInLists(nmo)
        dets = []

        for i in alphaO:
            for j in alphaU:
                det = self.copy()
                det.removeAlphaOrbital(i)
                det.addAlphaOrbital(j)
                dets.append(det)

        for k in betaO:
            for l in betaU:
                det = self.copy()
                det.removeBetaOrbital(k)
                det.addBetaOrbital(l)
                dets.append(det)

        return dets

    def generateDoubleExcitationsOfDet(self, nmo):
        """
        Generate all the double excitations of determinant in a list
        """

        alphaO, betaO = self.getOrbitalIndexLists()
        alphaU, betaU = self.getUnoccupiedOrbitalsInLists(nmo)
        dets = []

        for i in alphaO:
            for j in alphaU:
                for k in betaO:
                    for l in betaU:
                        det = self.copy()
                        det.removeAlphaOrbital(i)
                        det.addAlphaOrbital(j)
                        det.removeBetaOrbital(k)
                        det.addBetaOrbital(l)
                        dets.append(det)

        for i1, i2 in combinations(alphaO, 2):
            for j1, j2 in combinations(alphaU, 2):
                det = self.copy()
                det.removeAlphaOrbital(i1)
                det.addAlphaOrbital(j1)
                det.removeAlphaOrbital(i2)
                det.addAlphaOrbital(j2)
                dets.append(det)

        for k1, k2 in combinations(betaO, 2):
            for l1, l2 in combinations(betaU, 2):
                det = self.copy()
                det.removeBetaOrbital(k1)
                det.addBetaOrbital(l1)
                det.removeBetaOrbital(k2)
                det.addBetaOrbital(l2)
                dets.append(det)
        return dets

    def generateSingleAndDoubleExcitationsOfDet(self, nmo):
        """
        Generate all the single and double excitations of determinant in a list
        """

        return self.generateSingleExcitationsOfDet(
            nmo
        ) + self.generateDoubleExcitationsOfDet(nmo)

    def copy(self):
        """
        Return a deep copy of self
        """

        return Determinant(alphaObtBits=self.alphaObtBits, betaObtBits=self.betaObtBits)

    def __str__(self):
        """
        Print a representation of the Determinant
        """
        a, b = self.getOrbitalIndexLists()
        return "|" + str(a) + str(b) + ">"


import numpy as np


class PFHamiltonianGenerator:
    """
    class for Full CI matrix elements
    """

    def __init__(
        self,
        molecule_string,
        psi4_options_dict,
        cavity_options
    ):
        """
        Constructor for matrix elements of the PF Hamiltonian
        """
        # look at cavity options first
        cavity_options = {k.lower(): v for k, v in cavity_options.items()}
        self.parseCavityOptions(cavity_options)

        # generate orbital basis - if self.natural_orbitals == True then
        # self.C and self.Ca will be the QED-CIS natural orbitals
        # otherwise they will be the QED-RHF MOs
        psi4_wfn_o = self.generateOrbitalBasis(molecule_string, psi4_options_dict)

        # build arrays in orbital basis from last step
        self.buildArraysInOrbitalBasis(psi4_wfn_o)

        t_det_start = time.time()
        # build the determinant list
        if self.ci_level == "cis":
            self.generateCISDeterminants()
            H_dim = self.CISnumDets * 2 
        elif self.ci_level == "cas":
            self.generateCASCIDeterminants()
            H_dim = self.CASnumDets * 2 
        elif self.ci_level == "fci":
            self.generateFCIDeterminants()
            H_dim = self.FCInumDets * 2 

        t_det_end = time.time()
        print(F' Completed determinant list in {t_det_end - t_det_start} seconds ')


        # build Constant matrices
        self.buildConstantMatrices(self.ci_level)
        t_const_end = time.time()
        print(F' Completed constant offset matrix in {t_const_end - t_det_end} seconds')

        # Build Matrix
        self.generatePFHMatrix(self.ci_level)
        t_H_build = time.time()
        print(F' Completed Hamiltonian build in {t_H_build - t_const_end} seconds')
        
        if self.full_diagonalization:
            self.CIeigs, self.CIvecs = np.linalg.eigh(self.H_PF)
        
        else:
            indim = self.davidson_indim * self.davidson_roots
            maxdim = self.davidson_maxdim * self.davidson_roots
            if (indim > H_dim or maxdim > H_dim):
                print('subspace size is too large, try to set maxdim and indim <',H_dim//self.davidson_roots)
                sys.exit()


            dres = self.Davidson(self.H_PF, self.davidson_roots, self.davidson_threshold, indim, maxdim,self.davidson_maxiter)
            self.CIeigs = dres["DAVIDSON EIGENVALUES"]
            self.CIvecs = dres["DAVIDSON EIGENVECTORS"]
            t_dav_end = time.time()
            print(F' Completed Davidson iterations in {t_dav_end - t_H_build} seconds')        


    def parseCavityOptions(self, cavity_dictionary):
        """
        Parse the cavity dictionary for important parameters for the QED-CI calculation
        """
        if "omega_value" in cavity_dictionary:
            self.omega = cavity_dictionary["omega_value"]
        else:
            self.omega = 0

        if "lambda_vector" in cavity_dictionary:
            self.lambda_vector = cavity_dictionary["lambda_vector"]
        else:
            self.lambda_vector = np.array([0, 0, 0])
        if "number_of_photons" in cavity_dictionary:
            self.N_p = cavity_dictionary["number_of_photons"]
        else:
            self.N_p = 1
        if "ci_level" in cavity_dictionary:
            self.ci_level = cavity_dictionary["ci_level"]
        else:
            self.ci_level = "cis"
        if "ignore_coupling" in cavity_dictionary:
            self.ignore_coupling = cavity_dictionary["ignore_coupling"]
        else:
            self.ignore_coupling = False

        if "natural_orbitals" in cavity_dictionary:
            self.natural_orbitals = cavity_dictionary["natural_orbitals"]
        else:
            self.natural_orbitals = False

        if "canonical_mos" in cavity_dictionary:
            self.canonical_mos = cavity_dictionary["canonical_mos"]
        else:
            self.canonical_mos = False

        if self.natural_orbitals:
            if "rdm_weights" in cavity_dictionary:
                self.rdm_weights = cavity_dictionary["rdm_weights"]
            else:
                self.rdm_weights = np.array([1,1])

        if "full_diagonalization" in cavity_dictionary:
            self.full_diagonalization = cavity_dictionary["full_diagonalization"]
        else:
            self.full_diagonalization = False 

        if "davidson_roots" in cavity_dictionary:
            self.davidson_roots = cavity_dictionary["davidson_roots"]
        else:
            self.davidson_roots = 3
        if "davidson_guess" in cavity_dictionary:
            self.davidson_guess = cavity_dictionary["davidson_guess"]
        else:
            self.davidson_guess = "unit guess"
        if "davidson_threshold" in cavity_dictionary:
            self.davidson_threshold = cavity_dictionary["davidson_threshold"]
        else:
            self.davidson_threshold = 1e-5
        if "davidson_indim" in cavity_dictionary:
            self.davidson_indim = cavity_dictionary["davidson_indim"]
        else:
            self.davidson_indim = 4
        if "davidson_maxdim" in cavity_dictionary:
            self.davidson_maxdim = cavity_dictionary["davidson_maxdim"]
        else:
            self.davidson_maxdim = 20
        if "davidson_maxiter" in cavity_dictionary:
            self.davidson_maxiter = cavity_dictionary["davidson_maxiter"]
        else:
            self.davidson_maxiter = 100    

        # only need nact and nels if ci_level == "CAS"
        if self.ci_level == "cas" or self.ci_level == "CAS":
            if "nact_orbs" in cavity_dictionary:
                self.n_act_orb = cavity_dictionary["nact_orbs"]
            else:
                self.n_act_orb = 0
            if "nact_els" in cavity_dictionary:
                self.n_act_el = cavity_dictionary["nact_els"]
            else:
                self.n_act_el = 0

        else:
            self.n_act_orb = 0
            self.n_act_el = 0

    def parseArrays(self, cqed_rhf_dict):
        # grab quantities from cqed_rhf_dict
        self.rhf_reference_energy = cqed_rhf_dict["RHF ENERGY"]
        self.cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
        self.cqed_one_energy = cqed_rhf_dict["CQED-RHF ONE-ENERGY"]
        self.C = cqed_rhf_dict["CQED-RHF C"]
        self.dc = cqed_rhf_dict["DIPOLE ENERGY (1/2 (lambda cdot <mu>_e)^2)"]
        self.T_ao = cqed_rhf_dict["1-E KINETIC MATRIX AO"]
        self.V_ao = cqed_rhf_dict["1-E POTENTIAL MATRIX AO"]
        self.q_PF_ao = cqed_rhf_dict["PF 1-E QUADRUPOLE MATRIX AO"]
        self.d_PF_ao = cqed_rhf_dict["PF 1-E SCALED DIPOLE MATRIX AO"]
        self.d_cmo = cqed_rhf_dict["PF 1-E DIPOLE MATRIX MO"]
        self.d_ao = cqed_rhf_dict["PF 1-E DIPOLE MATRIX AO"]
        wfn = cqed_rhf_dict["PSI4 WFN"]
        self.d_exp = cqed_rhf_dict["EXPECTATION VALUE OF d"]
        self.Enuc = cqed_rhf_dict["NUCLEAR REPULSION ENERGY"]
        self.mu_nuc = cqed_rhf_dict["NUCLEAR DIPOLE MOMENT"]
        self.rhf_dipole_moment = cqed_rhf_dict["RHF DIPOLE MOMENT"]

        self.mu_x_ao = cqed_rhf_dict["DIPOLE AO X"]
        self.mu_y_ao = cqed_rhf_dict["DIPOLE AO Y"]
        self.mu_z_ao = cqed_rhf_dict["DIPOLE AO Z"]

        # collect rhf wfn object as dictionary
        wfn_dict = psi4.core.Wavefunction.to_file(wfn)

        # update wfn_dict with orbitals from CQED-RHF
        wfn_dict["matrix"]["Ca"] = self.C
        wfn_dict["matrix"]["Cb"] = self.C

        # update wfn object
        wfn = psi4.core.Wavefunction.from_file(wfn_dict)

        # Grab data from wavfunction class
        self.Ca = wfn.Ca()
        self.ndocc = wfn.doccpi()[0]
        self.nmo = wfn.nmo()
        self.nso = 2 * self.nmo
        self.nvirt = self.nmo - self.ndocc

        self.docc_list = [i for i in range(self.ndocc)]

        return wfn
    
    def build1MuSO(self):
        """ Will build the 1-electron dipole arrays in the spin orbital
            basis to be used for computing dipole moment expectation values
        """
        _mu_x_spin = np.einsum("uj,vi,uv", self.C, self.C, self.mu_x_ao)
        _mu_x_spin = np.repeat(_mu_x_spin, 2, axis=0)
        _mu_x_spin = np.repeat(_mu_x_spin, 2, axis=1)

        _mu_y_spin = np.einsum("uj,vi,uv", self.C, self.C, self.mu_y_ao)
        _mu_y_spin = np.repeat(_mu_y_spin, 2, axis=0)
        _mu_y_spin = np.repeat(_mu_y_spin, 2, axis=1)

        _mu_z_spin = np.einsum("uj,vi,uv", self.C, self.C, self.mu_z_ao)
        _mu_z_spin = np.repeat(_mu_z_spin, 2, axis=0)
        _mu_z_spin = np.repeat(_mu_z_spin, 2, axis=1)

        _spin_ind = np.arange(_mu_z_spin.shape[0], dtype=int) % 2

        self.mu_x_spin = _mu_x_spin * (_spin_ind.reshape(-1, 1) == _spin_ind)
        self.mu_y_spin = _mu_y_spin * (_spin_ind.reshape(-1, 1) == _spin_ind)
        self.mu_z_spin = _mu_z_spin * (_spin_ind.reshape(-1, 1) == _spin_ind)
        


    def build1HSO(self):
        """Will build the 1-electron arrays in
        the spin orbital basis that contribute to the A+Delta blocks

        """

        self.H_1e_ao = self.T_ao + self.V_ao
        self.D_1e_ao = self.q_PF_ao + self.d_PF_ao
        #if self.ignore_coupling == False:
        #    self.H_1e_ao += self.q_PF_ao + self.d_PF_ao
        # build H_spin
        # spatial part of 1-e integrals
        _H_spin = np.einsum("uj,vi,uv", self.C, self.C, self.H_1e_ao)
        _D1_spin = np.einsum("uj,vi,uv", self.C, self.C, self.D_1e_ao)

        _H_spin = np.repeat(_H_spin, 2, axis=0)
        _H_spin = np.repeat(_H_spin, 2, axis=1)

        _D1_spin = np.repeat(_D1_spin, 2, axis=0)
        _D1_spin = np.repeat(_D1_spin, 2, axis=1)



        # spin part of 1-e integrals
        spin_ind = np.arange(_H_spin.shape[0], dtype=int) % 2
        # product of spatial and spin parts
        self.Hspin = _H_spin * (spin_ind.reshape(-1, 1) == spin_ind)
        # keep a copy of only the D1 block for DSE matrix
        self.D1spin = _D1_spin * (spin_ind.reshape(-1, 1) == spin_ind)
        # this is for the Hamiltonian matrix
        self.Hspin += self.D1spin


    def build2DSO(self):
        """Will build the 2-electron arrays in the spin orbital basis
        that contribute to the A+Delta blocks

        """
        # try a more efficient way of computing the 2-e d terms
        _d_spin = np.copy(self.d_cmo)
        _d_spin = np.repeat(_d_spin, 2, axis=0)
        _d_spin = np.repeat(_d_spin, 2, axis=1)
        spin_ind = np.arange(_d_spin.shape[0], dtype=int) % 2
        self.d_spin = _d_spin * (spin_ind.reshape(-1, 1) == spin_ind)

        t1 = np.einsum("ik,jl->ijkl", self.d_spin, self.d_spin)
        t2 = np.einsum("il,jk->ijkl", self.d_spin, self.d_spin)
        # keep a copy of the TDI_spin for the DSE matrix
        self.TDI_spin = t1 - t2
        # this is for the Hamiltonian matrix
        self.antiSym2eInt = self.eri_so + self.TDI_spin

    def buildGSO(self):
        """
        Will build the 1-electron arrays in the spin orbital basis
        that contribute to the G blocks
        """

        # build g matrix
        _g = -np.sqrt(self.omega / 2) * self.d_cmo
        _g = np.repeat(_g, 2, axis=0)
        _g = np.repeat(_g, 2, axis=1)

        spin_ind = np.arange(_g.shape[0], dtype=int) % 2
        # product of spatial and spin parts
        self.g_so = _g * (spin_ind.reshape(-1, 1) == spin_ind)
        if self.ignore_coupling == True:
            self.g_so *= 0

    def buildConstantMatrices(self, ci_level):
        """
        Will build <G> * I, E_nuc * I, omega * I, and d_c * I
        """
        if ci_level == "cis" or ci_level == "CIS":
            _I = np.identity(self.CISnumDets)

        elif ci_level == "cas" or ci_level == "CAS":
            _I = np.identity(self.CASnumDets)

        elif ci_level == "fci" or ci_level == "FCI":
            _I = np.identity(self.FCInumDets)

        else:
            _I = np.identity(self.CISnumDets)

        self.Enuc_so = self.Enuc * _I
        self.G_exp_so = np.sqrt(self.omega / 2) * self.d_exp * _I
        self.Omega_so = self.omega * _I
        self.dc_so = self.dc * _I

        if self.ignore_coupling == True:
            self.G_exp_so *= 0
            self.Omega_so *= 0
            self.dc_so *= 0

    def generateCISTuple(self):
        """
        Generates the tuples that define the occupation strings for CIS
        """
        cis_dets = []
        cis_dets.append(tuple(self.docc_list))
        for i in range(self.ndocc - 1, -1, -1):
            for a in range(self.ndocc, self.nmo):
                ket = np.copy(self.docc_list)
                ket[i] = a
                ket = np.sort(ket)
                ket_tuple = tuple(ket)
                cis_dets.append(ket_tuple)

        return cis_dets
    
    def generateFCIDeterminants(self):
        """
        Generates the determinant list for building the FCI matrix
        """
        self.FCIDets = []
        self.FCInumDets = 0
        self.detmap = []
        for alpha in combinations(range(self.nmo), self.ndocc):
            for beta in combinations(range(self.nmo), self.ndocc):
                e = Determinant(alphaObtList=alpha, betaObtList=beta)
                alphabit = e.obtIndexList2ObtBits(alpha)
                betabit = e.obtIndexList2ObtBits(beta)
                self.detmap.append([alphabit,betabit])

                self.FCIDets.append(e)
                self.FCInumDets += 1
            
    def generateCISDeterminants(self):
        """
        Generates the determinant list for building the CIS matrix
        """
        self.CISdets = []
        self.CISsingdetsign = []
        self.CISnumDets = 0
        self.CISexcitation_index = []
        self.detmap = []

        # get list of tuples definining CIS occupations, including the reference
        cis_tuples = self.generateCISTuple()
        # loop through these occupations, compute excitation level, create determinants,
        # and keep track of excitation index
        for alpha in cis_tuples:
            alpha_ex_level = compute_excitation_level(alpha, self.ndocc)

            for beta in cis_tuples:
                beta_ex_level = compute_excitation_level(beta, self.ndocc)
                if alpha_ex_level + beta_ex_level <= 1:
                    e = Determinant(alphaObtList=alpha, betaObtList=beta)
                    self.CISdets.append(e)
                    alphabit = e.obtIndexList2ObtBits(alpha)
                    betabit = e.obtIndexList2ObtBits(beta)
                    self.detmap.append([alphabit,betabit])


                    self.CISnumDets += 1

        for i in range(len(self.CISdets)):
            # compare singly-excited determinant on the bra to reference ket
            # this order makes unique2[0] -> i and unique1[0] -> a
            unique1, unique2, sign = self.CISdets[
                i
            ].getUniqueOrbitalsInMixIndexListsPlusSign(self.CISdets[0])
            if i > 0:
                self.CISsingdetsign.append(sign)
                _i = unique2[0]
                _a = unique1[0]
                _ia = 2 * self.nvirt * _i + _a - 2 * self.ndocc
                self.CISexcitation_index.append(_ia)

    def generateCASCIDeterminants(self):
        """
        Generates the determinant list for building the CASCI matrix
        """
        self.CASdets = []
        self.CASdetlists = []
        self.CASsingdetsign = []
        self.CASnumDets = 0
        self.detmap = []

        n_in_orb = self.ndocc - self.n_act_el // 2
        n_ac_el_half = self.n_act_el // 2
        inactive_list = list(x for x in range(n_in_orb))

        print("Generating all determinants in active space")
        for alpha in combinations(range(self.n_act_orb), n_ac_el_half):
            alphalist = list(alpha)
            alphalist = [x + n_in_orb for x in alphalist]
            alphalist[0:0] = inactive_list
            alpha = tuple(alphalist)

            for beta in combinations(range(self.n_act_orb), n_ac_el_half):
                betalist = list(beta)
                betalist = [x + n_in_orb for x in betalist]
                betalist[0:0] = inactive_list
                beta = tuple(betalist)
                e = Determinant(alphaObtList=alpha, betaObtList=beta)
                self.CASdets.append(e)
                alphabit = e.obtIndexList2ObtBits(alpha)
                betabit = e.obtIndexList2ObtBits(beta)
                self.detmap.append([alphabit,betabit])
                self.CASnumDets += 1

        for i in range(len(self.CASdets)):
            #print(self.CASdets[i])
            unique1, unique2, sign = self.CASdets[
                i
            ].getUniqueOrbitalsInMixIndexListsPlusSign(self.CASdets[0])
            #print(unique1, unique2, sign)
            if i > 0:
                self.CASsingdetsign.append(sign)
        #for i in range(len(self.detmap)):
        #    print(self.detmap[i])
        #    a,b = self.detmap[i]
        #    print(Determinant.obtBits2ObtMixSpinIndexList(a,b))


    def generatePFHMatrix(self, ci_level):
        """
        Generate H_PF CI Matrix
        """
        if ci_level == "CIS" or ci_level == "cis":
            _dets = self.CISdets.copy()
            _numDets = self.CISnumDets

        elif ci_level == "CAS" or ci_level == "cas":
            _dets = self.CASdets.copy()
            _numDets = self.CASnumDets

        elif ci_level == "FCI" or ci_level == "fci":
            _dets = self.FCIDets.copy()
            _numDets = self.FCInumDets

        else:
            _dets = self.CISdets.copy()
            _numDets = self.CISnumDets

        # \Delta - the dse matrix
        self.Dmatrix = np.zeros((_numDets, _numDets))
        # A +\Delta 
        self.ApDmatrix = np.zeros((_numDets, _numDets))

        # one-electron only version of A+\Delta
        self.apdmatrix = np.zeros((_numDets, _numDets))

        # G matrix
        self.Gmatrix = np.zeros((_numDets, _numDets))
                
        # dipole matrix
        self.dipole_block_x = np.zeros((_numDets, _numDets))
        self.dipole_block_y = np.zeros((_numDets, _numDets))
        self.dipole_block_z = np.zeros((_numDets, _numDets))


        # build ApD, G, and dipole sub-blocks
        for i in range(_numDets):
            for j in range(i + 1):
                APD_Dict = self.calcApDMatrixElement(_dets[i], _dets[j])
                apd_dict = self.calcApDMatrixElement(
                    _dets[i], _dets[j], OneEpTwoE=False
                )
                _ApD = APD_Dict["A+Delta"]
                _D = APD_Dict["Delta"]
                _apd = apd_dict["A+Delta"]

                self.ApDmatrix[i, j] = _ApD
                self.apdmatrix[i, j] = _apd
                self.Dmatrix[i, j] = _D

                self.ApDmatrix[j, i] = _ApD
                self.apdmatrix[j, i] = _apd
                self.Dmatrix[j, i] = _D

                self.Gmatrix[i, j] = self.calcGMatrixElement(_dets[i], _dets[j])
                self.Gmatrix[j, i] = self.Gmatrix[i, j]

                _dipole_moment = self.calcMuMatrixElement(_dets[i], _dets[j])
                self.dipole_block_x[i, j] = _dipole_moment[0]
                self.dipole_block_x[j, i] = _dipole_moment[0]
                self.dipole_block_y[i, j] = _dipole_moment[1]
                self.dipole_block_y[j, i] = _dipole_moment[1]
                self.dipole_block_z[i, j] = _dipole_moment[2]
                self.dipole_block_z[j, i] = _dipole_moment[2]

        # dimension of composite dimension
        _compDim = (self.N_p + 1) * _numDets

        # build arrays in the full N_el * N_ph space
        self.H_PF = np.zeros((_compDim, _compDim))
        #_tryHPF = np.zeros((_compDim, _compDim))

        # one-electron version of Hamiltonian
        self.H_1E = np.zeros((_compDim, _compDim))

        # dipole matrix in CI basis
        self.MU_X = np.zeros((_compDim, _compDim))
        self.MU_Y = np.zeros((_compDim, _compDim))
        self.MU_Z = np.zeros((_compDim, _compDim))

        # if N_p = 0, build the DSE matrix
        if self.N_p == 0:
            self.H_DSE = np.zeros((_compDim, _compDim))

        if self.N_p == 0:
            self.H_DSE[:_numDets, :_numDets] = np.copy(self.Dmatrix)
            self.H_PF[:_numDets, :_numDets] = self.ApDmatrix + self.Enuc_so + self.dc_so 
            self.H_1E[:_numDets, :_numDets] = self.apdmatrix + self.Enuc_so + self.dc_so
            self.MU_X[:_numDets, :_numDets] = self.dipole_block_x
            self.MU_Y[:_numDets, :_numDets] = self.dipole_block_y
            self.MU_Z[:_numDets, :_numDets] = self.dipole_block_z

        elif self.N_p == 1:
            # full hamiltonian
            self.H_PF[:_numDets, :_numDets] = self.ApDmatrix + self.Enuc_so + self.dc_so
            # 1-e piece
            self.H_1E[:_numDets, :_numDets] = self.apdmatrix + self.Enuc_so + self.dc_so

            # Dipole Hamiltonian
            self.MU_X[:_numDets, :_numDets] = self.dipole_block_x
            self.MU_X[_numDets:, _numDets:] = self.dipole_block_x
            self.MU_Y[:_numDets, :_numDets] = self.dipole_block_y
            self.MU_Y[_numDets:, _numDets:] = self.dipole_block_y
            self.MU_Z[:_numDets, :_numDets] = self.dipole_block_z
            self.MU_Z[_numDets:, _numDets:] = self.dipole_block_z


            # full Hamiltonian
            self.H_PF[_numDets:, _numDets:] = (
                self.ApDmatrix + self.Enuc_so + self.dc_so + self.Omega_so
            )

            # 1-e piece
            self.H_1E[_numDets:, _numDets:] = (
                self.apdmatrix + self.Enuc_so + self.dc_so + self.Omega_so
            )
            self.H_PF[_numDets:, :_numDets] = self.Gmatrix + self.G_exp_so
            self.H_PF[:_numDets, _numDets:] = self.Gmatrix + self.G_exp_so

        else:

            for i in range(self.N_p+1):
                bra_s = i * _numDets
                bra_e = (i+1) * _numDets
                ket_s = bra_s
                ket_e = bra_e
                self.H_PF[bra_s:bra_e, ket_s:ket_e] = self.ApDmatrix + self.Enuc_so + self.dc_so + i * self.Omega_so

            for i in range(self.N_p+1):
                if i==0:
                    j = i + 1
                    bra_s = i * _numDets
                    bra_e = (i+1) * _numDets
                    ket_s = j * _numDets
                    ket_e = (j+1) * _numDets

                    self.H_PF[bra_s:bra_e, ket_s:ket_e] = self.Gmatrix * np.sqrt(j) + self.G_exp_so * np.sqrt(j)

                elif i == (self.N_p):
                    j  = i - 1 #<== equivalent to i = j + 1
                    bra_s = i * _numDets
                    bra_e = (i+1) * _numDets
                    ket_s = j * _numDets
                    ket_e = (j+1) * _numDets

                    # the <i|\hat{b}^{\dagger}|j> -> \sqrt{j+1} <i|j+1> term survives
                    self.H_PF[bra_s:bra_e, ket_s:ket_e] = self.Gmatrix * np.sqrt(j+1) + self.G_exp_so * np.sqrt(j+1)

                else:
                    j = i + 1 #<== equivalent to i = j - 1
                    bra_s = i * _numDets
                    bra_e = (i+1) * _numDets
                    ket_s = j * _numDets
                    ket_e = (j+1) * _numDets

                    # the <i|\hat{b}|j> -> \sqrt{j} <i|j-1> term survives
                    self.H_PF[bra_s:bra_e, ket_s:ket_e] = self.Gmatrix * np.sqrt(j) + self.G_exp_so * np.sqrt(j)

                    j  = i - 1 #<== equivalent to i = j + 1
                    bra_s = i * _numDets
                    bra_e = (i+1) * _numDets
                    ket_s = j * _numDets
                    ket_e = (j+1) * _numDets

                    # the <i|\hat{b}^{\dagger}|j> -> \sqrt{j+1} <i|j+1> term survives
                    self.H_PF[bra_s:bra_e, ket_s:ket_e] = self.Gmatrix * np.sqrt(j+1) + self.G_exp_so * np.sqrt(j+1)

            ##assert np.allclose(_tryHPF, self.H_PF)
            #self.tryHPF = np.copy(_tryHPF)



    def calcApDMatrixElement(self, det1, det2, OneEpTwoE=True):
        """
        Calculate a matrix element by two determinants
        """

        numUniqueOrbitals = None
        if det1.diff2OrLessOrbitals(det2) and OneEpTwoE:
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 0:
                # print(F' cal matrix element for {det1} and {det2}\n')
                return self.calcMatrixElementIdentialDet(det1)
            if numUniqueOrbitals == 2:
                return self.calcMatrixElementDiffIn2(det1, det2)
            elif numUniqueOrbitals == 1:
                return self.calcMatrixElementDiffIn1(det1, det2)
            else:
                #
                slater_condon_dict = {
                    "A+Delta" : 0.0,
                    "Delta" : 0.0
                }
                return slater_condon_dict
        elif det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 0:
                return self.calcMatrixElementIdentialDet(det1, omit2E=True)
            elif numUniqueOrbitals == 1:
                return self.calcMatrixElementDiffIn1(det1, det2, omit2E=True)
            else:
                slater_condon_dict = {
                    "A+Delta" : 0.0,
                    "Delta" : 0.0
                }
                return slater_condon_dict

        else:
            slater_condon_dict = {
                    "A+Delta" : 0.0,
                    "Delta" : 0.0
                }
            return slater_condon_dict
        
    def calcMuMatrixElement(self, det1, det2):
        """
        Calculate a Mu matrix element between two determinants
        """
        numUniqueOrbitals = None
        if det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 0:
                
                return self.calcMuMatrixElementIdentialDet(det1)
            elif numUniqueOrbitals == 1:
                return self.calcMuMatrixElementDiffIn1(det1, det2)
            else:
                #
                return np.array([0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])
        

    def calcGMatrixElement(self, det1, det2):
        """
        Calculate a matrix element by two determinants
        """

        numUniqueOrbitals = None
        if det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 0:
                # print(F' cal matrix element for {det1} and {det2}\n')
                return self.calcGMatrixElementIdentialDet(det1)
            elif numUniqueOrbitals == 1:
                return self.calcGMatrixElementDiffIn1(det1, det2)
            else:
                #
                return 0.0
        else:
            return 0.0

    def calcMatrixElementDiffIn2(self, det1, det2):
        """
        Calculate a matrix element by two determinants where the determinants differ by 2 spin orbitals
        """

        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        _ApD = sign * self.antiSym2eInt[unique1[0], unique1[1], unique2[0], unique2[1]]
        _D = sign * self.TDI_spin[unique1[0], unique1[1], unique2[0], unique2[1]]
        #return sign * self.antiSym2eInt[unique1[0], unique1[1], unique2[0], unique2[1]]
        slater_condon_dict = {
            "A+Delta" : _ApD,
            "Delta" : _D
        }
        return slater_condon_dict
        #return _ApD, _D

    def calcMatrixElementDiffIn1(self, det1, det2, omit2E=False):
        """
        Calculate a matrix element by two determinants where the determinants differ by 1 spin orbitals
        """

        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        Helem = self.Hspin[m, p]
        D1elem = self.D1spin[m, p]

        common = det1.getCommonOrbitalsInMixedSpinIndexList(det2)

        if omit2E == False:
            Relem = 0.0
            D2elem = 0.0
            for n in common:
                Relem += self.antiSym2eInt[m, n, p, n]
                D2elem += self.TDI_spin[m, n, p, n]

        else:
            Relem = 0.0
            D2elem = 0.0

        _ApD_elem = sign * (Helem + Relem)
        _D_elem = sign * (D1elem + D2elem)

        #return sign * (Helem + Relem)
        slater_condon_dict = {
            "A+Delta" : _ApD_elem,
            "Delta" : _D_elem
        }
        return slater_condon_dict

    def calcGMatrixElementDiffIn1(self, det1, det2):
        """
        Calculate a matrix element by two determinants where the determinants differ by 1 spin orbitals
        """

        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        Gelem = self.g_so[m, p]
        return sign * Gelem
    
    def calcMuMatrixElementDiffIn1(self, det1, det2):
        """
        Calculate a dipole matrix element by two determinants where the determinants differ by 1 spin orbitals
        """
        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        mu_x = self.mu_x_spin[m, p] * sign
        mu_y = self.mu_y_spin[m, p] * sign
        mu_z = self.mu_z_spin[m, p] * sign
        return np.array([mu_x, mu_y, mu_z])

    
    def calcMuMatrixElementIdentialDet(self, det):
        """
        Calculate a matrix element by two determinants where they are identical
        """

        spinObtList = det.getOrbitalMixedIndexList()
        mu_x = 0.0
        mu_y = 0.0
        mu_z = 0.0
        for m in spinObtList:
            mu_x += self.mu_x_spin[m, m]
            mu_y += self.mu_y_spin[m, m]
            mu_z += self.mu_z_spin[m, m]

        return np.array([mu_x, mu_y, mu_z])

    def calcGMatrixElementIdentialDet(self, det):
        """
        Calculate a matrix element by two determinants where they are identical
        """

        spinObtList = det.getOrbitalMixedIndexList()
        Gelem = 0.0
        for m in spinObtList:
            Gelem += self.g_so[m, m]

        return Gelem

    def calcMatrixElementIdentialDet(self, det, omit2E=False):
        """
        Calculate a matrix element by two determinants where they are identical
        """

        spinObtList = det.getOrbitalMixedIndexList()
        Helem = 0.0
        D1elem = 0.0
        for m in spinObtList:
            Helem += self.Hspin[m, m]
            D1elem += self.D1spin[m,m]

        length = len(spinObtList)
        if omit2E == False:
            Relem = 0.0
            D2elem = 0.0
            for m in range(length - 1):
                for n in range(m + 1, length):
                    Relem += self.antiSym2eInt[
                        spinObtList[m], spinObtList[n], spinObtList[m], spinObtList[n]
                    ]
                    D2elem += self.TDI_spin[
                        spinObtList[m], spinObtList[n], spinObtList[m], spinObtList[n]
                    ]

        else:
            Relem = 0.0
            D2elem = 0.0

        _ApDelem = Helem + Relem
        _Delem = D1elem + D2elem

        slater_condon_dict = {
            "A+Delta" : _ApDelem,
            "Delta" : _Delem
        }
        return slater_condon_dict

 
    
    def generateOrbitalBasis(self, molecule_string, psi4_options_dict):
        """
        Calculate the orbitals basis for the CI calculation
        Needs to:
            Determine the orbital type -
            if cqed-rhf:
               1. Run CQED-RHF
               2. Update 
            1. Build CIS determinant list
            2. Obtain CIS vectors
            3. Build CIS 1RDM
            4. Diagonalize 1RDM -> vectors U
            5. Dot original MO coefficients into vectors NO = C @ U
        """

        t_hf_start = time.time()
        if self.canonical_mos:
            cqed_rhf_dict = cqed_rhf(self.lambda_vector, molecule_string, psi4_options_dict, canonical_basis=True)
        else:
            cqed_rhf_dict = cqed_rhf(self.lambda_vector, molecule_string, psi4_options_dict)
        t_hf_end = time.time()
        print(F' Completed QED-RHF in {t_hf_end - t_hf_start} seconds')


        # Parse output of cqed-rhf calculation
        psi4_wfn = self.parseArrays(cqed_rhf_dict)


        if self.natural_orbitals: #<== need to run CIS
            self.buildArraysInOrbitalBasis(psi4_wfn)
            t_det_start = time.time()
            self.generateCISDeterminants()
            H_dim = self.CISnumDets * 2 
            t_det_end = time.time()
            print(F' Completed determinant list in {t_det_end - t_det_start} seconds ')
            #indim = self.davidson_indim * self.davidson_roots
            #maxdim = self.davidson_maxdim * self.davidson_roots
            #if (indim > H_dim or maxdim > H_dim):
            #    print('subspace size is too large, try to set maxdim and indim <',H_dim//self.davidson_roots)
            #    sys.exit()
                
            # build Constant matrices
            self.buildConstantMatrices("cis")
            t_const_end = time.time()
            print(F' Completed constant offset matrix in {t_const_end - t_det_end} seconds')
            
            # Build Matrix
            self.generatePFHMatrix("cis")
            t_H_build = time.time()
            print(F' Completed Hamiltonian build in {t_H_build - t_const_end} seconds')
            #dres = self.Davidson(self.H_PF, self.davidson_roots, self.davidson_threshold, indim, maxdim,self.davidson_maxiter)
            #self.cis_e = dres["DAVIDSON EIGENVALUES"]
            #self.cis_c = dres["DAVIDSON EIGENVECTORS"]
            #t_dav_end = time.time()
            #print(F' Completed Davidson iterations in {t_dav_end - t_H_build} seconds')
            self.cis_e, self.cis_c = np.linalg.eigh(self.H_PF)
            self.classifySpinState()

            # get RDM from CIS ground-state - THIS CAN BE GENERALIZED TO 
            # get RDM from different states!
            _D1_avg = np.zeros((self.nmo, self.nmo))
            N_states = len(self.rdm_weights)
            _norm = np.sum(self.rdm_weights)
            for i in range(N_states):
                _sidx = self.singlets[i]
                self.calc1RDMfromCIS(self.cis_c[:, _sidx])
                _D1_avg += self.rdm_weights[i] / _norm * self.D1_spatial 


            _eig, _vec = np.linalg.eigh(_D1_avg)
            _idx = _eig.argsort()[::-1]
            self.noocs = _eig[_idx]
            self.no_vec = _vec[:,_idx]
            self.nat_orbs = np.dot(self.C, self.no_vec)

            # now we have the natural orbitals, make sure we update the quantities
            # that use the orbitals
            self.C = np.copy(self.nat_orbs)

            # collect rhf wfn object as dictionary
            wfn_dict = psi4.core.Wavefunction.to_file(psi4_wfn)
            
            # update wfn_dict with orbitals from CQED-RHF
            wfn_dict["matrix"]["Ca"] = self.C
            wfn_dict["matrix"]["Cb"] = self.C
            
            # update wfn object
            psi4_wfn = psi4.core.Wavefunction.from_file(wfn_dict)
            
            # Grab data from wavfunction class
            self.Ca = psi4_wfn.Ca()

            # transform d_ao to d_cmo using natural orbitals
            self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)

        return psi4_wfn


    def buildArraysInOrbitalBasis(self, p4_wfn):

        # build 1H in orbital basis
        t_1H_start = time.time()
        self.build1HSO()
        t_1H_end = time.time()
        print(F' Completed 1HSO Build in {t_1H_end - t_1H_start} seconds')
        
        # build 2eInt in cqed-rhf basis
        mints = psi4.core.MintsHelper(p4_wfn.basisset())
        self.eri_so = np.asarray(mints.mo_spin_eri(self.Ca, self.Ca))
        t_eri_end = time.time()
        print(F' Completed ERI Build in {t_eri_end - t_1H_end} seconds ')
        
        # form the 2H in spin orbital basis
        self.build2DSO()
        t_2d_end = time.time()
        print(F' Completed 2D build in {t_2d_end - t_eri_end} seconds')
        
        # build the array to build G in the so basis
        self.buildGSO()
        t_1G_end = time.time()
        print(F' Completed 1G build in {t_1G_end - t_2d_end} seconds')

        # build the x, y, z components of the dipole array in the so basis
        self.build1MuSO()
        t_dipole_end = time.time()
        print(F' Completed the Dipole Matrix Build in {t_dipole_end - t_1G_end} seconds')


    def calc1RDMfromCIS(self, c_vec):
        _nDets = self.CISnumDets
        _nSingles = _nDets - 1
        self.nvirt = self.nmo - self.ndocc

        # get different terms in c_vector
        _c00 = c_vec[0]
        _c10 = c_vec[1:_nDets]
        _c01 = c_vec[_nDets]
        _c11 = c_vec[_nDets + 1 :]

        # initialize different blocks of 1RDM
        self.Dij = np.zeros((2 * self.ndocc, 2 * self.ndocc))
        self.Dab = np.zeros((2 * self.nvirt, 2 * self.nvirt))
        self.Dia = np.zeros((2 * self.ndocc, 2 * self.nvirt))

        # arrange the _c10 and _c11 elements in arrays
        # indexed by spin orbital excitation labels
        # keeping track of the sign!
        _c10_nso = np.zeros(4 * self.ndocc * self.nvirt)
        _c11_nso = np.zeros(4 * self.ndocc * self.nvirt)

        for n in range(_nSingles):
            ia = self.CISexcitation_index[n]
            _a = ia % (2 * self.nvirt)
            _i = (ia - _a) // (2 * self.nvirt)
            _c10_nso[ia] = _c10[n] * self.CISsingdetsign[n]
            _c11_nso[ia] = _c11[n] * self.CISsingdetsign[n]

        # build _Dij block
        for i in range(2 * self.ndocc):
            for j in range(2 * self.ndocc):
                self.Dij[i, j] = (_c00 * _c00 + _c01 * _c01) * (i == j)
                self.Dij[i, j] += (np.dot(_c10.T, _c10) + np.dot(_c11.T, _c11)) * (
                    i == j
                )
                dum = 0
                for a in range(2 * self.nvirt):
                    ia = i * 2 * self.nvirt + a
                    ja = j * 2 * self.nvirt + a
                    dum += np.conj(_c10_nso[ia]) * _c10_nso[ja]
                    dum += np.conj(_c11_nso[ia]) * _c11_nso[ja]
                self.Dij[i, j] -= dum

        for a in range(2 * self.nvirt):
            for b in range(2 * self.nvirt):
                dum = 0.0
                for i in range(2 * self.ndocc):
                    ia = i * 2 * self.nvirt + a
                    ib = i * 2 * self.nvirt + b
                    dum += np.conj(_c10_nso[ia]) * _c10_nso[ib]
                    dum += np.conj(_c11_nso[ia]) * _c11_nso[ib]

                self.Dab[a, b] += dum

        for i in range(2 * self.ndocc):
            for a in range(2 * self.nvirt):
                ia = i * 2 * self.nvirt + a
                self.Dia[i, a] = np.conj(_c10_nso[ia]) * _c00
                self.Dia[i, a] += np.conj(_c11_nso[ia]) * _c01

        _D1 = np.concatenate((self.Dij, self.Dia), axis=1)
        _D2 = np.concatenate((self.Dia.T, self.Dab), axis=1)

        # spin-orbital 1RDM
        self.D1 = np.concatenate((_D1, _D2), axis=0)

        # now build spatial orbital 1RDM
        _D_aa = np.zeros((self.nmo,self.nmo))
        _D_bb = np.zeros((self.nmo,self.nmo))
        
        for p in range(self.D1.shape[0]):
            for q in range(self.D1.shape[1]):
                
                i=p%2
                j=(p-i)//2
                
                k=q%2
                l=(q-k)//2
                
                if i==0 and k==0:
                    _D_aa[j,l]=self.D1[p,q]
                    
                if i==1 and k==1:
                    _D_bb[j,l]=self.D1[p,q]

        # spatial orbital 1RDM
        self.D1_spatial = _D_aa + _D_bb

    
    def Davidson(self, H, nroots, threshold,indim,maxdim,maxiter):
        H_diag = np.diag(H)
        H_dim = len(H[:,0])
        L = indim 

        # When L exceeds Lmax we will collapse the guess space so our sub-space
        # diagonalization problem does not grow too large
        Lmax = maxdim
        

        # An array to hold the excitation energies
        theta = [0.0] * L

        #generate initial guess
        Q_idx = H_diag.argsort()[:indim]
        #print(Q_idx)
        if self.davidson_guess == "unit guess":
            print("use unit guess")
            Q = np.eye(H_dim)[:, Q_idx]
        else:    
            print("use random guess")
            Q = np.random.rand(H_dim,indim)
        #print(Q)
        #print(np.shape(Q)) 

        num_iter = maxiter
        for a in range(0, num_iter):
            print("\n")
            #orthonormalization of basis vectors by QR
            Q, R = np.linalg.qr(Q)
            print(Q.shape)
            L = Q.shape[1]#dynamic dimension of subspace
            print('iteration', a+1, 'subspace dimension', L)
            theta_old = theta[:nroots]
            #print("CI Iter # {:>6} L = {}".format(EOMCCSD_iter, L))
            # singma build
            S = np.zeros_like(Q)
            S = np.einsum("pq,qi->pi", H, Q)  

            # Build the subspace Hamiltonian
            G = np.dot(Q.T, S)
            #print(np.allclose(G, G.T, 1e-12, 1e-12))    
            # Diagonalize it, and sort the eigenvector/eigenvalue pairs
            theta, alpha = np.linalg.eigh(G)
            #print(theta)
            idx = theta.argsort()[:nroots]
            theta = theta[idx]
            alpha = alpha[:, idx]
            # This vector will hold the new guess vectors to add to our space
            add_Q = []
            w = np.zeros((H_dim,nroots))
            residual_norm = np.zeros((nroots))
            unconverged_idx = []
            convergence_check = np.zeros((nroots),dtype=str)
            conv = 0
            for j in range(nroots):
                # Compute a residual vector "w" for each root we seek
                w[:,j] = np.dot(S, alpha[:, j]) - theta[j] * np.dot(Q, alpha[:, j])
                residual_norm[j] = np.sqrt(np.dot(w[:,j].T,w[:,j]))
                if (residual_norm[j] < threshold):
                    conv += 1
                    convergence_check[j] = 'Yes'
                else:
                    unconverged_idx.append(j)
                    convergence_check[j] = 'No'
            print(unconverged_idx) 
            
            print('root','residual norm','Eigenvalue','Convergence')
            for j in range(nroots):
                print(j+1,residual_norm[j],theta[j],convergence_check[j])


            if (conv == nroots):
                print("converged!")
                break

            if (conv < nroots and a+1==num_iter):
                print('maxiter reached, try to increase maxiter or subspace size')
                break
            
            preconditioned_w = np.zeros((H_dim,len(unconverged_idx)))
            #print('wshape',w.shape)
            if(len(unconverged_idx) >0):
                for n in range(H_dim):
                    for k in range(len(unconverged_idx)):
                    # Precondition the residual vector to form a correction vector
                        dum = (theta[unconverged_idx[k]] - H_diag[n])
                        
                        if np.abs(dum) <1e-20:
                            #print('error!!!!!!!!!!!')
                            preconditioned_w[n,k] = 0.0
                        else:
                            preconditioned_w[n,k] = w[n,unconverged_idx[k]] /dum
                        #print(preconditioned_w[n,k],n,k)
                add_Q.append(preconditioned_w)
                #print(add_Q)
            
            if (Lmax-L < len(unconverged_idx)):
                unconverged_w = np.zeros((H_dim,len(unconverged_idx)))
                Q=np.dot(Q, alpha)
                
                for i in range(len(unconverged_idx)):
                    unconverged_w[:,i]=w[:,unconverged_idx[i]] 
                
                #Q=np.append(Q,unconverged_w)
                #print(Q)
                #print(unconverged_w)
                Q = np.concatenate((Q,unconverged_w),axis=1)
                print(Q.shape)
                #Q=np.column_stack(Qtup)
                
                # These vectors will give the same eigenvalues at the next
                # iteration so to avoid a false convergence we reset the theta
                # vector to theta_old
                theta = theta_old
            else:
                # if not we add the preconditioned residuals to the guess
                # space, and continue. Note that the set will be orthogonalized
                # at the start of the next iteration
                Qtup = tuple(Q[:, i] for i in range(L)) + tuple(add_Q)
                Q = np.column_stack(Qtup)
                #print(Q)
        Q=np.dot(Q, alpha)
        for i in range(Q.shape[1]):
            #print("state",i, "energy =",theta[i])
            print("        amplitude","      position", "         most important determinants","             number of photon")
            index=np.argsort(np.abs(Q[:,i]))
            c0 = index[Q.shape[0]-1]%(H_dim//2)
            d0 = (index[Q.shape[0]-1]-c0)//(H_dim//2)
            a0,b0 = self.detmap[c0]
            alphalist = Determinant.obtBits2ObtIndexList(a0)
            betalist = Determinant.obtBits2ObtIndexList(b0)
            singlet = 1
            for j in range(min(H_dim,10)):
                c = index[Q.shape[0]-j-1]%(H_dim//2)
                d = (index[Q.shape[0]-j-1]-c)//(H_dim//2)
                a,b = self.detmap[c]
                if a == b0 and b == a0 and np.abs(Q[index[Q.shape[0]-j-1]][i]-(-1)*Q[index[Q.shape[0]-1]][i]) < 1e-4:
                    singlet = singlet * 0
                else:
                    singlet = singlet * 1
                alphalist = Determinant.obtBits2ObtIndexList(a)
                betalist = Determinant.obtBits2ObtIndexList(b)

                print("%20.12lf"%(Q[index[Q.shape[0]-j-1]][i]),"%9.3d"%(index[Q.shape[0]-j-1]),"      alpha",alphalist,"   beta",betalist,"%4.1d"%(d), "photon")
            #print("state",i, "energy =",theta[i], singlet)
            if singlet == 1:
                print("state",i, "energy =",theta[i], '  singlet',"%2.1d"%(d0), "photon")
            else:
                print("state",i, "energy =",theta[i], '  triplet',"%2.1d"%(d0), "photon")


        


        davidson_dict = {
        "DAVIDSON EIGENVALUES": theta,
        "DAVIDSON EIGENVECTORS": Q,
        }

        return davidson_dict
    
    def classifySpinState(self):
        self.triplets = []
        self.singlets = [item for item in range(2*self.CISnumDets)]
        Q = self.cis_c
        theta = self.cis_e
        singlesDets = self.CISnumDets - 1
        halfDets = singlesDets // 2

        H_dim = Q.shape[0]
        print("H_dim is",H_dim)
        print("dimensions of detmap", len(self.detmap))
        for i in range(Q.shape[1]):
            print("state",i, "energy =",theta[i])
            needs_assignment = True
            print("        amplitude","      position", "         most important determinants","             number of photon")
            for j in range(1, halfDets-1):
                #if j<H_dim/2: 
                # zero photon
                j_off = j + halfDets
                if np.abs(Q[j,i]) >0.2:
                    a1, b1 = self.detmap[j]
                    a2, b2 = self.detmap[j_off]
                    alpha1 = Determinant.obtBits2ObtIndexList(a1)
                    beta1 = Determinant.obtBits2ObtIndexList(b1)
                    alpha2 = Determinant.obtBits2ObtIndexList(a2)
                    beta2 = Determinant.obtBits2ObtIndexList(b2)
                    if needs_assignment:
                        if np.isclose(Q[j,i], -1*Q[j_off,i]):
                            self.triplets.append(i)
                            self.singlets.remove(i)
                            needs_assignment = False
                    

                    print("%20.12lf"%(Q[j,i]),"%9.3d"%(j),"      alpha",alpha1,"beta",beta1, "       0 photon")
                    print("%20.12lf"%(Q[j_off,i]),"%9.3d"%(j_off),"      alpha",alpha2,"beta",beta2, "       0 photon")

                # one photon
                j_o = j + singlesDets + 1
                j_o_off = j_o + halfDets
                if np.abs(Q[j_o, i]) > 0.2:
                    a1, b1 = self.detmap[j]
                    a2, b2 = self.detmap[j_off]
                    alpha1 = Determinant.obtBits2ObtIndexList(a1)
                    beta1 = Determinant.obtBits2ObtIndexList(b1)
                    alpha2 = Determinant.obtBits2ObtIndexList(a2)
                    beta2 = Determinant.obtBits2ObtIndexList(b2)
                    if needs_assignment:
                        if np.isclose(Q[j_o,i], -1*Q[j_o_off,i]):
                            self.triplets.append(i)
                            self.singlets.remove(i)
                            needs_assignment = False
                        
                    print("%20.12lf"%(Q[j_o,i]),"%9.3d"%(j_o),"      alpha",alpha1,"beta",beta1, "       1 photon")
                    print("%20.12lf"%(Q[j_o_off,i]),"%9.3d"%(j_o_off),"      alpha",alpha2,"beta",beta2, "       1 photon")


    def reduceCISpace(self):
        """
        function to map QED-CI space to electronic subspace for when 
        there is no cavity effect
        """
        # make the CI matrix in the electronic subspace only
        _H_CI_NO_CAV = self.ApDmatrix + self.Enuc_so + self.dc_so
        
        # diagonize only the electronic subspace CI matrix
        eigs, vecs = np.linalg.eigh(_H_CI_NO_CAV)
        self.CIeigs = np.copy(eigs)
        self.CIvecs = np.copy(vecs)
        #self.CIeigs, self.CIvecs = np.linalg.eigh(_H_CI_NO_CAV)
        
        # get the number of determinants
        _numDets = len(self.CIeigs)
        
        # reduce the dipole arrays to the electronic subspace
        self.MU_X = np.copy(self.MU_X[:_numDets, :_numDets])
        self.MU_Y = np.copy(self.MU_Y[:_numDets, :_numDets])
        self.MU_Z = np.copy(self.MU_Z[:_numDets, :_numDets])


    def sort_dipole_allowed_states(self, N_el):
        """
        A function to capture the indices of states with a dipole-allowed
        singlet transition from the ground-state
        """
        
        # we want to find the first N_el states that are coupled to the ground state through a dipole transition
        _singlet_states = [0]
        _num_kets = 0
        _ket_idx = 1
        _sing_idx = 1
        zero_vec = np.array([0., 0., 0.])
        while _num_kets < N_el - 1:
            _tmp_mu = self.compute_dipole_moment(0, _ket_idx)
            if np.allclose(zero_vec, _tmp_mu):
                _ket_idx += 1
            else:
                _singlet_states.append(_ket_idx)
                _ket_idx += 1
                _num_kets += 1
                _sing_idx += 1
                
        return _singlet_states
    
    def compute_dipole_moment(self, braI, ketJ):
        """
        Compute the dipole moment expectation value  using CIvecs indexed
        by braI and ketJ <braI | \hat{mu} | ketJ> where \hat{mu} is built
        in the CI basis.  Note that this will be a transition dipole moment
        when braI refers to a different state than ketJ
        
        """
        # array for the dipole moment
        _dm = np.zeros(3)

        # x-component 
        _tmpX = np.dot(self.MU_X, self.CIvecs[:,ketJ])
        _dm[0] = np.dot(self.CIvecs[:,braI].T, _tmpX)
        
        # y-component
        _tmpY = np.dot(self.MU_Y, self.CIvecs[:,ketJ])
        _dm[1] = np.dot(self.CIvecs[:,braI].T, _tmpY)

        # z-component
        _tmpZ = np.dot(self.MU_Z, self.CIvecs[:,ketJ])
        _dm[2] = np.dot(self.CIvecs[:,braI].T, _tmpZ)

        return _dm
    
    def compute_dipole_self_energy(self, braI, ketJ):
        """
        Compute the dipole self energy expectation value  using CIvecs indexed
        by braI and ketJ <braI | \hat{H}_{DSE} | ketJ> where \hat{H}_{DSE} is built
        in the CI basis. 
        """
        _tmpDSE = np.dot(self.H_DSE, self.CIvecs[:,ketJ])
        _DSE = np.dot(self.CIvecs[:,braI], _tmpDSE)

        return _DSE


    def compute_dipole_moments(self, states):
        """
        Given an array of states, compute all the dipole moments between those states
        """
        _Ns = len(states)
        _singlet_dipole_moments = np.zeros((_Ns, _Ns, 3))
        
        for i in range(_Ns):
            a = states[i]
            for j in range(_Ns):
                b = states[j]
                _singlet_dipole_moments[i, j, :] = self.compute_dipole_moment(a, b) + self.mu_nuc * (a==b)
        
        return _singlet_dipole_moments
    

    def compute_dipole_self_energies(self, states):
        """
        Given an array of states, compute all the dipole moments between those states
        """
        _Ns = len(states)
        _singlet_dses = np.zeros((_Ns, _Ns, 3))
        
        for i in range(_Ns):
            a = states[i]
            for j in range(_Ns):
                b = states[j]
                _singlet_dses[i, j, :] = self.compute_dipole_self_energy(a, b) 
        
        return _singlet_dses


                        