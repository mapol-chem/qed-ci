"""
Helper Classes for Cavity Quantum Electrodynamics Configuration Interaction methods.
Adapted from a helper class for Configuration Interaction by the Psi4Numpy Developers, specifically
Tianyuan Zhang, Jeffrey B. Schriber, and Daniel G. A. Smith.

References:
- Equations from [Szabo:1996], [Foley:2022], [Koch:2020]
"""

__authors__ = "Jonathan J. Foley IV"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2023, The Psi4NumPy Developers, Foley Lab, Mapol Project"
__license__ = "GNU-GPL-3"
__date__ = "2023-01-21"

import psi4
from helper_cqed_rhf import cqed_rhf
from itertools import combinations


def compute_excitation_level(ket, ndocc):
    level = 0
    homo = ndocc - 1
    for i in range(0, len(ket)):
        if ket[i] > homo:
            level += 1
    return level


def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return list(a - b) + list(b - a)


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
        N_photon,
        molecule_string,
        psi4_options_dict,
        lambda_vector,
        omega_val,
        n_act_el,
        n_act_orb,
        ignore_coupling=False,
        cas=False,
    ):
        """
        Constructor for matrix elements of the PF Hamiltonian
        """
        self.cas = cas
        self.n_act_el = n_act_el
        self.n_act_orb = n_act_orb
        # now compute cqed-rhf to get transformation vectors with cavity
        self.ignore_coupling = ignore_coupling
        self.N_p = N_photon
        cqed_rhf_dict = cqed_rhf(lambda_vector, molecule_string, psi4_options_dict)
        self.omega = omega_val

        # Parse
        p4_wfn = self.parseArrays(cqed_rhf_dict)

        # build 1H in spin orbital basis
        self.build1HSO()

        # build 2eInt in cqed-rhf basis
        mints = psi4.core.MintsHelper(p4_wfn.basisset())
        self.eri_so = np.asarray(mints.mo_spin_eri(self.Ca, self.Ca))

        # form the 2H in spin orbital basis
        self.build2DSO()

        # build the array to build G in the so basis
        self.buildGSO()

        # build the determinant list
        self.generateDeterminants(psi4_options_dict)

        # build Constant matrices
        self.buildConstantMatrices()

        # Build Matrix
        self.generatePFHMatrix()

    def parseArrays(self, cqed_rhf_dict):
        # grab quantities from cqed_rhf_dict
        self.rhf_reference_energy = cqed_rhf_dict["RHF ENERGY"]
        self.cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
        self.C = cqed_rhf_dict["CQED-RHF C"]
        self.dc = cqed_rhf_dict["DIPOLE ENERGY (1/2 (\lambda \cdot <\mu>_e)^2)"]
        self.T_ao = cqed_rhf_dict["1-E KINETIC MATRIX AO"]
        self.V_ao = cqed_rhf_dict["1-E POTENTIAL MATRIX AO"]
        self.q_PF_ao = cqed_rhf_dict["PF 1-E QUADRUPOLE MATRIX AO"]
        self.d_PF_ao = cqed_rhf_dict["PF 1-E SCALED DIPOLE MATRIX AO"]
        self.d_cmo = cqed_rhf_dict["PF 1-E DIPOLE MATRIX MO"]
        wfn = cqed_rhf_dict["PSI4 WFN"]
        self.d_exp = cqed_rhf_dict["EXPECTATION VALUE OF d"]
        self.Enuc = cqed_rhf_dict["NUCLEAR REPULSION ENERGY"]

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

        return wfn

    def build1HSO(self):
        """Will build the 1-electron arrays in
        the spin orbital basis that contribute to the A+\Delta blocks

        """
        self.H_1e_ao = self.T_ao + self.V_ao
        if self.ignore_coupling == False:
            self.H_1e_ao += self.q_PF_ao + self.d_PF_ao
        # build H_spin
        # spatial part of 1-e integrals
        _H_spin = np.einsum("uj,vi,uv", self.Ca, self.Ca, self.H_1e_ao)
        _H_spin = np.repeat(_H_spin, 2, axis=0)
        _H_spin = np.repeat(_H_spin, 2, axis=1)
        # spin part of 1-e integrals
        spin_ind = np.arange(_H_spin.shape[0], dtype=int) % 2
        # product of spatial and spin parts
        self.Hspin = _H_spin * (spin_ind.reshape(-1, 1) == spin_ind)

    def build2DSO(self):
        """Will build the 2-electron arrays in the spin orbital basis
        that contribute to the A+\Delta blocks

        """
        self.TDI_spin = np.zeros((self.nso, self.nso, self.nso, self.nso))
        if self.ignore_coupling == False:
            # get the dipole-dipole integrals in the spin-orbital basis with physicist convention
            for i in range(self.nso):
                for j in range(self.nso):
                    for k in range(self.nso):
                        for l in range(self.nso):
                            self.TDI_spin[i, j, k, l] = map_spatial_dipole_to_spin(
                                self.d_cmo, i, j, k, l
                            )

        # add dipole-dipole integrals to ERIs
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

    def buildConstantMatrices(self):
        """
        Will build <G> * I, E_nuc * I, omega * I, and d_c * I
        """

        _I = np.identity(self.numDets)
        self.Enuc_so = self.Enuc * _I
        self.G_exp_so = np.sqrt(self.omega / 2) * self.d_exp * _I
        self.Omega_so = self.omega * _I
        self.dc_so = self.dc * _I
        if self.ignore_coupling == True:
            self.G_exp_so *= 0
            self.Omega_so *= 0
            self.dc_so *= 0

    def generateDeterminants(self, options_dict):
        """
        Generates the determinant list for building the CI matrix
        """
        self.dets = []
        self.detlists = []
        self.numDets = 0
        self.excitation_index = []
        if self.cas == False:
            docc_list = list(x for x in range(self.ndocc))
            print(docc_list)
            for alpha in combinations(range(self.nmo), self.ndocc):
                alpha_ex_level = compute_excitation_level(alpha, self.ndocc)
                for beta in combinations(range(self.nmo), self.ndocc):
                    beta_ex_level = compute_excitation_level(beta, self.ndocc)
                    if alpha_ex_level + beta_ex_level <= 1:
                        # print(F' adding alpha: {alpha} and beta: {beta}\n')
                        self.dets.append(
                            Determinant(alphaObtList=alpha, betaObtList=beta)
                        )
                        self.numDets += 1
                    if alpha_ex_level + beta_ex_level == 1:
                        alphalist = list(alpha)
                        betalist = list(beta)
                        if beta_ex_level == 1:
                            print("betalist")
                            print(betalist)
                            new_list = returnNotMatches(docc_list, betalist)
                            new_list = [x * 2 + 1 for x in new_list]
                        else:
                            print("alphalist")
                            print(alphalist)
                            new_list = returnNotMatches(docc_list, alphalist)
                            new_list = [x * 2 for x in new_list]

                        print(new_list)
                        self.excitation_index.append(
                            new_list[0] * 2 * (self.nmo - self.ndocc)
                            + new_list[1]
                            - 2 * self.ndocc
                        )
                        # print(alphalist)
                        # print(betalist)
                        jointlist = alphalist + betalist
                        self.detlists.append(jointlist)

        else:
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
                    self.dets.append(Determinant(alphaObtList=alpha, betaObtList=beta))
                    self.numDets += 1
        for i in range(len(self.dets)):
            print(self.dets[i])
        for i in range(len(self.detlists)):
            print(self.detlists[i])
        print(self.excitation_index)

    def generatePFHMatrix(self):
        """
        Generate H_PF CI Matrix
        """

        self.ApDmatrix = np.zeros((self.numDets, self.numDets))
        # one-electron only version of A+\Delta
        self.apdmatrix = np.zeros((self.numDets, self.numDets))

        self.Gmatrix = np.zeros((self.numDets, self.numDets))

        self.H_PF = np.zeros((2 * self.numDets, 2 * self.numDets))
        # one-electron version of Hamiltonian
        self.H_1E = np.zeros((2 * self.numDets, 2 * self.numDets))

        for i in range(self.numDets):
            for j in range(i + 1):
                self.ApDmatrix[i, j] = self.calcApDMatrixElement(
                    self.dets[i], self.dets[j]
                )
                self.apdmatrix[i, j] = self.calcApDMatrixElement(
                    self.dets[i], self.dets[j], OneEpTwoE=False
                )
                self.ApDmatrix[j, i] = self.ApDmatrix[i, j]
                self.apdmatrix[j, i] = self.apdmatrix[i, j]
                self.Gmatrix[i, j] = self.calcGMatrixElement(self.dets[i], self.dets[j])
                self.Gmatrix[j, i] = self.Gmatrix[i, j]

        # full hamiltonian
        self.H_PF[: self.numDets, : self.numDets] = (
            self.ApDmatrix + self.Enuc_so + self.dc_so
        )
        # 1-e piece
        self.H_1E[: self.numDets, : self.numDets] = (
            self.apdmatrix + self.Enuc_so + self.dc_so
        )

        # full Hamiltonian
        self.H_PF[self.numDets :, self.numDets :] = (
            self.ApDmatrix + self.Enuc_so + self.dc_so + self.Omega_so
        )

        # 1-e piece
        self.H_1E[self.numDets :, self.numDets :] = (
            self.apdmatrix + self.Enuc_so + self.dc_so + self.Omega_so
        )
        self.H_PF[self.numDets :, : self.numDets] = self.Gmatrix + self.G_exp_so
        self.H_PF[: self.numDets, self.numDets :] = self.Gmatrix + self.G_exp_so

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
                return 0.0
        elif det1.diff2OrLessOrbitals(det2):
            numUniqueOrbitals = det1.numberOfTotalDiffOrbitals(det2)
            if numUniqueOrbitals == 0:
                return self.calcMatrixElementIdentialDet(det1, omit2E=True)
            elif numUniqueOrbitals == 1:
                return self.calcMatrixElementDiffIn1(det1, det2, omit2E=True)
            else:
                return 0.0

        else:
            return 0.0

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
        return sign * self.antiSym2eInt[unique1[0], unique1[1], unique2[0], unique2[1]]

    def calcMatrixElementDiffIn1(self, det1, det2, omit2E=False):
        """
        Calculate a matrix element by two determinants where the determinants differ by 1 spin orbitals
        """

        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        Helem = self.Hspin[m, p]
        common = det1.getCommonOrbitalsInMixedSpinIndexList(det2)

        if omit2E == False:
            Relem = 0.0
            for n in common:
                Relem += self.antiSym2eInt[m, n, p, n]

        else:
            Relem = 0.0
        return sign * (Helem + Relem)

    def calcGMatrixElementDiffIn1(self, det1, det2):
        """
        Calculate a matrix element by two determinants where the determinants differ by 1 spin orbitals
        """

        unique1, unique2, sign = det1.getUniqueOrbitalsInMixIndexListsPlusSign(det2)
        m = unique1[0]
        p = unique2[0]
        Gelem = self.g_so[m, p]
        return sign * Gelem

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
        for m in spinObtList:
            Helem += self.Hspin[m, m]
        length = len(spinObtList)
        if omit2E == False:
            Relem = 0.0
            for m in range(length - 1):
                for n in range(m + 1, length):
                    Relem += self.antiSym2eInt[
                        spinObtList[m], spinObtList[n], spinObtList[m], spinObtList[n]
                    ]
        else:
            Relem = 0.0

        return Helem + Relem
