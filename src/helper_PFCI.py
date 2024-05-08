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
from memory_profiler import profile
from helper_cqed_rhf import cqed_rhf
from itertools import combinations
import math
import time
import ctypes
import numpy as np
from ctypes import *
import os
import psutil

script_dir = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(script_dir, "cfunctions.so")

# import shared lib
cfunctions = cdll.LoadLibrary(lib_path)

cfunctions.get_graph.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
]

cfunctions.index_to_string.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
]

cfunctions.index_to_string.restype = ctypes.c_size_t

cfunctions.get_string.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

cfunctions.build_sigma.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_bool,
]

cfunctions.get_roots.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
]

cfunctions.build_one_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.build_two_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.build_symmetrized_active_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.build_photon_electron_one_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.build_active_photon_electron_one_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]

cfunctions.build_sigma_s_square.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,
]

cfunctions.build_S_diag.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,
]


def c_string(
    h1e,
    h2e,
    H_diag,
    b_array,
    table,
    table_creation,
    table_annihilation,
    N_p,
    num_alpha,
    nmo,
    N,
    n_o,
    n_in_a,
    omega,
    Enuc,
    dc,
):
    cfunctions.get_string(
        h1e,
        h2e,
        H_diag,
        b_array,
        table,
        table_creation,
        table_annihilation,
        N_p,
        num_alpha,
        nmo,
        N,
        n_o,
        n_in_a,
        omega,
        Enuc,
        dc,
    )


def c_index_to_string(index, n_act_a, n_act_orb, Y):
    string = cfunctions.index_to_string(index, n_act_a, n_act_orb, Y)
    return string


def c_graph(N, n_o, Y):
    cfunctions.get_graph(N, n_o, Y)


def c_sigma(
    h1e,
    h2e,
    d_cmo,
    c_vectors,
    s_vectors,
    table,
    table_creation,
    table_annihilation,
    N_ac,
    n_o_ac,
    n_o_in,
    nmo,
    num_state,
    N_p,
    Enuc,
    dc,
    omega,
    d_exp,
    E_core,
    break_degeneracy,
):
    cfunctions.build_sigma(
        h1e,
        h2e,
        d_cmo,
        c_vectors,
        s_vectors,
        table,
        table_creation,
        table_annihilation,
        N_ac,
        n_o_ac,
        n_o_in,
        nmo,
        num_state,
        N_p,
        Enuc,
        dc,
        omega,
        d_exp,
        E_core,
        break_degeneracy,
    )


def c_sigma_s_square(
    c_vectors,
    c1_vectors,
    S_diag,
    b_array,
    table,
    num_links,
    n_o_ac,
    num_alpha,
    num_state,
    N_p,
    scale,
):
    cfunctions.build_sigma_s_square(
        c_vectors,
        c1_vectors,
        S_diag,
        b_array,
        table,
        num_links,
        n_o_ac,
        num_alpha,
        num_state,
        N_p,
        scale,
    )


def c_s_diag(S_diag, num_alpha, nmo, N, n_o, n_in_a, shift):
    cfunctions.build_S_diag(S_diag, num_alpha, nmo, N, n_o, n_in_a, shift)


def c_get_roots(
    h1e,
    h2e,
    d_cmo,
    Hdiag,
    eigenvals,
    eigenvecs,
    table,
    table_creation,
    table_annihilation,
    constint,
    constdouble,
):
    cfunctions.get_roots(
        h1e,
        h2e,
        d_cmo,
        Hdiag,
        eigenvals,
        eigenvecs,
        table,
        table_creation,
        table_annihilation,
        constint,
        constdouble,
    )


def c_build_one_rdm(
    eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
):
    cfunctions.build_one_rdm(
        eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
    )


def c_build_two_rdm(
    eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
):
    cfunctions.build_two_rdm(
        eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
    )


def c_build_symmetrized_active_rdm(
    eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2
):
    cfunctions.build_symmetrized_active_rdm(
        eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2
    )


def c_build_photon_electron_one_rdm(
    eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
):
    cfunctions.build_photon_electron_one_rdm(
        eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
    )


def c_build_active_photon_electron_one_rdm(
    eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2
):
    cfunctions.build_active_photon_electron_one_rdm(
        eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2
    )


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


class PFHamiltonianGenerator:
    """
    class for Full CI matrix elements
    """

    def __init__(self, molecule_string, psi4_options_dict, cavity_options):
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
        np1 = self.N_p + 1

        # build the determinant list (for full diagonalization) or determinant tables (for direct) based on CI type

        # cis - currently only full diagonalization available; build all QED-CIS determinants
        if self.ci_level == "cis":
            self.generateCISDeterminants()
            H_dim = self.CISnumDets * np1
            self.H_diag = np.zeros((H_dim))

        # cas - can do full diagonalization of direct
        elif self.ci_level == "cas":
            if self.test_mode:
                # Build all QED-CAS determinants
                self.generateCASCIDeterminants()
                H_dim = self.CASnumDets * np1
                self.H_diag2 = np.zeros((H_dim))

            if self.full_diagonalization:
                # Build all QED-CAS determinants
                self.generateCASCIDeterminants()
                H_dim = self.CASnumDets * np1
                self.H_diag = np.zeros((H_dim))

            # direct is default
            else:
                # build determinant tables but not all determinants
                self.n_act_a = self.n_act_el // 2  # number of active alpha electrons
                self.n_in_a = (
                    self.ndocc - self.n_act_a
                )  # number of inactive alpha electrons
                self.num_alpha = math.comb(
                    self.n_act_orb, self.n_act_a
                )  # number of alpha strings
                self.num_det = self.num_alpha * self.num_alpha  # number of determinants
                self.CASnumDets = self.num_det
                H_dim = self.CASnumDets * np1
                self.H_diag = np.zeros(H_dim)

                # build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                for k in range(self.nmo):
                    for l in range(self.nmo):
                        kl = k * self.nmo + l
                        self.fock_core[k][l] = self.H_spatial2[k][l]
                        for j in range(self.n_in_a):
                            jj = j * self.nmo + j
                            kj = k * self.nmo + j
                            jl = j * self.nmo + l
                            self.fock_core[k][l] += (
                                2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]
                            )
                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                for k in range(self.n_act_orb):
                    for l in range(self.n_act_orb):
                        self.gkl2[k][l] = self.fock_core[k + self.n_in_a][
                            l + self.n_in_a
                        ]
                        for j in range(self.n_act_orb):
                            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]

                self.E_core = 0.0
                for i in range(self.n_in_a):
                    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i]
                self.table = np.zeros(
                    self.num_alpha
                    * (self.n_act_a * (self.n_act_orb - self.n_act_a) + self.n_act_a)
                    * 4,
                    dtype=np.int32,
                )
                num_links1 = self.n_act_orb - self.n_act_a + 1
                rows1 = math.comb(self.n_act_orb, self.n_act_a - 1) * num_links1
                self.table_creation = np.zeros(rows1 * 3, dtype=np.int32)
                num_links2 = self.n_act_a
                rows2 = self.num_alpha * num_links2
                self.table_annihilation = np.zeros(rows2 * 3, dtype=np.int32)

                self.b_array = np.zeros(
                    self.num_alpha * self.n_act_orb * self.n_act_orb * 2, dtype=np.int32
                )
                c_string(
                    self.H_spatial2,
                    self.twoeint,
                    self.H_diag,
                    self.b_array,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.N_p,
                    self.num_alpha,
                    self.nmo,
                    self.n_act_a,
                    self.n_act_orb,
                    self.n_in_a,
                    self.omega,
                    self.Enuc,
                    self.d_c,
                )
                self.S_diag = np.zeros(H_dim)
                shift = 0.0
                c_s_diag(
                    self.S_diag,
                    self.num_alpha,
                    self.nmo,
                    self.n_act_a,
                    self.n_act_orb,
                    self.n_in_a,
                    shift,
                )
                num_alpha1 = math.comb(self.n_act_orb, self.n_act_a - 1)
                print(
                    "mem_D+T",
                    self.num_alpha * num_alpha1 * self.n_act_orb * 8 * 2 / 1024 / 1024,
                )  # intermediate in sigma3

        # fci - can do full diagonalization or direct
        elif self.ci_level == "fci":
            if self.test_mode:
                # build all QED-FCI determiants
                self.generateFCIDeterminants()
                H_dim = self.FCInumDets * np1
                self.H_diag2 = np.zeros((H_dim))

            if self.full_diagonalization:
                # build all QED-FCI determiants
                self.generateFCIDeterminants()
                H_dim = self.FCInumDets * np1
                self.H_diag = np.zeros((H_dim))

            # default is direct
            else:
                # build determinant tables but not all determinants
                self.n_act_a = self.ndocc  # number of active alpha electrons
                self.n_in_a = 0  # number of inactive alpha electrons
                self.n_act_orb = self.nmo  # number of active alpha orbitals

                self.num_alpha = math.comb(
                    self.nmo, self.ndocc
                )  # number of alpha strings
                self.num_det = self.num_alpha * self.num_alpha  # number of determinants
                self.FCInumDets = self.num_det
                H_dim = self.FCInumDets * np1
                self.H_diag = np.zeros(H_dim)

                # build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                for k in range(self.nmo):
                    for l in range(self.nmo):
                        kl = k * self.nmo + l
                        self.fock_core[k][l] = self.H_spatial2[k][l]
                        for j in range(self.n_in_a):
                            jj = j * self.nmo + j
                            kj = k * self.nmo + j
                            jl = j * self.nmo + l
                            self.fock_core[k][l] += (
                                2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]
                            )
                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                for k in range(self.n_act_orb):
                    for l in range(self.n_act_orb):
                        self.gkl2[k][l] = self.fock_core[k + self.n_in_a][
                            l + self.n_in_a
                        ]
                        for j in range(self.n_act_orb):
                            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]

                self.E_core = 0.0
                for i in range(self.n_in_a):
                    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i]
                self.table = np.zeros(
                    self.num_alpha
                    * (self.n_act_a * (self.n_act_orb - self.n_act_a) + self.n_act_a)
                    * 4,
                    dtype=np.int32,
                )
                num_links1 = self.n_act_orb - self.n_act_a + 1
                rows1 = math.comb(self.n_act_orb, self.n_act_a - 1) * num_links1
                self.table_creation = np.zeros(rows1 * 3, dtype=np.int32)
                num_links2 = self.n_act_a
                rows2 = self.num_alpha * num_links2
                self.table_annihilation = np.zeros(rows2 * 3, dtype=np.int32)

                self.b_array = np.zeros(
                    self.num_alpha * self.n_act_orb * self.n_act_orb * 2, dtype=np.int32
                )
                c_string(
                    self.H_spatial2,
                    self.twoeint,
                    self.H_diag,
                    self.b_array,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.N_p,
                    self.num_alpha,
                    self.nmo,
                    self.n_act_a,
                    self.n_act_orb,
                    self.n_in_a,
                    self.omega,
                    self.Enuc,
                    self.d_c,
                )

                self.S_diag = np.zeros(H_dim)
                shift = 0.0
                c_s_diag(
                    self.S_diag,
                    self.num_alpha,
                    self.nmo,
                    self.n_act_a,
                    self.n_act_orb,
                    self.n_in_a,
                    shift,
                )

                num_alpha1 = math.comb(self.n_act_orb, self.n_act_a - 1)
                print(
                    "mem_D+T",
                    self.num_alpha * num_alpha1 * self.n_act_orb * 8 * 2 / 1024 / 1024,
                )  # size of intermediate in sigma3

        t_det_end = time.time()
        print(
            f" Completed determinant list in {t_det_end - t_det_start} seconds ",
            flush=True,
        )

        # if doing full-diagonalization, next step will be to build full Hamiltonian matrix
        if self.ci_level == "cis":
            # build Constant matrices
            self.buildConstantMatrices(self.ci_level)
            t_const_end = time.time()
            print(
                f" Completed constant offset matrix in {t_const_end - t_det_end} seconds"
            )

            # Build Matrix
            self.generatePFHMatrix(self.ci_level)
            t_H_build = time.time()
            print(f" Completed Hamiltonian build in {t_H_build - t_const_end} seconds")
        else:
            if self.test_mode:
                # build Constant matrices
                self.buildConstantMatrices(self.ci_level)
                t_const_end = time.time()
                print(
                    f" Completed constant offset matrix in {t_const_end - t_det_end} seconds"
                )

                # Build full Hamiltonian Matrix
                self.generatePFHMatrix(self.ci_level)
                t_H_build = time.time()
                print(
                    f" Completed Hamiltonian build in {t_H_build - t_const_end} seconds"
                )
                self.H_diag2 = np.diag(self.H_PF)

            if self.full_diagonalization:
                # build Constant matrices
                self.buildConstantMatrices(self.ci_level)
                t_const_end = time.time()
                print(
                    f" Completed constant offset matrix in {t_const_end - t_det_end} seconds"
                )

                # Build Matrix
                self.generatePFHMatrix(self.ci_level)
                t_H_build = time.time()
                print(
                    f" Completed Hamiltonian build in {t_H_build - t_const_end} seconds"
                )
            else:
                # if using iterative solver, pass a small matrix to the solver that doesn't take up too much memory
                self.H_PF = np.eye(2)

        if self.test_mode:
            # call full diagonalization
            self.CIeigs, self.CIvecs = np.linalg.eigh(self.H_PF)

        if self.full_diagonalization:
            # call full diagonalization
            self.CIeigs, self.CIvecs = np.linalg.eigh(self.H_PF)

        # if doing direct method, call Davidson routine
        else:
            indim = self.davidson_indim * self.davidson_roots
            maxdim = self.davidson_maxdim * self.davidson_roots
            # print(H_dim, self.n_act_a,self.nmo)
            if indim > H_dim or maxdim > H_dim:
                print(
                    "subspace size is too large, try to set maxdim and indim <",
                    H_dim // self.davidson_roots,
                )
                sys.exit()
            t_H_build = time.time()
            print(
                "memory required for sigma and CI vectors",
                maxdim * H_dim * 8 * 2 / 1024 / 1024,
                flush=True,
            )

            print(psutil.Process().memory_info().rss / (1024 * 1024))
            if self.ci_level == "cas" or self.ci_level == "fci":
                sys.stderr.flush()
                d_diag = 0.0
                for i in range(self.n_in_a):
                    d_diag += 2.0 * self.d_cmo[i][i]
                # print(d_diag, self.d_exp, self.N_p, self.n_act_orb, self.nmo, self.omega, self.num_alpha)
                self.constint = np.zeros(9, dtype=np.int32)
                self.constint[0] = self.n_act_a
                self.constint[1] = self.n_act_orb
                self.constint[2] = self.n_in_a
                self.constint[3] = self.nmo
                self.constint[4] = self.N_p
                self.constint[5] = indim
                self.constint[6] = maxdim
                self.constint[7] = self.davidson_roots
                self.constint[8] = self.davidson_maxiter
                self.constdouble = np.zeros(6)
                self.constdouble[0] = self.Enuc
                if self.ignore_dse_terms:
                    self.constdouble[1] = 0.0
                else:
                    self.constdouble[1] = self.d_c
                self.constdouble[2] = self.omega
                self.constdouble[3] = self.d_exp - d_diag
                self.constdouble[4] = self.davidson_threshold
                self.constdouble[5] = self.E_core
                eigenvals = np.zeros((self.davidson_roots))
                eigenvecs = np.zeros((self.davidson_roots, H_dim))
                # dres = self.Davidson(self.H_PF, self.davidson_roots, self.davidson_threshold, indim, maxdim,self.davidson_maxiter,self.build_sigma,self.H_diag)
                c_get_roots(
                    self.gkl2,
                    self.twoeint,
                    self.d_cmo,
                    self.H_diag,
                    eigenvals,
                    eigenvecs,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.constint,
                    self.constdouble,
                )
                self.CIeigs = eigenvals
                self.CIvecs = eigenvecs
                self.CISingletEigs = np.zeros_like(self.CIeigs)
                self.CITripletEigs = np.zeros_like(self.CIeigs)

                print(
                    "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
                )
                Y = np.zeros(
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                    dtype=np.int32,
                )
                c_graph(self.n_act_a, self.n_act_orb, Y)
                np1 = self.N_p + 1
                self.singlet_count = 0
                self.triplet_count = 0
                self.singlet_indices = []
                for i in range(eigenvecs.shape[0]):
                    total_spin = self.check_total_spin(eigenvecs[i : (i + 1), :])
                    print(
                        "state",
                        i,
                        "energy =",
                        eigenvals[i],
                        "<S^2>=",
                        total_spin,
                        end="",
                    )
                    if np.abs(total_spin) < 1e-5:
                        self.CISingletEigs[self.singlet_count] = eigenvals[i]
                        self.singlet_count += 1
                        self.singlet_indices.append(i)
                        print("\tsinglet", self.singlet_count)
                    elif np.abs(total_spin - 2.0) < 1e-5:
                        self.CITripletEigs[self.triplet_count] = eigenvals[i]
                        self.triplet_count += 1
                        print("\ttriplet", self.triplet_count)
                    elif np.abs(total_spin - 6.0) < 1e-5:
                        print("\tquintet")

                    # print("state",i, "energy =",theta[i])
                    print(
                        "        amplitude",
                        "      position",
                        "         most important determinants",
                        "             number of photon",
                    )
                    index = np.argsort(np.abs(eigenvecs[i, :]))
                    # print(index)
                    Idet0 = (
                        index[eigenvecs.shape[1] - 1] % self.num_det
                    )  # determinant index of most significant contribution
                    photon_p0 = (
                        index[eigenvecs.shape[1] - 1] - Idet0
                    ) // self.num_det  # photon number block of determinant
                    Ib0 = Idet0 % self.num_alpha
                    Ia0 = Idet0 // self.num_alpha
                    a0 = c_index_to_string(Ia0, self.n_act_a, self.n_act_orb, Y)
                    b0 = c_index_to_string(Ib0, self.n_act_a, self.n_act_orb, Y)

                    alphalist = Determinant.obtBits2ObtIndexList(a0)
                    betalist = Determinant.obtBits2ObtIndexList(b0)
                    for j in range(min(H_dim, 10)):
                        Idet = index[eigenvecs.shape[1] - j - 1] % self.num_det
                        photon_p = (
                            index[eigenvecs.shape[1] - j - 1] - Idet
                        ) // self.num_det
                        Ib = Idet % self.num_alpha
                        Ia = Idet // self.num_alpha
                        a = c_index_to_string(Ia, self.n_act_a, self.n_act_orb, Y)
                        b = c_index_to_string(Ib, self.n_act_a, self.n_act_orb, Y)

                        alphalist = Determinant.obtBits2ObtIndexList(a)
                        betalist = Determinant.obtBits2ObtIndexList(b)

                        inactive_list = list(x for x in range(self.n_in_a))
                        alphalist2 = [x + self.n_in_a for x in alphalist]
                        # alphalist2[0:0] = inactive_list
                        betalist2 = [x + self.n_in_a for x in betalist]
                        # betalist2[0:0] = inactive_list

                        print(
                            "%20.12lf"
                            % (eigenvecs[i][index[eigenvecs.shape[1] - j - 1]]),
                            "%9.3d" % (index[eigenvecs.shape[1] - j - 1]),
                            "alpha",
                            alphalist2,
                            "   beta",
                            betalist2,
                            "%4.1d" % (photon_p),
                            "photon",
                        )

                if self.compute_properties:
                    print(" GOING TO COMPUTE 1-E PROPERTIES!")
                    self.n_occupied = self.n_act_orb + self.n_in_a
                    _mu_x_spin = np.einsum(
                        "uj,vi,uv",
                        self.C[:, : self.n_occupied],
                        self.C[:, : self.n_occupied],
                        self.mu_x_ao,
                    )
                    _mu_y_spin = np.einsum(
                        "uj,vi,uv",
                        self.C[:, : self.n_occupied],
                        self.C[:, : self.n_occupied],
                        self.mu_y_ao,
                    )
                    _mu_z_spin = np.einsum(
                        "uj,vi,uv",
                        self.C[:, : self.n_occupied],
                        self.C[:, : self.n_occupied],
                        self.mu_z_ao,
                    )
                    _mu_x_spin = np.ascontiguousarray(_mu_x_spin)
                    _mu_y_spin = np.ascontiguousarray(_mu_y_spin)
                    _mu_z_spin = np.ascontiguousarray(_mu_z_spin)

                    # store dipole moments as attributes
                    # total dipole moments, mu_el + mu_nuc
                    self.dipole_array = np.zeros(
                        (self.davidson_roots, self.davidson_roots, 3)
                    )
                    self.singlet_dipole_array = np.zeros(
                        (self.singlet_count, self.singlet_count, 3)
                    )

                    # only electronic contribution
                    self.electronic_dipole_array = np.zeros_like(self.dipole_array)

                    # only nuclear contribution
                    self.nuclear_dipole_array = np.zeros_like(self.dipole_array)

                    self.nuclear_dipole_array[:, :, 0] = (
                        np.eye(self.davidson_roots) * self.nuclear_dipole_moment[0]
                    )
                    self.nuclear_dipole_array[:, :, 1] = (
                        np.eye(self.davidson_roots) * self.nuclear_dipole_moment[1]
                    )
                    self.nuclear_dipole_array[:, :, 2] = (
                        np.eye(self.davidson_roots) * self.nuclear_dipole_moment[2]
                    )
                    self.nat_obt_number = np.zeros(
                        (self.davidson_roots, self.n_occupied)
                    )

                    print(
                        "{:^15s}".format(" "),
                        "{:^20s}".format("dipole x"),
                        "{:^20s}".format("dipole y"),
                        "{:^20s}".format("dipole z"),
                    )
                    for i in range(self.davidson_roots):
                        for j in range(i, self.davidson_roots):
                            one_rdm = np.zeros((self.n_occupied * self.n_occupied))
                            c_build_one_rdm(
                                eigenvecs,
                                one_rdm,
                                self.table,
                                self.n_act_a,
                                self.n_act_orb,
                                self.n_in_a,
                                np1,
                                i,
                                j,
                            )
                            dipole_x = np.dot(_mu_x_spin.flatten(), one_rdm)
                            dipole_y = np.dot(_mu_y_spin.flatten(), one_rdm)
                            dipole_z = np.dot(_mu_z_spin.flatten(), one_rdm)
                            # dipole_x = c_one_electron_properties(_mu_x_spin, eigenvecs, rdm_eig, self.table, self.n_act_a, self.n_act_orb, self.n_in_a, self.nmo, np1, i, j)
                            print(
                                "{:4d}".format(i),
                                "->",
                                "{:4d}".format(j),
                                "{:20.12f}".format(dipole_x),
                                "{:20.12f}".format(dipole_y),
                                "{:20.12f}".format(dipole_z),
                            )
                            if i == j:
                                one_rdm = np.reshape(
                                    one_rdm, (self.n_occupied, self.n_occupied)
                                )
                                rdm_eig = np.linalg.eigvalsh(one_rdm)
                                self.nat_obt_number[i, :] = rdm_eig[
                                    np.argsort(-rdm_eig)
                                ][:]
                            self.electronic_dipole_array[i, j, 0] = dipole_x
                            self.electronic_dipole_array[i, j, 1] = dipole_y
                            self.electronic_dipole_array[i, j, 2] = dipole_z

                    # combine nuclear and electronic parts for the total dipole array
                    self.dipole_array = (
                        self.electronic_dipole_array + self.nuclear_dipole_array
                    )
                    self.singlet_dipole_array = self.dipole_array[
                        self.singlet_indices, self.singlet_indices, :
                    ]
                    # print(self.nat_obt_number)
                ###check total energy
                if self.check_rdms:
                    print("check total energy using full rdms")
                    twoeint2 = self.twoeint.reshape(
                        (self.nmo, self.nmo, self.nmo, self.nmo)
                    )
                    twoeint2 = twoeint2[
                        : self.n_occupied,
                        : self.n_occupied,
                        : self.n_occupied,
                        : self.n_occupied,
                    ]
                    for i in range(self.davidson_roots):
                        sum_energy = 0.0
                        off_diagonal_constant_energy = 0.0
                        photon_energy = 0.0
                        eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
                        eigenvecs2 = eigenvecs2.transpose(1, 0)
                        for m in range(np1):
                            if self.N_p == 0:
                                continue
                            if m > 0 and m < self.N_p:
                                off_diagonal_constant_energy += (
                                    np.sqrt(m * self.omega / 2)
                                    * self.d_exp
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m - 1) : m].flatten(),
                                    )
                                )
                                off_diagonal_constant_energy += (
                                    np.sqrt((m + 1) * self.omega / 2)
                                    * self.d_exp
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                                    )
                                )
                            elif m == self.N_p:
                                off_diagonal_constant_energy += (
                                    np.sqrt(m * self.omega / 2)
                                    * self.d_exp
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m - 1) : m].flatten(),
                                    )
                                )
                            else:
                                off_diagonal_constant_energy += (
                                    np.sqrt((m + 1) * self.omega / 2)
                                    * self.d_exp
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                                    )
                                )
                            photon_energy += (
                                m
                                * self.omega
                                * np.dot(
                                    eigenvecs2[:, m : (m + 1)].flatten(),
                                    eigenvecs2[:, m : (m + 1)].flatten(),
                                )
                            )

                        one_rdm = np.zeros((self.n_occupied * self.n_occupied))
                        c_build_one_rdm(
                            eigenvecs,
                            one_rdm,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            self.n_in_a,
                            np1,
                            i,
                            i,
                        )
                        two_rdm = np.zeros(
                            (
                                self.n_occupied
                                * self.n_occupied
                                * self.n_occupied
                                * self.n_occupied
                            )
                        )
                        c_build_two_rdm(
                            eigenvecs,
                            two_rdm,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            self.n_in_a,
                            np1,
                            i,
                            i,
                        )
                        Dpe = np.zeros((self.n_occupied * self.n_occupied))
                        c_build_photon_electron_one_rdm(
                            eigenvecs,
                            Dpe,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            self.n_in_a,
                            np1,
                            i,
                            i,
                        )

                        # two_rdm2 = two_rdm.reshape((self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied))
                        # print(two_rdm2[:(self.n_in_a*self.n_in_a),:(self.n_in_a*self.n_in_a)])
                        # np.savetxt('correct_rdm.txt', two_rdm2)

                        # one_rdm2 = np.zeros((self.nmo * self.nmo))
                        # for p in range(self.nmo):
                        #    for q in range(self.nmo):
                        #        dum = 0.0
                        #        for r in range(self.nmo):
                        #            dum += 0.5/(self.n_act_a+self.n_in_a-0.5) * two_rdm[p * self.nmo * self.nmo * self.nmo + r * self.nmo * self.nmo + q * self.nmo + r]
                        #        one_rdm2[p * self.nmo + q] = dum

                        one_e_energy = np.dot(
                            self.H_spatial2[
                                : self.n_occupied, : self.n_occupied
                            ].flatten(),
                            one_rdm,
                        )
                        two_e_energy = 0.5 * np.dot(twoeint2.flatten(), two_rdm)
                        one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                            self.d_cmo[: self.n_occupied, : self.n_occupied].flatten(),
                            Dpe,
                        )
                        sum_energy = (
                            one_e_energy
                            + two_e_energy
                            + self.Enuc
                            + one_pe_energy
                            + off_diagonal_constant_energy
                            + self.d_c
                            + photon_energy
                        )

                        # store the RDMs as a self attribute if the current state matches the rdm root
                        if self.rdm_root == i:
                            self.one_electron_rdm = np.copy(one_rdm)
                            self.one_electron_one_photon_rdm = np.copy(Dpe)
                            self.two_electron_rdm = np.copy(two_rdm)
                            self.total_energy_from_rdms = sum_energy
                        print(
                            "{:4d}".format(i),
                            "{:20.12f}".format(eigenvals[i]),
                            "{:20.12f}".format(sum_energy),
                            "{:20.12f}".format(eigenvals[i] - sum_energy),
                        )

                    print("check total energy using active rdms")
                    print(
                        "{:10s}".format("state"),
                        "{:20s}".format("total energies"),
                        "{:20s}".format("eigenvalues"),
                        "error",
                    )
                    active_twoeint = twoeint2[
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                    ]
                    active_fock_core = self.fock_core[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ]
                    for i in range(self.davidson_roots):
                        self.D_tu = np.zeros((self.n_act_orb * self.n_act_orb))
                        self.Dpe_tu = np.zeros((self.n_act_orb * self.n_act_orb))
                        self.D_tuvw = np.zeros(
                            (
                                self.n_act_orb
                                * self.n_act_orb
                                * self.n_act_orb
                                * self.n_act_orb
                            )
                        )
                        sum_energy = 0.0
                        off_diagonal_constant_energy = 0.0
                        photon_energy = 0.0
                        eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
                        eigenvecs2 = eigenvecs2.transpose(1, 0)
                        for m in range(np1):
                            if self.N_p == 0:
                                continue
                            if m > 0 and m < self.N_p:
                                off_diagonal_constant_energy += (
                                    np.sqrt(m * self.omega / 2)
                                    * (self.d_exp - d_diag)
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m - 1) : m].flatten(),
                                    )
                                )
                                off_diagonal_constant_energy += (
                                    np.sqrt((m + 1) * self.omega / 2)
                                    * (self.d_exp - d_diag)
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                                    )
                                )
                            elif m == self.N_p:
                                off_diagonal_constant_energy += (
                                    np.sqrt(m * self.omega / 2)
                                    * (self.d_exp - d_diag)
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m - 1) : m].flatten(),
                                    )
                                )
                            else:
                                off_diagonal_constant_energy += (
                                    np.sqrt((m + 1) * self.omega / 2)
                                    * (self.d_exp - d_diag)
                                    * np.dot(
                                        eigenvecs2[:, m : (m + 1)].flatten(),
                                        eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                                    )
                                )
                            photon_energy += (
                                m
                                * self.omega
                                * np.dot(
                                    eigenvecs2[:, m : (m + 1)].flatten(),
                                    eigenvecs2[:, m : (m + 1)].flatten(),
                                )
                            )
                        c_build_symmetrized_active_rdm(
                            eigenvecs,
                            self.D_tu,
                            self.D_tuvw,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            np1,
                            i,
                            i,
                        )
                        c_build_active_photon_electron_one_rdm(
                            eigenvecs,
                            self.Dpe_tu,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            np1,
                            i,
                            i,
                        )
                        active_one_e_energy = np.dot(
                            active_fock_core.flatten(), self.D_tu
                        )
                        active_two_e_energy = 0.5 * np.dot(
                            active_twoeint.flatten(), self.D_tuvw
                        )
                        active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                            self.d_cmo[
                                self.n_in_a : self.n_occupied,
                                self.n_in_a : self.n_occupied,
                            ].flatten(),
                            self.Dpe_tu,
                        )
                        sum_energy = (
                            active_one_e_energy
                            + active_two_e_energy
                            + active_one_pe_energy
                            + self.E_core
                            + self.Enuc
                            + self.d_c
                            + off_diagonal_constant_energy
                            + photon_energy
                        )

                        print(
                            "{:4d}".format(i),
                            "{:20.12f}".format(eigenvals[i]),
                            "{:20.12f}".format(sum_energy),
                            "{:20.12f}".format(eigenvals[i] - sum_energy),
                        )

                    # self.gkl3 = np.zeros((self.n_act_orb, self.n_act_orb))
                    # self.twoeint3 = np.zeros((self.nmo * self.nmo, self.nmo * self.nmo))
                    #
                    # S = np.zeros((self.davidson_roots, H_dim))
                    # c_sigma(
                    #    self.gkl2,
                    #    self.twoeint,
                    #    self.d_cmo,
                    #    eigenvecs,
                    #    S,
                    #    self.table,
                    #    self.table_creation,
                    #    self.table_annihilation,
                    #    self.n_act_a,
                    #    self.n_act_orb,
                    #    self.n_in_a,
                    #    self.nmo,
                    #    self.davidson_roots,
                    #    self.N_p,
                    #    self.Enuc-self.Enuc,
                    #    self.d_c-self.d_c,
                    #    self.omega-self.omega,
                    #    self.d_exp-d_diag,
                    #    self.E_core-self.E_core,
                    #    self.break_degeneracy,
                    # )
                    # eig_mat = np.einsum("pq,rq",S, eigenvecs)
                    # np.set_printoptions(precision=12)
                    # print(np.diag(eig_mat))
            elif self.ci_level == "cis":
                dres = self.Davidson(
                    self.H_PF,
                    self.davidson_roots,
                    self.davidson_threshold,
                    indim,
                    maxdim,
                    self.davidson_maxiter,
                    self.build_sigma,
                    self.H_diag,
                )
                self.CIeigs = dres["DAVIDSON EIGENVALUES"]
                self.CIvecs = dres["DAVIDSON EIGENVECTORS"]

            t_dav_end = time.time()
            print(f" Completed Davidson iterations in {t_dav_end - t_H_build} seconds")

    def parseCavityOptions(self, cavity_dictionary):
        """
        Parse the cavity dictionary for important parameters for the QED-CI calculation
        """
        if "check_rdms" in cavity_dictionary:
            self.check_rdms = cavity_dictionary["check_rdms"]
        else:
            self.check_rdms = False

        if "compute_properties" in cavity_dictionary:
            self.compute_properties = cavity_dictionary["compute_properties"]
        else:
            self.compute_properties = False

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

        # this will set options for performing in the coherent state or photon number basis.
        # For consistency, canonical mos and photon number basis must go together.

        # default is coherent state basis
        if "canonical_mos" in cavity_dictionary:
            self.canonical_mos = cavity_dictionary["canonical_mos"]
        else:
            self.canonical_mos = False

        if "photon_number_basis" in cavity_dictionary:
            self.photon_number_basis = cavity_dictionary["photon_number_basis"]
        else:
            self.photon_number_basis = False

        if "coherent_state_basis" in cavity_dictionary:
            self.coherent_state_basis = cavity_dictionary["coherent_state_basis"]
        elif self.canonical_mos or self.photon_number_basis:
            self.coherent_state_basis = False
        else:
            self.coherent_state_basis = True

        # if we set either canonical mo or photon number basis, make sure
        # all options align
        if self.canonical_mos or self.photon_number_basis:
            self.canonical_mos = True
            self.photon_number_basis = True
            self.coherent_state_basis = False

        # if we set coherent state basis, make sure all options align
        if self.coherent_state_basis:
            self.canonical_mos = False
            self.photon_number_basis = False

        if "ignore_dse_terms" in cavity_dictionary:
            self.ignore_dse_terms = cavity_dictionary["ignore_dse_terms"]
        else:
            self.ignore_dse_terms = False

        if self.ignore_dse_terms:
            self.ignore_dse_terms = True
            self.photon_number_basis = True
            self.coherent_state_basis = False

        if self.natural_orbitals:
            if "rdm_weights" in cavity_dictionary:
                self.rdm_weights = cavity_dictionary["rdm_weights"]
            else:
                self.rdm_weights = np.array([1, 1])

        if "full_diagonalization" in cavity_dictionary:
            self.full_diagonalization = cavity_dictionary["full_diagonalization"]
        else:
            self.full_diagonalization = False

        if "test_mode" in cavity_dictionary:
            self.test_mode = cavity_dictionary["test_mode"]
        else:
            self.test_mode = False

        if "break_degeneracy" in cavity_dictionary:
            self.break_degeneracy = cavity_dictionary["break_degeneracy"]
        else:
            self.break_degeneracy = False

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
        # rdm will be stored for this state
        if "rdm_root" in cavity_dictionary:
            self.rdm_root = cavity_dictionary["rdm_root"]
        else:
            self.rdm_root = 0

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
        # grab quantities from cqed_rhf_dict that apply to both number state and coherent state bases
        wfn = cqed_rhf_dict["PSI4 WFN"]
        self.rhf_reference_energy = cqed_rhf_dict["RHF ENERGY"]
        self.Enuc = cqed_rhf_dict["NUCLEAR REPULSION ENERGY"]
        self.nuclear_dipole_moment = cqed_rhf_dict["NUCLEAR DIPOLE MOMENT"]
        self.mu_x_ao = cqed_rhf_dict["DIPOLE AO X"]
        self.mu_y_ao = cqed_rhf_dict["DIPOLE AO Y"]
        self.mu_z_ao = cqed_rhf_dict["DIPOLE AO Z"]
        self.T_ao = cqed_rhf_dict["1-E KINETIC MATRIX AO"]
        self.V_ao = cqed_rhf_dict["1-E POTENTIAL MATRIX AO"]
        self.q_PF_ao = cqed_rhf_dict["1-E QUADRUPOLE MATRIX AO"]
        self.d_ao = cqed_rhf_dict["1-E DIPOLE MATRIX AO"]
        self.d_cmo = cqed_rhf_dict["1-E DIPOLE MATRIX MO"]

        if self.photon_number_basis:
            self.d_c = cqed_rhf_dict["NUMBER STATE NUCLEAR DIPOLE ENERGY"]
            self.d_PF_ao = cqed_rhf_dict["NUMBER STATE 1-E SCALED DIPOLE MATRIX AO"]
            self.d_exp = -1 * cqed_rhf_dict["NUMBER STATE NUCLEAR DIPOLE TERM"]
            self.C = cqed_rhf_dict["RHF C"]
            self.D = cqed_rhf_dict["RHF DENSITY MATRIX"]

        else:
            self.d_c = cqed_rhf_dict["COHERENT STATE DIPOLE ENERGY"]
            self.d_exp = cqed_rhf_dict["COHERENT STATE EXPECTATION VALUE OF d"]
            self.d_PF_ao = cqed_rhf_dict["COHERENT STATE 1-E SCALED DIPOLE MATRIX AO"]
            self.C = cqed_rhf_dict["CQED-RHF C"]
            self.D = cqed_rhf_dict["CQED-RHF DENSITY MATRIX"]
            self.cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
            self.cqed_one_energy = cqed_rhf_dict["CQED-RHF ONE-ENERGY"]

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
        """Will build the 1-electron dipole arrays in the spin orbital
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

        # Standard 1-e integrals, kinetic and electron-nuclear attraction
        self.H_1e_ao = self.T_ao + self.V_ao

        if self.ignore_coupling == False and self.ignore_dse_terms == False:
            # cavity-specific 1-e integrals, including quadrupole and 1-e dipole integrals
            # note that d_PF_ao is <d_e> \hat{d}_e in the coherent state basis
            # and       d_PF_ao is d_nuc \hat{d} in the photon number basis
            # this assignment is taken care of in the parseArrays() method
            self.H_1e_ao += self.q_PF_ao + self.d_PF_ao

        # build H_spin
        # spatial part of 1-e integrals
        _H_spin = np.einsum("uj,vi,uv", self.C, self.C, self.H_1e_ao)
        self.H_spatial = _H_spin
        self.H_spatial2 = np.ascontiguousarray(_H_spin)
        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            _H_spin = np.repeat(_H_spin, 2, axis=0)
            _H_spin = np.repeat(_H_spin, 2, axis=1)
            # spin part of 1-e integrals
            spin_ind = np.arange(_H_spin.shape[0], dtype=int) % 2
            # product of spatial and spin parts
            self.Hspin = _H_spin * (spin_ind.reshape(-1, 1) == spin_ind)

    def build2DSO(self):
        """Will build the 2-electron arrays in the spin orbital basis
        that contribute to the A+Delta blocks

        """

        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            _d_spin = np.copy(self.d_cmo)
            self.d_spatial = np.einsum("ij,kl->ijkl", _d_spin, _d_spin)
            # try a more efficient way of computing the 2-e d terms
            _d_spin = np.repeat(_d_spin, 2, axis=0)
            _d_spin = np.repeat(_d_spin, 2, axis=1)
            spin_ind = np.arange(_d_spin.shape[0], dtype=int) % 2
            self.d_spin = _d_spin * (spin_ind.reshape(-1, 1) == spin_ind)

            t1 = np.einsum("ik,jl->ijkl", self.d_spin, self.d_spin)
            t2 = np.einsum("il,jk->ijkl", self.d_spin, self.d_spin)
            self.TDI_spin = t1 - t2
            self.antiSym2eInt = self.eri_so + self.TDI_spin
        # self.d_spatial = np.einsum("ij,kl->ijkl", _d_spin, _d_spin)
        if self.ignore_dse_terms:
            self.twoeint = self.twoeint1
        else:
            self.twoeint = self.twoeint1 + np.einsum(
                "ij,kl->ijkl", self.d_cmo, self.d_cmo
            )
        self.twoeint1 = None
        del self.twoeint1
        # self.contracted_twoeint = -0.5 * np.einsum("illj->ij", self.twoeint)
        self.twoeint = np.reshape(
            self.twoeint, (self.nmo * self.nmo, self.nmo * self.nmo)
        )

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

        # these terms are used for coherent state and number state basis
        self.Enuc_so = self.Enuc * _I
        self.Omega_so = self.omega * _I

        # these terms are different depending on if we are in coherent state or number basis
        self.dc_so = (
            self.d_c * _I
        )  # <== we have already taken care of differentiating between d_c in lines 1117 and 1124

        # <== G_exp = +w/2 <d_el> * I in the coherent state basis and G_exp = -w/2 * d_N * I in the photon number basis
        # the negative sign is taken care of in the parsing around line 1119
        self.G_exp_so = np.sqrt(self.omega / 2) * self.d_exp * _I

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
        self.num_alpha = 0
        self.detmap = []
        for alpha in combinations(range(self.nmo), self.ndocc):
            self.num_alpha += 1
            for beta in combinations(range(self.nmo), self.ndocc):
                e = Determinant(alphaObtList=alpha, betaObtList=beta)
                alphabit = e.obtIndexList2ObtBits(alpha)
                betabit = e.obtIndexList2ObtBits(beta)
                self.detmap.append([alphabit, betabit])

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
                    self.detmap.append([alphabit, betabit])

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
        # self.num_alpha = 0

        n_in_orb = self.ndocc - self.n_act_el // 2
        n_ac_el_half = self.n_act_el // 2
        inactive_list = list(x for x in range(n_in_orb))

        print("Generating all determinants in active space")
        for alpha in combinations(range(self.n_act_orb), n_ac_el_half):
            # self.num_alpha += 1
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
                self.detmap.append([alphabit, betabit])
                self.CASnumDets += 1

        for i in range(len(self.CASdets)):
            # print(self.CASdets[i])
            unique1, unique2, sign = self.CASdets[
                i
            ].getUniqueOrbitalsInMixIndexListsPlusSign(self.CASdets[0])
            # print(unique1, unique2, sign)
            if i > 0:
                self.CASsingdetsign.append(sign)
        # for i in range(len(self.detmap)):
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

        self.ApDmatrix = np.zeros((_numDets, _numDets))
        # one-electron only version of A+\Delta
        self.apdmatrix = np.zeros((_numDets, _numDets))

        self.Gmatrix = np.zeros((_numDets, _numDets))

        self.H_PF = np.zeros((2 * _numDets, 2 * _numDets))
        # one-electron version of Hamiltonian
        self.H_1E = np.zeros((2 * _numDets, 2 * _numDets))

        # dipole matrix
        self.dipole_block_x = np.zeros((_numDets, _numDets))
        self.dipole_block_y = np.zeros((_numDets, _numDets))
        self.dipole_block_z = np.zeros((_numDets, _numDets))

        for i in range(_numDets):
            for j in range(i + 1):
                self.ApDmatrix[i, j] = self.calcApDMatrixElement(_dets[i], _dets[j])
                self.apdmatrix[i, j] = self.calcApDMatrixElement(
                    _dets[i], _dets[j], OneEpTwoE=False
                )
                self.ApDmatrix[j, i] = self.ApDmatrix[i, j]
                self.apdmatrix[j, i] = self.apdmatrix[i, j]
                self.Gmatrix[i, j] = self.calcGMatrixElement(_dets[i], _dets[j])
                self.Gmatrix[j, i] = self.Gmatrix[i, j]

                _dipole_moment = self.calcMuMatrixElement(_dets[i], _dets[j])
                self.dipole_block_x[i, j] = _dipole_moment[0]
                self.dipole_block_x[j, i] = _dipole_moment[0]
                self.dipole_block_y[i, j] = _dipole_moment[1]
                self.dipole_block_y[j, i] = _dipole_moment[1]
                self.dipole_block_z[i, j] = _dipole_moment[2]
                self.dipole_block_z[i, j] = _dipole_moment[2]

        # full hamiltonian
        self.H_PF[:_numDets, :_numDets] = self.ApDmatrix + self.Enuc_so + self.dc_so
        # 1-e piece
        self.H_1E[:_numDets, :_numDets] = self.apdmatrix + self.Enuc_so + self.dc_so

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
            cqed_rhf_dict = cqed_rhf(
                self.lambda_vector,
                molecule_string,
                psi4_options_dict,
                canonical_basis=True,
            )
        else:
            cqed_rhf_dict = cqed_rhf(
                self.lambda_vector, molecule_string, psi4_options_dict
            )
        t_hf_end = time.time()
        print(f" Completed QED-RHF in {t_hf_end - t_hf_start} seconds")

        # Parse output of cqed-rhf calculation
        psi4_wfn = self.parseArrays(cqed_rhf_dict)

        if self.natural_orbitals:  # <== need to run CIS
            self.buildArraysInOrbitalBasis(psi4_wfn)
            t_det_start = time.time()
            self.generateCISDeterminants()
            H_dim = self.CISnumDets * 2
            t_det_end = time.time()
            print(f" Completed determinant list in {t_det_end - t_det_start} seconds ")

            # build Constant matrices
            self.buildConstantMatrices("cis")
            t_const_end = time.time()
            print(
                f" Completed constant offset matrix in {t_const_end - t_det_end} seconds"
            )

            # Build Matrix
            self.generatePFHMatrix("cis")
            t_H_build = time.time()
            print(f" Completed Hamiltonian build in {t_H_build - t_const_end} seconds")

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
            self.no_vec = _vec[:, _idx]
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
        print(f" Completed 1HSO Build in {t_1H_end - t_1H_start} seconds")

        # build 2eInt in cqed-rhf basis
        print("number of MO", self.nmo)
        mints = psi4.core.MintsHelper(p4_wfn.basisset())
        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            self.eri_so = np.asarray(mints.mo_spin_eri(self.Ca, self.Ca))

        self.twoeint1 = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca))
        t_eri_end = time.time()
        print(f" Completed ERI Build in {t_eri_end - t_1H_end} seconds ")

        # form the 2H in spin orbital basis
        self.build2DSO()
        t_2d_end = time.time()

        print(psutil.Process().memory_info().rss / (1024 * 1024))
        self.gkl = np.zeros((self.nmo, self.nmo))
        for k in range(self.nmo):
            for l in range(self.nmo):
                self.gkl[k][l] = self.H_spatial[k][l]
                for j in range(self.nmo):
                    if j >= k:
                        continue
                    else:
                        kj = k * self.nmo + j
                        jl = j * self.nmo + l
                        self.gkl[k][l] -= self.twoeint[kj][jl]
                if k >= l:
                    kk = k * self.nmo + k
                    kl = k * self.nmo + l
                    self.gkl[k][l] -= self.twoeint[kk][kl] / (1 + (k == l))
        print(psutil.Process().memory_info().rss / (1024 * 1024))
        print("mem_gkl", self.gkl.size * self.gkl.itemsize / 1024 / 1024)
        print("mem_twoeint", self.twoeint.size * self.twoeint.itemsize / 1024 / 1024)
        print("mem_d_cmo", self.d_cmo.size * self.d_cmo.itemsize / 1024 / 1024)

        print(f" Completed 2D build in {t_2d_end - t_eri_end} seconds")
        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            # build the array to build G in the so basis
            self.buildGSO()
            t_1G_end = time.time()
            print(f" Completed 1G build in {t_1G_end - t_2d_end} seconds")

            # build the x, y, z components of the dipole array in the so basis
            self.build1MuSO()
            t_dipole_end = time.time()
            print(
                f" Completed the Dipole Matrix Build in {t_dipole_end - t_1G_end} seconds"
            )

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
        _D_aa = np.zeros((self.nmo, self.nmo))
        _D_bb = np.zeros((self.nmo, self.nmo))

        for p in range(self.D1.shape[0]):
            for q in range(self.D1.shape[1]):
                i = p % 2
                j = (p - i) // 2

                k = q % 2
                l = (q - k) // 2

                if i == 0 and k == 0:
                    _D_aa[j, l] = self.D1[p, q]

                if i == 1 and k == 1:
                    _D_bb[j, l] = self.D1[p, q]

        # spatial orbital 1RDM
        self.D1_spatial = _D_aa + _D_bb

    def graph(self, N, n_o):
        """
        lexical ordering graph with unoccupied arc set to be zero, return vertex weight
        """
        rows, cols = ((N + 1) * (n_o - N + 1), 3)
        rows1, cols1 = (N + 1, n_o + 1)
        graph = [[0 for i in range(cols)] for j in range(rows)]
        graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
        graph_big[N][n_o] = 1
        # weight of vertex
        for e in range(N, -1, -1):
            for o in range(n_o - 1, -1, -1):
                # print(e,o)
                if e == N and o >= e:
                    graph_big[e][o] = graph_big[e][o + 1]
                elif e <= N and o < e:
                    graph_big[e][o] = 0
                else:
                    graph_big[e][o] = graph_big[e + 1][o + 1] + graph_big[e][o + 1]

        count = 0
        for e in range(0, N + 1):
            for o in range(0, n_o + 1):
                # print(e,o,graph_big[e][o])
                if graph_big[e][o] != 0:
                    graph[count][0] = e
                    graph[count][1] = o
                    graph[count][2] = graph_big[e][o]
                    count += 1

        return graph, graph_big

    def arc_weight(self, graph, graph_big, N, n_o):
        """
        return diagonal (occupied) arc weight for lexical ordering graph above
        Y=arc weight
        """
        Y = []
        for row in range(0, len(graph)):
            # print(graph[i])
            e = graph[row][0]
            o = graph[row][1]
            B = []
            if e == N:
                continue
            i = o - e
            c = 0
            if i == 0:
                c = 0
                B.extend([e, o, c])
                # print(B)
            else:
                for j in range(1, i + 1):
                    c += graph_big[e + 1][o + 2 - j]
                B.extend([e, o, c])
                # print(B)
            Y.append(B)

        return Y

    def string_to_binary(self, string, n_o):
        b = [int(d) for d in str(bin(string))[2:]]
        if len(b) < n_o:
            c = [0] * (n_o - len(b))
            d = c + b
        else:
            d = b
        d.reverse()
        return d

    def string_to_index(self, string, N, n_o, Y):
        """
        return index using arc weight for a string(integer represents binary string)
        """
        count = 0
        index = 0  # counting from 0 so that determinant index can be counted from 0
        b = [int(d) for d in str(bin(string))[2:]]
        if len(b) < n_o:
            c = [0] * (n_o - len(b))
            d = c + b
        else:
            d = b
        d.reverse()
        for i in range(n_o):
            if d[i] == 1:
                e = count
                o = i
                for j in range(0, len(Y)):
                    if Y[j][0] == e and Y[j][1] == o:
                        index += Y[j][2]
                count += 1
        return index

    def binary_to_index(self, binary_string, N, n_o, Y):
        count = 0
        index = 0  # count from 0
        string = 0
        for i in range(0, len(binary_string)):
            if binary_string[i] == 1:
                e = count
                o = i
                string += pow(2, i)
                for j in range(0, len(Y)):
                    if Y[j][0] == e and Y[j][1] == o:
                        index += Y[j][2]
                count += 1
        # print(string)
        return index, string

    def index_to_string(self, index, N, n_o, Y, return_binary=False):
        """
        return string for an index using graph
        """
        I = 0  # count from 0 in stead of 1 so that determinant index can be counted from 0
        e = N
        o = n_o
        count = 0
        record = []
        while I <= index and e <= o:
            # print(count,e,o,I,record)
            if e == o and I < index:
                count2 = 0
                for i in range(len(record) - 1, -1, -1):
                    if record[i] == 0:
                        count2 += 1
                        record.pop(i)
                    else:
                        record[i] = 0
                        break
                o = o + count2
                e = e + 1
                for j in range(0, len(Y)):
                    if Y[j][0] == e - 1 and Y[j][1] == o:
                        b = Y[j][2]
                I = I - b
                # print(record,o,e,b,I)

            else:
                if e > 0:
                    for j in range(0, len(Y)):
                        if Y[j][0] == e - 1 and Y[j][1] == o - 1:
                            a = Y[j][2]
                            if a <= index - I:
                                e = e - 1
                                o = o - 1
                                I = I + a
                                record.append(1)
                            else:
                                o = o - 1
                                record.append(0)
                else:
                    o = o - 1
                    record.append(0)
            count += 1
            if count == 15000000 or (e == 0 and o == 0 and I == index):
                break
        string = 0
        for i in range(len(record)):
            if record[i] == 1:
                string += pow(2, len(record) - i - 1)
        # print(string)
        # print(record)

        if return_binary == True:
            return record, string
        else:
            return string

    def phase_single_excitation(self, p, q, string):
        """getting the phase(-1 or 1) for E_pq\ket{I_a} where I_a is an alpha or beta string
        p=particle,q=hole
        """
        if p > q:
            mask = (1 << p) - (1 << (q + 1))
        else:
            mask = (1 << q) - (1 << (p + 1))
        if bin(mask & string).count("1") % 2:
            return -1
        else:
            return 1

    def single_replacement_list2(self, num_strings, N, n_o, n_in_a, Y):
        """getting the sign, string address and pq for sign(pq)E_pq\ket{I_a}
        N = number of active alpha electrons
        n_o = number of active alpha orbitals
        n_in_a = number of inactive alpha orbitals
        p=particle, q=hole
        """
        rows, cols = (num_strings * (N * (n_o - N) + N + n_in_a), 4)
        self.num_links = N * (n_o - N) + N + n_in_a
        table = [[0 for i in range(cols)] for j in range(rows)]
        count = 0
        for index in range(0, num_strings):
            string = self.index_to_string(index, N, n_o, Y)
            # d = self.string_to_binary(string,n_o)
            # print('single replacement list for binary string',d,'string',string,'index',index)
            occ = []
            vir = []
            # print(occ,vir)
            for i in range(n_o):
                if string & (1 << i):
                    occ.append(i)
                else:
                    vir.append(i)
            if n_in_a > 0:
                for i in range(n_in_a):
                    table[count][0] = index
                    table[count][1] = 1
                    table[count][2] = i
                    table[count][3] = i
                    count += 1

            for i in range(N):
                table[count][0] = index
                table[count][1] = 1
                table[count][2] = occ[i] + n_in_a
                table[count][3] = occ[i] + n_in_a
                count += 1
            for i in range(N):
                for a in range(n_o - N):
                    string1 = (string ^ (1 << occ[i])) | (1 << vir[a])
                    # c=string_to_binary(string1,n_o)
                    table[count][0] = self.string_to_index(string1, N, n_o, Y)
                    table[count][1] = self.phase_single_excitation(
                        vir[a], occ[i], string
                    )
                    table[count][2] = vir[a] + n_in_a
                    table[count][3] = occ[i] + n_in_a
                    count += 1
        return table

    def one_e_contraction(
        self, h1e, c_vectors, c1_vectors, num_strings, num_links, scale
    ):
        for indexa in range(0, num_strings):
            stride = indexa * num_links
            # print('indexa',indexa,'stride',stride)
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                # print(index1,sign,a,i)
                tmp = sign * h1e[a, i]
                for indexb in range(0, num_strings):
                    # composite index of determinant
                    index_I = indexa * num_strings + indexb
                    index_J = index1 * num_strings + indexb
                    c1_vectors[index_I] += scale * tmp * c_vectors[index_J]
        for indexb in range(0, num_strings):
            stride = indexb * num_links
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                tmp = sign * h1e[a, i]
                for indexa in range(0, num_strings):
                    index_I = indexa * num_strings + indexb
                    index_J = indexa * num_strings + index1
                    c1_vectors[index_I] += scale * tmp * c_vectors[index_J]
        # c1_vectors = scale * c1_vectors

    def two_e_contraction(self, h2e, c_vectors, c1_vectors, num_strings, num_links):
        # build intermediate D^K_kl = \gamma^KJ_kl * c_J
        D = np.zeros((self.num_det, self.nmo * self.nmo))
        for indexa in range(0, num_strings):
            stride = indexa * num_links
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                for indexb in range(0, num_strings):
                    # composite index of determinant
                    index_I = indexa * num_strings + indexb
                    index_J = index1 * num_strings + indexb
                    ai = a * self.nmo + i
                    D[index_I][ai] = sign * c_vectors[index_J]
        for indexb in range(0, num_strings):
            stride = indexb * num_links
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                for indexa in range(0, num_strings):
                    index_I = indexa * num_strings + indexb
                    index_J = indexa * num_strings + index1
                    ai = a * self.nmo + i
                    D[index_I][ai] += sign * c_vectors[index_J]

        # build intermediate E^K_ij = D^K_kl * (ij|kl)
        E = np.einsum("pr,kr->kp", h2e, D)

        # build \sigma_I = 0.5 * \gamma^IK_ij * E^K_ij
        for indexa in range(0, num_strings):
            stride = indexa * num_links
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                for indexb in range(0, num_strings):
                    # composite index of determinant
                    index_I = indexa * num_strings + indexb
                    index_J = index1 * num_strings + indexb
                    ai = a * self.nmo + i
                    c1_vectors[index_I] += 0.5 * sign * E[index_J][ai]
        for indexb in range(0, num_strings):
            stride = indexb * num_links
            for excitation in range(num_links):
                index1 = self.table[stride + excitation][0]
                sign = self.table[stride + excitation][1]
                a = self.table[stride + excitation][2]
                i = self.table[stride + excitation][3]
                for indexa in range(0, num_strings):
                    index_I = indexa * num_strings + indexb
                    index_J = indexa * num_strings + index1
                    ai = a * self.nmo + i
                    c1_vectors[index_I] += 0.5 * sign * E[index_J][ai]

    def constant_terms_contraction(self, c_vectors, c1_vectors, A):
        for i in range(len(c_vectors)):
            c1_vectors[i] += A * c_vectors[i]

    def build_sigma(self, c_vectors, s_vectors, H_dim):
        """
        build sigma for FCI and CASCI. The function use Knowles-Handy algorithm for fci, and Olsen algorithm for casci
        """

        if self.ci_level == "fci":
            print("using Knowles-Handy algorithm")
            np1 = self.N_p + 1
            for n in range(c_vectors.shape[1]):
                for m in range(self.N_p + 1):
                    start = m * H_dim // np1
                    end = (m + 1) * H_dim // np1
                    self.one_e_contraction(
                        self.oneeint,
                        c_vectors[start:end, n],
                        s_vectors[start:end, n],
                        self.num_alpha,
                        self.num_links,
                        1,
                    )
                    self.two_e_contraction(
                        self.twoeint,
                        c_vectors[start:end, n],
                        s_vectors[start:end, n],
                        self.num_alpha,
                        self.num_links,
                    )
                    someconstant = m * self.omega + self.Enuc + self.d_c
                    self.constant_terms_contraction(
                        c_vectors[start:end, n], s_vectors[start:end, n], someconstant
                    )
                    start0 = (m - 1) * H_dim // np1
                    end0 = m * H_dim // np1
                    start2 = (m + 1) * H_dim // np1
                    end2 = (m + 2) * H_dim // np1
                    if 0 < m < self.N_p:
                        someconstant = -np.sqrt(m * self.omega / 2)
                        self.one_e_contraction(
                            self.d_cmo,
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                        someconstant = -np.sqrt((m + 1) * self.omega / 2)
                        self.one_e_contraction(
                            self.d_cmo,
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                    elif m == self.N_p:
                        someconstant = -np.sqrt(m * self.omega / 2)
                        self.one_e_contraction(
                            self.d_cmo,
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                    else:
                        someconstant = -np.sqrt((m + 1) * self.omega / 2)
                        self.one_e_contraction(
                            self.d_cmo,
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
        elif self.ci_level == "cas":
            print("using Olsen algorithm")
            np1 = self.N_p + 1
            for n in range(c_vectors.shape[1]):
                for m in range(self.N_p + 1):
                    start = m * H_dim // np1
                    end = (m + 1) * H_dim // np1
                    self.sigma12(
                        self.gkl,
                        self.twoeint,
                        c_vectors[start:end, n],
                        s_vectors[start:end, n],
                        self.num_alpha,
                        self.num_links,
                    )
                    self.sigma3(
                        self.twoeint,
                        c_vectors[start:end, n],
                        s_vectors[start:end, n],
                        self.num_alpha,
                        self.num_links,
                    )
                    someconstant = m * self.omega + self.Enuc + self.d_c
                    if self.break_degeneracy == True:
                        # print('only ground state')
                        someconstant = m * (self.omega + 1) + self.Enuc + self.d_c

                    self.constant_terms_contraction(
                        c_vectors[start:end, n], s_vectors[start:end, n], someconstant
                    )
                    start0 = (m - 1) * H_dim // np1
                    end0 = m * H_dim // np1
                    start2 = (m + 1) * H_dim // np1
                    end2 = (m + 2) * H_dim // np1
                    if 0 < m < self.N_p:
                        someconstant = -np.sqrt(m * self.omega / 2)
                        self.sigma_dipole(
                            self.d_cmo,
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                        someconstant = -np.sqrt((m + 1) * self.omega / 2)
                        self.sigma_dipole(
                            self.d_cmo,
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                    elif m == self.N_p:
                        someconstant = -np.sqrt(m * self.omega / 2)
                        self.sigma_dipole(
                            self.d_cmo,
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start0:end0, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )
                    else:
                        someconstant = -np.sqrt((m + 1) * self.omega / 2)
                        self.sigma_dipole(
                            self.d_cmo,
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            self.num_alpha,
                            self.num_links,
                            someconstant,
                        )
                        self.constant_terms_contraction(
                            c_vectors[start2:end2, n],
                            s_vectors[start:end, n],
                            -self.d_exp * someconstant,
                        )

    def sigma3(self, h2e, c_vectors, c1_vectors, num_strings, num_links):
        for k in range(self.nmo):
            for l in range(self.nmo):
                kl = k * self.nmo + l
                # print('l','r','sgn')
                L = []
                R = []
                sgn = []
                dim = 0
                for N in range(len(self.table)):
                    if self.table[N][2] == k and self.table[N][3] == l:
                        index = N // num_links
                        L.append(self.table[N][0])
                        R.append(index)
                        sgn.append(self.table[N][1])
                        dim += 1
                # print(L,R,sgn,dim)
                cp = np.zeros((dim, num_strings))
                for I in range(dim):
                    for index_jb in range(0, num_strings):
                        index_ljb = L[I] * num_strings + index_jb
                        cp[I][index_jb] = c_vectors[index_ljb] * sgn[I]
                for index_ib in range(0, num_strings):
                    F = np.zeros((num_strings))
                    stride = index_ib * num_links
                    for excitation in range(num_links):
                        index_jb = self.table[stride + excitation][0]
                        sign = self.table[stride + excitation][1]
                        i = self.table[stride + excitation][2]
                        j = self.table[stride + excitation][3]
                        ij = i * self.nmo + j
                        F[index_jb] += sign * h2e[ij][kl]
                    v = np.zeros((dim))
                    v = np.einsum("pq,q->p", cp, F)
                    for I in range(dim):
                        index_I = R[I] * num_strings + index_ib
                        c1_vectors[index_I] += v[I]

    def sigma12(self, h1e, h2e, c_vectors, c1_vectors, num_strings, num_links):
        for index_ib in range(0, num_strings):
            F = np.zeros((num_strings))
            stride1 = index_ib * num_links
            # print('index_ib',index_ib,'stride',stride1)
            for excitation1 in range(num_links):
                index_kb = self.table[stride1 + excitation1][0]
                sign1 = self.table[stride1 + excitation1][1]
                k = self.table[stride1 + excitation1][2]
                l = self.table[stride1 + excitation1][3]
                # print(index_kb,sign1,k,l)
                kl = k * self.nmo + l
                F[index_kb] += sign1 * h1e[k, l]
                stride2 = index_kb * num_links
                for excitation2 in range(num_links):
                    index_jb = self.table[stride2 + excitation2][0]
                    sign2 = self.table[stride2 + excitation2][1]
                    i = self.table[stride2 + excitation2][2]
                    j = self.table[stride2 + excitation2][3]
                    ij = i * self.nmo + j
                    if ij >= kl:
                        F[index_jb] += (sign1 * sign2 * h2e[ij][kl]) / (1 + (ij == kl))
            for index_jb in range(0, num_strings):
                for index_ia in range(0, num_strings):
                    # composite index of determinant
                    index_I = index_ia * num_strings + index_ib
                    index_J = index_ia * num_strings + index_jb
                    c1_vectors[index_I] += F[index_jb] * c_vectors[index_J]
        for index_ia in range(0, num_strings):
            F = np.zeros((num_strings))
            stride1 = index_ia * num_links
            for excitation1 in range(num_links):
                index_ka = self.table[stride1 + excitation1][0]
                sign1 = self.table[stride1 + excitation1][1]
                k = self.table[stride1 + excitation1][2]
                l = self.table[stride1 + excitation1][3]
                kl = k * self.nmo + l
                F[index_ka] += sign1 * h1e[k, l]
                stride2 = index_ka * num_links
                for excitation2 in range(num_links):
                    index_ja = self.table[stride2 + excitation2][0]
                    sign2 = self.table[stride2 + excitation2][1]
                    i = self.table[stride2 + excitation2][2]
                    j = self.table[stride2 + excitation2][3]
                    ij = i * self.nmo + j
                    if ij >= kl:
                        F[index_ja] += (sign1 * sign2 * h2e[ij][kl]) / (1 + (ij == kl))
            for index_ja in range(0, num_strings):
                for index_ib in range(0, num_strings):
                    # composite index of determinant
                    index_I = index_ia * num_strings + index_ib
                    index_J = index_ja * num_strings + index_ib
                    c1_vectors[index_I] += F[index_ja] * c_vectors[index_J]

    def sigma_dipole(
        self, h1e, c_vectors, c1_vectors, num_strings, num_links, someconstant
    ):
        for index_ib in range(0, num_strings):
            F = np.zeros((num_strings))
            stride1 = index_ib * num_links
            # print('index_ib',index_ib,'stride',stride1)
            for excitation1 in range(num_links):
                index_kb = self.table[stride1 + excitation1][0]
                sign1 = self.table[stride1 + excitation1][1]
                k = self.table[stride1 + excitation1][2]
                l = self.table[stride1 + excitation1][3]
                # print(index_kb,sign1,k,l)
                kl = k * self.nmo + l
                F[index_kb] += sign1 * h1e[k, l]
            for index_jb in range(0, num_strings):
                for index_ia in range(0, num_strings):
                    # composite index of determinant
                    index_I = index_ia * num_strings + index_ib
                    index_J = index_ia * num_strings + index_jb
                    c1_vectors[index_I] += (
                        someconstant * F[index_jb] * c_vectors[index_J]
                    )
        for index_ia in range(0, num_strings):
            F = np.zeros((num_strings))
            stride1 = index_ia * num_links
            for excitation1 in range(num_links):
                index_ka = self.table[stride1 + excitation1][0]
                sign1 = self.table[stride1 + excitation1][1]
                k = self.table[stride1 + excitation1][2]
                l = self.table[stride1 + excitation1][3]
                kl = k * self.nmo + l
                F[index_ka] += sign1 * h1e[k, l]
            for index_ja in range(0, num_strings):
                for index_ib in range(0, num_strings):
                    # composite index of determinant
                    index_I = index_ia * num_strings + index_ib
                    index_J = index_ja * num_strings + index_ib
                    c1_vectors[index_I] += (
                        someconstant * F[index_ja] * c_vectors[index_J]
                    )

    def build_H_diag(self, H_dim, h1e, h2e, num_strings, n_act_a, n_act_orb, n_in_a):
        d = np.zeros((H_dim))
        np1 = self.N_p + 1
        for m in range(self.N_p + 1):
            start = m * H_dim // np1
            end = (m + 1) * H_dim // np1

            for I in range(self.num_det):
                index_a = I // num_strings
                index_b = I % num_strings
                string_a = self.index_to_string(
                    index_a, n_act_a, n_act_orb, self.Y, return_binary=False
                )
                string_b = self.index_to_string(
                    index_b, n_act_a, n_act_orb, self.Y, return_binary=False
                )
                inactive_list = list(x for x in range(n_in_a))
                double_occupation = Determinant.obtBits2ObtIndexList(
                    string_a & string_b
                )
                double_occupation = [x + n_in_a for x in double_occupation]
                double_occupation[0:0] = inactive_list
                dim_d = len(double_occupation)
                e = string_a ^ string_b
                ea = e & string_a
                eb = e & string_b
                single_occupation_a = Determinant.obtBits2ObtIndexList(ea)
                single_occupation_a = [x + n_in_a for x in single_occupation_a]
                single_occupation_b = Determinant.obtBits2ObtIndexList(eb)
                single_occupation_b = [x + n_in_a for x in single_occupation_b]
                dim_sa = len(single_occupation_a)
                dim_sb = len(single_occupation_b)
                # print(double_occupation,single_occupation_a,single_occupation_b)
                occupation_list_spin = np.zeros((dim_sa + dim_sb + dim_d, 3), dtype=int)
                for i in range(dim_d):
                    occupation_list_spin[i][0] = double_occupation[i]
                    occupation_list_spin[i][1] = 1
                    occupation_list_spin[i][2] = 1
                for i in range(dim_sa):
                    occupation_list_spin[i + dim_d][0] = single_occupation_a[i]
                    occupation_list_spin[i + dim_d][1] = 1
                for i in range(dim_sb):
                    occupation_list_spin[i + dim_d + dim_sa][0] = single_occupation_b[i]
                    occupation_list_spin[i + dim_d + dim_sa][2] = 1
                c = 0
                for a in range(dim_d + dim_sa + dim_sb):
                    i = occupation_list_spin[a][0]
                    n_ia = occupation_list_spin[a][1]
                    n_ib = occupation_list_spin[a][2]
                    n_i = n_ia + n_ib
                    ii = i * self.nmo + i
                    c += n_i * h1e[i, i]
                    for b in range(dim_d + dim_sa + dim_sb):
                        j = occupation_list_spin[b][0]
                        n_ja = occupation_list_spin[b][1]
                        n_jb = occupation_list_spin[b][2]
                        n_j = n_ja + n_jb
                        jj = j * self.nmo + j
                        c += 0.5 * n_i * n_j * h2e[ii][jj]
                        ij = i * self.nmo + j
                        c -= 0.5 * (n_ia * n_ja + n_ib * n_jb) * h2e[ij][ij]
                d[I + start] = c + m * self.omega + self.Enuc + self.d_c
        # print('d',d)
        return d

    def check_total_spin(self, v):
        norm = np.sqrt(np.dot(v, v.T))
        v1 = v / norm
        newS = np.zeros_like(v)
        num_links0 = self.n_act_a * (self.n_act_orb - self.n_act_a) + self.n_act_a
        c_sigma_s_square(
            v1,
            newS,
            self.S_diag,
            self.b_array,
            self.table,
            num_links0,
            self.n_act_orb,
            self.num_alpha,
            1,
            self.N_p,
            1.0,
        )
        total_spin = np.dot(v1, newS.T)
        return total_spin

    def Davidson(
        self, H, nroots, threshold, indim, maxdim, maxiter, build_sigma, H_diag
    ):
        if self.ci_level == "cis":
            H_diag = np.diag(H)
        H_dim = len(H_diag)
        print(H_dim, indim)
        L = indim
        # When L exceeds Lmax we will collapse the guess space so our sub-space
        # diagonalization problem does not grow too large
        Lmax = maxdim
        if self.ci_level == "cas" or self.ci_level == "fci":
            rows = self.num_alpha * (
                self.n_act_a * (self.n_act_orb - self.n_act_a)
                + self.n_act_a
                + self.n_in_a
            )
            num_links = rows // self.num_alpha

        # An array to hold the excitation energies
        theta = [0.0] * L

        # generate initial guess
        Q_idx = H_diag.argsort()[:indim]
        # print(Q_idx)
        if self.davidson_guess == "unit guess":
            Q = np.zeros((indim, H_dim))
            print("use unit guess")
            BIGNUM = 10**100
            H_diag2 = np.copy(H_diag)
            for i in range(indim):
                minimum = H_diag2[0]
                min_pos = 0
                for j in range(H_dim):
                    if H_diag2[j] < minimum:
                        minimum = H_diag2[j]
                        min_pos = j
                Q[i][min_pos] = 1.0
                H_diag2[min_pos] = BIGNUM
        else:
            print("use random guess")
            Q = np.random.rand(indim, H_dim)
        # print(Q)
        print("qshape", np.shape(Q))
        print("initial_mem_q", Q.size * Q.itemsize / 1024 / 1024)

        num_iter = maxiter
        for a in range(0, num_iter):
            print("\n")
            print("mem_q1", Q.size * Q.itemsize / 1024 / 1024)
            # orthonormalization of basis vectors by QR
            t_qr_begin = time.time()
            Q, R = np.linalg.qr(Q.T)
            t_qr_end = time.time()
            print("QR took", t_qr_end - t_qr_begin, "seconds")
            Q = np.ascontiguousarray(Q.T)
            R = None
            del R
            L = Q.shape[0]  # dynamic dimension of subspace
            print(np.shape(Q))
            print("iteration", a + 1, "subspace dimension", L)
            theta_old = theta[:nroots]
            # singma build
            S = np.zeros_like(Q)
            if self.ci_level == "fci" or self.ci_level == "cas":
                # S = np.einsum("pq,qi->pi", H, Q)
                t_sigma_begin = time.time()
                # build_sigma(Q,S,H_dim)
                if self.ignore_dse_terms:
                    self.d_c = 0.0
                print(psutil.Process().memory_info().rss / (1024 * 1024))

                c_sigma(
                    self.gkl2,
                    self.twoeint,
                    self.d_cmo,
                    Q,
                    S,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.n_act_a,
                    self.n_act_orb,
                    self.n_in_a,
                    self.nmo,
                    L,
                    self.N_p,
                    self.Enuc,
                    self.d_c,
                    self.omega,
                    self.d_exp,
                    self.E_core,
                    self.break_degeneracy,
                )

                print(psutil.Process().memory_info().rss / (1024 * 1024))

                t_sigma_end = time.time()
                print("build sigma took", t_sigma_end - t_sigma_begin, "seconds")
            else:
                S = np.einsum("ip,pq->iq", Q, H)
            # Build the subspace Hamiltonian
            G = np.dot(Q, S.T)
            # Diagonalize it, and sort the eigenvector/eigenvalue pairs
            theta, alpha = np.linalg.eigh(G)
            # print(theta)
            idx = theta.argsort()[:nroots]
            theta = theta[idx]
            alpha = alpha[:, idx]
            # This vector will hold the new guess vectors to add to our space
            # add_Q = []
            w = np.zeros((nroots, H_dim))
            residual_norm = np.zeros((nroots))
            unconverged_idx = []
            convergence_check = np.zeros((nroots), dtype=str)
            conv = 0
            for j in range(nroots):
                # Compute a residual vector "w" for each root we seek
                w[j, :] = np.dot(alpha[:, j].T, S) - theta[j] * np.dot(alpha[:, j].T, Q)
                residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                if residual_norm[j] < threshold:
                    conv += 1
                    convergence_check[j] = "Yes"
                else:
                    unconverged_idx.append(j)
                    convergence_check[j] = "No"
            print(unconverged_idx)

            print("root", "residual norm", "Eigenvalue", "Convergence")
            for j in range(nroots):
                print(
                    j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
                )

            if conv == nroots:
                print("converged!")
                break

            if conv < nroots and a + 1 == num_iter:
                print("maxiter reached, try to increase maxiter or subspace size")
                break

            t_precondition_begin = time.time()
            # preconditioned_w = np.zeros((len(unconverged_idx),H_dim))
            if len(unconverged_idx) > 0:
                preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
                preconditioned_w = theta[unconverged_idx].reshape(
                    len(unconverged_idx), 1
                ) - H_diag.reshape(1, H_dim)
                # print(np.shape(preconditioned_w))
                preconditioned_w = np.divide(
                    w[unconverged_idx],
                    preconditioned_w,
                    out=np.zeros_like(w[unconverged_idx]),
                    where=preconditioned_w != 0,
                )

            t_precondition_end = time.time()
            print(
                "build precondition took",
                t_precondition_end - t_precondition_begin,
                "seconds",
            )
            if Lmax - L < len(unconverged_idx):
                t_collapsing_begin = time.time()
                # unconverged_w = np.zeros((len(unconverged_idx), H_dim))
                Q = np.dot(alpha.T, Q)

                # for i in range(len(unconverged_idx)):
                #    unconverged_w[i, :] = w[unconverged_idx[i], :]

                # Q=np.append(Q,unconverged_w)
                # print(Q)
                # print(unconverged_w)
                Q = np.concatenate((Q, preconditioned_w), axis=0)
                print(Q.shape)
                # Q=np.column_stack(Qtup)

                # These vectors will give the same eigenvalues at the next
                # iteration so to avoid a false convergence we reset the theta
                # vector to theta_old
                theta = theta_old
                t_collapsing_end = time.time()
                print("restart took", t_collapsing_end - t_collapsing_begin, "seconds")
            else:
                t_expanding_begin = time.time()
                # if not we add the preconditioned residuals to the guess
                # space, and continue. Note that the set will be orthogonalized
                # at the start of the next iteration
                Q = tuple(Q[i, :] for i in range(L)) + tuple(preconditioned_w)
                Q = np.row_stack(Q)
                t_expanding_end = time.time()
                print("expand took", t_expanding_end - t_expanding_begin, "seconds")
                # print(Q)
        newQ = np.dot(alpha.T, Q)
        Q = np.zeros((H_dim, nroots))
        Q = newQ.T
        singletcount = 0
        print(Q.shape)
        for i in range(Q.shape[1]):
            print("STATE", i, "energy =", theta[i])
        if self.ci_level == "cis":
            for i in range(Q.shape[1]):
                # print("state",i, "energy =",theta[i])
                print(
                    "        amplitude",
                    "      position",
                    "         most important determinants",
                    "             number of photon",
                )
                index = np.argsort(np.abs(Q[:, i]))
                c0 = index[Q.shape[0] - 1] % (H_dim // 2)
                d0 = (index[Q.shape[0] - 1] - c0) // (H_dim // 2)
                a0, b0 = self.detmap[c0]
                alphalist = Determinant.obtBits2ObtIndexList(a0)
                betalist = Determinant.obtBits2ObtIndexList(b0)

                singlet = 1
                for j in range(min(H_dim, 10)):
                    c = index[Q.shape[0] - j - 1] % (H_dim // 2)
                    d = (index[Q.shape[0] - j - 1] - c) // (H_dim // 2)
                    a, b = self.detmap[c]
                    if (
                        a == b0
                        and b == a0
                        and np.abs(
                            Q[index[Q.shape[0] - j - 1]][i]
                            - (-1) * Q[index[Q.shape[0] - 1]][i]
                        )
                        < 1e-4
                    ):
                        singlet = singlet * 0
                    else:
                        singlet = singlet * 1
                    alphalist = Determinant.obtBits2ObtIndexList(a)
                    betalist = Determinant.obtBits2ObtIndexList(b)

                    print(
                        "%20.12lf" % (Q[index[Q.shape[0] - j - 1]][i]),
                        "%9.3d" % (index[Q.shape[0] - j - 1]),
                        "      alpha",
                        alphalist,
                        "   beta",
                        betalist,
                        "%4.1d" % (d),
                        "photon",
                    )
                # print("state",i, "energy =",theta[i], singlet)
                if singlet == 1:
                    print("state", i, "energy =", theta[i], "  singlet", singletcount)
                    singletcount += 1
                else:
                    print(
                        "state",
                        i,
                        "energy =",
                        theta[i],
                        "  triplet",
                        "%2.1d" % (d0),
                        "photon",
                    )
        else:
            print(
                "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
            )
            Y = np.zeros(
                self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3, dtype=np.int32
            )
            c_graph(self.n_act_a, self.n_act_orb, Y)
            np1 = self.N_p + 1
            singlet_count = 0
            triplet_count = 0
            for i in range(Q.shape[1]):
                total_spin = self.check_total_spin(newQ[i : (i + 1), :])
                print("state", i, "energy =", theta[i], "<S^2>=", total_spin, end="")
                if np.abs(total_spin) < 1e-5:
                    singlet_count += 1
                    print("\tsinglet", singlet_count)
                elif np.abs(total_spin - 2.0) < 1e-5:
                    triplet_count += 1
                    print("\ttriplet", triplet_count)
                elif np.abs(total_spin - 6.0) < 1e-5:
                    print("\tquintet")
                # print("state",i, "energy =",theta[i])
                print(
                    "        amplitude",
                    "      position",
                    "         most important determinants",
                    "             number of photon",
                )
                index = np.argsort(np.abs(Q[:, i]))
                # print(index)
                Idet0 = (
                    index[Q.shape[0] - 1] % self.num_det
                )  # determinant index of most significant contribution
                photon_p0 = (
                    index[Q.shape[0] - 1] - Idet0
                ) // self.num_det  # photon number block of determinant
                Ib0 = Idet0 % self.num_alpha
                Ia0 = Idet0 // self.num_alpha
                a0 = c_index_to_string(Ia0, self.n_act_a, self.n_act_orb, Y)
                b0 = c_index_to_string(Ib0, self.n_act_a, self.n_act_orb, Y)

                alphalist = Determinant.obtBits2ObtIndexList(a0)
                betalist = Determinant.obtBits2ObtIndexList(b0)
                for j in range(min(H_dim, 10)):
                    Idet = index[Q.shape[0] - j - 1] % self.num_det
                    photon_p = (index[Q.shape[0] - j - 1] - Idet) // self.num_det
                    Ib = Idet % self.num_alpha
                    Ia = Idet // self.num_alpha
                    a = c_index_to_string(Ia, self.n_act_a, self.n_act_orb, Y)
                    b = c_index_to_string(Ib, self.n_act_a, self.n_act_orb, Y)

                    alphalist = Determinant.obtBits2ObtIndexList(a)
                    betalist = Determinant.obtBits2ObtIndexList(b)

                    inactive_list = list(x for x in range(self.n_in_a))
                    alphalist2 = [x + self.n_in_a for x in alphalist]
                    # alphalist2[0:0] = inactive_list
                    betalist2 = [x + self.n_in_a for x in betalist]
                    # betalist2[0:0] = inactive_list

                    print(
                        "%20.12lf" % (Q[index[Q.shape[0] - j - 1]][i]),
                        "%9.3d" % (index[Q.shape[0] - j - 1]),
                        "      alpha",
                        alphalist2,
                        "   beta",
                        betalist2,
                        "%4.1d" % (photon_p),
                        "photon",
                    )

        davidson_dict = {
            "DAVIDSON EIGENVALUES": theta,
            "DAVIDSON EIGENVECTORS": Q,
        }

        return davidson_dict

    def classifySpinState(self):
        self.triplets = []
        self.singlets = [item for item in range(2 * self.CISnumDets)]
        Q = self.cis_c
        theta = self.cis_e
        singlesDets = self.CISnumDets - 1
        halfDets = singlesDets // 2

        H_dim = Q.shape[0]
        print("H_dim is", H_dim)
        print("dimensions of detmap", len(self.detmap))
        for i in range(Q.shape[1]):
            print("state", i, "energy =", theta[i])
            needs_assignment = True
            print(
                "        amplitude",
                "      position",
                "         most important determinants",
                "             number of photon",
            )
            for j in range(1, halfDets):
                # if j<H_dim/2:
                # zero photon
                j_off = j + halfDets
                if np.abs(Q[j, i]) > 0.2:
                    a1, b1 = self.detmap[j]
                    a2, b2 = self.detmap[j_off]
                    alpha1 = Determinant.obtBits2ObtIndexList(a1)
                    beta1 = Determinant.obtBits2ObtIndexList(b1)
                    alpha2 = Determinant.obtBits2ObtIndexList(a2)
                    beta2 = Determinant.obtBits2ObtIndexList(b2)
                    if needs_assignment:
                        if np.isclose(Q[j, i], -1 * Q[j_off, i]):
                            self.triplets.append(i)
                            self.singlets.remove(i)
                            needs_assignment = False

                    print(
                        "%20.12lf" % (Q[j, i]),
                        "%9.3d" % (j),
                        "      alpha",
                        alpha1,
                        "beta",
                        beta1,
                        "       0 photon",
                    )
                    print(
                        "%20.12lf" % (Q[j_off, i]),
                        "%9.3d" % (j_off),
                        "      alpha",
                        alpha2,
                        "beta",
                        beta2,
                        "       0 photon",
                    )

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
                        if np.isclose(Q[j_o, i], -1 * Q[j_o_off, i]):
                            self.triplets.append(i)
                            self.singlets.remove(i)
                            needs_assignment = False

                    print(
                        "%20.12lf" % (Q[j_o, i]),
                        "%9.3d" % (j_o),
                        "      alpha",
                        alpha1,
                        "beta",
                        beta1,
                        "       1 photon",
                    )
                    print(
                        "%20.12lf" % (Q[j_o_off, i]),
                        "%9.3d" % (j_o_off),
                        "      alpha",
                        alpha2,
                        "beta",
                        beta2,
                        "       1 photon",
                    )
