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
#from memory_profiler import profile
from helper_cqed_rhf import cqed_rhf
from itertools import combinations
import math
import time
import ctypes
import numpy as np
from ctypes import *
import os
import psutil
import copy
import scipy.sparse
import ortho_script
from ortho_script import ortho_orbs
from scipy.stats import ortho_group
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import LinearOperator
from timeit import default_timer as timer
import numba as nb

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
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_bool,
    ctypes.c_double,
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

cfunctions.build_active_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_double,
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
    ctypes.c_double,
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

cfunctions.build_H_diag_cas.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
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
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
]
cfunctions.build_H_diag_cas_spin.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
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
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_double,
]

cfunctions.gram_schmidt_orthogonalization.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32]
cfunctions.gram_schmidt_add.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32]

cfunctions.full_transformation_macroiteration.argtypes = [ 
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int32,
        ctypes.c_int32]
cfunctions.full_transformation_internal_optimization.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int32,
        ctypes.c_int32]
cfunctions.build_sigma_reduced.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32]


def c_H_diag_cas(h1e, h2e, H_diag, N_p, num_alpha, nmo, n_act_a, n_act_orb, n_in_a, E_core, omega, Enuc, dc, Y):
    cfunctions.build_H_diag_cas(h1e, h2e, H_diag, N_p, num_alpha, nmo, n_act_a, n_act_orb, n_in_a, E_core, omega, Enuc, dc, Y)


def c_H_diag_cas_spin(h1e, h2e, H_diag, N_p, num_alpha, nmo, n_act_a, n_act_orb, n_in_a, E_core, omega, Enuc, dc, Y, target_spin):
    cfunctions.build_H_diag_cas_spin(h1e, h2e, H_diag, N_p, num_alpha, nmo, n_act_a, n_act_orb, n_in_a, E_core, omega, Enuc, dc, Y, target_spin)



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
    E_core,
    omega,
    Enuc,
    dc,
    target_spin
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
        E_core,
        omega,
        Enuc,
        dc,
        target_spin
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
    Sdiag,
    Sdiag_projection,
    eigenvals,
    eigenvecs,
    table,
    table_creation,
    table_annihilation,
    b_array,
    constint,
    constdouble,
    index_Hdiag,
    casscf,
    target_spin,
):
    cfunctions.get_roots(
        h1e,
        h2e,
        d_cmo,
        Hdiag,
        Sdiag,
        Sdiag_projection,
        eigenvals,
        eigenvecs,
        table,
        table_creation,
        table_annihilation,
        b_array,
        constint,
        constdouble,
        index_Hdiag,
        casscf,
        target_spin,
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

def c_build_active_rdm(
    eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_rdm(
        eigvec, D_tu, D_tuvw,  table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
    )
def c_build_photon_electron_one_rdm(
    eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
):
    cfunctions.build_photon_electron_one_rdm(
        eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
    )

def c_build_active_photon_electron_one_rdm(
    eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_photon_electron_one_rdm(
        eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
    )


def c_gram_schmidt_orthogonalization(Q, rows, cols):
    cfunctions.gram_schmidt_orthogonalization(Q, rows, cols)
def c_gram_schmidt_add(Q, rows, cols, rows2):
    cfunctions.gram_schmidt_add(Q, rows, cols, rows2)
    
def c_full_transformation_macroiteration(U, h2e, J, K, index_map_pq, index_map_kl, nmo, n_occupied): 
    cfunctions.full_transformation_macroiteration(U, h2e, J, K, index_map_pq, index_map_kl, nmo, n_occupied)

def c_full_transformation_internal_optimization(U, J, K, h1, d_cmo, Jt, Kt, h1t, d_cmot, index_map_ab, index_map_kl, nmo, n_occupied):
    cfunctions.full_transformation_internal_optimization(U, J, K, h1, d_cmo, Jt, Kt, h1t, d_cmot, index_map_ab, index_map_kl, nmo, n_occupied)

def c_build_sigma_reduced(U, A_tilde, index_map, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
    cfunctions.build_sigma_reduced(U, A_tilde, index_map, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied)
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
                self.n_occupied = self.n_act_orb + self.n_in_a
                self.num_alpha = math.comb(
                    self.n_act_orb, self.n_act_a
                )  # number of alpha strings
                self.num_det = self.num_alpha * self.num_alpha  # number of determinants
                self.CASnumDets = self.num_det
                H_dim = self.CASnumDets * np1
                self.H_diag = np.zeros(H_dim)
                self.H1temp= copy.deepcopy(self.H_spatial2) 
                self.d_cmo_temp= copy.deepcopy(self.d_cmo) 
                self.build_JK()
                self.occupied_J = copy.deepcopy(self.J[:,:,:self.n_occupied,:self.n_occupied])
                self.occupied_K = copy.deepcopy(self.K[:,:,:self.n_occupied,:self.n_occupied])
                self.occupied_h1 = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied])
                self.occupied_d_cmo = copy.deepcopy(self.d_cmo[:self.n_occupied,:self.n_occupied])
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                
                #build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                #for k in range(self.nmo):
                #    for l in range(self.nmo):
                #        kl = k * self.nmo + l
                #        self.fock_core[k][l] = self.H_spatial2[k][l]
                #        for j in range(self.n_in_a):
                #            jj = j * self.nmo + j
                #            kj = k * self.nmo + j
                #            jl = j * self.nmo + l
                #            self.fock_core[k][l] += 2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]
                
                self.fock_core = copy.deepcopy(self.H_spatial2) 
                self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
        
                self.occupied_fock_core = copy.deepcopy(self.fock_core[:self.n_occupied, :self.n_occupied]) 



                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                #for k in range(self.n_act_orb):
                #    for l in range(self.n_act_orb):
                #        self.gkl2[k][l] = self.fock_core[k+self.n_in_a][l+self.n_in_a]
                #        for j in range(self.n_act_orb):
                #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                #            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]
                self.gkl2 = copy.deepcopy(self.fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
                self.gkl2 -= 0.5 * np.einsum("kjjl->kl", 
                    self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 


                self.E_core = 0.0  
                #for i in range(self.n_in_a):
                #    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i] 
                self.E_core += np.einsum("jj->", self.occupied_h1[:self.n_in_a,:self.n_in_a]) 
                self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 

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
                self.Y = np.zeros(
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3, dtype=np.int32
                )
                c_graph(self.n_act_a, self.n_act_orb, self.Y)

                #self.H_diag3 = np.zeros(H_dim)
                #c_H_diag_cas(
                #        self.occupied_fock_core, 
                #        self.occupied_J3, 
                #        self.H_diag3, 
                #        self.N_p, 
                #        self.num_alpha, 
                #        self.nmo, 
                #        self.n_act_a, 
                #        self.n_act_orb, 
                #        self.n_in_a, 
                #        self.E_core, 
                #        self.omega, 
                #        self.Enuc, 
                #        self.d_c, 
                #        self.Y)
                #print(self.H_diag3)
                
                #self.target_spin = 0.0
                self.H_diag3 = np.zeros(H_dim)
                c_H_diag_cas_spin(
                        self.occupied_fock_core, 
                        self.occupied_J3, 
                        self.H_diag3, 
                        self.N_p, 
                        self.num_alpha, 
                        self.nmo, 
                        self.n_act_a, 
                        self.n_act_orb, 
                        self.n_in_a, 
                        self.E_core, 
                        self.omega, 
                        self.Enuc, 
                        self.d_c, 
                        self.Y,
                        self.target_spin)
                #print(self.H_diag3, flush = True)
                self.index_Hdiag = np.asarray(self.H_diag3.argsort(),dtype = np.int32)
                #np.savetxt("H_diag.out", self.H_diag3)
                #print(np.sort(self.H_diag3))
                c_string(
                    self.occupied_fock_core,
                    self.occupied_J3,
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
                    self.E_core,
                    self.omega,
                    self.Enuc,
                    self.d_c,
                    self.target_spin
                )
                #print(self.H_diag, flush = True)
                #print(self.H_diag - self.H_diag3, flush = True)
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
                self.S_diag_projection = np.zeros(H_dim)
                shift = self.target_spin * (self.target_spin + 1)
                c_s_diag(self.S_diag_projection, self.num_alpha, self.nmo, self.n_act_a, self.n_act_orb, self.n_in_a, shift)
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
                self.n_occupied = self.n_act_orb + self.n_in_a

                self.num_alpha = math.comb(
                    self.nmo, self.ndocc
                )  # number of alpha strings
                self.num_det = self.num_alpha * self.num_alpha  # number of determinants
                self.FCInumDets = self.num_det
                H_dim = self.FCInumDets * np1
                self.H_diag = np.zeros(H_dim)
  
                self.build_JK()
                self.occupied_K = copy.deepcopy(self.K[:,:,:self.n_occupied,:self.n_occupied])
                self.occupied_J = copy.deepcopy(self.J[:,:,:self.n_occupied,:self.n_occupied])
                self.occupied_h1 = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied])
                self.occupied_d_cmo = copy.deepcopy(self.d_cmo[:self.n_occupied,:self.n_occupied])
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

                #build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                #for k in range(self.nmo):
                #    for l in range(self.nmo):
                #        kl = k * self.nmo + l
                #        self.fock_core[k][l] = self.H_spatial2[k][l]
                #        for j in range(self.n_in_a):
                #            jj = j * self.nmo + j
                #            kj = k * self.nmo + j
                #            jl = j * self.nmo + l
                #            self.fock_core[k][l] += 2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]
                self.fock_core = copy.deepcopy(self.H_spatial2) 
                self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
        
                self.occupied_fock_core = copy.deepcopy(self.fock_core[:self.n_occupied, :self.n_occupied]) 

                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                #for k in range(self.n_act_orb):
                #    for l in range(self.n_act_orb):
                #        self.gkl2[k][l] = self.fock_core[k+self.n_in_a][l+self.n_in_a]
                #        for j in range(self.n_act_orb):
                #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                #            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]
                self.gkl2 = copy.deepcopy(self.fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
                self.gkl2 -= 0.5 * np.einsum("kjjl->kl", 
                    self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 


                self.E_core = 0.0  
                #for i in range(self.n_in_a):
                #    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i] 
                self.E_core += np.einsum("jj->", self.occupied_h1[:self.n_in_a,:self.n_in_a]) 
                self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 



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
                self.Y = np.zeros(
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3, dtype=np.int32
                )
                c_graph(self.n_act_a, self.n_act_orb, self.Y)


                #self.target_spin = 0.0
                self.H_diag3 = np.zeros(H_dim)
                c_H_diag_cas_spin(
                        self.occupied_fock_core, 
                        self.occupied_J3, 
                        self.H_diag3, 
                        self.N_p, 
                        self.num_alpha, 
                        self.nmo, 
                        self.n_act_a, 
                        self.n_act_orb, 
                        self.n_in_a, 
                        self.E_core, 
                        self.omega, 
                        self.Enuc, 
                        self.d_c, 
                        self.Y,
                        self.target_spin)
                #print(self.H_diag3, flush = True)
                self.index_Hdiag = np.asarray(self.H_diag3.argsort(),dtype = np.int32)
                #np.savetxt("H_diag.out", self.H_diag3)
                #print(np.sort(self.H_diag3))
                c_string(
                    self.occupied_fock_core,
                    self.occupied_J3,
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
                    self.E_core,
                    self.omega,
                    self.Enuc,
                    self.d_c,
                    self.target_spin
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
                self.S_diag_projection = np.zeros(H_dim)
                shift = self.target_spin * (self.target_spin + 1)
                c_s_diag(self.S_diag_projection, self.num_alpha, self.nmo, self.n_act_a, self.n_act_orb, self.n_in_a, shift) 
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
            #print(H_dim, self.n_act_a,self.nmo)
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
                    d_diag += 2.0 * self.occupied_d_cmo[i][i]
                #print(d_diag, self.d_exp, self.N_p, self.n_act_orb, self.nmo, self.omega, self.num_alpha)
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
                    self.occupied_J3,
                    self.occupied_d_cmo,
                    self.H_diag,
                    self.S_diag,
                    self.S_diag_projection,
                    eigenvals,
                    eigenvecs,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.b_array,
                    self.constint,
                    self.constdouble,
                    self.index_Hdiag,
                    False,
                    self.target_spin,
                )
                



                self.CIeigs = eigenvals
                self.CIvecs = eigenvecs

                print(
                    "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS",flush = True
                )
                Y = np.zeros(
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                    dtype=np.int32,
                )
                c_graph(self.n_act_a, self.n_act_orb, Y)
                np1 = self.N_p + 1
                singlet_count = 0
                triplet_count = 0
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

                print(" GOING TO COMPUTE 1-E PROPERTIES!", flush = True)
                _mu_x_spin = np.einsum("uj,vi,uv", self.C[:,:self.n_occupied], self.C[:,:self.n_occupied], self.mu_x_ao)
                _mu_y_spin = np.einsum("uj,vi,uv", self.C[:,:self.n_occupied], self.C[:,:self.n_occupied], self.mu_y_ao)
                _mu_z_spin = np.einsum("uj,vi,uv", self.C[:,:self.n_occupied], self.C[:,:self.n_occupied], self.mu_z_ao)
                _mu_x_spin = np.ascontiguousarray(_mu_x_spin)
                _mu_y_spin = np.ascontiguousarray(_mu_y_spin)
                _mu_z_spin = np.ascontiguousarray(_mu_z_spin)

                # store dipole moments as attributes
                # total dipole moments, mu_el + mu_nuc
                self.dipole_array = np.zeros(
                    (self.davidson_roots, self.davidson_roots, 3)
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
                self.nat_obt_number = np.zeros((self.davidson_roots, self.n_occupied))

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
                            j
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
                            "{:20.12f}".format(dipole_z), flush = True
                        )
                        if i == j:
                            one_rdm = np.reshape(one_rdm, (self.n_occupied, self.n_occupied))
                            rdm_eig = np.linalg.eigvalsh(one_rdm)
                            self.nat_obt_number[i, :] = rdm_eig[np.argsort(-rdm_eig)][:]
                        self.electronic_dipole_array[i, j, 0] = dipole_x
                        self.electronic_dipole_array[i, j, 1] = dipole_y
                        self.electronic_dipole_array[i, j, 2] = dipole_z

                # combine nuclear and electronic parts for the total dipole array
                self.dipole_array = (
                    self.electronic_dipole_array + self.nuclear_dipole_array
                )
                # print(self.nat_obt_number)
                ###check total energy
                print("check total energy using full rdms", flush = True)
                twoeint2 = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
                twoeint2 = twoeint2[:self.n_occupied,:self.n_occupied,:self.n_occupied,:self.n_occupied]
                for i in range(self.davidson_roots):
                    sum_energy = 0.0
                    off_diagonal_constant_energy = 0.0
                    photon_energy = 0.0
                    eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
                    eigenvecs2 = eigenvecs2.transpose(1,0)
                    for m in range(np1):
                        if (self.N_p == 0): continue
                        if (m > 0 and m < self.N_p):
                            off_diagonal_constant_energy  += np.sqrt(m * self.omega/2) * self.d_exp * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                            off_diagonal_constant_energy += np.sqrt((m+1) * self.omega/2) * self.d_exp * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                        elif (m == self.N_p):
                            off_diagonal_constant_energy  += np.sqrt(m * self.omega/2) * self.d_exp * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                        else:
                            off_diagonal_constant_energy  += np.sqrt((m+1) * self.omega/2) * self.d_exp * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                        photon_energy  += m * self.omega * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,m:(m+1)].flatten())

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
                        i
                    )
                    two_rdm = np.zeros((self.n_occupied * self.n_occupied * self.n_occupied * self.n_occupied))
                    c_build_two_rdm(
                        eigenvecs,
                        two_rdm,
                        self.table,
                        self.n_act_a,
                        self.n_act_orb,
                        self.n_in_a,
                        np1,
                        i,
                        i
                    )
                    #for t in range(self.n_occupied):
                    #    for u in range(self.n_occupied):
                    #        tu = t * self.n_occupied + u
                    #        for v in range(self.n_occupied):
                    #            for w in range(self.n_occupied):
                    #                vw = v * self.n_occupied + w
                    #                print(two_rdm[tu * self.n_occupied * self.n_occupied + vw],  
                    #                   two_rdm2[tu * self.n_occupied * self.n_occupied + vw],  
                    #                   two_rdm[tu * self.n_occupied * self.n_occupied + vw]-  
                    #                   two_rdm2[tu * self.n_occupied * self.n_occupied + vw], t,u,v,w,
                    #                   (t * self.n_occupied + u)*self.n_occupied*self.n_occupied + v*self.n_occupied + w
                    #                   )





                    Dpe = np.zeros((self.n_occupied * self.n_occupied))
                    c_build_photon_electron_one_rdm(eigenvecs,
                            Dpe,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            self.n_in_a,
                            np1,
                            i,
                            i
                    )

                    #two_rdm2 = two_rdm.reshape((self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied))
                    #print(two_rdm2[:(self.n_in_a*self.n_in_a),:(self.n_in_a*self.n_in_a)])
                    #np.savetxt('correct_rdm.txt', two_rdm2) 
                    
                    #one_rdm2 = np.zeros((self.nmo * self.nmo))
                    #for p in range(self.nmo):
                    #    for q in range(self.nmo):
                    #        dum = 0.0
                    #        for r in range(self.nmo):
                    #            dum += 0.5/(self.n_act_a+self.n_in_a-0.5) * two_rdm[p * self.nmo * self.nmo * self.nmo + r * self.nmo * self.nmo + q * self.nmo + r]
                    #        one_rdm2[p * self.nmo + q] = dum
                            
                    one_e_energy = np.dot(self.H_spatial2[:self.n_occupied,:self.n_occupied].flatten(), one_rdm)
                    two_e_energy = 0.5 * np.dot(twoeint2.flatten(), two_rdm)
                    one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[:self.n_occupied,:self.n_occupied].flatten(), Dpe)
                    sum_energy = one_e_energy + two_e_energy + self.Enuc + one_pe_energy + off_diagonal_constant_energy + self.d_c + photon_energy
                    #print("1e integral")
                    #for k in range(self.n_occupied): 
                    #    for l in range(self.n_occupied):
                    #        print("{:20.16f}".format(self.H_spatial2[k,l]), k, l, flush = True)
                    #print("1-rdm")   
                    #for k in range(self.n_occupied): 
                    #    for l in range(self.n_occupied):
                    #        print("{:20.16f}".format(one_rdm[k * self.n_occupied + l]), k, l, flush = True)
                    #print("2e integral")
                    #for k in range(self.n_occupied): 
                    #    for l in range(self.n_occupied):
                    #        for m in range(self.n_occupied):
                    #            for n in range(self.n_occupied):
                    #                print("{:20.16f}".format(twoeint2[k,l,m,n]), k, l, m, n, flush = True)
                    #print("2-rdm")
                    #for k in range(self.n_occupied): 
                    #    for l in range(self.n_occupied):
                    #        for m in range(self.n_occupied):
                    #            for n in range(self.n_occupied):
                    #                print("{:20.16f}".format(two_rdm[k * self.n_occupied * self.n_occupied * self.n_occupied + 
                    #                    l* self.n_occupied * self.n_occupied + m * self.n_occupied +n]), k, l, m, n, flush = True)


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
                        "{:20.12f}".format(eigenvals[i] - sum_energy, flush = True)
                    )
                
                print("check total energy using active rdms")
                print("{:10s}".format("state"), "{:20s}".format("active_one"), "{:20s}".format("active_two"), 
                        "{:20s}".format("eigenvalues"), "{:20s}".format("total energies"), "error")
                active_twoeint = twoeint2[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]
                active_fock_core = self.fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]
                for i in range(self.davidson_roots):
                    self.D_tu = np.zeros((self.n_act_orb * self.n_act_orb))
                    self.Dpe_tu = np.zeros((self.n_act_orb * self.n_act_orb))
                    self.D_tuvw = np.zeros((self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb))
                    sum_energy = 0.0
                    off_diagonal_constant_energy = 0.0
                    photon_energy = 0.0
                    eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
                    eigenvecs2 = eigenvecs2.transpose(1,0)
                    for m in range(np1):
                        if (self.N_p ==0): continue
                        if (m > 0 and m < self.N_p):
                            off_diagonal_constant_energy  += np.sqrt(m * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                            off_diagonal_constant_energy += np.sqrt((m+1) * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                        elif (m == self.N_p):
                            off_diagonal_constant_energy  += np.sqrt(m * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                        else:
                            off_diagonal_constant_energy  += np.sqrt((m+1) * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                        photon_energy  += m * self.omega * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,m:(m+1)].flatten())
                    c_build_active_rdm(eigenvecs,
                            self.D_tu,
                            self.D_tuvw,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            np1,
                            i,
                            i,
                            1.0
                    )
                    c_build_active_photon_electron_one_rdm(eigenvecs,
                            self.Dpe_tu,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            np1,
                            i,
                            i,
                            1.0
                    )
                    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu)
                    active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw)
                    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu)
                    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                            self.Enuc + self.d_c + off_diagonal_constant_energy + photon_energy)

                    
                    print(
                        "{:4d}".format(i),
                        "{:20.12f}".format(active_one_e_energy),
                        "{:20.12f}".format(active_two_e_energy),
                        "{:20.12f}".format(eigenvals[i]),
                        "{:20.12f}".format(sum_energy),
                        "{:20.12f}".format(eigenvals[i] - sum_energy)

                    )
                #print(self.Dpe_tu.reshape((self.n_act_orb,self.n_act_orb))-self.Dpe_tu.reshape((self.n_act_orb,self.n_act_orb)).transpose()) 
                #self.gkl3 = np.zeros((self.n_act_orb, self.n_act_orb))
                #self.twoeint3 = np.zeros((self.nmo * self.nmo, self.nmo * self.nmo))
                #
                #S = np.zeros((self.davidson_roots, H_dim))
                #c_sigma(
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
                #)
                #eig_mat = np.einsum("pq,rq",S, eigenvecs)
                #np.set_printoptions(precision=12)
                #print(np.diag(eig_mat))
                ################################################################self.build_JK()
                self.build_state_avarage_rdms(eigenvecs)
                #print(self.D_tu_avg)
                #print("qrqr")
                #print(self.D_tu_avg2)
                #print("qrqr2")
                ##print(self.D_tuvw_avg)
                #print("qrqr3")
                ##print(self.D_tuvw_avg2)
                #for t in range(self.n_act_orb):
                #    for u in range(self.n_act_orb):
                #        tu = t * self.n_act_orb + u
                #        ut = u * self.n_act_orb + t
                #        for v in range(self.n_act_orb):
                #            for w in range(self.n_act_orb):
                #                vw = v * self.n_act_orb + w
                #                print(self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw],  
                #                 self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw],
                #                 self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw]-
                #                 self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw],t+self.n_in_a,u+self.n_in_a,v+self.n_in_a,w+self.n_in_a,
                #                 tu * self.n_act_orb * self.n_act_orb + vw,
                #                 ((t+self.n_in_a) * self.n_occupied + u+self.n_in_a)*self.n_occupied*self.n_occupied + (v+self.n_in_a)*self.n_occupied + w+self.n_in_a)

                #####get state-avaraged rdms
                ####self.D_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
                ####self.Dpe_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
                ####self.D_tuvw_avg = np.zeros((self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb))
                ####for i in range(self.davidson_roots):
                ####    c_build_active_rdm(eigenvecs,
                ####            self.D_tu_avg,
                ####            self.D_tuvw_avg,
                ####            self.table,
                ####            self.n_act_a,
                ####            self.n_act_orb,
                ####            np1,
                ####            i,
                ####            i,
                ####            self.weight[i]
                ####    )
                ####    c_build_active_photon_electron_one_rdm(eigenvecs,
                ####            self.Dpe_tu_avg,
                ####            self.table,
                ####            self.n_act_a,
                ####            self.n_act_orb,
                ####            np1,
                ####            i,
                ####            i,
                ####            self.weight[i]
                ####    )
                #####for t in range(self.n_act_orb):
                #####    for v in range(self.n_act_orb):
                #####        for u in range(self.n_act_orb):
                #####            for w in range(self.n_act_orb):
                #####                tv = t * self.n_act_orb + v
                #####                uw = u * self.n_act_orb + w
                #####                dum = self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + uw]-self.D_tuvw_avg[uw * self.n_act_orb * self.n_act_orb + tv]
                #####                if dum > 1e-9: print('{:.5f}'.format(dum))



                ####self.D_tuvw_avg2 = np.zeros((self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb))
                #####symmetrize 2-rdm
                ####for t in range(self.n_act_orb):
                ####    for u in range(t,self.n_act_orb):
                ####        tu = t * self.n_act_orb + u
                ####        ut = u * self.n_act_orb + t
                ####        for vw in range(self.n_act_orb * self.n_act_orb):
                ####            dum = (self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] + 
                ####             self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw])
                ####            #dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu] 
                ####            self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                ####            self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                ####            #self.D_tu_avg[tu] = dum2/2.0
                ####            #self.D_tu_avg[ut] = dum2/2.0
                
                #for t in range(self.n_act_orb):
                #    for v in range(self.n_act_orb):
                #        for u in range(self.n_act_orb):
                #            for w in range(self.n_act_orb):
                #                tv = t * self.n_act_orb + v
                #                vt = v * self.n_act_orb + t
                #                uw = u * self.n_act_orb + w
                #                wu = w * self.n_act_orb + u
                #                dum = self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + uw]-self.D_tuvw_avg[vt * self.n_act_orb * self.n_act_orb + uw]
                #                #dum = self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + uw]-self.D_tuvw_avg[uw * self.n_act_orb * self.n_act_orb + tv]
                #                #dum = self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + uw]-self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + wu]
                #                if dum > 1e-14: print("hiha",'{:.12f}'.format(dum))
                
                #test avarage energy
                print("test avarage energy")
                avg_energy = 0.0
                for i in range(self.davidson_roots):
                    avg_energy += self.weight[i] * eigenvals[i]
                    #print(self.weight[i], eigenvals[i], self.weight[i] * eigenvals[i])
                active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                        self.Enuc + self.d_c + ci_dependent_energy)
                print(
                    "{:20.12f}".format(sum_energy),
                    "{:20.12f}".format(avg_energy),
                )
                #print(
                #    "{:20.12f}".format(sum_energy),
                #    "{:20.12f}".format(active_one_e_energy),
                #    "{:20.12f}".format(active_two_e_energy),
                #    "{:20.12f}".format(active_one_pe_energy),
                #    "{:20.12f}".format(self.E_core),
                #    "{:20.12f}".format(self.Enuc),
                #    "{:20.12f}".format(self.d_c),
                #    "{:20.12f}".format(ci_dependent_energy),
                #)

                self.n_virtual = self.nmo - self.n_occupied
                n_ai = self.n_in_a * self.n_act_orb
                n_vi = self.n_in_a * self.n_virtual
                n_va = self.n_act_orb * self.n_virtual
                #build index map
                self.index_map_size = n_ai + n_vi + n_va
                self.index_map = np.zeros((self.index_map_size, 2), dtype = int)
                index_count = 0 
                for r in range(self.nmo):
                    for s in range(r+1,self.nmo):
                        if (r < self.n_in_a and s < self.n_in_a): continue
                        if (self.n_in_a <= r < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                        if (r >= self.n_occupied and s >= self.n_occupied): continue
                        self.index_map[index_count][0] = s 
                        self.index_map[index_count][1] = r
                        index_count += 1

                self.index_map1 = self.index_map.astype('int32')
                #print(self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.index_map)        
                ############################################TEST SIGMA BUILD#############################################
                ####Rai= np.random.rand(self.n_act_orb,self.n_in_a)
                #####Rvi = np.zeros((self.n_virtual,self.n_in_a))
                ####Rvi= np.random.rand(self.n_virtual,self.n_in_a)
                #####Rva = np.zeros((self.n_virtual,self.n_act_orb))
                ####Rva= np.random.rand(self.n_virtual,self.n_act_orb)


                ####Rai2= np.random.rand(self.n_act_orb,self.n_in_a)
                #####Rvi = np.zeros((self.n_virtual,self.n_in_a))
                ####Rvi2= np.random.rand(self.n_virtual,self.n_in_a)
                #####Rva = np.zeros((self.n_virtual,self.n_act_orb))
                ####Rva2= np.random.rand(self.n_virtual,self.n_act_orb)
                ####self.build_unitary_matrix(Rai2, Rvi2, Rva2)
                ####print("U_delta", self.U_delta)



                ####R_reduced = np.zeros(self.index_map_size)
                ####for i in range(self.index_map_size):
                ####    s = self.index_map[i][0]   
                ####    l = self.index_map[i][1] 
                ####    if s >=self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                ####        R_reduced[i] = Rai[s-self.n_in_a][l]
                ####    if s >= self.n_occupied and l < self.n_in_a:
                ####        R_reduced[i] = Rvi[s-self.n_occupied][l]
                ####    if s >= self.n_occupied and l >= self.n_in_a and l < self.n_occupied:
                ####        R_reduced[i] = Rva[s-self.n_occupied][l-self.n_in_a]


                ####print(R_reduced)
                ####R_total = np.zeros((self.nmo,self.n_occupied))
                ####for k in range(self.n_occupied):
                ####    for r in range(self.nmo):
                ####        if r >= self.n_in_a and r < self.n_occupied and k < self.n_in_a:
                ####            R_total[r][k] = Rai[r-self.n_in_a][k]
                ####        if r >= self.n_occupied and k < self.n_in_a:
                ####            R_total[r][k] = Rvi[r-self.n_occupied][k]
                ####        if r >= self.n_occupied and k >= self.n_in_a and k < self.n_occupied:
                ####            R_total[r][k] = Rva[r-self.n_occupied][k-self.n_in_a]
                ####
                ####self.U = np.eye(self.nmo)
                ####rot_dim = self.nmo
                ####A = np.zeros((rot_dim, rot_dim))
                ####G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####print("initial e_core", self.E_core)
                ####self.build_intermediates(eigenvecs, A, G, True)
                ####A2 = np.zeros((rot_dim, rot_dim))
                ####G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####self.build_intermediates2(eigenvecs, A2, G2, True)
                ####print("test second order energy and ci updated integrals")
                ####exact_t_energy = self.microiteration_exact_energy(self.U_delta, A, G)
                ####print("exact_energy from second order expansion", exact_t_energy + avg_energy)
                ####active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
                ####active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                ####d_cmo = np.zeros((self.nmo, self.nmo))
                ####self.microiteration_ci_integrals_transform(self.U_delta, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                ####active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                ####active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                ####active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ####ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                ####sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                ####        self.Enuc + self.d_c + ci_dependent_energy)
                ####print("gfhgy",
                ####    "{:20.12f}".format(sum_energy),
                ####    "{:20.12f}".format(active_one_e_energy),
                ####    "{:20.12f}".format(active_two_e_energy),
                ####    "{:20.12f}".format(self.E_core),
                ####    "{:20.12f}".format(active_one_pe_energy),
                ####    "{:20.12f}".format(self.Enuc),
                ####)

                #####self.U_delta = np.eye(self.nmo)

                ####self.build_sigma_reduced(self.U_delta, A, G, R_reduced)
                ####print("compare rtotal and rtotal2")
                ####for l in range(self.n_occupied):
                ####    for s in range(self.nmo):
                ####        print('{:.12f}'.format(R_total[s][l]), '{:.12f}'.format(self.R_total[s][l]), s,l)
                ####gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                ####hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####self.build_gradient_and_hessian(self.U_delta, A, G, gradient_tilde, hessian_tilde, True)
                ####print(np.shape(gradient_tilde))
                ####print("occupied-full gradient",gradient_tilde)
                ####print(A)
                ####






                ####hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                ####hessian_tilde4 = np.zeros((self.nmo,self.n_occupied, self.nmo,self.n_occupied))
                ####hessian_tilde4[:,:,:,:] = hessian_tilde.transpose(2,0,3,1)[:,:,:,:]
                ####hessian_tilde3=hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))
                ####omega2, eig_vecs2 = np.linalg.eigh(hessian_tilde3)


                ####self.reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                ####index_count1 = 0 
                ####for k in range(self.n_occupied):
                ####    for r in range(k+1,self.nmo):
                ####        if (k < self.n_in_a and r < self.n_in_a): continue
                ####        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                ####        #print(r,k,index_count1)
                ####        index_count2 = 0 
                ####        for l in range(self.n_occupied):
                ####            for s in range(l+1,self.nmo):
                ####                if (l < self.n_in_a and s < self.n_in_a): continue
                ####                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                ####                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                ####                self.reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                ####                #print(r,k,s,l,index_count1,index_count2)
                ####                index_count2 += 1
                ####        index_count1 += 1

                ####print("eere", np.shape(R_total),np.shape(self.reduced_hessian))
                ####omega3, eig_vecs3 = np.linalg.eigh(self.reduced_hessian)
                #####print(omega2)
                ####print(omega3)
                ####
                #####test sigma from reduced and occupied-full hessian
                ####sigma_reduced = np.einsum("pq,q->p", self.reduced_hessian, R_reduced)
                ####sigma_total = np.einsum("pq,q->p", hessian_tilde3, R_total.reshape(self.nmo * self.n_occupied))
                ####sigma_total=sigma_total.reshape((self.nmo, self.n_occupied))
                ########for k in range(self.n_occupied):
                ########    for r in range(self.nmo):
                ########        for l in range(self.n_occupied):
                ########            for s in range(self.nmo):
                ########                print('{:.12f}'.format(hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]), '{:.12f}'.format(R_total[s][l]), r,k,s,l)
                ####print("test sigma reduced and sigma")
                ########for i in range(self.index_map_size):
                ########    s = self.index_map[i][0]   
                ########    l = self.index_map[i][1]  
                ########    for j in range(self.index_map_size):
                ########        r = self.index_map[j][0] 
                ########        k = self.index_map[j][1] 
                ########        print(self.reduced_hessian[i][j] - hessian_tilde3[s * self.n_occupied + l][r * self.n_occupied + k],'{:.12f}'.format(R_reduced[j]),'{:.12f}'.format(R_total[r][k]), s,l,r,k)


                ####for i in range(self.index_map_size):
                ####    s = self.index_map[i][0] 
                ####    l = self.index_map[i][1] 
                ####    print(sigma_reduced[i], sigma_total[s][l], self.sigma_total[s][l])






                ####R_reduced2 = np.random.rand(3, self.index_map_size)
                ####
                ####sigma_reduced2 = np.zeros((3, self.index_map_size)) 
                ####sigma_reduced3 = np.einsum("pq,iq->ip", self.reduced_hessian, R_reduced2) 

 
                ####gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
                ####A_tilde2 = np.zeros((rot_dim, rot_dim))
                ####self.build_gradient(self.U_delta, A, G, gradient_tilde2, A_tilde2, True)




                ####self.build_sigma_reduced2(self.U_delta, A_tilde2, G, R_reduced2, sigma_reduced2, 3, 0)



                ####print("vobk")
                ####for i in range(3):
                ####    for j in range(self.index_map_size):
                ####        s = self.index_map[j][0] 
                ####        l = self.index_map[j][1] 
                ####        print(sigma_reduced3[i][j], sigma_reduced2[i][j])




                #################################end_test_sigma_build###########################################################
















                #np.savetxt('gradient1.txt', gradient_tilde) 
                #np.savetxt('hessian1.txt', hessian_tilde.transpose(0,2,1,3).reshape(self.n_occupied * self.nmo, self.n_occupied * self.nmo)) 
                

                #####internal rotations
                ####Rvi = np.zeros((self.n_virtual,self.n_in_a))
                ####Rva = np.zeros((self.n_virtual,self.n_act_orb))

                ####self.build_unitary_matrix(Rai, Rvi, Rva)
                ####rot_dim = self.n_occupied
                ####A1 = np.zeros((rot_dim, rot_dim))
                ####G1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####self.build_intermediates(eigenvecs, A1, G1, False)
                ####
                ####gradient_tilde1 = np.zeros((rot_dim, self.n_occupied))
                ####hessian_tilde1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####gradient_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a))
                ####gradient_tilde_ai2 = np.zeros((self.n_act_orb, self.n_in_a))
                ####hessian_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a))
                ####hessian_tilde_ai2 = np.zeros((self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a))
               
                ####self.build_gradient_and_hessian(self.U, A1, G1, gradient_tilde1, hessian_tilde1, False)
                ####hessian_tilde_ai[:,:,:,:] = hessian_tilde1.transpose(2,0,3,1)[self.n_in_a:self.n_occupied, :self.n_in_a, self.n_in_a:self.n_occupied, :self.n_in_a]
                ####gradient_tilde_ai[:,:] = gradient_tilde1[self.n_in_a:self.n_occupied, :self.n_in_a]
                ####
                ####for a in range(self.n_act_orb):
                ####    for i in range(self.n_in_a):
                ####        gradient_tilde_ai2[a][i] = gradient_tilde[a+self.n_in_a][i]
                ####        print(gradient_tilde_ai2[a][i] - gradient_tilde_ai[a][i], a+self.n_in_a,i)
                ####        for b in range(self.n_act_orb):
                ####            for j in range(self.n_in_a):
                ####                hessian_tilde_ai2[a][i][b][j] = hessian_tilde3[(a+self.n_in_a) * self.n_occupied + i][(b+self.n_in_a) * self.n_occupied + j]
                ####                #print(hessian_tilde_ai2[a][i][b][j] - hessian_tilde_ai[a][i][b][j], a+self.n_in_a,i,b+self.n_in_a,j)
                ####
                ####print("dfbg", gradient_tilde1)
                ####print(np.shape(gradient_tilde1))
                ####print("mjys", np.allclose(gradient_tilde[:self.n_occupied,:],gradient_tilde1, rtol=1e-14,atol=1e-14))
                ####print("kljh", np.allclose(hessian_tilde[:,:,:self.n_occupied,:self.n_occupied],hessian_tilde1, rtol=1e-14,atol=1e-14))
                ####print("ntbf", np.allclose(G[:,:,:self.n_occupied,:self.n_occupied],G1, rtol=1e-14,atol=1e-14))
                ####print("xcuf", np.allclose(A[:self.n_occupied,:self.n_occupied],A1, rtol=1e-14,atol=1e-14))
                ####
                ####A10 = np.zeros((rot_dim, rot_dim))
                ####G10 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####self.build_intermediates_internal(eigenvecs, A10, G10, self.occupied_fock_core, self.occupied_d_cmo, self.occupied_J, self.occupied_K)
                ####print("ynbi", np.allclose(G[:,:,:self.n_occupied,:self.n_occupied],G10, rtol=1e-14,atol=1e-14))
                ####print("omlq", np.allclose(A[:self.n_occupied,:self.n_occupied],A10, rtol=1e-14,atol=1e-14))
                

                ####J2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####K2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ####h1 = np.zeros((rot_dim, rot_dim))
                ####d_cmo2 = np.zeros((rot_dim, rot_dim))
                ####self.internal_transformation(self.U_delta,h1,d_cmo2, J2,K2)
                ####self.internal_transformation(self.U_delta,self.H_spatial2, self.d_cmo, self.J,self.K)

                ####print("internal transformation test")
                ####print("nguh", np.allclose(K2,self.K[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
                ####print("gtro", np.allclose(J2,self.J[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
                ####print("vfao", np.allclose(h1,self.H_spatial2[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
                ####print("tyuo", np.allclose(d_cmo2,self.d_cmo[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))


                #####print("full internal transformation test")
                #####J5 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #####K5 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #####h1e = np.zeros((self.nmo, self.nmo))
                #####d_cmo1 = np.zeros((self.nmo, self.nmo))
                #####self.full_transformation_internal_optimization(self.U_delta, h1e, d_cmo1, J5, K5) 
                #####print("full internal transformation test")
                #####self.full_transformation_internal_optimization(self.U_delta, self.H_spatial2, self.d_cmo, self.J, self.K) 
                ##### 
                ##### print("ygno", np.allclose(K5,self.K, rtol=1e-14,atol=1e-14))
                ##### print("mkgl", np.allclose(J5,self.J, rtol=1e-14,atol=1e-14))
                ##### print("vdbe", np.allclose(h1e,self.H_spatial2, rtol=1e-14,atol=1e-14))
                ##### print("nmtv", np.allclose(d_cmo1,self.d_cmo, rtol=1e-14,atol=1e-14))
                ####
                ####
                ######test energy
                #####internal_exact_energy = self.internal_optimization_exact_energy(avg_energy, eigenvecs)
                #####internal_predicted_energy = self.internal_optimization_predicted_energy(gradient_tilde_ai, hessian_tilde_ai, Rai)
                #####print(internal_exact_energy, internal_predicted_energy)
                


                #####self.full_one_rdm = np.zeros((self.nmo, self.nmo))
                #####one_rdm = np.zeros((self.n_occupied * self.n_occupied))
                #####c_build_one_rdm(
                #####    eigenvecs,
                #####    one_rdm,
                #####    self.table,
                #####    self.n_act_a,
                #####    self.n_act_orb,
                #####    self.n_in_a,
                #####    np1,
                #####    0,
                #####    0
                #####)
                #####self.full_one_rdm[:self.n_occupied,:self.n_occupied] = one_rdm.reshape((self.n_occupied, self.n_occupied))
                #####self.full_two_rdm = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                #####two_rdm = np.zeros((self.n_occupied * self.n_occupied * self.n_occupied * self.n_occupied))
                #####two_rdm2 = np.zeros((self.n_occupied * self.n_occupied * self.n_occupied * self.n_occupied))
                #####c_build_two_rdm(
                #####    eigenvecs,
                #####    two_rdm,
                #####    two_rdm2,
                #####    self.table,
                #####    self.n_act_a,
                #####    self.n_act_orb,
                #####    self.n_in_a,
                #####    np1,
                #####    0,
                #####    0
                #####)
                #####self.full_two_rdm[:self.n_occupied,:self.n_occupied,:self.n_occupied,:self.n_occupied] = two_rdm.reshape((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #####self.twoeint4=self.twoeint.reshape((self.nmo,self.nmo,self.nmo,self.nmo)) 
                ######test full gradient and hessian with usual formulation
                #####hessian_full = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                #####gradient_full = np.zeros((self.nmo, self.nmo))
                #####hessian_full1 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                dum = 0.0
                #####                dum -= 2.0 * self.H_spatial2[s][p] * self.full_one_rdm[r][q]
                #####                dum -= 2.0 * self.H_spatial2[q][r] * self.full_one_rdm[p][s]
                #####                for u in range(self.nmo):
                #####                    dum += (q==r) * self.H_spatial2[u][p] * self.full_one_rdm[u][s]
                #####                    dum += (q==r) * self.H_spatial2[s][u] * self.full_one_rdm[p][u]
                #####                    dum += (p==s) * self.H_spatial2[u][r] * self.full_one_rdm[u][q]
                #####                    dum += (p==s) * self.H_spatial2[q][u] * self.full_one_rdm[r][u]
                #####                hessian_full[p][r][q][s] = 0.25 * dum
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                dum = 0.0
                #####                for t in range(self.nmo):
                #####                    for u in range(self.nmo):
                #####                        for v in range(self.nmo):
                #####                            dum += (q==r) * self.twoeint4[u][p][v][t] * self.full_two_rdm[u][s][v][t]
                #####                            dum += (q==r) * self.twoeint4[s][u][t][v] * self.full_two_rdm[p][u][t][v]
                #####                            dum += (p==s) * self.twoeint4[q][u][t][v] * self.full_two_rdm[r][u][t][v]
                #####                            dum += (p==s) * self.twoeint4[u][r][v][t] * self.full_two_rdm[u][q][v][t]
                #####                hessian_full[p][r][q][s] += 0.25 * dum
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                dum = 0.0
                #####                for u in range(self.nmo):
                #####                    for v in range(self.nmo):
                #####                        dum += self.twoeint4[u][p][v][r] * self.full_two_rdm[u][q][v][s]
                #####                        dum += self.twoeint4[q][u][s][v] * self.full_two_rdm[p][u][r][v]
                #####                hessian_full[p][r][q][s] += 0.5 * dum
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                dum = 0.0
                #####                for t in range(self.nmo):
                #####                    for u in range(self.nmo):
                #####                        dum += self.twoeint4[s][p][t][u] * self.full_two_rdm[r][q][t][u]
                #####                        dum += self.twoeint4[t][p][s][u] * self.full_two_rdm[t][q][r][u]
                #####                        dum += self.twoeint4[q][r][u][t] * self.full_two_rdm[p][s][u][t]
                #####                        dum += self.twoeint4[q][t][u][r] * self.full_two_rdm[p][t][u][s]
                #####                hessian_full[p][r][q][s] -= 0.5 * dum
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                hessian_full1[p][q][r][s]  = hessian_full[p][r][q][s]
                #####                hessian_full1[p][q][r][s] -= hessian_full[q][r][p][s]
                #####                hessian_full1[p][q][r][s] -= hessian_full[p][s][q][r]
                #####                hessian_full1[p][q][r][s] += hessian_full[q][s][p][r]
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        dum = 0.0
                #####        for r in range(self.nmo):
                #####            dum += 1.0 * self.H_spatial2[r][p] * self.full_one_rdm[r][q]
                #####            dum -= 1.0 * self.H_spatial2[q][r] * self.full_one_rdm[p][r]
                #####        gradient_full[p][q] = dum
                #####for p in range(self.nmo):
                #####    for q in range(self.nmo):
                #####        dum = 0.0
                #####        for r in range(self.nmo):
                #####            for s in range(self.nmo):
                #####                for t in range(self.nmo):
                #####                    dum += self.twoeint4[r][p][s][t] * self.full_two_rdm[r][q][s][t]
                #####                    dum -= self.twoeint4[q][r][t][s] * self.full_two_rdm[p][r][t][s]
                #####        gradient_full[p][q] += dum
               


                #####reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                #####reduced_gradient = np.zeros(self.index_map_size)
                #####index_count1 = 0 
                #####for k in range(self.n_occupied):
                #####    for r in range(k+1,self.nmo):
                #####        if (k < self.n_in_a and r < self.n_in_a): continue
                #####        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #####        reduced_gradient[index_count1] = gradient_full[r][k]
                #####        #print(r,k,index_count1)
                #####        index_count2 = 0 
                #####        for l in range(self.n_occupied):
                #####            for s in range(l+1,self.nmo):
                #####                if (l < self.n_in_a and s < self.n_in_a): continue
                #####                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #####                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #####                reduced_hessian[index_count1][index_count2] = hessian_full1[r][k][s][l]
                #####                index_count2 += 1
                #####        index_count1 += 1
                #####omega100, eig_vecs100 = np.linalg.eigh(reduced_hessian)
                #####np.set_printoptions(precision = 14, suppress = True)  
                #####print(np.linalg.norm(reduced_gradient))
                #####print(omega100)
                #####reduced_hessian2 = np.zeros((self.index_map_size, self.index_map_size))
                #####reduced_gradient2 = np.zeros(self.index_map_size)
                #####for p in range(self.index_map_size):
                #####    r = self.index_map[p][0]  
                #####    k = self.index_map[p][1]  
                #####    reduced_gradient2[p] = gradient_full[r][k]
                #####    for q in range(self.index_map_size):
                #####        s = self.index_map[q][0]  
                #####        l = self.index_map[q][1]  
                #####        reduced_hessian2[p][q] = hessian_full1[r][k][s][l]
                #####omega200, eig_vecs200 = np.linalg.eigh(reduced_hessian2)
                #####print(np.linalg.norm(reduced_gradient2))
                #####print(omega200)


                #########hessian_full2 = hessian_full1.reshape(self.nmo * self.nmo,self.nmo * self.nmo)
                #########omega10, eig_vecs10 = np.linalg.eigh(hessian_full2)
                #########print("cgfn",omega10)
                #####hessian_ai = np.zeros((self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a))
                #####gradient_ai = np.zeros((self.n_act_orb, self.n_in_a))
                #####for a in range(self.n_act_orb):
                #####    for i in range(self.n_in_a):
                #####        gradient_ai[a][i]  = gradient_full[a+self.n_in_a][i];
                #####        for b in range(self.n_act_orb):
                #####            for j in range(self.n_in_a):
                #####                hessian_ai[a][i][b][j]  = hessian_full1[a+self.n_in_a][i][b+self.n_in_a][j];
                #####hessian_ai1 = hessian_ai.reshape((self.n_act_orb * self.n_in_a, self.n_act_orb * self.n_in_a))
                #####omega11, eig_vecs11 = np.linalg.eigh(hessian_ai1)
                #####print("zofd",omega11)
                #####print("trgm",gradient_ai)
                #####print("vcgi",hessian_ai1)
                #####gradient_ai1 = gradient_ai.reshape((self.n_act_orb * self.n_in_a))

                #####dim4 = self.n_act_orb * self.n_in_a + 1
                #####aug_hessian = np.zeros((dim4, dim4))
                #####aug_hessian[0, 1:] = gradient_ai1
                #####aug_hessian[1:, 0] = gradient_ai1.T
                #####aug_hessian[1:, 1:] = hessian_ai1
                #####omega13, eig_vecs13 = np.linalg.eigh(aug_hessian)
                #####print("eezo", aug_hessian) 
                #####print("vktj",omega13)


                #####self.printA(hessian_ai1)
                #####print("fcer")
                #####self.printA(aug_hessian)

                
                ####hessian_reduced10 = np.zeros((self.nmo, self.n_occupied, self.nmo, self.n_occupied))
                ####for r in range(self.nmo):
                ####    for k in range(self.n_occupied):
                ####        rk = r*self.n_occupied +k
                ####        for s in range(self.nmo):
                ####            for l in range(self.n_occupied):
                ####                sl = s*self.n_occupied +l
                ####                hessian_reduced10[r][k][s][l] = hessian_full1[r][k][s][l]


                ####hessian_reduced10 = hessian_reduced10.reshape((self.nmo * self.n_occupied, self.nmo * self.n_occupied))
                ####print("utto", np.allclose(hessian_reduced10,hessian_tilde3, rtol=1e-14,atol=1e-14))


                ####self.full_two_rdm2 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                ####self.full_two_rdm2[:,:,:,:] = self.full_two_rdm[:,:,:,:]
                #####for t in range(self.nmo):
                #####    for u in range(t,self.nmo):
                #####        tu = t * self.nmo + u
                #####        ut = u * self.nmo + t
                #####        for v in range(self.nmo):
                #####            for w in range(self.nmo):
                #####                print("{:20.12f}".format(self.full_two_rdm2[t][u][v][w])) 
                ####



                #####symmetrize 2-rdm
                ####for t in range(self.nmo):
                ####    for u in range(t,self.nmo):
                ####        tu = t * self.nmo + u
                ####        ut = u * self.nmo + t
                ####        for v in range(self.nmo):
                ####            for w in range(self.nmo):
                ####                dum = self.full_two_rdm[t][u][v][w]+ self.full_two_rdm[u][t][v][w]  
                ####                #dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu] 
                ####                self.full_two_rdm[t][u][v][w] = dum/2.0
                ####                self.full_two_rdm[u][t][v][w] = dum/2.0
                ####                #self.D_tu_avg[tu] = dum2/2.0
                ####                #self.D_tu_avg[ut] = dum2/2.0
                ####
 
                ####A3 = np.zeros((self.nmo, self.nmo))
                ####A3[:,:] = np.einsum("rt,tu->ru", self.H_spatial2,
                ####        self.full_one_rdm)
                #####print(np.shape(self.active_twoeint))
                ####A3[:,:] += np.einsum("rtvw,tuvw->ru", self.twoeint4, 
                ####   self.full_two_rdm)
                ####gradient_tilde3 = A3[:,:] - A3.T[:,:]
                #####gradient2_tilde = np.zeros((self.nmo, self.n_occupied))
                #####for k in range(self.n_occupied):
                #####    for r in range(self.nmo):
                #####        gradient2_tilde[r][k] = A_tilde[r][k] - A_tilde[k][r]
                #####print("ytuu", np.allclose(gradient2_tilde,gradient_tilde, rtol=1e-14,atol=1e-14))
                ####gradient_norm = np.dot(gradient_tilde3.flatten(), gradient_tilde3.flatten())
                ####gradient_norm = np.sqrt(gradient_norm)
                ####np.set_printoptions(precision=9, suppress=True)
                ####for k in range(self.n_occupied):
                ####    for r in range(self.nmo):
                ####        if(gradient_tilde3[r][k]>1e-10):
                ####            print("efgfb",gradient_tilde[r][k]) 
                ####print("full gradient before antisymmetrizing",A3)
                ####print(gradient_norm) 
           


                ####G3 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                ####G3 = np.einsum("rs,tu->turs", self.H_spatial2,
                ####   self.full_one_rdm)
                ####G3 += np.einsum("vwrs,tuvw->turs", self.twoeint4,
                ####   self.full_two_rdm)
                ####G3 += 2.0 * np.einsum("vrws,tvuw->turs", self.twoeint4,
                ####   self.full_two_rdm)
                ####A4 = A3 + A3.T

                ####G3 -= 0.5  * np.einsum("ij,rs->ijrs", np.eye(self.nmo),
                ####   A4)

                ####
                ####G4 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))
                ####G4 = G3 -G3.transpose(2,1,0,3)
                ####G4 -= G3.transpose(0,3,2,1)
                ####G4 += G3.transpose(2,3,0,1)
                ####G5 = G4.transpose(2,0,3,1)
                ####print("yrbt", np.allclose(hessian_tilde4,G5[:,:self.n_occupied,:,:self.n_occupied], rtol=1e-14,atol=1e-14))
               
                ####G6 = G5.reshape(self.nmo * self.nmo,self.nmo * self.nmo)
                ####print(np.shape(G5))
                ####print(np.shape(hessian_tilde3))
                #####with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
                #####   print(G5)
                ####for r in range(self.nmo):
                ####    for k in range(self.n_occupied):
                ####        rk = r*self.n_occupied +k
                ####        for s in range(self.nmo):
                ####            for l in range(self.n_occupied):
                ####                sl = s*self.n_occupied +l
                ####                dum = hessian_tilde4[r][k][s][l]-G6[r*self.nmo+k][s*self.nmo+l]
                ####                dum2 = hessian_tilde3[rk][sl]-G6[r*self.nmo+k][s*self.nmo+l]
                ####                #if dum2 >1e-14: print(dum2,r,k,l,s)
                ####for r in range(self.nmo):
                ####    for k in range(self.n_occupied,self.nmo):
                ####        for s in range(self.nmo):
                ####            for l in range(self.n_occupied,self.nmo):
                ####                if np.abs(G6[r*self.nmo+k][s*self.nmo+l]) >= 1e-10:
                ####                    print("{:20.12f}".format(G6[r*self.nmo+k][s*self.nmo+l]),"{:20.12f}".format(G6[k*self.nmo+r][l*self.nmo+s]),r,k,l,s)
                #### 
                ####G7 = np.zeros(((self.nmo * (self.nmo-1))//2, (self.nmo* (self.nmo-1))//2))
                ####index1 = 0
                ####for q in range(self.nmo):
                ####    for p in range(q+1,self.nmo):
                ####        index2 =0
                ####        for s in range(self.nmo):
                ####            for r in range(s+1,self.nmo):
                ####                G7[index1][index2] = G6[p * self.nmo + q][r * self.nmo +s]
                ####                index2 += 1
                ####        index1 += 1
                ####omega4, eig_vecs4 = np.linalg.eigh(G7)
                ####print("eigenvalues of the hessian",omega4)



                ####hessian_tilde4 = hessian_tilde4.reshape(self.nmo * self.n_occupied, self.nmo * self.n_occupied)
                #####hessian_tilde3 = hessian_tilde.transpose(0,2,1,3)
                #####hessian_tilde3=hessian_tilde3.reshape((self.nmo*self.nmo, self.nmo*self.nmo))
                ####omega, eig_vecs = np.linalg.eigh(G6)
                ####omega2, eig_vecs2 = np.linalg.eigh(hessian_tilde4)
                #####omega2, eig_vecs2 = np.linalg.eigh(hessian_tilde3)
                ####print("eigenvalues of the full orbital hessian",omega)
                ####print("eigenvalues of the reduced orbital hessian",omega2)
                #####print("dfds",omega2)
                #####print("gdgt", np.allclose(hessian_tilde3,G5, rtol=1e-14,atol=1e-14))
                







                ##########macroiteration = 0
                ##########self.U_total = np.eye(self.nmo)
                ##########while (macroiteration < 10):
                ##########    print("heyhey",eigenvecs) 
                ##########    self.internal_optimization2(avg_energy, eigenvecs) 
                ##########    print("heyhey",eigenvecs)
 
 
                ##########    ###print("LETS TEST THE TWO WAYS TO CALCULATE SECOND ORDER ENERGY AGAIN")
                ##########    ###A = np.zeros((rot_dim, rot_dim))
                ##########    ###G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ##########    ###print("initial e_core", self.E_core)
                ##########    ###self.build_intermediates(eigenvecs, A, G, True)
                ##########    ###A2 = np.zeros((rot_dim, rot_dim))
                ##########    ###G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                ##########    ###self.build_intermediates2(eigenvecs, A2, G2, True)
                ##########    ###print("test second order energy and ci updated integrals")
                ##########    ###exact_t_energy = self.microiteration_exact_energy(self.U_delta, A, G)
                ##########    ###print("exact_energy from second order expansion", exact_t_energy + avg_energy)
                ##########    ###active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
                ##########    ###active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                ##########    ###d_cmo = np.zeros((self.nmo, self.nmo))
                ##########    ###self.microiteration_ci_integrals_transform(self.U_delta, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                ##########    ###active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                ##########    ###active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                ##########    ###active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ##########    ###ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                ##########    ###sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                ##########    ###        self.Enuc + self.d_c + ci_dependent_energy)
                ##########    ###print("gfhgy",
                ##########    ###    "{:20.12f}".format(sum_energy),
                ##########    ###    "{:20.12f}".format(active_one_e_energy),
                ##########    ###    "{:20.12f}".format(active_two_e_energy),
                ##########    ###    "{:20.12f}".format(self.E_core),
                ##########    ###    "{:20.12f}".format(active_one_pe_energy),
                ##########    ###    "{:20.12f}".format(self.Enuc),
                ##########    ###)


                ##########    self.avg_energy = avg_energy
                ##########    if macroiteration == 0:
                ##########        convergence_threshold = 1e-3
                ##########    else:
                ##########        convergence_threshold = 1e-4
                ##########    print(self.avg_energy) 
                ##########    self.microiteration_optimization3(eigenvecs, c_get_roots, convergence_threshold) 
                ##########    print("full transformation test")
                ##########    #print(self.U_total)
                ##########    #JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                ##########    #KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                ##########    #self.full_transformation_macroiteration(self.U_total, JJ, KK)
                ##########    self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                ##########    print("tvhj", np.allclose(self.K3,self.K, rtol=1e-14,atol=1e-14))
                ##########    print("oins", np.allclose(self.J3,self.J, rtol=1e-14,atol=1e-14))
                ##########    print("tc5k", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
                ##########    print("p0ba", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
                ##########    active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
                ##########    self.fock_core = copy.deepcopy(self.H_spatial2) 
                ##########    self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                ##########    self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
                ##########    
                ##########    self.E_core = 0.0  
                ##########    self.E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
                ##########    self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 


                ##########    print(eigenvecs)
                ##########    active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                ##########    active_fock_core[:,:] = self.fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
                ##########    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                ##########    active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                ##########    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ##########    ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                ##########    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                ##########            self.Enuc + self.d_c + ci_dependent_energy)
                ##########    print("gfhgy end macroiteration",
                ##########        "{:20.12f}".format(sum_energy),
                ##########        "{:20.12f}".format(active_one_e_energy),
                ##########        "{:20.12f}".format(active_two_e_energy),
                ##########        "{:20.12f}".format(self.E_core),
                ##########        "{:20.12f}".format(active_one_pe_energy),
                ##########        "{:20.12f}".format(self.Enuc),
                ##########    )
                ##########    print("end one macroiteration")
                ##########    avg_energy = sum_energy
                ##########    self.occupied_K = copy.deepcopy(self.K[:,:,:self.n_occupied,:self.n_occupied])
                ##########    self.occupied_J = copy.deepcopy(self.J[:,:,:self.n_occupied,:self.n_occupied])
                ##########    self.occupied_h1 = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied])
                ##########    self.occupied_d_cmo = copy.deepcopy(self.d_cmo[:self.n_occupied,:self.n_occupied])
                ##########    self.occupied_fock_core = copy.deepcopy(self.fock_core[:self.n_occupied, :self.n_occupied]) 

                ##########    macroiteration += 1
               

                 


                #Rai= np.random.rand(self.n_act_orb,self.n_in_a)
                ##Rvi= np.random.rand(self.n_virtual,self.n_in_a)
                ##Rva= np.random.rand(self.n_virtual,self.n_act_orb)
                #Rvi= np.zeros((self.n_virtual,self.n_in_a))
                #Rva= np.zeros((self.n_virtual,self.n_act_orb))


                #self.build_unitary_matrix(Rai, Rvi, Rva)
                ####print("U_delta", self.U_delta)

                self.index_map_pq = np.zeros((self.nmo * (self.nmo +1)//2, 2), dtype = np.int32)
                index_count = 0 
                for r in range(self.nmo):
                    for s in range(r,self.nmo):
                        self.index_map_pq[index_count][0] = s 
                        self.index_map_pq[index_count][1] = r
                        index_count += 1
                self.index_map_kl = np.zeros((self.n_occupied * (self.n_occupied +1)//2, 2), dtype = np.int32)
                index_count = 0 
                for r in range(self.n_occupied):
                    for s in range(r,self.n_occupied):
                        self.index_map_kl[index_count][0] = s 
                        self.index_map_kl[index_count][1] = r
                        index_count += 1
                self.index_map_ab = np.zeros((self.n_virtual * (self.n_virtual +1)//2, 2), dtype = np.int32)
                index_count = 0 
                for r in range(self.n_virtual):
                    for s in range(r,self.n_virtual):
                        self.index_map_ab[index_count][0] = s 
                        self.index_map_ab[index_count][1] = r
                        index_count += 1


                #print(index_map_pq)
                #print(index_map_kl)

                #print("nmo, n_occupied", self.nmo, self.n_occupied, flush = True)
                #JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #start = timer()
                #c_full_transformation_macroiteration(self.U_delta, self.twoeint, JJ, KK, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
                #end   = timer()
                #print("build JK with blas took", end -start)
                #JJ2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #KK2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #start = timer()
                #self.full_transformation_macroiteration(self.U_delta, JJ2, KK2)
                #end   = timer()
                #print("build JK with numpy took", end -start)
                #print("test J build", flush = True)
                #for k in range(self.n_occupied):
                #    for l in range(self.n_occupied):
                #        for r in range(self.nmo):
                #            for s in range(self.nmo):
                #                a = JJ[k,l,r,s] - JJ2[k,l,r,s]
                #                if np.abs(a) > 1e-14: print("LARGE")
                #                print(JJ[k,l,r,s], JJ2[k,l,r,s], JJ[k,l,r,s] - JJ2[k,l,r,s], flush = True)
                #print("test K build", flush = True)
                #for k in range(self.n_occupied):
                #    for l in range(self.n_occupied):
                #        for r in range(self.nmo):
                #            for s in range(self.nmo):
                #                a = KK[k,l,r,s] - KK2[k,l,r,s]
                #                if np.abs(a) > 1e-14: print("LARGE")
                #                print(KK[k,l,r,s], KK2[k,l,r,s], KK[k,l,r,s] - KK2[k,l,r,s], flush = True)

                #JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #h1 = np.zeros((self.nmo, self.nmo))
                #d_cmo1 = np.zeros((self.nmo, self.nmo))
                #start = timer()
                #self.K =np.ascontiguousarray(self.K)
                #c_full_transformation_internal_optimization(self.U_delta, self.J, self.K, self.H_spatial2, self.d_cmo, JJ, KK, h1, d_cmo1, 
                #        self.index_map_ab, self.index_map_kl, self.nmo, self.n_occupied) 
                #end   = timer()
                #print("build JK with blas took", end -start)

                #h3 = np.zeros((self.nmo, self.nmo))
                #d_cmo3 = np.zeros((self.nmo, self.nmo))
                #JJ2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #KK2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #start = timer()
                #self.full_transformation_internal_optimization(self.U_delta, h3, d_cmo3, JJ2, KK2)
                #end   = timer()
                #print("build JK with numpy took", end -start)
                #print("test J build", flush = True)
                #for k in range(self.n_occupied):
                #    for l in range(self.n_occupied):
                #        for r in range(self.nmo):
                #            for s in range(self.nmo):
                #                a = JJ[k,l,r,s] - JJ2[k,l,r,s]
                #                if np.abs(a) > 1e-14: print("LARGE")
                #                print(JJ[k,l,r,s], JJ2[k,l,r,s], JJ[k,l,r,s] - JJ2[k,l,r,s], flush = True)
                #print("test K build", flush = True)
                #for k in range(self.n_occupied):
                #    for l in range(self.n_occupied):
                #        for r in range(self.nmo):
                #            for s in range(self.nmo):
                #                a = KK[k,l,r,s] - KK2[k,l,r,s]
                #                if np.abs(a) > 1e-14: print("LARGE")
                #                print(KK[k,l,r,s], KK2[k,l,r,s], KK[k,l,r,s] - KK2[k,l,r,s], flush = True)
                #print("test h build", flush = True)
                #for r in range(self.nmo):
                #    for s in range(self.nmo):
                #        a = h1[r,s] - h3[r,s]
                #        if np.abs(a) > 1e-14: print("LARGE")
                #        print(h1[r,s], h3[r,s], h1[r,s] - h3[r,s], flush = True)
                #print("test d_cmo build", flush = True)
                #for r in range(self.nmo):
                #    for s in range(self.nmo):
                #        a = d_cmo1[r,s] - d_cmo3[r,s]
                #        if np.abs(a) > 1e-14: print("LARGE")
                #        print(d_cmo1[r,s], d_cmo3[r,s], d_cmo1[r,s] - d_cmo3[r,s], flush = True)




                if self.n_in_a == 0 and self.n_act_orb == self.nmo:
                    pass
                else:
                    #self.ah_orbital_optimization(eigenvecs, c_get_roots)
                    self.bfgs_orbital_optimization(eigenvecs, c_get_roots)
                    ########start = timer()
                    ########macroiteration = 0
                    ########self.U_total = np.eye(self.nmo)
                    ########old_avg_energy = avg_energy
                    ########new_avg_energy = 0
                    ########convergence = 0
                    ########while (macroiteration < 20000):
                    ########    if np.abs(new_avg_energy - old_avg_energy) < 1e-9:
                    ########        convergence = 1
                    ########    if macroiteration >0:

                    ########        #print("U total")
                    ########        #self.printA(self.U_total)
                    ########        print("old energy",old_avg_energy, "new energy", new_avg_energy, flush = True)
                    ########        occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                    ########        occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,
                    ########                self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(self.J[self.n_in_a: self.n_occupied,
                    ########                    self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied])        
                    ########        active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
                    ########        self.H_diag3 = np.zeros(H_dim)
                    ########        fock_core = copy.deepcopy(self.H_spatial2) 
                    ########        fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                    ########        fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 

                    ########        active_fock_core = copy.deepcopy(fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
                    ########        occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
                    ########        occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
                    ########        occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
                    ########        occupied_d_cmo = copy.deepcopy(self.d_cmo[: self.n_occupied,: self.n_occupied]) 
                    ########        gkl2 = copy.deepcopy(active_fock_core) 
                    ########        gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
                    ########        
                    ########        occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                    ########        c_H_diag_cas_spin(
                    ########                occupied_fock_core, 
                    ########                occupied_J, 
                    ########                self.H_diag3, 
                    ########                self.N_p, 
                    ########                self.num_alpha, 
                    ########                self.nmo, 
                    ########                self.n_act_a, 
                    ########                self.n_act_orb, 
                    ########                self.n_in_a, 
                    ########                self.E_core, 
                    ########                self.omega, 
                    ########                self.Enuc, 
                    ########                self.d_c, 
                    ########                self.Y,
                    ########                self.target_spin)
                    ########        d_diag = 2.0 * np.einsum("ii->", self.d_cmo[:self.n_in_a,:self.n_in_a])
                    ########        self.constdouble[3] = self.d_exp - d_diag
                    ########        #self.constdouble[4] = 1e-5 
                    ########        self.constdouble[4] = self.davidson_threshold
                    ########        self.constdouble[5] = self.E_core
                    ########        self.constint[8] = self.davidson_maxiter
                    ########        eigenvals = np.zeros((self.davidson_roots))
                    ########        #eigenvecs = np.zeros((self.davidson_roots, H_dim))
                    ########        #eigenvecs[:,:] = 0.0
                    ########        #print("heyhey5", eigenvecs)
                    ########        c_get_roots(
                    ########            gkl2,
                    ########            occupied_J,
                    ########            occupied_d_cmo,
                    ########            self.H_diag3,
                    ########            self.S_diag,
                    ########            self.S_diag_projection,
                    ########            eigenvals,
                    ########            eigenvecs,
                    ########            self.table,
                    ########            self.table_creation,
                    ########            self.table_annihilation,
                    ########            self.b_array,
                    ########            self.constint,
                    ########            self.constdouble,
                    ########            self.index_Hdiag,
                    ########            True,
                    ########            self.target_spin,
                    ########        )
                    ########        avg_energy = 0.0 
                    ########        for i in range(self.davidson_roots):
                    ########            avg_energy += self.weight[i] * eigenvals[i]
                    ########        print("avg energy", macroiteration, avg_energy)
                    ########        print("average energy at the start of macroiteration", avg_energy) 
                    ########        self.build_state_avarage_rdms(eigenvecs)

                    ########    if macroiteration > 0 and convergence == 1:

                    ########        print(
                    ########            "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
                    ########        )
                    ########        Y = np.zeros(
                    ########            self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                    ########            dtype=np.int32,
                    ########        )
                    ########        c_graph(self.n_act_a, self.n_act_orb, Y)
                    ########        np1 = self.N_p + 1
                    ########        singlet_count = 0
                    ########        triplet_count = 0
                    ########        for i in range(eigenvecs.shape[0]):
                    ########            total_spin = self.check_total_spin(eigenvecs[i : (i + 1), :])
                    ########            print(
                    ########                "STATE",
                    ########                i,
                    ########                "ENERGY =",
                    ########                eigenvals[i],
                    ########                "<S^2>=",
                    ########                total_spin,
                    ########                "WEIGHT =",
                    ########                self.weight[i],
                    ########                end="",
                    ########            )
                    ########            if np.abs(total_spin) < 1e-5:
                    ########                singlet_count += 1
                    ########                print("\tSINGLET", singlet_count)
                    ########            elif np.abs(total_spin - 2.0) < 1e-5:
                    ########                triplet_count += 1
                    ########                print("\tTRIPLET", triplet_count)
                    ########            elif np.abs(total_spin - 6.0) < 1e-5:
                    ########                print("\tQUINTET")

                    ########            # print("state",i, "energy =",theta[i])
                    ########            print(
                    ########                "        amplitude",
                    ########                "      position",
                    ########                "         most important determinants",
                    ########                "             number of photon",
                    ########            )
                    ########            index = np.argsort(np.abs(eigenvecs[i, :]))
                    ########            # print(index)
                    ########            Idet0 = (
                    ########                index[eigenvecs.shape[1] - 1] % self.num_det
                    ########            )  # determinant index of most significant contribution
                    ########            photon_p0 = (
                    ########                index[eigenvecs.shape[1] - 1] - Idet0
                    ########            ) // self.num_det  # photon number block of determinant
                    ########            Ib0 = Idet0 % self.num_alpha
                    ########            Ia0 = Idet0 // self.num_alpha
                    ########            a0 = c_index_to_string(Ia0, self.n_act_a, self.n_act_orb, Y)
                    ########            b0 = c_index_to_string(Ib0, self.n_act_a, self.n_act_orb, Y)

                    ########            alphalist = Determinant.obtBits2ObtIndexList(a0)
                    ########            betalist = Determinant.obtBits2ObtIndexList(b0)
                    ########            for j in range(min(H_dim, 10)):
                    ########                Idet = index[eigenvecs.shape[1] - j - 1] % self.num_det
                    ########                photon_p = (
                    ########                    index[eigenvecs.shape[1] - j - 1] - Idet
                    ########                ) // self.num_det
                    ########                Ib = Idet % self.num_alpha
                    ########                Ia = Idet // self.num_alpha
                    ########                a = c_index_to_string(Ia, self.n_act_a, self.n_act_orb, Y)
                    ########                b = c_index_to_string(Ib, self.n_act_a, self.n_act_orb, Y)

                    ########                alphalist = Determinant.obtBits2ObtIndexList(a)
                    ########                betalist = Determinant.obtBits2ObtIndexList(b)

                    ########                inactive_list = list(x for x in range(self.n_in_a))
                    ########                alphalist2 = [x + self.n_in_a for x in alphalist]
                    ########                # alphalist2[0:0] = inactive_list
                    ########                betalist2 = [x + self.n_in_a for x in betalist]
                    ########                # betalist2[0:0] = inactive_list

                    ########                print(
                    ########                    "%20.12lf"
                    ########                    % (eigenvecs[i][index[eigenvecs.shape[1] - j - 1]]),
                    ########                    "%9.3d" % (index[eigenvecs.shape[1] - j - 1]),
                    ########                    "alpha",
                    ########                    alphalist2,
                    ########                    "   beta",
                    ########                    betalist2,
                    ########                    "%4.1d" % (photon_p),
                    ########                    "photon",
                    ########                )


                    ########        print("OPTIMIZATION CONVERGED", flush = True)
                    ########        print("avg energy final", macroiteration, avg_energy)
                    ########        if self.save_orbital == True:
                    ########            new_C = np.einsum("pq,qr->pr", self.C, self.U_total)
                    ########            #print(new_C)
                    ########            np.savetxt("orbital.out", new_C)
                    ########        
                    ########        break
                    ########    old_avg_energy = avg_energy
                    ########    self.avg_energy = avg_energy
                    ########    #if macroiteration >0 and self.n_in_a > 0:
                    ########    #    start1 = timer()
                    ########    #    self.internal_optimization2(avg_energy, eigenvecs) 
                    ########    #    end1 = timer()
                    ########    #    print("internal optimization took", end1 - start1) 
 
                    ########    ###print("LETS TEST THE TWO WAYS TO CALCULATE SECOND ORDER ENERGY AGAIN")
                    ########    ###A = np.zeros((rot_dim, rot_dim))
                    ########    ###G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                    ########    ###print("initial e_core", self.E_core)
                    ########    ###self.build_intermediates(eigenvecs, A, G, True)
                    ########    ###A2 = np.zeros((rot_dim, rot_dim))
                    ########    ###G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                    ########    ###self.build_intermediates2(eigenvecs, A2, G2, True)
                    ########    ###print("test second order energy and ci updated integrals")
                    ########    ###exact_t_energy = self.microiteration_exact_energy(self.U_delta, A, G)
                    ########    ###print("exact_energy from second order expansion", exact_t_energy + avg_energy)
                    ########    ###active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
                    ########    ###active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                    ########    ###d_cmo = np.zeros((self.nmo, self.nmo))
                    ########    ###self.microiteration_ci_integrals_transform(self.U_delta, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                    ########    ###active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                    ########    ###active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                    ########    ###active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                    ########    ###ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                    ########    ###sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                    ########    ###        self.Enuc + self.d_c + ci_dependent_energy)
                    ########    ###print("gfhgy",
                    ########    ###    "{:20.12f}".format(sum_energy),
                    ########    ###    "{:20.12f}".format(active_one_e_energy),
                    ########    ###    "{:20.12f}".format(active_two_e_energy),
                    ########    ###    "{:20.12f}".format(self.E_core),
                    ########    ###    "{:20.12f}".format(active_one_pe_energy),
                    ########    ###    "{:20.12f}".format(self.Enuc),
                    ########    ###)


                    ########    np.set_printoptions(precision=14)
                    ########    if macroiteration == 0:
                    ########        convergence_threshold = 1e-3
                    ########    else:
                    ########        convergence_threshold = 1e-4
                    ########    print("avg energy", macroiteration, self.avg_energy)
                    ########    U0 = np.eye(self.nmo)
                    ########    #print("heyhey3",eigenvecs)
                    ########    start1 = timer()
                    ########    #self.microiteration_optimization4(U0,eigenvecs, c_get_roots, convergence_threshold)
                    ########    self.ah_orbital_optimization(U0,eigenvecs, c_get_roots, convergence_threshold)
                    ########    end1   = timer()
                    ########    print("microiteration took", end1 - start1)
                    ########    #print("heyhey",eigenvecs)
                    ########    #print("u2",self.U2)
                    ########    print("full transformation test")
                    ########    
                    ########    R = 0.5 * (self.U2 - self.U2.T)
                    ########    Rai = np.zeros((self.n_act_orb, self.n_in_a))
                    ########    Rai[:,:] = R[self.n_in_a:self.n_occupied,:self.n_in_a]
                    ########    #print ("norm of internal step", np.linalg.norm(Rai), flush = True)
                    ########    #if (np.linalg.norm(Rai) > 1e-4):
                    ########    #    #print("u2 before", self.U2,flush = True)
                    ########    #    print("RESTART MICROITERATION TO CORRECT INTERNAL ROTATION")
                    ########    #    Rvi = np.zeros((self.n_virtual,self.n_in_a))
                    ########    #    Rva = np.zeros((self.n_virtual,self.n_act_orb))
                    ########    #    self.build_unitary_matrix(Rai, Rvi, Rva)
                    ########    #    self.K = np.ascontiguousarray(self.K)
                    ########    #    start1 = timer()
                    ########    #    c_full_transformation_internal_optimization(self.U_delta, self.J, self.K, self.H_spatial2, self.d_cmo, self.J, self.K, self.H_spatial2, self.d_cmo, 
                    ########    #                               self.index_map_ab, self.index_map_kl, self.nmo, self.n_occupied) 
                    ########    #    end1   = timer()
                    ########    #    print("full internal transformation took", end1 - start1)
                    ########    #    #self.full_transformation_internal_optimization(self.U_delta, self.H_spatial2, self.d_cmo, self.J, self.K)
                    ########    #    self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U_delta)
                    ########    #    Rai[:,:] = 0.0
                    ########    #    Rvi[:,:] = R[self.n_occupied:,:self.n_in_a]
                    ########    #    Rva[:,:] = R[self.n_occupied:,self.n_in_a:self.n_occupied]
                    ########    #    self.build_unitary_matrix(Rai, Rvi, Rva)
                    ########    #    start1 = timer()
                    ########    #    self.microiteration_optimization4(self.U_delta,eigenvecs, c_get_roots, convergence_threshold)
                    ########    #    end1   = timer()
                    ########    #    print("microiteration took", end1 - start1)
                    ########    #    Rkl = 0.5 * (self.U2[:self.n_occupied,:self.n_occupied] - self.U2.T[:self.n_occupied,:self.n_occupied])
                    ########    #    Rai = np.zeros((self.n_act_orb, self.n_in_a))
                    ########    #    Rai[:,:] = Rkl[self.n_in_a:self.n_occupied,:self.n_in_a]
                    ########    #print ("norm of internal step after", np.linalg.norm(Rai), flush = True)
                    ########    ##print("u2 after", self.U2,flush = True)
                    ########    
                    ########    temp8 = np.zeros((self.nmo, self.nmo))
                    ########    temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                    ########    self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                    ########    temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                    ########    self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)

                    ########    self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
   
                    ########    #print(self.U_total)
                    ########    new_C = np.einsum("pq,qr->pr", self.C, self.U_total)
                    ########    print(new_C)
                    ########    np.savetxt("orbital.out", new_C)
                    ########        

                    ########    ##JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                    ########    ##KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                    ########    ##self.full_transformation_macroiteration(self.U_total, JJ, KK)
                    ########    start1 = timer()
                    ########    self.K = np.ascontiguousarray(self.K)
                    ########    c_full_transformation_macroiteration(self.U_total, self.twoeint, self.J, self.K, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
                    ########    #self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                    ########    end1 = timer()
                    ########    print("full JK transformation took", end1-start1)
                    ########    ##print("tvhj", np.allclose(self.K3,self.K, rtol=1e-14,atol=1e-14))
                    ########    ##print("oins", np.allclose(self.J3,self.J, rtol=1e-14,atol=1e-14))
                    ########    ##print("tc5k", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
                    ########    ##print("p0ba", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
                    ########    active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
                    ########    self.fock_core = copy.deepcopy(self.H_spatial2) 
                    ########    self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                    ########    self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
                    ########    
                    ########    self.E_core = 0.0  
                    ########    self.E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
                    ########    self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 


                    ########    #print(eigenvecs)
                    ########    active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                    ########    active_fock_core[:,:] = self.fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
                    ########    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                    ########    active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                    ########    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                    ########    ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                    ########    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                    ########            self.Enuc + self.d_c + ci_dependent_energy)
                    ########    print("gfhgy end macroiteration",
                    ########        "{:20.12f}".format(sum_energy),
                    ########        "{:20.12f}".format(active_one_e_energy),
                    ########        "{:20.12f}".format(active_two_e_energy),
                    ########        "{:20.12f}".format(self.E_core),
                    ########        "{:20.12f}".format(active_one_pe_energy),
                    ########        "{:20.12f}".format(self.Enuc),
                    ########    )
                    ########    print("end one macroiteration")
                    ########    avg_energy = sum_energy
                    ########    self.occupied_K = copy.deepcopy(self.K[:,:,:self.n_occupied,:self.n_occupied])
                    ########    self.occupied_J = copy.deepcopy(self.J[:,:,:self.n_occupied,:self.n_occupied])
                    ########    self.occupied_h1 = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied])
                    ########    self.occupied_d_cmo = copy.deepcopy(self.d_cmo[:self.n_occupied,:self.n_occupied])
                    ########    self.occupied_fock_core = copy.deepcopy(self.fock_core[:self.n_occupied, :self.n_occupied]) 
                    ########    new_avg_energy = avg_energy

                    ########    macroiteration += 1
                    ########end = timer()
                    ########print("optimization took", end - start)


































                #back up wmk algorithm
                #if self.n_in_a == 0 and self.n_act_orb == self.nmo:
                #    pass
                #else:
                #    start = timer()
                #    macroiteration = 0
                #    self.U_total = np.eye(self.nmo)
                #    old_avg_energy = avg_energy
                #    new_avg_energy = 0
                #    convergence = 0
                #    while (macroiteration < 20000):
                #        if np.abs(new_avg_energy - old_avg_energy) < 1e-9:
                #            convergence = 1
                #        if macroiteration >0:

                #            #print("U total")
                #            #self.printA(self.U_total)
                #            print("old energy",old_avg_energy, "new energy", new_avg_energy, flush = True)
                #            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,
                #                    self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(self.J[self.n_in_a: self.n_occupied,
                #                        self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied])        
                #            active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
                #            self.H_diag3 = np.zeros(H_dim)
                #            fock_core = copy.deepcopy(self.H_spatial2) 
                #            fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                #            fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 

                #            active_fock_core = copy.deepcopy(fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
                #            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
                #            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
                #            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
                #            occupied_d_cmo = copy.deepcopy(self.d_cmo[: self.n_occupied,: self.n_occupied]) 
                #            gkl2 = copy.deepcopy(active_fock_core) 
                #            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
                #            
                #            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                #            c_H_diag_cas_spin(
                #                    occupied_fock_core, 
                #                    occupied_J, 
                #                    self.H_diag3, 
                #                    self.N_p, 
                #                    self.num_alpha, 
                #                    self.nmo, 
                #                    self.n_act_a, 
                #                    self.n_act_orb, 
                #                    self.n_in_a, 
                #                    self.E_core, 
                #                    self.omega, 
                #                    self.Enuc, 
                #                    self.d_c, 
                #                    self.Y,
                #                    self.target_spin)
                #            d_diag = 2.0 * np.einsum("ii->", self.d_cmo[:self.n_in_a,:self.n_in_a])
                #            self.constdouble[3] = self.d_exp - d_diag
                #            #self.constdouble[4] = 1e-5 
                #            self.constdouble[4] = self.davidson_threshold
                #            self.constdouble[5] = self.E_core
                #            self.constint[8] = self.davidson_maxiter
                #            eigenvals = np.zeros((self.davidson_roots))
                #            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
                #            #eigenvecs[:,:] = 0.0
                #            #print("heyhey5", eigenvecs)
                #            c_get_roots(
                #                gkl2,
                #                occupied_J,
                #                occupied_d_cmo,
                #                self.H_diag3,
                #                self.S_diag,
                #                self.S_diag_projection,
                #                eigenvals,
                #                eigenvecs,
                #                self.table,
                #                self.table_creation,
                #                self.table_annihilation,
                #                self.b_array,
                #                self.constint,
                #                self.constdouble,
                #                self.index_Hdiag,
                #                True,
                #                self.target_spin,
                #            )
                #            avg_energy = 0.0 
                #            for i in range(self.davidson_roots):
                #                avg_energy += self.weight[i] * eigenvals[i]
                #            print("avg energy", macroiteration, avg_energy)
                #            print("average energy at the start of macroiteration", avg_energy) 
                #            self.build_state_avarage_rdms(eigenvecs)

                #        if macroiteration > 0 and convergence == 1:

                #            print(
                #                "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
                #            )
                #            Y = np.zeros(
                #                self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                #                dtype=np.int32,
                #            )
                #            c_graph(self.n_act_a, self.n_act_orb, Y)
                #            np1 = self.N_p + 1
                #            singlet_count = 0
                #            triplet_count = 0
                #            for i in range(eigenvecs.shape[0]):
                #                total_spin = self.check_total_spin(eigenvecs[i : (i + 1), :])
                #                print(
                #                    "STATE",
                #                    i,
                #                    "ENERGY =",
                #                    eigenvals[i],
                #                    "<S^2>=",
                #                    total_spin,
                #                    "WEIGHT =",
                #                    self.weight[i],
                #                    end="",
                #                )
                #                if np.abs(total_spin) < 1e-5:
                #                    singlet_count += 1
                #                    print("\tSINGLET", singlet_count)
                #                elif np.abs(total_spin - 2.0) < 1e-5:
                #                    triplet_count += 1
                #                    print("\tTRIPLET", triplet_count)
                #                elif np.abs(total_spin - 6.0) < 1e-5:
                #                    print("\tQUINTET")

                #                # print("state",i, "energy =",theta[i])
                #                print(
                #                    "        amplitude",
                #                    "      position",
                #                    "         most important determinants",
                #                    "             number of photon",
                #                )
                #                index = np.argsort(np.abs(eigenvecs[i, :]))
                #                # print(index)
                #                Idet0 = (
                #                    index[eigenvecs.shape[1] - 1] % self.num_det
                #                )  # determinant index of most significant contribution
                #                photon_p0 = (
                #                    index[eigenvecs.shape[1] - 1] - Idet0
                #                ) // self.num_det  # photon number block of determinant
                #                Ib0 = Idet0 % self.num_alpha
                #                Ia0 = Idet0 // self.num_alpha
                #                a0 = c_index_to_string(Ia0, self.n_act_a, self.n_act_orb, Y)
                #                b0 = c_index_to_string(Ib0, self.n_act_a, self.n_act_orb, Y)

                #                alphalist = Determinant.obtBits2ObtIndexList(a0)
                #                betalist = Determinant.obtBits2ObtIndexList(b0)
                #                for j in range(min(H_dim, 10)):
                #                    Idet = index[eigenvecs.shape[1] - j - 1] % self.num_det
                #                    photon_p = (
                #                        index[eigenvecs.shape[1] - j - 1] - Idet
                #                    ) // self.num_det
                #                    Ib = Idet % self.num_alpha
                #                    Ia = Idet // self.num_alpha
                #                    a = c_index_to_string(Ia, self.n_act_a, self.n_act_orb, Y)
                #                    b = c_index_to_string(Ib, self.n_act_a, self.n_act_orb, Y)

                #                    alphalist = Determinant.obtBits2ObtIndexList(a)
                #                    betalist = Determinant.obtBits2ObtIndexList(b)

                #                    inactive_list = list(x for x in range(self.n_in_a))
                #                    alphalist2 = [x + self.n_in_a for x in alphalist]
                #                    # alphalist2[0:0] = inactive_list
                #                    betalist2 = [x + self.n_in_a for x in betalist]
                #                    # betalist2[0:0] = inactive_list

                #                    print(
                #                        "%20.12lf"
                #                        % (eigenvecs[i][index[eigenvecs.shape[1] - j - 1]]),
                #                        "%9.3d" % (index[eigenvecs.shape[1] - j - 1]),
                #                        "alpha",
                #                        alphalist2,
                #                        "   beta",
                #                        betalist2,
                #                        "%4.1d" % (photon_p),
                #                        "photon",
                #                    )


                #            print("OPTIMIZATION CONVERGED", flush = True)
                #            print("avg energy final", macroiteration, avg_energy)
                #            if self.save_orbital == True:
                #                new_C = np.einsum("pq,qr->pr", self.C, self.U_total)
                #                #print(new_C)
                #                np.savetxt("orbital.out", new_C)
                #            
                #            break
                #        old_avg_energy = avg_energy
                #        self.avg_energy = avg_energy
                #        if macroiteration >0 and self.n_in_a > 0:
                #            start1 = timer()
                #            self.internal_optimization2(avg_energy, eigenvecs) 
                #            end1 = timer()
                #            print("internal optimization took", end1 - start1) 
 
                #        ###print("LETS TEST THE TWO WAYS TO CALCULATE SECOND ORDER ENERGY AGAIN")
                #        ###A = np.zeros((rot_dim, rot_dim))
                #        ###G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                #        ###print("initial e_core", self.E_core)
                #        ###self.build_intermediates(eigenvecs, A, G, True)
                #        ###A2 = np.zeros((rot_dim, rot_dim))
                #        ###G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                #        ###self.build_intermediates2(eigenvecs, A2, G2, True)
                #        ###print("test second order energy and ci updated integrals")
                #        ###exact_t_energy = self.microiteration_exact_energy(self.U_delta, A, G)
                #        ###print("exact_energy from second order expansion", exact_t_energy + avg_energy)
                #        ###active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
                #        ###active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                #        ###d_cmo = np.zeros((self.nmo, self.nmo))
                #        ###self.microiteration_ci_integrals_transform(self.U_delta, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                #        ###active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                #        ###active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                #        ###active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                #        ###ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                #        ###sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                #        ###        self.Enuc + self.d_c + ci_dependent_energy)
                #        ###print("gfhgy",
                #        ###    "{:20.12f}".format(sum_energy),
                #        ###    "{:20.12f}".format(active_one_e_energy),
                #        ###    "{:20.12f}".format(active_two_e_energy),
                #        ###    "{:20.12f}".format(self.E_core),
                #        ###    "{:20.12f}".format(active_one_pe_energy),
                #        ###    "{:20.12f}".format(self.Enuc),
                #        ###)


                #        np.set_printoptions(precision=14)
                #        if macroiteration == 0:
                #            convergence_threshold = 1e-3
                #        else:
                #            convergence_threshold = 1e-4
                #        print("avg energy", macroiteration, self.avg_energy)
                #        U0 = np.eye(self.nmo)
                #        #print("heyhey3",eigenvecs)
                #        start1 = timer()
                #        self.microiteration_optimization4(U0,eigenvecs, c_get_roots, convergence_threshold)
                #        end1   = timer()
                #        print("microiteration took", end1 - start1)
                #        #print("heyhey",eigenvecs)
                #        #print("u2",self.U2)
                #        print("full transformation test")
                #        
                #        R = 0.5 * (self.U2 - self.U2.T)
                #        Rai = np.zeros((self.n_act_orb, self.n_in_a))
                #        Rai[:,:] = R[self.n_in_a:self.n_occupied,:self.n_in_a]
                #        print ("norm of internal step", np.linalg.norm(Rai), flush = True)
                #        if (np.linalg.norm(Rai) > 1e-4):
                #            #print("u2 before", self.U2,flush = True)
                #            print("RESTART MICROITERATION TO CORRECT INTERNAL ROTATION")
                #            Rvi = np.zeros((self.n_virtual,self.n_in_a))
                #            Rva = np.zeros((self.n_virtual,self.n_act_orb))
                #            self.build_unitary_matrix(Rai, Rvi, Rva)
                #            self.K = np.ascontiguousarray(self.K)
                #            start1 = timer()
                #            c_full_transformation_internal_optimization(self.U_delta, self.J, self.K, self.H_spatial2, self.d_cmo, self.J, self.K, self.H_spatial2, self.d_cmo, 
                #                                       self.index_map_ab, self.index_map_kl, self.nmo, self.n_occupied) 
                #            end1   = timer()
                #            print("full internal transformation took", end1 - start1)
                #            #self.full_transformation_internal_optimization(self.U_delta, self.H_spatial2, self.d_cmo, self.J, self.K)
                #            self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U_delta)
                #            Rai[:,:] = 0.0
                #            Rvi[:,:] = R[self.n_occupied:,:self.n_in_a]
                #            Rva[:,:] = R[self.n_occupied:,self.n_in_a:self.n_occupied]
                #            self.build_unitary_matrix(Rai, Rvi, Rva)
                #            start1 = timer()
                #            self.microiteration_optimization4(self.U_delta,eigenvecs, c_get_roots, convergence_threshold)
                #            end1   = timer()
                #            print("microiteration took", end1 - start1)
                #            Rkl = 0.5 * (self.U2[:self.n_occupied,:self.n_occupied] - self.U2.T[:self.n_occupied,:self.n_occupied])
                #            Rai = np.zeros((self.n_act_orb, self.n_in_a))
                #            Rai[:,:] = Rkl[self.n_in_a:self.n_occupied,:self.n_in_a]
                #        print ("norm of internal step after", np.linalg.norm(Rai), flush = True)
                #        #print("u2 after", self.U2,flush = True)
                #        
                #        temp8 = np.zeros((self.nmo, self.nmo))
                #        temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #        self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #        temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                #        self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)

                #        self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
   
                #        #print(self.U_total)
                #            

                #        ##JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #        ##KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                #        ##self.full_transformation_macroiteration(self.U_total, JJ, KK)
                #        start1 = timer()
                #        self.K = np.ascontiguousarray(self.K)
                #        c_full_transformation_macroiteration(self.U_total, self.twoeint, self.J, self.K, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
                #        #self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                #        end1 = timer()
                #        print("full JK transformation took", end1-start1)
                #        ##print("tvhj", np.allclose(self.K3,self.K, rtol=1e-14,atol=1e-14))
                #        ##print("oins", np.allclose(self.J3,self.J, rtol=1e-14,atol=1e-14))
                #        ##print("tc5k", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
                #        ##print("p0ba", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
                #        active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
                #        self.fock_core = copy.deepcopy(self.H_spatial2) 
                #        self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
                #        self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
                #        
                #        self.E_core = 0.0  
                #        self.E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
                #        self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 


                #        #print(eigenvecs)
                #        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                #        active_fock_core[:,:] = self.fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
                #        active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                #        active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                #        active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                #        ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                #        sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                #                self.Enuc + self.d_c + ci_dependent_energy)
                #        print("gfhgy end macroiteration",
                #            "{:20.12f}".format(sum_energy),
                #            "{:20.12f}".format(active_one_e_energy),
                #            "{:20.12f}".format(active_two_e_energy),
                #            "{:20.12f}".format(self.E_core),
                #            "{:20.12f}".format(active_one_pe_energy),
                #            "{:20.12f}".format(self.Enuc),
                #        )
                #        print("end one macroiteration")
                #        avg_energy = sum_energy
                #        self.occupied_K = copy.deepcopy(self.K[:,:,:self.n_occupied,:self.n_occupied])
                #        self.occupied_J = copy.deepcopy(self.J[:,:,:self.n_occupied,:self.n_occupied])
                #        self.occupied_h1 = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied])
                #        self.occupied_d_cmo = copy.deepcopy(self.d_cmo[:self.n_occupied,:self.n_occupied])
                #        self.occupied_fock_core = copy.deepcopy(self.fock_core[:self.n_occupied, :self.n_occupied]) 
                #        new_avg_energy = avg_energy

                #        macroiteration += 1
                #    end = timer()
                #    print("optimization took", end - start)







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
            #print(f" Completed Davidson iterations in {t_dav_end - t_H_build} seconds", flush = True)

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
        if "use_orbital_guess" in cavity_dictionary:
            self.use_orbital_guess = cavity_dictionary["use_orbital_guess"]
        else:
            self.use_orbital_guess = False
        if "save_orbital" in cavity_dictionary:
            self.save_orbital = cavity_dictionary["save_orbital"]
        else:
            self.save_orbital = False
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

        if self.ci_level == "cas" or self.ci_level == "fci":
            if "spin_adaptation" in cavity_dictionary:
                self.spin_adaptation = cavity_dictionary["spin_adaptation"]
                if self.spin_adaptation == "singlet":
                    self.target_spin = 0.0
                if self.spin_adaptation == "triplet":
                    self.target_spin = 1.0
                if self.spin_adaptation == "quintet":
                    self.target_spin = 2.0
            else:
                self.spin_adaptation = "no"
                self.target_spin = -1.0
                #print(self.target_spin)
           
        if "casscf_weight" in cavity_dictionary:
            self.weight = cavity_dictionary["casscf_weight"]
        else:
            self.weight = np.full(self.davidson_roots,1.0)  
        if np.sum(self.weight) <= 1e-14:
            print("the program requires posive weight for at least one root")
            exit()
        if np.size(self.weight) > self.davidson_roots:
            print("the dimension of the weight array exceeds the number of roots")
            exit()
        elif np.size(self.weight) < self.davidson_roots:
            extra_dim = self.davidson_roots - np.size(self.weight)
            weight1 = np.full(extra_dim, 0.0)
            print(weight1)
            self.weight = np.concatenate((self.weight, weight1))
        self.weight = self.weight/np.sum(self.weight)
        print("weight", self.weight)

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
            self.fock_alpha = cqed_rhf_dict["CQED-RHF FOCK MATRIX"]
            self.D = cqed_rhf_dict["CQED-RHF DENSITY MATRIX"]
            self.cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
            self.cqed_one_energy = cqed_rhf_dict["CQED-RHF ONE-ENERGY"]

        # collect rhf wfn object as dictionary
        wfn_dict = psi4.core.Wavefunction.to_file(wfn)
        ##print(self.C)
        #U = ortho_group.rvs(wfn.nmo())
        #new_C = np.einsum("pq,qr->pr", self.C, U)

        #self.C[:,:] = new_C[:,:]       
        ##update d_cmo
        #self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)

        #print("Unitary matrix") 
        #print(U)
        # update wfn_dict with orbitals from CQED-RHF
        wfn_dict["matrix"]["Ca"] = self.C
        wfn_dict["matrix"]["Cb"] = self.C
        #mints = psi4.core.MintsHelper(wfn.basisset())
        #overlap_matrix = mints.ao_overlap()
        #overlap_matrix = np.asarray(overlap_matrix) 
        #print(np.shape(overlap_matrix))
        #temp = np.einsum("pq, qr, rs->ps", self.C.T, overlap_matrix, self.C)
        #print("thrbf", np.allclose(temp,np.eye(wfn.nmo()), rtol=1e-14,atol=1e-14))

        
        #print(self.fock_alpha)
        if self.use_orbital_guess == True:
            if self.omega != 0.0:
                wfn_dict["matrix"]["Fa"] = self.fock_alpha
                wfn_dict["matrix"]["Fb"] = self.fock_alpha
            print("load file")
            wfn_dict["matrix"]["Ca"] = np.loadtxt("orbital.out")
            wfn_dict["matrix"]["Cb"] = np.loadtxt("orbital.out")
            # update wfn object
            wfn = psi4.core.Wavefunction.from_file(wfn_dict)
            wfn.Ca().copy(ortho_orbs(wfn, wfn))
            wfn.Cb().copy(wfn.Ca())
            self.C = np.asarray(wfn.Ca())
            #update d_cmo
            self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)
            #mints = psi4.core.MintsHelper(wfn.basisset())
            #overlap_matrix = mints.ao_overlap()
            #overlap_matrix = np.asarray(overlap_matrix) 
            #print(np.shape(overlap_matrix))
            #temp = np.einsum("pq, qr, rs->ps", self.C.T, overlap_matrix, self.C)
            #print("thrbf", np.allclose(temp,np.eye(wfn.nmo()), rtol=1e-14,atol=1e-14))
            #print(temp)
            
      



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
            self.twoeint = self.twoeint1 + np.einsum("ij,kl->ijkl", self.d_cmo, self.d_cmo)
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


    def build_JK(self):
        #naive build of J_klrs and K_klrs from the full integrals
        #self.J = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.K = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.J = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:self.n_occupied,:self.n_occupied,:,:]
        #self.K = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:,:self.n_occupied,:,:self.n_occupied]
        self.J = copy.deepcopy(self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:self.n_occupied,:self.n_occupied,:,:])
        self.J_temp = copy.deepcopy(self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:self.n_occupied,:self.n_occupied,:,:])
        self.K = copy.deepcopy(self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:,:self.n_occupied,:,:self.n_occupied])
        self.K_temp = copy.deepcopy(self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:,:self.n_occupied,:,:self.n_occupied])
        #print (np.shape(self.J),np.shape(self.K))
        self.K = self.K.transpose(1,3,0,2)
        self.K_temp = self.K_temp.transpose(1,3,0,2)
        #print (np.shape(self.K))
    def build_intermediates_internal(self, eigenvecs, A, G, occupied_fock_core, occupied_d_cmo, occupied_J, occupied_K):
        rot_dim = self.n_occupied
        L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        fock_general = np.zeros((rot_dim, rot_dim))
        #print (np.shape(self.L))
        L = 4.0 * occupied_K[:,:self.n_in_a,:rot_dim,:rot_dim] - occupied_K.transpose(0,1,3,2)[:,:self.n_in_a,:rot_dim,:rot_dim] - occupied_J[:,:self.n_in_a,:rot_dim,:rot_dim]
        fock_general += occupied_fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
           occupied_J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim])
        fock_general -= 0.5 * np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), occupied_K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim])
       
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)
        A[:,:self.n_in_a] = 2.0 * (fock_general[:,:self.n_in_a] + occupied_d_cmo[:rot_dim,:self.n_in_a] * off_diagonal_constant)
        
        A[:,self.n_in_a:self.n_occupied] = np.einsum("rt,tu->ru", occupied_fock_core[:rot_dim,self.n_in_a:self.n_occupied],
                self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        twoeint_rtvw = occupied_J[:,self.n_in_a:self.n_occupied,
                self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]
        #print(np.shape(self.active_twoeint))
        A[:,self.n_in_a:self.n_occupied] += np.einsum("rtvw,tuvw->ru", twoeint_rtvw[:rot_dim,:,:,:], 
           self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        A[:,self.n_in_a:self.n_occupied] += -np.sqrt(self.omega/2) * np.einsum("rt,tu->ru", occupied_d_cmo[:rot_dim,self.n_in_a:self.n_occupied],
                self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        #print(self.A-self.A.transpose())
       


        #if full_space == True: 
            #self.A2 = np.zeros((self.nmo, self.nmo))
            #for r in range(self.nmo):
            #    for i in range(self.n_in_a):
            #        self.A2[r][i] = 2.0 * self.fock_general[r][i]
            #        self.A2[r][i] += 2.0 * off_diagonal_constant * self.d_cmo[r][i]

            #for r in range(self.nmo):
            #    for u in range(self.n_act_orb):
            #        for t in range(self.n_act_orb):
            #            self.A2[r][u+self.n_in_a] += self.fock_core[r][t+self.n_in_a] * self.D_tu_avg[t*self.n_act_orb + u]
            #            self.A2[r][u+self.n_in_a] += -np.sqrt(self.omega/2) * self.d_cmo[r][t+self.n_in_a] * self.Dpe_tu_avg[t*self.n_act_orb + u]
            #            for v in range(self.n_act_orb):
            #                for w in range(self.n_act_orb):
            #                    tu = t * self.n_act_orb + u            
            #                    vw = v * self.n_act_orb + w            
            #                    self.A2[r][u+self.n_in_a] += self.twoeint_rtvw[r][t][v][w] * self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw]
            #print("uue", np.allclose(self.A2,self.A, rtol=1e-14,atol=1e-14))









        G[:self.n_in_a,:self.n_in_a,:,:] = 2.0 * np.einsum("rs,ij->ijrs", fock_general[:rot_dim,:rot_dim],
           np.eye(self.n_in_a))
        G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * L[:self.n_in_a,:,:,:]
        G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * off_diagonal_constant * np.einsum("rs,ij->ijrs", occupied_d_cmo[:rot_dim,:rot_dim],
           np.eye(self.n_in_a))

        G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:] = np.einsum("tv,vjrs->tjrs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)),
           L[self.n_in_a:,:,:,:])
        G[:self.n_in_a,self.n_in_a:self.n_occupied,:,:] = G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:].transpose(1,0,3,2) 


        G[self.n_in_a:,self.n_in_a:,:,:] = np.einsum("rs,tu->turs", occupied_fock_core[:rot_dim,:rot_dim],
           self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        G[self.n_in_a:,self.n_in_a:,:,:] += np.einsum("vwrs,tuvw->turs", occupied_J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
           self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        G[self.n_in_a:,self.n_in_a:,:,:] += 2.0 * np.einsum("vwrs,tvuw->turs", occupied_K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
           self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        G[self.n_in_a:,self.n_in_a:,:,:] += -np.sqrt(self.omega/2) * np.einsum("rs,tu->turs", occupied_d_cmo[:rot_dim,:rot_dim],
                self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        #for r in range(self.nmo):
        #    for s in range(self.nmo):
        #        for v in range(self.n_act_orb):
        #            for w in range(self.n_act_orb):
        #                dum = self.K[v][w][r][s]-self.K[w][v][s][r]
        #                if dum > 1e-12: print('{:.5f}'.format(dum))





        print("n_act_orb,n_in_orb, nmo,nvir",self.n_act_orb,self.n_in_a,self.nmo,self.n_virtual)  
        #print("ooq", np.allclose(self.G2,G, rtol=1e-14,atol=1e-14))
        #self.G3 = G.transpose(0,2,1,3)
        #self.G3 = self.G3.reshape(self.n_occupied * rot_dim,self.n_occupied * rot_dim)
        #print("symmetric test", np.allclose(self.G3.T,self.G3, rtol=1e-14,atol=1e-14))
        ##print(self.G3)
        #for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for k in range(self.n_occupied):
        #            for l in range(self.n_occupied):
        #                dum = self.G3.T[k*rot_dim+r][l*rot_dim+s]-self.G3[k*rot_dim+r][l*rot_dim+s]
        #                #if dum > 1e-14: print('{:.14f}'.format(dum), k, l, r, s)
        #                #if dum > 1e-14: print('{:.16f}'.format(self.G3.T[k*self.nmo+r][l*self.nmo+s]),'{:.16f}'.format(self.G3[k*self.nmo+r][l*self.nmo+s]) , 2.0 * self.fock_general[r][s],k, l, r, s)
        #                if dum > 1e-14: print('{:.16f}'.format(self.G3[l*rot_dim+s][k*rot_dim+r]),'{:.16f}'.format(self.G3[k*rot_dim+r][l*rot_dim+s]) ,
        #                        #2.0 * self.fock_general[r][s],2.0 * self.fock_general[s][r],k, l, r, s)
        #                        2.0 * self.fock_core[r][s],2.0 * self.fock_core[s][r],k, l, r, s)



        #print("trq", self.G3.T-self.G3)
        #print(self.G2-self.G)

    def build_intermediates(self, eigenvecs, A, G, full_space):
        if full_space == True: rot_dim = self.nmo
        else: rot_dim = self.n_occupied
        #start = timer()
        self.fock_core = copy.deepcopy(self.H_spatial2) 
        self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:], optimize = "optimal") 
        self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:], optimize = "optimal") 
        D_tu_avg = self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)) 
        Dpe_tu_avg = self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb))
        D_tuvw_avg = self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb))
        self.E_core = 0.0  
        self.E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
        self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 
        #end   = timer()
        #print("build intermediate step 1", end - start)


        self.active_fock_core = copy.deepcopy(self.fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
        self.active_twoeint = copy.deepcopy(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied])
        self.L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        self.fock_general = np.zeros((rot_dim, rot_dim))
        #print (np.shape(self.L))
        #start = timer()
        self.L = 4.0 * self.K[:,:self.n_in_a,:rot_dim,:rot_dim] - self.K.transpose(0,1,3,2)[:,:self.n_in_a,:rot_dim,:rot_dim] - self.J[:,:self.n_in_a,:rot_dim,:rot_dim]
        #end   = timer()
        #print("build intermediate step 2", end - start)
        ###self.L2 = np.zeros((self.n_occupied, self.n_in_a, self.nmo, self.nmo))
        ###for k in range(self.n_occupied):
        ###    for j in range(self.n_in_a):
        ###        for r in range(self.nmo):
        ###            for s in range(self.nmo):
        ###                self.L2[k][j][r][s] = (4.0 * self.K[k][j][r][s] -
        ###                    self.K[k][j][s][r] - self.J[k][j][r][s])
        ###print((self.L==self.L2).all())
        
        #self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
        #        self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        #self.fock_general -= 0.5 * np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
        #        self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        temp1 = self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim] -0.5 * self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim]
        #start = timer()
        self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", D_tu_avg, 
                temp1, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 3", end - start)
        ###self.fock_general2 = np.zeros((self.nmo, self.nmo))
        ###for r in range(self.nmo):
        ###    for s in range(self.nmo):
        ###        self.fock_general2[r][s] = self.fock_core[r][s]
        ###        for t in range(self.n_act_orb):
        ###            for u in range(self.n_act_orb):
        ###                self.fock_general2[r][s] += self.D_tu_avg[t*self.n_act_orb +u] * self.J[t+self.n_in_a][u+self.n_in_a][r][s]
        ###                self.fock_general2[r][s] -= 0.5 * self.D_tu_avg[t*self.n_act_orb +u] * self.K[t+self.n_in_a][u+self.n_in_a][r][s]
        ###print("rqq", np.allclose(self.fock_general,self.fock_general2, rtol=1e-14,atol=1e-14))
        #start = timer()
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)
        A[:,:self.n_in_a] = 2.0 * (self.fock_general[:,:self.n_in_a] + self.d_cmo[:rot_dim,:self.n_in_a] * off_diagonal_constant)
        
        #off_diagonal_constant2 = 0.0
        #np1 = self.N_p + 1
        #for i in range(self.davidson_roots):
        #    eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
        #    eigenvecs2 = eigenvecs2.transpose(1,0)
        #    for m in range(np1):
        #        for I in range(self.num_det):
        #            if (self.N_p ==0): continue
        #            if (m > 0 and m < self.N_p):
        #                off_diagonal_constant2+= -np.sqrt(m * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m-1)*self.num_det+I]
        #                off_diagonal_constant2 += -np.sqrt((m+1) * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m+1)*self.num_det+I]
        #            elif (m == self.N_p):
        #                off_diagonal_constant2 += -np.sqrt(m * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m-1)*self.num_det+I]
        #            else:
        #                off_diagonal_constant2 += -np.sqrt((m+1) * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m+1)*self.num_det+I]
        #print("rtty", off_diagonal_constant2, off_diagonal_constant)
        
        A[:,self.n_in_a:self.n_occupied] = np.einsum("rt,tu->ru", self.fock_core[:rot_dim,self.n_in_a:self.n_occupied],
                D_tu_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 4", end - start)
        #print(np.shape(self.active_twoeint))
        #start = timer()
        A[:,self.n_in_a:self.n_occupied] += np.einsum("vwrt,tuvw->ru", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,self.n_in_a:self.n_occupied], 
           D_tuvw_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 5", end - start)
        #start = timer()
        A[:,self.n_in_a:self.n_occupied] += -np.sqrt(self.omega/2) * np.einsum("rt,tu->ru", self.d_cmo[:rot_dim,self.n_in_a:self.n_occupied],
                Dpe_tu_avg, optimize = "optimal")
        #print(self.A-self.A.transpose())
        #end   = timer()
        #print("build intermediate step 6", end - start)


        #if full_space == True: 
            #self.A2 = np.zeros((self.nmo, self.nmo))
            #for r in range(self.nmo):
            #    for i in range(self.n_in_a):
            #        self.A2[r][i] = 2.0 * self.fock_general[r][i]
            #        self.A2[r][i] += 2.0 * off_diagonal_constant * self.d_cmo[r][i]

            #for r in range(self.nmo):
            #    for u in range(self.n_act_orb):
            #        for t in range(self.n_act_orb):
            #            self.A2[r][u+self.n_in_a] += self.fock_core[r][t+self.n_in_a] * self.D_tu_avg[t*self.n_act_orb + u]
            #            self.A2[r][u+self.n_in_a] += -np.sqrt(self.omega/2) * self.d_cmo[r][t+self.n_in_a] * self.Dpe_tu_avg[t*self.n_act_orb + u]
            #            for v in range(self.n_act_orb):
            #                for w in range(self.n_act_orb):
            #                    tu = t * self.n_act_orb + u            
            #                    vw = v * self.n_act_orb + w            
            #                    self.A2[r][u+self.n_in_a] += self.twoeint_rtvw[r][t][v][w] * self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw]
            #print("uue", np.allclose(self.A2,self.A, rtol=1e-14,atol=1e-14))



        #G[:self.n_in_a,:self.n_in_a,:,:] = 2.0 * np.einsum("rs,ij->ijrs", self.fock_general[:rot_dim,:rot_dim],
        #  np.eye(self.n_in_a))
        #G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * self.L[:self.n_in_a,:,:,:]
        #G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * off_diagonal_constant * np.einsum("rs,ij->ijrs", self.d_cmo[:rot_dim,:rot_dim],
        #   np.eye(self.n_in_a))
        #start = timer()
        #start1 = timer()
        temp2 = self.fock_general[:rot_dim,:rot_dim] + off_diagonal_constant * self.d_cmo[:rot_dim,:rot_dim]
        #end1   = timer()
        #print("build intermediate step 7_0", end1 - start1)
        #start1 = timer()
        G[:self.n_in_a,:self.n_in_a,:,:] = 2.0 * np.einsum("rs,ij->ijrs", temp2,
          np.eye(self.n_in_a), optimize = "optimal")
        #end1   = timer()
        #print("build intermediate step 7_1", end1 - start1)
        #start1 = timer()
        G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * self.L[:self.n_in_a,:,:,:]
        #end1   = timer()
        #print("build intermediate step 7_2", end1 - start1)
        #end   = timer()
        #print("build intermediate step 7", end - start)





        #start = timer()
        G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:] = np.einsum("tv,vjrs->tjrs", D_tu_avg,
           self.L[self.n_in_a:,:,:,:], optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 8", end - start)
        #start = timer()
        G[:self.n_in_a,self.n_in_a:self.n_occupied,:,:] = G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:].transpose(1,0,3,2) 

        #end   = timer()
        #print("build intermediate step 9", end - start)
        #start = timer()
        G[self.n_in_a:,self.n_in_a:,:,:] = np.einsum("rs,tu->turs", self.fock_core[:rot_dim,:rot_dim],
           D_tu_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 10", end - start)
        #start = timer()
        G[self.n_in_a:,self.n_in_a:,:,:] += np.einsum("vwrs,tuvw->turs", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
           D_tuvw_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 11", end - start)
        #start = timer()
        G[self.n_in_a:,self.n_in_a:,:,:] += 2.0 * np.einsum("vwrs,tvuw->turs", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
           D_tuvw_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 12", end - start)
        #start = timer()
        G[self.n_in_a:,self.n_in_a:,:,:] += -np.sqrt(self.omega/2) * np.einsum("rs,tu->turs", self.d_cmo[:rot_dim,:rot_dim],
                Dpe_tu_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 13", end - start)
        #for r in range(self.nmo):
        #    for s in range(self.nmo):
        #        for v in range(self.n_act_orb):
        #            for w in range(self.n_act_orb):
        #                dum = self.K[v][w][r][s]-self.K[w][v][s][r]
        #                if dum > 1e-12: print('{:.5f}'.format(dum))
        ##test symmetry of 1rdm and fock matrix
        #self.D3 = self.D_tu_avg.reshape(self.n_act_orb,self.n_act_orb)
        #print("wew", np.allclose(self.D3.T,self.D3, rtol=1e-14,atol=1e-14))

        #print("gsw", np.allclose(self.fock_general.T,self.fock_general, rtol=1e-14,atol=1e-14))




        #self.G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for i in range(self.n_in_a):
        #            for j in range(self.n_in_a):
        #                self.G2[i][j][r][s] = 2.0 * (i==j) * self.fock_general[r][s]
        #                self.G2[i][j][r][s] += 2.0 * self.L[i][j][r][s]
        #                self.G2[i][j][r][s] += 2.0 * off_diagonal_constant * (i==j) * self.d_cmo[r][s]
        #for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for t in range(self.n_act_orb):
        #            for j in range(self.n_in_a):
        #                for v in range(self.n_act_orb):
        #                    self.G2[t+self.n_in_a][j][r][s] += self.D_tu_avg[t * self.n_act_orb + v] * self.L[v+self.n_in_a][j][r][s]
        #                self.G2[j][t+self.n_in_a][s][r] = self.G2[t+self.n_in_a][j][r][s] 
        #for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for t in range(self.n_act_orb):
        #            for u in range(self.n_act_orb):
        #                tu = t * self.n_act_orb + u
        #                self.G2[t+self.n_in_a][u+self.n_in_a][r][s] = self.D_tu_avg[t * self.n_act_orb + u] * self.fock_core[r][s]
        #                self.G2[t+self.n_in_a][u+self.n_in_a][r][s] += -np.sqrt(self.omega/2) * self.Dpe_tu_avg[t * self.n_act_orb + u] * self.d_cmo[r][s]
        #                for v in range(self.n_act_orb):
        #                    for w in range(self.n_act_orb):
        #                        vw = v * self.n_act_orb + w
        #                        tv = t * self.n_act_orb + v
        #                        uw = u * self.n_act_orb + w
        #                        self.G2[t+self.n_in_a][u+self.n_in_a][r][s] += self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] * self.J[v+self.n_in_a][w+self.n_in_a][r][s]
        #                        self.G2[t+self.n_in_a][u+self.n_in_a][r][s] += 2.0 * self.D_tuvw_avg[tv * self.n_act_orb * self.n_act_orb + uw] * self.K[v+self.n_in_a][w+self.n_in_a][r][s]
        #print("n_act_orb,n_in_orb, nmo,nvir",self.n_act_orb,self.n_in_a,self.nmo,self.n_virtual)  
        #print("ooq", np.allclose(self.G2,G, rtol=1e-14,atol=1e-14))
        #self.G3 = G.transpose(0,2,1,3)
        #self.G3 = self.G3.reshape(self.n_occupied * rot_dim,self.n_occupied * rot_dim)
        #print("symmetric test", np.allclose(self.G3.T,self.G3, rtol=1e-14,atol=1e-14))
        ##print(self.G3)
        #for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for k in range(self.n_occupied):
        #            for l in range(self.n_occupied):
        #                dum = self.G3.T[k*rot_dim+r][l*rot_dim+s]-self.G3[k*rot_dim+r][l*rot_dim+s]
        #                #if dum > 1e-14: print('{:.14f}'.format(dum), k, l, r, s)
        #                #if dum > 1e-14: print('{:.16f}'.format(self.G3.T[k*self.nmo+r][l*self.nmo+s]),'{:.16f}'.format(self.G3[k*self.nmo+r][l*self.nmo+s]) , 2.0 * self.fock_general[r][s],k, l, r, s)
        #                if dum > 1e-14: print('{:.16f}'.format(self.G3[l*rot_dim+s][k*rot_dim+r]),'{:.16f}'.format(self.G3[k*rot_dim+r][l*rot_dim+s]) ,
        #                        #2.0 * self.fock_general[r][s],2.0 * self.fock_general[s][r],k, l, r, s)
        #                        2.0 * self.fock_core[r][s],2.0 * self.fock_core[s][r],k, l, r, s)



        #print("trq", self.G3.T-self.G3)
        #print(self.G2-self.G)
   

    def build_intermediates2(self, eigenvecs, A, G, full_space):
        if full_space == True: rot_dim = self.nmo
        else: rot_dim = self.n_occupied
        self.fock_core = copy.deepcopy(self.H_spatial2) 
        self.fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
        self.fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
        
        self.E_core = 0.0  
        self.E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
        self.E_core += np.einsum("jj->", self.fock_core[:self.n_in_a,:self.n_in_a]) 


        self.active_fock_core = copy.deepcopy(self.fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
        self.active_twoeint = copy.deepcopy(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied])
        self.L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        self.fock_general = np.zeros((rot_dim, rot_dim))
        print (np.shape(self.L))
        self.L = 4.0 * self.K[:,:self.n_in_a,:rot_dim,:rot_dim] - self.K.transpose(0,1,3,2)[:,:self.n_in_a,:rot_dim,:rot_dim] - self.J[:,:self.n_in_a,:rot_dim,:rot_dim]
        
        #self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
        self.fock_general +=  np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
                self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim])
        self.fock_general -= 0.5 * np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim])
        
        
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)
        
        #A[:,:self.n_in_a] = 2.0 * (self.fock_general[:,:self.n_in_a] + self.d_cmo[:rot_dim,:self.n_in_a] * off_diagonal_constant)
        A[:,:self.n_in_a] = 2.0 * (self.fock_general[:,:self.n_in_a] )
        
        
        
        A[:,self.n_in_a:self.n_occupied] = np.einsum("rt,tu->ru", self.fock_core[:rot_dim,self.n_in_a:self.n_occupied],
                self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        #print(np.shape(self.active_twoeint))
        #A[:,self.n_in_a:self.n_occupied] += np.einsum("vwrt,tuvw->ru", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,self.n_in_a:self.n_occupied], 
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        #A[:,self.n_in_a:self.n_occupied] += -np.sqrt(self.omega/2) * np.einsum("rt,tu->ru", self.d_cmo[:rot_dim,self.n_in_a:self.n_occupied],
        #        self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        #print(self.A-self.A.transpose())
       


        


        G[:self.n_in_a,:self.n_in_a,:,:] = 2.0 * np.einsum("rs,ij->ijrs", self.fock_general[:rot_dim,:rot_dim],
          np.eye(self.n_in_a))
        #G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * self.L[:self.n_in_a,:,:,:]
        #G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * off_diagonal_constant * np.einsum("rs,ij->ijrs", self.d_cmo[:rot_dim,:rot_dim],
        #   np.eye(self.n_in_a))

        G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:] = np.einsum("tv,vjrs->tjrs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)),
           self.L[self.n_in_a:,:,:,:])
        G[:self.n_in_a,self.n_in_a:self.n_occupied,:,:] = G[self.n_in_a:self.n_occupied,:self.n_in_a,:,:].transpose(1,0,3,2) 


        G[self.n_in_a:,self.n_in_a:,:,:] = np.einsum("rs,tu->turs", self.fock_core[:rot_dim,:rot_dim],
           self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        
        #G[self.n_in_a:,self.n_in_a:,:,:] += np.einsum("vwrs,tuvw->turs", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        #G[self.n_in_a:,self.n_in_a:,:,:] += 2.0 * np.einsum("vwrs,tvuw->turs", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        #G[self.n_in_a:,self.n_in_a:,:,:] += -np.sqrt(self.omega/2) * np.einsum("rs,tu->turs", self.d_cmo[:rot_dim,:rot_dim],
        #        self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        

       

        




   
    def calculate_ci_dependent_energy(self, eigenvecs, occupied_d_cmo):
        off_diagonal_constant_energy = 0.0
        photon_energy = 0.0
        d_diag = 0.0
        #for i in range(self.n_in_a):
        #    d_diag += 2.0 * self.d_cmo[i][i]
        d_diag = 2.0 * np.einsum("ii->", occupied_d_cmo[:self.n_in_a,:self.n_in_a])
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            #print("weight", self.weight[i])
            eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
            eigenvecs2 = eigenvecs2.transpose(1,0)
            for m in range(np1):
                if (self.N_p ==0): continue
                if (m > 0 and m < self.N_p):
                    off_diagonal_constant_energy += self.weight[i] * np.sqrt(m * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                    off_diagonal_constant_energy += self.weight[i] * np.sqrt((m+1) * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                elif (m == self.N_p):
                    off_diagonal_constant_energy += self.weight[i] * np.sqrt(m * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                else:
                    off_diagonal_constant_energy += self.weight[i] * np.sqrt((m+1) * self.omega/2) * (self.d_exp - d_diag) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                photon_energy  += self.weight[i] * m * self.omega * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,m:(m+1)].flatten())

        #ci_dependent_energy = (off_diagonal_constant_energy + photon_energy) / self.davidson_roots
        ci_dependent_energy = (off_diagonal_constant_energy + photon_energy) 
        return ci_dependent_energy 


    def calculate_off_diagonal_photon_constant(self, eigenvecs):
        off_diagonal_constant = 0.0
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            #print("weight", self.weight[i])
            eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
            eigenvecs2 = eigenvecs2.transpose(1,0)
            for m in range(np1):
                if (self.N_p ==0): continue
                if (m > 0 and m < self.N_p):
                    off_diagonal_constant += -self.weight[i] * np.sqrt(m * self.omega/2) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                    off_diagonal_constant += -self.weight[i] * np.sqrt((m+1) * self.omega/2) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())
                elif (m == self.N_p):
                    off_diagonal_constant += -self.weight[i] * np.sqrt(m * self.omega/2) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m-1):m].flatten())
                else:
                    off_diagonal_constant += -self.weight[i] * np.sqrt((m+1) * self.omega/2) * np.dot(eigenvecs2[:,m:(m+1)].flatten(),eigenvecs2[:,(m+1):(m+2)].flatten())

        return off_diagonal_constant

    def build_unitary_matrix(self, Rai, Rvi, Rva):
        R = np.zeros((self.nmo, self.nmo))
        self.U_delta = np.zeros((self.nmo, self.nmo))

        R[self.n_in_a:self.n_occupied,:self.n_in_a] = Rai 
        R[:self.n_in_a,self.n_in_a:self.n_occupied] = -Rai.T 
        R[self.n_occupied:,:self.n_in_a] = Rvi 
        R[:self.n_in_a,self.n_occupied:] = -Rvi.T 
        R[self.n_occupied:,self.n_in_a:self.n_occupied] = Rva 
        R[self.n_in_a:self.n_occupied,self.n_occupied:] = -Rva.T 
        np.set_printoptions(precision=5)
        #print("Rai",Rai)
        #print("Rvi",Rvi)
        #print("Rva",Rva)
        #print("R",R)

        #print("nbm", np.allclose(R.T,-R, rtol=1e-14,atol=1e-14))
        #print((R.T==-R).all())
        #R1 = np.einsum("pq,qr->pr", -R,R)
        R1 = np.dot(-R,R)
        tau_square, W = np.linalg.eigh(R1)
        sine_product_array = np.zeros(self.nmo)
        cosine_array = np.zeros(self.nmo)
        for i in range(self.nmo):
            if tau_square[i]<0.0: 
                tau_square[i] =0.0
            tau = np.sqrt(tau_square[i])
            cosine_array[i] = np.cos(tau)
            if np.sqrt(tau_square[i]) > 1e-15:
                sine_product_array[i] = np.sin(tau)/tau
            else:
                #print("dfgsg")
                sine_product_array[i] = 1 - pow(tau,2)/6 + pow(tau,4)/120
        #self.U_delta = np.einsum("pq,qs,st->pt", W, np.diag(cosine_array), W.T)
        #temp = np.einsum("pq,qs->ps", W.T, R)
        #self.U_delta += np.einsum("pq,qs,st->pt", W, np.diag(sine_product_array), temp)
        temp = np.dot(np.diag(cosine_array), W.T)
        self.U_delta = np.dot(W, temp)
        temp = np.dot(W.T, R)
        R1 = np.dot(np.diag(sine_product_array), temp)
        self.U_delta += np.dot(W, R1)
        
    
    def build_gradient(self, U, A, G, gradient_tilde, A_tilde, full_space):
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
           
            A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
            #A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)
           
            gradient_tilde[:,:] = A_tilde[:,:self.n_occupied] - A_tilde.T[:,:self.n_occupied]
           
        else: 

            gradient_tilde[:,:] = A[:,:] - A.T[:,:]
    
    def build_gradient2(self, U, A, G, hessian_tilde, gradient_tilde, A_tilde, full_space):
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied], optimize ='optimal')
           
            A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied], optimize ='optimal')
            #A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)
           
            gradient_tilde[:,:] = A_tilde[:,:self.n_occupied] - A_tilde.T[:,:self.n_occupied]
            A3_tilde =  (A_tilde + A_tilde.T)
        
            #hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            
            hessian_tilde[:,:,:,:] = -0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde, optimize ='optimal')
            hessian_tilde[:,:,:,:] -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied], optimize ='optimal')
            hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:], optimize ='optimal')
            hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied], optimize ='optimal')

  
        else: 

            gradient_tilde[:,:] = A[:,:] - A.T[:,:]
            A3_tilde = A + A.T 
            hessian_tilde[:,:,:,:] = -0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde, optimize ='optimal')
            hessian_tilde[:,:,:,:] -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:self.n_occupied], optimize ='optimal')
            hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:], optimize ='optimal')
            hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied], optimize ='optimal')
       


    def build_gradient_and_hessian(self, U, A, G, gradient_tilde,hessian_tilde, full_space):
        print("omppm")
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
            B2 = np.zeros((self.nmo, self.nmo))
            for r in range(self.nmo):
                for k in range(self.n_occupied):
                    B2[r][k] = A[r][k]
                    for s in range(self.nmo):
                        for l in range(self.n_occupied):
                            B2[r][k] += G[k][l][r][s] * T[s][l]
            print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))
            
            A_tilde = np.zeros((self.nmo, self.nmo))
            A2_tilde = np.zeros((self.nmo, self.nmo))
            #A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
            A_tilde = np.einsum("rs,sk->rk",U.T, B)
            A2_tilde = np.zeros((self.nmo, self.nmo))
            for r in range(self.nmo):
                for k in range(self.n_occupied):
                    for s in range(self.nmo):
                        A2_tilde[r][k] += U[s][r] * B[s][k]
            print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))
            
            G_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            G_tilde = np.einsum("pr,klpq,qs->klrs",U, G, U)
            G2_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            for k in range(self.n_occupied):
                for l in range(self.n_occupied):
                    for r in range(self.nmo):
                        for s in range(self.nmo):
                            for p in range(self.nmo):
                                for q in range(self.nmo):
                                    G2_tilde[k][l][r][s] += U[p][r] * G[k][l][p][q] * U[q][s] 
            print("ngfn", np.allclose(G2_tilde,G_tilde, rtol=1e-14,atol=1e-14))
            
            gradient_tilde[:,:] = A_tilde[:,:self.n_occupied] - A_tilde.T[:,:self.n_occupied]
            gradient2_tilde = np.zeros((self.nmo, self.n_occupied))
            for k in range(self.n_occupied):
                for r in range(self.nmo):
                    gradient2_tilde[r][k] = A_tilde[r][k] - A_tilde[k][r]
            print("ytuu", np.allclose(gradient2_tilde,gradient_tilde, rtol=1e-14,atol=1e-14))
            gradient_norm = np.dot(gradient_tilde.flatten(), gradient_tilde.flatten())
            np.set_printoptions(precision=9, suppress=True)
            gradient_norm = np.sqrt(gradient_norm)
            #for k in range(self.n_occupied):
            #    for r in range(self.nmo):
            #        if(gradient_tilde[r][k]>1e-10):
            #            print("efgfb",gradient_tilde[r][k]) 
            #print(A_tilde)
            print(gradient_norm) 
            A3_tilde =  (A_tilde + A_tilde.T) 
            #hessian_tilde = np.copy(G_tilde)
            hessian_tilde[:,:,:,:] = G_tilde[:,:,:,:]
            hessian_tilde[:,:,:self.n_occupied,:] -= G_tilde[:,:,:self.n_occupied,:].transpose(2,1,0,3)
            hessian_tilde[:,:,:,:self.n_occupied] -= G_tilde[:,:,:,:self.n_occupied].transpose(0,3,2,1)
            hessian_tilde[:,:,:self.n_occupied,:self.n_occupied] += G_tilde[:,:,:self.n_occupied,:self.n_occupied].transpose(2,3,0,1)
            hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
            hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
            hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
            hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
            hessian2_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            np.set_printoptions(suppress=True)
            for k in range(self.n_occupied):
                for l in range(self.n_occupied):
                    for r in range(self.nmo):
                        for s in range(self.nmo):
                            hessian2_tilde[k][l][r][s] = G_tilde[k][l][r][s]
                            hessian2_tilde[k][l][r][s] -= 0.5 * (k==l) * (A_tilde[r][s]+A_tilde[s][r])
                            hessian2_tilde[k][l][r][s] += 0.5 * (r==l) * (A_tilde[k][s]+A_tilde[s][k])
                            hessian2_tilde[k][l][r][s] += 0.5 * (k==s) * (A_tilde[r][l]+A_tilde[l][r])
                            hessian2_tilde[k][l][r][s] -= 0.5 * (r==s) * (A_tilde[k][l]+A_tilde[l][k])
                            if r < self.n_occupied: 
                                hessian2_tilde[k][l][r][s] -= G_tilde[r][l][k][s]
                            if s < self.n_occupied: 
                                hessian2_tilde[k][l][r][s] -= G_tilde[k][s][r][l]
                            if r < self.n_occupied and s < self.n_occupied: 
                                hessian2_tilde[k][l][r][s] += G_tilde[r][s][k][l]

            #print(A_tilde)
            #print(B)
            print("bfad", np.allclose(hessian2_tilde,hessian_tilde, rtol=1e-14,atol=1e-14))
            #print(gradient_tilde)
            G3 = hessian_tilde.transpose(0,2,1,3)
            G3 = G3.reshape(self.n_occupied * self.nmo,self.n_occupied * self.nmo)
            self.G7 = G3 
            omega, eig_vecs = np.linalg.eigh(G3)
            print("nbjfgbnjtryrt",omega)
        else: 

            gradient_tilde[:,:] = A[:,:] - A.T[:,:]

            hessian_tilde[:,:,:,:] = G[:,:,:,:]
            hessian_tilde[:,:,:self.n_occupied,:] -= G[:,:,:self.n_occupied,:].transpose(2,1,0,3)
            hessian_tilde[:,:,:,:self.n_occupied] -= G[:,:,:,:self.n_occupied].transpose(0,3,2,1)
            hessian_tilde[:,:,:self.n_occupied,:self.n_occupied] += G[:,:,:self.n_occupied,:self.n_occupied].transpose(2,3,0,1)
            A3_tilde = A + A.T 
            hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
            hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:self.n_occupied])
            hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
            hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
       
    def internal_transformation(self, U, h1, d_cmo1, J, K):
        #self.occupied_J2 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        #self.occupied_K2 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        #for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for m_p in range(self.n_occupied):
        #            for n_p in range(self.n_occupied):
        #                for k in range(self.n_occupied):
        #                    for l in range(self.n_occupied):
        #                        for m in range(self.n_occupied):
        #                            for n in range(self.n_occupied):
        #                                self.occupied_J2[k_p][l_p][m_p][n_p] += U[k][k_p] * U[l][l_p] * U[m][m_p] * U[n][n_p] * self.J[k][l][m][n]
        #                                self.occupied_K2[k_p][l_p][m_p][n_p] += U[k][k_p] * U[l][l_p] * U[m][m_p] * U[n][n_p] * self.K[k][l][m][n]

        #self.occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
        #self.occupied_d_cmo1 = np.zeros((self.n_occupied, self.n_occupied))
        #for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for k in range(self.n_occupied):
        #            for l in range(self.n_occupied):
        #                self.occupied_h1[k_p][l_p] += U[k][k_p] * U[l][l_p] * self.H_spatial2[k][l]
        #                self.occupied_d_cmo1[k_p][l_p] += U[k][k_p] * U[l][l_p] * self.d_cmo[k][l]


        occupied_twoeint1 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        occupied_twoeint2 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        #occupied_twoeint1 = np.einsum("klmn,ns->klms", self.J[:,:,:self.n_occupied,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        occupied_twoeint1 = np.einsum("klmn,ns->klms", self.occupied_J[:,:,:self.n_occupied,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        occupied_twoeint2 = np.einsum("klms,mr->klrs", occupied_twoeint1, U[:self.n_occupied,:self.n_occupied])
        occupied_twoeint1 = np.einsum("klrs,lq->kqrs", occupied_twoeint2, U[:self.n_occupied,:self.n_occupied])
        occupied_twoeint2 = np.einsum("kqrs,kp->pqrs", occupied_twoeint1, U[:self.n_occupied,:self.n_occupied])
        J[:,:,:self.n_occupied,:self.n_occupied] = occupied_twoeint2[:,:,:,:] 
        K[:,:,:self.n_occupied,:self.n_occupied] = occupied_twoeint2[:,:,:,:].transpose(1,3,0,2) 
        
        occupied_oneeint1 = np.zeros((self.n_occupied, self.n_occupied))
        occupied_oneeint1 = np.einsum("ij,jl->il", self.occupied_h1[:self.n_occupied,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        h1[:self.n_occupied,:self.n_occupied] = np.einsum("il,ik->kl", occupied_oneeint1, U[:self.n_occupied,:self.n_occupied])
        occupied_oneeint1 = np.einsum("ij,jl->il", self.occupied_d_cmo[:self.n_occupied,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        d_cmo1[:self.n_occupied,:self.n_occupied] = np.einsum("il,ik->kl", occupied_oneeint1, U[:self.n_occupied,:self.n_occupied])
        #print("fdrx", np.allclose(self.occupied_K2,K[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        #print("lknx", np.allclose(self.occupied_J2,J[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        #print("vcvx", np.allclose(self.occupied_h1,h1[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        #print("trgx", np.allclose(self.occupied_d_cmo1,d_cmo1[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
         
    #def internal_update_ci_integrals(self, E_core, occupied_fock_core, occupied_h1, occupied_d_cmo, occupied_J, occupied_K):
    #    self.J[:,:,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
    #    self.twoeint = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
    #    self.twoeint[:self.n_occupied,:self.n_occupied,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
    #    self.twoeint = self.twoeint.reshape((self.nmo * self.nmo, self.nmo * self.nmo))
    #    self.K[:,:,:self.n_occupied, :self.n_occupied] = occupied_K[:,:,:,:]
    #    self.H_spatial2[:self.n_occupied, :self.n_occupied] = occupied_h1[:,:]
    #    self.d_cmo[:self.n_occupied, :self.n_occupied] = occupied_d_cmo[:,:]
    #    self.fock_core[:self.n_occupied, :self.n_occupied] = occupied_fock_core[:,:]
    #    self.E_core = E_core

    def full_transformation_internal_optimization(self, U, h1, d_cmo1, J, K):
        #self.occupied_J2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.occupied_K2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.occupied_h1 = np.zeros((self.nmo, self.nmo))
        #self.occupied_d_cmo1 = np.zeros((self.nmo, self.nmo))
        #for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for r_p in range(self.nmo):
        #            for s_p in range(self.nmo):
        #                for k in range(self.n_occupied):
        #                    for l in range(self.n_occupied):
        #                        for r in range(self.nmo):
        #                            for s in range(self.nmo):
        #                                self.occupied_J2[k_p][l_p][r_p][s_p] += U[k][k_p] * U[l][l_p] * U[r][r_p] * U[s][s_p] * self.J[k][l][r][s]
        #                                self.occupied_K2[k_p][l_p][r_p][s_p] += U[k][k_p] * U[l][l_p] * U[r][r_p] * U[s][s_p] * self.K[k][l][r][s]
        #for r_p in range(self.nmo):
        #    for s_p in range(self.nmo):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                self.occupied_h1[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.H_spatial2[r][s]
        #                self.occupied_d_cmo1[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.d_cmo[r][s]





        twoeint1_ab = np.zeros((self.n_occupied, self.n_occupied, self.n_virtual, self.n_virtual))
        #twoeint2_ab = np.zeros((self.n_occupied, self.n_occupied, self.n_virtual, self.n_virtual))
        
        twoeint1_ab = np.einsum("klab,kp->plab", self.J[:,:,self.n_occupied:,self.n_occupied:], U[:self.n_occupied,:self.n_occupied])
        twoeint1_ms = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.nmo))
        twoeint2_ms = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.nmo))
        twoeint1_ms = np.einsum("klmn,kp->plmn", self.J[:,:,:self.n_occupied,:], U[:self.n_occupied,:self.n_occupied])
        twoeint2_ms = np.einsum("plmn,lq->pqmn", twoeint1_ms, U[:self.n_occupied,:self.n_occupied])
        twoeint1_ms = np.einsum("pqmn,mr->pqrn", twoeint2_ms, U[:self.n_occupied,:self.n_occupied])
        #J^kl_ab
        J[:,:,self.n_occupied:,self.n_occupied:] = np.einsum("plab,lq->pqab", twoeint1_ab, U[:self.n_occupied,:self.n_occupied])
        #J^kl_mn
        J[:,:,:self.n_occupied,:self.n_occupied] = np.einsum("pqrn,ns->pqrs", twoeint1_ms[:,:,:,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        #J^kl_ma
        J[:,:,:self.n_occupied,self.n_occupied:] = twoeint1_ms[:,:,:,self.n_occupied:]
        #J^kl_am = J^kl_ma
        J[:,:,self.n_occupied:,:self.n_occupied] = twoeint1_ms[:,:,:,self.n_occupied:].transpose(0,1,3,2)
        #J = np.copy(J9)
        twoeint1_ab = np.einsum("klab,kp->plab", self.K[:,:,self.n_occupied:,self.n_occupied:], U[:self.n_occupied,:self.n_occupied])
        twoeint1_ms = np.einsum("klmn,kp->plmn", self.K[:,:,:self.n_occupied,:], U[:self.n_occupied,:self.n_occupied])
        twoeint2_ms = np.einsum("plmn,lq->pqmn", twoeint1_ms, U[:self.n_occupied,:self.n_occupied])
        twoeint1_ms = np.einsum("pqmn,mr->pqrn", twoeint2_ms, U[:self.n_occupied,:self.n_occupied])
        #K^kl_ab
        K[:,:,self.n_occupied:,self.n_occupied:] = np.einsum("plab,lq->pqab", twoeint1_ab, U[:self.n_occupied,:self.n_occupied])
        #K^kl_mn
        K[:,:,:self.n_occupied,:self.n_occupied] = np.einsum("pqrn,ns->pqrs", twoeint1_ms[:,:,:,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        #K^kl_ma
        K[:,:,:self.n_occupied,self.n_occupied:] = twoeint1_ms[:,:,:,self.n_occupied:]
        #K^lk_am = K^kl_ma
        K[:,:,self.n_occupied:,:self.n_occupied] = twoeint1_ms[:,:,:,self.n_occupied:].transpose(1,0,3,2)
       
        oneeint1 = np.zeros((self.nmo, self.nmo))
        oneeint1 = np.einsum("rs,sq->rq", self.H_spatial2, U)
        h1[:,:] = np.einsum("rq,rp->pq", oneeint1, U)
        oneeint1 = np.einsum("rs,sq->rq", self.d_cmo, U)
        d_cmo1[:,:] = np.einsum("rq,rp->pq", oneeint1, U)
        #print("vdwt", np.allclose(self.occupied_K2,K, rtol=1e-14,atol=1e-14))
        #print("ujhn", np.allclose(self.occupied_J2,J, rtol=1e-14,atol=1e-14))
        #print("nhfd", np.allclose(self.occupied_h1,h1, rtol=1e-14,atol=1e-14))
        #print("eerd", np.allclose(self.occupied_d_cmo1,d_cmo1, rtol=1e-14,atol=1e-14))
    
     
    def full_transformation_macroiteration(self, U, J, K):
        
        self.twoeint = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
        #self.J3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.K3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        #self.h3 = np.zeros((self.nmo, self.nmo))
        #self.d_cmo3 = np.zeros((self.nmo, self.nmo))
        #for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for r_p in range(self.nmo):
        #            for s_p in range(self.nmo):
        #                for p in range(self.nmo):
        #                    for q in range(self.nmo):
        #                        for r in range(self.nmo):
        #                            for s in range(self.nmo):
        #                                self.J3[k_p][l_p][r_p][s_p] += U[p][k_p] * U[q][l_p] * U[r][r_p] * U[s][s_p] * self.twoeint[p][q][r][s]
        #                                self.K3[k_p][l_p][r_p][s_p] += U[r][r_p] * U[p][k_p] * U[s][s_p] * U[q][l_p] * self.twoeint[r][p][s][q]
        #for r_p in range(self.nmo):
        #    for s_p in range(self.nmo):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                self.h3[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.H1temp[r][s]
        #                self.d_cmo3[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.d_cmo_temp[r][s]


        temp1 = np.zeros((self.n_occupied, self.nmo, self.nmo, self.nmo))
        temp2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        temp3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        temp1 = np.einsum("pqrs,pk->kqrs", self.twoeint, U[:,:self.n_occupied])
        temp2 = np.einsum("kqrs,ql->klrs", temp1, U[:,:self.n_occupied])
        temp3 = np.einsum("klrs,rm->klms", temp2, U)
        J[:,:,:,:] = np.einsum("klms,sn->klmn", temp3, U)

        temp1 = temp1.transpose(0,2,1,3)
        temp2 = np.einsum("kqrs,ql->klrs", temp1, U[:,:self.n_occupied])
        temp3 = np.einsum("klrs,rm->klms", temp2, U)
        K[:,:,:,:] = np.einsum("klms,sn->klmn", temp3, U)
        #print("obne", np.allclose(self.K3,K, rtol=1e-14,atol=1e-14))
        #print("ztlv", np.allclose(self.J3,J, rtol=1e-14,atol=1e-14))
        #print("tdck", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
        #print("pgxq", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
          
    def internal_optimization_exact_energy(self, E0, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K):
        #for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                  dum = self.J[k][l][r][s] - self.J_temp[k][l][r][s]
        #                  if np.abs(dum) == 0.0:
        #                      print("{:20.12f}".format(dum), k, l, r, s)
        #                  #print("{:20.12f}".format(self.J[k][l][r][s]), "{:20.12f}".format(self.J_temp[k][l][r][s]), k, l, r, s)
        #for r in range(self.nmo):
        #    for s in range(self.nmo):
        #          dum = self.H_spatial2[r][s] - self.H1temp[r][s]
        #          ##if np.abs(dum) == 0.0:
        #          ##    print("{:20.12f}".format(dum), r, s)
        #          print("{:20.12f}".format(self.H_spatial2[r][s]), "{:20.12f}".format(self.H1temp[r][s]), r, s)

        #occupied fock core
        occupied_fock_core = copy.deepcopy(occupied_h1[:self.n_occupied,:self.n_occupied]) 
        occupied_fock_core += 2.0 * np.einsum("jjrs->rs", occupied_J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        occupied_fock_core -= np.einsum("jjrs->rs", occupied_K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        #occupied_fock_core1 = np.zeros((self.n_occupied, self.n_occupied))
        #for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        occupied_fock_core1[k][l] = self.H_spatial2[k][l]
        #        for j in range(self.n_in_a):
        #            occupied_fock_core1[k][l] += 2.0 * self.J[j][j][k][l] - self.K[j][j][k][l]
        #print("test occupied fock core")
        #for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        print("{:20.12f}".format(occupied_fock_core[k][l] - occupied_fock_core1[k][l]), k, l)
        E_core = 0.0  
        E_core += np.einsum("jj->", occupied_h1[:self.n_in_a,:self.n_in_a]) 
        E_core += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
        print("vtrhtr", E_core)
        active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
        active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
        active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
        ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, occupied_d_cmo)
        sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
                self.Enuc + self.d_c + ci_dependent_energy)
        if sum_energy - E0 < 0.0:
            self.occupied_J[:,:,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
            #self.twoeint = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
            #self.twoeint[:self.n_occupied,:self.n_occupied,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
            #self.twoeint = self.twoeint.reshape((self.nmo * self.nmo, self.nmo * self.nmo))
            self.occupied_K[:,:,:self.n_occupied, :self.n_occupied] = occupied_K[:,:,:,:]
            self.occupied_h1[:self.n_occupied, :self.n_occupied] = occupied_h1[:,:]
            self.occupied_d_cmo[:self.n_occupied, :self.n_occupied] = occupied_d_cmo[:,:]
            self.occupied_fock_core[:self.n_occupied, :self.n_occupied] = occupied_fock_core[:,:]
            print("fsdb",self.E_core,E_core)
            self.E_core = E_core
            #self.gkl3 = np.zeros((self.n_act_orb, self.n_act_orb))
            #for k in range(self.n_act_orb):
            #    for l in range(self.n_act_orb):
            #        self.gkl3[k][l] = occupied_fock_core[k+self.n_in_a][l+self.n_in_a]
            #        for j in range(self.n_act_orb):
            #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
            #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
            #            self.gkl3[k][l] -= 0.5 * self.twoeint[kj][jl]
            self.gkl2 = copy.deepcopy(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
            self.gkl2 -= 0.5 * np.einsum("kjjl->kl", 
                    self.occupied_J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 


            #print("vffs", np.allclose(self.gkl3,self.gkl2, rtol=1e-14,atol=1e-14))

        #print("test exact energy")
        #print(
        #            "{:20.12f}".format(sum_energy),
        #            "{:20.12f}".format(active_one_e_energy),
        #            "{:20.12f}".format(active_two_e_energy),
        #            "{:20.12f}".format(active_one_pe_energy),
        #            "{:20.12f}".format(self.E_core),
        #            "{:20.12f}".format(self.Enuc),
        #            "{:20.12f}".format(self.d_c),
        #            "{:20.12f}".format(ci_dependent_energy),
        #        )
        return (sum_energy - E0) 
    def internal_optimization_predicted_energy(self, gradient_ai, hessian_ai, Rai):
        print(np.shape(gradient_ai), np.shape(Rai))
        energy = 2.0 * np.einsum("a,a->", gradient_ai, Rai)
        energy += np.einsum("a,ab,b->", Rai, hessian_ai, Rai)
        return energy
    def internal_optimization(self, E0, eigenvecs):
        print("avg_energy", E0)
        self.U1 = np.eye(self.nmo)
        current_energy = E0
        trust_radius = 0.5
        rot_dim = self.n_occupied
        microiteration = 0
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        while(True):
            print("\n", current_energy)
            print("Internal optimization iteration", microiteration + 1)
            A1 = np.zeros((rot_dim, rot_dim))
            G1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
            self.build_intermediates_internal(eigenvecs, A1, G1, self.occupied_fock_core, self.occupied_d_cmo, self.occupied_J, self.occupied_K)
 
            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_K = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))

            gradient_tilde1 = np.zeros((rot_dim, self.n_occupied))
            hessian_tilde1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
            gradient_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a))
            hessian_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a))
            
            self.build_gradient_and_hessian(self.U, A1, G1, gradient_tilde1, hessian_tilde1, False)
            gradient_tilde_ai[:,:] = gradient_tilde1[self.n_in_a:self.n_occupied, :self.n_in_a]
            hessian_tilde_ai[:,:,:,:] = hessian_tilde1.transpose(2,0,3,1)[self.n_in_a:self.n_occupied, :self.n_in_a, self.n_in_a:self.n_occupied, :self.n_in_a]
            gradient_tilde_ai = gradient_tilde_ai.reshape(self.n_act_orb * self.n_in_a)
            hessian_tilde_ai = hessian_tilde_ai.reshape(self.n_act_orb * self.n_in_a,self.n_act_orb * self.n_in_a)
            #print(hessian_tilde_ai)
            mu1, w1 = np.linalg.eigh(hessian_tilde_ai)
            print("eigenvalue of active-inactive hessian", mu1)
                   
            #print("dfbg", gradient_tilde1)
            #print(np.shape(gradient_tilde_ai))
            gradient_norm = np.dot(gradient_tilde_ai, gradient_tilde_ai.T)
            gradient_norm = np.sqrt(gradient_norm)
            print("gradient norm", gradient_norm)
            if gradient_norm < 1e-4 or microiteration == 40: 
                print("internal rotation converged!")
                #print("qims", np.allclose(self.J, self.J_temp, rtol=1e-14,atol=1e-14))
                #print("ynss", np.allclose(self.K, self.K_temp, rtol=1e-14,atol=1e-14))
                #print("ibsp", np.allclose(self.H_spatial2, self.H1temp, rtol=1e-14,atol=1e-14))
                self.full_transformation_internal_optimization(self.U1, self.H_spatial2, self.d_cmo, self.J, self.K)
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U1) 
                #test energy       
                occupied_fock_core = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied]) 
                occupied_fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
                occupied_fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        
                


                E_core = 0.0  
                E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
                E_core += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
                active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
                active_two_e_energy = 0.5 * np.dot(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
                active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
                        self.Enuc + self.d_c + ci_dependent_energy)

                #print("heyhey1",eigenvecs) 
                print("imgf", sum_energy, E_core, active_one_e_energy, active_two_e_energy, active_one_pe_energy, ci_dependent_energy, self.Enuc, self.d_c)
                break
            #print(np.shape(hessian_tilde_ai))
            alpha = 1  
            alpha_min = 1  
            alpha_max = 1  
            dim0 = self.n_act_orb * self.n_in_a + 1
            step = np.zeros((self.n_act_orb * self.n_in_a))
            sub_microiteration = 0
            step2 = np.zeros((self.n_act_orb * self.n_in_a))
            mu2 = np.zeros(dim0)
            step_norm2, scale = self.internal_step2(gradient_tilde_ai, hessian_tilde_ai, step2, mu2, 1, dim0)
            print("scale", scale, flush = True) 
            if np.abs(scale) > 0.05:
                #find (alpha_min, alpha_max)
                while True: 
                    step_norm = self.internal_step(gradient_tilde_ai, hessian_tilde_ai, step, alpha, dim0)
                    print("step", step)
                    if step_norm > trust_radius:
                        alpha_min = alpha
                        alpha = alpha * 10
                    else:
                        alpha_max = alpha
                        break
                print("alpha range", alpha_min, alpha_max)
                #bisection search
                if alpha_max != 1:
                    while True:
                        #print(alpha_min, alpha_max)
                        alpha = 0.5 * (alpha_min + alpha_max)
                        step_norm = self.internal_step(gradient_tilde_ai, hessian_tilde_ai, step, alpha, dim0)
                        if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                            break
                        elif trust_radius - step_norm > 1e-2:
                            alpha_max = alpha
                        else:
                            alpha_min = alpha
            else:

                H_lambda = hessian_tilde_ai - mu1[0] * np.eye(dim0-1)
                step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), gradient_tilde_ai)
                H_lambda_inverse = np.linalg.pinv(H_lambda)
                print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                print("step_limit", step_limit)
                print("limit of step norm", np.linalg.norm(step_limit))
                Q = np.zeros((1, dim0-1))
                x, exitCode= minres(H_lambda, -gradient_tilde_ai)
                print("step limit from scipy solver", np.linalg.norm(x))
                print("PROBLEM!!!!!!!!!")
                step = trust_radius * step2/step_norm2
            Rai = step.reshape(self.n_act_orb, self.n_in_a)
            Rvi = np.zeros((self.n_virtual,self.n_in_a))
            Rva = np.zeros((self.n_virtual,self.n_act_orb))
            self.build_unitary_matrix(Rai, Rvi, Rva)
            self.internal_transformation(self.U_delta, occupied_h1, occupied_d_cmo, occupied_J, occupied_K)
            #print("jnti",self.E_core)
            exact_energy = self.internal_optimization_exact_energy(current_energy, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K)
            predicted_energy = self.internal_optimization_predicted_energy(gradient_tilde_ai,hessian_tilde_ai, step)
            print("vfrb",exact_energy, predicted_energy, flush = True)

            if exact_energy < 0.0:
                self.U1 = np.einsum("pq,qs->ps", self.U1, self.U_delta) 
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                self.H_diag3 = np.zeros(H_dim)
                c_H_diag_cas_spin(
                        self.occupied_fock_core, 
                        self.occupied_J3, 
                        self.H_diag3, 
                        self.N_p, 
                        self.num_alpha, 
                        self.nmo, 
                        self.n_act_a, 
                        self.n_act_orb, 
                        self.n_in_a, 
                        self.E_core, 
                        self.omega, 
                        self.Enuc, 
                        self.d_c, 
                        self.Y,
                        self.target_spin)
                #self.J[:,:,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
                #self.K[:,:,:self.n_occupied, :self.n_occupied] = occupied_K[:,:,:,:]
                #self.H_spatial2[:self.n_occupied, :self.n_occupied] = occupied_h1[:,:]
                #self.d_cmo[:self.n_occupied, :self.n_occupied] = occupied_d_cmo[:,:]
                current_energy = current_energy + exact_energy
                print("wezn",current_energy, flush=True)
                d_diag = 2.0 * np.einsum("ii->", self.occupied_d_cmo[:self.n_in_a,:self.n_in_a])
                #self.constdouble = np.zeros(6)
                #self.constdouble[0] = self.Enuc
                #if self.ignore_dse_terms:
                #    self.constdouble[1] = 0.0
                #else:  
                #    self.constdouble[1] = self.d_c
                #self.constdouble[2] = self.omega
                self.constdouble[3] = self.d_exp - d_diag
                self.constdouble[4] = 1e-4 
                self.constdouble[5] = self.E_core
                self.constint[8] = 2 
                eigenvals = np.zeros((self.davidson_roots))
                #eigenvecs = np.zeros((self.davidson_roots, H_dim))
                #eigenvecs[:,:] = 0.0
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                c_get_roots(
                    self.gkl2,
                    self.occupied_J3,
                    self.occupied_d_cmo,
                    self.H_diag3,
                    self.S_diag,
                    self.S_diag_projection,
                    eigenvals,
                    eigenvecs,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.b_array,
                    self.constint,
                    self.constdouble,
                    self.index_Hdiag,
                    True,
                    self.target_spin,
                )
                avg_energy = 0.0
                for i in range(self.davidson_roots):
                    avg_energy += self.weight[i] * eigenvals[i]
                self.avg_energy = avg_energy
                predicted_energy = self.internal_optimization_predicted_energy(gradient_tilde_ai,hessian_tilde_ai, step)
                print("internal optimization iteration",microiteration + 1, exact_energy, predicted_energy, current_energy, avg_energy, exact_energy/predicted_energy, flush = True)
                ratio = exact_energy/predicted_energy
                trust_radius = self.step_control(ratio, trust_radius)
                current_energy = avg_energy
                self.build_state_avarage_rdms(eigenvecs)
            else:
                trust_radius = 0.7 * trust_radius
                print("Reject step, restart")
            microiteration += 1
            print(trust_radius)




    def internal_optimization2(self, E0, eigenvecs):
        print("avg_energy", E0)
        current_residual = 1.0 
        
        self.U = np.eye(self.nmo)
        self.U1 = np.eye(self.nmo)
        current_energy = E0
        trust_radius = 0.5
        trust_radius_hard_case = trust_radius
        rot_dim = self.n_occupied
        microiteration = 0
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1

        A1 = np.zeros((rot_dim, rot_dim))
        G1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        self.build_intermediates_internal(eigenvecs, A1, G1, self.occupied_fock_core, self.occupied_d_cmo, self.occupied_J, self.occupied_K)
 
        occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        occupied_K = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
        occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
        occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))

        gradient_tilde1 = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a))
        hessian_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a))
        
        self.build_gradient_and_hessian(self.U, A1, G1, gradient_tilde1, hessian_tilde1, False)
        gradient_tilde_ai[:,:] = gradient_tilde1[self.n_in_a:self.n_occupied, :self.n_in_a]
        hessian_tilde_ai[:,:,:,:] = hessian_tilde1.transpose(2,0,3,1)[self.n_in_a:self.n_occupied, :self.n_in_a, self.n_in_a:self.n_occupied, :self.n_in_a]
        gradient_tilde_ai = gradient_tilde_ai.reshape(self.n_act_orb * self.n_in_a)
        hessian_tilde_ai = hessian_tilde_ai.reshape(self.n_act_orb * self.n_in_a,self.n_act_orb * self.n_in_a)



        while(True):
            hard_case = 0
            print("trust radius", trust_radius)
            print("\n", current_energy)
            print("Internal optimization iteration", microiteration + 1)
            
            #print(hessian_tilde_ai)
            mu1, w1 = np.linalg.eigh(hessian_tilde_ai)
            #print("eigenvalue of active-inactive hessian", mu1)
                   
            #print("dfbg", gradient_tilde1)
            #print(np.shape(gradient_tilde_ai))
            gradient_norm = np.dot(gradient_tilde_ai, gradient_tilde_ai.T)
            gradient_norm = np.sqrt(gradient_norm)
            print("gradient norm", gradient_norm)
            ##07/25/24: move this part to the end of microiteration for experiment
            ##if gradient_norm < 1e-4 or microiteration == 40: 
            ##    print("internal rotation converged!")
            ##    #print("qims", np.allclose(self.J, self.J_temp, rtol=1e-14,atol=1e-14))
            ##    #print("ynss", np.allclose(self.K, self.K_temp, rtol=1e-14,atol=1e-14))
            ##    #print("ibsp", np.allclose(self.H_spatial2, self.H1temp, rtol=1e-14,atol=1e-14))
            ##    start = timer()

            ##    c_full_transformation_internal_optimization(self.U1, self.J, self.K, self.H_spatial2, self.d_cmo, 
            ##            self.J, self.K, self.H_spatial2, self.d_cmo, self.index_map_ab, self.index_map_kl, self.nmo, self.n_occupied) 
            ##    #self.full_transformation_internal_optimization(self.U1, self.H_spatial2, self.d_cmo, self.J, self.K)
            ##    end = timer()
            ##    print("full internal transformation took", end - start)
            ##    self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U1) 
            ##    #test energy       
            ##    occupied_fock_core = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied]) 
            ##    occupied_fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
            ##    occupied_fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        
            ##    


            ##    E_core = 0.0  
            ##    E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            ##    E_core += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
            ##    active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
            ##    active_two_e_energy = 0.5 * np.dot(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
            ##    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            ##    ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            ##    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
            ##            self.Enuc + self.d_c + ci_dependent_energy)

            ##    #print("heyhey1",eigenvecs) 
            ##    print("imgf", sum_energy, E_core, active_one_e_energy, active_two_e_energy, active_one_pe_energy, ci_dependent_energy, self.Enuc, self.d_c)
            ##    break
            #print(np.shape(hessian_tilde_ai))
            alpha = 1  
            alpha_min = 1  
            alpha_max = 1  
            dim0 = self.n_act_orb * self.n_in_a + 1
            step = np.zeros((self.n_act_orb * self.n_in_a))
            sub_microiteration = 0
            step2 = np.zeros((self.n_act_orb * self.n_in_a))
            mu2 = np.zeros(dim0)
            step_norm2, scale = self.internal_step2(gradient_tilde_ai, hessian_tilde_ai, step2, mu2, 1, dim0)
            #x1, TR = trlib.trlib_solve(hessian_tilde_ai, gradient_tilde_ai, trust_radius)
            #print(x1)
            #print("norm of x from trlib", np.linalg.norm(x1))
            print("scale", scale, flush = True) 
            if np.abs(scale) > 1e-4:
                hard_case = 0
                #find (alpha_min, alpha_max)
                while True: 
                    step_norm = self.internal_step(gradient_tilde_ai, hessian_tilde_ai, step, alpha, dim0)
                    #print("step", step)
                    if step_norm > trust_radius:
                        alpha_min = alpha
                        alpha = alpha * 10
                    else:
                        alpha_max = alpha
                        break
                print("alpha range", alpha_min, alpha_max)
                #bisection search
                if alpha_max != 1:
                    while True:
                        #print(alpha_min, alpha_max)
                        alpha = 0.5 * (alpha_min + alpha_max)
                        step_norm = self.internal_step(gradient_tilde_ai, hessian_tilde_ai, step, alpha, dim0)
                        if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                            break
                        elif trust_radius - step_norm > 1e-2:
                            alpha_max = alpha
                        else:
                            alpha_min = alpha
            else:
                hard_case = 1
                H_lambda = hessian_tilde_ai - mu1[0] * np.eye(dim0-1)
                step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), gradient_tilde_ai)
                H_lambda_inverse = np.linalg.pinv(H_lambda)
                print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                #print("step_limit", step_limit)
                print("limit of step norm", np.linalg.norm(step_limit))
                Q = np.zeros((1, dim0-1))
                x, exitCode= minres(H_lambda, -gradient_tilde_ai)
                print("step limit from scipy solver", np.linalg.norm(x))
                print("PROBLEM!!!!!!!!!")
                #step = trust_radius * step2/step_norm2
                print("adjust step")
                if (np.linalg.norm(x) < trust_radius):
                    step = x
                    xy_square = np.dot(x, step2) * np.dot(x, step2)
                    x_square = np.dot(x, x) 
                    y_square = np.dot(step2, step2) 
                    delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                    #print(delta)
                    t1= (-2 * np.dot(x,step2) - np.sqrt(delta))/ (2*y_square)
                    t2= (-2 * np.dot(x,step2) + np.sqrt(delta))/ (2*y_square)
                    #print("x^2, xy, y^2, t", x_square, np.dot(x, step2), y_square, t1)
                    adjusted_step = step + min(t1,t2) * step2
                    print("adjusted step norm", np.linalg.norm(adjusted_step))
                    trust_radius_hard_case = np.linalg.norm(x)
                    step = adjusted_step
                else:
                    step = x/np.linalg.norm(x) * trust_radius 
            Rai = step.reshape(self.n_act_orb, self.n_in_a)
            Rvi = np.zeros((self.n_virtual,self.n_in_a))
            Rva = np.zeros((self.n_virtual,self.n_act_orb))
            self.build_unitary_matrix(Rai, Rvi, Rva)

            start = timer()
            self.internal_transformation(self.U_delta, occupied_h1, occupied_d_cmo, occupied_J, occupied_K)
            end = timer()
            print("internal transformation took", end - start)
            #print("jnti",self.E_core)
            exact_energy = self.internal_optimization_exact_energy(current_energy, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K)
            predicted_energy = self.internal_optimization_predicted_energy(gradient_tilde_ai,hessian_tilde_ai, step)
            print("internal exact energy",exact_energy,  "internal predicted energy", predicted_energy, flush = True)

            if exact_energy < 0.0:
                self.U1 = np.einsum("pq,qs->ps", self.U1, self.U_delta) 
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                self.H_diag3 = np.zeros(H_dim)
                c_H_diag_cas_spin(
                        self.occupied_fock_core, 
                        self.occupied_J3, 
                        self.H_diag3, 
                        self.N_p, 
                        self.num_alpha, 
                        self.nmo, 
                        self.n_act_a, 
                        self.n_act_orb, 
                        self.n_in_a, 
                        self.E_core, 
                        self.omega, 
                        self.Enuc, 
                        self.d_c, 
                        self.Y,
                        self.target_spin)
                #self.J[:,:,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
                #self.K[:,:,:self.n_occupied, :self.n_occupied] = occupied_K[:,:,:,:]
                #self.H_spatial2[:self.n_occupied, :self.n_occupied] = occupied_h1[:,:]
                #self.d_cmo[:self.n_occupied, :self.n_occupied] = occupied_d_cmo[:,:]
                current_energy = current_energy + exact_energy
                print("wezn",current_energy, flush=True)
                d_diag = 2.0 * np.einsum("ii->", self.occupied_d_cmo[:self.n_in_a,:self.n_in_a])
                #self.constdouble = np.zeros(6)
                #self.constdouble[0] = self.Enuc
                #if self.ignore_dse_terms:
                #    self.constdouble[1] = 0.0
                #else:  
                #    self.constdouble[1] = self.d_c
                #self.constdouble[2] = self.omega
                self.constdouble[3] = self.d_exp - d_diag
                self.constdouble[4] = 1e-5 
                self.constdouble[5] = self.E_core
                self.constint[8] = 2 
                eigenvals = np.zeros((self.davidson_roots))
                #eigenvecs = np.zeros((self.davidson_roots, H_dim))
                #eigenvecs[:,:] = 0.0
                self.occupied_J3 = self.occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)
                c_get_roots(
                    self.gkl2,
                    self.occupied_J3,
                    self.occupied_d_cmo,
                    self.H_diag3,
                    self.S_diag,
                    self.S_diag_projection,
                    eigenvals,
                    eigenvecs,
                    self.table,
                    self.table_creation,
                    self.table_annihilation,
                    self.b_array,
                    self.constint,
                    self.constdouble,
                    self.index_Hdiag,
                    True,
                    self.target_spin,
                )

                current_residual = self.constdouble[4]
                print("current CI residual", current_residual)
                avg_energy = 0.0
                for i in range(self.davidson_roots):
                    avg_energy += self.weight[i] * eigenvals[i]
                self.avg_energy = avg_energy
                predicted_energy = self.internal_optimization_predicted_energy(gradient_tilde_ai,hessian_tilde_ai, step)
                print("internal optimization iteration",microiteration + 1, exact_energy, predicted_energy, current_energy, avg_energy, exact_energy/predicted_energy, flush = True)
                ratio = exact_energy/predicted_energy
                trust_radius = self.step_control(ratio, trust_radius)
                current_energy = avg_energy
                self.build_state_avarage_rdms(eigenvecs)


                self.build_intermediates_internal(eigenvecs, A1, G1, self.occupied_fock_core, self.occupied_d_cmo, self.occupied_J, self.occupied_K)
                
                self.build_gradient_and_hessian(self.U, A1, G1, gradient_tilde1, hessian_tilde1, False)
                gradient_tilde_ai = gradient_tilde_ai.reshape(self.n_act_orb, self.n_in_a)
                hessian_tilde_ai = hessian_tilde_ai.reshape(self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a)
                gradient_tilde_ai[:,:] = gradient_tilde1[self.n_in_a:self.n_occupied, :self.n_in_a]
                hessian_tilde_ai[:,:,:,:] = hessian_tilde1.transpose(2,0,3,1)[self.n_in_a:self.n_occupied, :self.n_in_a, self.n_in_a:self.n_occupied, :self.n_in_a]
                gradient_tilde_ai = gradient_tilde_ai.reshape(self.n_act_orb * self.n_in_a)
                hessian_tilde_ai = hessian_tilde_ai.reshape(self.n_act_orb * self.n_in_a,self.n_act_orb * self.n_in_a)
  
    
            else:
                trust_radius = 0.5 * trust_radius
                #if hard_case == 0:
                #    trust_radius = 0.7 * trust_radius
                #if hard_case == 1:
                #    trust_radius = 0.7 * trust_radius_hard_case 
                print("Reject step, restart")
            gradient_norm = np.linalg.norm(gradient_tilde_ai)   
            print(gradient_norm, current_residual)
            if (gradient_norm < 1e-4 and current_residual < 1e-5) or microiteration == 20: 
                print("internal rotation converged!")
                #print("qims", np.allclose(self.J, self.J_temp, rtol=1e-14,atol=1e-14))
                #print("ynss", np.allclose(self.K, self.K_temp, rtol=1e-14,atol=1e-14))
                #print("ibsp", np.allclose(self.H_spatial2, self.H1temp, rtol=1e-14,atol=1e-14))
                start = timer()

                c_full_transformation_internal_optimization(self.U1, self.J, self.K, self.H_spatial2, self.d_cmo, 
                        self.J, self.K, self.H_spatial2, self.d_cmo, self.index_map_ab, self.index_map_kl, self.nmo, self.n_occupied) 
                #self.full_transformation_internal_optimization(self.U1, self.H_spatial2, self.d_cmo, self.J, self.K)
                end = timer()
                print("full internal transformation took", end - start)
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U1) 
                #test energy       
                occupied_fock_core = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied]) 
                occupied_fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
                occupied_fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        
                


                E_core = 0.0  
                E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
                E_core += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
                active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
                active_two_e_energy = 0.5 * np.dot(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
                active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
                sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
                        self.Enuc + self.d_c + ci_dependent_energy)
                self.avg_energy = sum_energy
                #print("heyhey1",eigenvecs) 
                print("imgf", sum_energy, E_core, active_one_e_energy, active_two_e_energy, active_one_pe_energy, ci_dependent_energy, self.Enuc, self.d_c)
                break

            microiteration += 1
            print(trust_radius)
    
    
    
    
   





    def printA(self, a):
        for row in a:
            for col in row:
                print("{:13.10f}".format(col), end=" ")
            print("")



    def internal_step(self, gradient_tilde_ai, hessian_tilde_ai, step, alpha, dim0):
        gradient_tilde_ai = alpha * gradient_tilde_ai
        augmented_hessian = np.zeros((dim0, dim0))

        augmented_hessian[0,1:] = gradient_tilde_ai
        augmented_hessian[1:,0] = gradient_tilde_ai.T
        augmented_hessian[1:,1:] = hessian_tilde_ai
        #print("hessian")
        #self.printA(hessian_tilde_ai)
        #print("augmented_hessian")
        #self.printA(augmented_hessian)

        #print(augmented_hessian)
        mu, w = np.linalg.eigh(augmented_hessian)
        #print("w",w)
        #print("first eigenvalues of the augmented hessian", mu[0], flush=True)
        scale = w[0][0]
        print(scale, flush=True)
        w[:,0] = w[:,0]/scale
        #print(w)
        step0 = w[1:,0]
        #print(step0)
        step[:] = step0/alpha
        #h_inverse = np.linalg.inv(hessian_tilde_ai)
        #step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai) 
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        #print("step norm", step_norm, flush=True)
        return step_norm
    def internal_step2(self, gradient_tilde_ai, hessian_tilde_ai, step, mu, alpha, dim0):
        gradient_tilde_ai = alpha * gradient_tilde_ai
        augmented_hessian = np.zeros((dim0, dim0))

        augmented_hessian[0,1:] = gradient_tilde_ai
        augmented_hessian[1:,0] = gradient_tilde_ai.T
        augmented_hessian[1:,1:] = hessian_tilde_ai
        #print(augmented_hessian)
        mu[:], w = np.linalg.eigh(augmented_hessian)
        #print("w",w)
        #print("eigenvalues of the augmented hessian", mu, flush=True)
        scale = w[0][0]
        print(scale, flush=True)
        #w[:,0] = w[:,0]/scale
        #print(w)
        step0 = w[1:,0]
        #print(step0)
        step[:] = step0/alpha
        #h_inverse = np.linalg.inv(hessian_tilde_ai)
        #step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai) 
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        #print("step norm", step_norm, flush=True)
        return step_norm, scale, 
 

    def step_control(self, ratio, trust_radius):
        if ratio > 0 and ratio < 0.25: trust_radius = 0.7 * trust_radius
        if ratio > 0.75: 
            trust_radius = 1.2 * trust_radius
            if trust_radius > 0.75: trust_radius = 0.75
        return trust_radius
 
    def build_state_avarage_rdms(self, eigenvecs):
        self.D_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.Dpe_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.D_tuvw_avg = np.zeros((self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb))
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            c_build_active_rdm(eigenvecs,
                    self.D_tu_avg,
                    self.D_tuvw_avg,
                    self.table,
                    self.n_act_a,
                    self.n_act_orb,
                    np1,
                    i,
                    i,
                    self.weight[i]
            )
            c_build_active_photon_electron_one_rdm(eigenvecs,
                    self.Dpe_tu_avg,
                    self.table,
                    self.n_act_a,
                    self.n_act_orb,
                    np1,
                    i,
                    i,
                    self.weight[i]
            )
        

        ###symmetrize 2-rdm
        for t in range(self.n_act_orb):
            for u in range(t,self.n_act_orb):
                tu = t * self.n_act_orb + u
                ut = u * self.n_act_orb + t
                for vw in range(self.n_act_orb * self.n_act_orb):
                    dum = (self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] + 
                     self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw])
                    #dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu] 
                    self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                    self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                    

                    #dum = (self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw] + 
                    # self.D_tuvw_avg2[ut * self.n_act_orb * self.n_act_orb + vw])
                    ##dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu] 
                    #self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                    #self.D_tuvw_avg2[ut * self.n_act_orb * self.n_act_orb + vw] = dum/2.0

                    #self.D_tu_avg[tu] = dum2/2.0
                    #self.D_tu_avg[ut] = dum2/2.0
    def build_sigma_reduced(self, U, A, G, step):            
        self.reduced_hessian2 = np.zeros((self.index_map_size, self.index_map_size))
        # build the component of tranformed hessian from gradient
        B = np.zeros((self.nmo, self.nmo))
        self.T = U - np.eye(self.nmo)
        B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], self.T[:,:self.n_occupied])
        #B2 = np.zeros((self.nmo, self.nmo))
        #for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        B2[r][k] = A[r][k]
        #        for s in range(self.nmo):
        #            for l in range(self.n_occupied):
        #                B2[r][k] += G[k][l][r][s] * self.T[s][l]
        #print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))
        
        A_tilde = np.zeros((self.nmo, self.nmo))
        #A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
        #A2_tilde = np.zeros((self.nmo, self.nmo))
        A_tilde = np.einsum("rs,sk->rk",U.T, B)
        #A2_tilde = np.zeros((self.nmo, self.nmo))
        #for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        for s in range(self.nmo):
        #            A2_tilde[r][k] += U[s][r] * B[s][k]
        #print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))
        
           
        A3_tilde =  (A_tilde + A_tilde.T)
        
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        
        hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])


        self.R_total = np.zeros((self.nmo, self.n_occupied))
        self.sigma_total = np.zeros((self.nmo, self.n_occupied))
        for i in range(self.index_map_size):
            s = self.index_map[i][0]   
            l = self.index_map[i][1] 
            self.R_total[s][l] = step[i]
        temp1 = np.zeros((self.nmo, self.n_occupied))
        temp2 = np.zeros((self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,sl->ql", U, self.R_total)
        temp2 = np.einsum("klpq,ql->pk", G, temp1)
        self.sigma_total = np.einsum("pk,pr->rk", temp2, U)
        self.sigma_total[:self.n_occupied,:] -= np.einsum("pr,pk->rk", temp2, U[:,:self.n_occupied])
        
        temp1 = np.einsum("ql,sl->qs", U[:,:self.n_occupied], self.R_total[:self.n_occupied,:])
        temp2 = np.einsum("kspq,qs->pk", G, temp1)
        self.sigma_total -= np.einsum("pk,pr->rk", temp2, U)
        self.sigma_total[:self.n_occupied,:] += np.einsum("pr,pk->rk", temp2, U[:,:self.n_occupied])
        self.sigma_total += np.einsum("klrs,sl->rk", hessian_tilde, self.R_total)
   

    def build_sigma_reduced2(self, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer):
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw",num_states)
        #print(self.index_map)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                R_total[i][r][k] = R_reduced[i][j+pointer]
        #print("R total") 
        #for i in range(num_states):
        #    for r in range(self.nmo):
        #        for k in range(self.n_occupied):
        #            print("%20.12lf" %(R_total[i,r,k]))
        #print("\n")

        ###B = np.zeros((self.nmo, self.nmo))
        ###T = U - np.eye(self.nmo)
        ###B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
        ####B2 = np.zeros((self.nmo, self.nmo))
        ####for r in range(self.nmo):
        ####    for k in range(self.n_occupied):
        ####        B2[r][k] = A[r][k]
        ####        for s in range(self.nmo):
        ####            for l in range(self.n_occupied):
        ####                B2[r][k] += G[k][l][r][s] * self.T[s][l]
        ####print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))
        ###
        ###A_tilde = np.zeros((self.nmo, self.nmo))
        ####A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
        ####A2_tilde = np.zeros((self.nmo, self.nmo))
        ###A_tilde = np.einsum("rs,sk->rk",U.T, B)
        #A2_tilde = np.zeros((self.nmo, self.nmo))
        #for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        for s in range(self.nmo):
        #            A2_tilde[r][k] += U[s][r] * B[s][k]
        #print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))
        
           
        A3_tilde =  (A_tilde + A_tilde.T)
        
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        
        hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])


        sigma_total = np.zeros((num_states, self.nmo, self.n_occupied))
        
        temp1 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp2 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,isl->iql", U, R_total)
        temp2 = np.einsum("klpq,iql->ipk", G, temp1)
        sigma_total = np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:,:self.n_occupied,:] -= np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        
        temp1 = np.einsum("ql,isl->iqs", U[:,:self.n_occupied], R_total[:,:self.n_occupied,:])
        temp2 = np.einsum("kspq,iqs->ipk", G, temp1)
        sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:,:self.n_occupied,:] += np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        sigma_total += np.einsum("klrs,isl->irk", hessian_tilde, R_total)
   
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                sigma_reduced[i][j+pointer] = sigma_total[i][r][k] 
         
    def build_sigma_reduced3(self, U, hessian_tilde, G, R_reduced, sigma_reduced, num_states, pointer):
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw",num_states) 
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                R_total[i][r][k] = R_reduced[i][j+pointer]
                
       
       
        sigma_total = np.zeros((num_states, self.nmo, self.n_occupied))
        
        temp1 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp2 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,isl->iql", U, R_total, optimize ="optimal")
        temp1 -= np.einsum("ql,isl->iqs", U[:,:self.n_occupied], R_total[:,:self.n_occupied,:], optimize ="optimal")
        temp2 = np.einsum("klpq,iql->ipk", G, temp1, optimize ="optimal")
        sigma_total = np.einsum("ipk,pr->irk", temp2, U, optimize ="optimal")
        sigma_total[:,:self.n_occupied,:] -= np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied], optimize ="optimal")
        
        #temp1 = np.einsum("ql,isl->iqs", U[:,:self.n_occupied], R_total[:,:self.n_occupied,:])
        #temp2 = np.einsum("kspq,iqs->ipk", G, temp1, optimize = "optimal")
        #sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
        #sigma_total[:,:self.n_occupied,:] += np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        sigma_total += np.einsum("klrs,isl->irk", hessian_tilde, R_total, optimize = "optimal")
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                sigma_reduced[i][j+pointer] = sigma_total[i][r][k] 
     
     
    def orbital_sigma(self, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer):
        nmo = self.nmo
        index_map = self.index_map
        index_map_size = self.index_map_size
        n_occupied = self.n_occupied
        return self.build_sigma_reduced4(U, A_tilde, index_map, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied)
    
    @staticmethod    
    #@nb.njit("""void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,:,:,::1], float64[:,::1], float64[:,::1],
    #        int64, int64, int64, int64, int64)""", fastmath = True, parallel = True) 
    #def build_sigma_reduced6(U, A_tilde, index_map, G1, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
    @nb.jit("""void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1],
            int64, int64, int64, int64, int64)""", nopython = True, cache = True, fastmath = True, parallel = True) 
    def build_sigma_reduced4(U, A_tilde, index_map, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
      
        assert U.shape == (nmo, nmo)
        assert G.shape == (nmo * n_occupied, nmo * n_occupied)
        assert A_tilde.shape == (nmo, nmo)
        R_total = np.zeros((num_states, nmo, n_occupied))
        print("oivdpw",num_states) 
        for j in nb.prange(index_map_size):
            r = index_map[j][0] 
            k = index_map[j][1] 
            for i in range(num_states):
                R_total[i][r][k] = R_reduced[i][j+pointer]
                
       
       
        sigma_total = np.zeros((num_states, nmo, n_occupied))
        temp1 = np.zeros((num_states, nmo, n_occupied))
        temp2 = np.zeros((num_states, nmo, n_occupied))


        R1 = R_total.transpose(1,0,2)
        R1 = np.ascontiguousarray(R1) 
        R1 = np.reshape(R1,(nmo, num_states * n_occupied)) 
        temp2 = np.dot(U, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)
        R1 = np.ascontiguousarray(R_total[:,:n_occupied,:].transpose(2,0,1)) 
        #R_total1 = np.ascontiguousarray(R_total1) 
        R1 = np.reshape(R1,(n_occupied, num_states * n_occupied))       
        U1 = np.ascontiguousarray(U[:,:n_occupied])
        temp2 = np.dot(U1, R1)
        temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)

        ##sigma_total1 = np.zeros((num_states, nmo, n_occupied))
        ##temp11 = np.einsum("qs,isl->iql", U, R_total)
        #for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for l in range(n_occupied):
        #            #a = 0.0
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += U[q,s] * R_total[i,s,l]
        #                #temp1[i,q,l] += U[q,s] * R_total[i,s,l]
        #            temp1[i,q,l] = a 
        #

        #for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for s in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a -= U[q,l] * R_total[i,s,l]
        #                #temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
        #            temp1[i,q,s] += a 
        R1 = np.ascontiguousarray(temp1) 
        R1 = np.reshape(R1, (num_states, nmo * n_occupied))
        #R2 = np.zeros((num_states, nmo * n_occupied))
        ##for i in nb.prange(num_states):
        ##    for r in range(nmo):
        ##        for k in range(n_occupied):
        ##            a = R1[i][r*n_occupied+k] - temp1[i,r,k]
        ##            if np.abs(a) > 1e-14: print("LARGE error")

        print(np.shape(R1))
        temp1 = np.dot(R1, G)
        #temp2 = temp1.reshape(num_states, nmo, n_occupied)
        #for r in nb.prange(nmo):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for p in range(nmo):
        #                a += U[p,r] * temp2[i,p,k]
        #                #sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
        #            temp1[i,r,k] = a 
        #            sigma_total[i,r,k] = a 
        #

        #for r in nb.prange(n_occupied):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] -= temp1[i,k,r] 
        

        #######for i in nb.prange(num_states):
        #######    R3 = np.dot(R1[i,:], G)
        #######    R2[i,:] = R3 

        temp2 = temp1.reshape(num_states, nmo, n_occupied)
        temp2 = temp2.transpose(0,2,1) 
        temp2 = np.ascontiguousarray(temp2)
        temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
        temp1 = np.dot(temp2, U)
        temp2 = temp1.reshape(num_states, n_occupied, nmo)
        sigma_total[:,:,:] = temp2.transpose(0,2,1)[:,:,:]  
        sigma_total[:,:n_occupied,:] -= temp2[:,:,:n_occupied]
                
        A3_tilde =  (A_tilde + A_tilde.T)
        #temp1 = np.zeros((num_states, nmo, n_occupied))
        #temp2 = np.zeros((num_states, nmo, n_occupied))

        ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        ##hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += A3_tilde[r,s] * R_total[i,s,k]
        #            temp1[i,r,k] = a 
        #            sigma_total[i,r,k] -= 0.5 * a 
        #
        #for i in nb.prange(num_states):
        #    for r in range(n_occupied):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r] 
        #
        ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[k,l] * R_total[i,r,l]
        #            sigma_total[i,r,k] -= 0.5 * a

        ##hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[r,l] * R_total[i,k,l]
        #            sigma_total[i,r,k] += 0.5 * a
        R1 = R_total.transpose(1,0,2) 
        R1 = np.ascontiguousarray(R1) 
        R1 = np.reshape(R1,(nmo, num_states * n_occupied))
        temp2 = np.dot(A3_tilde, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)
        sigma_total[:,:,:] -= 0.5 * temp1[:,:,:]
        sigma_total[:,:n_occupied,:] += 0.5 * temp1.transpose(0,2,1)[:,:,:n_occupied]
        A3 = A3_tilde[:n_occupied, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total)
        R1 = np.reshape(R1,(num_states * nmo, n_occupied))
        temp2 = np.dot(R1, A3)
        sigma_total[:,:,:] -= 0.5 * temp2.reshape(num_states, nmo, n_occupied)[:,:,:]
        A3 = A3_tilde[:, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total[:,:n_occupied,:])
        R1 = np.reshape(R1,(num_states * n_occupied, n_occupied))
        temp2 = np.dot(R1, A3)
        temp1 = temp2.reshape(num_states, n_occupied, nmo)
        sigma_total[:,:,:] += 0.5 * temp1.transpose(0,2,1)[:,:,:]
        
   
        for j in nb.prange(index_map_size):
            r = index_map[j][0] 
            k = index_map[j][1] 
            for i in range(num_states):
                sigma_reduced[i][j+pointer] = sigma_total[i][r][k] 
     




    def microiteration_ci_integrals_transform(self, U, eigenvecs, d_cmo, active_fock_core, active_twoeint):
        #print("test U2")
        #print(U)


        #print("E_core before second order transformation", self.E_core)
        T = U - np.eye(self.nmo)
        E_core = self.E_core
        E_core += 4.0 * np.einsum("ir,ri->", self.fock_core[:self.n_in_a,:], T[:,:self.n_in_a])
        temp1 = np.zeros((self.n_in_a, self.n_in_a, self.nmo, self.nmo))
        temp1 = np.einsum("rs,ij->ijrs", self.fock_core, np.eye(self.n_in_a))
        temp1 += self.L[:self.n_in_a,:,:,:] 
        temp2 = np.zeros((self.nmo, self.n_in_a))
        temp2 = np.einsum("ijrs,sj->ri", temp1, T[:,:self.n_in_a])
        E_core += 2.0 * np.einsum("ri,ri->", T[:,:self.n_in_a], temp2)
        #print("E_core from ci_integral", E_core)


        #E_core3 = 0.0
        #for i in range(self.n_in_a):
        #    E_core3 += self.H_spatial2[i][i]
        #    E_core3 -= self.fock_core[i][i]
        #for i in range(self.n_in_a):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                E_core3 += 2.0 * U[r][i] * self.fock_core[r][s] *  U[s][i] 
        #for i in range(self.n_in_a):
        #    for j in range(self.n_in_a):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                E_core3 += 2.0 * T[r][i] * self.L[i][j][r][s] * T[s][j] 
        #
        ##print("test",E_core, E_core3)



        #E_core2 = self.E_core 
        #for i in range(self.n_in_a):
        #    for r in range(self.nmo):
        #        E_core2 += 4.0 * self.fock_core[i][r] * T[r][i] 
        #for i in range(self.n_in_a):
        #    for j in range(self.n_in_a):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                E_core2 += 2.0 * T[r][i] * (self.fock_core[r][s] *(i==j) + self.L[i][j][r][s]) * T[s][j] 
        ##print(E_core, E_core2)

     
        self.E_core2 = E_core




        active_fock_core[:,:] = 0.0 
        #active_fock_core[:,:] = copy.deepcopy(self.active_fock_core)
        temp3 = np.zeros((self.nmo, self.n_act_orb))
        temp3 = np.einsum("rs,su->ru", self.fock_core, U[:,self.n_in_a:self.n_occupied])
        active_fock_core[:,:] = np.einsum("rt,ru->tu", U[:,self.n_in_a:self.n_occupied], temp3)
        
        temp4 = np.zeros((self.n_act_orb, self.n_act_orb, self.nmo, self.n_in_a))
        temp4 = np.einsum("turs,si->turi", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:,:], U[:,:self.n_in_a])
        active_fock_core[:,:] += 2.0 * np.einsum("ri,turi->tu", U[:,:self.n_in_a], temp4)
        active_fock_core[:,:] -= 2.0 * np.einsum("tuii->tu", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:self.n_in_a,:self.n_in_a])
        temp4 = np.einsum("turs,si->turi", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:,:], U[:,:self.n_in_a])
        active_fock_core[:,:] -= np.einsum("ri,turi->tu", U[:,:self.n_in_a], temp4)
        active_fock_core[:,:] += np.einsum("tuii->tu", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:self.n_in_a,:self.n_in_a])

        temp3 = np.einsum("uirs,si->ru", self.L[self.n_in_a:self.n_occupied,:,:,:], T[:,:self.n_in_a])
        temp5 = np.zeros((self.n_act_orb, self.n_act_orb))
        temp5 = np.einsum("rt,ru->tu", T[:,self.n_in_a:self.n_occupied], temp3)
        active_fock_core[:,:] += temp5 + temp5.T

        active_twoeint[:,:,:,:] = -1.0 * self.active_twoeint
        temp6 = np.zeros((self.n_act_orb, self.n_act_orb, self.nmo, self.n_act_orb))
        temp7 = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        temp6 = np.einsum("vwrs,su->vwru", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:,:], U[:,self.n_in_a:self.n_occupied])
        temp7 = np.einsum("vwru,rt->vwtu", temp6, U[:,self.n_in_a:self.n_occupied])
        #print(np.shape(active_twoeint),np.shape(temp7) )
      
        active_twoeint[:,:,:,:] += temp7[:,:,:,:]  
        active_twoeint[:,:,:,:] += temp7[:,:,:,:].transpose(2,3,0,1)  
        temp6 = np.einsum("tvrs,sw->tvrw", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:,:], T[:,self.n_in_a:self.n_occupied])
        temp7 = np.einsum("tvrw,ru->tvuw", temp6, T[:,self.n_in_a:self.n_occupied])
        #active_twoeint[:,:,:,:] = temp7[:,:,:,:]  
        #active_twoeint[:,:,:,:] += temp7.transpose(2,1,0,3)  
        #active_twoeint[:,:,:,:] += temp7.transpose(0,3,2,1)  
        #active_twoeint[:,:,:,:] += temp7.transpose(2,3,0,1)  
        temp7 = np.einsum("tvrw,ru->tuvw", temp6, T[:,self.n_in_a:self.n_occupied])
        active_twoeint[:,:,:,:] += temp7[:,:,:,:]  
        active_twoeint[:,:,:,:] += temp7.transpose(1,0,2,3)  
        active_twoeint[:,:,:,:] += temp7.transpose(0,1,3,2)  
        active_twoeint[:,:,:,:] += temp7.transpose(1,0,3,2)  

        temp8 = np.einsum("pq,qs->ps", self.d_cmo, U)
        d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, U)

        ##build integrals from loop 
        #active_fock2 = np.zeros((self.n_act_orb, self.n_act_orb))
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                active_fock2[t][u] += U[r][t+self.n_in_a] * self.fock_core[r][s] * U[s][u+self.n_in_a]
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for i in range(self.n_in_a):
        #            active_fock2[t][u] -= 2.0 * self.J[t+self.n_in_a][u+self.n_in_a][i][i] 
        #            for r in range(self.nmo):
        #                for s in range(self.nmo):
        #                    active_fock2[t][u] += 2.0 * U[r][i] * self.J[t+self.n_in_a][u+self.n_in_a][r][s] * U[s][i]
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for i in range(self.n_in_a):
        #            active_fock2[t][u] += self.K[t+self.n_in_a][u+self.n_in_a][i][i] 
        #            for r in range(self.nmo):
        #                for s in range(self.nmo):
        #                    active_fock2[t][u] -= U[r][i] * self.K[t+self.n_in_a][u+self.n_in_a][r][s] * U[s][i]
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                for i in range(self.n_in_a):
        #                    active_fock2[t][u] += T[r][t+self.n_in_a] * self.L[u+self.n_in_a][i][r][s] * T[s][i]
        #                    active_fock2[t][u] += T[r][u+self.n_in_a] * self.L[t+self.n_in_a][i][r][s] * T[s][i]



        #print("hnbo", np.allclose(active_fock2,active_fock_core, rtol=1e-14,atol=1e-14))
        #active_twoeint2 = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for v in range(self.n_act_orb):
        #            for w in range(self.n_act_orb):
        #                active_twoeint2[t][u][v][w] = -self.J[t+self.n_in_a][u+self.n_in_a][v+self.n_in_a][w+self.n_in_a]
        #                for r in range(self.nmo):
        #                    for s in range(self.nmo):
        #                        active_twoeint2[t][u][v][w] += U[r][t+self.n_in_a] * self.J[v+self.n_in_a][w+self.n_in_a][r][s] * U[s][u+self.n_in_a]
        #                        active_twoeint2[t][u][v][w] += U[r][v+self.n_in_a] * self.J[t+self.n_in_a][u+self.n_in_a][r][s] * U[s][w+self.n_in_a]
        #                        active_twoeint2[t][u][v][w] += T[r][u+self.n_in_a] * self.K[t+self.n_in_a][v+self.n_in_a][r][s] * T[s][w+self.n_in_a]
        #                        active_twoeint2[t][u][v][w] += T[r][t+self.n_in_a] * self.K[u+self.n_in_a][v+self.n_in_a][r][s] * T[s][w+self.n_in_a]
        #                        active_twoeint2[t][u][v][w] += T[r][u+self.n_in_a] * self.K[t+self.n_in_a][w+self.n_in_a][r][s] * T[s][v+self.n_in_a]
        #                        active_twoeint2[t][u][v][w] += T[r][t+self.n_in_a] * self.K[u+self.n_in_a][w+self.n_in_a][r][s] * T[s][v+self.n_in_a]
        #print("gnzy", np.allclose(active_twoeint,active_twoeint2, rtol=1e-14,atol=1e-14))
        
        #####active_twoeint3 = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        #####for t in range(self.n_act_orb):
        #####    for u in range(self.n_act_orb):
        #####        for v in range(self.n_act_orb):
        #####            for w in range(self.n_act_orb):
        #####                #active_twoeint3[t][u][v][w] = self.J[t+self.n_in_a][u+self.n_in_a][v+self.n_in_a][w+self.n_in_a]
        #####                for r in range(self.nmo):
        #####                    active_twoeint3[t][u][v][w] += 4.0 * T[r][u+self.n_in_a] * self.J[v+self.n_in_a][w+self.n_in_a][r][t+self.n_in_a] 
        #####                    for s in range(self.nmo):
        #####                        active_twoeint3[t][u][v][w] += 2.0 * T[r][t+self.n_in_a] * self.J[v+self.n_in_a][w+self.n_in_a][r][s] * T[s][u+self.n_in_a]
        #####                        active_twoeint3[t][u][v][w] += 4.0 * T[r][t+self.n_in_a] * self.K[u+self.n_in_a][w+self.n_in_a][r][s] * T[s][v+self.n_in_a]
        #####active_two_e_energy2 = 0.5 * np.dot(active_twoeint3.flatten(), self.D_tuvw_avg)
        #####print("active two energy with different integrals",
        #####    "{:20.12f}".format(active_two_e_energy2)
        #####    )


        #####rot_dim = self.nmo
        #####self.A10 = np.zeros((rot_dim, rot_dim))
        #####self.G10 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))

        #####self.A10[:,self.n_in_a:self.n_occupied] += np.einsum("vwrt,tuvw->ru", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,self.n_in_a:self.n_occupied], 
        #####   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))

        #####self.G10[self.n_in_a:,self.n_in_a:,:,:] += np.einsum("vwrs,tuvw->turs", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #####   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        #####self.G10[self.n_in_a:,self.n_in_a:,:,:] += 2.0 * np.einsum("vwrs,tvuw->turs", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #####   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        #####

        #####B20 = np.zeros((self.nmo, self.n_occupied))
        #####B20 = 2.0 * self.A10[:,:self.n_occupied] + np.einsum("klrs,sl->rk", self.G10, T[:,:self.n_occupied])
        #####E = np.einsum("rk, rk->",T[:,:self.n_occupied], B20)
        #####print("heyheeeqq", E)
        #print("mjhy", np.allclose(active_twoeint,self.active_twoeint, rtol=1e-14,atol=1e-14))
        #for t in range(self.n_act_orb):
        #    for u in range(self.n_act_orb):
        #        for v in range(self.n_act_orb):
        #            for w in range(self.n_act_orb):
        #                print("{:20.12f}".format(active_twoeint2[t][u][v][w]), "{:20.12f}".format(active_twoeint[t][u][v][w])) 

        ##test energy from loop 
        #active_one_e_energy = np.dot(active_fock2.flatten(), self.D_tu_avg)

        #active_two_e_energy = 0.5 * np.dot(active_twoeint2.flatten(), self.D_tuvw_avg)
        #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
        #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
        #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
        #        self.Enuc + self.d_c + ci_dependent_energy)
        #print("test sum energy from explicit loop",
        #    "{:20.12f}".format(sum_energy),
        #    "{:20.12f}".format(active_one_e_energy),
        #    "{:20.12f}".format(active_two_e_energy),
        #    "{:20.12f}".format(E_core),
        #    "{:20.12f}".format(self.Enuc),
        #)

    def microiteration_exact_energy(self, U, A, G):
        #print("test U")
        #print(U)

        T = U - np.eye(self.nmo)
        T_rk = T[:,:self.n_occupied].reshape(self.nmo * self.n_occupied)
        A_rk = A[:,:self.n_occupied].reshape(self.nmo * self.n_occupied)
        #B = np.zeros((self.nmo, self.n_occupied))
        #B = 2.0 * A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G, T[:,:self.n_occupied])
        B = 2.0 * A_rk + np.dot(T_rk, G)
        #E = np.einsum("rk, rk->",T[:,:self.n_occupied], B)
        E = np.dot(T_rk, B)
        #print("ccvcvc", B) 
        #c= np.einsum("rk, rk->",T[:,:self.n_occupied], B)
        c= np.dot(T_rk, B)
        #d=np.einsum("rk, rk->",T[:,:self.n_in_a], B[:,:self.n_in_a])
        #E1 = E0 + np.einsum("rk, rk->",T[:,:self.n_in_a], B[:,:self.n_in_a])
        #print("uuunn", c)
        #B3 = np.zeros((self.nmo, self.n_in_a))
        #B3 = 2.0 * self.fock_core[:,:self.n_in_a]
        #temp = np.zeros((self.nmo, self.nmo, self.n_in_a, self.n_in_a))
        #temp = np.einsum("rs, ij->ijrs",self.fock_core, np.eye(self.n_in_a))
        #temp += self.L[:self.n_in_a,:,:,:]
        #B3 += np.einsum("ijrs,sj->ri", temp, T[:,:self.n_in_a]) 
        #E10 = 2.0 * np.einsum("ri,ri->", T[:,:self.n_in_a], B3)
        #print("rmmngghg", E0, E0+E10)
        return E

    def rdm_exact_energy(self, J, K, h1, d_cmo, eigenvecs):

        # corresponds to Eq. (2) in Nam's notes
        active_twoeint = J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
        fock_core = copy.deepcopy(h1) 
        fock_core += 2.0 * np.einsum("jjrs->rs", J[:self.n_in_a,:self.n_in_a,:,:]) 
        fock_core -= np.einsum("jjrs->rs", K[:self.n_in_a,:self.n_in_a,:,:]) 
        
        # corresponds to Eq. (3) in Nam's notes
        E_core = 0.0  
        E_core += np.einsum("jj->", h1[:self.n_in_a,:self.n_in_a]) 
        E_core += np.einsum("jj->", fock_core[:self.n_in_a,:self.n_in_a]) 


        #print(eigenvecs)
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        active_fock_core[:,:] = fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
        active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg) # (F')_tu D^{IJ}_tu
        active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg) # 1/2 (tu|vw)' D_tu;vw
        # -\sqrt{w/2} d_tu D_tu
        active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
        ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
        sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
                self.Enuc + self.d_c + ci_dependent_energy)
        print("sum energy",
            "{:20.12f}".format(sum_energy),
            "{:20.12f}".format(active_one_e_energy),
            "{:20.12f}".format(active_two_e_energy),
            "{:20.12f}".format(E_core),
            "{:20.12f}".format(active_one_pe_energy),
            "{:20.12f}".format(self.Enuc),
        )
        return sum_energy
   

    def energy_function(self, step):
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual,self.n_in_a))
        Rva = np.zeros((self.n_virtual,self.n_act_orb))
        for i in range(self.index_map_size):
            s = self.index_map[i][0] 
            l = self.index_map[i][1]
            if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                Rai[s-self.n_in_a][l] = step[i]
            elif s >= self.n_occupied and l < self.n_in_a:
                Rvi[s-self.n_occupied][l] = step[i]
            else:
                Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

        self.build_unitary_matrix(Rai, Rvi, Rva)
        U_temp = self.U_delta 
        J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        K_temp = np.ascontiguousarray(K_temp)
        c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
        #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

        temp8 = np.zeros((self.nmo, self.nmo))
        temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
        h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
        d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)


        sum_energy = self.rdm_exact_energy(J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs)

        
        return sum_energy

    def energy_grad_finite_difference_element(self, step):
        """
        A function to compute the elements of the orbital gradient using centered finite differences
        """
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual,self.n_in_a))
        Rva = np.zeros((self.n_virtual,self.n_act_orb))

        rot_dim = self.nmo
        A_num = np.zeros((rot_dim, rot_dim))

        # define step size for orbitals - is this reasonable???
        _h = 0.001

        # perform outter-loop to select gradient element to compute
        for j in range(self.index_map):
            gradS = self.index_map[j][0]
            gradL = self.index_map[j][1]
            
            # perform inner-loop to build R with forward displacement
            for i in range(self.index_map_size):
                s = self.index_map[i][0] 
                l = self.index_map[i][1]

                # if this is the element to displace, do forward displacement
                if i==j:
                    step_val = step[i] + _h
                # otherwise, don't displace
                else:
                    step_val = step[i]

                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s-self.n_in_a][l] = step_val
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s-self.n_occupied][l] = step_val
                else:
                    Rva[s-self.n_occupied][l-self.n_in_a] = step_val

            # build unitary matrix with one element displaced - the resulting unitary matrix will be antisymmetrized 
            self.build_unitary_matrix(Rai, Rvi, Rva)
            U_temp = self.U_delta 
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)

            # transform with displaced 
            c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
            #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

            # compute energy from forward displaced 
            forward_energy = self.rdm_exact_energy(J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs)

            # perform inner-loop to build R with forward displacement
            for i in range(self.index_map_size):
                s = self.index_map[i][0] 
                l = self.index_map[i][1]

                # if this is the element to displace, do forward displacement
                if i==j:
                    step_val = step[i] - _h
                # otherwise, don't displace
                else:
                    step_val = step[i]

                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s-self.n_in_a][l] = step_val
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s-self.n_occupied][l] = step_val
                else:
                    Rva[s-self.n_occupied][l-self.n_in_a] = step_val

            # build unitary matrix with one element displaced - the resulting unitary matrix will be antisymmetrized 
            self.build_unitary_matrix(Rai, Rvi, Rva)
            U_temp = self.U_delta 
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)

            # transform with displaced 
            c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
            #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)


            backward_energy = self.rdm_exact_energy(J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs)

            A_num[gradS, gradL] = (forward_energy - backward_energy) / (4 * _h)
        
        return A_num

    
   
    def energy_grad(self, step):
     
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual,self.n_in_a))
        Rva = np.zeros((self.n_virtual,self.n_act_orb))
        for i in range(self.index_map_size):
            s = self.index_map[i][0] 
            l = self.index_map[i][1]
            if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                Rai[s-self.n_in_a][l] = step[i]
            elif s >= self.n_occupied and l < self.n_in_a:
                Rvi[s-self.n_occupied][l] = step[i]
            else:
                Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

        self.build_unitary_matrix(Rai, Rvi, Rva)
        U_temp = self.U_delta 
        J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        K_temp = np.ascontiguousarray(K_temp)
        c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
        #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

        temp8 = np.zeros((self.nmo, self.nmo))
        temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
        h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
        d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        
        rot_dim = self.nmo       
        A = np.zeros((rot_dim, rot_dim))
        fock_core = copy.deepcopy(h1_temp) 
        fock_core += 2.0 * np.einsum("jjrs->rs", J_temp[:self.n_in_a,:self.n_in_a,:,:], optimize = "optimal") 
        fock_core -= np.einsum("jjrs->rs", K_temp[:self.n_in_a,:self.n_in_a,:,:], optimize = "optimal") 
        D_tu_avg = self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)) 
        Dpe_tu_avg = self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb))
        D_tuvw_avg = self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb))
        E_core = 0.0  
        E_core += np.einsum("jj->", h1_temp[:self.n_in_a,:self.n_in_a]) 
        E_core += np.einsum("jj->", fock_core[:self.n_in_a,:self.n_in_a]) 
        
        active_fock_core = copy.deepcopy(fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied]) 
        active_twoeint = copy.deepcopy(J_temp[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied])
        L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        fock_general = np.zeros((rot_dim, rot_dim))
        #print (np.shape(self.L))
        #start = timer()
        L = 4.0 * K_temp[:,:self.n_in_a,:rot_dim,:rot_dim] - K_temp.transpose(0,1,3,2)[:,:self.n_in_a,:rot_dim,:rot_dim] - J_temp[:,:self.n_in_a,:rot_dim,:rot_dim]
        #end   = timer()
        #print("build intermediate step 2", end - start)
        ###self.L2 = np.zeros((self.n_occupied, self.n_in_a, self.nmo, self.nmo))
        ###for k in range(self.n_occupied):
        ###    for j in range(self.n_in_a):
        ###        for r in range(self.nmo):
        ###            for s in range(self.nmo):
        ###                self.L2[k][j][r][s] = (4.0 * self.K[k][j][r][s] -
        ###                    self.K[k][j][s][r] - self.J[k][j][r][s])
        ###print((self.L==self.L2).all())
        
        #self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
        #        self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        #self.fock_general -= 0.5 * np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)), 
        #        self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        temp1 = J_temp[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim] -0.5 * K_temp[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim]
        #start = timer()
        fock_general += fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", D_tu_avg, 
                temp1, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 3", end - start)
        ###self.fock_general2 = np.zeros((self.nmo, self.nmo))
        ###for r in range(self.nmo):
        ###    for s in range(self.nmo):
        ###        self.fock_general2[r][s] = self.fock_core[r][s]
        ###        for t in range(self.n_act_orb):
        ###            for u in range(self.n_act_orb):
        ###                self.fock_general2[r][s] += self.D_tu_avg[t*self.n_act_orb +u] * self.J[t+self.n_in_a][u+self.n_in_a][r][s]
        ###                self.fock_general2[r][s] -= 0.5 * self.D_tu_avg[t*self.n_act_orb +u] * self.K[t+self.n_in_a][u+self.n_in_a][r][s]
        ###print("rqq", np.allclose(self.fock_general,self.fock_general2, rtol=1e-14,atol=1e-14))
        #start = timer()
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(self.eigenvecs)
        A[:,:self.n_in_a] = 2.0 * (fock_general[:,:self.n_in_a] + d_cmo_temp[:rot_dim,:self.n_in_a] * off_diagonal_constant)
        
        #off_diagonal_constant2 = 0.0
        #np1 = self.N_p + 1
        #for i in range(self.davidson_roots):
        #    eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
        #    eigenvecs2 = eigenvecs2.transpose(1,0)
        #    for m in range(np1):
        #        for I in range(self.num_det):
        #            if (self.N_p ==0): continue
        #            if (m > 0 and m < self.N_p):
        #                off_diagonal_constant2+= -np.sqrt(m * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m-1)*self.num_det+I]
        #                off_diagonal_constant2 += -np.sqrt((m+1) * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m+1)*self.num_det+I]
        #            elif (m == self.N_p):
        #                off_diagonal_constant2 += -np.sqrt(m * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m-1)*self.num_det+I]
        #            else:
        #                off_diagonal_constant2 += -np.sqrt((m+1) * self.omega/2) * eigenvecs[i][m*self.num_det+I] * eigenvecs[i][(m+1)*self.num_det+I]
        #print("rtty", off_diagonal_constant2, off_diagonal_constant)
        
        A[:,self.n_in_a:self.n_occupied] = np.einsum("rt,tu->ru", fock_core[:rot_dim,self.n_in_a:self.n_occupied],
                D_tu_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 4", end - start)
        #print(np.shape(self.active_twoeint))
        #start = timer()
        A[:,self.n_in_a:self.n_occupied] += np.einsum("vwrt,tuvw->ru", J_temp[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,self.n_in_a:self.n_occupied], 
           D_tuvw_avg, optimize = "optimal")
        #end   = timer()
        #print("build intermediate step 5", end - start)
        #start = timer()
        A[:,self.n_in_a:self.n_occupied] += -np.sqrt(self.omega/2) * np.einsum("rt,tu->ru", d_cmo_temp[:rot_dim,self.n_in_a:self.n_occupied],
                Dpe_tu_avg, optimize = "optimal")



        #B = np.zeros((self.nmo, self.nmo))
        #T = U - np.eye(self.nmo)
        #B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
        #A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
        ##A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)
        #gradient_tilde[:,:] = A_tilde[:,:self.n_occupied] - A_tilde.T[:,:self.n_occupied]
        
        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        gradient_tilde[:,:] = A[:,:self.n_occupied] - A.T[:,:self.n_occupied]
        reduced_gradient = np.zeros(self.index_map_size)
        index_count1 = 0
        for k in range(self.n_occupied):
            for r in range(k+1,self.nmo):
                if (k < self.n_in_a and r < self.n_in_a): continue
                if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                reduced_gradient[index_count1] = gradient_tilde[r][k]
                #print(r,k,index_count1)
                #index_count2 = 0 
                #for l in range(self.n_occupied):
                #    for s in range(l+1,self.nmo):
                #        if (l < self.n_in_a and s < self.n_in_a): continue
                #        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                #        #print(r,k,s,l,index_count1,index_count2)
                #        index_count2 += 1
                index_count1 += 1
        return reduced_gradient 












    def microiteration_predicted_energy(self, reduced_gradient, reduced_hessian, step):
        print(np.shape(reduced_gradient), np.shape(step))
        energy = 2.0 * np.einsum("a,a->", reduced_gradient, step)
        energy += np.einsum("a,ab,b->", step, reduced_hessian, step)
        return energy
    
    def microiteration_predicted_energy2(self, U, reduced_gradient, A_tilde, G, step):
        #energy = 2.0 * np.einsum("a,a->", reduced_gradient, step)
        energy = 2.0 * np.dot(reduced_gradient, step)

        step2 = step.reshape(1,self.index_map_size)
        sigma_reduced = np.zeros((1, self.index_map_size)) 
        #self.build_sigma_reduced2(U, A_tilde, G, step2, sigma_reduced, 1, 0)
        self.orbital_sigma(U, A_tilde, G, step2, sigma_reduced, 1, 0)

        #energy += np.einsum("ip,p->", sigma_reduced,step)
        sigma_reduced = sigma_reduced.reshape((self.index_map_size)) 
        energy += np.dot(sigma_reduced,step)
        return energy


    def microiteration_optimization(self, E0, eigenvecs, c_get_roots):
        print("avg_energy", E0)
        print("E_core", self.E_core)
        self.U2 = np.eye(self.nmo)
        trust_radius = 0.5
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        A2 = np.zeros((rot_dim, rot_dim))
        G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #self.build_intermediates(eigenvecs, A, G, True)
        #self.build_intermediates2(eigenvecs, A2, G2, True)
        print(eigenvecs)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        zero_energy2 = E0
        convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        current_energy = E0
        old_energy = E0
        N_orbital_optimization_steps = 1
        N_microiterations = 20
        microiteration = 0
        while(microiteration < N_microiterations):
        #while(microiteration < 2):
            A[:,:] = 0.0
            G[:,:,:,:] = 0.0
            self.build_intermediates(eigenvecs, A, G, True)
            A2[:,:] = 0.0
            G2[:,:,:,:] = 0.0
            self.build_intermediates2(eigenvecs, A2, G2, True)
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION") 
            if np.abs(current_energy - old_energy < 0.01 * convergence_threshold) and microiteration >=2:
                print("microiteration converged (small energy change)")
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                self.d_cmo[:,:] = d_cmo[:,:]
                break

            zero_energy = self.E_core
            zero_energy += np.dot(self.active_fock_core.flatten(), self.D_tu_avg)
            zero_energy += 0.5 * np.dot(self.active_twoeint.flatten(), self.D_tuvw_avg)  
            zero_energy += -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            zero_energy += self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            zero_energy += self.Enuc
            zero_energy += self.d_c
            print("zero energy", zero_energy2, zero_energy, flush = True)
            initial_energy_change = self.microiteration_exact_energy(self.U2, A, G)
            old_energy = zero_energy + initial_energy_change 
            print("current energy from zero energy + second order energy change", zero_energy + initial_energy_change, flush = True)
            current_energy = old_energy
            #occupied_fock_core = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied]) 
            #occupied_fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
            #occupied_fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        
            #


            #E_core2 = 0.0  
            #E_core2 += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            #E_core2 += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
            #active_one_e_energy2 = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
            #active_two_e_energy2 = 0.5 * np.dot(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
            #active_one_pe_energy2 = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy2 = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            #sum_energy2 = (active_one_e_energy2 + active_two_e_energy2 + active_one_pe_energy2 + E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy2)

            #print("vrfw", sum_energy2, E_core2, active_one_e_energy2, active_two_e_energy2, active_one_pe_energy2, ci_dependent_energy2, self.Enuc, self.d_c)
        
            orbital_optimization_step = 0 
            while(orbital_optimization_step < N_orbital_optimization_steps):
                self.U3 = copy.deepcopy(self.U2)
                print("\n", current_energy)
                print("Microiteration", microiteration + 1, "orbital optimization step", orbital_optimization_step + 1, flush=True)
                #occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #occupied_K = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
                #occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
                #occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))

                gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                A_tilde = np.zeros((rot_dim, rot_dim))
                self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde, True)
                print(np.shape(gradient_tilde), flush = True)
               


                hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))


                reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                reduced_gradient =  np.zeros(self.index_map_size)
                index_count1 = 0 
                for k in range(self.n_occupied):
                    for r in range(k+1,self.nmo):
                        if (k < self.n_in_a and r < self.n_in_a): continue
                        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                        reduced_gradient[index_count1] = gradient_tilde[r][k]
                        #print(r,k,index_count1)
                        index_count2 = 0 
                        for l in range(self.n_occupied):
                            for s in range(l+1,self.nmo):
                                if (l < self.n_in_a and s < self.n_in_a): continue
                                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                                #print(r,k,s,l,index_count1,index_count2)
                                index_count2 += 1
                        index_count1 += 1
                print("reduced_gradient",reduced_gradient, flush = True) 
                np.set_printoptions(precision = 14)  
                mu1, w1 = np.linalg.eigh(reduced_hessian)
                print("eigenvalue of the non-redundant hessian", mu1, flush = True)
                

                #aa = np.einsum("rp,rs,sq->pq", w1, reduced_hessian, w1)
                #self.w1 = copy.deepcopy(w1)
                #print(aa)
                gradient_norm = np.dot(reduced_gradient, reduced_gradient.T)
                gradient_norm = np.sqrt(gradient_norm)
                print("gradient norm", gradient_norm, flush = True)
                print("convergence_threshold", convergence_threshold, flush = True)    
                if (gradient_norm < 0.1 * convergence_threshold):
                    convergence = 1
                    print("Microiteration converged (small gradient norm)")
                    break




                #print(np.shape(hessian_tilde_ai))
                alpha = 1  
                alpha_min = 1  
                alpha_max = 1  
                dim0 = self.index_map_size + 1
                step = np.zeros((self.index_map_size))
                step2 =np.zeros((self.index_map_size))
                step_norm2, scale = self.microiteration_step2(reduced_gradient, reduced_hessian, step2, alpha, dim0)

                sub_microiteration = 0
                #if np.abs(scale) > 1e-9:
                #find (alpha_min, alpha_max)
                while True: 
                    step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                    print("step", step, flush=True)
                    if step_norm > trust_radius:
                        alpha_min = alpha
                        alpha = alpha * 10
                    else:
                        alpha_max = alpha
                        break
                print("alpha range", alpha_min, alpha_max)
                #bisection search
                if alpha_max != 1:
                    while True:
                        #print(alpha_min, alpha_max)
                        alpha = 0.5 * (alpha_min + alpha_max)
                        step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                        if trust_radius - step_norm <= 1e-3 and trust_radius - step_norm >= 0.0:
                            break
                        elif trust_radius - step_norm > 1e-3:
                            alpha_max = alpha
                        else:
                            alpha_min = alpha
                #else:
                #    print("PROBLEM!!!!!!!!!")
                #    step = trust_radius * step2/step_norm2
                #    step_norm = trust_radius
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual,self.n_in_a))
                Rva = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s-self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s-self.n_occupied][l] = step[i]
                    else:
                        Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                #print("jnti",self.E_core)

                self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta) 
                second_order_energy_change = self.microiteration_exact_energy(self.U3, A, G)
                exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                energy_change = zero_energy + second_order_energy_change - current_energy
                print("exact energy2", exact_energy2, flush = True)
                print("exact energy", current_energy, energy_change, second_order_energy_change, flush = True)
                print("new energy", current_energy + energy_change, flush = True)
                if energy_change < 0.0:
                    predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                    predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde, G, step)
                    print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                    self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                    if microiteration == 0 and orbital_optimization_step == 0: 
                        convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                        if step_norm > 0.1: 
                            N_microiterations = 5
                            N_orbital_optimization_steps = 4
                        elif step_norm <= 0.1 and step_norm > 0.01:
                            N_microiterations = 7 
                            N_orbital_optimization_steps = 3
                        print("number of microiteration", N_microiterations, flush = True)    
                        print("number of optimization steps", N_orbital_optimization_steps, flush = True)    
                    orbital_optimization_step += 1   
                    ratio = energy_change/predicted_energy2
                    trust_radius = self.step_control(ratio, trust_radius)
                         
                    active_twoeint[:,:,:,:] = 0.0 
                    active_fock_core[:,:] = 0.0 
                    d_cmo[:,:] = 0.0 

                        
                    self.microiteration_ci_integrals_transform(self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                    active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                    ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                            self.Enuc + self.d_c + ci_dependent_energy)
                    print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
                    print("gfhgy",
                        "{:20.12f}".format(sum_energy),
                        "{:20.12f}".format(active_one_e_energy),
                        "{:20.12f}".format(active_two_e_energy),
                        "{:20.12f}".format(self.E_core2),
                        "{:20.12f}".format(active_one_pe_energy),
                        "{:20.12f}".format(ci_dependent_energy),
                        "{:20.12f}".format(self.Enuc),
                        flush = True
                    )
                    current_energy = zero_energy + second_order_energy_change
              
                else:
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush = True)
                    print(trust_radius, flush = True)
            if convergence == 1:
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                break


            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            print("recheck energy", flush = True)
            active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                    self.Enuc + self.d_c + ci_dependent_energy)
            print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            print("gfhgy",
                "{:20.12f}".format(sum_energy),
                "{:20.12f}".format(active_one_e_energy),
                "{:20.12f}".format(active_two_e_energy),
                "{:20.12f}".format(self.E_core2),
                "{:20.12f}".format(active_one_pe_energy),
                "{:20.12f}".format(ci_dependent_energy),
                "{:20.12f}".format(self.Enuc),
                flush = True
            )
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    self.E_core2, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = self.E_core2
            eigenvals = np.zeros((self.davidson_roots))
            self.constint[8] = 2 
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            #print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            #print("iteration",microiteration + 1, avg_energy, flush = True)
            current_energy = avg_energy
            self.build_state_avarage_rdms(eigenvecs)
            total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2)) 
            if total_norm < convergence_threshold: 
                print("total norm", total_norm, flush = True)
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                self.d_cmo[:,:] = d_cmo[:,:]
                print(eigenvecs)
                print("microiteration converged! (small total norm)", flush = True)
                break 


            microiteration += 1

    def microiteration_optimization2(self, eigenvecs, c_get_roots):
        print("E_core", self.E_core)
        self.U2 = np.eye(self.nmo)
        trust_radius = 0.5
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        A2 = np.zeros((rot_dim, rot_dim))
        G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #self.build_intermediates(eigenvecs, A, G, True)
        #self.build_intermediates2(eigenvecs, A2, G2, True)
        print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))
        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size+1))

        convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        current_energy = 0.0
        old_energy = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 20
        microiteration = 0
        while(microiteration < N_microiterations):
        #while(microiteration < 2):
            A[:,:] = 0.0
            G[:,:,:,:] = 0.0
            self.build_intermediates(eigenvecs, A, G, True)
            A2[:,:] = 0.0
            G2[:,:,:,:] = 0.0
            self.build_intermediates2(eigenvecs, A2, G2, True)
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION") 
            if np.abs(current_energy - old_energy < 0.01 * convergence_threshold) and microiteration >=2:
                print("microiteration converged (small energy change)")
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                self.d_cmo[:,:] = d_cmo[:,:]
                break

            zero_energy = self.E_core
            zero_energy += np.dot(self.active_fock_core.flatten(), self.D_tu_avg)
            zero_energy += 0.5 * np.dot(self.active_twoeint.flatten(), self.D_tuvw_avg)  
            zero_energy += -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            zero_energy += self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            zero_energy += self.Enuc
            zero_energy += self.d_c
            print("zero energy", zero_energy, flush = True)
            initial_energy_change = self.microiteration_exact_energy(self.U2, A, G)
            old_energy = zero_energy + initial_energy_change 
            print("current energy from zero energy + second order energy change", zero_energy + initial_energy_change, flush = True)
            current_energy = old_energy
            #occupied_fock_core = copy.deepcopy(self.H_spatial2[:self.n_occupied,:self.n_occupied]) 
            #occupied_fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
            #occupied_fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:self.n_occupied,:self.n_occupied]) 
        
            #


            #E_core2 = 0.0  
            #E_core2 += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            #E_core2 += np.einsum("jj->",occupied_fock_core[:self.n_in_a,:self.n_in_a]) 
            #active_one_e_energy2 = np.dot(occupied_fock_core[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tu_avg)
            #active_two_e_energy2 = 0.5 * np.dot(self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.D_tuvw_avg)
            #active_one_pe_energy2 = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy2 = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            #sum_energy2 = (active_one_e_energy2 + active_two_e_energy2 + active_one_pe_energy2 + E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy2)

            #print("vrfw", sum_energy2, E_core2, active_one_e_energy2, active_two_e_energy2, active_one_pe_energy2, ci_dependent_energy2, self.Enuc, self.d_c)
            restart = False 
            orbital_optimization_step = 0 
            while(orbital_optimization_step < N_orbital_optimization_steps):
                self.U3 = copy.deepcopy(self.U2)
                print("\n", current_energy)
                print("Microiteration", microiteration + 1, "orbital optimization step", orbital_optimization_step + 1, flush=True)
                #occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #occupied_K = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
                #occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
                #occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
                #occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))

                gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                print(np.shape(gradient_tilde), flush = True)
               
                gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
                A_tilde2 = np.zeros((rot_dim, rot_dim))
                self.build_gradient(self.U2, A, G, gradient_tilde2, A_tilde2, True)

                hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                reduced_hessian_diagonal = np.zeros(self.index_map_size)
                index_count1 = 0 
                for k in range(self.n_occupied):
                    for r in range(k+1,self.nmo):
                        if (k < self.n_in_a and r < self.n_in_a): continue
                        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                        index_count1 += 1
                self.build_hessian_diagonal(self.U2, G, A_tilde2)
                #print("diagonal elements of the non redundant hessian")   
                #for a in range(self.nmo):
                #    for i in range(self.n_occupied):
                #        aa = hessian_diagonal3[a][i] - self.hessian_diagonal[a][i]
                #        if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                print("diagonal elements of the reduced hessian")   
                for i in range(self.index_map_size):
                    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                reduced_gradient = np.zeros(self.index_map_size)
                index_count1 = 0 
                for k in range(self.n_occupied):
                    for r in range(k+1,self.nmo):
                        if (k < self.n_in_a and r < self.n_in_a): continue
                        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                        reduced_gradient[index_count1] = gradient_tilde[r][k]
                        #print(r,k,index_count1)
                        index_count2 = 0 
                        for l in range(self.n_occupied):
                            for s in range(l+1,self.nmo):
                                if (l < self.n_in_a and s < self.n_in_a): continue
                                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                                #print(r,k,s,l,index_count1,index_count2)
                                index_count2 += 1
                        index_count1 += 1
                print("reduced_gradient",reduced_gradient, flush = True)
                restart = False
                davidson_step = np.zeros((1, self.index_map_size))
                guess_vector = np.zeros((1, self.index_map_size))
                self.mu_min = 0.0
                trial_vector, hard_case = self.Davidson_augmented_hessian_solve(self.U2, A_tilde2, G, self.reduced_hessian_diagonal, reduced_gradient,
                        davidson_step, trust_radius, guess_vector, restart)
                np.set_printoptions(precision = 14)  
                mu1, w1 = np.linalg.eigh(reduced_hessian)
                print("eigenvalue of the non-redundant hessian", mu1, flush = True)
                product = np.einsum("p,p->",reduced_gradient, w1[:,0])
                H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                H_lambda_inverse = np.linalg.pinv(H_lambda)
                print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                print("step_limit", step_limit)
                print("limit of step norm", np.linalg.norm(step_limit))
                print("check dot product of gradient and first eigenvector", product)
                Q = np.zeros((1, self.index_map_size))
                H_diag = self.reduced_hessian_diagonal - mu1[0] 
                Q[0,:] = np.divide(-reduced_gradient, H_diag, out=np.zeros_like(H_diag), where=H_diag!=0)
                S = np.zeros_like(Q)
                self.A_tilde2 = A_tilde2
                self.G = G
                print(np.shape(Q), self.index_map_size)
                #H_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, mu1[0]), 
                #        rmatvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, mu1[0]))
                H_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, mu1[0])) 
                print(H_op)
                S=H_op.matvec(Q.T)
                print(S)
                S2 = np.einsum("pq,qr->pr", H_lambda, Q.T)
                print(S2)
                print("diagonal element of the reduced hessian", np.diagonal(reduced_hessian))   
                x, exitCode= minres(H_op, -reduced_gradient)
                #x, istop, itn, normr, normar, norma, conda, normx = lsmr(H_op, -reduced_gradient)[:8]
                #print("reo0", x, istop, itn, normr, normx, conda)
                print("exitcode", exitCode)
                print("step norm from scipy solver", np.linalg.norm(x))
                print("step from scipy solver", x)
                test_g = np.einsum("pq,q->p", H_lambda,x)
                print(test_g)
                print(reduced_gradient)



                #x2, istop2, itn2, normr2, normar2, norma2, conda2, normx2 = lsmr(H_lambda, reduced_gradient)[:8]
                #print("reo9", x2, istop2, itn2, normr2, normx2, conda2)
                Rai2 = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi2 = np.zeros((self.n_virtual,self.n_in_a))
                Rva2 = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai2[s-self.n_in_a][l] = step_limit[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi2[s-self.n_occupied][l] = step_limit[i]
                    else:
                        Rva2[s-self.n_occupied][l-self.n_in_a] = step_limit[i]

                self.build_unitary_matrix(Rai2, Rvi2, Rva2)
                #print("jnti",self.E_core)
                U4 = copy.deepcopy(self.U3)
                U4 = np.einsum("pq,qs->ps", U4, self.U_delta) 
                second_order_energy_change1 = self.microiteration_exact_energy(U4, A, G)
                energy_change1 = zero_energy + second_order_energy_change1 - current_energy
                print("energy change1", energy_change1, flush = True)

                if np.abs(product) < 1e-5: 
                    self.Davidson_linear_matrix_equation_solve(self.U2, A_tilde2, G, self.reduced_hessian_diagonal, reduced_gradient, mu1[0], reduced_hessian, hessian_tilde3)

                #aa = np.einsum("rp,rs,sq->pq", w1, reduced_hessian, w1)
                #self.w1 = copy.deepcopy(w1)
                #print(aa)
                gradient_norm = np.dot(reduced_gradient, reduced_gradient.T)
                gradient_norm = np.sqrt(gradient_norm)
                print("gradient norm", gradient_norm, flush = True)
                print("convergence_threshold", convergence_threshold, flush = True)    
                if (gradient_norm < 0.1 * convergence_threshold):
                    convergence = 1
                    print("Microiteration converged (small gradient norm)")
                    break




                #print(np.shape(hessian_tilde_ai))
                alpha = 1  
                alpha_min = 1  
                alpha_max = 1  
                dim0 = self.index_map_size + 1
                step = np.zeros((self.index_map_size))
                step2 =np.zeros((self.index_map_size))
                step_norm2, scale = self.microiteration_step2(reduced_gradient, reduced_hessian, step2, alpha, dim0)
                print("scale", scale)
                sub_microiteration = 0
                if np.abs(scale) > 1e-4:
                    ##find (alpha_min, alpha_max)
                    while True: 
                        step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                        print("step", step, flush=True)
                        if step_norm > trust_radius:
                            alpha_min = alpha
                            alpha = alpha * 10
                        else:
                            alpha_max = alpha
                            break
                    print("alpha range", alpha_min, alpha_max)
                    #bisection search
                    if alpha_max != 1:
                        while True:
                            #print(alpha_min, alpha_max)
                            alpha = 0.5 * (alpha_min + alpha_max)
                            step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                            if trust_radius - step_norm <= 1e-3 and trust_radius - step_norm >= 0.0:
                                break
                            elif trust_radius - step_norm > 1e-3:
                                alpha_max = alpha
                            else:
                                alpha_min = alpha
                else:
                    print("PROBLEM!!!!!!!!!")
                    step = trust_radius * step2/step_norm2
                    step_norm = trust_radius
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual,self.n_in_a))
                Rva = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s-self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s-self.n_occupied][l] = step[i]
                    else:
                        Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                #print("jnti",self.E_core)

                self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta) 
                second_order_energy_change = self.microiteration_exact_energy(self.U3, A, G)
                exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                energy_change = zero_energy + second_order_energy_change - current_energy
                print("exact energy2", exact_energy2, flush = True)
                print("exact energy", current_energy, energy_change, second_order_energy_change, flush = True)
                print("new energy", current_energy + energy_change, flush = True)
                if energy_change < 0.0:
                    restart = False
                    predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                    predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                    print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                    self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                    if microiteration == 0 and orbital_optimization_step == 0: 
                        convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                        if step_norm > 0.1: 
                            N_microiterations = 5
                            N_orbital_optimization_steps = 4
                        elif step_norm <= 0.1 and step_norm > 0.01:
                            N_microiterations = 7 
                            N_orbital_optimization_steps = 3
                        print("number of microiteration", N_microiterations, flush = True)    
                        print("number of optimization steps", N_orbital_optimization_steps, flush = True)    
                    orbital_optimization_step += 1   
                    ratio = energy_change/predicted_energy2
                    trust_radius = self.step_control(ratio, trust_radius)
                         
                    active_twoeint[:,:,:,:] = 0.0 
                    active_fock_core[:,:] = 0.0 
                    d_cmo[:,:] = 0.0 

                        
                    self.microiteration_ci_integrals_transform(self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                    active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                    active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                    ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                    sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                            self.Enuc + self.d_c + ci_dependent_energy)
                    print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
                    print("gfhgy",
                        "{:20.12f}".format(sum_energy),
                        "{:20.12f}".format(active_one_e_energy),
                        "{:20.12f}".format(active_two_e_energy),
                        "{:20.12f}".format(self.E_core2),
                        "{:20.12f}".format(active_one_pe_energy),
                        "{:20.12f}".format(ci_dependent_energy),
                        "{:20.12f}".format(self.Enuc),
                        flush = True
                    )
                    current_energy = zero_energy + second_order_energy_change
              
                else:
                    restart = True
                    guess_vector = trial_vector 
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush = True)
                    print(trust_radius, flush = True)
            if convergence == 1:
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                break


            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            print("recheck energy", flush = True)
            active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                    self.Enuc + self.d_c + ci_dependent_energy)
            print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            print("gfhgy",
                "{:20.12f}".format(sum_energy),
                "{:20.12f}".format(active_one_e_energy),
                "{:20.12f}".format(active_two_e_energy),
                "{:20.12f}".format(self.E_core2),
                "{:20.12f}".format(active_one_pe_energy),
                "{:20.12f}".format(ci_dependent_energy),
                "{:20.12f}".format(self.Enuc),
                flush = True
            )
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    self.E_core2, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = self.E_core2
            self.constint[8] = 2 
            eigenvals = np.zeros((self.davidson_roots))
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print("iteration",microiteration + 1, avg_energy, flush = True)
            current_energy = avg_energy
            self.build_state_avarage_rdms(eigenvecs)
            total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2)) 
            if total_norm < convergence_threshold: 
                print("total norm", total_norm, flush = True)
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                self.d_cmo[:,:] = d_cmo[:,:]
                print(eigenvecs)
                print("microiteration converged! (small total norm)", flush = True)
                break 


            microiteration += 1


            
    def microiteration_optimization3(self, U, eigenvecs, c_get_roots, convergence_threshold):
        print("E_core", self.E_core)
        self.U2 = copy.deepcopy(U)
        trust_radius = 0.4
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        A2 = np.zeros((rot_dim, rot_dim))
        G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #self.build_intermediates(eigenvecs, A, G, True)
        #self.build_intermediates2(eigenvecs, A2, G2, True)
        print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size+1))
        #convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        current_energy = 0.0
        old_energy = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 20
        microiteration = 0
        while(microiteration < N_microiterations):
        #while(microiteration < 2):
            A[:,:] = 0.0
            G[:,:,:,:] = 0.0
            self.build_intermediates(eigenvecs, A, G, True)
            A2[:,:] = 0.0
            G2[:,:,:,:] = 0.0
            self.build_intermediates2(eigenvecs, A2, G2, True)
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION") 
            if np.abs(current_energy - old_energy < 0.01 * convergence_threshold) and microiteration >=2:
                print("microiteration converged (small energy change)")
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                #temp8 = np.zeros((self.nmo, self.nmo))
                #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                #self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                break

            zero_energy = self.E_core
            zero_energy += np.dot(self.active_fock_core.flatten(), self.D_tu_avg)
            zero_energy += 0.5 * np.dot(self.active_twoeint.flatten(), self.D_tuvw_avg)  
            zero_energy += -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            zero_energy += self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            zero_energy += self.Enuc
            zero_energy += self.d_c
            print("zero energy", zero_energy, flush = True)
            initial_energy_change = self.microiteration_exact_energy(self.U2, A, G)
            old_energy = zero_energy + initial_energy_change 
            print("current energy from zero energy + second order energy change", zero_energy + initial_energy_change, flush = True)
            current_energy = old_energy
           



            #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            #print(np.shape(gradient_tilde), flush = True)
            
            #self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            #this  function build only a part of the hessian
            #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)

            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            reduced_gradient = np.zeros(self.index_map_size)
            index_count1 = 0 
            for k in range(self.n_occupied):
                for r in range(k+1,self.nmo):
                    if (k < self.n_in_a and r < self.n_in_a): continue
                    if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    #print(r,k,index_count1)
                    ####index_count2 = 0 
                    ####for l in range(self.n_occupied):
                    ####    for s in range(l+1,self.nmo):
                    ####        if (l < self.n_in_a and s < self.n_in_a): continue
                    ####        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                    ####        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                    ####        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                    ####        #print(r,k,s,l,index_count1,index_count2)
                    ####        index_count2 += 1
                    index_count1 += 1
            print("reduced_gradient",reduced_gradient, flush = True)
            ####mu1, w1 = np.linalg.eigh(reduced_hessian)
            


            orbital_optimization_step = 0 
            restart = False
            while(orbital_optimization_step < N_orbital_optimization_steps):
                self.U3 = copy.deepcopy(self.U2)
                print("\n", current_energy)
                print("Microiteration", microiteration + 1, "orbital optimization step", orbital_optimization_step + 1, flush=True)
                
                #########gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                #########hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                #########self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                #########print(np.shape(gradient_tilde), flush = True)
               
                #########gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
                #########A_tilde2 = np.zeros((rot_dim, rot_dim))
                #########self.build_gradient(self.U2, A, G, gradient_tilde2, A_tilde2, True)

                #########hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                #########hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                #########hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                #########reduced_hessian_diagonal = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                #########        index_count1 += 1
                #########self.build_hessian_diagonal(self.U2, G, A_tilde2)
                ##########print("diagonal elements of the non redundant hessian")   
                ##########for a in range(self.nmo):
                ##########    for i in range(self.n_occupied):
                ##########        aa = hessian_diagonal3[a][i] - self.hessian_diagonal[a][i]
                ##########        if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########print("diagonal elements of the reduced hessian")   
                #########for i in range(self.index_map_size):
                #########    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                #########    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                #########reduced_gradient = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_gradient[index_count1] = gradient_tilde[r][k]
                #########        #print(r,k,index_count1)
                #########        index_count2 = 0 
                #########        for l in range(self.n_occupied):
                #########            for s in range(l+1,self.nmo):
                #########                if (l < self.n_in_a and s < self.n_in_a): continue
                #########                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #########                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #########                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                #########                #print(r,k,s,l,index_count1,index_count2)
                #########                index_count2 += 1
                #########        index_count1 += 1
                #########print("reduced_gradient",reduced_gradient, flush = True)
                self.mu_min = 0.0
                trial_vector, hard_case = self.Davidson_augmented_hessian_solve(self.U2, A_tilde2, G, self.reduced_hessian_diagonal, reduced_gradient,
                        davidson_step, trust_radius, guess_vector, restart)
                print("hard case", hard_case) 
                print("smallest eigenvalue of the augmented hessian", self.mu_min)
                step2 = davidson_step.flatten()
                product = np.einsum("p,p->",reduced_gradient, step2)
                np.set_printoptions(precision = 14)  
                print("check dot product of gradient and first eigenvector", product)
                trust_radius_hard_case = trust_radius 
                if hard_case == 0:
                    step = step2
                else:
                    ####print("eigenvalue of the non-redundant hessian", mu1, flush = True)
                    ####product = np.einsum("p,p->",reduced_gradient, w1[:,0])
                    ####H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                    ####step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                    ####H_lambda_inverse = np.linalg.pinv(H_lambda)
                    ####print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                    ####print("step_limit", step_limit)
                    ####print("limit of step norm", np.linalg.norm(step_limit))
                    ####print("check dot product of gradient and first eigenvector", product)
                    Q = np.zeros((1, self.index_map_size))
                    #H_diag = self.reduced_hessian_diagonal - mu1[0] 
                    #Q[0,:] = np.divide(-reduced_gradient, H_diag, out=np.zeros_like(H_diag), where=H_diag!=0)
                    #S = np.zeros_like(Q)
                    
                    H_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, self.mu_min)) 
                    #S=H_op.matvec(Q.T)
                    #print(S)
                    #S2 = np.einsum("pq,qr->pr", H_lambda, Q.T)
                    #print(S2)
                    #print("diagonal element of the reduced hessian", np.diagonal(reduced_hessian))   
                    x, exitCode= minres(H_op, -reduced_gradient, tol = 1e-6)
                    #x, istop, itn, normr, normar, norma, conda, normx = lsmr(H_op, -reduced_gradient)[:8]
                    #print("reo0", x, istop, itn, normr, normx, conda)
                    print("exitcode", exitCode)
                    print("step norm from scipy solver", np.linalg.norm(x))
                    print("step from scipy solver", x)
                    print(np.shape(x))
                    S = np.zeros((1, self.index_map_size))
                    S = H_op.matvec(davidson_step.T)
                    print(S)
                    #test_g = np.einsum("pq,q->p", H_lambda,x)
                    #test_g2 = np.einsum("pq,q->p", H_lambda,step_limit)
                    #print(test_g)
                    #print(test_g2)
                    #print(reduced_gradient)
                    if (np.linalg.norm(x) < trust_radius):
                        step = x
                        #product2 = np.dot(x, step2)
                        xy_square = np.dot(x, step2) * np.dot(x, step2)
                        x_square = np.dot(x, x) 
                        y_square = np.dot(step2, step2) 
                        delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                        #print(delta)
                        t1= (-2 * np.dot(x,step2) - np.sqrt(delta))/ (2*y_square)
                        t2= (-2 * np.dot(x,step2) + np.sqrt(delta))/ (2*y_square)
                        print("x^2, xy, y^2, t", x_square, np.dot(x, step2), y_square, t1)
                        adjusted_step = step + min(t1,t2) * step2
                        print("adjusted step norm", np.linalg.norm(adjusted_step))
                        step = adjusted_step
                        trust_radius_hard_case = np.linalg.norm(adjusted_step)
                        trust_radius = trust_radius_hard_case
                    else:
                        step = x/np.linalg.norm(x) * trust_radius

                    #step = step2/np.linalg.norm(step2) * trust_radius 


                step_norm = np.linalg.norm(step)
                gradient_norm = np.dot(reduced_gradient, reduced_gradient.T)
                gradient_norm = np.sqrt(gradient_norm)
                print("gradient norm", gradient_norm, flush = True)
                print("convergence_threshold", convergence_threshold, flush = True)    
                if (gradient_norm < 0.1 * convergence_threshold):
                    convergence = 1
                    print("Microiteration converged (small gradient norm)")
                    break




                ####alpha = 1  
                ####alpha_min = 1  
                ####alpha_max = 1  
                ####dim0 = self.index_map_size + 1
                ####step = np.zeros((self.index_map_size))
                ####step2 =np.zeros((self.index_map_size))
                ####step_norm2, scale = self.microiteration_step2(reduced_gradient, reduced_hessian, step2, alpha, dim0)
                ####print("scale", scale)
                ####if np.abs(scale) > 1e-4:
                ####    ##find (alpha_min, alpha_max)
                ####    while True: 
                ####        step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                ####        #print("step", step, flush=True)
                ####        if step_norm > trust_radius:
                ####            alpha_min = alpha
                ####            alpha = alpha * 10
                ####        else:
                ####            alpha_max = alpha
                ####            break
                ####    print("alpha range", alpha_min, alpha_max)
                ####    #bisection search
                ####    if alpha_max != 1:
                ####        while True:
                ####            #print(alpha_min, alpha_max)
                ####            alpha = 0.5 * (alpha_min + alpha_max)
                ####            step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                ####            if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                ####                break
                ####            elif trust_radius - step_norm > 1e-2:
                ####                alpha_max = alpha
                ####            else:
                ####                alpha_min = alpha
                ####else:
                ####    print("PROBLEM!!!!!!!!!")
                ####    step = trust_radius * step2/step_norm2
                ####    step_norm = trust_radius
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual,self.n_in_a))
                Rva = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s-self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s-self.n_occupied][l] = step[i]
                    else:
                        Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                #print("jnti",self.E_core)

                self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta) 
                second_order_energy_change = self.microiteration_exact_energy(self.U3, A, G)
                exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                energy_change = zero_energy + second_order_energy_change - current_energy
                print("exact energy2", exact_energy2, flush = True)
                print("exact energy", current_energy, energy_change, second_order_energy_change, flush = True)
                print("new energy", current_energy + energy_change, flush = True)
                



                #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                print("microinteration predicted energy", predicted_energy2, flush = True)

                if energy_change < 0.0:
                     #restart = False
                     #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                     #predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                     #print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                     self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                     if microiteration == 0 and orbital_optimization_step == 0: 
                         convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                         if step_norm > 0.1: 
                             N_microiterations = 5
                             N_orbital_optimization_steps = 4
                         elif step_norm <= 0.1 and step_norm > 0.01:
                             N_microiterations = 7 
                             N_orbital_optimization_steps = 3
                         print("number of microiteration", N_microiterations, flush = True)    
                         print("number of optimization steps", N_orbital_optimization_steps, flush = True)    
                     orbital_optimization_step += 1   
                     ratio = energy_change/predicted_energy2
                     trust_radius = self.step_control(ratio, trust_radius)
                         

                     #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                     #print(np.shape(gradient_tilde), flush = True)
                     
                     self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
                     #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)

                     #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                     #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                     #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                     #reduced_hessian_diagonal = np.zeros(self.index_map_size)
                     #index_count1 = 0 
                     #for k in range(self.n_occupied):
                     #    for r in range(k+1,self.nmo):
                     #        if (k < self.n_in_a and r < self.n_in_a): continue
                     #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                     #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                     #        index_count1 += 1
                     self.build_hessian_diagonal(self.U2, G, A_tilde2)
                     #print("diagonal elements of the reduced hessian")   
                     #for i in range(self.index_map_size):
                     #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                     #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                     #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                     reduced_gradient = np.zeros(self.index_map_size)
                     index_count1 = 0 
                     for k in range(self.n_occupied):
                         for r in range(k+1,self.nmo):
                             if (k < self.n_in_a and r < self.n_in_a): continue
                             if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                             reduced_gradient[index_count1] = gradient_tilde[r][k]
                             #print(r,k,index_count1)
                             #index_count2 = 0 
                             #for l in range(self.n_occupied):
                             #    for s in range(l+1,self.nmo):
                             #        if (l < self.n_in_a and s < self.n_in_a): continue
                             #        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                             #        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                             #        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                             #        #print(r,k,s,l,index_count1,index_count2)
                             #        index_count2 += 1
                             index_count1 += 1
                     print("reduced_gradient",reduced_gradient, flush = True)
            






                     active_twoeint[:,:,:,:] = 0.0 
                     active_fock_core[:,:] = 0.0 
                     d_cmo[:,:] = 0.0 

                         
                     self.microiteration_ci_integrals_transform(self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                     active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                     active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                     active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                     ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                     sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                             self.Enuc + self.d_c + ci_dependent_energy)
                     print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
                     print("gfhgy",
                         "{:20.12f}".format(sum_energy),
                         "{:20.12f}".format(active_one_e_energy),
                         "{:20.12f}".format(active_two_e_energy),
                         "{:20.12f}".format(self.E_core2),
                         "{:20.12f}".format(active_one_pe_energy),
                         "{:20.12f}".format(ci_dependent_energy),
                         "{:20.12f}".format(self.Enuc),
                         flush = True
                     )
                     current_energy = zero_energy + second_order_energy_change
              
                else:
                    restart = True 
                    guess_vector = trial_vector
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush = True)
                    print(trust_radius, flush = True)
            if convergence == 1:
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                #temp8 = np.zeros((self.nmo, self.nmo))
                #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                #self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                break


            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            print("recheck energy", flush = True)
            active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                    self.Enuc + self.d_c + ci_dependent_energy)
            print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            print("gfhgy",
                "{:20.12f}".format(sum_energy),
                "{:20.12f}".format(active_one_e_energy),
                "{:20.12f}".format(active_two_e_energy),
                "{:20.12f}".format(self.E_core2),
                "{:20.12f}".format(active_one_pe_energy),
                "{:20.12f}".format(ci_dependent_energy),
                "{:20.12f}".format(self.Enuc),
                flush = True
            )
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    self.E_core2, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = self.E_core2
            self.constint[8] = 2 
            eigenvals = np.zeros((self.davidson_roots))
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print("iteration",microiteration + 1, avg_energy, flush = True)
            current_energy = avg_energy
            self.build_state_avarage_rdms(eigenvecs)
            total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2)) 
            if total_norm < convergence_threshold: 
                print("total norm", total_norm, flush = True)
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                #temp8 = np.zeros((self.nmo, self.nmo))
                #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #self.d_cmo[:,:] = d_cmo[:,:]
                print(eigenvecs)
                print("microiteration converged! (small total norm)", flush = True)
                break 


            microiteration += 1


    def microiteration_optimization4(self, U, eigenvecs, c_get_roots, convergence_threshold):
        #print("E_core", self.E_core)
        self.U2 = copy.deepcopy(U)
        trust_radius = 0.4   
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #A2 = np.zeros((rot_dim, rot_dim))
        #G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        #hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size+1))
        #convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        current_energy = 0.0
        old_energy = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 1  
        microiteration = 0
        while(microiteration < N_microiterations):
            print("\n")
            print("\n")
            print("MICROITERATION", microiteration+1,flush = True)

        #while(microiteration < 2):
            trust_radius = 0.35 
            A[:,:] = 0.0
            G[:,:,:,:] = 0.0
            start = timer()
            self.build_intermediates(eigenvecs, A, G, True)
            end   = timer()
            print("build intermediates took", end - start)
            #A2[:,:] = 0.0
            #G2[:,:,:,:] = 0.0
            #self.build_intermediates2(eigenvecs, A2, G2, True)
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            print(old_energy, current_energy)
            print("initial convergence threshold", convergence_threshold)
            if (np.abs(current_energy - old_energy) < 0.01 * convergence_threshold) and microiteration >=2:
            #if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)")
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                break

            start = timer()
            start1 = timer()
            zero_energy = self.E_core
            zero_energy += np.dot(self.active_fock_core.flatten(), self.D_tu_avg)
            zero_energy += 0.5 * np.dot(self.active_twoeint.flatten(), self.D_tuvw_avg)  
            zero_energy += -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            end1   = timer()
            print("check zero energy took", end1 - start1)
            start1 = timer()
            zero_energy += self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            end1   = timer()
            print("check ci dependent energy took", end1 - start1)
            zero_energy += self.Enuc
            zero_energy += self.d_c
            print("zero energy", zero_energy, flush = True)
            start1 = timer()
            G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            end1   = timer()
            print("transpose matrix G took", end1 - start1)
            start1 = timer()
            initial_energy_change = self.microiteration_exact_energy(self.U2, A, G1)
            end1   = timer()
            print("calculate initial exact energy took", end1 - start1)
            old_energy = zero_energy + initial_energy_change 
            print("current energy from zero energy + second order energy change", zero_energy + initial_energy_change, flush = True)
            current_energy = old_energy
            end   = timer()
            print("check initial energy took", end - start)
           



            #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            #print(np.shape(gradient_tilde), flush = True)
            
            start = timer() 
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            end   = timer() 
            #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            
            print("build gradient took", end - start)
            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()    
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end   = timer() 
            print("build hessian diagonal took", end - start)
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            reduced_gradient = np.zeros(self.index_map_size)
            index_count1 = 0 
            for k in range(self.n_occupied):
                for r in range(k+1,self.nmo):
                    if (k < self.n_in_a and r < self.n_in_a): continue
                    if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    #print(r,k,index_count1)
                    ####index_count2 = 0 
                    ####for l in range(self.n_occupied):
                    ####    for s in range(l+1,self.nmo):
                    ####        if (l < self.n_in_a and s < self.n_in_a): continue
                    ####        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                    ####        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                    ####        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                    ####        #print(r,k,s,l,index_count1,index_count2)
                    ####        index_count2 += 1
                    index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            ####mu1, w1 = np.linalg.eigh(reduced_hessian)
            #gradient_norm = np.linalg.norm(reduced_gradient)
            #print("gradient norm", gradient_norm)
            ##if microiteration ==0:  
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            #if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break

            orbital_optimization_step = 0 
            restart = False
            count = 0
            while(orbital_optimization_step < N_orbital_optimization_steps):
                self.U3 = copy.deepcopy(self.U2)
                print("\n", current_energy)
                print("Microiteration", microiteration + 1, "orbital optimization step", orbital_optimization_step + 1, flush=True)
                gradient_norm = np.linalg.norm(reduced_gradient)
                print("gradient norm", gradient_norm, flush = True)
                print("convergence_threshold", convergence_threshold, flush = True)    
                
                #if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                #if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                #if (gradient_norm < 1e-5 and microiteration > 0):
                if (gradient_norm < 0.1 * convergence_threshold and microiteration > 0):
                    convergence = 1
                    #print("Microiteration converged (small gradient norm)")
                    break
                if (gradient_norm < 1e-7):
                    convergence = 1
                    break


                #########gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                #########hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                #########self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                #########print(np.shape(gradient_tilde), flush = True)
               
                #########gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
                #########A_tilde2 = np.zeros((rot_dim, rot_dim))
                #########self.build_gradient(self.U2, A, G, gradient_tilde2, A_tilde2, True)

                #########hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                #########hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                #########hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                #########reduced_hessian_diagonal = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                #########        index_count1 += 1
                #########self.build_hessian_diagonal(self.U2, G, A_tilde2)
                ##########print("diagonal elements of the non redundant hessian")   
                ##########for a in range(self.nmo):
                ##########    for i in range(self.n_occupied):
                ##########        aa = hessian_diagonal3[a][i] - self.hessian_diagonal[a][i]
                ##########        if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########print("diagonal elements of the reduced hessian")   
                #########for i in range(self.index_map_size):
                #########    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                #########    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                #########reduced_gradient = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_gradient[index_count1] = gradient_tilde[r][k]
                #########        #print(r,k,index_count1)
                #########        index_count2 = 0 
                #########        for l in range(self.n_occupied):
                #########            for s in range(l+1,self.nmo):
                #########                if (l < self.n_in_a and s < self.n_in_a): continue
                #########                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #########                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #########                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                #########                #print(r,k,s,l,index_count1,index_count2)
                #########                index_count2 += 1
                #########        index_count1 += 1
                #########print("reduced_gradient",reduced_gradient, flush = True)
                self.mu_min = 0.0
                #start = timer() 
                #trial_vector, hard_case = self.Davidson_augmented_hessian_solve(self.U2, A_tilde2, G, G1, self.reduced_hessian_diagonal, reduced_gradient,
                #        davidson_step, trust_radius, guess_vector, restart)
                #end   = timer() 
                #print("solving augmented hessian took", end - start)
                start = timer() 
                trial_vector, hard_case = self.Davidson_augmented_hessian_solve2(self.U2, A_tilde2, G, G1, self.reduced_hessian_diagonal, reduced_gradient,
                        davidson_step, trust_radius, guess_vector, restart)
                end   = timer() 
                print("solving augmented hessian2 took", end - start)


                print("hard case", hard_case) 
                print("smallest eigenvalue of the augmented hessian", self.mu_min)
                step2 = davidson_step.flatten()
                product = np.einsum("p,p->",reduced_gradient, step2)
                np.set_printoptions(precision = 14)  
                print("check dot product of gradient and first eigenvector", product)
                trust_radius_hard_case = trust_radius 
                if hard_case == 0:
                    step = step2
                else:
                    ####print("eigenvalue of the non-redundant hessian", mu1, flush = True)
                    ####product = np.einsum("p,p->",reduced_gradient, w1[:,0])
                    ####H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                    ####step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                    ####H_lambda_inverse = np.linalg.pinv(H_lambda)
                    ####print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                    ####print("step_limit", step_limit)
                    ####print("limit of step norm", np.linalg.norm(step_limit))
                    ####print("check dot product of gradient and first eigenvector", product)
                    Q = np.zeros((1, self.index_map_size))
                    #H_diag = self.reduced_hessian_diagonal - mu1[0] 
                    #Q[0,:] = np.divide(-reduced_gradient, H_diag, out=np.zeros_like(H_diag), where=H_diag!=0)
                    #S = np.zeros_like(Q)
                    
                    #H_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, self.mu_min)) 
                    H1_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv2(self.U2, A_tilde2, G1, Q, 1,0, self.mu_min)) 
                    #S=H_op.matvec(Q.T)
                    #print(S)
                    #S2 = np.einsum("pq,qr->pr", H_lambda, Q.T)
                    #print(S2)
                    #print("diagonal element of the reduced hessian", np.diagonal(reduced_hessian))   
                    x, exitCode= minres(H1_op, -reduced_gradient, tol = 1e-6)
                    #x, istop, itn, normr, normar, norma, conda, normx = lsmr(H_op, -reduced_gradient)[:8]
                    #print("reo0", x, istop, itn, normr, normx, conda)
                    print("exitcode", exitCode)
                    print("step norm from scipy solver", np.linalg.norm(x))
                    #Q1 = np.random.rand(1, self.index_map_size)
                    #start = timer()
                    #S1=H_op.matvec(Q1.T)
                    #end = timer()
                    #print("build sigma with numpy took", end - start)
                    #start = timer()
                    #S2=H1_op.matvec(Q1.T)
                    #end = timer()
                    #print("build sigma with numba took", end - start)
                    #print("rg8l", np.allclose(S1,S2, rtol=1e-14,atol=1e-14))
                    #for j in range(self.index_map_size):
                    #    abc = S1[j] -S2[j]
                    #    if np.abs(abc) > 1e-14: print(j, abc)
                    #for i in range(1):
                    #   for r in range(self.nmo):
                    #       for l in range(self.n_occupied):
                    #           a = sigma_total4[i][r][l]-self.sigma_total5[i][r][l]
                    #           if np.absolute(a) > 1e-14: print (i,r,l,a, "large error")
                    

                    #print("step from scipy solver", x)
                    #print(np.shape(x))
                    #S = np.zeros((1, self.index_map_size))
                    #S = H_op.matvec(davidson_step.T)
                    #print(S)
                    #test_g = np.einsum("pq,q->p", H_lambda,x)
                    #test_g2 = np.einsum("pq,q->p", H_lambda,step_limit)
                    #print(test_g)
                    #print(test_g2)
                    #print(reduced_gradient)
                    if (np.linalg.norm(x) < trust_radius):
                        step = x
                        #product2 = np.dot(x, step2)
                        xy_square = np.dot(x, step2) * np.dot(x, step2)
                        x_square = np.dot(x, x) 
                        y_square = np.dot(step2, step2) 
                        delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                        #print(delta)
                        t1= (-2 * np.dot(x,step2) - np.sqrt(delta))/ (2*y_square)
                        t2= (-2 * np.dot(x,step2) + np.sqrt(delta))/ (2*y_square)
                        print("x^2, xy, y^2, t", x_square, np.dot(x, step2), y_square, t1)
                        adjusted_step = step + min(t1,t2) * step2
                        print("adjusted step norm", np.linalg.norm(adjusted_step))
                        step = adjusted_step
                        trust_radius_hard_case = np.linalg.norm(adjusted_step)
                        trust_radius = trust_radius_hard_case
                    else:
                        step = x/np.linalg.norm(x) * trust_radius

                    #step = step2/np.linalg.norm(step2) * trust_radius 


                step_norm = np.linalg.norm(step)
                

                ####alpha = 1  
                ####alpha_min = 1  
                ####alpha_max = 1  
                ####dim0 = self.index_map_size + 1
                ####step = np.zeros((self.index_map_size))
                ####step2 =np.zeros((self.index_map_size))
                ####step_norm2, scale = self.microiteration_step2(reduced_gradient, reduced_hessian, step2, alpha, dim0)
                ####print("scale", scale)
                ####if np.abs(scale) > 1e-4:
                ####    ##find (alpha_min, alpha_max)
                ####    while True: 
                ####        step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                ####        #print("step", step, flush=True)
                ####        if step_norm > trust_radius:
                ####            alpha_min = alpha
                ####            alpha = alpha * 10
                ####        else:
                ####            alpha_max = alpha
                ####            break
                ####    print("alpha range", alpha_min, alpha_max)
                ####    #bisection search
                ####    if alpha_max != 1:
                ####        while True:
                ####            #print(alpha_min, alpha_max)
                ####            alpha = 0.5 * (alpha_min + alpha_max)
                ####            step_norm = self.microiteration_step(reduced_gradient, reduced_hessian, step, alpha, dim0)
                ####            if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                ####                break
                ####            elif trust_radius - step_norm > 1e-2:
                ####                alpha_max = alpha
                ####            else:
                ####                alpha_min = alpha
                ####else:
                ####    print("PROBLEM!!!!!!!!!")
                ####    step = trust_radius * step2/step_norm2
                ####    step_norm = trust_radius
                start = timer()
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual,self.n_in_a))
                Rva = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s-self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s-self.n_occupied][l] = step[i]
                    else:
                        Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                #print("jnti",self.E_core)

                self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta) 
                second_order_energy_change = self.microiteration_exact_energy(self.U3, A, G1)
                #exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                #print("exact energy2", exact_energy2, flush = True)
                energy_change = zero_energy + second_order_energy_change - current_energy
                print("old energy", current_energy, "energy change", energy_change, "second order energy change", second_order_energy_change, flush = True)
                print("new energy", current_energy + energy_change, flush = True)
                



                #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G1, step)
                print("microiteration predicted energy", predicted_energy2, flush = True)
                end   = timer()
                print("build unitary matrix and recheck energy took", end - start)
           
                if microiteration == 0 and orbital_optimization_step == 0: 
                   convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                if energy_change < 0.0:
                     #restart = False
                     #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                     #predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                     #print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                     self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                     if microiteration == 0 and orbital_optimization_step == 0: 
                     #    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                         #convergence_threshold = 0.01 * gradient_norm
                         ###10/10/2024 comment out this part to use standard trust region method 
                         #if step_norm > 0.1: 
                         #    N_microiterations = 5
                         #    N_orbital_optimization_steps = 4
                         #elif step_norm <= 0.1 and step_norm > 0.01:
                         #    N_microiterations = 7 
                         #    N_orbital_optimization_steps = 3
                         ###end comment
                         print("number of microiteration", N_microiterations, flush = True)    
                         print("number of optimization steps", N_orbital_optimization_steps, flush = True)    
                     orbital_optimization_step += 1   
                     ratio = energy_change/predicted_energy2
                     print("compare model with actual energy change",ratio)
                     trust_radius = self.step_control(ratio, trust_radius)
                         

                     #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                     #print(np.shape(gradient_tilde), flush = True)
                     
                     start = timer() 
                     self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
                     end   = timer() 
                     #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
                     G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
                     print("build gradient took", end - start)

                     #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                     #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                     #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                     #reduced_hessian_diagonal = np.zeros(self.index_map_size)
                     #index_count1 = 0 
                     #for k in range(self.n_occupied):
                     #    for r in range(k+1,self.nmo):
                     #        if (k < self.n_in_a and r < self.n_in_a): continue
                     #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                     #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                     #        index_count1 += 1
                     start = timer() 
                     self.build_hessian_diagonal(self.U2, G, A_tilde2)
                     end   = timer() 
                     print("build hessian diagonal took", end - start)
                     #print("diagonal elements of the reduced hessian")   
                     #for i in range(self.index_map_size):
                     #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                     #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                     #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                     reduced_gradient = np.zeros(self.index_map_size)
                     index_count1 = 0 
                     np.set_printoptions(precision = 14)
                     for k in range(self.n_occupied):
                         for r in range(k+1,self.nmo):
                             if (k < self.n_in_a and r < self.n_in_a): continue
                             if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                             reduced_gradient[index_count1] = gradient_tilde[r][k]
                             #print(r,k,index_count1)
                             #index_count2 = 0 
                             #for l in range(self.n_occupied):
                             #    for s in range(l+1,self.nmo):
                             #        if (l < self.n_in_a and s < self.n_in_a): continue
                             #        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                             #        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                             #        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                             #        #print(r,k,s,l,index_count1,index_count2)
                             #        index_count2 += 1
                             index_count1 += 1
                     #print("reduced_gradient",reduced_gradient, flush = True)
            
                     mu1, w1 = np.linalg.eigh(reduced_hessian)
                     
                     active_twoeint[:,:,:,:] = 0.0 
                     active_fock_core[:,:] = 0.0 
                     d_cmo[:,:] = 0.0
                     start1 = timer()
                     self.microiteration_ci_integrals_transform(self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                     end1 = timer()
                     print("second order integral transformation took", end1 - start1)
                     count += 1
                     #active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                     #active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                     #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                     #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                     #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
                     #        self.Enuc + self.d_c + ci_dependent_energy)
                     #print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
                     #print("gfhgy",
                     #    "{:20.12f}".format(sum_energy),
                     #    "{:20.12f}".format(active_one_e_energy),
                     #    "{:20.12f}".format(active_two_e_energy),
                     #    "{:20.12f}".format(self.E_core2),
                     #    "{:20.12f}".format(active_one_pe_energy),
                     #    "{:20.12f}".format(ci_dependent_energy),
                     #    "{:20.12f}".format(self.Enuc),
                     #    flush = True
                     #)
                     current_energy = zero_energy + second_order_energy_change
              
                else:
                    restart = False
                    guess_vector = trial_vector
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush = True)
                    print("new trust radius", trust_radius, flush = True)
            #if convergence == 1:
            #    #active_twoeint[:,:,:,:] = self.active_twoeint[:,:,:,:] 
            #    #active_fock_core[:,:] = self.active_fock_core[:,:] 
            #    #d_cmo[:,:] = self.d_cmo[:,:]

            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    #temp8 = np.zeros((self.nmo, self.nmo))
            #    #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
            #    #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    #temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
            #    #self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    break


            if convergence == 1 and count == 0:
                active_twoeint[:,:,:,:] = self.active_twoeint[:,:,:,:] 
                active_fock_core[:,:] = self.active_fock_core[:,:] 
                d_cmo[:,:] = self.d_cmo[:,:]

                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                #temp8 = np.zeros((self.nmo, self.nmo))
                #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                #self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #break

            start = timer() 
            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            #print("recheck energy", flush = True)
            #active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            #active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy)
            #print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            #print("gfhgy",
            #    "{:20.12f}".format(sum_energy),
            #    "{:20.12f}".format(active_one_e_energy),
            #    "{:20.12f}".format(active_two_e_energy),
            #    "{:20.12f}".format(self.E_core2),
            #    "{:20.12f}".format(active_one_pe_energy),
            #    "{:20.12f}".format(ci_dependent_energy),
            #    "{:20.12f}".format(self.Enuc),
            #    flush = True
            #)
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    self.E_core2, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = self.E_core2
            self.constint[8] = 0 
            eigenvals = np.zeros((self.davidson_roots))
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            #print("heyhey5", eigenvecs)
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            end   = timer() 
            print("CI step took", end - start)


            #print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print("microiteration",microiteration + 1, "current average energy", avg_energy, flush = True)
            current_energy = avg_energy
            
            start = timer() 
            self.build_state_avarage_rdms(eigenvecs)
            end   = timer() 
            print("building RDM took", end - start)

            print("current gradient_norm and residual", gradient_norm, current_residual)
            print("current convergence_threshold", convergence_threshold)
            total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2)) 
            if total_norm < convergence_threshold: 
                print("total norm", total_norm, flush = True)
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                #temp8 = np.zeros((self.nmo, self.nmo))
                #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                #self.d_cmo[:,:] = d_cmo[:,:]
                #print(eigenvecs)
                #print("u2i",self.U2)
                print("microiteration converged! (small total norm)", flush = True)
                break 


            microiteration += 1





    def ah_orbital_optimization(self, eigenvecs, c_get_roots):
        self.H1 = copy.deepcopy(self.H_spatial2)
        self.d_cmo1 = copy.deepcopy(self.d_cmo)
        #print("E_core", self.E_core)
        self.U2 = np.eye(self.nmo)
        self.U_total = np.eye(self.nmo)
        U_temp = np.eye(self.nmo)
        convergence_threshold = 1e-5
        trust_radius = 0.4   
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #A2 = np.zeros((rot_dim, rot_dim))
        #G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size+1))
        #convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        macroiteration_energy_initial = 0.0
        macroiteration_energy_current = 0.0
        N_orbital_optimization_steps = 4
        N_microiterations = 25  
        microiteration = 0
        while(microiteration < N_microiterations):
            print("\n")
            print("\n")
            print("MICROITERATION", microiteration+1,flush = True)

        #while(microiteration < 2):
            trust_radius = 0.35
            A[:,:] = 0.0
            G[:,:,:,:] = 0.0
            start = timer()
            self.build_intermediates(eigenvecs, A, G, True)
            end   = timer()
            print("build intermediates took", end - start)
            #A2[:,:] = 0.0
            #G2[:,:,:,:] = 0.0
            #self.build_intermediates2(eigenvecs, A2, G2, True)
            #print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            #print(old_energy, current_energy)
            #print("initial convergence threshold", convergence_threshold)
            #if (np.abs(current_energy - old_energy) < 0.01 * convergence_threshold) and microiteration >=2:
            ##if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
            #    print("microiteration converged (small energy change)")
            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    break

            start = timer()
            start1 = timer()
            macroiteration_energy_initial = macroiteration_energy_current 
            active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
            fock_core = copy.deepcopy(self.H_spatial2) 
            fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
            fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
            
            E_core = 0.0  
            E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            E_core += np.einsum("jj->", fock_core[:self.n_in_a,:self.n_in_a]) 


            #print(eigenvecs)
            active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
            active_fock_core[:,:] = fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
            active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
            active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
            active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(self.d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            macroiteration_energy_current = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core +
                    self.Enuc + self.d_c + ci_dependent_energy)
            print("macroiteration_energy_current",
                "{:20.12f}".format(macroiteration_energy_current),
                "{:20.12f}".format(active_one_e_energy),
                "{:20.12f}".format(active_two_e_energy),
                "{:20.12f}".format(E_core),
                "{:20.12f}".format(active_one_pe_energy),
                "{:20.12f}".format(self.Enuc),
            )


            end   = timer()
            print("check initial energy took", end - start)
           



            self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            #print(np.shape(gradient_tilde), flush = True)
            
            start = timer() 
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            end   = timer() 
            #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            

            
            print("build gradient took", end - start)
            hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()    
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end   = timer() 
            print("build hessian diagonal took", end - start)
            reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            reduced_gradient = np.zeros(self.index_map_size)
            index_count1 = 0 
            for k in range(self.n_occupied):
                for r in range(k+1,self.nmo):
                    if (k < self.n_in_a and r < self.n_in_a): continue
                    if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    #print(r,k,index_count1)
                    index_count2 = 0 
                    for l in range(self.n_occupied):
                        for s in range(l+1,self.nmo):
                            if (l < self.n_in_a and s < self.n_in_a): continue
                            if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                            #if (k >= self.n_occupied and r >= self.n_occupied): continue
                            reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                            #print(r,k,s,l,index_count1,index_count2)
                            index_count2 += 1
                    index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            np.set_printoptions(precision = 14)
            mu1, w1 = np.linalg.eigh(reduced_hessian)
            print("eigenvalue of the reduced hessian", mu1)
            print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]))
            print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]))
            H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            print("norm of critical step", np.linalg.norm(step_limit))
            print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
            gradient_norm = np.linalg.norm(reduced_gradient)
            print("gradient norm", gradient_norm)
            print("norm of reduced hessian", np.linalg.norm(reduced_hessian))
            ##if microiteration ==0:  
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            #if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            print(macroiteration_energy_initial, macroiteration_energy_current)
            print("initial convergence threshold", convergence_threshold)
            if (np.abs(macroiteration_energy_current - macroiteration_energy_initial) < 0.0001 * convergence_threshold) and microiteration >=2:
            #if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)")
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                break


            orbital_optimization_step = 0 
            restart = False
            count = 0
            current_energy = macroiteration_energy_current 
            hard_case = 0
            reduce_step = 0
            first_component = np.zeros(self.index_map_size)
            second_component = np.zeros(self.index_map_size)
            critical_step_norm = 0
            while(orbital_optimization_step < N_orbital_optimization_steps):
                print("\n")
                old_energy = current_energy
                print("Microiteration", microiteration + 1, "orbital optimization step", orbital_optimization_step + 1, flush=True)
                print("current energy", current_energy)
                print("reduced_gradient", reduced_gradient)
                gradient_norm = np.linalg.norm(reduced_gradient)
                print("gradient norm", gradient_norm, flush = True)
                print("convergence_threshold", convergence_threshold, flush = True)    
                
                #if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                #if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                #if (gradient_norm < 1e-5 and microiteration > 0):
                if (gradient_norm < 0.001 * convergence_threshold and microiteration > 0):
                    convergence = 1
                    #print("Microiteration converged (small gradient norm)")
                    break
                if (gradient_norm < 1e-7):
                    convergence = 1
                    break
                if microiteration == 2 and orbital_optimization_step == 0:
                    np.savetxt("gradient.out", reduced_gradient)
                    np.savetxt("hessian.out", reduced_hessian)

                #########gradient_tilde = np.zeros((rot_dim, self.n_occupied))
                #########hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                #########self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                #########print(np.shape(gradient_tilde), flush = True)
               
                #########gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
                #########A_tilde2 = np.zeros((rot_dim, rot_dim))
                #########self.build_gradient(self.U2, A, G, gradient_tilde2, A_tilde2, True)

                #########hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                #########hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                #########hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                #########reduced_hessian_diagonal = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                #########        index_count1 += 1
                #########self.build_hessian_diagonal(self.U2, G, A_tilde2)
                ##########print("diagonal elements of the non redundant hessian")   
                ##########for a in range(self.nmo):
                ##########    for i in range(self.n_occupied):
                ##########        aa = hessian_diagonal3[a][i] - self.hessian_diagonal[a][i]
                ##########        if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########print("diagonal elements of the reduced hessian")   
                #########for i in range(self.index_map_size):
                #########    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                #########    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                #########reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                #########reduced_gradient = np.zeros(self.index_map_size)
                #########index_count1 = 0 
                #########for k in range(self.n_occupied):
                #########    for r in range(k+1,self.nmo):
                #########        if (k < self.n_in_a and r < self.n_in_a): continue
                #########        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                #########        reduced_gradient[index_count1] = gradient_tilde[r][k]
                #########        #print(r,k,index_count1)
                #########        index_count2 = 0 
                #########        for l in range(self.n_occupied):
                #########            for s in range(l+1,self.nmo):
                #########                if (l < self.n_in_a and s < self.n_in_a): continue
                #########                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #########                #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #########                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                #########                #print(r,k,s,l,index_count1,index_count2)
                #########                index_count2 += 1
                #########        index_count1 += 1
                #########print("reduced_gradient",reduced_gradient, flush = True)
               

                #new augmented hessian
                beta = 0
                dim00 = self.index_map_size + 1
                augmented_hessian9= np.zeros((dim00, dim00))
                w9= np.zeros((dim00, dim00))

                step = np.zeros(self.index_map_size)
                step10_norm = 0.0
                #print("w",w)
                #count10 = 0
                #while np.abs((step10_norm - trust_radius)/trust_radius) > 1e-3:
                #    print("iteration", count10, "beta", beta)
                #    augmented_hessian9[:,:] = 0.0
                #    augmented_hessian9[0,0] = beta
                #    augmented_hessian9[0,1:] = reduced_gradient
                #    augmented_hessian9[1:,0] = reduced_gradient.T
                #    augmented_hessian9[1:,1:] = reduced_hessian

                #    mu9, w9 = np.linalg.eigh(augmented_hessian9)
                #    print("eig",mu9)
                #    idx = mu9.argsort()[:dim00]
                #    scale9 = w9[0][0]
                #    print("scale",scale9)
                #    if np.abs(scale9) < 1e-15:
                #        break
                #    step10 = w9[1:,0]/scale9
                #    step10_norm = np.linalg.norm(step10)
                #    if count10 == 0 and step10_norm < trust_radius: break
                #    print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                #    print(mu9, flush=True)
                #    print("step10 norm", step10_norm, flush=True)
                #    beta = beta + (beta - mu9[0])/step10_norm * (trust_radius - step10_norm)/trust_radius * (trust_radius + 1.0/step10_norm)
                #    count10 +=1 
                #    if count10 == 20: break
                ##w[:,0] = w[:,0]/scale
                ##print(w)
                #step9 = w9[1:,0]
                if np.linalg.norm(reduced_gradient) > 1e-4:
                    if hard_case == 1 and reduce_step == 1 and np.linalg.norm(first_component) < trust_radius:
                        print(np.linalg.norm(first_component))
                        xy_square = np.dot(first_component, second_component) * np.dot(first_component, second_component)
                        x_square = np.dot(first_component, first_component) 
                        y_square = np.dot(second_component, second_component) 
                        delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                        #print(delta)
                        t1= (-2 * np.dot(first_component,second_component) - np.sqrt(delta))/ (2*y_square)
                        t2= (-2 * np.dot(first_component,second_component) + np.sqrt(delta))/ (2*y_square)
                        print("x^2, xy, y^2, t", x_square, np.dot(first_component,second_component), y_square, t1)
                        adjusted_step = first_component + min(t1,t2) * second_component
                        print("adjusted step norm", np.linalg.norm(adjusted_step))
                        step = adjusted_step   
                    else:
                        mu9 = np.zeros(dim00)
                        beta0 = 0
                        both_roots_normalization = 0
                        v1 = 0
                        v2 = 0
                        u1 = np.zeros(self.index_map_size)
                        u2 = np.zeros(self.index_map_size)
                        x1 = np.zeros(self.index_map_size)
                        x2 = np.zeros(self.index_map_size)
                        phi1 = 0
                        phi2 = 0

                        lambda1 = 0
                        lambda2 = 0
                        lambda_c = 0
                        idx = self.reduced_hessian_diagonal.argsort()
                        delta_u = self.reduced_hessian_diagonal[idx[0]]
                        delta_l = 0
                        alpha_u = delta_u + np.linalg.norm(reduced_gradient) * trust_radius
                        alpha_l = 0
                        count10 = 0
                        #beta = min(0, alpha_u)
                        beta = alpha_u
                        print("trust radius", trust_radius)
                        while np.abs((step10_norm - trust_radius)/trust_radius) > 1e-3:
                            print("****************************") 
                            print("iteration", count10, "beta", beta)
                            beta0 = beta 
                            mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, beta, dim00)
                            #augmented_hessian9[:,:] = 0.0
                            #augmented_hessian9[0,0] = beta
                            #augmented_hessian9[0,1:] = reduced_gradient
                            #augmented_hessian9[1:,0] = reduced_gradient.T
                            #augmented_hessian9[1:,1:] = reduced_hessian
                            #mu9, w9 = np.linalg.eigh(augmented_hessian9)
                            #print("eig",mu9)
                            if count10 == 0:
                                delta_l = mu9[0]
                                alpha_l = delta_l - np.linalg.norm(reduced_gradient)/trust_radius
                            print("ALPHA range", alpha_l, alpha_u)
                            #idx = mu9.argsort()[:dim00]
                            #print(idx)
                            #w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                            #w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                            v1 = w9[0,0]/np.linalg.norm(w9[:,0])
                            v2 = w9[0,1]/np.linalg.norm(w9[:,1])
                            u1[:] = w9[1:,0]/np.linalg.norm(w9[:,0])
                            u2[:] = w9[1:,1]/np.linalg.norm(w9[:,1])
                            aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                            bb1 = np.sqrt(1 - v1 * v1)
                            aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2) 
                            bb2 =  np.sqrt(1 - v2 * v2)
                            print("aa1, bb1", aa1, bb1, "aa2, bb2", aa2, bb2)
                            if mu9[0] > -1e-8 and np.linalg.norm(u1) < trust_radius * v1: 
                                hard_case = 2
                                print("use newton step")
                                break


                            epsilon_v = 1e-4
                            print (aa1 <= epsilon_v * bb1) 
                            print (aa2 <= epsilon_v * bb2) 
                            alpha = beta
                            while (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2) and np.abs(alpha_u-alpha_l) > max(1e-15, 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))):
                                alpha_u = alpha
                                alpha = (alpha_l + alpha_u)/2
                                w9[:,:] = 0.0
                                mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, alpha, dim00)
                                #w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                                #w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                                v1 = w9[0,0]/np.linalg.norm(w9[:,0])
                                v2 = w9[0,1]/np.linalg.norm(w9[:,1])
                                u1[:] = w9[1:,0]/np.linalg.norm(w9[:,0])
                                u2[:] = w9[1:,1]/np.linalg.norm(w9[:,1])
                                aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                bb1 = np.sqrt(1 - v1 * v1)
                                aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2) 
                                bb2 =  np.sqrt(1 - v2 * v2)
                                print(alpha, aa1 <= epsilon_v * bb1, aa2 <= epsilon_v * bb2, mu9[0], mu9[1], bb1,bb2)
                            if (aa1 > epsilon_v * bb1) and (aa2 > epsilon_v * bb2):
                                both_roots_normalization = 2
                            beta = alpha
                            #update delta_u
                            temptemp = np.dot(reduced_hessian, u1)
                            temptemp2 = np.dot(u1, temptemp)

                            delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1)/np.dot(u1,u1)
                            delta_u = min(delta_u, temptemp2/np.dot(u1,u1))
                            print("new delta_u", delta_u)
                            ####scale9 = w9[0][0]
                            ####print("scale",scale9)
                            ####if np.abs(scale9) < 1e-15:
                            ####    break
                            #assign (k-1) and k-iteration values
                            lambda1 = lambda2
                            x1 = x2
                            phi1 = phi2
                            print("two roots", mu9[0], mu9[1])
                            if (aa1 > epsilon_v * bb1):
                                step10 = u1/v1
                                second_component[:] = u2
                                lambda2 = mu9[0]
                                print("norm from first root", np.linalg.norm(step10))
                                if aa2 > epsilon_v * bb2:
                                    second_component[:] = u2/v2
                                    print("second root norm", np.linalg.norm(second_component))
                                if np.linalg.norm(step10) < trust_radius: alpha_l = beta
                                if np.linalg.norm(step10) > trust_radius: alpha_u = beta
                            else:
                                step10 = u2/v2
                                second_component[:] = u1
                                lambda2 = mu9[1]
                                print("norm from second root", np.linalg.norm(step10))
                                alpha_u = beta
                            print("NEW ALPHA range", alpha_l, alpha_u)
                            step10_norm = np.linalg.norm(step10)
                            x2 = step10
                            if mu9[0] > -1e-8 and np.linalg.norm(u1) < trust_radius * v1: 
                                hard_case = 2
                                print("use newton step")
                                break
                            phi2 = -np.dot(reduced_gradient, x2)
                            #print(step10)
                            #print(x1)
                            #print(x2)
                            #if count10 == 0 and step10_norm < trust_radius: break
                            #print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                            #print(mu9, flush=True)
                            print("step10 norm", step10_norm, "current beta", beta, flush=True)
                            x1_norm = np.linalg.norm(x1)
                            x2_norm = np.linalg.norm(x2)
                            phi2_p = np.dot(x2, x2)
                            phi1_p = np.dot(x1, x1)
                            print("phi1", phi1, "phi1_p", phi1_p, "phi2", phi2, "phi2_p", phi2_p)
                            if np.abs((step10_norm - trust_radius)/trust_radius) <= 1e-3:
                                step = step10
                                hard_case = 0 
                                break
                            print("check orthogonality of the first two roots")
                            vv1 = w9[0,0]
                            vv2 = w9[0,1]
                            uu1 = w9[1:,0]
                            uu2 = w9[1:,1]
                            #check if the quasi-optimal condition is satisfied
                            print("quasi-optimal condition", (1+ trust_radius * trust_radius) * (vv1 * vv1 + vv2 * vv2), flush = True)
                            qs = (1+ trust_radius * trust_radius) * (vv1 * vv1 + vv2 * vv2)
                            print(np.dot(w9[:,:2].T, w9[:,:2]))

                            #construct quasi-optimal step
                            epsilon_hc = 1e-6
                            eta = epsilon_hc/(1-epsilon_hc)
                            tau1 = 1
                            tau2 = 1
                            if qs > 1.0:
                                tau1 = (vv1 - vv2 * np.sqrt(qs -1))/(vv1 * vv1 + vv2 * vv2)/np.sqrt(1 + trust_radius * trust_radius)
                                tau2 = (vv2 + vv1 * np.sqrt(qs -1))/(vv1 * vv1 + vv2 * vv2)/np.sqrt(1 + trust_radius * trust_radius)
                            elif np.abs(qs - 1.0) < 1e-15:
                                tau1 = vv1/np.sqrt(vv1 * vv1 + vv2 * vv2)
                                tau2 = vv2/np.sqrt(vv1 * vv1 + vv2 * vv2)
                            x_tilde = (tau1 * uu1 + tau2 * uu2) / (tau1 * vv1 + tau2 * vv2)

                            lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                            psi_tilde = 0.5 * self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G1, x_tilde)
                            if qs > 1.0 or np.abs(qs - 1.0) < 1e-15:
                                print("x_tilde norm1", np.linalg.norm(x_tilde), 2.0 * psi_tilde)
                                if (mu9[1] - mu9[0]) * tau2 * tau2 * (1 + trust_radius * trust_radius) < -2.0 * eta * psi_tilde:
                                    print("find quasi-optimal solution1", flush = True)
                                    hard_case = 3
                                    break
                                else:
                                    if qs > 1.0:
                                        tau1 = (vv1 + vv2 * np.sqrt(qs -1))/(vv1 * vv1 + vv2 * vv2)/np.sqrt(1 + trust_radius * trust_radius)
                                        tau2 = (vv2 - vv1 * np.sqrt(qs -1))/(vv1 * vv1 + vv2 * vv2)/np.sqrt(1 + trust_radius * trust_radius)
                                        x_tilde = (tau1 * uu1 + tau2 * uu2) / (tau1 * vv1 + tau2 * vv2)
                                        lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                                        psi_tilde = 0.5 * self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G1, x_tilde)
                                        print("x_tilde norm2", np.linalg.norm(x_tilde), 2.0 * psi_tilde, flush = True)
                                        if (mu9[1] - mu9[0]) * tau2 * tau2 * (1 + trust_radius * trust_radius) < -2.0 * eta * psi_tilde:
                                            print("find quasi-optimal solution2", flush = True)
                                            hard_case = 3
                                            break
                            if np.abs(alpha_u-alpha_l) <= max(1e-15, 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))):
                                print("interval too small")
                                if x2_norm < trust_radius: 
                                    hard_case = 1
                                break


                            if count10 == 0 or (np.abs(x2_norm - x1_norm) <= 1e-15):
                                beta = beta + (beta - lambda2)/x2_norm * (trust_radius - x2_norm)/trust_radius * (trust_radius + 1.0/x2_norm)
                            else:
                                print("denominator", x2_norm-x1_norm)
                                if (np.abs(x2_norm - x1_norm) > 1e-15):
                                    lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                else:
                                    lambda_c = delta_u
                                if lambda_c > delta_u:
                                    print("need to safeguard delta_u", lambda_c, delta_u)
                                    lambda_c = delta_u
                                #lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                #print(lambda_c)
                                omega_k = (lambda2 - lambda_c)/(lambda2 - lambda1)
                                ratio1 = x1_norm * x2_norm * (x2_norm - x1_norm)/(omega_k * x2_norm + (1-omega_k) * x1_norm)
                                ratio2 = (lambda1 - lambda_c) * (lambda2 - lambda_c) / (lambda2 - lambda1)
                                beta = lambda_c + omega_k *phi1 + (1-omega_k) * phi2 + ratio1 * ratio2
                            print("beta after interpolation", beta)
                            #safeguard alpha
                            if beta < alpha_l or beta > alpha_u:
                                print("need to safeguard alpha")
                                if count10 == 0: 
                                    beta = delta_u + phi2 + phi2_p * (delta_u - lambda2)
                                elif x2_norm < x1_norm:
                                    print("use phi2")
                                    beta = delta_u + phi2 + phi2_p * (delta_u - lambda2)
                                else:
                                    print("use phi1")
                                    beta = delta_u + phi1 + phi1_p * (delta_u - lambda1)
                                if beta < alpha_l or beta > alpha_u:
                                    print("rwttrye")
                                    beta = (alpha_l + alpha_u)/2 
                            count10 +=1
                            #if np.abs(beta -beta0) <= 1e-12: 
                            #    print("possible convergence")
                            #    print(beta, beta0)
                            #    print(x2_norm)

                            #    if x2_norm < trust_radius:
                            #        print("11111111")
                            #        hard_case = 1
                            #    elif x2_norm > trust_radius and both_roots_normalization == 2:    
                            #        print("22222222")
                            #        hard_case = 1

                            #    break
                            if count10 == 50: break
                        print("check hard case", hard_case)
                        if hard_case == 1:
                            first_component[:] = x2[:]
                            if x2_norm > trust_radius:
                                temp_vector = np.zeros(self.index_map_size)
                                temp_vector[:] = second_component[:]
                                second_component[:] = first_component[:]
                                first_component[:] = temp_vector[:]
                            print(first_component)
                            print(second_component)
                            #product2 = np.dot(x, step2)
                            print("norm of current root", np.linalg.norm(first_component))
                            xy_square = np.dot(first_component, second_component) * np.dot(first_component, second_component)
                            x_square = np.dot(first_component, first_component) 
                            y_square = np.dot(second_component, second_component) 
                            delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                            #print(delta)
                            t1= (-2 * np.dot(first_component,second_component) - np.sqrt(delta))/ (2*y_square)
                            t2= (-2 * np.dot(first_component,second_component) + np.sqrt(delta))/ (2*y_square)
                            print("x^2, xy, y^2, t", x_square, np.dot(first_component,second_component), y_square, t1)
                            adjusted_step = first_component + min(t1,t2) * second_component
                            print("adjusted step norm", np.linalg.norm(adjusted_step))
                            step = adjusted_step

                        if hard_case == 2:
                            Q = np.zeros((1, self.index_map_size))
                            H1_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv2(self.U2, A_tilde2, G1, Q, 1,0, 0)) 
                            x, exitCode= minres(H1_op, -reduced_gradient, tol = 1e-6)
                            print("exitcode", exitCode)
                            step = x

                        if hard_case == 3:
                            step = x_tilde

                else:
                    print("gradient is small, use Newton step")
                    Q = np.zeros((1, self.index_map_size))
                    H1_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv2(self.U2, A_tilde2, G1, Q, 1,0, 0)) 
                    x, exitCode= minres(H1_op, -reduced_gradient, tol = 1e-6)
                    print("exitcode", exitCode)
                    hard_case = 2
                    step = x
                    print(step)
                #w[:,0] = w[:,0]/scale
                #print(w)
                step9 = w9[1:,0]




                ###############alpha = 1  
                ###############alpha_min = 1  
                ###############alpha_max = 1  
                ###############dim0 = self.index_map_size + 1
                ###############step3= np.zeros((self.index_map_size))
                ###############step_norm3 = 0
                ###############step2 =np.zeros((self.index_map_size))
                ###############step_norm2, scale = self.microiteration_step2(reduced_gradient, reduced_hessian, step2, alpha, dim0)
                ###############print("scale", scale, flush = True)
                ###############if np.abs(scale) > 1e-4:
                ###############    ##find (alpha_min, alpha_max)
                ###############    while True: 
                ###############        step_norm3 = self.microiteration_step(reduced_gradient, reduced_hessian, step3, alpha, dim0)
                ###############        #print("step", step, flush=True)
                ###############        if step_norm3 > trust_radius:
                ###############            alpha_min = alpha
                ###############            alpha = alpha * 10
                ###############        else:
                ###############            alpha_max = alpha
                ###############            break
                ###############    print("alpha range", alpha_min, alpha_max, flush = True)
                ###############    #bisection search
                ###############    count1 =0
                ###############    if alpha_max != 1:
                ###############        while True:
                ###############            #print(alpha_min, alpha_max)
                ###############            alpha = 0.5 * (alpha_min + alpha_max)
                ###############            step_norm3 = self.microiteration_step(reduced_gradient, reduced_hessian, step3, alpha, dim0)
                ###############            if np.abs((step_norm3 - trust_radius)/trust_radius) <= 1e-3:
                ###############            #if trust_radius - step_norm3 <= 1e-2 and trust_radius - step_norm3 >= 0.0:
                ###############                break
                ###############            elif (trust_radius - step_norm3)/trust_radius >  1e-3:
                ###############            #elif trust_radius - step_norm3 > 1e-2:
                ###############                alpha_max = alpha
                ###############            else:
                ###############                alpha_min = alpha
                ###############            count1 +=1
                ###############            if count1 ==150: break
                ###############else:
                ###############    print("PROBLEM!!!!!!!!!", flush = True)
                ###############    step3 = trust_radius * step2/step_norm2
                ###############    step_norm3 = trust_radius



                ###############self.mu_min = 0.0
                ################start = timer() 
                ################trial_vector, hard_case = self.Davidson_augmented_hessian_solve(self.U2, A_tilde2, G, G1, self.reduced_hessian_diagonal, reduced_gradient,
                ################        davidson_step, trust_radius, guess_vector, restart)
                ################end   = timer() 
                ################print("solving augmented hessian took", end - start)
                ###############start = timer() 
                ###############trial_vector, hard_case = self.Davidson_augmented_hessian_solve2(self.U2, A_tilde2, G, G1, self.reduced_hessian_diagonal, reduced_gradient,
                ###############        davidson_step, trust_radius, guess_vector, restart)
                ###############end   = timer() 
                ###############print("solving augmented hessian2 took", end - start)


                ###############print("hard case", hard_case) 
                ###############print("smallest eigenvalue of the augmented hessian", self.mu_min)
                ###############step2 = davidson_step.flatten()
                ###############product = np.einsum("p,p->",reduced_gradient, step2)
                ###############np.set_printoptions(precision = 14)  
                ###############print("check dot product of gradient and first eigenvector", product)
                ###############trust_radius_hard_case = trust_radius 
                ###############if hard_case == 0:
                ###############    step = step2
                ###############else:
                ###############    ####print("eigenvalue of the non-redundant hessian", mu1, flush = True)
                ###############    ####product = np.einsum("p,p->",reduced_gradient, w1[:,0])
                ###############    ####H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                ###############    ####step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                ###############    ####H_lambda_inverse = np.linalg.pinv(H_lambda)
                ###############    ####print("conditional number from product", np.linalg.norm(H_lambda) * np.linalg.norm(H_lambda_inverse))
                ###############    ####print("step_limit", step_limit)
                ###############    ####print("limit of step norm", np.linalg.norm(step_limit))
                ###############    ####print("check dot product of gradient and first eigenvector", product)
                ###############    Q = np.zeros((1, self.index_map_size))
                ###############    #H_diag = self.reduced_hessian_diagonal - mu1[0] 
                ###############    #Q[0,:] = np.divide(-reduced_gradient, H_diag, out=np.zeros_like(H_diag), where=H_diag!=0)
                ###############    #S = np.zeros_like(Q)
                ###############    
                ###############    #H_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv(self.U2, A_tilde2, G, Q, 1,0, self.mu_min)) 
                ###############    H1_op = LinearOperator((self.index_map_size, self.index_map_size), matvec = lambda Q:  self.mv2(self.U2, A_tilde2, G1, Q, 1,0, self.mu_min)) 
                ###############    #S=H_op.matvec(Q.T)
                ###############    #print(S)
                ###############    #S2 = np.einsum("pq,qr->pr", H_lambda, Q.T)
                ###############    #print(S2)
                ###############    #print("diagonal element of the reduced hessian", np.diagonal(reduced_hessian))   
                ###############    x, exitCode= minres(H1_op, -reduced_gradient, tol = 1e-6)
                ###############    #x, istop, itn, normr, normar, norma, conda, normx = lsmr(H_op, -reduced_gradient)[:8]
                ###############    #print("reo0", x, istop, itn, normr, normx, conda)
                ###############    print("exitcode", exitCode)
                ###############    print("step norm from scipy solver", np.linalg.norm(x))
                ###############    #Q1 = np.random.rand(1, self.index_map_size)
                ###############    #start = timer()
                ###############    #S1=H_op.matvec(Q1.T)
                ###############    #end = timer()
                ###############    #print("build sigma with numpy took", end - start)
                ###############    #start = timer()
                ###############    #S2=H1_op.matvec(Q1.T)
                ###############    #end = timer()
                ###############    #print("build sigma with numba took", end - start)
                ###############    #print("rg8l", np.allclose(S1,S2, rtol=1e-14,atol=1e-14))
                ###############    #for j in range(self.index_map_size):
                ###############    #    abc = S1[j] -S2[j]
                ###############    #    if np.abs(abc) > 1e-14: print(j, abc)
                ###############    #for i in range(1):
                ###############    #   for r in range(self.nmo):
                ###############    #       for l in range(self.n_occupied):
                ###############    #           a = sigma_total4[i][r][l]-self.sigma_total5[i][r][l]
                ###############    #           if np.absolute(a) > 1e-14: print (i,r,l,a, "large error")
                ###############    

                ###############    #print("step from scipy solver", x)
                ###############    #print(np.shape(x))
                ###############    #S = np.zeros((1, self.index_map_size))
                ###############    #S = H_op.matvec(davidson_step.T)
                ###############    #print(S)
                ###############    #test_g = np.einsum("pq,q->p", H_lambda,x)
                ###############    #test_g2 = np.einsum("pq,q->p", H_lambda,step_limit)
                ###############    #print(test_g)
                ###############    #print(test_g2)
                ###############    #print(reduced_gradient)
                ###############    if (np.linalg.norm(x) < trust_radius):
                ###############        step = x
                ###############        #product2 = np.dot(x, step2)
                ###############        xy_square = np.dot(x, step2) * np.dot(x, step2)
                ###############        x_square = np.dot(x, x) 
                ###############        y_square = np.dot(step2, step2) 
                ###############        delta = 4 * xy_square - 4 * y_square * (x_square - trust_radius * trust_radius)
                ###############        #print(delta)
                ###############        t1= (-2 * np.dot(x,step2) - np.sqrt(delta))/ (2*y_square)
                ###############        t2= (-2 * np.dot(x,step2) + np.sqrt(delta))/ (2*y_square)
                ###############        print("x^2, xy, y^2, t", x_square, np.dot(x, step2), y_square, t1)
                ###############        adjusted_step = step + min(t1,t2) * step2
                ###############        print("adjusted step norm", np.linalg.norm(adjusted_step))
                ###############        step = adjusted_step
                ###############        trust_radius_hard_case = np.linalg.norm(adjusted_step)
                ###############        trust_radius = trust_radius_hard_case
                ###############    else:
                ###############        step = x/np.linalg.norm(x) * trust_radius

                ###############    #step = step2/np.linalg.norm(step2) * trust_radius 


                step_norm = np.linalg.norm(step)
                

                
                start = timer()
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual,self.n_in_a))
                Rva = np.zeros((self.n_virtual,self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0] 
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s-self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s-self.n_occupied][l] = step[i]
                    else:
                        Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                #print("jnti",self.E_core)

                U_temp = np.einsum("pq,qs->ps", self.U_total, self.U_delta) 
                J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

                K_temp = np.ascontiguousarray(K_temp)
                c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
                #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
                h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
                temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
                d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)


                new_energy = self.rdm_exact_energy(J_temp, K_temp, h1_temp, d_cmo_temp, eigenvecs)
                energy_change = new_energy - old_energy
                print("old energy", old_energy, "new energy", new_energy, "energy change", energy_change, flush = True)
                














                #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G1, step)
                print("microiteration predicted energy", predicted_energy2, flush = True)
                end   = timer()
                print("build unitary matrix and recheck energy took", end - start)
           
                if microiteration == 0 and orbital_optimization_step == 0: 
                   convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                if energy_change < 0.0 or hard_case == 2:
                     #restart = False
                     #predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                     #predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                     #print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                     self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U_delta)
                     if microiteration == 0 and orbital_optimization_step == 0: 
                     #    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                         #convergence_threshold = 0.01 * gradient_norm
                         ###10/10/2024 comment out this part to use standard trust region method 
                         #if step_norm > 0.1: 
                         #    N_microiterations = 5
                         #    N_orbital_optimization_steps = 4
                         #elif step_norm <= 0.1 and step_norm > 0.01:
                         #    N_microiterations = 7 
                         #    N_orbital_optimization_steps = 3
                         ###end comment
                         print("number of microiteration", N_microiterations, flush = True)    
                         print("number of optimization steps", N_orbital_optimization_steps, flush = True)    
                     orbital_optimization_step += 1   
                     ratio = energy_change/predicted_energy2
                     print("compare model with actual energy change",ratio)
                     trust_radius = self.step_control(ratio, trust_radius)
                         

                     #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                     #print(np.shape(gradient_tilde), flush = True)
                     
                     start = timer()
                     self.J[:,:,:,:] = J_temp
                     self.K[:,:,:,:] = K_temp
                     self.H_spatial2[:,:] = h1_temp
                     self.d_cmo[:,:] = d_cmo_temp
                     #self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                     temp_energy = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs)
                     print("check energy again",temp_energy)
                     A[:,:] = 0
                     A_tilde2[:,:] = 0
                     G[:,:,:,:] = 0
                     gradient_tilde[:] = 0
                     hessian_tilde[:,:,:,:] = 0
                     self.build_intermediates(eigenvecs, A, G, True)
                     self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                     self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
                     end   = timer() 
                     #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
                     G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
                     print("build gradient took", end - start)

                     hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                     hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                     #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                     #reduced_hessian_diagonal = np.zeros(self.index_map_size)
                     #index_count1 = 0 
                     #for k in range(self.n_occupied):
                     #    for r in range(k+1,self.nmo):
                     #        if (k < self.n_in_a and r < self.n_in_a): continue
                     #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                     #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                     #        index_count1 += 1
                     start = timer() 
                     self.build_hessian_diagonal(self.U2, G, A_tilde2)
                     end   = timer() 
                     print("build hessian diagonal took", end - start)
                     #print("diagonal elements of the reduced hessian")   
                     #for i in range(self.index_map_size):
                     #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                     #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                     #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                     reduced_gradient[:] = 0 
                     reduced_hessian[:,:] = 0 
                     index_count1 = 0 
                     np.set_printoptions(precision = 14)
                     for k in range(self.n_occupied):
                         for r in range(k+1,self.nmo):
                             if (k < self.n_in_a and r < self.n_in_a): continue
                             if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                             reduced_gradient[index_count1] = gradient_tilde[r][k]
                             #print(r,k,index_count1)
                             index_count2 = 0 
                             for l in range(self.n_occupied):
                                 for s in range(l+1,self.nmo):
                                     if (l < self.n_in_a and s < self.n_in_a): continue
                                     if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                                     #if (k >= self.n_occupied and r >= self.n_occupied): continue
                                     reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                                     #print(r,k,s,l,index_count1,index_count2)
                                     index_count2 += 1
                             index_count1 += 1
                     #print("reduced_gradient",reduced_gradient, flush = True)
                     mu1, w1 = np.linalg.eigh(reduced_hessian)
                     print("eigenvalue of the reduced hessian", mu1)
                     print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]))
                     print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]))
                     H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                     step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                     print("norm of critical step", np.linalg.norm(step_limit))
                     print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
                     print("gradient norm", np.linalg.norm(reduced_gradient))



                     
                     current_energy = old_energy + energy_change
                     print("now accept step, check current energy", current_energy)
                     hard_case = 0
                else:
                    restart = False
                    #guess_vector = trial_vector
                    reduce_step = 1
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush = True)
                    print("new trust radius", trust_radius, flush = True)
            #if convergence == 1:
            #    #active_twoeint[:,:,:,:] = self.active_twoeint[:,:,:,:] 
            #    #active_fock_core[:,:] = self.active_fock_core[:,:] 
            #    #d_cmo[:,:] = self.d_cmo[:,:]

            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    #temp8 = np.zeros((self.nmo, self.nmo))
            #    #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
            #    #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    #temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
            #    #self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    break


            start = timer()
            active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
            fock_core = copy.deepcopy(self.H_spatial2) 
            fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
            fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
            
            E_core = 0.0  
            E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            E_core += np.einsum("jj->", fock_core[:self.n_in_a,:self.n_in_a]) 


            active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
            active_fock_core[:,:] = fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
            d_cmo = self.d_cmo
            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            #print("recheck energy", flush = True)
            #active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            #active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy)
            #print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            #print("gfhgy",
            #    "{:20.12f}".format(sum_energy),
            #    "{:20.12f}".format(active_one_e_energy),
            #    "{:20.12f}".format(active_two_e_energy),
            #    "{:20.12f}".format(self.E_core2),
            #    "{:20.12f}".format(active_one_pe_energy),
            #    "{:20.12f}".format(ci_dependent_energy),
            #    "{:20.12f}".format(self.Enuc),
            #    flush = True
            #)
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    E_core, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            print(self.H_diag3)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = E_core
            self.constint[8] = 4 
            eigenvals = np.zeros((self.davidson_roots))
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            #print("heyhey5", eigenvecs)
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            end   = timer() 
            print("CI step took", end - start, flush = True)


            #print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print("microiteration",microiteration + 1, "current average energy", avg_energy, flush = True)
            current_energy = avg_energy
            
            start = timer() 
            self.build_state_avarage_rdms(eigenvecs)
            end   = timer() 
            print("building RDM took", end - start)

            current_energy2 = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs)
            
            self.build_intermediates(eigenvecs, A, G, True)
            self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            #print(np.shape(gradient_tilde), flush = True)
            
            start = timer() 
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            end   = timer() 
            #self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            

            
            print("build gradient took", end - start)
            hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()    
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end   = timer() 
            print("build hessian diagonal took", end - start)
            reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            reduced_gradient = np.zeros(self.index_map_size)
            index_count1 = 0 
            for k in range(self.n_occupied):
                for r in range(k+1,self.nmo):
                    if (k < self.n_in_a and r < self.n_in_a): continue
                    if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    #print(r,k,index_count1)
                    index_count2 = 0 
                    for l in range(self.n_occupied):
                        for s in range(l+1,self.nmo):
                            if (l < self.n_in_a and s < self.n_in_a): continue
                            if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                            #if (k >= self.n_occupied and r >= self.n_occupied): continue
                            reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                            #print(r,k,s,l,index_count1,index_count2)
                            index_count2 += 1
                    index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            np.set_printoptions(precision = 14)
            mu1, w1 = np.linalg.eigh(reduced_hessian)
            print("eigenvalue of the reduced hessian", mu1)
            print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]))
            print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]))
            H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            print("norm of critical step", np.linalg.norm(step_limit))
            print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
            gradient_norm = np.linalg.norm(reduced_gradient)
            print("gradient norm", gradient_norm)
            alpha_critical = mu1[0] - np.dot(reduced_gradient, step_limit)
            augmented_hessian90= np.zeros((dim00, dim00))
            augmented_hessian90[0,0] = alpha_critical
            augmented_hessian90[0,1:] = reduced_gradient
            augmented_hessian90[1:,0] = reduced_gradient.T
            augmented_hessian90[1:,1:] = reduced_hessian

            mu90, w90 = np.linalg.eigh(augmented_hessian90)
            print(mu90[0], mu90[1])
            if w90[0,0] > 1e-15:
                zzzz = w90[1:,0]/w90[0,0]
                print(np.linalg.norm(zzzz))
            elif w90[0,1] > 1e-15:
                yyyy = w90[1:,1]/w90[0,1]
                print("second case", np.linalg.norm(yyyy))
            print("ci energy", current_energy, "rdm_energy", current_energy2)
            print("current gradient_norm and residual", gradient_norm, current_residual)
            print("current convergence_threshold", convergence_threshold)
            total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2)) 
            #macroiteration_energy_current = current_energy2
            #if total_norm < convergence_threshold: 
            #    print("total norm", total_norm, flush = True)
            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    #temp8 = np.zeros((self.nmo, self.nmo))
            #    #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
            #    #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    #self.d_cmo[:,:] = d_cmo[:,:]
            #    #print(eigenvecs)
            #    #print("u2i",self.U2)
            #    print("microiteration converged! (small total norm)", flush = True)
            #    break 


            microiteration += 1









    def bfgs_orbital_optimization(self, eigenvecs, c_get_roots):
        self.eigenvecs = eigenvecs
        self.H1 = copy.deepcopy(self.H_spatial2)
        self.d_cmo1 = copy.deepcopy(self.d_cmo)
        #print("E_core", self.E_core)
        self.U2 = np.eye(self.nmo)
        self.U_total = np.eye(self.nmo)
        U_temp = np.eye(self.nmo)
        convergence_threshold = 1e-5
        trust_radius = 0.4   
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #A2 = np.zeros((rot_dim, rot_dim))
        #G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        #print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size+1))
        #convergence_threshold = 1e-4
        convergence = 0
        #while(True):
        macroiteration_energy_initial = 0.0
        macroiteration_energy_current = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 5000 
        microiteration = 0
        x00 = np.zeros(self.index_map_size)
        while(microiteration < N_microiterations):
            print("\n")
            print("\n")
            print("MACROITERATION", microiteration+1,flush = True)

        #while(microiteration < 2):
            #trust_radius = 0.35 
            #A[:,:] = 0.0
            #G[:,:,:,:] = 0.0
            #start = timer()
            #self.build_intermediates(eigenvecs, A, G, True)
            #end   = timer()
            #print("build intermediates took", end - start,flush = True)
           

            start = timer()
            start1 = timer()
            macroiteration_energy_initial = macroiteration_energy_current 
            
            macroiteration_energy_current = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, self.eigenvecs)
            


            end   = timer()
            print("check initial energy took", end - start,flush = True)
           



            #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            
            #start = timer() 
            #self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            #G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            #end   = timer() 
            ##self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            #

            #
            #print("build gradient took", end - start,flush = True)
            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            ##hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            #reduced_hessian_diagonal = np.zeros(self.index_map_size)
            #start = timer()    
            #self.build_hessian_diagonal(self.U2, G, A_tilde2)
            #end   = timer() 
            #print("build hessian diagonal took", end - start,flush = True)
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            #reduced_gradient = np.zeros(self.index_map_size)
            #print(np.shape(reduced_gradient), np.shape(gradient_tilde))
            #index_count1 = 0 
            #for k in range(self.n_occupied):
            #    for r in range(k+1,self.nmo):
            #        if (k < self.n_in_a and r < self.n_in_a): continue
            #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
            #        reduced_gradient[index_count1] = gradient_tilde[r][k]
            #        #index_count2 = 0 
            #        #for l in range(self.n_occupied):
            #        #    for s in range(l+1,self.nmo):
            #        #        if (l < self.n_in_a and s < self.n_in_a): continue
            #        #        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
            #        #        #if (k >= self.n_occupied and r >= self.n_occupied): continue
            #        #        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
            #        #        #print(r,k,s,l,index_count1,index_count2)
            #        #        index_count2 += 1
            #        index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            ####np.set_printoptions(precision = 14)
            ####mu1, w1 = np.linalg.eigh(reduced_hessian)
            ####print("eigenvalue of the reduced hessian", mu1,flush = True)
            ####print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]),flush = True)
            ####print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]),flush = True)
            ####H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            ####step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            ####print("norm of critical step", np.linalg.norm(step_limit),flush = True)
            ####print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit),flush = True)
            #gradient_norm = np.linalg.norm(reduced_gradient)
            #print("gradient norm", gradient_norm,flush = True)
            ##if microiteration ==0:  
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            #if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION",flush = True)
            print(macroiteration_energy_initial, macroiteration_energy_current,flush = True)
            print("initial convergence threshold", convergence_threshold,flush = True)
            if (np.abs(macroiteration_energy_current - macroiteration_energy_initial) < 1e-9) and microiteration >=2:
            #if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)",flush = True)
                #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                break


            #print("eigenvec", self.eigenvecs,flush = True)
            x0 = np.zeros(self.index_map_size)
            x0[:] = x00
            res = scipy.optimize.minimize(self.energy_function, x0, method='BFGS', jac=self.energy_grad, options={'disp': True}) 
            step = res.x
            #print(res.x,flush = True)
            step_norm = np.linalg.norm(step)
            x00 = res.x 

            
            start = timer()
            Rai = np.zeros((self.n_act_orb, self.n_in_a))
            Rvi = np.zeros((self.n_virtual,self.n_in_a))
            Rva = np.zeros((self.n_virtual,self.n_act_orb))
            for i in range(self.index_map_size):
                s = self.index_map[i][0] 
                l = self.index_map[i][1]
                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s-self.n_in_a][l] = step[i]
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s-self.n_occupied][l] = step[i]
                else:
                    Rva[s-self.n_occupied][l-self.n_in_a] = step[i]

            self.build_unitary_matrix(Rai, Rvi, Rva)
            #print("jnti",self.E_core)

            U_temp = self.U_delta 
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)
            c_full_transformation_macroiteration(U_temp, self.twoeint, J_temp, K_temp, self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied) 
            #self.full_transformation_macroiteration(self.U_total, self.J, self.K)

            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)


            new_energy = self.rdm_exact_energy(J_temp, K_temp, h1_temp, d_cmo_temp, eigenvecs)

            end   = timer()
            print("build unitary matrix and recheck energy took", end - start,flush = True)
            self.J[:,:,:,:] = J_temp
            self.K[:,:,:,:] = K_temp
            self.H_spatial2[:,:] = h1_temp
            self.d_cmo[:,:] = d_cmo_temp
            #self.full_transformation_macroiteration(self.U_total, self.J, self.K)
            #temp_energy = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs)
            #print("check energy again",temp_energy,flush = True)
            A[:,:] = 0
            A_tilde2[:,:] = 0
            #G[:,:,:,:] = 0
            #gradient_tilde[:] = 0
            #hessian_tilde[:,:,:,:] = 0
            ###self.build_intermediates(eigenvecs, A, G, True)
            ###self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            ###self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            ###end   = timer() 
            ####self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            ###G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            ###print("build gradient took", end - start,flush = True)

            ###hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            ###hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            #reduced_hessian_diagonal = np.zeros(self.index_map_size)
            #index_count1 = 0 
            #for k in range(self.n_occupied):
            #    for r in range(k+1,self.nmo):
            #        if (k < self.n_in_a and r < self.n_in_a): continue
            #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
            #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
            #        index_count1 += 1
            #start = timer() 
            #self.build_hessian_diagonal(self.U2, G, A_tilde2)
            #end   = timer() 
            #print("build hessian diagonal took", end - start,flush = True)
            #print("diagonal elements of the reduced hessian")   
            #for i in range(self.index_map_size):
            #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
            #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            #reduced_gradient[:] = 0 
            #reduced_hessian[:,:] = 0 
            #index_count1 = 0 
            #np.set_printoptions(precision = 14)
            #for k in range(self.n_occupied):
            #    for r in range(k+1,self.nmo):
            #        if (k < self.n_in_a and r < self.n_in_a): continue
            #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
            #        reduced_gradient[index_count1] = gradient_tilde[r][k]
            #        #print(r,k,index_count1)
            #        index_count2 = 0 
            #        for l in range(self.n_occupied):
            #            for s in range(l+1,self.nmo):
            #                if (l < self.n_in_a and s < self.n_in_a): continue
            #                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
            #                #if (k >= self.n_occupied and r >= self.n_occupied): continue
            #                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
            #                #print(r,k,s,l,index_count1,index_count2)
            #                index_count2 += 1
            #        index_count1 += 1
            ##print("reduced_gradient",reduced_gradient, flush = True)
            #mu1, w1 = np.linalg.eigh(reduced_hessian)
            #print("eigenvalue of the reduced hessian", mu1,flush = True)
            #print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]),flush = True)
            #print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]),flush = True)
            #H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            #step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            #print("norm of critical step", np.linalg.norm(step_limit),flush = True)
            #print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit),flush = True)
            #print("gradient norm", np.linalg.norm(reduced_gradient),flush = True)




            start = timer()
            active_twoeint = self.J[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] 
            fock_core = copy.deepcopy(self.H_spatial2) 
            fock_core += 2.0 * np.einsum("jjrs->rs", self.J[:self.n_in_a,:self.n_in_a,:,:]) 
            fock_core -= np.einsum("jjrs->rs", self.K[:self.n_in_a,:self.n_in_a,:,:]) 
            
            E_core = 0.0  
            E_core += np.einsum("jj->", self.H_spatial2[:self.n_in_a,:self.n_in_a]) 
            E_core += np.einsum("jj->", fock_core[:self.n_in_a,:self.n_in_a]) 


            active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
            active_fock_core[:,:] = fock_core[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied]
            d_cmo = self.d_cmo
            occupied_J = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
            occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_twoeint)        
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied] = copy.deepcopy(active_fock_core) 
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied,: self.n_occupied]) 
            gkl2 = copy.deepcopy(active_fock_core) 
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint) 
            #print("recheck energy", flush = True)
            #active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            #active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy)
            #print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            #print("gfhgy",
            #    "{:20.12f}".format(sum_energy),
            #    "{:20.12f}".format(active_one_e_energy),
            #    "{:20.12f}".format(active_two_e_energy),
            #    "{:20.12f}".format(self.E_core2),
            #    "{:20.12f}".format(active_one_pe_energy),
            #    "{:20.12f}".format(ci_dependent_energy),
            #    "{:20.12f}".format(self.Enuc),
            #    flush = True
            #)
            occupied_J = occupied_J.reshape(self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied)

            c_H_diag_cas_spin(
                    occupied_fock_core, 
                    occupied_J, 
                    self.H_diag3, 
                    self.N_p, 
                    self.num_alpha, 
                    self.nmo, 
                    self.n_act_a, 
                    self.n_act_orb, 
                    self.n_in_a, 
                    E_core, 
                    self.omega, 
                    self.Enuc, 
                    self.d_c, 
                    self.Y,
                    self.target_spin)
            #print(self.H_diag3,flush = True)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[:self.n_in_a,:self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9 
            self.constdouble[5] = E_core
            self.constint[8] = 4 
            eigenvals = np.zeros((self.davidson_roots))
            #eigenvecs = np.zeros((self.davidson_roots, H_dim))
            #eigenvecs[:,:] = 0.0
            #print("heyhey5", eigenvecs)
            c_get_roots(
                gkl2,
                occupied_J,
                occupied_d_cmo,
                self.H_diag3,
                self.S_diag,
                self.S_diag_projection,
                eigenvals,
                eigenvecs,
                self.table,
                self.table_creation,
                self.table_annihilation,
                self.b_array,
                self.constint,
                self.constdouble,
                self.index_Hdiag,
                True,
                self.target_spin,
            )
            end   = timer() 
            print("CI step took", end - start, flush = True)


            #print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print("microiteration",microiteration + 1, "current average energy", avg_energy, flush = True)
            current_energy = avg_energy
            
            start = timer() 
            self.build_state_avarage_rdms(eigenvecs)
            end   = timer() 
            print("building RDM took", end - start,flush = True)

            current_energy2 = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs)
            
            ###self.build_intermediates(eigenvecs, A, G, True)
            ###self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            ####print(np.shape(gradient_tilde), flush = True)
            ###
            ###start = timer() 
            ###self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            ###G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            ###end   = timer() 
            ####self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            ###

            ###
            ###print("build gradient took", end - start)
            ###hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            ###hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            ####hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            ###reduced_hessian_diagonal = np.zeros(self.index_map_size)
            ###start = timer()    
            ###self.build_hessian_diagonal(self.U2, G, A_tilde2)
            ###end   = timer() 
            ###print("build hessian diagonal took", end - start)
            ###reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            ###reduced_gradient = np.zeros(self.index_map_size)
            ###index_count1 = 0 
            ###for k in range(self.n_occupied):
            ###    for r in range(k+1,self.nmo):
            ###        if (k < self.n_in_a and r < self.n_in_a): continue
            ###        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
            ###        reduced_gradient[index_count1] = gradient_tilde[r][k]
            ###        #print(r,k,index_count1)
            ###        index_count2 = 0 
            ###        for l in range(self.n_occupied):
            ###            for s in range(l+1,self.nmo):
            ###                if (l < self.n_in_a and s < self.n_in_a): continue
            ###                if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
            ###                #if (k >= self.n_occupied and r >= self.n_occupied): continue
            ###                reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
            ###                #print(r,k,s,l,index_count1,index_count2)
            ###                index_count2 += 1
            ###        index_count1 += 1
            ###        
            ####print("reduced_gradient",reduced_gradient, flush = True)
            ###np.set_printoptions(precision = 14)
            ###mu1, w1 = np.linalg.eigh(reduced_hessian)
            ###print("eigenvalue of the reduced hessian", mu1)
            ###print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]),flush = True)
            ###print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]),flush = True)
            ###H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            ###step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            ###print("norm of critical step", np.linalg.norm(step_limit),flush = True)
            ###print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit),flush = True)
            ###gradient_norm = np.linalg.norm(reduced_gradient)
            ###print("gradient norm", gradient_norm,flush = True)

            print("ci energy", current_energy, "rdm_energy", current_energy2,flush = True)
            ###print("current gradient_norm and residual", gradient_norm, current_residual,flush = True)
            ###print("current convergence_threshold", convergence_threshold,flush = True)
            ###total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2))
            ###self.eigenvecs = eigenvecs
            ###print("eigenvec", self.eigenvecs,flush = True)
            #macroiteration_energy_current = current_energy2
            #if total_norm < convergence_threshold: 
            #    print("total norm", total_norm, flush = True)
            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    #temp8 = np.zeros((self.nmo, self.nmo))
            #    #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
            #    #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #    #self.d_cmo[:,:] = d_cmo[:,:]
            #    #print(eigenvecs)
            #    #print("u2i",self.U2)
            #    print("microiteration converged! (small total norm)", flush = True)
            #    break 


            microiteration += 1






















    def build_hessian_diagonal(self, U, G, A_tilde):
        dim1 = self.n_occupied * self.nmo
        #start = timer() 
        temp1 = np.zeros((self.n_occupied, self.nmo, self.nmo))
        temp1 = np.einsum("kkrs->krs", G[:self.n_occupied,:self.n_occupied,:,:], optimize = "optimal") 
        temp2 = np.zeros((self.n_occupied, self.nmo, self.n_virtual))
        temp2 = np.einsum("krs,sa->kra", temp1, U[:, self.n_occupied:self.nmo], optimize = "optimal")
        self.hessian_diagonal = np.zeros((self.nmo, self.n_occupied))
        self.hessian_diagonal[self.n_occupied:self.nmo,:] = np.einsum("kra,ra->ak", temp2, U[:, self.n_occupied:self.nmo], optimize = "optimal")

        #end = timer()
        #print("hessian diagonal step took", end - start)
        #B = np.zeros((self.nmo, self.nmo))
        #T = U - np.eye(self.nmo)
        #B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
        #A_tilde = np.zeros((self.nmo, self.nmo))
        #A_tilde = np.einsum("rs,sk->rk",U.T, B)
        temp4 = np.zeros((self.n_occupied, self.n_occupied))
       
        #start = timer() 
        o = np.full((self.n_occupied), 1)
        v = np.full((self.n_virtual), 1)
        self.hessian_diagonal[self.n_occupied:self.nmo,:] -= np.einsum("k,a->ak", o, np.diagonal(A_tilde[self.n_occupied:self.nmo,self.n_occupied:self.nmo]))
        self.hessian_diagonal[self.n_occupied:self.nmo,:] -= np.einsum("a,k->ak", v, np.diagonal(A_tilde[:self.n_occupied,:self.n_occupied]))
        #end = timer()
        #print("hessian diagonal step 1 took", end - start)
        #start = timer() 
        temp3 = np.zeros((self.n_occupied, self.nmo, self.n_occupied))
        temp3 = np.einsum("krs,sl->krl", temp1, U[:, :self.n_occupied], optimize = "optimal")
        temp4 = np.einsum("krl,rl->lk", temp3, U[:, :self.n_occupied], optimize = "optimal")
        self.hessian_diagonal[:self.n_occupied,:] += 1.0 * temp4 
        self.hessian_diagonal[:self.n_occupied,:] += 1.0 * temp4.T
        #end = timer()
        #print("hessian diagonal step 2 took", end - start)
        #start = timer() 
        temp3 = np.einsum("klrs,sk->krl", G, U[:, :self.n_occupied], optimize = "optimal")
        #end = timer()
        #print("hessian diagonal step 3 took", end - start)
        #start = timer()  
        self.hessian_diagonal[:self.n_occupied,:] -= 2.0 * np.einsum("krl,rl->kl", temp3, U[:, :self.n_occupied], optimize = "optimal")
        self.hessian_diagonal[:self.n_occupied,:] -= np.einsum("k,l->kl", o, np.diagonal(A_tilde[:self.n_occupied,:self.n_occupied]))
        self.hessian_diagonal[:self.n_occupied,:] -= np.einsum("l,k->kl", o, np.diagonal(A_tilde[:self.n_occupied,:self.n_occupied]))
        temp4 = A_tilde[:self.n_occupied,:self.n_occupied] + A_tilde.T[:self.n_occupied,:self.n_occupied]
        self.hessian_diagonal[:self.n_occupied,:] += np.einsum("kl,kl->kl", np.eye(self.n_occupied), temp4, optimize = "optimal")
        #end = timer()
        #print("hessian diagonal step 4 took", end - start)
        #print("i7gt",hessian_diagonal)
        #start = timer() 
        index_count1 = 0
        for k in range(self.n_occupied):
            for r in range(k+1,self.nmo):
                if (k < self.n_in_a and r < self.n_in_a): continue
                if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                self.reduced_hessian_diagonal[index_count1] = self.hessian_diagonal[r][k]
                index_count1 += 1
        #end = timer()
        #print("hessian diagonal step 5 took", end - start)
    def Davidson_augmented_hessian_solve(self, U, A_tilde, G, G1, reduced_hessian_diagonal, reduced_gradient, davidson_step, trust_radius, guess_vector, restart):
        start = timer()
        hard_case = 0
        threshold = 1e-7 
        count = 0
        print("trust radius", trust_radius, "restart", restart)
        for i in range(self.index_map_size):
            if reduced_hessian_diagonal[i] <= 1e-14:
                count +=1
        if self.index_map_size > 600:
            dim0 = 200
        else:
            dim0 = self.index_map_size//3
        dim1 = max(count, dim0)  
        #print("eepp",reduced_hessian_diagonal)
        alpha = 1  
        alpha_min = 1  
        alpha_max = 1  
        dim2 = self.index_map_size + 1
        H_dim = dim2
        H_diag = copy.deepcopy(reduced_hessian_diagonal)
        #print(H_diag)
        H_diag = np.concatenate(([0.0], H_diag ))
        #print(H_diag)
        indim = 1
        maxdim = 10
        unconverged_idx = []
        nroots = 1
        end   = timer()
        print("guess1 took", end - start)

        #start = timer()
        #G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
        #end   = timer()
        #print("transpose took", end - start)


        
        start = timer()
        if restart == False:
            out = np.empty(self.index_map_size)
            out.fill(1e9)
            d = np.divide(np.absolute(reduced_gradient), reduced_hessian_diagonal, out, where=reduced_hessian_diagonal>1e-14)
            idx = (-d).argsort()[:dim1]
            #print(d)
            #print("order", idx)
            guess_hessian = np.zeros((dim1, dim1))
            guess_gradient = np.zeros(dim1)
            sym_A_tilde = A_tilde + A_tilde.T        
            
            start1 = timer() 
            #for i in range(dim1):
            #    index1 = idx[i]
            #    r = self.index_map[index1][0] 
            #    k = self.index_map[index1][1]
            #    guess_gradient[i] = reduced_gradient[index1]
            #    for j in range(dim1):
            #        index2 = idx[j]
            #        s = self.index_map[index2][0] 
            #        l = self.index_map[index2][1]
            #        #print(r,k,s,l)
            #        #b = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
            #        guess_hessian[i][j] = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
            #        #print(b, hessian_tilde[r*self.n_occupied+k][s*self.n_occupied+l])
            
            self.build_orbital_hessian_guess(U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, idx)
            end1   = timer() 
            print("building orbital guess took", end1 - start1)


            mu, w = np.linalg.eigh(guess_hessian)
            #print(mu)
            aug_eigvecs = np.zeros((dim1+1, dim1+1))
            indim = dim1 + 1
            maxdim = dim1 + min(dim1, 40) 
            Lmax = maxdim
            L = indim
            print(H_dim, indim)
            Q = np.zeros((indim, H_dim))
            projected_step = np.zeros(dim1)
            #print("norm of guess gradient", np.linalg.norm(guess_gradient))
            #print(guess_gradient)
            #print(guess_hessian)
            while True: 
                mu0 = self.projection_step(guess_gradient, guess_hessian, aug_eigvecs, alpha, dim1+1)
                if np.absolute(aug_eigvecs[0][0]) < 1e-4 and alpha == 1:
                    hard_case = 1
                    break
                else:
                    hard_case = 0
                    projected_step = aug_eigvecs[1:,0]/aug_eigvecs[0][0]
                    projected_step /=alpha
                    projected_step_norm = np.sqrt(np.dot(projected_step, projected_step.T))
                    #print("qqrwfwe1", alpha, projected_step_norm, mu0[0], aug_eigvecs[:,0])
                    #print("projected step norm", projected_step_norm)
                    if projected_step_norm > trust_radius:
                        alpha_min = alpha
                        alpha = alpha * 10
                    else:
                        alpha_max = alpha
                        break

            print("alpha range", alpha_min, alpha_max, flush = True)
            if alpha_max != 1:
                #count =0
                while True:
                    #print(alpha_min, alpha_max)
                    alpha = 0.5 * (alpha_min + alpha_max)
                    mu0 = self.projection_step(guess_gradient, guess_hessian, aug_eigvecs, alpha, dim1+1)
                    projected_step = aug_eigvecs[1:,0]/aug_eigvecs[0][0] 
                    projected_step /=alpha
                    projected_step_norm = np.sqrt(np.dot(projected_step, projected_step.T))
                    #print("qqrwfwe1", alpha, projected_step_norm)
                    #print(mu0[0], aug_eigvecs[:,0])
                    #if count >= 100: break
                    if trust_radius - projected_step_norm <= 1e-2 and trust_radius - projected_step_norm >= 0.0:
                        break
                    elif trust_radius - projected_step_norm > 1e-2:
                        alpha_max = alpha
                    else:
                        alpha_min = alpha
                    #count +=1

            reduced_gradient1 = alpha * reduced_gradient
            #augmented_hessian = np.zeros((dim2, dim2))
            #augmented_hessian[0,1:] = reduced_gradient1
            #augmented_hessian[1:,0] = reduced_gradient1.T
            #augmented_hessian[1:,1:] = reduced_hessian
            #print("augmented_hessian")
            #self.printA(augmented_hessian)
            #print(" ", flush = True)

            #theta1, alpha1 = np.linalg.eigh(augmented_hessian)
            #print(theta1)
            for i in range(dim1+1):
                for j in range(dim1):
                    index1 = idx[j]
                    Q[i][0] = aug_eigvecs[0][i]
                    Q[i][index1+1] = aug_eigvecs[j+1][i]
            #QQ = np.einsum("pq,qr->pr", Q, Q.T)
            #print("check orthogonality",QQ)
            #S1 = np.einsum("pq, qr->rp", augmented_hessian, Q.T)
            #print("first Q",Q[0,:], flush = True)
            S = np.zeros((indim, H_dim))
            #S1 = np.zeros((indim, H_dim))
                     


            #self.build_sigma_reduced2(U, A_tilde, G, Q, S, indim, 1)
            #S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
            #S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )

            #self.sigma_total4 = np.zeros((indim, self.nmo, self.n_occupied))
            #self.build_sigma_reduced3(U, hessian_tilde, G, Q, S, indim, 1)
            self.orbital_sigma(U, A_tilde, G1, Q, S, indim, 1)
            #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S, indim, 1, self.nmo, self.index_map_size, self.n_occupied)
            S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
            S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )
            
            #for i in range(indim):
            #   for j in range(H_dim):
            #       print(S1[i][j], S[i][j], S1[i][j]-S[i][j])


            #print("dsoq", np.allclose(S1,S, rtol=1e-14,atol=1e-14))

            projected_augmented_hessian = np.dot(Q, S.T)
            #print(projected_augmented_hessian)
            # Diagonalize it, and sort the eigenvector/eigenvalue pairs
            theta, eigvecs = np.linalg.eigh(projected_augmented_hessian)
            #print("rdsi",theta)
            #w1 = np.zeros(H_dim)
            #w1 = np.copy(S[0,:])
            #print(np.shape(w1))
            #w1 -= theta[0] * Q[0,:]
            #print(w1)
            #print(np.linalg.norm(w1))
            w = np.zeros((nroots, H_dim))
            residual_norm = np.zeros((nroots))
            unconverged_idx = []
            convergence_check = np.zeros((nroots), dtype=str)
            conv = 0
            for j in range(nroots):
                # Compute a residual vector "w" for each root we seek
                w[j, :] = np.dot(eigvecs[:, j].T, S) - theta[j] * np.dot(eigvecs[:, j].T, Q)
                residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                if residual_norm[j] < threshold:
                    conv += 1
                    convergence_check[j] = "Yes"
                else:
                    unconverged_idx.append(j)
                    convergence_check[j] = "No"
            print(unconverged_idx)

            print("root", "AH residual norm", "Eigenvalue", "Convergence")
            for j in range(nroots):
                print(
                    j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
                )

            if conv == nroots:
                Q1=np.dot(eigvecs.T[:nroots,:], Q)
                if hard_case == 0:
                    print(np.shape(Q1))
                    davidson_step[:,:] = Q1[0,1:]/Q1[0,0]
                    davidson_step[:,:] /= alpha
                else:
                    davidson_step[:,:] = Q1[0,1:]
                self.mu_min = theta[0]    
                #print("davidson step",davidson_step, flush = True)
                print("norm of solution", np.linalg.norm(davidson_step), flush = True)
                print("converged from the first iteration!", flush = True)


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

            
            

                Q = np.concatenate((Q,preconditioned_w),axis=0)
                c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                L = Q.shape[0]
            #QQ = np.einsum("pq,qr->pr", Q, Q.T)
            #print("check orthogonality",QQ)
            #print("current dimension", L, flush = True)

        else:
            indim = guess_vector.shape[0]
            #maxdim = dim1 + 3 
            maxdim = dim1 + min(dim1, 20)
            #if indim == maxdim:
            #    maxdim +=1
            print(H_dim, indim)
            Q = copy.deepcopy(guess_vector)
            Lmax = maxdim
            L = indim
        
        end   = timer()
        print("guess2 took", end - start)



        if (len(unconverged_idx) > 0 and restart == False) or restart == True:
            num_iter = 10000 
            for davidson_iteration in range(1, num_iter):
                print("\n")
                print("iteration", davidson_iteration+1)
                alpha = 1  
                alpha_min = 1  
                alpha_max = 1  
                L = Q.shape[0]
                projected_step = np.zeros(L-1)
                #S3 = np.zeros_like(Q)
                #self.build_sigma_reduced2(U, A_tilde, G, Q, S3, L, 1)
                
                #S4 = np.zeros_like(Q)
                #start = timer() 
                #self.build_sigma_reduced3(U, hessian_tilde, G, Q, S4, L, 1)
                #end = timer()
                #print("buil sigma took ", end - start)
                #S6 = np.zeros_like(Q) 
                #start = timer()

                #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S6, L, 1, self.nmo, self.index_map_size, self.n_occupied)
                ###self.build_sigma_reduced3(U, hessian_tilde, G, Q, S3, L, 1)
                ##self.orbital_sigma(U, A_tilde, G1, Q, S6, L, 1)
                #end = timer()
                #print("build sigma2 with c took ", end - start)
                

                
                start = timer() 
                S3 = np.zeros_like(Q)
                #self.build_sigma_reduced3(U, hessian_tilde, G, Q, S3, L, 1)
                #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S3, L, 1, self.nmo, self.index_map_size, self.n_occupied)
                self.orbital_sigma(U, A_tilde, G1, Q, S3, L, 1)
                end = timer()
                print("build orbital sigma took", end - start)
                #print("build sigma with numba took ", end - start)
                #for i in range(L):
                #   for j in range(H_dim):
                #       #a = S3[i][j]-S6[i][j]
                #       #if np.absolute(a) > 1e-14: print (i,j,a, "large error")
                #       print("%20.12lf" %(S3[i][j]), "%20.12lf" %(S6[i][j]), "%20.12lf" %(S3[i][j]-S6[i][j]))
                #for i in range(L):
                #   for r in range(self.nmo):
                #       for l in range(self.n_occupied):
                #           a = sigma_total2[i][r][l]-self.sigma_total4[i][r][l]
                #           if np.absolute(a) > 1e-14: print (i,r,l,a, "large error")


                #print("aipo", np.allclose(S3,S6, rtol=1e-14,atol=1e-14))

                start = timer()
                while True:
                    reduced_gradient1 = alpha * reduced_gradient
                    #augmented_hessian[0,1:] = reduced_gradient1
                    #augmented_hessian[1:,0] = reduced_gradient1.T
                    #augmented_hessian[1:,1:] = reduced_hessian
                    #S2 = np.einsum("pq, qr->rp", augmented_hessian, Q.T)

                    S = copy.deepcopy(S3)
                             
                    #print(np.shape(S))

                    S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
                    S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )

                    #print("aipo", np.allclose(S2,S, rtol=1e-14,atol=1e-14))

                    projected_augmented_hessian = np.dot(Q, S.T)
                    #print(alpha)
                    print(np.shape(projected_augmented_hessian))
                    start1 = timer()
                    theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                    end1  = timer()
                    print("diagonalization took",end1 - start1 )
                    #print("zopp",theta)
                    #print(np.shape(aug_eigvecs))
                    full_eigvecs=np.dot(aug_eigvecs.T[:nroots,:], Q)

                    if np.absolute(full_eigvecs[0][0]) < 1e-4 and alpha == 1:
                        hard_case = 1
                        break
                    else:
                        hard_case = 0
                        step = full_eigvecs[0,1:]/full_eigvecs[0,0]
                        step /= alpha
                        step_norm = np.sqrt(np.dot(step, step.T))

                        #print("step norm2",step_norm)
                        if step_norm > trust_radius:
                            alpha_min = alpha
                            alpha = alpha * 10
                        else:
                            alpha_max = alpha
                            break

                print("alpha range", alpha_min, alpha_max)
                end = timer()
                print("finding alpha range0 took", end - start)
                start = timer()
                if alpha_max != 1:
                    while True:
                        #print(alpha_min, alpha_max)
                        alpha = 0.5 * (alpha_min + alpha_max)
                        #reduced_gradient1 = alpha * reduced_gradient
                        #augmented_hessian[0,1:] = reduced_gradient1
                        #augmented_hessian[1:,0] = reduced_gradient1.T
                        #augmented_hessian[1:,1:] = reduced_hessian
                        #S2 = np.einsum("pq, qr->rp", augmented_hessian, Q.T)
                        S = copy.deepcopy(S3)
                        S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
                        S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )

                        #print("aipo2", np.allclose(S2,S, rtol=1e-14,atol=1e-14))


                        projected_augmented_hessian = np.dot(Q, S.T)
                        theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                        #print("tymn",theta)
                        full_eigvecs=np.dot(aug_eigvecs.T[:nroots,:], Q)
                        #print(np.shape(full_eigvecs))
                        step = full_eigvecs[0,1:]/full_eigvecs[0,0]
                        #print(step)
                        step /= alpha
                        step_norm = np.sqrt(np.dot(step, step.T))
                        if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                            break
                        elif trust_radius - step_norm > 1e-2:
                            alpha_max = alpha
                        else:
                            alpha_min = alpha

                end = timer()
                print("finding valid alpha0 took", end - start)
                start = timer()
                #print("alpha", alpha, flush = True)
                #print(aug_eigvecs.T[0,:])
                #print("full eigenvector")
                #full_eigvecs=np.dot(aug_eigvecs.T[:nroots,:], Q)
                #print(full_eigvecs[0,:])
                w = np.zeros((nroots, H_dim))
                residual_norm = np.zeros((nroots))
                unconverged_idx = []
                convergence_check = np.zeros((nroots), dtype=str)
                conv = 0
                #print(np.shape(Q),np.shape(S), np.shape(aug_eigvecs) )
                for j in range(nroots):
                    # Compute a residual vector "w" for each root we seek
                    w[j, :] = np.dot(aug_eigvecs[:, j].T, S) - theta[j] * np.dot(aug_eigvecs[:, j].T, Q)
                    residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                    if residual_norm[j] < threshold:
                        conv += 1
                        convergence_check[j] = "Yes"
                    else:
                        unconverged_idx.append(j)
                        convergence_check[j] = "No"
                print(unconverged_idx)

                print("root", "AH residual norm", "Eigenvalue", "Convergence")
                for j in range(nroots):
                    print(
                        j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
                    )

                if conv == nroots:
                    Q1=np.dot(aug_eigvecs.T[:nroots,:], Q)
                    if hard_case == 0:
                        #print(np.shape(Q1))
                        davidson_step[:,:] = Q1[0,1:]/Q1[0,0]
                        davidson_step[:,:] /= alpha
                    else:
                        davidson_step[:,:] = Q1[0,1:]
                    self.mu_min = theta[0]    
                    #print("davidson step", davidson_step, flush = True)
                    print("norm of solution", np.linalg.norm(davidson_step), flush = True)
                    print("converged!", flush = True)
                    break

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
                end = timer()
                print("residual took", end - start)


                if (Lmax-L < len(unconverged_idx)):
                    t_collapsing_begin = time.time()
                    #unconverged_w = np.zeros((len(unconverged_idx),H_dim))
                    Q=np.dot(aug_eigvecs.T[:nroots,:], Q)
                   
                    Q = np.concatenate((Q,preconditioned_w),axis=0)
                    print(Q.shape)
                    #Q=np.column_stack(Qtup)
                    c_gram_schmidt_orthogonalization(Q, nroots+len(unconverged_idx), H_dim)
                    #gc.collect()
                    t_collapsing_end = time.time()
                    print('restart took',t_collapsing_end-t_collapsing_begin,'seconds')
                else:
                    t_expanding_begin = time.time()
                    
                    Q = np.concatenate((Q,preconditioned_w),axis=0)
                    c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                    #gc.collect() 
                    t_expanding_end = time.time()
                    print('expand took',t_expanding_end-t_expanding_begin,'seconds')
                    #print(Q)
 
        return Q, hard_case









    def Davidson_augmented_hessian_solve2(self, U, A_tilde, G, G1, reduced_hessian_diagonal, reduced_gradient, davidson_step, trust_radius, guess_vector, restart):
        hard_case = 0
        threshold = 1e-7 
        count = 0
        #start = timer()
        print("trust radius", trust_radius, "restart", restart)
        for i in range(self.index_map_size):
            if reduced_hessian_diagonal[i] <= 1e-14:
                count +=1
        if self.index_map_size > 600:
            dim0 = 200
        else:
            dim0 = self.index_map_size//2
        dim1 = max(count, dim0)  
        #print("eepp",reduced_hessian_diagonal)
        alpha = 1  
        alpha_min = 1  
        alpha_max = 1  
        dim2 = self.index_map_size + 1
        H_dim = dim2
        H_diag = copy.deepcopy(reduced_hessian_diagonal)
        #print(H_diag)
        H_diag = np.concatenate(([0.0], H_diag ))
        #print(H_diag)
        indim = 1
        maxdim = 10
        unconverged_idx = []
        nroots = 1
        #end   = timer()
        #print("guess1 took", end - start)
        #start = timer()
        #G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
        #end = timer()
        #print("transpose took", end - start)
         
        #start = timer()
        if restart == False:
            #start1 = timer() 
            out = np.empty(self.index_map_size)
            out.fill(1e9)
            d = np.divide(np.absolute(reduced_gradient), reduced_hessian_diagonal, out, where=reduced_hessian_diagonal>1e-14)
            idx = (-d).argsort()[:dim1]
            #print(d)
            #print("order", idx)
            guess_hessian = np.zeros((dim1, dim1))
            guess_gradient = np.zeros(dim1)
            sym_A_tilde = A_tilde + A_tilde.T        
            #end1   = timer() 
            #print("building orbital guess1 took1", end1 - start1, flush = True)
            
            #start1 = timer() 
            #for i in range(dim1):
            #    index1 = idx[i]
            #    r = self.index_map[index1][0] 
            #    k = self.index_map[index1][1]
            #    guess_gradient[i] = reduced_gradient[index1]
            #    for j in range(dim1):
            #        index2 = idx[j]
            #        s = self.index_map[index2][0] 
            #        l = self.index_map[index2][1]
            #        #print(r,k,s,l)
            #        #b = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
            #        guess_hessian[i][j] = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
            #        #print(b, hessian_tilde[r*self.n_occupied+k][s*self.n_occupied+l])
            
            self.build_orbital_hessian_guess(U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, idx)
            #end1   = timer() 
            #print("building orbital guess2 took", end1 - start1, flush = True)
            
            ##try to build orbital guess from sigma vector but it is much slower
            ####start1 = timer() 
            ####guess_gradient2 = np.zeros(dim1)
            ####hq = np.zeros((dim1, self.index_map_size))
            ####QQ = np.zeros((dim1, self.index_map_size))
            ####for i in range(dim1):
            ####    index1 = idx[i]
            ####    r = self.index_map[index1][0] 
            ####    k = self.index_map[index1][1]
            ####    guess_gradient2[i] = reduced_gradient[index1]
            ####    QQ[i][index1] = 1.0

            ####self.orbital_sigma(U, A_tilde, G1, QQ, hq, dim1, 0)
            ####guess_hessian2 = np.dot(hq,QQ.T)
            ####print(np.allclose(guess_hessian, guess_hessian2, rtol=1e-14,atol=1e-14))
            ####print(np.allclose(guess_gradient, guess_gradient2, rtol=1e-14,atol=1e-14))
            ####end1   = timer() 
            ####print("building orbital guess3 took", end1 - start1, flush = True)
            ####print("\n")
            ####print("\n")

            #start1 = timer() 
            #mu, w = np.linalg.eigh(guess_hessian)
            #end1   = timer() 
            #print("building orbital guess3 took", end1 - start1, flush = True)
            #start1 = timer() 
            #print(mu)
            aug_eigvecs = np.zeros((dim1+1, dim1+1))
            indim = dim1 + 1
            maxdim = dim1 + min(dim1, 40) 
            Lmax = maxdim
            L = indim
            print(H_dim, indim)
            Q = np.zeros((indim, H_dim))
            projected_step = np.zeros(dim1)
            while True: 
                theta = self.projection_step(guess_gradient, guess_hessian, aug_eigvecs, alpha, dim1+1)
                if np.absolute(aug_eigvecs[0][0]) < 1e-4 and alpha == 1:
                    hard_case = 1
                    break
                else:
                    hard_case = 0
                    normalized_eigvecs = aug_eigvecs/np.linalg.norm(aug_eigvecs)
                    print(normalized_eigvecs)
                    bb = normalized_eigvecs[0,0]
                    aa = np.linalg.norm(guess_gradient) * np.abs(bb) 
                    cc = np.sqrt(1-bb*bb)
                    print("epsilon",aa/cc, "norm of hessian", np.linalg.norm(guess_hessian))
                    projected_step = aug_eigvecs[1:,0]/aug_eigvecs[0][0]
                    projected_step /=alpha
                    projected_step_norm = np.sqrt(np.dot(projected_step, projected_step.T))
                    print(alpha, aug_eigvecs[0][0],projected_step_norm)
                    #print("projected step norm", projected_step_norm)
                    if projected_step_norm > trust_radius:
                        alpha_min = alpha
                        alpha = alpha * 10
                    else:
                        alpha_max = alpha
                        break

            print("alpha range", alpha_min, alpha_max, flush = True)
            #end1   = timer() 
            #print("building orbital guess took4", end1 - start1, flush = True)
            #start1 = timer() 
            if alpha_max != 1:
                while True:
                    #print(alpha_min, alpha_max)
                    alpha = 0.5 * (alpha_min + alpha_max)
                    theta = self.projection_step(guess_gradient, guess_hessian, aug_eigvecs, alpha, dim1+1)
                    projected_step = aug_eigvecs[1:,0]/aug_eigvecs[0][0] 
                    projected_step /=alpha
                    projected_step_norm = np.sqrt(np.dot(projected_step, projected_step.T))
                    if trust_radius - projected_step_norm <= 1e-2 and trust_radius - projected_step_norm >= 0.0:
                        break
                    elif trust_radius - projected_step_norm > 1e-2:
                        alpha_max = alpha
                    else:
                        alpha_min = alpha

            #end1   = timer() 
            #print("building orbital guess5 took", end1 - start1, flush = True)
            #reduced_gradient1 = alpha * reduced_gradient
            #augmented_hessian = np.zeros((dim2, dim2))
            #augmented_hessian[0,1:] = reduced_gradient1
            #augmented_hessian[1:,0] = reduced_gradient1.T
            #augmented_hessian[1:,1:] = reduced_hessian
            #print("augmented_hessian")
            #self.printA(augmented_hessian)
            #print(" ", flush = True)

            #theta1, alpha1 = np.linalg.eigh(augmented_hessian)
            #print(theta1)
            #start1 = timer() 
            for i in range(dim1+1):
                for j in range(dim1):
                    index1 = idx[j]
                    Q[i][0] = aug_eigvecs[0][i]
                    Q[i][index1+1] = aug_eigvecs[j+1][i]
            #end1   = timer() 
            #print("building orbital guess6 took", end1 - start1, flush = True)
            #print("first QQ", Q[0,:])        
            #QQ = np.einsum("pq,qr->pr", Q, Q.T)
            #print("check orthogonality",QQ)
            #S1 = np.einsum("pq, qr->rp", augmented_hessian, Q.T)
            #print("guess augmented hessian eigenvalue", theta)
            #S = np.zeros((indim, H_dim))
            #S2 = np.zeros((1, H_dim))
            #self.orbital_sigma(U, A_tilde, G1, Q, S2, 1, 1)
            #S2[0,1:] += alpha * Q[0,0] * reduced_gradient   
            #S2[0,0] = alpha * np.dot(reduced_gradient, Q[0,1:] )
            #residual = S2 - theta[0] * Q[0,:]
            #print("zero residual norm", np.linalg.norm(residual), "eigenvalue", theta[0])
        


            #self.build_sigma_reduced2(U, A_tilde, G, Q, S, indim, 1)
            #S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
            #S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )

            #self.sigma_total4 = np.zeros((indim, self.nmo, self.n_occupied))
            #self.build_sigma_reduced3(U, hessian_tilde, G, Q, S, indim, 1)
            #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S, indim, 1, self.nmo, self.index_map_size, self.n_occupied)
            #self.orbital_sigma(U, A_tilde, G1, Q, S, indim, 1)
            #S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
            #S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:])
            
            #for i in range(indim):
            #   for j in range(H_dim):
            #       print(S1[i][j], S[i][j], S1[i][j]-S[i][j])


            #print("dsoq", np.allclose(S1,S, rtol=1e-14,atol=1e-14))

            #projected_augmented_hessian = np.dot(Q, S.T)
            #theta, eigvecs = np.linalg.eigh(projected_augmented_hessian)
            
            #start1 = timer() 
            w = np.zeros((nroots, H_dim))
            self.orbital_sigma(U, A_tilde, G1, Q, w, nroots, 1)
            #self.build_sigma_reduced2(U, A_tilde, G, Q, w, nroots, 1)
            #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, w, nroots, 1, self.nmo, self.index_map_size, self.n_occupied)
            w[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:nroots,0])
            w[:,0] = alpha * np.dot(Q[:nroots,1:], reduced_gradient)
            residual_norm = np.zeros((nroots))
            unconverged_idx = []
            convergence_check = np.zeros((nroots), dtype=str)
            conv = 0
            for j in range(nroots):
                # Compute a residual vector "w" for each root we seek
                #w[j, :] = np.dot(eigvecs[:, j].T, S) - theta[j] * np.dot(eigvecs[:, j].T, Q)
                w[j, :] -= theta[j] * Q[j,:]
                #residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                residual_norm[j] = np.linalg.norm(w[j, :])
                if residual_norm[j] < threshold:
                    conv += 1
                    convergence_check[j] = "Yes"
                else:
                    unconverged_idx.append(j)
                    convergence_check[j] = "No"
            print(unconverged_idx)

            print("root", "AH residual norm", "Eigenvalue", "Convergence")
            for j in range(nroots):
                print(
                    j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
                )

            if conv == nroots:
                Q1=Q
                if hard_case == 0:
                    print(np.shape(Q1))
                    davidson_step[:,:] = Q1[0,1:]/Q1[0,0]
                    davidson_step[:,:] /= alpha
                else:
                    davidson_step[:,:] = Q1[0,1:]
                self.mu_min = theta[0]    
                #print("davidson step",davidson_step, flush = True)
                print("norm of solution", np.linalg.norm(davidson_step), flush = True)
                print("converged from the first iteration!", flush = True)

            #end1   = timer() 
            #print("building orbital guess7 took", end1 - start1, flush = True)
            
            #start1 = timer() 
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

            
            

                Q = np.concatenate((Q,preconditioned_w),axis=0)
                c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                L = Q.shape[0]
            #end1   = timer() 
            #print("building orbital guess8 took", end1 - start1, flush = True)
            #QQ = np.einsum("pq,qr->pr", Q, Q.T)
            #print("check orthogonality",QQ)
            #print("current dimension", L, flush = True)

        else:
            indim = guess_vector.shape[0]
            #maxdim = dim1 + 3 
            maxdim = dim1 + min(dim1, 20)
            #if indim == maxdim:
            #    maxdim +=1
            print(H_dim, indim)
            Q = copy.deepcopy(guess_vector)
            Lmax = maxdim
            L = indim
        #end   = timer()
        #print("guess step2 took", end - start)




        if (len(unconverged_idx) > 0 and restart == False) or restart == True:
            num_iter = 10000 
            #num_iter = 20      
            for davidson_iteration in range(1, num_iter):
                #if num_iter == 18: exit()
                print("\n")
                print("iteration", davidson_iteration+1)
                alpha = 1  
                alpha_min = 1  
                alpha_max = 1  
                L = Q.shape[0]
                projected_step = np.zeros(L-1)

                
                #S5 = np.zeros_like(Q)
                #self.build_sigma_reduced2(U, A_tilde, G, Q, S5, L, 1)
                
                #S4 = np.zeros_like(Q)
                #start = timer() 
                #self.build_sigma_reduced3(U, hessian_tilde, G, Q, S4, L, 1)
                #end = timer()
                #print("buil sigma took ", end - start)
                #S6 = np.zeros_like(Q) 
                #start = timer()

                #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S6, L, 1, self.nmo, self.index_map_size, self.n_occupied)
                ###self.build_sigma_reduced3(U, hessian_tilde, G, Q, S3, L, 1)
                ##self.orbital_sigma(U, A_tilde, G1, Q, S6, L, 1)
                #end = timer()
                #print("build sigma2 with c took ", end - start)
                

                
                #S3 = np.zeros_like(Q)
                #start = timer() 
                ##self.build_sigma_reduced3(U, hessian_tilde, G, Q, S3, L, 1)
                ##c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q, S3, L, 1, self.nmo, self.index_map_size, self.n_occupied)
                #self.orbital_sigma(U, A_tilde, G1, Q, S3, L, 1)
                #end = timer()
                #print("build orbital sigma took", end - start)
                #print("build sigma with numba took ", end - start)
                #for i in range(L):
                #   for j in range(H_dim):
                #       #a = S3[i][j]-S6[i][j]
                #       #if np.absolute(a) > 1e-14: print (i,j,a, "large error")
                #       print("%20.12lf" %(S3[i][j]), "%20.12lf" %(S6[i][j]), "%20.12lf" %(S3[i][j]-S6[i][j]))
                #for i in range(L):
                #   for r in range(self.nmo):
                #       for l in range(self.n_occupied):
                #           a = sigma_total2[i][r][l]-self.sigma_total4[i][r][l]
                #           if np.absolute(a) > 1e-14: print (i,r,l,a, "large error")


                #print("aipo", np.allclose(S3,S6, rtol=1e-14,atol=1e-14))
                
                #print("index map", idx)
                start = timer()
                b_dim = L - indim
                Sq0 = np.zeros((b_dim, H_dim))
                #self.build_sigma_reduced2(U, A_tilde, G, Q[indim:,:], Sq0, b_dim, 1)
                #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q[indim:,:], Sq0, b_dim, 1, self.nmo, self.index_map_size, self.n_occupied)
                self.orbital_sigma(U, A_tilde, G1, Q[indim:,:], Sq0, b_dim, 1)
                gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                gradient2 = np.dot(Q[indim:,1:], reduced_gradient)
                end = timer()
                print("build orbital sigma for Q space took", end - start)
                full_eigvecs = np.zeros((nroots, H_dim))
                full_eigvecs2 = np.zeros((nroots, H_dim))
                #start = timer()
                while True:

                    #start1 = timer()
                    H_pp = np.zeros((indim, indim))
                    scaled_guess_gradient = guess_gradient
                    H_pp[0,1:] = alpha * scaled_guess_gradient
                    H_pp[1:,0] = alpha * scaled_guess_gradient.T
                    H_pp[1:,1:] = guess_hessian
                    H_qp = np.zeros((b_dim, indim))
                    Sq = copy.deepcopy(Sq0)
                    #end1 = timer()
                    #print("alpha step1 took", end1 - start1)
                    #start1 = timer()
                    #Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                    Sq[:,1:] += alpha * gradient1 
                    #Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                    Sq[:,0] = alpha * gradient2 
                    for i in range(b_dim):
                        for j in range(dim1):
                            index1 = idx[j]
                            H_qp[i][0] = Sq[i][0]
                            H_qp[i][j+1] = Sq[i][index1+1]
                    #end1 = timer()
                    #print("alpha step2 took", end1 - start1)

                    #start1 = timer()
                    H_qq = np.dot(Sq, Q[indim:,:].T)
                    #end1 = timer()
                    #print("alpha step3 took", end1 - start1)
                    #start1 = timer()
                    H1 = np.concatenate((H_pp, H_qp), axis = 0)
                    H2 = np.concatenate((H_qp.T, H_qq), axis = 0)
                    projected_augmented_hessian = np.concatenate((H1, H2), axis = 1)
                    #end1 = timer()
                    #print("alpha step4 took", end1 - start1)
                    #print("test second way to build projected hessian")
                    #print(alpha)
                    #print(projected_augmented_hessian)
       
                    #start1 = timer()
                    theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                    #end1 = timer()
                    #print("alpha step5 took", end1 - start1)
                    #print("zopp",theta)
                    #print(np.shape(aug_eigvecs))
                    #start1 = timer()
                    full_eigvecs=np.dot(aug_eigvecs.T[:nroots,indim:], Q[indim:,])
                    full_eigvecs2 = copy.deepcopy(full_eigvecs)
                    #end1 = timer()
                    #print("alpha step6 took", end1 - start1)
                    #start1 = timer()
                    for i in range(nroots):
                        for j in range(dim1):
                            index1 = idx[j]
                            full_eigvecs[i][0] = aug_eigvecs[0][i]
                            full_eigvecs[i][index1+1] = aug_eigvecs[j+1][i]
                    #end1 = timer()
                    #print("alpha step7 took", end1 - start1)
 
                    #print("scale", full_eigvecs[0][0], "alpha", alpha, flush = True)     
                    if np.absolute(full_eigvecs[0][0]) < 1e-4 and alpha == 1:
                        hard_case = 1
                        break
                    else:
                        hard_case = 0
                        step = full_eigvecs[0,1:]/full_eigvecs[0,0]
                        step /= alpha
                        step_norm = np.sqrt(np.dot(step, step.T))

                        #print("step norm2",step_norm)
                        if step_norm > trust_radius:
                            alpha_min = alpha
                            alpha = alpha * 10
                        else:
                            alpha_max = alpha
                            break

                print("alpha range", alpha_min, alpha_max, flush = True)
                #end = timer()
                #print("finding alpha range took", end - start)
                #start = timer()
                if alpha_max != 1:
                    while True:
                        #print(alpha_min, alpha_max)
                        alpha = 0.5 * (alpha_min + alpha_max)
                        ##reduced_gradient1 = alpha * reduced_gradient
                        ##augmented_hessian[0,1:] = reduced_gradient1
                        ##augmented_hessian[1:,0] = reduced_gradient1.T
                        ##augmented_hessian[1:,1:] = reduced_hessian
                        ##S2 = np.einsum("pq, qr->rp", augmented_hessian, Q.T)
                        #S = copy.deepcopy(S3)
                        #S[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[:,0])
                        #S[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[:,1:] )

                        ##print("aipo2", np.allclose(S2,S, rtol=1e-14,atol=1e-14))


                        #projected_augmented_hessian = np.dot(Q, S.T)
                        #theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                        ##print("tymn",theta)
                        #full_eigvecs=np.dot(aug_eigvecs.T[:nroots,:], Q)
                        #print(alpha) 
                        H_pp = np.zeros((indim, indim))
                        scaled_guess_gradient = guess_gradient
                        H_pp[0,1:] = alpha * scaled_guess_gradient
                        H_pp[1:,0] = alpha * scaled_guess_gradient.T
                        H_pp[1:,1:] = guess_hessian
                        H_qp = np.zeros((b_dim, indim))
                        Sq = copy.deepcopy(Sq0)
                        #Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                        Sq[:,1:] += alpha * gradient1 
                        #Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                        Sq[:,0] = alpha * gradient2 
                        for i in range(b_dim):
                            for j in range(dim1):
                                index1 = idx[j]
                                H_qp[i][0] = Sq[i][0]
                                H_qp[i][j+1] = Sq[i][index1+1]

                        H_qq = np.dot(Sq, Q[indim:,:].T)
                        H1 = np.concatenate((H_pp, H_qp), axis = 0)
                        H2 = np.concatenate((H_qp.T, H_qq), axis = 0)
                        projected_augmented_hessian = np.concatenate((H1, H2), axis = 1)

                        
                        theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                        #print("zopp",theta)
                        #print(np.shape(aug_eigvecs))
                        #print(aug_eigvecs.T[0,:])
                        full_eigvecs=np.dot(aug_eigvecs.T[:nroots,indim:], Q[indim:,])
                        full_eigvecs2 = copy.deepcopy(full_eigvecs)
                        #print("q component")
                        #print(full_eigvecs[0,:])

                        for i in range(nroots):
                            for j in range(dim1):
                                index1 = idx[j]
                                full_eigvecs[i][0] = aug_eigvecs[0][i]
                                #full_eigvecs2[i][0] = aug_eigvecs[0][i]
                                full_eigvecs[i][index1+1] = aug_eigvecs[j+1][i]
                                #full_eigvecs2[i][index1+1] = aug_eigvecs[j+1][i]
                        #print("p component")
                        #print(full_eigvecs2[0,:])


                        #print(np.shape(full_eigvecs))
                        step = full_eigvecs[0,1:]/full_eigvecs[0,0]
                        #print(step)
                        step /= alpha
                        step_norm = np.sqrt(np.dot(step, step.T))
                        #if np.abs((step_norm - trust_radius)/trust_radius) > 1e-3:
                        if trust_radius - step_norm <= 1e-2 and trust_radius - step_norm >= 0.0:
                            break
                        elif trust_radius - step_norm > 1e-2:
                            alpha_max = alpha
                        else:
                            alpha_min = alpha
                #end = timer()
                #print("finding valid alpha took", end - start)

                ##reduced_gradient1 = alpha * reduced_gradient
                ##augmented_hessian = np.zeros((dim2, dim2))
                ##augmented_hessian[0,1:] = reduced_gradient1
                ##augmented_hessian[1:,0] = reduced_gradient1.T
                ##augmented_hessian[1:,1:] = reduced_hessian
                ##print("augmented_hessian")
                ##self.printA(augmented_hessian)
                #print("alpha", alpha, flush = True)
                #print("projection of augmented hessian by blocks")
                #H_pp = np.zeros((indim, indim))
                #scaled_guess_gradient = guess_gradient
                #H_pp[0,1:] = alpha * scaled_guess_gradient
                #H_pp[1:,0] = alpha * scaled_guess_gradient.T
                #H_pp[1:,1:] = guess_hessian
                #b_dim = L - indim
                #H_qp = np.zeros((b_dim, indim))
                #Sq = np.zeros((b_dim, H_dim))
                #self.orbital_sigma(U, A_tilde, G1, Q[indim:,:], Sq, b_dim, 1)
                #Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                #Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                #for i in range(b_dim):
                #    for j in range(dim1):
                #        index1 = idx[j]
                #        H_qp[i][0] = Sq[i][0]
                #        H_qp[i][j+1] = Sq[i][index1+1]

                #H_qq = np.dot(Sq, Q[indim:,:].T)
                #H1 = np.concatenate((H_pp, H_qp), axis = 0)
                #H2 = np.concatenate((H_qp.T, H_qq), axis = 0)
                #H3 = np.concatenate((H1, H2), axis = 1)
                #theta1, aug_eigvecs1 = np.linalg.eigh(H3)

                #theta1, alpha1 = np.linalg.eigh(augmented_hessian)
                #print(theta1[0])
                #print(full_eigvecs[0,:])
                start = timer()
                w = np.zeros((nroots, H_dim))
                #self.build_sigma_reduced2(U, A_tilde, G, full_eigvecs, w, nroots, 1)
                self.orbital_sigma(U, A_tilde, G1, full_eigvecs, w, nroots, 1)
                #c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, full_eigvecs, w, nroots, 1, self.nmo, self.index_map_size, self.n_occupied)
                w[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, full_eigvecs[:,0])
                w[:,0] = alpha * np.dot(full_eigvecs[:,1:], reduced_gradient)
                residual_norm = np.zeros((nroots))
                unconverged_idx = []
                convergence_check = np.zeros((nroots), dtype=str)
                conv = 0
                #print(np.shape(Q),np.shape(S), np.shape(aug_eigvecs) )
                for j in range(nroots):
                    # Compute a residual vector "w" for each root we seek
                    w[j, :] -= theta[j] * full_eigvecs[j,:]
                    #w[j, :] = np.dot(aug_eigvecs[:, j].T, S) - theta[j] * np.dot(aug_eigvecs[:, j].T, Q)
                    #residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                    residual_norm[j] = np.linalg.norm(w[j, :])
                    if residual_norm[j] < threshold:
                        conv += 1
                        convergence_check[j] = "Yes"
                    else:
                        unconverged_idx.append(j)
                        convergence_check[j] = "No"
                print(unconverged_idx)

                print("root", "AH residual norm", "Eigenvalue", "Convergence")
                for j in range(nroots):
                    print(
                        j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
                    )

                if conv == nroots:
                    Q1 = full_eigvecs
                    if hard_case == 0:
                        #print(np.shape(Q1))
                        davidson_step[:,:] = Q1[0,1:]/Q1[0,0]
                        davidson_step[:,:] /= alpha
                    else:
                        davidson_step[:,:] = Q1[0,1:]
                    self.mu_min = theta[0]    
                    #print("davidson step", davidson_step, flush = True)
                    print("norm of solution", np.linalg.norm(davidson_step), flush = True)
                    print("converged!", flush = True)
                    break

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

                end   = timer()
                print("residual took", end -start)
                start = timer()
                if (Lmax-L < len(unconverged_idx)):
                    t_collapsing_begin = time.time()
                    #unconverged_w = np.zeros((len(unconverged_idx),H_dim))
                    Q= Q[:indim,:]                   
                    Q = np.concatenate((Q,full_eigvecs2),axis=0)
                    Q = np.concatenate((Q,preconditioned_w),axis=0)
                    print(Q.shape)
                    #Q=np.column_stack(Qtup)
                    c_gram_schmidt_add(Q, indim, H_dim, nroots+len(preconditioned_w))
                    #gc.collect()
                    t_collapsing_end = time.time()
                    print('restart took',t_collapsing_end-t_collapsing_begin,'seconds')
                else:
                    t_expanding_begin = time.time()
                    
                    Q = np.concatenate((Q,preconditioned_w),axis=0)
                    c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                    #gc.collect() 
                    t_expanding_end = time.time()
                    print('expand took',t_expanding_end-t_expanding_begin,'seconds')
                    #print(Q)
                end   = timer()
 
        return Q, hard_case



















































    def Davidson_linear_matrix_equation_solve(self, U, A_tilde, G, reduced_hessian_diagonal, reduced_gradient, eigval, reduced_hessian, hessian_tilde):
        threshold = 1e-6
        
        A3_tilde =  (A_tilde + A_tilde.T)
        
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        
        hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])


        H_diag = copy.deepcopy(reduced_hessian_diagonal)
        print(H_diag)
        H_diag = H_diag - eigval
        indim = 1
        #maxdim = indim + min(self.index_map_size//3, 100) 
        maxdim = 5  
        Lmax = maxdim
        H_dim = self.index_map_size
        print(H_dim, indim, maxdim)
        L = indim
        theta = [0.0] * L
        Q = np.zeros((indim, H_dim))
        
        Q[0,:] = np.divide(-reduced_gradient, H_diag, out=np.zeros_like(H_diag), where=H_diag!=0)
        print(Q)
        Q = Q/np.linalg.norm(Q)
        H_lambda = reduced_hessian - eigval * np.eye(self.index_map_size)
        #step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
        #        
        #Q = step_limit.reshape((1, H_dim))
        self.printA(H_lambda)
        aaa,ccc = np.linalg.eigh(H_lambda)
        print(np.linalg.matrix_rank(H_lambda))
        print(aaa)
        print(np.linalg.cond(H_lambda))
        num_iter = 20
        for davidson_iteration in range(0, num_iter):
            print("\n")
            L = Q.shape[0]
            print("iteration", davidson_iteration+1, "dimension", L, flush = True)
            S = np.zeros_like(Q)
            #S1 = np.einsum("pq, qr->rp", reduced_hessian, Q.T)
            S1 = np.einsum("pq, qr->rp", H_lambda, Q.T)
            QQ = np.einsum("pq,qr->pr", Q, Q.T)
            print("check orthogonality",QQ)


            
          
            #self.build_sigma_reduced2(U, A_tilde, G, Q, S, L, 0)
            self.build_sigma_reduced3(U, hessian_tilde, G, Q, S, L, 0)
            S -= eigval * Q
            print("bn5k", np.allclose(S1,S, rtol=1e-14,atol=1e-14))
            projected_hessian = np.dot(Q, S.T)
            print("conditioner number of projected matrix", np.linalg.cond(projected_hessian))
            projected_gradient = np.einsum("pq,q->p", Q, reduced_gradient)
            print(np.shape(Q), np.shape(reduced_gradient), np.shape(projected_gradient))
            projected_step = np.linalg.solve(projected_hessian, -projected_gradient)
            step = np.einsum("pq,q->p", Q.T, projected_step)
            print(step)
            step1 = np.zeros((1, H_dim))
            step1[0,:] = np.copy(step)
            S2 = np.zeros_like(step1)
            #self.build_sigma_reduced2(U, A_tilde, G, step1, S2, 1, 0)
            self.build_sigma_reduced3(U, hessian_tilde, G, step1, S2, 1, 0)
            S2 -= eigval * step1
            S3 = S2.flatten()
            w = np.zeros((1,H_dim))
            nroots =1
            unconverged_idx = []
            convergence_check = np.zeros((nroots), dtype=str)
            conv = 0
            residual_norm = np.zeros((nroots))
            # Compute a residual vector "w" for each root we seek
            for j in range(nroots):
                # Compute a residual vector "w" for each root we seek
                w[j, :] = -reduced_gradient.reshape((1, H_dim))[j,:] - S2[j,:] 
                residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
                if residual_norm[j] < threshold:
                    conv += 1
                    convergence_check[j] = "Yes"
                else:
                    unconverged_idx.append(j)
                    convergence_check[j] = "No"
            print(unconverged_idx)
            print("root", "linear equation residual norm",  "Convergence")
            for j in range(nroots):
                print(
                    j + 1, residual_norm[j], convergence_check[j], flush=True
                )

            if conv == nroots:
                print("converged!")
                break




            # preconditioned_w = np.zeros((len(unconverged_idx),H_dim))
            if len(unconverged_idx) > 0:
                
                preconditioned_w = np.divide(
                    w[unconverged_idx],
                    H_diag,
                    out=np.zeros_like(w[unconverged_idx]),
                    where=H_diag != 0,
                )

            if (Lmax-L < len(unconverged_idx)):
                t_collapsing_begin = time.time()
                #unconverged_w = np.zeros((len(unconverged_idx),H_dim))
                Q=np.copy(step1)
                
                #for i in range(len(unconverged_idx)):
                #    unconverged_w[i,:]=w[unconverged_idx[i],:] 
                
                #Q=np.append(Q,unconverged_w)
                #print(Q)
                #print(unconverged_w)
                Q = np.concatenate((Q,preconditioned_w),axis=0)
                print(Q.shape)
                #Q=np.column_stack(Qtup)
                c_gram_schmidt_orthogonalization(Q, nroots+len(unconverged_idx), H_dim)
                # These vectors will give the same eigenvalues at the next
                # iteration so to avoid a false convergence we reset the theta
                # vector to theta_old
                #gc.collect()
                t_collapsing_end = time.time()
                print('restart took',t_collapsing_end-t_collapsing_begin,'seconds')
            else:
                t_expanding_begin = time.time()
                Q = np.concatenate((Q,preconditioned_w),axis=0)
                c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                #gc.collect() 
                t_expanding_end = time.time()
                print('expand took',t_expanding_end-t_expanding_begin,'seconds')
                #print(Q)
 







    def build_orbital_hessian_element(self, U, sym_A_tilde, G, r, k, s, l):
        temp1 = np.copy(G[k,l,:,:])
        temp2 = np.copy(U[:,s])
        temp3 = np.copy(U[:,r])
        temp4 = np.zeros((self.nmo))
        temp4 = np.einsum("pq, q ->p", temp1, temp2)
        a = np.einsum("p, p ->", temp3, temp4)
        a -= 0.5 * (k==l) * (sym_A_tilde[r][s])
        a -= 0.5 * (r==s) * (sym_A_tilde[k][l])
        #print(np.shape(temp1))
        if r < self.n_occupied: 
            temp1 = np.copy(G[r,l,:,:])
            temp3 = np.copy(U[:,k])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a -= np.einsum("p, p ->", temp3, temp4)
            a += 0.5 * (r==l) * (sym_A_tilde[k][s])
        if s < self.n_occupied: 
            temp1 = np.copy(G[k,s,:,:])
            temp2 = np.copy(U[:,l])
            temp3 = np.copy(U[:,r])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a -= np.einsum("p, p ->", temp3, temp4)
            a += 0.5 * (k==s) * (sym_A_tilde[r][l])
        if r < self.n_occupied and s < self.n_occupied: 
            temp1 = np.copy(G[r,s,:,:])
            temp2 = np.copy(U[:,l])
            temp3 = np.copy(U[:,k])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a += np.einsum("p, p ->", temp3, temp4)
        return a
    def build_orbital_hessian_guess(self, U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, idx):
        index_map = self.index_map
        nmo = self.nmo
        n_occupied = self.n_occupied
        return self.hessian_guess(U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, nmo, n_occupied,index_map, idx)
    @staticmethod    
    #@nb.njit("""void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,:,:,::1], float64[:,::1], float64[:,::1],
    #        int64, int64, int64, int64, int64)""", fastmath = True, parallel = True) 
    #def build_sigma_reduced6(U, A_tilde, index_map, G1, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
    @nb.njit("""void(float64[:,::1], float64[:,::1], float64[::1], float64[:,:,:,::1], float64[:,::1], float64[::1],
            int64, int64, int64, int64[:,::1], int64[::1])""", fastmath = True, parallel = True) 

    def hessian_guess(U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, nmo, n_occupied, index_map, idx):
        for i in nb.prange(dim1):
            index1 = idx[i]
            r = index_map[index1][0] 
            k = index_map[index1][1]
            guess_gradient[i] = reduced_gradient[index1]
            for j in range(dim1):
                index2 = idx[j]
                s = index_map[index2][0] 
                l = index_map[index2][1]
                #print(r,k,s,l)
                #b = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
                temp1 = np.copy(G[k,l,:,:])
                temp2 = np.copy(U[:,s])
                temp3 = np.copy(U[:,r])
                temp4 = np.zeros((nmo))
                #temp4 = np.einsum("pq, q ->p", temp1, temp2)
                temp4 = np.dot(temp1, temp2)
                #a = np.einsum("p, p ->", temp3, temp4)
                a = np.dot(temp3, temp4)
                a -= 0.5 * (k==l) * (sym_A_tilde[r][s])
                a -= 0.5 * (r==s) * (sym_A_tilde[k][l])
                #print(np.shape(temp1))
                if r < n_occupied: 
                    temp1 = np.copy(G[r,l,:,:])
                    temp3 = np.copy(U[:,k])
                    #temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    #a -= np.einsum("p, p ->", temp3, temp4)
                    a -= np.dot(temp3, temp4)
                    a += 0.5 * (r==l) * (sym_A_tilde[k][s])
                if s < n_occupied: 
                    temp1 = np.copy(G[k,s,:,:])
                    temp2 = np.copy(U[:,l])
                    temp3 = np.copy(U[:,r])
                    #temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    #a -= np.einsum("p, p ->", temp3, temp4)
                    a -= np.dot(temp3, temp4)
                    a += 0.5 * (k==s) * (sym_A_tilde[r][l])
                if r < n_occupied and s < n_occupied: 
                    temp1 = np.copy(G[r,s,:,:])
                    temp2 = np.copy(U[:,l])
                    temp3 = np.copy(U[:,k])
                    #temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    #a += np.einsum("p, p ->", temp3, temp4)
                    a += np.dot(temp3, temp4)
                guess_hessian[i][j] = a 
                #print(b, hessian_tilde[r*self.n_occupied+k][s*self.n_occupied+l])

    def microiteration_step(self, gradient_tilde, hessian_tilde, step, alpha, dim0):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))
        #print("gradient_tilde", gradient_tilde, flush = True)
        augmented_hessian[0,1:] = gradient_tilde
        augmented_hessian[1:,0] = gradient_tilde.T
        augmented_hessian[1:,1:] = hessian_tilde
        #print("hessian")
        #self.printA(hessian_tilde)
        #print("augmented_hessian")
        #self.printA(augmented_hessian)
        #print(" ", flush = True)
        #U_a = np.zeros((dim0, dim0))
        #U_a[0][0] = 1
        #U_a[1:,1:] = self.w1
        #A_b = np.einsum("rp,rs,sq->pq", U_a, augmented_hessian, U_a)
        #mu2, w2 = np.linalg.eigh(A_b)
        #print("eigenvalues of transformed augmented hessian", mu2)
        #print("eigenvectors of transformed augmented hessian",w2)

        #print("augmented_hessian", augmented_hessian, flush = True)
        #print("transformed augmented hesssian", A_b)
        mu, w = np.linalg.eigh(augmented_hessian)

        idx = mu.argsort()[:dim0]
        #print(idx)
        np.set_printoptions(precision = 14)
        print("check eigenvalues and eigenvectors of the augmented hessian", flush = True)
        print(mu[0], mu[1], flush=True)
        #print("w",w, flush = True)
        print("alpha",alpha, flush = True)
        scale = w[0][0]
        w[:,0] = w[:,0]/scale
        #print(w)
        step0 = w[1:,0]
        #print(step0)
        step[:] = step0/alpha
        #h_inverse = np.linalg.inv(hessian_tilde_ai)
        #step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai) 
        step_norm = np.linalg.norm(step)
        #print("step", step)
        print("step norm", step_norm, flush=True)
        return step_norm
 
    def microiteration_step2(self, gradient_tilde, hessian_tilde, step, alpha, dim0):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))

        augmented_hessian[0,1:] = gradient_tilde
        augmented_hessian[1:,0] = gradient_tilde.T
        augmented_hessian[1:,1:] = hessian_tilde
        #U_a = np.zeros((dim0, dim0))
        #U_a[0][0] = 1
        #U_a[1:,1:] = self.w1
        #A_b = np.einsum("rp,rs,sq->pq", U_a, augmented_hessian, U_a)
        #mu2, w2 = np.linalg.eigh(A_b)
        #print("eigenvalues of transformed augmented hessian", mu2)
        #print("eigenvectors of transformed augmented hessian",w2)

        #print("augmented_hessian", augmented_hessian)
        #print("transformed augmented hesssian", A_b)
        mu, w = np.linalg.eigh(augmented_hessian)
        idx = mu.argsort()[:dim0]
        #print(idx)

        np.set_printoptions(precision = 14)
        print("check eigenvalues and eigenvectors of the augmented hessian")
        #print("w",w)
        print(mu, flush=True)
        scale = w[0][0]
        #w[:,0] = w[:,0]/scale
        #print(w)
        step0 = w[1:,0]
        product = np.dot(gradient_tilde, step0.flatten())
        print("product of the gradient and the step from augmented hessian", product)
        #print(step0)
        step[:] = step0/alpha
        #h_inverse = np.linalg.inv(hessian_tilde_ai)
        #step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai) 
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        #print("step norm", step_norm, flush=True)
        return step_norm, scale
 




    def projection_step(self, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))
        #print("projected_gradient", gradient_tilde, flush = True)
        augmented_hessian[0,1:] = gradient_tilde
        augmented_hessian[1:,0] = gradient_tilde.T
        augmented_hessian[1:,1:] = hessian_tilde
        #print("projected_hessian")
        #self.printA(hessian_tilde)
        #print("projected_augmented_hessian")
        #self.printA(augmented_hessian)
        #print(" ", flush = True)
        
        mu, aug_eigenvecs[:,:] = np.linalg.eigh(augmented_hessian)
        #np.set_printoptions(precision = 14)
        #print("check eigenvalues and eigenvectors of the projected augmented hessian", flush = True)
        #print("eigenvecs",aug_eigenvecs, flush = True)
        #print("alpha",alpha, flush = True)
        #print(mu, flush=True)
        return mu 
 
    def projection_step2(self, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0):
        augmented_hessian = np.zeros((dim0, dim0))
        #print("projected_gradient", gradient_tilde, flush = True)
        
        augmented_hessian[0,0] = alpha 
        augmented_hessian[0,1:] = gradient_tilde
        augmented_hessian[1:,0] = gradient_tilde.T
        augmented_hessian[1:,1:] = hessian_tilde
        #print("projected_hessian")
        #self.printA(hessian_tilde)
        #print("projected_augmented_hessian")
        #self.printA(augmented_hessian)
        #print(" ", flush = True)

        mu, aug_eigenvecs[:,:] = np.linalg.eigh(augmented_hessian)
        #np.set_printoptions(precision = 14)
        #print("check eigenvalues and eigenvectors of the projected augmented hessian", flush = True)
        #print("eigenvecs",aug_eigenvecs, flush = True)
        #print("alpha",alpha, flush = True)
        #print(mu, flush=True)
        return mu
    

    def mv(self, U, A_tilde, G, R_reduced, num_states, pointer, eigval):
        R_reduced = R_reduced.reshape((1, self.index_map_size))
        sigma_reduced = np.zeros_like(R_reduced) 
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw", num_states) 
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                R_total[i][r][k] = R_reduced[i][j+pointer]
                
           
        A3_tilde =  (A_tilde + A_tilde.T)
        
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        
        hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])


        sigma_total = np.zeros((num_states, self.nmo, self.n_occupied))
        
        temp1 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp2 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,isl->iql", U, R_total)
        temp2 = np.einsum("klpq,iql->ipk", G, temp1)
        sigma_total = np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:,:self.n_occupied,:] -= np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        
        temp1 = np.einsum("ql,isl->iqs", U[:,:self.n_occupied], R_total[:,:self.n_occupied,:])
        temp2 = np.einsum("kspq,iqs->ipk", G, temp1)
        sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:,:self.n_occupied,:] += np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        sigma_total += np.einsum("klrs,isl->irk", hessian_tilde, R_total)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0] 
                k = self.index_map[j][1] 
                sigma_reduced[i][j+pointer] = sigma_total[i][r][k]
        sigma_reduced -= eigval * R_reduced        
        sigma_reduced = sigma_reduced.flatten()
        return sigma_reduced


    def mv2(self, U, A_tilde, G, R_reduced, num_states, pointer, eigval):
        R_reduced = R_reduced.reshape((1, self.index_map_size))
        nmo = self.nmo
        index_map = self.index_map
        index_map_size = self.index_map_size
        n_occupied = self.n_occupied
        return self.build_sigma_reduced5(U, A_tilde, index_map, G, R_reduced, num_states, pointer, nmo, index_map_size, n_occupied, eigval)
    
    @staticmethod    
    @nb.jit("""float64[::1](float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,::1], 
            int64, int64, int64, int64, int64, float64)""", nopython=True, cache = True, fastmath = True, parallel = True) 
    def build_sigma_reduced5(U, A_tilde, index_map, G, R_reduced, num_states, pointer, nmo, index_map_size, n_occupied, eigval):
        assert U.shape == (nmo, nmo)
        assert A_tilde.shape == (nmo, nmo)
        assert G.shape == (nmo * n_occupied, nmo * n_occupied)
        sigma_reduced = np.zeros((1, index_map_size))
        R_total = np.zeros((num_states, nmo, n_occupied))
        print("oivdpw",num_states) 
        for i in range(num_states):
            for j in range(index_map_size):
                r = index_map[j][0] 
                k = index_map[j][1] 
                R_total[i][r][k] = R_reduced[i][j+pointer]
                
         
       
        sigma_total = np.zeros((num_states, nmo, n_occupied))
        

        R1 = R_total.transpose(1,0,2)
        R1 = np.ascontiguousarray(R1) 
        R1 = np.reshape(R1,(nmo, num_states * n_occupied)) 
        temp2 = np.dot(U, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)
        R1 = np.ascontiguousarray(R_total[:,:n_occupied,:].transpose(2,0,1)) 
        #R_total1 = np.ascontiguousarray(R_total1) 
        R1 = np.reshape(R1,(n_occupied, num_states * n_occupied))       
        U1 = np.ascontiguousarray(U[:,:n_occupied])
        temp2 = np.dot(U1, R1)
        temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)

        ##sigma_total1 = np.zeros((num_states, nmo, n_occupied))
        ##temp11 = np.einsum("qs,isl->iql", U, R_total)
        #for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for l in range(n_occupied):
        #            #a = 0.0
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += U[q,s] * R_total[i,s,l]
        #                #temp1[i,q,l] += U[q,s] * R_total[i,s,l]
        #            temp1[i,q,l] = a 
        #

        #for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for s in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a -= U[q,l] * R_total[i,s,l]
        #                #temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
        #            temp1[i,q,s] += a 
        R1 = np.ascontiguousarray(temp1) 
        R1 = np.reshape(R1, (num_states, nmo * n_occupied))
        R2 = np.zeros((num_states, nmo * n_occupied))
        ##for i in nb.prange(num_states):
        ##    for r in range(nmo):
        ##        for k in range(n_occupied):
        ##            a = R1[i][r*n_occupied+k] - temp1[i,r,k]
        ##            if np.abs(a) > 1e-14: print("LARGE error")

        temp1 = np.dot(R1, G)
        #temp2 = temp1.reshape(num_states, nmo, n_occupied)
        #for r in nb.prange(nmo):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for p in range(nmo):
        #                a += U[p,r] * temp2[i,p,k]
        #                #sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
        #            temp1[i,r,k] = a 
        #            sigma_total[i,r,k] = a 
        #

        #for r in nb.prange(n_occupied):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] -= temp1[i,k,r] 
        

        #######for i in nb.prange(num_states):
        #######    R3 = np.dot(R1[i,:], G)
        #######    R2[i,:] = R3 

        temp2 = temp1.reshape(num_states, nmo, n_occupied)
        temp2 = temp2.transpose(0,2,1) 
        temp2 = np.ascontiguousarray(temp2)
        temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
        temp1 = np.dot(temp2, U)
        temp2 = temp1.reshape(num_states, n_occupied, nmo)
        sigma_total[:,:,:] = temp2.transpose(0,2,1)[:,:,:]  
        sigma_total[:,:n_occupied,:] -= temp2[:,:,:n_occupied]
                
        A3_tilde =  (A_tilde + A_tilde.T)
        #temp1 = np.zeros((num_states, nmo, n_occupied))
        #temp2 = np.zeros((num_states, nmo, n_occupied))

        ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        ##hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += A3_tilde[r,s] * R_total[i,s,k]
        #            temp1[i,r,k] = a 
        #            sigma_total[i,r,k] -= 0.5 * a 
        #
        #for i in nb.prange(num_states):
        #    for r in range(n_occupied):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r] 
        #
        ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[k,l] * R_total[i,r,l]
        #            sigma_total[i,r,k] -= 0.5 * a

        ##hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
        #for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[r,l] * R_total[i,k,l]
        #            sigma_total[i,r,k] += 0.5 * a
        R1 = R_total.transpose(1,0,2) 
        R1 = np.ascontiguousarray(R1) 
        R1 = np.reshape(R1,(nmo, num_states * n_occupied))
        temp2 = np.dot(A3_tilde, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1,0,2)
        sigma_total[:,:,:] -= 0.5 * temp1[:,:,:]
        sigma_total[:,:n_occupied,:] += 0.5 * temp1.transpose(0,2,1)[:,:,:n_occupied]
        A3 = A3_tilde[:n_occupied, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total)
        R1 = np.reshape(R1,(num_states * nmo, n_occupied))
        temp2 = np.dot(R1, A3)
        sigma_total[:,:,:] -= 0.5 * temp2.reshape(num_states, nmo, n_occupied)[:,:,:]
        A3 = A3_tilde[:, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total[:,:n_occupied,:])
        R1 = np.reshape(R1,(num_states * n_occupied, n_occupied))
        temp2 = np.dot(R1, A3)
        temp1 = temp2.reshape(num_states, n_occupied, nmo)
        sigma_total[:,:,:] += 0.5 * temp1.transpose(0,2,1)[:,:,:]
        


        #####sigma_total1 = np.zeros((num_states, nmo, n_occupied))
        #####temp11 = np.einsum("qs,isl->iql", U, R_total)
        ####for q in nb.prange(nmo):
        ####    for i in range(num_states):
        ####        for l in range(n_occupied):
        ####            for s in range(nmo):
        ####                temp1[i,q,l] += U[q,s] * R_total[i,s,l]
        ####

        ####for q in nb.prange(nmo):
        ####    for i in range(num_states):
        ####        for s in range(n_occupied):
        ####            for l in range(n_occupied):
        ####                temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
        ####

        #####print("aipo1", np.allclose(temp11,temp1, rtol=1e-14,atol=1e-14))
        #####temp22 = np.einsum("klpq,iql->ipk", G, temp11)
        ####for p in nb.prange(nmo):
        ####    for i in range(num_states):
        ####        for k in range(n_occupied):
        ####            for q in range(nmo):
        ####                for l in range(n_occupied):
        ####                    temp2[i,p,k] += G[k,l,p,q] * temp1[i,q,l]
        ####
        #####print("aipo2", np.allclose(temp22,temp2, rtol=1e-14,atol=1e-14))
        #####sigma_total1 = np.einsum("ipk,pr->irk", temp22, U, optimize ="optimal")
        ####for r in nb.prange(nmo):
        ####    for i in range(num_states):
        ####        for k in range(n_occupied):
        ####            for p in range(nmo):
        ####                sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
        #####print("aipo3", np.allclose(sigma_total1,sigma_total, rtol=1e-14,atol=1e-14))
        ####

        #####sigma_total1[:,:n_occupied,:] -= np.einsum("ipr,pk->irk", temp22, U[:,:n_occupied])
        ####for r in nb.prange(n_occupied):
        ####    for i in range(num_states):
        ####        for k in range(n_occupied):
        ####            for p in range(nmo):
        ####                sigma_total[i,r,k] -= U[p,k] * temp2[i,p,r]
        #####print("aipo4", np.allclose(sigma_total1,sigma_total, rtol=1e-14,atol=1e-14))
        ####
        #####temp11 = np.einsum("ql,isl->iqs", U[:,:n_occupied], R_total[:,:n_occupied,:])
        #####temp1[:,:,:] = 0.0
        #####for q in nb.prange(nmo):
        #####    for i in range(num_states):
        #####        for s in range(n_occupied):
        #####            for l in range(n_occupied):
        #####                temp1[i,q,s] += U[q,l] * R_total[i,s,l]
        #####print("aipo5", np.allclose(temp11,temp1, rtol=1e-14,atol=1e-14))
        ####
        #####temp22 = np.einsum("kspq,iqs->ipk", G, temp11, optimize = "optimal")
        #####temp2[:,:,:] = 0.0
        #####for p in nb.prange(nmo):
        #####    for i in range(num_states):
        #####        for k in range(n_occupied):
        #####            for q in range(nmo):
        #####                for l in range(n_occupied):
        #####                    temp2[i,p,k] += G[k,l,p,q] * temp1[i,q,l]
        #####print("aipo6", np.allclose(temp22,temp2, rtol=1e-14,atol=1e-14))
        ####
        #####sigma_total1 -= np.einsum("ipk,pr->irk", temp22, U)
        #####for r in nb.prange(nmo):
        #####    for i in range(num_states):
        #####        for k in range(n_occupied):
        #####            for p in range(nmo):
        #####                sigma_total[i,r,k] -= U[p,r] * temp2[i,p,k]
        #####print("aipo7", np.allclose(sigma_total1,sigma_total, rtol=1e-14,atol=1e-14))
        ####
        #####sigma_total1[:,:n_occupied,:] += np.einsum("ipr,pk->irk", temp22, U[:,:n_occupied])
        #####for r in nb.prange(n_occupied):
        #####    for i in range(num_states):
        #####        for k in range(n_occupied):
        #####            for p in range(nmo):
        #####                sigma_total[i,r,k] += U[p,k] * temp2[i,p,r]
        #####print("aipo8", np.allclose(sigma_total1,sigma_total, rtol=1e-14,atol=1e-14))

        #####sigma_total1 += np.einsum("klrs,isl->irk", hessian_tilde, R_total, optimize = "optimal")
        ####for r in nb.prange(nmo):
        ####    for i in range(num_states):
        ####        for k in range(n_occupied):
        ####            for s in range(nmo):
        ####                for l in range(n_occupied):
        ####                    sigma_total[i,r,k] += hessian_tilde[k,l,r,s] * R_total[i,s,l]
        #####print("aipo9", np.allclose(sigma_total1,sigma_total, rtol=1e-14,atol=1e-14))


        #####sigma_total4[:,:,:] =sigma_total[:,:,:] 
        #####sigma_total3[:,:,:] =sigma_total[:,:,:]
        
   
         
     
        for i in range(num_states):
            for j in range(index_map_size):
                r = index_map[j][0] 
                k = index_map[j][1] 
                sigma_reduced[i][j+pointer] = sigma_total[i][r][k] 
                #sigma_reduced[i][j+pointer] -= eigval * R_reduced[i][j+pointer]        
        sigma_reduced -= eigval * R_reduced        
        sigma_reduced = sigma_reduced.flatten()
        return sigma_reduced






































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
