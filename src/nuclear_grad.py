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

# from memory_profiler import profile
from helper_cqed_rhf import cqed_rhf
from helper_PFCI import *
from residual_minimization import *
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
from scipy.linalg import solve, LinAlgError
from typing import List, Tuple, Optional, Union
from timeit import default_timer as timer
import numba as nb
from numba import objmode


class nuclear_grad(PFHamiltonianGenerator):
    def build_B_t(self, Z_vector, eigenvecs, B_t, A):
        #self.h_tilde = np.einsum("sr,sj->rj", Z_vector, self.H_spatial2[:, :self.n_occupied], optimize = "optimal")
        #self.h_tilde += np.einsum("rs,sj->rj", self.H_spatial2, Z_vector[:, :self.n_occupied], optimize = "optimal")

        #self.d_cmo_tilde = np.einsum("sr,sj->rj", Z_vector, self.d_cmo[:, :self.n_occupied], optimize = "optimal")
        #self.d_cmo_tilde += np.einsum("rs,sj->rj", self.d_cmo, Z_vector[:, :self.n_occupied], optimize = "optimal")
        #self.J_tilde = np.einsum("sr,klsj->klrj", Z_vector, self.J[:, :, :, :self.n_occupied], optimize = "optimal")
        #self.J_tilde += np.einsum("sj,klrs->klrj", Z_vector[:, :self.n_occupied], self.J, optimize = "optimal")
        #self.J_tilde += np.einsum("sk,jlrs->klrj", Z_vector[:, :self.n_occupied], self.K, optimize = "optimal")
        #self.J_tilde += np.einsum("sl,jkrs->klrj", Z_vector[:, :self.n_occupied], self.K, optimize = "optimal")
        self.occupied_h_t = np.einsum("sr,sj->rj", Z_vector[:,:self.n_occupied], self.H_spatial2[:, :self.n_occupied], optimize = "optimal")
        self.occupied_h_t += np.einsum("rs,sj->rj", self.H_spatial2[:self.n_occupied,:], Z_vector[:, :self.n_occupied], optimize = "optimal")

        self.occupied_d_cmo_t = np.einsum("sr,sj->rj", Z_vector[:,:self.n_occupied], self.d_cmo[:, :self.n_occupied], optimize = "optimal")
        self.occupied_d_cmo_t += np.einsum("rs,sj->rj", self.d_cmo[:self.n_occupied,:], Z_vector[:, :self.n_occupied], optimize = "optimal")
        self.occupied_J_t = np.einsum("sr,klsj->klrj", Z_vector[:,:self.n_occupied], self.J[:, :, :, :self.n_occupied], optimize = "optimal")
        self.occupied_J_t += np.einsum("sj,klrs->klrj", Z_vector[:, :self.n_occupied], self.J[:,:,:self.n_occupied,:], optimize = "optimal")
        self.occupied_J_t += np.einsum("sk,jlrs->klrj", Z_vector[:, :self.n_occupied], self.K[:,:,:self.n_occupied,:], optimize = "optimal")
        self.occupied_J_t += np.einsum("sl,jkrs->klrj", Z_vector[:, :self.n_occupied], self.K[:,:,:self.n_occupied,:], optimize = "optimal")
        

        self.occupied_fock_core_t = copy.deepcopy(self.occupied_h_t)
        self.occupied_fock_core_t += 2.0 * np.einsum(
            "jjrs->rs", self.occupied_J_t[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        self.occupied_fock_core_t -= np.einsum(
            "rjsj->rs", self.occupied_J_t[:, :self.n_in_a, :, :self.n_in_a], optimize="optimal"
        )
        
        self.E_core_t = 0.0
        self.E_core_t += np.einsum("jj->", self.occupied_h_t[: self.n_in_a, : self.n_in_a])
        self.E_core_t += np.einsum("jj->", self.occupied_fock_core_t[: self.n_in_a, : self.n_in_a])
        self.active_fock_core_t = copy.deepcopy(
            self.occupied_fock_core_t[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied]
        )
        self.active_twoeint_t = copy.deepcopy(
            self.occupied_J_t[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ]
        )

        gkl2_t = copy.deepcopy(self.active_fock_core_t)
        gkl2_t -= 0.5 * np.einsum("kjjl->kl", self.active_twoeint_t)
        d_diag_t = 2.0 * np.einsum(
            "ii->", self.occupied_d_cmo_t[: self.n_in_a, : self.n_in_a]
        )
        occupied_J_t = self.occupied_J_t.reshape(
          self.n_occupied * self.n_occupied,
          self.n_occupied * self.n_occupied,
        )
        gkl2_t = np.ascontiguousarray(gkl2_t)
        occupied_J_t = np.ascontiguousarray(occupied_J_t)
        self.occupied_d_cmo_t = np.ascontiguousarray(self.occupied_d_cmo_t)

        L = self.davidson_roots
        S = np.zeros_like(eigenvecs)
        c_sigma_2(
               gkl2_t,
               occupied_J_t,
               self.occupied_d_cmo_t,
               eigenvecs,
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
               0,
               0,
               0,
               self.omega,
               -d_diag_t,
               self.E_core_t,
               self.break_degeneracy,
               )
        #diag_elements0 = np.sum(eigenvecs * S, axis = 1)
        #diag_elements  = diag_elements0 * self.weight
        ##print("diag element", diag_elements)
        ##B_t = W_n.2 H_t .c - 2 W_n(c^T .H_t .c) .c
        #B_t[:,:] = 2.0 * S * self.weight[:, np.newaxis]
        #B_t[:,:] += -2.0 * eigenvecs * diag_elements[:, np.newaxis]
        
        diag_elements0 = np.sum(eigenvecs * S, axis = 1)
        B_t[:,:] = 2.0 * S[:,:]
        B_t[:,:] += -2.0 * eigenvecs * diag_elements0[:, np.newaxis]
        

    def build_occupied_rdm(self, z_vector, eigenvecs, one_rdm_avg, two_rdm_avg, one_rdm_pe_avg, weight, z_state):
        np1 = self.N_p + 1
        one_rdm_avg[:] = 0.0 
        one_rdm_pe_avg[:] = 0.0 
        two_rdm_avg[:] = 0.0 
        for i in range(self.davidson_roots):
            if weight[i] == 0: continue
            one_rdm = np.zeros((self.n_occupied * self.n_occupied))
            c_build_one_rdm(
                z_vector,    
                eigenvecs,
                one_rdm,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                self.n_in_a,
                np1,
                i,
                i,
                z_state
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
                z_vector,    
                eigenvecs,
                two_rdm,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                self.n_in_a,
                np1,
                i,
                i,
                z_state
            )
            Dpe = np.zeros((self.n_occupied * self.n_occupied))
            c_build_photon_electron_one_rdm(
                z_vector,    
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
            one_rdm_avg[:] += weight[i] * one_rdm
            two_rdm_avg[:] += weight[i] * two_rdm
            one_rdm_pe_avg[:] += weight[i] *Dpe 
        ###symmetrize rdm
        for t in range(self.n_occupied):
            for u in range(t, self.n_occupied):
                tu = t * self.n_occupied + u
                ut = u * self.n_occupied + t
                dum = one_rdm_avg[tu] + one_rdm_avg[ut]
                one_rdm_avg[tu] = dum/2.0
                one_rdm_avg[ut] = dum/2.0
                dum = one_rdm_pe_avg[tu] + one_rdm_pe_avg[ut]
                one_rdm_pe_avg[tu] = dum/2.0
                one_rdm_pe_avg[ut] = dum/2.0
                for vw in range(self.n_occupied * self.n_occupied):
                    dum = (
                        two_rdm_avg[tu * self.n_occupied * self.n_occupied + vw]
                        + two_rdm_avg[ut * self.n_occupied * self.n_occupied + vw]
                    )
                    # dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    two_rdm_avg[tu * self.n_occupied * self.n_occupied + vw] = (
                        dum / 2.0
                    )
                    two_rdm_avg[ut * self.n_occupied * self.n_occupied + vw] = (
                        dum / 2.0
                    )
        #one_rdm_avg = self.one_rdm_avg.reshape(self.n_occupied, self.n_occupied)
        #one_rdm_pe_avg = self.one_rdm_pe_avg.reshape(self.n_occupied, self.n_occupied)
        #two_rdm_avg = self.two_rdm_avg.reshape(self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
    def build_effective_densities_mo(self, Z_vector, z_vector, eigenvecs, state):
        self.one_rdm_eff_mo = np.zeros((self.nmo, self.nmo))
        self.one_rdm_pe_eff_mo= np.zeros((self.nmo, self.nmo))
        self.two_rdm_eff_mo = np.zeros(
            (
                self.nmo
                , self.n_occupied
                , self.nmo
                , self.n_occupied
            )
        )
        one_rdm_avg = np.zeros((self.n_occupied * self.n_occupied))
        one_rdm_pe_avg = np.zeros((self.n_occupied * self.n_occupied))
        two_rdm_avg = np.zeros(
            (
                self.n_occupied
                * self.n_occupied
                * self.n_occupied
                * self.n_occupied
            )
        )
        weight = np.zeros_like(self.weight)
        weight[state] = 1.0
        self.build_occupied_rdm(eigenvecs, eigenvecs, one_rdm_avg, two_rdm_avg, one_rdm_pe_avg, weight, False)
        self.one_rdm_eff_mo[:self.n_occupied,:self.n_occupied] = one_rdm_avg.reshape(self.n_occupied, self.n_occupied)[:,:]
        self.one_rdm_pe_eff_mo[:self.n_occupied,:self.n_occupied] = one_rdm_pe_avg.reshape(self.n_occupied, self.n_occupied)[:,:]
        self.two_rdm_eff_mo[:self.n_occupied,:,:self.n_occupied,:] = 0.5 * two_rdm_avg.reshape(self.n_occupied, self.n_occupied,
                                                                           self.n_occupied, self.n_occupied)[:,:,:,:]
        
 
        #build intermediate for cp-hf equations
        self.E_d_state = np.dot(self.occupied_d_cmo.flatten(), one_rdm_avg)


        self.build_occupied_rdm(z_vector, eigenvecs, one_rdm_avg, two_rdm_avg, one_rdm_pe_avg, self.weight, True)
        self.one_rdm_eff_mo[:self.n_occupied,:self.n_occupied] += one_rdm_avg.reshape(self.n_occupied, self.n_occupied)[:,:]
        self.one_rdm_pe_eff_mo[:self.n_occupied,:self.n_occupied] += one_rdm_pe_avg.reshape(self.n_occupied, self.n_occupied)[:,:]
        self.two_rdm_eff_mo[:self.n_occupied,:,:self.n_occupied,:] += 0.5 *two_rdm_avg.reshape(self.n_occupied, self.n_occupied,
                                                                           self.n_occupied, self.n_occupied)[:,:,:,:]
       
        #self.print_matrix_nice(one_rdm_avg.reshape(self.n_occupied, self.n_occupied), precision=10, width=14, cols_per_line=6)
        #build intermediate for cp-hf equations
        self.E_d_z  = np.dot(self.occupied_d_cmo.flatten(), one_rdm_avg)
        

        self.build_occupied_rdm(eigenvecs, eigenvecs, one_rdm_avg, two_rdm_avg, one_rdm_pe_avg, self.weight, False)
        one_rdm_avg_full = np.zeros((self.nmo , self.nmo))
        one_rdm_avg_full[:self.n_occupied, :self.n_occupied] = one_rdm_avg.reshape(self.n_occupied, self.n_occupied)
        self.one_rdm_eff_mo[:,:] += np.dot(Z_vector, one_rdm_avg_full) + np.dot(one_rdm_avg_full, Z_vector.T)
       
        #build intermediate for cp-hf equations
        temp_Zd = 2.0 * np.einsum("ri,rj->ij",Z_vector[:,:self.n_occupied], self.d_cmo[:,:self.n_occupied])
        self.E_d_Z  = np.dot(temp_Zd.flatten(), one_rdm_avg)
               
        one_rdm_pe_avg_full = np.zeros((self.nmo , self.nmo))
        one_rdm_pe_avg_full[:self.n_occupied, :self.n_occupied] = one_rdm_pe_avg.reshape(self.n_occupied, self.n_occupied)
        self.one_rdm_pe_eff_mo[:,:] += np.dot(Z_vector, one_rdm_pe_avg_full) + np.dot(one_rdm_pe_avg_full, Z_vector.T)
        two_rdm_hf = np.zeros(
            (
                 self.n_occupied
                , self.n_occupied
                , self.nmo
                , self.nmo
                            )
        )
        two_rdm_hf[:,:,:self.n_occupied, :self.n_occupied] = two_rdm_avg.reshape(self.n_occupied, self.n_occupied,
                                                               self.n_occupied, self.n_occupied)[:,:,:,:].transpose(1,3,0,2)
        self.two_rdm_eff_mo[:,:,:,:] +=  np.einsum("pr, klrq->qlpk", Z_vector, two_rdm_hf)
        self.two_rdm_eff_mo[:,:,:,:] +=  np.einsum("klpr, qr->qlpk", two_rdm_hf, Z_vector)

    def update_effective_densities_ao(self, state, kappa, z_vector, eigenvecs, C):
        kappa_matrix = kappa.reshape(self.n_v_hf, self.ndocc)
        kappa_full = np.zeros((self.nmo, self.nmo))
        kappa_full[self.ndocc:,:self.ndocc] = kappa_matrix[:,:]
        kappa_temp1 = np.einsum("ai,ma->mi", kappa_matrix, C[:,self.ndocc:])
        one_rdm_kappa= np.einsum("mi,ni->mn", kappa_temp1, C[:,:self.ndocc])
       

        self.one_rdm_eff_ao += one_rdm_kappa 
        one_rdm_hf = 2.0 * np.einsum("mi,ni->mn", C[:,:self.ndocc], C[:,:self.ndocc])

        self.two_rdm_eff_ao += np.einsum("mn,pq->mnpq", one_rdm_kappa, one_rdm_hf)
        self.two_rdm_eff_ao -= 0.5 * np.einsum("mp,nq->mnpq", one_rdm_kappa, one_rdm_hf)
      

        off_diagonal_constant_state = self.calculate_off_diagonal_photon_constant_z(eigenvecs, eigenvecs, state)
        off_diagonal_constant_z = self.calculate_off_diagonal_photon_constant_z(z_vector, eigenvecs, -1)
        constant_hf = np.einsum("ai,ai->", kappa_matrix, self.d_hf[self.ndocc:,:self.ndocc])
        scale_factor = -self.E_d_state + self.d_exp - self.E_d_Z - self.E_d_z + off_diagonal_constant_state + off_diagonal_constant_z - constant_hf

        #self.one_rdm_pe_eff_ao3 = self.one_rdm_pe_eff_ao + (off_diagonal_constant_state + off_diagonal_constant_z) * one_rdm_hf
        self.one_rdm_pe_eff_ao += scale_factor * one_rdm_hf
        #print("constant_hf", constant_hf)
        #scale_factor2 = -self.E_d_state + self.d_exp - self.E_d_Z - self.E_d_z  - constant_hf
        #print("scale factor2", scale_factor2)
        #self.one_rdm_pe_eff_ao2 = scale_factor2 * one_rdm_hf

    def transform_effective_densities(self, C):
        two_rdm_1 = np.einsum("slpk,ms->mlpk", self.two_rdm_eff_mo, C)
        two_rdm_2 = np.einsum("mlpk,xp->mlxk", two_rdm_1, C)
        two_rdm_3 = np.einsum("mlxk,nl->mnxk", two_rdm_2, C[:,:self.n_occupied])
        self.two_rdm_eff_ao = np.einsum("mnxk,yk->mnxy", two_rdm_3, C[:,:self.n_occupied])

        one_rdm_mj = np.einsum("rs,mr->ms", self.one_rdm_eff_mo, C)
        self.one_rdm_eff_ao = np.einsum("ms,ns->mn", one_rdm_mj, C)
        #print("effective one-rdm in ao basis") 
        #self.print_matrix_nice(self.one_rdm_eff_ao, precision=10, width=14, cols_per_line=6)

        one_rdm_pe_mj = np.einsum("rs,mr->ms", self.one_rdm_pe_eff_mo, C)
        self.one_rdm_pe_eff_ao = -np.sqrt(self.omega/2.0) * np.einsum("ms,ns->mn", one_rdm_pe_mj, C)


    
    def build_full_rdm_avg(self, eigenvecs):
        np1 = self.N_p + 1
        self.one_rdm_avg = np.zeros((self.n_occupied * self.n_occupied))
        self.one_rdm_pe_avg = np.zeros((self.n_occupied * self.n_occupied))
        self.two_rdm_avg = np.zeros(
            (
                self.n_occupied
                * self.n_occupied
                * self.n_occupied
                * self.n_occupied
            )
        )
        for i in range(self.davidson_roots):
            one_rdm = np.zeros((self.n_occupied * self.n_occupied))
            c_build_one_rdm(
                eigenvecs,
                eigenvecs,
                one_rdm,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                self.n_in_a,
                np1,
                i,
                i,
                False
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
                eigenvecs,
                two_rdm,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                self.n_in_a,
                np1,
                i,
                i,
                False
            )
            Dpe = np.zeros((self.n_occupied * self.n_occupied))
            c_build_photon_electron_one_rdm(
                eigenvecs,
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
            self.one_rdm_avg += self.weight[i] * one_rdm
            self.two_rdm_avg += self.weight[i] * two_rdm
            self.one_rdm_pe_avg += self.weight[i] *Dpe 
        ###symmetrize rdm
        for t in range(self.n_occupied):
            for u in range(t, self.n_occupied):
                tu = t * self.n_occupied + u
                ut = u * self.n_occupied + t
                dum = self.one_rdm_avg[tu] + self.one_rdm_avg[ut]
                self.one_rdm_avg[tu] = dum/2.0
                self.one_rdm_avg[ut] = dum/2.0
                dum = self.one_rdm_pe_avg[tu] + self.one_rdm_pe_avg[ut]
                self.one_rdm_pe_avg[tu] = dum/2.0
                self.one_rdm_pe_avg[ut] = dum/2.0
                for vw in range(self.n_occupied * self.n_occupied):
                    dum = (
                        self.two_rdm_avg[tu * self.n_occupied * self.n_occupied + vw]
                        + self.two_rdm_avg[ut * self.n_occupied * self.n_occupied + vw]
                    )
                    # dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    self.two_rdm_avg[tu * self.n_occupied * self.n_occupied + vw] = (
                        dum / 2.0
                    )
                    self.two_rdm_avg[ut * self.n_occupied * self.n_occupied + vw] = (
                        dum / 2.0
                    )
        self.one_rdm_avg = self.one_rdm_avg.reshape(self.n_occupied, self.n_occupied)
        self.one_rdm_pe_avg = self.one_rdm_pe_avg.reshape(self.n_occupied, self.n_occupied)
        self.two_rdm_avg = self.two_rdm_avg.reshape(self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)

    def build_B_b(self, z_vector, B_b):
        L = self.davidson_roots
        S = np.zeros_like(z_vector)
        c_sigma(
               self.gkl2,
               self.occupied_J,
               self.occupied_d_cmo,
               z_vector,
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
               self.d_exp - self.d_diag,
               self.E_core,
               self.break_degeneracy,
               )
        #####B_b = W_n(H-E_n)z_n
        #diagonal_elements = self.eigenvals * self.weight
        #B_b[:,:] = S[:,:] * self.weight[:, np.newaxis]
        #B_b[:,:] -=  z_vector * diagonal_elements[:, np.newaxis]
        
        B_b[:,:] = copy.deepcopy(S) 
        B_b[:,:] -=  z_vector * self.eigenvals[:, np.newaxis]


    def build_state_average_rdms_z(self, z_vector, eigenvecs):
        self.D_tu_avg_b = np.zeros((self.n_act_orb * self.n_act_orb))
        self.Dpe_tu_avg_b = np.zeros((self.n_act_orb * self.n_act_orb))
        self.D_tuvw_avg_b = np.zeros(
            (self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb)
        )
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            c_build_active_rdm_z(
                z_vector,    
                eigenvecs,
                self.D_tu_avg_b,
                self.D_tuvw_avg_b,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                self.weight[i],
            )
            c_build_active_photon_electron_one_rdm_z(
                z_vector,    
                eigenvecs,
                self.Dpe_tu_avg_b,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                self.weight[i],
            )

        ###symmetrize rdm
        for t in range(self.n_act_orb):
            for u in range(t, self.n_act_orb):
                tu = t * self.n_act_orb + u
                ut = u * self.n_act_orb + t
                dum = self.D_tu_avg_b[tu] + self.D_tu_avg_b[ut]
                self.D_tu_avg_b[tu] = dum/2.0
                self.D_tu_avg_b[ut] = dum/2.0
                dum = self.Dpe_tu_avg_b[tu] + self.Dpe_tu_avg_b[ut]
                self.Dpe_tu_avg_b[tu] = dum/2.0
                self.Dpe_tu_avg_b[ut] = dum/2.0
                for vw in range(self.n_act_orb * self.n_act_orb):
                    dum = (
                        self.D_tuvw_avg_b[tu * self.n_act_orb * self.n_act_orb + vw]
                        + self.D_tuvw_avg_b[ut * self.n_act_orb * self.n_act_orb + vw]
                    )
                    # dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    self.D_tuvw_avg_b[tu * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )
                    self.D_tuvw_avg_b[ut * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )
    def unpack_Z_vector(self, Z, Z_vector):
        index_map = self.index_map
        index_map_size = self.index_map_size
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            #print(r, k, flush = True)
            Z_vector[r][k] = Z[j]
            Z_vector[k][r] = -Z[j]
    def build_residual(self, z_vector, Z_vector, A, G1, Y):
        self.build_state_average_rdms_z(z_vector, self.eigenvecs)
        B_t = np.zeros_like(self.eigenvecs)
        B_b = np.zeros_like(self.eigenvecs)
        self.build_B_t(Z_vector, self.eigenvecs, B_t, A)
        self.build_B_b(z_vector, B_b)
        A_b = np.zeros((self.nmo, self.nmo))
        self.build_A_b(z_vector, self.eigenvecs, A_b)

        A_t = np.zeros((self.nmo, self.nmo))
        self.build_A_t(Z_vector, A, G1, A_t)
        Z_residual_total = A_t + A_b + Y 
        Z_residual_asym = Z_residual_total - Z_residual_total.T
        self.r_Z= np.zeros(self.index_map_size)
        index_map = self.index_map
        index_map_size = self.index_map_size
        
        self.r_z = np.zeros_like(self.eigenvecs)
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            self.r_Z[j] = Z_residual_asym[r][k]
        self.r_z[:,:] = B_t + B_b
        r = self.pack_solution(self.r_Z,self.r_z)
        return r
    def matvec_product(self, x, A, G1):
        z_vector, Z_vector = self.unpack_solution(x)
        #print("Z_Vector", Z_vector, "z_vector", z_vector, flush = True)
        z_vector=self.project_out_all(z_vector, self.eigenvecs) 
        self.build_state_average_rdms_z(z_vector, self.eigenvecs)
        B_t = np.zeros_like(self.eigenvecs)
        B_b = np.zeros_like(self.eigenvecs)
        self.build_B_t(Z_vector, self.eigenvecs, B_t, A)
        self.build_B_b(z_vector, B_b)
        A_b = np.zeros((self.nmo, self.nmo))
        self.build_A_b(z_vector, self.eigenvecs, A_b)
        #print("A_b", A_b)
        #print("data", flush = True)
        A_t = np.zeros((self.nmo, self.nmo))
        Z_residual_asym = np.zeros((self.nmo, self.nmo))
        self.build_A_t(Z_vector, A, G1, A_t)
        Z_residual_total = A_t + A_b  
        #Z_residual_total = copy.deepcopy(2.0 * A_b)  
        Z_residual_asym[:,:] = Z_residual_total - Z_residual_total.T
        self.H_Z= np.zeros(self.index_map_size)
        index_map = self.index_map
        index_map_size = self.index_map_size
        

        self.H_z = np.zeros_like(self.eigenvecs)
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            self.H_Z[j] = 0.5 * Z_residual_asym[r][k]
        self.H_z[:,:] = B_t + B_b
        #self.H_z[:,:] = B_t
        r = self.pack_solution(self.H_Z,self.H_z)
        #print("data2", flush = True)
        return r



    def build_H0_op(self, x, A, G1):
        Z_vector = copy.deepcopy(x[:self.nmo * self.nmo]) 
        z_vector = copy.deepcopy(x[self.nmo * self.nmo:])
        z_vector = z_vector.reshape(self.davidson_roots, self.H_dim)
        Z_vector = Z_vector.reshape(self.nmo, self.nmo)
        #print("Z_Vector", Z_vector, "z_vector", z_vector, flush = True)
        self.build_state_average_rdms_z(z_vector, self.eigenvecs)
        B_t = np.zeros_like(self.eigenvecs)
        B_b = np.zeros_like(self.eigenvecs)
        self.build_B_t(Z_vector, self.eigenvecs, B_t, A)
        self.build_B_b(z_vector, B_b)
        A_b = np.zeros((self.nmo, self.nmo))
        self.build_A_b(z_vector, self.eigenvecs, A_b)
        #print("A_b", A_b)
        #print("data", flush = True)
        A_t = np.zeros((self.nmo, self.nmo))
        self.build_A_t(Z_vector, A, G1, A_t)
        self.H_Z= np.zeros(self.nmo * self.nmo)
        self.H_Z = A_t + A_b 

        self.H_z = np.zeros_like(self.eigenvecs)
        self.H_z[:,:] = B_t + B_b
        r = np.zeros(self.nmo * self.nmo + self.davidson_roots * self.H_dim)
        r[:self.nmo * self.nmo] = self.H_Z.flatten()[:] 
        r[self.nmo * self.nmo:] = self.H_z.flatten()[:] 
        #print("data2", flush = True)
        return r



    def build_total_gradient(self, Y):
        
        Y_asym = 0.5 * (Y-Y.T)
        print("gradient norm of state:", np.linalg.norm(Y_asym.flatten()))
        #Y_zero = np.zeros_like(Y_asym)
        Y_Z= np.zeros(self.index_map_size)
        index_map = self.index_map
        index_map_size = self.index_map_size
        #print("Y_asym", Y_asym)
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            Y_Z[j] = Y_asym[r][k]
            #Y_zero[r][k] = Y_asym[r][k]
            #Y_zero[k][r] = -Y_asym[r][k]
        #print("Y_zero", Y_zero)
        Y_z = np.zeros_like(self.eigenvecs)
        y = self.pack_solution(Y_Z,Y_z)
        return y
    def solve2(self, A, G1, matvec_product, denom, max_iter, conv_thresh=1e-7):
        """
        Main driver loop to solve the linear system Ax + b = 0.
        """
        #self._reset()
        residual = self.reduced_state_gradient.copy()
        if np.linalg.norm(residual < 1e-3):
            #random guess
            dim00 = self.index_map_size + self.davidson_roots * self.H_dim
            trial_0 = np.random.rand(dim00)

            temp2 = trial_0[self.index_map_size:].reshape(self.davidson_roots, self.H_dim)
            temp2=self.project_out_all(temp2, self.eigenvecs)
            trial_0[self.index_map_size:] = temp2.flatten()[:]
            norm = np.linalg.norm(trial_0)
            if norm > 1e-12:
                trial_0 /= norm

            sigma = matvec_product(trial_0, A, G1)
            temp3 = sigma[self.index_map_size:].reshape(self.davidson_roots, self.H_dim)
            temp3=self.project_out_all(temp3, self.eigenvecs)
            sigma[self.index_map_size:] = temp3.flatten()[:]
            residual = sigma + self.reduced_state_gradient 

        #print("initial guess", residual)
        #residual = self.apply_preconditioner(A, G1, z)
        #residual /= np.linalg.norm(residual)
        solver_sym = LinearRMSolver(b_vector=self.reduced_state_gradient, max_subspace=max_iter)
        #solution_sym = solver_sym.solve(matvec_prod_sym, precond_sym, max_iter=100, conv_thresh=1e-7)
        #print(denom[:self.index_map_size])
        print("--------------------------------------------")
        print("--- Start solving CP-SA-CASSCF equations ---")
        print("--------------------------------------------")
        for i in range(max_iter):
            #sigma0 = matvec_product(residual, A, G1)
            #residual0 = sigma0 + self.total_gradient
            #print("zzz", np.linalg.norm(residual0))
            residual_norm = np.linalg.norm(residual)
            print(f"Iter: {i+1:3d}   Residual Norm: {residual_norm:.4e}")

            if residual_norm < conv_thresh:
                print("\n--- Convergence Achieved ---")
                return solver_sym.get_solution()

            trial_c = residual/denom
            #trial_c = np.zeros_like(residual)
            #Q = np.zeros(self.index_map_size)
            #H1_op = LinearOperator(
            #            (self.index_map_size, self.index_map_size),
            #            matvec=lambda Q: self.mv3(A, G1, Q),
            #        )
            #residual_slice1 = copy.deepcopy(residual[:self.index_map_size])
            #residual_slice2 = copy.deepcopy(residual[self.index_map_size:])
            #delta_Z, exitCode = minres(H1_op,residual_slice1, rtol=1e-6)
            #print("exit code", exitCode)
            #trial_c[:self.index_map_size] = copy.deepcopy(delta_Z)
            #trial_c[self.index_map_size:] = residual_slice2/denom[self.index_map_size:] 




            #z_vector, Z_vector = self.unpack_solution(trial_c)
            #for i in range(self.davidson_roots):
            #    print("check dot product", np.dot(z_vector[i], self.eigenvecs[i]))
            #print(trial_c[:self.index_map_size])
            #self.print_matrix_nice(trial_c[self.index_map_size:].reshape(self.davidson_roots,self.H_dim)[0].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
            #self.print_matrix_nice(trial_c[self.index_map_size:].reshape(self.davidson_roots,self.H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)


            temp2 = trial_c[self.index_map_size:].reshape(self.davidson_roots, self.H_dim)
            
            temp2=self.project_out_all(temp2, self.eigenvecs)
            trial_c[self.index_map_size:] = temp2.flatten()[:]
            #self.projection(trial_c[self.index_map_size:])
            #z_vector, Z_vector = self.unpack_solution(trial_c)
            #for i in range(self.davidson_roots):
            #    print("check dot product after", np.dot(z_vector[i], self.eigenvecs[i]))


            #print("trial_c", trial_c)
            norm = np.linalg.norm(trial_c)
            if norm > 1e-12:
                trial_c /= norm

            sigma = matvec_product(trial_c, A, G1)
            #print("sigma before projection")
            #print(sigma[:self.index_map_size])
            #self.print_matrix_nice(sigma[self.index_map_size:].reshape(2,self.H_dim)[0].reshape(4,4), precision=10, width=14, cols_per_line=6)
            #self.print_matrix_nice(sigma[self.index_map_size:].reshape(2,self.H_dim)[1].reshape(4,4), precision=10, width=14, cols_per_line=6)


            #z_vector, Z_vector = self.unpack_solution(sigma)
            #for i in range(self.davidson_roots):
            #    print("check dot product0", np.dot(z_vector[i], self.eigenvecs[i]))

            temp3 = sigma[self.index_map_size:].reshape(self.davidson_roots, self.H_dim)
            temp3=self.project_out_all(temp3, self.eigenvecs)
            sigma[self.index_map_size:] = temp3.flatten()[:]

            #self.projection(sigma[self.index_map_size:])
            #z_vector, Z_vector = self.unpack_solution(trial_c)
            #for i in range(self.davidson_roots):
            #    print("check dot product after0", np.dot(z_vector[i], self.eigenvecs[i]))


            #print("sigma", sigma)
            residual_old =copy.deepcopy(residual)
            residual = solver_sym.update_subspace_and_extrapolate(trial_c, sigma)
            residual_new = copy.deepcopy(residual)
            error = residual_new - residual_old
            if i > 0 and np.linalg.norm(error) < 1e-8:
                print("\n--- Convergence Achieved (solution becomes self-consistent)---")
                return solver_sym.get_solution()


        print("\n--- Solver did not converge within max iterations ---")
        #return solver_sym.get_solution()



    def pack_solution(self, x_Z, x_z):
        r = np.zeros(self.index_map_size + self.davidson_roots * self.H_dim)
        r[:self.index_map_size] = copy.deepcopy(x_Z)
        r[self.index_map_size:] = copy.deepcopy(x_z.flatten())
        return r
    def unpack_solution(self, x):
        Z = copy.deepcopy(x[:self.index_map_size])
        z_vector = copy.deepcopy(x[self.index_map_size:]) 
        index_map = self.index_map
        index_map_size = self.index_map_size
        
        Z_vector = np.zeros((self.nmo, self.nmo))
        self.unpack_Z_vector(Z, Z_vector)
        z_vector = z_vector.reshape(self.davidson_roots, self.H_dim).astype(np.float64)
        return z_vector, Z_vector


    def mv3(self, A, G1, Z):
        Z_vector = np.zeros((self.nmo, self.nmo))
        self.unpack_Z_vector(Z, Z_vector)
        A_t = np.zeros((self.nmo, self.nmo))
        self.build_A_t(Z_vector, A, G1, A_t)
 
        A_temp = 0.5 *(A_t - A_t.transpose())
        A_t_asym= np.zeros(self.index_map_size)
        index_map = self.index_map
        index_map_size = self.index_map_size
        
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            A_t_asym[j] = A_temp[r][k]
            
        return A_t_asym

    #def projection(self, z):
    #    d, n = self.davidson_roots, self.H_dim          # for brevity
    #    for i in range(d):
    #        # slice that corresponds to the i-th row
    #        start, stop = i * n, (i + 1) * n
    #        row_z = z[start:stop]
    #    
    #        # projection coefficient
    #        coeff = np.dot(row_z, self.eigenvecs[i])
    #    
    #        # subtract the projection from that slice
    #        z[start:stop] -= coeff * self.eigenvecs[i]



    def build_A_t(self, Z_vector, A, G, A_t):
        ##symmetrize A
        #A_temp = 0.5 * (A + A.T)
        A_t[:,:] = 0 
        A_t[:,:self.n_occupied] = 2.0 * np.dot(Z_vector.T, A[:,:self.n_occupied])
        #A_t[:,:] = 2.0 * np.dot(Z_vector.T, A_temp)
        A_t[:,:self.n_occupied] += 2.0 * np.dot(G, Z_vector[:,:self.n_occupied].flatten()).reshape((self.nmo,self.n_occupied))
    
    def calculate_off_diagonal_photon_constant_z(self, z_vector, eigenvecs, state):
        off_diagonal_constant = 0.0
        np1 = self.N_p + 1
        weight = np.zeros_like(self.weight)
        if state < 0: weight[:] = copy.deepcopy(self.weight)
        else: 
            weight[state] = 1
        #print("weight", weight)
        for i in range(self.davidson_roots):
            if weight[i] == 0: continue
            #print(i)
            # print("weight", self.weight[i])
            eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
            eigenvecs2 = eigenvecs2.transpose(1, 0)
            z_vector2 = z_vector[i].reshape((np1, self.num_det))
            z_vector2 = z_vector2.transpose(1, 0)
            for m in range(np1):
                if self.N_p == 0:
                    continue
                if m > 0 and m < self.N_p:
                    off_diagonal_constant += (
                        weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * np.dot(
                            z_vector2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                    off_diagonal_constant += (
                        weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * np.dot(
                            z_vector2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )
                elif m == self.N_p:
                    off_diagonal_constant += (
                        weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * np.dot(
                            z_vector2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                else:
                    off_diagonal_constant += (
                        weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * np.dot(
                            z_vector2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )

        return off_diagonal_constant

   
    def build_A_b(self, z_vector, eigenvecs, A_b):
        A_b[:,:] = 0
        rot_dim = self.nmo
        constant_Dij = 0 
        # this constant should be 0 because z^n.c^n =0
        for i in range(self.davidson_roots):
            constant_Dij += self.weight[i] * np.dot(z_vector[i], eigenvecs[i])
        constant_Dij = 0 
        #print("constant_Dij",constant_Dij)
        D_tu_avg_b = self.D_tu_avg_b.reshape((self.n_act_orb, self.n_act_orb))
        Dpe_tu_avg_b = self.Dpe_tu_avg_b.reshape((self.n_act_orb, self.n_act_orb))
        D_tuvw_avg_b = self.D_tuvw_avg_b.reshape(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        #print("D_tu_avg_b",D_tu_avg_b, flush = True)

        
        fock_general = np.zeros((rot_dim, rot_dim))
        
        temp1 = (
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
            - 0.5
            * self.K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
        )
        # start = timer()
        fock_general += self.fock_core[:rot_dim, :rot_dim] * constant_Dij + np.einsum(
            "tu,turs->rs", D_tu_avg_b, temp1, optimize="optimal"
        )
        
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant_z(z_vector, eigenvecs,-1)
        A_b[:, : self.n_in_a] = 4.0 * (
            fock_general[:, : self.n_in_a]
            - self.d_cmo[:rot_dim, : self.n_in_a] * off_diagonal_constant
        )


        A_b[:, self.n_in_a : self.n_occupied] = 2.0 * np.einsum(
            "rt,tu->ru",
            self.fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            D_tu_avg_b,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 4", end - start)
        # print(np.shape(self.active_twoeint))
        # start = timer()
        A_b[:, self.n_in_a : self.n_occupied] += 2.0 * np.einsum(
            "vwrt,tuvw->ru",
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                self.n_in_a : self.n_occupied,
            ],
            D_tuvw_avg_b,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 5", end - start)
        # start = timer()
        A_b[:, self.n_in_a : self.n_occupied] += -2.0 * np.sqrt(self.omega / 2) * np.einsum(
            "rt,tu->ru",
            self.d_cmo[:rot_dim, self.n_in_a : self.n_occupied],
            Dpe_tu_avg_b,
            optimize="optimal",
        )
       

    def build_Y(self, state, eigenvecs, Y):
        Y[:,:] = 0.0
        self.D_tu_n = np.zeros((self.n_act_orb * self.n_act_orb))
        self.Dpe_tu_n = np.zeros((self.n_act_orb * self.n_act_orb))
        self.D_tuvw_n = np.zeros(
            (self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb)
        )

        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            if state != i: continue
            c_build_active_rdm(
                eigenvecs,
                self.D_tu_n,
                self.D_tuvw_n,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                1.0,
            )
            c_build_active_photon_electron_one_rdm(
                eigenvecs,
                self.Dpe_tu_n,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                1.0,
            )

        ###symmetrize rdm
        for t in range(self.n_act_orb):
            for u in range(t, self.n_act_orb):
                tu = t * self.n_act_orb + u
                ut = u * self.n_act_orb + t
                dum = self.D_tu_n[tu] + self.D_tu_n[ut]
                self.D_tu_n[tu] = dum/2.0
                self.D_tu_n[ut] = dum/2.0
                dum = self.Dpe_tu_n[tu] + self.Dpe_tu_n[ut]
                self.Dpe_tu_n[tu] = dum/2.0
                self.Dpe_tu_n[ut] = dum/2.0
                for vw in range(self.n_act_orb * self.n_act_orb):
                    dum = (
                        self.D_tuvw_n[tu * self.n_act_orb * self.n_act_orb + vw]
                        + self.D_tuvw_n[ut * self.n_act_orb * self.n_act_orb + vw]
                    )
                    # dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    self.D_tuvw_n[tu * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )
                    self.D_tuvw_n[ut * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )

        rot_dim = self.nmo
        D_tu_n = self.D_tu_n.reshape((self.n_act_orb, self.n_act_orb))
        Dpe_tu_n = self.Dpe_tu_n.reshape((self.n_act_orb, self.n_act_orb))
        D_tuvw_n = self.D_tuvw_n.reshape(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )


        
        fock_general = np.zeros((rot_dim, rot_dim))
        
        temp1 = (
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
            - 0.5
            * self.K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
        )
        # start = timer()
        fock_general += self.fock_core[:rot_dim, :rot_dim] + np.einsum(
            "tu,turs->rs", D_tu_n, temp1, optimize="optimal"
        )
        
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant_z(eigenvecs, eigenvecs, state)
        Y[:, : self.n_in_a] = 4.0 * (
            fock_general[:, : self.n_in_a]
            - self.d_cmo[:rot_dim, : self.n_in_a] * off_diagonal_constant
        )


        Y[:, self.n_in_a : self.n_occupied] = 2.0 * np.einsum(
            "rt,tu->ru",
            self.fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            D_tu_n,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 4", end - start)
        # print(np.shape(self.active_twoeint))
        # start = timer()
        Y[:, self.n_in_a : self.n_occupied] += 2.0 * np.einsum(
            "vwrt,tuvw->ru",
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                self.n_in_a : self.n_occupied,
            ],
            D_tuvw_n,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 5", end - start)
        # start = timer()
        Y[:, self.n_in_a : self.n_occupied] += -2.0 * np.sqrt(self.omega / 2) * np.einsum(
            "rt,tu->ru",
            self.d_cmo[:rot_dim, self.n_in_a : self.n_occupied],
            Dpe_tu_n,
            optimize="optimal",
        )
    
    def build_Y_hf(self, state, Z_vector, z_vector, eigenvecs, Y_hf):
        Y_hf[:,:] = 0.0
        off_diagonal_constant_state = self.calculate_off_diagonal_photon_constant_z(eigenvecs, eigenvecs, state)
        off_diagonal_constant_z = self.calculate_off_diagonal_photon_constant_z(z_vector, eigenvecs, -1)
        #print(self.d_exp)
        #print(self.E_d_state)
        #print(self.E_d_Z)
        #print(self.E_d_z)
        #print(off_diagonal_constant_state)
        #print(off_diagonal_constant_z)
        scale_factor = -self.E_d_state + off_diagonal_constant_state + self.d_exp - self.E_d_Z - self.E_d_z + off_diagonal_constant_z
        Y_hf[:,:self.ndocc] = 4.0 * scale_factor * self.d_hf[:,:self.ndocc]
    
    def build_sigma_hf(self, energy_diff, kappa, temp_aibj):
        kappa_matrix = kappa.reshape(self.n_v_hf, self.ndocc)
        sigma_matrix = energy_diff * kappa_matrix

        sigma_matrix += np.einsum("aibj, bj->ai", temp_aibj, kappa_matrix)
        return sigma_matrix.flatten()
    def build_A_tilde_hf(self, fock_hf, kappa, A_tilde_hf):
        z = np.zeros((self.nmo, self.nmo))
        z[self.ndocc:, :self.ndocc] = kappa.reshape(self.n_v_hf, self.ndocc)[:,:]
        z_bar = z + z.T
        A_tilde_hf[:,:] = np.dot(fock_hf, z_bar)
        g_bar = np.einsum("mn, rqmn-> rq", z_bar, self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo))
        g_bar += -np.einsum("mn, rqmn-> rq", z_bar, self.d_spatial.reshape(self.nmo, self.nmo, self.nmo, self.nmo))
        g_bar += -0.5 * np.einsum("mn, rmqn-> rq", z_bar, self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo))
        diagonal_values = np.concatenate([np.full(self.ndocc, 2), np.zeros(self.n_v_hf)])
        d_zero = np.diag(diagonal_values)
        A_tilde_hf[:,:] += np.dot(g_bar, d_zero)

    def purify(self, sigma, dim0):
        temp = sigma[:dim0*dim0]
        temp = temp.reshape(dim0, dim0)
        for i in range(self.n_in_a):
            for j in range(self.n_in_a):
                temp[i,j] = 0
        for t in range(self.n_act_orb):
            for u in range(self.n_act_orb):
                temp[self.n_in_a + t, self.n_in_a + u] = 0
        for a in range(self.n_virtual):
            for b in range(self.n_virtual):
                temp[self.n_occupied + a, self.n_occupied + b] = 0
        sigma[:dim0*dim0] = temp.flatten()        
     
    def antisymmetrize(self, sigma, dim0):
        temp = sigma[:dim0*dim0]
        temp = temp.reshape(dim0, dim0)
        temp0 = 0.5 * (temp-temp.T)
        
        sigma[:dim0*dim0] = temp0.flatten()        


    def print_matrix_nice(self, matrix, precision=6, width=12, cols_per_line=6, threshold=1e-10):
        """
        Print matrix with custom formatting
        """
        rows, cols = matrix.shape
        
        for start_col in range(0, cols, cols_per_line):
            end_col = min(start_col + cols_per_line, cols)
            
            print(f"\nColumns {start_col} to {end_col-1}:")
            print("-" * (width * (end_col - start_col)))
            
            for i in range(rows):
                row_str = ""
                for j in range(start_col, end_col):
                    value = matrix[i, j]
                    if abs(value) < threshold:  # Treat very small numbers as zero
                        value = 0.0
                    row_str += f"{value:{width}.{precision}f}"
                print(f"Row {i}: {row_str}") 


        #"""
        #Print matrix with custom formatting
        #
        #Parameters:
        #- matrix: numpy array
        #- precision: number of decimal places
        #- width: width of each number field
        #- cols_per_line: maximum columns per line
        #"""
        #rows, cols = matrix.shape
        #
        #for start_col in range(0, cols, cols_per_line):
        #    end_col = min(start_col + cols_per_line, cols)
        #    
        #    print(f"\nColumns {start_col} to {end_col-1}:")
        #    print("-" * (width * (end_col - start_col)))
        #    
        #    for i in range(rows):
        #        row_str = ""
        #        for j in range(start_col, end_col):
        #            if abs(matrix[i, j]) < 1e-10:  # Treat very small numbers as zero
        #                row_str += f"{'0.0000':{width}.{precision}f}"
        #            else:
        #                row_str += f"{matrix[i, j]:{width}.{precision}f}"
        #        print(f"Row {i}: {row_str}")

    def project_out_all(self, sigma_vectors, ci_eigenvectors):
        """
        Projects each vector in sigma_vectors to be orthogonal to every vector
        in ci_eigenvectors.

        This function implements the 'project_out_all' logic, which is equivalent
        to a Gram-Schmidt orthogonalization of each sigma vector against the full
        set of CI basis vectors.

        Args:
            sigma_vectors (np.ndarray): A 2D NumPy array of shape (n_states, H_dim)
                                         where each row is a sigma vector.
            ci_eigenvectors (np.ndarray): A 2D NumPy array of shape (n_states, H_dim)
                                          where each row is a CI eigenvector. It is
                                          assumed that these vectors form an
                                          orthonormal basis.

        Returns:
            np.ndarray: A new 2D array of the same shape as sigma_vectors, where
                        each row vector has been made orthogonal to all rows in
                        ci_eigenvectors.
        """
        # Verify that the dimensions are compatible
        if sigma_vectors.shape[1] != ci_eigenvectors.shape[1]:
            raise ValueError("The dimensionality (H_dim) of sigma and CI vectors must match.")

        # Create a copy of the sigma vectors to store the projected results
        projected_sigma = np.copy(sigma_vectors)

        # Loop over each sigma vector that needs to be projected
        for i in range(projected_sigma.shape[0]):
            # Start with the original sigma vector for this state
            vec_to_project = projected_sigma[i, :]

            # Sequentially subtract the projection onto each CI eigenvector
            for j in range(ci_eigenvectors.shape[0]):
                ci_vec = ci_eigenvectors[j, :]
                
                # Calculate the dot product (the projection coefficient)
                dot_product = np.dot(vec_to_project, ci_vec)
                
                # Subtract the component parallel to the CI eigenvector
                vec_to_project = vec_to_project - dot_product * ci_vec
            
            # Store the final, fully projected vector
            projected_sigma[i, :] = vec_to_project

        return projected_sigma 



    def compute_grad(self, state):
        print("\n\n")
        print("--------------------------------------------")
        print("Begin computing the analytical gradient for state",state)
        rot_dim = self.nmo
        self.H_dim = self.eigenvecs.shape[1]
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        self.build_state_average_rdms(self.eigenvecs)



        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        self.build_intermediates(self.eigenvecs, A, G, True)
        G1 = G.transpose(3, 1, 2, 0).reshape(
                self.nmo * self.n_occupied, self.nmo * self.n_occupied
            )

       
        self.fock_core = copy.deepcopy(self.H_spatial2)
        self.fock_core += 2.0 * np.einsum(
            "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        self.fock_core -= np.einsum(
            "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        
        self.E_core = 0.0
        self.E_core += np.einsum("jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a])
        self.E_core += np.einsum("jj->", self.fock_core[: self.n_in_a, : self.n_in_a])
        # end   = timer()
        # print("build intermediate step 1", end - start)

        self.active_fock_core = copy.deepcopy(
            self.fock_core[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied]
        )
        self.active_twoeint = copy.deepcopy(
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ]
        )





        self.occupied_J = np.zeros(
            (
                self.n_occupied,
                self.n_occupied,
                self.n_occupied,
                self.n_occupied,
            )
        )
        self.occupied_J[
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
        ] = copy.deepcopy(
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ]
        )
        
        self.H_diag3 = np.zeros(H_dim)
        

        
        self.occupied_fock_core = np.zeros(
            (self.n_occupied, self.n_occupied)
        )
        self.occupied_fock_core[
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
        ] = copy.deepcopy(self.active_fock_core)
        self.occupied_d_cmo = np.zeros(
            (self.n_occupied, self.n_occupied)
        )
        self.occupied_d_cmo = copy.deepcopy(
            self.d_cmo[: self.n_occupied, : self.n_occupied]
        )
        self.gkl2 = copy.deepcopy(self.active_fock_core)
        self.gkl2 -= 0.5 * np.einsum("kjjl->kl", self.active_twoeint)

        self.occupied_J = self.occupied_J.reshape(
            self.n_occupied * self.n_occupied,
            self.n_occupied * self.n_occupied,
        )
        
        #build exact diagonal elements of the Hamiltonian
        c_H_diag_cas_spin(
            self.occupied_fock_core,
            self.occupied_J,
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
            self.target_spin,
        )
        self.d_diag = 2.0 * np.einsum(
            "ii->", self.d_cmo[: self.n_in_a, : self.n_in_a]
        )
        Y = np.zeros((rot_dim, rot_dim))
        #self.build_Y(0, self.eigenvecs, Y)
        #self.b = np.zeros(self.index_map_size + self.davidson_roots * self.H_dim)
        #Y_asym = Y - Y.T
        index_map = self.index_map
        index_map_size = self.index_map_size
        #
        #self.r_z = np.zeros_like(self.eigenvecs)
        #for j in range(index_map_size):
        #    r = index_map[j][0]
        #    k = index_map[j][1]
        #    self.b[j] = Y_asym[r][k]

        #self.reduced_hessian_diagonal[:] = 0.0 
                

        self.reduced_hessian_diagonal[:] = 0.0 
        U = np.eye(self.nmo)
        self.build_hessian_diagonal(U, G, A)
        


        self.build_Y(state, self.eigenvecs, Y)
        state_gradient = copy.deepcopy(Y)
        self.reduced_state_gradient = self.build_total_gradient(Y)
        denom0 = np.zeros(self.index_map_size + self.davidson_roots * self.H_dim)
        #denom0[:self.index_map_size] = 2.0 * self.reduced_hessian_diagonal[:]
        denom0[:self.index_map_size] = self.reduced_hessian_diagonal[:]
        for i in range(self.davidson_roots):
            for j in range(self.H_dim):
                denom0[self.index_map_size + i * self.H_dim + j] = self.H_diag3[j] - self.eigenvals[i]
        solution = self.solve2(A, G1, self.matvec_product, denom0, max_iter=2000, conv_thresh=1e-7)





        ####print(self.reduced_hessian_diagonal)
        ###total_diagonal = np.full((self.nmo, self.nmo), 1e20)
        ###for j in range(index_map_size):
        ###    r = index_map[j][0]
        ###    k = index_map[j][1]
        ###    total_diagonal[r][k] = self.reduced_hessian_diagonal[j] 
        ###    total_diagonal[k][r] = self.reduced_hessian_diagonal[j]

        ####self.print_matrix_nice(total_diagonal, precision=10, width=14, cols_per_line=6)
        ###self.build_Y(state, self.eigenvecs, Y)
        ###state_gradient = copy.deepcopy(Y)
        ####print("raw gradient")
        ####self.print_matrix_nice(Y, precision=10, width=14, cols_per_line=6)
        ###Y = Y.flatten() 
        ###self.antisymmetrize(Y, self.nmo)
        ###self.purify(Y, self.nmo)
        ###print("gradient norm of state", state, ":", np.linalg.norm(Y))
        ####Y = Y.reshape(self.nmo, self.nmo)
        ####self.print_matrix_nice(Y.reshape(self.nmo,self.nmo), precision=10, width=14, cols_per_line=6)
        ###
        ###residual = np.zeros(self.nmo * self.nmo + self.davidson_roots * self.H_dim)
        ###denom = np.zeros(self.nmo * self.nmo + self.davidson_roots * self.H_dim)
        ###denom[:self.nmo * self.nmo] = total_diagonal.flatten()[:] 
        ###for i in range(self.davidson_roots):
        ###    for j in range(self.H_dim):
        ###        denom[self.nmo * self.nmo + i * self.H_dim + j] = self.H_diag3[j] - self.eigenvals[i]




        ###residual[:self.nmo * self.nmo] = Y[:]
        ###bb = copy.deepcopy(residual)
        ###b_vector = bb 

        ###b_norm = np.linalg.norm(Y)
        ###print(f"Initial gradient norm ||b||: {b_norm:.4e}")
        ###
        ####level_shift = 0 
        ####level_shift = 1e-3
        ####denom = denom + level_shift
        #####denom += 1e-3
        ####denom[abs(denom) < 1e-8] = 1e-8



 
        ####residual[:self.nmo * self.nmo].fill(1)
        ###solver_sym1 = LinearRMSolver(b_vector=residual, max_subspace=1000)
        ###max_iter = 1000
        ###conv_thresh = 1e-7
        ###print("--------------------------------------------")
        ###print("--- Start solving CP-SA-CASSCF equations ---")
        ###print("--------------------------------------------")
        ###for i in range(max_iter):
        ###    residual_orb = residual[:self.nmo * self.nmo]
        ###    residual_ci = residual[self.nmo * self.nmo:]

        ###    rms_value1 = np.sqrt(np.mean(residual_orb**2))
        ###    rms_value2 = np.sqrt(np.mean(residual_ci **2))
        ###    print("rms", rms_value1, rms_value2)

        ###    residual_norm = np.linalg.norm(residual)
        ###    print(f"Iter: {i+1:3d}   Residual Norm: {residual_norm:.4e}")

        ###    if residual_norm < conv_thresh and i > 0:
        ###        print("\n--- Convergence Achieved ---")
        ###        self.c_vector = solver_sym1.get_solution()
        ###        break
        ###    trial_c = np.zeros(self.nmo * self.nmo + self.davidson_roots * self.H_dim)
        ###    #print("denom")
        ###    
        ###    trial_c[:] = residual/denom
        ###    #print("trial c after precondition")
        ###    #self.print_matrix_nice(trial_c[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(trial_c[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(trial_c[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)


        ###    temp = trial_c[self.nmo * self.nmo:].reshape((self.davidson_roots, H_dim))
        ###    #for j in range(2):
        ###    #    temp[j] -= np.dot(temp[j], ci_vecs[j]) * ci_vecs[j]
        ###    temp=self.project_out_all(temp, self.eigenvecs) 
        ###    trial_c[self.nmo * self.nmo:] = temp.flatten()    
        ###    norm = np.linalg.norm(trial_c)
        ###    if norm > 1e-12:
        ###        trial_c /= norm
        ###    print("trial c after normalization")
        ###    sigma = self.build_H0_op(trial_c, A, G1)
        ###    level_shift = 1e-3
        ###    level_shift = 0
        ###    sigma += level_shift * trial_c
        ###    self.antisymmetrize(sigma, self.nmo)
        ###    self.purify(sigma, self.nmo) 
        ###    temp3 = sigma[self.nmo * self.nmo:].reshape((self.davidson_roots, H_dim))
        ###    #print("sigma before projection")
        ###    #self.print_matrix_nice(sigma[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(
        ###    #    self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        ###    #temp2 = sigma[self.nmo * self.nmo:].reshape((2, H_dim))
        ###    #for j in range(2):
        ###    #    temp2[j] -= np.dot(temp2[j], ci_vecs[j]) * ci_vecs[j]
        ###    #print(np.dot(temp2[0], ci_vecs[0]))
        ###    temp2 = sigma[self.nmo * self.nmo:].reshape((self.davidson_roots, H_dim))
        ###    #for j in range(2):
        ###    #    dot_prod = np.dot(temp2[j], ci_vecs[j])
        ###    #    projection = dot_prod * ci_vecs[j]
        ###    #    print(f"State {j}: dot_product = {dot_prod}")
        ###    #    print(f"State {j}: projection norm = {np.linalg.norm(projection)}")
        ###    #    temp2[j] -= projection
        ###    #    # Check dot product after projection (should be ~0)
        ###    #    print(f"State {j}: dot product after projection = {np.dot(temp2[j], ci_vecs[j])}")
        ###    temp2=self.project_out_all(temp2, self.eigenvecs) 
        ###    sigma[self.nmo * self.nmo:] = temp2.flatten()   
        ###    # After projection
        ###   
        ###    #print("sigma after projection")
        ###    #self.print_matrix_nice(sigma[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)

        ###    residual_old =copy.deepcopy(residual)
        ###    residual = solver_sym1.update_subspace_and_extrapolate(trial_c, sigma)
        ###    #print("residual after linear rm")
        ###    #self.print_matrix_nice(residual[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(residual[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        ###    #self.print_matrix_nice(residual[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)

        ###    residual_new = copy.deepcopy(residual)
        ###    error = residual_new - residual_old
        ###    if i > 0 and np.linalg.norm(error) < 1e-10:
        ###        print("\n--- Convergence Achieved (solution becomes self-consistent)---")
        ###        self.c_vector = solver_sym1.get_solution()
        ###        break


        #print("\n--- Solver did not converge within max iterations ---")
        #print("z_vector")
        #self.print_matrix_nice(trial_c[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        #self.print_matrix_nice(trial_c[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)
        #self.print_matrix_nice(trial_c[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(self.num_alpha,self.num_alpha), precision=10, width=14, cols_per_line=6)

        #sigma = self.build_H0_op(self.c_vector, A, G1)
        #self.antisymmetrize(sigma, self.nmo)
        #self.purify(sigma, self.nmo) 
        #temp2 = sigma[self.nmo * self.nmo:].reshape((self.davidson_roots, H_dim))
        #temp2=self.project_out_all(temp2, self.eigenvecs) 
        #sigma[self.nmo * self.nmo:] = temp2.flatten()   
        #print("last sigma")
        #self.print_matrix_nice(sigma[:self.nmo * self.nmo].reshape(self.nmo, self.nmo), precision=10, width=14, cols_per_line=6)
        #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[0].reshape(4,4), precision=10, width=14, cols_per_line=6)
        #self.print_matrix_nice(sigma[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim)[1].reshape(4,4), precision=10, width=14, cols_per_line=6)
        #print (np.linalg.norm(sigma+bb))

        ##Z_final = self.c_vector[:self.nmo * self.nmo].reshape(self.nmo, self.nmo).copy()
        ##z_final = self.c_vector[self.nmo * self.nmo:].reshape(self.davidson_roots,H_dim).copy()
        z_final,Z_final = self.unpack_solution(solution)

        self.build_state_average_rdms_z(z_final, self.eigenvecs)
        #print("Z_final",Z_final)
        #print("z_final",z_final)
        A_t = np.zeros((self.nmo,self.nmo))
        A_b = np.zeros((self.nmo,self.nmo))
        self.build_A_t(Z_final, A, G1, A_t)
        self.build_A_b(z_final, self.eigenvecs, A_b)

        X_temp = A_t + A_b + state_gradient
        X_temp[:,:self.n_occupied] += 4.0*np.dot(Z_final[:,:self.n_occupied], A[:self.n_occupied, :self.n_occupied])
        X = 0.25 *(X_temp + X_temp.T)
        #print(X)
        #state = 0
        self.build_effective_densities_mo(Z_final, z_final, self.eigenvecs, state)
        self.transform_effective_densities(self.opt_C)

        #mints = psi4.core.MintsHelper(wfn.basisset())
        
        # C_np = np.zeros((self.nmo,self.nmo))
        self.Ca_like = psi4.core.Matrix.from_array(self.opt_C)
        self.Ca_like.basis = self.wfn.basisset()
         

        Z_vector = Z_final
        z_vector = z_final


        n_atoms = self.n_atoms
        n_orbitals = self.nmo
        # initialize array for the integral derivatives
        self.overlap_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.potential_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.kinetic_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.eri_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals, n_orbitals, n_orbitals))

        # initialize the gradient arrays
        self.pulay_force = np.zeros(3 * n_atoms)
        self.kinetic_gradient = np.zeros(3 * n_atoms)
        self.potential_gradient = np.zeros(3 * n_atoms)
        self.repulsion_gradient = np.zeros(3 * n_atoms)
        self.total_gradient = np.zeros(3 * n_atoms)

        ## loop over the atoms
        #for i in range(self.n_atoms):
        #    # loop over the cartesian coordinates
        #    for j in range(3):
        #        # define the derivative index
        #        deriv_index = 3 * i + j

        #        # get the one-electron integral derivatives
        #        # overlap is in the MO basis
        #        self.overlap_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.mo_oei_deriv1("OVERLAP", i, self.Ca_like, self.Ca_like )[j])

        #        # all others are in the AO basis
        #        self.potential_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("POTENTIAL", i)[j])
        #        self.kinetic_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("KINETIC", i)[j])

        #        # get the two-electron integral derivatives
        #        self.eri_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_tei_deriv1(i)[j])

        #        # compute the J and K derivetives
        #        self.repulsion_gradient[deriv_index] = np.einsum("mnxy,mnxy->", self.eri_deriv_matrix_ao[deriv_index, :, :, :, :], self.two_rdm_eff_ao)


        #        # now contract each of the derivatives with the density matrix to get the respective gradient components
        #        # Pulay force first
        #        self.pulay_force[deriv_index] =  -  np.einsum("pq,pq->", X, self.overlap_deriv_matrix_ao[deriv_index])

        #        # kinetic gradient
        #        self.kinetic_gradient[deriv_index] = np.einsum("uv,uv->", self.one_rdm_eff_ao, self.kinetic_deriv_matrix_ao[deriv_index, :, :])

        #        # potential gradient
        #        self.potential_gradient[deriv_index] = np.einsum("uv,uv->", self.one_rdm_eff_ao, self.potential_deriv_matrix_ao[deriv_index, :, :])


        #                                                            

        #total_gradient = self.nuclear_energy_gradient + self.pulay_force + self.kinetic_gradient + self.potential_gradient + self.repulsion_gradient 
        ##print("total_gradient")
        ##print(total_gradient)
        #print("pulay force")
        #print(self.pulay_force)
        #print("kinetic+potential")
        #print(self.kinetic_gradient+self.potential_gradient)

        ##print(self.pulay_force)
        #print(self.nuclear_energy_gradient)

        #print(np.linalg.norm(total_gradient))

        #self.print_matrix_nice((self.twoeint-self.twoeint_hf).reshape(self.nmo*self.nmo, self.nmo *self.nmo), precision=10, width=14, cols_per_line=6)
        self.twoeint = np.copy(self.twoeint_hf)        
        self.n_v_hf = self.nmo - self.ndocc

        self.fock_hf = copy.deepcopy(self.H_hf)
        self.fock_hf += 2.0 * np.einsum(
            "rsjj->rs", self.twoeint.reshape(self.nmo,self.nmo,self.nmo,self.nmo)[:, :, :self.ndocc, :self.ndocc], optimize="optimal"
        )
        self.fock_hf -= np.einsum(
            "rjsj->rs", self.twoeint.reshape(self.nmo,self.nmo,self.nmo,self.nmo)[:, :self.ndocc, :, :self.ndocc], optimize="optimal"
        )
        #print("fock HF ")
        #self.print_matrix_nice(self.fock_hf, precision=10, width=14, cols_per_line=6)
        fock_diag = np.diag(self.fock_hf)
        
        eps_occ = fock_diag[:self.ndocc]      # ε_i (occupied)
        eps_virt = fock_diag[self.ndocc:]     # ε_a (virtual)
        energy_diff = eps_virt[:, np.newaxis] - eps_occ[np.newaxis, :]   
        self.d_spatial = np.einsum("ij,kl-> ijkl", self.d_hf, self.d_hf)
        temp_aibj = np.zeros((self.n_v_hf, self.ndocc, self.n_v_hf, self.ndocc))
        for a in range(self.n_v_hf):
            for i in range(self.ndocc):
                for b in range(self.n_v_hf):
                    for j in range(self.ndocc):
                        temp_aibj[a][i][b][j] = 4.0 * self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc+a][i][self.ndocc+b][j]
                        temp_aibj[a][i][b][j] += -self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc+a][self.ndocc+b][i][j]
                        temp_aibj[a][i][b][j] += -self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc+a][j][self.ndocc+b][i]
                        temp_aibj[a][i][b][j] -= 4.0 * self.d_spatial[self.ndocc+a][i][self.ndocc+b][j]
        Y_hf = np.zeros((self.nmo, self.nmo))
        self.build_Y_hf(state, Z_vector, z_vector, self.eigenvecs, Y_hf)
        grad_hf = Y_hf - Y_hf.T
        grad_hf_ai = grad_hf[self.ndocc:, :self.ndocc].flatten()







        kappa = np.zeros(self.n_v_hf * self.ndocc)
        if np.linalg.norm(self.lambda_vector) > 0:
            residual = np.zeros(self.n_v_hf * self.ndocc)
            denom_hf = energy_diff.flatten()[:] 
             
            residual[:] = grad_hf_ai[:]
            bb = copy.deepcopy(residual)
            max_iter = 2000
            conv_thresh = 1e-7

            solver_sym3 = LinearRMSolver(b_vector=residual, max_subspace=2000)
            print("---------------------------------------------")
            print("------- Start solving CP-HF equations -------")
            print("---------------------------------------------")
            for i in range(max_iter):
                residual_norm = np.linalg.norm(residual)
                print(f"Iter: {i+1:3d}   Residual Norm: {residual_norm:.4e}")

                if residual_norm < conv_thresh and i > 0:
                    print("\n--- Convergence Achieved ---")
                    self.cp_hf_solution = solver_sym3.get_solution()
                    break
                trial_c = np.zeros(self.n_v_hf * self.ndocc)
                #print("denom")
                trial_c[:] = residual/denom_hf
                    
                norm = np.linalg.norm(trial_c)
                if norm > 1e-12:
                    trial_c /= norm
                #print("trial c after normalization")
                sigma2 = self.build_sigma_hf(energy_diff, trial_c, temp_aibj)
               
                residual = solver_sym3.update_subspace_and_extrapolate(trial_c, sigma2)
                #print("residual after linear rm")
            kappa[:] = self.cp_hf_solution
        else:
            kappa[:] = 0.0 
        #print("kappa", kappa)        
        Z_vector = Z_final
        z_vector = z_final
        Y_hf = np.zeros((self.nmo, self.nmo))
        A_tilde_hf = np.zeros((self.nmo, self.nmo))
        self.build_Y_hf(state, Z_vector, z_vector, self.eigenvecs, Y_hf)
        self.build_A_tilde_hf(self.fock_hf, kappa, A_tilde_hf)
        #A_tilde_hf2 = np.zeros((self.nmo, self.nmo))
        #A_tilde_hf2[:,:self.ndocc] = np.dot(self.fock_hf[:,self.ndocc:], kappa.reshape(self.n_v_hf, self.ndocc))
        #A_tilde_hf2[:,self.ndocc:] += np.dot(self.fock_hf[:,:self.ndocc], kappa.reshape(self.n_v_hf, self.ndocc).T)
        #A_tilde_hf2[:,:self.ndocc] += 4.0 * np.einsum("aixy,ai->xy", self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc:,:self.ndocc,:,:self.ndocc], 
        #                                              kappa.reshape(self.n_v_hf, self.ndocc))
        #A_tilde_hf2[:,:self.ndocc] += -np.einsum("ayxi,ai->xy", self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc:,:self.ndocc,:,:self.ndocc], 
        #                                              kappa.reshape(self.n_v_hf, self.ndocc))
        #A_tilde_hf2[:,:self.ndocc] += -np.einsum("axyi,ai->xy", self.twoeint.reshape(self.nmo, self.nmo, self.nmo, self.nmo)[self.ndocc:,:,:self.ndocc,:self.ndocc], 
        #                                              kappa.reshape(self.n_v_hf, self.ndocc))
        #A_tilde_hf2[:,:self.ndocc] += -4.0 * np.einsum("aixy,ai->xy", self.d_spatial[self.ndocc:,:self.ndocc,:,:self.ndocc], 
        #                                              kappa.reshape(self.n_v_hf, self.ndocc))
        ##self.print_matrix_nice(A_tilde_hf, precision=10, width=14, cols_per_line=6)
        #self.print_matrix_nice(A_tilde_hf2-A_tilde_hf, precision=10, width=14, cols_per_line=6)
        X_hf = 0.25 * (A_tilde_hf + Y_hf + A_tilde_hf.T + Y_hf.T)
        #self.print_matrix_nice(X_hf, precision=10, width=14, cols_per_line=6)
        self.update_effective_densities_ao(state, kappa, z_vector, self.eigenvecs, self.C_hf)
        
        #symmetrize effective density
        temp_rdm = self.two_rdm_eff_ao.reshape(self.nmo * self.nmo, self.nmo * self.nmo)
        temp_rdm2 = 0.5 * (temp_rdm + temp_rdm.T)
        temp_rdm3 = temp_rdm2
        self.two_rdm_eff_ao = temp_rdm3.reshape(self.nmo, self.nmo, self.nmo, self.nmo)

        temp_rdm = self.one_rdm_eff_ao.reshape(self.nmo, self.nmo)
        temp_rdm2 = 0.5 * (temp_rdm + temp_rdm.T)
        temp_rdm3 = temp_rdm2
        self.one_rdm_eff_ao = temp_rdm3


        #update 1rdm_pe_eff with contributions from 1rdm_eff and 2rdm_eff
        self.one_rdm_pe_eff_ao += 2.0 *np.einsum("mn, pqmn", self.d_ao, self.two_rdm_eff_ao)
        self.one_rdm_pe_eff_ao += -self.d_exp * self.one_rdm_eff_ao
        #self.one_rdm_pe_eff_ao2 += 2.0 *np.einsum("mn, pqmn", self.d_ao, self.two_rdm_eff_ao)
        #self.one_rdm_pe_eff_ao2 += -self.d_exp * self.one_rdm_eff_ao


        temp_rdm = self.one_rdm_pe_eff_ao.reshape(self.nmo, self.nmo)
        temp_rdm2 = 0.5 * (temp_rdm + temp_rdm.T)
        temp_rdm3 = temp_rdm2
        self.one_rdm_pe_eff_ao = temp_rdm3


        self.Ca_hf = psi4.core.Matrix.from_array(self.C_hf)

        self.overlap_deriv_matrix_ao2 = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.total_gradient = np.zeros(3 * n_atoms)

        self.pulay_force2 = np.zeros(3 * n_atoms)
        # loop over the atoms
        for i in range(self.n_atoms):
            # loop over the cartesian coordinates
            for j in range(3):
                # define the derivative index
                deriv_index = 3 * i + j

                # get the one-electron integral derivatives
                # overlap is in the MO basis
                self.overlap_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.mo_oei_deriv1("OVERLAP", i, self.Ca_like, self.Ca_like)[j])
                self.overlap_deriv_matrix_ao2[deriv_index] = np.asarray(self.mints.mo_oei_deriv1("OVERLAP", i, self.Ca_hf, self.Ca_hf)[j])

                # all others are in the AO basis
                self.potential_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("POTENTIAL", i)[j])
                self.kinetic_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("KINETIC", i)[j])

                # get the two-electron integral derivatives
                self.eri_deriv_matrix_ao[deriv_index] = np.asarray(self.mints.ao_tei_deriv1(i)[j])

                # compute the J and K derivetives
                self.repulsion_gradient[deriv_index] = np.einsum("mnxy,mnxy->", self.eri_deriv_matrix_ao[deriv_index, :, :, :, :], self.two_rdm_eff_ao)


                # now contract each of the derivatives with the density matrix to get the respective gradient components
                # Pulay force first
                self.pulay_force[deriv_index] =  - np.einsum("pq,pq->", X, self.overlap_deriv_matrix_ao[deriv_index])
                self.pulay_force[deriv_index] += - np.einsum("pq,pq->", X_hf, self.overlap_deriv_matrix_ao2[deriv_index])

                # kinetic gradient
                self.kinetic_gradient[deriv_index] = np.einsum("uv,uv->", self.one_rdm_eff_ao, self.kinetic_deriv_matrix_ao[deriv_index, :, :])

                # potential gradient
                self.potential_gradient[deriv_index] = np.einsum("uv,uv->", self.one_rdm_eff_ao, self.potential_deriv_matrix_ao[deriv_index, :, :])


                                                                    

        self.total_gradient = self.nuclear_energy_gradient + self.pulay_force + self.kinetic_gradient + self.potential_gradient + self.repulsion_gradient 
        print("Nuclear energy gradient")
        self.print_matrix_nice(self.nuclear_energy_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        print("Pulay force")
        self.print_matrix_nice(self.pulay_force.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        print("Kinetic gradient")
        self.print_matrix_nice(self.kinetic_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        print("Potential gradient")
        self.print_matrix_nice(self.potential_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        print("Repulsion gradient")
        self.print_matrix_nice(self.repulsion_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)

        # initialize the two-electron integrals derivative matrices
        n_atoms = self.n_atoms
        n_orbitals = self.nmo           

        # initialize three arrays for the K_dse terms
        d_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.dipole_gradient = np.zeros(3 * n_atoms)


        # loop over all of the atoms
        for atom_index in range(n_atoms):
            # Derivatives with respect to x, y, and z of the current atom
            _dip_deriv = np.asarray(self.mints.ao_elec_dip_deriv1(atom_index))
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # get element of d_deriv:
                if cart_index == 0:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[0] + self.lambda_vector[1] * _dip_deriv[3] + self.lambda_vector[2] * _dip_deriv[6]

                elif cart_index == 1:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[1] + self.lambda_vector[1] * _dip_deriv[4] + self.lambda_vector[2] * _dip_deriv[7]

                elif cart_index == 2:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[2] + self.lambda_vector[1] * _dip_deriv[5] + self.lambda_vector[2] * _dip_deriv[8]

                else:
                    raise ValueError("cart_index must be 0, 1, or 2.")
                self.dipole_gradient[deriv_index] = np.einsum("uv, uv->", self.one_rdm_pe_eff_ao, d_derivs[deriv_index, :, :], optimize="optimal")

        print("Dipole gradient")
        self.print_matrix_nice(self.dipole_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        #print("dipole gradient2")
        #print(self.dipole_gradient2)
        #print("dipole gradient3")
        #print(self.dipole_gradient3)




        c_origin = [0.0, 0.0, 0.0]
        max_order = 2

        Dp4 = psi4.core.Matrix.from_array(self.one_rdm_eff_ao)
        self.multipole_grad = np.asarray(self.mints.multipole_grad(Dp4, max_order, c_origin))



        self.o_dse_gradient = np.zeros(3 * self.n_atoms)

        for atom_index in range(self.n_atoms):
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[0] ** 2 * self.multipole_grad[deriv_index, 3]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[1] ** 2 * self.multipole_grad[deriv_index, 6]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[2] ** 2 * self.multipole_grad[deriv_index, 8]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[1] * self.multipole_grad[deriv_index, 4]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 5]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[1] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 7]

        print("Quadrupole gradient")
        self.print_matrix_nice(self.o_dse_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)

        self.total_gradient += self.dipole_gradient + self.o_dse_gradient
        #print("total_gradient 3")
        #total_gradient2 += self.dipole_gradient2 + self.o_dse_gradient
        #print(total_gradient2)
        #print("norm", np.linalg.norm(total_gradient2))
        #print("dse1 +ose")
        #print(self.dipole_gradient + self.o_dse_gradient)
        #print("dse2 +ose")
        #print(self.dipole_gradient2 + self.o_dse_gradient)
 
        print("Total gradient")
        self.print_matrix_nice(self.total_gradient.reshape(self.n_atoms, 3), precision=10, width=14, cols_per_line=6)
        print(self.total_gradient)
        print("gradient norm", np.linalg.norm(self.total_gradient))




