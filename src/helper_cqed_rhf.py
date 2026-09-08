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
import os
import psi4
import numpy as np
import time

#   QED_SCF_TIMING : print a phase breakdown of cqed_rhf     (default 0)
#
# `Completed QED-RHF` is ~155 s on c10h8 and does not thread (24 threads buys
# 4%) -- nothing had ever profiled inside this file.  The accumulators below
# split it into psi4's own SCF, the AO integral builds, and the per-iteration
# SCF kernels.  Off by default: it costs a few time.time() calls per iteration
# and prints a block that existing drivers do not expect.
_QED_SCF_TIMING = os.environ.get("QED_SCF_TIMING", "0") not in ("0", "", "false")
_scf_t = {}

#   QED_SCF_JK : build J/K with psi4's JK object          (default 1)
#   QED_SCF_JK_MEM : doubles of scratch handed to it      (default 1e9 = 8 GB)
#
# The conventional (non-density-fitted) branch used to call mints.ao_eri() and
# hold the whole nbf^4 AO ERI in core -- 8.4 GB and 176-210 s at c10h8 cc-pVDZ
# dimensions -- purely to contract it twice per SCF iteration.  psi4's JK object
# does the same contractions from the occupied orbitals, honouring the deck's
# own SCF_TYPE, so a `pk` deck still gets exact integrals.  Set QED_SCF_JK=0 to
# restore the dense path.
_QED_SCF_JK = os.environ.get("QED_SCF_JK", "1") not in ("0", "", "false")
_QED_SCF_JK_MEM = float(os.environ.get("QED_SCF_JK_MEM", 1e9))
# QED_SCF_REUSE_JK: reuse the JK object psi4's own SCF already built, instead of
# calling psi4.core.JK.build() and forming the PK supermatrix a SECOND time.
# Default 0 = off, so every existing run is unchanged.
#
# MEASURED (c10h8/aug-cc-pVDZ, QED_SCF_TIMING=1): QED-RHF is 344.78 s, of which
# `ao_eri / JK init` is 75.21 s -- and psi4's own timer.dat reports
# "Total PK formation time 231.800w, 2 CALLS".  That second call IS this JK
# build: it collapses to 0.001 s under scf_type=direct, which forms no PK.
# psi4 frees its JK after the SCF (wfn.jk() returns None), but the `save_jk`
# option keeps it alive and it is fully usable afterwards (probe: C_left_add /
# compute() / J,K correct).  So set save_jk and take psi4's object.
# NOT a memory regression: we hold psi4's JK instead of building our own, so at
# most one PK supermatrix is live either way.
_QED_SCF_REUSE_JK = os.environ.get("QED_SCF_REUSE_JK", "0") not in ("0", "", "false")

# QED_SCF_GUESS_DF: run psi4's OWN SCF density-fitted, then restore the deck's
# scf_type before anything else.  psi4's SCF is a GUESS GENERATOR for us -- the
# only product consumed downstream is `C = np.asarray(wfn.Ca())` (:190, "use
# canonical RHF orbitals for guess CQED-RHF orbitals"), which the CQED-RHF
# iterations below re-converge against OUR jk object.
#
# THE KNOB DOES NOT DENSITY-FIT THE REFERENCE.  In the coherent-state basis the
# CASSCF Hamiltonian carries three quantities frozen at the reference:
#     <d_HF>  = d_exp_el          (:577, from the CONVERGED CQED-RHF density)
#     d_c     = 1/2 <d_HF>^2      (:580)
#     d_cmo   = d_el_mo           (:630, from the CONVERGED CQED-RHF orbitals)
# consumed as helper_PFCI.py:6318-6320 -> self.d_exp/self.d_c/self.d_cmo and
# used as CONSTANTS of the Hamiltonian (constdouble[3] = self.d_exp - d_diag,
# helper_PFCI.py:4773).  The CASSCF re-optimises orbitals *within* that
# Hamiltonian; it does not re-optimise its coherent-state origin.  A
# density-fitted CQED-RHF would therefore define a DIFFERENT problem, not a
# worse starting point for the same one.  All three stay exact here because
# `jk` is built at :281 AFTER scf_type is restored, so every CQED-RHF iteration
# uses exact integrals -- only the guess is approximate, and a guess is free.
#
# MUTUALLY EXCLUSIVE with QED_SCF_REUSE_JK, which hands us psi4's *DF* JK object
# and so density-fits the reference after all.  Refused below rather than
# silently preferred, because the failure is a wrong Hamiltonian, not a crash.
#
# WHY IT IS WORTH A KNOB (c10h8/aug-cc-pVDZ, nmo 302, QED_SCF_TIMING=1):
#   default  (pk, reuse off)   psi4 scf 255.4 s + ao_eri/JK init 73.8 s = 341.8
#   QED_SCF_REUSE_JK=1 (pk)    psi4 scf 255.2 s +                  0.0 s = 268.3
#   this knob                  psi4 scf ~ 30  s + ao_eri/JK init 73.8 s = ~104
# psi4's pk SCF pays a PK supermatrix formation AND ~13 iterations against it,
# when all we take from it is Ca().  Here PK is formed once, by us.
#
# psi4's plain-RHF energy IS density-fitted under this knob ("RHF ENERGY" ->
# helper_PFCI.py:6298 self.rhf_reference_energy, which is stored and never
# read, but is also printed).  The knob says so on stdout rather than leaving a
# DF number labelled "Psi4 SCF energy" unremarked.
#
# Value: "1"/"df" = df; any other non-zero string is handed to psi4 as the
# scf_type verbatim (e.g. "disk_df", "mem_df").  QED_SCF_GUESS_DF=0 restores
# psi4's SCF to the deck's own scf_type.
#
# DEFAULT "1" SINCE 2026-09-08 (was "0").  It was measured and validated in
# item 370 and then shipped off, so the ~238 s it saves at nmo 302 never
# reached a real run.  Nothing downstream of psi4's SCF changes: only Ca() is
# consumed, and the CQED-RHF iterations that produce <d_HF>, d_c and d_cmo run
# against the exact `jk` built at :281 after scf_type is restored.
_QED_SCF_GUESS_DF_RAW = os.environ.get("QED_SCF_GUESS_DF")
_QED_SCF_GUESS_DF_EXPLICIT = _QED_SCF_GUESS_DF_RAW is not None
_QED_SCF_GUESS_DF = "1" if _QED_SCF_GUESS_DF_RAW is None else _QED_SCF_GUESS_DF_RAW
_QED_SCF_GUESS_DF = None if _QED_SCF_GUESS_DF in ("0", "", "false") else (
    "df" if _QED_SCF_GUESS_DF in ("1", "true") else _QED_SCF_GUESS_DF)
if _QED_SCF_GUESS_DF is not None and _QED_SCF_REUSE_JK:
    if not _QED_SCF_GUESS_DF_EXPLICIT:
        # Only the DEFAULT collided.  A default must never turn another knob
        # into an import-time crash, so QED_SCF_REUSE_JK=1 alone still means
        # what it meant before this became a default: yield to it, and say so.
        _QED_SCF_GUESS_DF = None
        print("QED_SCF_GUESS_DF: default off because QED_SCF_REUSE_JK=1 "
              "(mutually exclusive); set QED_SCF_GUESS_DF=1 explicitly to see "
              "the conflict reported instead.")
    else:
        raise ValueError(
            "QED_SCF_GUESS_DF and QED_SCF_REUSE_JK are mutually exclusive: "
            "reusing psi4's SCF JK object under a density-fitted SCF would "
            "density-fit the CQED-RHF iterations, and with them <d_HF>, d_c "
            "and d_cmo -- the frozen coherent-state constants of the CASSCF "
            "Hamiltonian.  Pick one.")

#   QED_PSI4_MEM : memory handed to psi4          (default unset = psi4's own)
#
# THIS IS NOT A SOLVER KNOB -- it is the companion to NTHREAD (helper_PFCI.py
# :471) and exists for the same class of defect: a process-wide psi4 default
# that silently degrades an algorithm the deck asked for.  psi4 defaults to
# 500 MiB.  At that size a `scf_type pk` deck does NOT get in-core PK -- psi4
# falls back to `DiskJK: Disk-Based J/K Matrices` and writes the AO integrals
# out in batches, then re-reads all of them on every SCF iteration.  Measured
# on c10h8 cc-pVDZ (nbf=180, 1.06 GB of PK integrals, 13 psi4 SCF iterations,
# 8 threads), psi4.energy("scf") alone:
#     default 500 MiB (DiskJK)   35.9 s
#     8 GB     (in-core PK)      14.0 s
#
# DELIBERATELY OPT-IN, for the same reason NTHREAD is: the right value is a
# property of the SYSTEM and the machine, not something this file can guess,
# and over-promising memory on a shared node is worse than a slow SCF.  Leaving
# it unset preserves today's exact behaviour for every existing driver.
#
# Accepts anything psi4.set_memory does: a byte count ("8e9", "8000000000") or
# a unit string ("8 GB", "800 MB").
_QED_PSI4_MEM = os.environ.get("QED_PSI4_MEM")
if _QED_PSI4_MEM:
    try:
        psi4.set_memory(int(float(_QED_PSI4_MEM)))
    except ValueError:
        # not a bare number -- hand the string to psi4 to parse the unit
        psi4.set_memory(_QED_PSI4_MEM)


def _tick(key, t0):
    """Accumulate elapsed time since t0 under `key`; return a fresh timestamp."""
    now = time.time()
    if _QED_SCF_TIMING:
        _scf_t[key] = _scf_t.get(key, 0.0) + (now - t0)
    return now


def b_coefficient(error_vectors):
    b_mat = np.zeros((len(error_vectors) + 1, len(error_vectors) + 1))
    b_mat[-1, :] = -1
    b_mat[:, -1] = -1
    b_mat[-1, -1] = 0
    rhs = np.zeros((len(error_vectors) + 1, 1))
    rhs[-1, -1] = -1
    for i in range(len(error_vectors)):
        for j in range(i + 1):
            # .item(): error_vectors are (N,1) column vectors, so this dot is a
            # (1,1) array, not a scalar.  numpy <=2.2 squeezed it implicitly on
            # assignment; numpy >=2.3 raises "setting an array element with a
            # sequence".  Value-identical to the old implicit squeeze.
            b_mat[i, j] = np.dot(error_vectors[i].transpose(), error_vectors[j]).item()
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
    if _QED_SCF_REUSE_JK:
        # must be set BEFORE the SCF, or psi4 frees the JK when it finishes
        psi4.set_options({"save_jk": True})
    # QED_SCF_GUESS_DF: density-fit psi4's SCF, which supplies only the guess
    # orbitals.  The deck's own scf_type is restored immediately afterwards, so
    # the jk built at :281 -- and every quantity the coherent-state Hamiltonian
    # freezes at the reference -- is exact.  See the knob's note above.
    _deck_scf_type = None
    if _QED_SCF_GUESS_DF is not None:
        _deck_scf_type = next(
            (v for k, v in psi4_options_dict.items() if k.lower() == "scf_type"),
            psi4.core.get_global_option("SCF_TYPE"))
        print(f"QED_SCF_GUESS_DF: psi4 SCF runs scf_type={_QED_SCF_GUESS_DF} for "
              f"the guess orbitals only; scf_type={_deck_scf_type} is restored "
              f"before the CQED-RHF jk is built.  The 'Psi4 SCF energy' printed "
              f"below is therefore density-fitted; <d_HF>, d_c and d_cmo are not.",
              flush=True)
        psi4.set_options({"scf_type": _QED_SCF_GUESS_DF})
    # run psi4 to get ordinary scf energy and wavefunction object
    _t = time.time()
    psi4_rhf_energy, wfn = psi4.energy("scf", return_wfn=True)
    _t = _tick("psi4 scf", _t)
    if _deck_scf_type is not None:
        # restore BEFORE mints/JK.build: psi4.core.JK.build() reads the global
        # SCF_TYPE, and under disk_df it does not merely return a DF object --
        # it raises "Bad BraKet type in Libint2TwoElectronInt".
        psi4.set_options({"scf_type": _deck_scf_type})

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    #basisset = wfn.basisset()
    #mol = basisset.molecule()
    #ao_idx = 0
    #for ishell in range(basisset.nshell()):
    #    shell = basisset.shell(ishell)
    #    center = shell.ncenter
    #    am = shell.am  # 0=s, 1=p, 2=d
    #    nfunc = shell.nfunction  # 1 for s, 3 for p, 5 for d (spherical)
    #    symbol = mol.symbol(center)
    #    am_label = 'spdfg'[am]
    #    
    #    for f in range(nfunc):
    #        print(f"AO {ao_idx:3d}: {symbol}{center}  {am_label}  (shell {ishell})")
    #        ao_idx += 1
           
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
    _t = _tick("ao 1-e (V,T)", _t)
    jk = None
    if "df_basis_scf" in psi4_options_dict:
        df_basis_name = psi4_options_dict["df_basis_scf"]
        density_fitting = True
        primary_basis = wfn.basisset()
        mol = wfn.molecule()
        nbf = primary_basis.nbf()
        use_spherical = primary_basis.has_puream()

        df_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", df_basis_name, puream=use_spherical)
        naux = df_basis.nbf()

        #print(f"DF reconstruction: {nbf} primary, {naux} auxiliary")

        ## Get DF tensor
        #dfh = psi4.core.DFHelper(primary_basis, df_basis)
        ##dfh.set_schwarz_cutoff(1e-20)   # effectively disable screening
        #dfh.set_schwarz_cutoff(0.0)   # disable screening entirely 
        ##dfh.set_fitting_condition(1e-12) # match Psi4's internal metric threshold
        #dfh.initialize()

        #identity = psi4.core.Matrix("Identity", nbf, nbf)
        #identity.identity()

        #dfh.add_space("AO", identity)
        #dfh.add_transformation("B", "AO", "AO")
        #dfh.transform()

        #B_tensor = dfh.get_tensor("B")
        #B_array = np.asarray(B_tensor)
        #print(f"DF tensor shape: {B_array.shape}")
       

        zero = psi4.core.BasisSet.zero_ao_basis_set()
        orb_basis = wfn.basisset()
        #aux_basis = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", orb_basis.name())
        aux_basis = wfn.get_basisset("DF_BASIS_SCF")  # Get SCF's actual DF basis
        raw_3c = mints.ao_eri(aux_basis, zero, orb_basis, orb_basis)
        # Convert to numpy and reshape
        naux = aux_basis.nbf()
        nbf = orb_basis.nbf()
        raw_3c_np = np.asarray(raw_3c).reshape(naux, nbf, nbf)

        metric_obj = psi4.core.FittingMetric(aux_basis, True)
        metric_obj.form_eig_inverse(1.0e-10)
        J_inv_half = np.asarray(metric_obj.get_metric())

        # Apply metric: (pq|Q) = J^{-1/2} @ (pq|A)
        #
        # B_Q,mn = sum_A Jinv[Q,A] raw[A,m,n] is a single GEMM once the (m,n)
        # pair index is flattened -- the leading index is the one being
        # contracted and (m,n) are pure spectators.  Was
        # np.einsum('QA,Amn->Qmn', ...), which numpy runs as a naive nested
        # loop over Q,A,m,n with no BLAS and no threading: naux^2 * nbf^2
        # scalar operations.
        B_array = (J_inv_half @ raw_3c_np.reshape(naux, nbf * nbf)).reshape(
            naux, nbf, nbf
        )
        del raw_3c, raw_3c_np

        #diff = np.max(np.abs(B_array - B_ref))
        #print(f"Difference: {diff:.2e}")  # Should be ~1e-10 or smaller

        #Correct contraction for (naux, nbf, nbf) format
        #if B_array.shape == (naux, nbf, nbf):
        #    print("Using Einstein summation: (μν|λσ) = Σ_P B_P^μν * B_P^λσ")
        #    I = np.einsum('Pmn,Pls->mnls', B_array, B_array)

    else:
        density_fitting = False
        if _QED_SCF_JK:
            # Build J and K through psi4's own JK object instead of
            # materialising the dense nbf^4 AO ERI.  `I` was used for NOTHING
            # but the two contractions in the SCF loop and is never returned,
            # so nothing else in the package sees this change.
            #
            # jk_type defaults to the deck's SCF_TYPE, so a `pk` deck still
            # gets exact (non-density-fitted) integrals -- this is not a
            # switch to DF.  Measured on c10h8 cc-pVDZ (nbf=180): the
            # mints.ao_eri() call alone was 176-210 s and the array 8.4 GB.
            I = None
            jk = wfn.jk() if _QED_SCF_REUSE_JK else None
            if jk is None:
                jk = psi4.core.JK.build(wfn.basisset())
                jk.set_memory(int(_QED_SCF_JK_MEM))
                jk.initialize()
            else:
                # psi4's SCF already initialised it; re-initialising would
                # re-form PK, which is the whole cost being avoided here.
                print("QED_SCF_REUSE_JK: reusing psi4's SCF JK object",
                      flush=True)
        else:
            jk = None
            I = np.asarray(mints.ao_eri())
    _t = _tick("ao_eri / JK init", _t)
    print("density_fitting", density_fitting)


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
    S = np.asarray(mints.ao_overlap())
    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.0e-16)
    A = np.asarray(A)
    _t = _tick("ao dipole/quadrupole/overlap", _t)

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
            _t = time.time()
            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141\
            if density_fitting == False:
                if jk is not None:
                    # D is idempotent (D = Cocc Cocc^T) at every iteration, so
                    # handing psi4 Cocc reproduces exactly the two contractions
                    # below with the same D.
                    jk.C_clear()
                    jk.C_left_add(psi4.core.Matrix.from_array(Cocc))
                    jk.compute()
                    J = np.asarray(jk.J()[0])
                    K = np.asarray(jk.K()[0])
                else:
                    J = np.einsum("pqrs,rs->pq", I, D)
                    K = np.einsum("prqs,rs->pq", I, D)
            else:
                # All four contractions below were 2-operand np.einsum calls,
                # which numpy evaluates as naive nested loops -- no BLAS, no
                # threading.  The K pair was the expensive one at
                # naux * nbf^3 scalar operations each.  Rewritten as GEMMs;
                # `B_flat` and `Cocc` views cost nothing (B_array is
                # C-contiguous, so the reshape is a view, not a copy).
                B_flat = B_array.reshape(naux, nbf * nbf)

                # J: contract B with D over the flattened (m,n) pair, then
                # expand back -- two GEMVs.
                Q_temp = B_flat @ D.ravel()
                J = (Q_temp @ B_flat).reshape(nbf, nbf)

                # K: route through the OCCUPIED index instead of the full AO
                # pair.  D = Cocc Cocc^T at every iteration, so
                #   K_pq = sum_{Q,r,s} B_Qpr D_rs B_Qqs
                #        = sum_{Q,i} Bo_Qpi Bo_Qqi,   Bo_Qpi = sum_r B_Qpr C_ri
                # This is naux*nbf^2*ndocc work instead of naux*nbf^3, and Bo
                # is ndocc/nbf the size of the old `temp1` intermediate.
                Bo = (B_array.reshape(naux * nbf, nbf) @ Cocc).reshape(
                    naux, nbf, ndocc
                )
                Bo = Bo.transpose(1, 0, 2).reshape(nbf, naux * ndocc)
                K = Bo @ Bo.T

                #jk = psi4.core.JK.build_JK(primary_basis, df_basis)
                #jk.set_memory(int(1e9))
                #jk.initialize()
                #
                #C_occ = psi4.core.Matrix.from_array(C[:, :ndocc])  # occupied MO coefficients
                #jk.C_left_add(C_occ)
                #jk.compute()
                #
                #J_ref = np.asarray(jk.J()[0])
                #K_ref = np.asarray(jk.K()[0])
                ## Compare with your computed J, K
                #print("J max diff:", np.max(np.abs(J - J_ref)))
                #print("K max diff:", np.max(np.abs(K - K_ref)))                
            _t = _tick("  J/K build", _t)
            # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
            # M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
            #
            # N_pq = sum_rs d_pr d_qs D_rs = (d @ D @ d.T)_pq.  Was
            # np.einsum("pr,qs,rs->pq", d_el_ao, d_el_ao, D): a THREE-operand
            # einsum, which numpy evaluates as a single nested loop over
            # p,r,q,s with no BLAS and no threading -- i.e. O(nbf^4) scalar
            # work for a quantity that is two GEMMs.  2.21 s -> 0.2 ms at
            # nbf=180 (c10h8 cc-pVDZ).  Same contraction, agrees to 1e-11.
            N = d_el_ao @ D @ d_el_ao.T

            # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
            # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
            F = H + 2 * J - K - N
            # save current Fock matrix into memory
            fock_list.append(F)
            # FDS - SDF.  The second term is the transpose of the first because
            # F, D and S are all symmetric, so one triple product suffices.
            # Was two three-operand einsums, again O(nbf^4) scalar work:
            # 1.48 s -> 0.3 ms at nbf=180.
            FDS = F @ D @ S
            diis_e = FDS - FDS.T
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
            _t = _tick("  DIIS error + energy", _t)
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
            _t = _tick("  DIIS extrap + eigh + density", _t)

            if SCF_ITER == maxiter:
                psi4.core.clean()
                raise Exception("Maximum number of SCF cycles exceeded.")

        print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
        if jk is not None:
            jk.finalize()
        if _QED_SCF_TIMING:
            print(" cqed_rhf phase breakdown (%d SCF iterations):" % SCF_ITER)
            for k, v in _scf_t.items():
                print("   %-32s %9.3f s" % (k, v))
            print("   %-32s %9.3f s" % ("TOTAL ACCOUNTED", sum(_scf_t.values())))
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
