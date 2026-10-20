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
from residual_minimization import *
from itertools import combinations
import math
import time
import ctypes
import numpy as np
from ctypes import *
import os
import atexit      # QED_JK_H2E_DISK: remove the spilled tensor at exit
import tempfile    # QED_JK_H2E_DISK
import psutil
import copy
import weakref  # build_intermediates' L cache invalidates by weak ref to J/K
import scipy.sparse
import ortho_script
from ortho_script import ortho_orbs
from scipy.stats import ortho_group
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import LinearOperator
from timeit import default_timer as timer
from scipy.linalg import eigh_tridiagonal
from collections import deque

# --- CI-cost experiment knobs (both default-no-op) --------------------------
# Profiling of the real MgH+ CAS(8,12) runs (see
# cpp_casscf/validation/ci_cost_experiment/README.md) showed the CI step is
# 71% of total runtime, and within it the COUPLED (qn_count > 0) CI solves in
# microiteration_optimization6 are ~79% of all Davidson time -- 833 calls,
# mean 25.9 iterations, 16.2 s each. By contrast the whole orbital side
# (orbital sigma, gradient, Hessian diagonal, intermediates, all integral
# transforms, LSTRS/GLTR expand+residual) is 0.05% of runtime.
#
#   QED_CI_UNCOUPLED_MAXITER : the Kreplin uncoupled CI cap        (default 5)
#   QED_CI_COUPLED_FACTOR    : the 0.1 in 0.1*||g_orb||            (default 0.1)
#   QED_CI_COUPLED_MAXITER   : iteration budget for the COUPLED CI (default 5;
#                              set 10000 to recover the effectively-uncapped
#                              behaviour this code had before 2026-08-25)
#
# The coupled threshold is tied to the orbital gradient, so it tightens as the
# optimization converges; measured cost of that tightening is mean iterations
# rising 23.9 -> 29.9 from the largest to the smallest gradient-norm quartile.
#
# WHERE 0.1 COMES FROM.  It is Kreplin et al.'s own constant: their QN-coupled
# microiteration tightens the Davidson solve at step k to
# |r_ci| < 0.1 * |g_orb^(k-1)| (the PREVIOUS microiteration's orbital gradient,
# which is the alignment used below -- reduced_gradient is built before this
# iteration's orbital step).  So the factor is prescribed, not a free choice;
# an earlier comment here calling it "a choice, not a derived value" was wrong.
#
# WHY A BUDGET IS NEEDED ANYWAY.  Molpro does not chase that threshold.  In its
# QN phase (the micro-table rows with NQN >= 1) NCI is 2 -- occasionally 3 --
# on every row of both reference outputs, and it simply accepts whatever
# residual 2 iterations produce:
#   examples/c10h8_detail.out    job 2  51 QN rows  |r_ci|/|g_orb^(k-1)| median
#                                       0.22, 33% of rows below 0.1
#   examples/mghplus_detail_v2.out job 2  26 QN rows  median 2.42, min 0.21,
#                                       NOT ONE row below 0.1
# i.e. the 0.1 acts as an EARLY EXIT that fires only where a 2-3 iteration
# budget can reach it (small single-root CI), and never binds on the larger
# state-averaged case.  Before 2026-08-25 we had no budget at all
# (constint[8] = 10000), so on MgH+ we chased 0.1 to a mean 8.09 Davidson
# iterations per coupled solve against Molpro's 2, and coupled solves were 89.6%
# of that run's CI time.  QED_CI_COUPLED_MAXITER supplies the missing budget;
# threshold and budget race, first to fire wins, exactly as in Molpro.
#
# WHY THE DEFAULT IS 5 (shipped 2026-08-25; evidence in items 267-271).  On MgH+
# CAS(8,12) it is -45.1% wall from the canonical guess (351.6 -> 193.0 s, 8 macros
# vs 11) and -55.2% median over 14 RANDOM orbital guesses, with the mean macro
# count EXACTLY unchanged (13.86 -> 13.86) and every seed landing on the same
# stationary point (max dE 2.1e-9).  A c10h8 CAS(10,10) QN-ON cross-check shows no
# regression (macros 4=4, microiterations 17=17, energy identical to 5.5e-12, wall
# 1.002x) -- though the cap barely binds there (5.38 -> 4.78 iterations/solve), so
# that leg proves no-harm, not benefit.  KNOWN COST, measured and accepted: on an
# already-near-converged guess the CI is nearly solved and 5 iterations
# under-resolve it, so the orbital step can need redoing -- 2 of 14 random seeds
# spent extra macros (worst +7).  Wave 2 tested budgets 8 and 10 and neither fixed
# that (the penalty is flat in the budget: base needs ~21-25 iterations on those
# seeds), while both gave back win on the hard seeds.  So the cost is WASTED TIME
# ON EASY CASES, never a wrong answer.  Set 10000 to restore the old behaviour.
_QED_CI_UNCOUPLED_MAXITER = 5  # resolved per calculation by qed_ci.runtime
_QED_CI_COUPLED_FACTOR = 0.1  # resolved per calculation by qed_ci.runtime
_QED_CI_COUPLED_MAXITER = 5  # resolved per calculation by qed_ci.runtime
#
#   QED_ORBITAL_SYMMETRY : constrain the orbital optimization to preserve the
#                          AXIAL symmetry of a linear molecule (default 0 = off,
#                          an exact no-op).
#
# WHY.  On MgH+ CAS(8,12) we converge 4.08e-04 Ha BELOW Molpro, and item 277
# established the reason: our converged active space BREAKS the axial symmetry,
# while Molpro's "extra symmetry" machinery labels every MO sigma/pi/delta and
# projects out the rest, so Molpro cannot represent our solution.  Handing our
# orbitals to Molpro with `noextra` reproduced our CI energy to 1.7e-10 and it
# STAYED there.  This knob reproduces Molpro's constraint so the two codes can be
# compared on the same footing.
#
# *** LEAVE IT OFF FOR CAVITY WORK. *** The PF coupling carries a DIRECTION
# (`lambda_vector`); a mode polarised off the molecular axis genuinely breaks the
# axial symmetry, so constraining the orbitals would exclude solutions the QED
# problem really has.  Validation mode only, i.e. at lambda = 0.
#
# HOW.  Nothing in the microiteration, the internal optimization, the
# trust-region solvers or the C backend changes.  It (a) projects the starting
# orbitals to be symmetry pure and (b) adds ONE filter -- "only rotate two MOs of
# the same symmetry species" -- everywhere the orbital-rotation list is
# enumerated.  See orbital_symmetry.py.
_QED_ORBITAL_SYMMETRY = int(os.environ.get("QED_ORBITAL_SYMMETRY", "0"))
#
#   QED_RANDOM_SEED : seed numpy's global RNG once, at import, so a run is
#                     REPRODUCIBLE.  Several solver sites draw random numbers
#                     with no seed -- `solve_gltr_trust_region`'s noise
#                     (trust_region_solvers.py:673,965), its `trial_0` vectors
#                     (:1439,1590) and the sigma/Davidson starts below -- so two
#                     runs of the same input do NOT agree to all digits.  That is
#                     noise in the reference, not in the theory: a numerical
#                     gradient/NAC check repeats with ~1.7x scatter (gradients)
#                     and ~13x (couplings), which is enough to make a validation
#                     figure unreproducible for a referee.  Seeding the GLOBAL
#                     legacy RNG covers every `np.random.*` call in the package
#                     with one knob, rather than threading a generator through
#                     four modules.
#                     NECESSARY BUT NOT SUFFICIENT.  Measured 2026-09-17 on
#                     numerical_gradient_check.py (h2o_cav, 3 components):
#                       8 threads + fixed seed -> 5.95e-07 vs 1.00e-06  (differs)
#                       1 thread  + no seed    -> 3.15e-07 vs 4.64e-07  (differs)
#                       1 thread  + fixed seed -> BITWISE identical
#                     There are TWO independent sources: this RNG, and
#                     non-deterministic reduction order in the threaded C/MKL
#                     backend (floating-point addition is not associative, and a
#                     CASSCF amplifies the last bit).  A reproducible run needs
#                     BOTH this seed AND OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1.
#                     DEFAULT OFF (empty): behaviour is bit-for-bit unchanged
#                     unless the variable is set.
_QED_RANDOM_SEED = os.environ.get("QED_RANDOM_SEED", "")
#   QED_D2H_PROJECT : group-average the orbital rotation generator over D2h at
#                     every step, so a symmetric solution STAYS symmetric.
#                     Default OFF -- a no-op on every existing run.  Only
#                     meaningful at a D2h geometry; the projector validates the
#                     geometry itself and disables if it does not hold.
_QED_D2H_PROJECT = os.environ.get("QED_D2H_PROJECT", "0") == "1"
#
#   QED_MACRO_MAXITER : hard cap on CASSCF macroiterations (default 1000, i.e.
#                       the previous hardcoded bound, so behaviour is unchanged
#                       unless set).  A run that lands on a DISFAVOURED solution
#                       is not merely worse in energy, it is far more expensive:
#                       at lambda = 0.15 CAS(4,7) the three seeds on the good
#                       branch finished in ~6 min with 8 MB logs, while the one
#                       on a branch 0.109 eV higher was still going at 41 min
#                       with a 59 MB log.  Capping lets a multiplicity SCAN --
#                       where only the converged SA is needed to identify which
#                       family a guess found -- skip the expensive branches
#                       instead of paying full convergence for solutions that
#                       will be rejected anyway.  Energies from a capped run are
#                       NOT converged: use them to classify, never to quote.
#                       HAZARD, measured 2026-09-19: capping BREAKS any run
#                       that subsequently asks for a GRADIENT.  `opt_C` is
#                       assigned only on the converged branch, so a capped
#                       CASSCF leaves it unset and nuclear_grad.py:2746
#                       dies with "AttributeError: 'nuclear_grad' object
#                       has no attribute 'opt_C'".  Energy-only paths are
#                       untested.  To cap an OPTIMIZATION scan, put a
#                       wall-clock timeout on the process instead.
_QED_MACRO_MAXITER = int(os.environ.get("QED_MACRO_MAXITER", "1000"))
if _QED_RANDOM_SEED != "":
    np.random.seed(int(_QED_RANDOM_SEED))
#
#   QED_CI_INTERNAL_FACTOR : scale the internal-optimization CI threshold to the
#                            current internal-internal orbital gradient
#                            (default 0 = off => the fixed 1e-5 as before)
#   QED_CI_INTERNAL_FLOOR / _CEIL : clamp on that scaled threshold
#   QED_INTERNAL_MAXITER   : the internal-optimization iteration cap (default 20)
#
# WHY.  internal_optimization3 pins constdouble[4] = 1e-5 (:10476) while its exit
# test (:10629) is
#     if (gradient_norm < 1e-4 and self.constint[8] == 0) or microiteration == 20
# and constint[8] is the C solver's CONVERGED flag -- zeroed only on the
# converged exit, left nonzero on the maxiter exit.  So the loop may terminate
# only if the CI reached a FIXED 1e-5 within QED_CI_UNCOUPLED_MAXITER (5)
# Davidson iterations; if it does not, the loop cannot exit however small the
# internal-internal orbital gradient becomes, and runs all 20 iterations.
#
# MEASURED IN MOLPRO (src/examples/c10h8.out, 10 "Starting internal optimizaton"
# blocks; NOTE every row of those blocks has 10 fields -- the 8-field first row
# hazard applies to the MICRO-ITERATION blocks, not these):
#   * CI RES / ORB GRAD at the SAME step has median 0.46 over the well-behaved
#     blocks (1,2,3,6: 0.29-0.57), rising to ~1.0 in blocks 4-5.  The same ratio
#     taken against the PREVIOUS step's gradient is far more scattered
#     (median 0.12, range 0.001-0.55), so same-step is the alignment in use.
#   * ORB GRAD falls 26-83x per block of 4 steps -- a RELATIVE reduction, not a
#     chase to an absolute floor.
#   * Blocks run 4,4,4,4,4,4,4,2,1 steps: 4 is a MAX WITH AN EARLY EXIT.
#     Early exits seen at ORB GRAD 3.0e-8 and 2.6e-7; no exit at 2.7e-6 or
#     8.7e-6, bracketing an absolute floor near 1e-6.
# So the CI is never solved tighter than the orbital step can use.  Against a
# fixed 1e-5 we over-solve early (ORB GRAD ~1e-2 wants ~5e-3) and under-solve
# late.  This is the same device already used at the coupled microiteration site
# (:16000, _QED_CI_COUPLED_FACTOR * ||reduced_gradient||), with a constant near
# 0.5 instead of 0.1.
#
# Scaling the threshold should also make the EXISTING exit test work rather than
# needing to be relaxed: a target proportional to the current gradient stays
# reachable inside the 5-iteration cap, so constint[8] comes back 0 and the loop
# exits on the gradient instead of grinding to 20.
#
#   QED_DUMP_CI_VECTORS = <dir> : dump every converged CI vector (default "" = off)
#
# Measures whether a CI-VECTOR HISTORY makes a better guess than the single
# previous vector the warm start uses now.  The question is geometric and is the
# same one that killed the P-space idea (item 225: the warm vector was measured
# 96.4% INSIDE a 378-CSF P-space, so adding that space re-spanned what we
# already had).  Here the candidate subspace is instead the span of the last k
# CONVERGED CI vectors -- directions the CI actually needed as the orbitals
# moved, which is the failure mode: warm-start quality tracks the SIZE OF THE
# ORBITAL STEP, not the macroiteration index.  Measured on c10h8, the guess
# residual at Davidson iteration 1 runs 2.27e-2 (macro 0) -> 6.09e-8 (macro 9),
# a factor of ~370,000, but is NOT monotone -- macros 5 and 6 spike back to
# 8.1e-3 / 3.4e-3 and cost 13 and 14 CI iterations.
# Each file is one converged solve: seq, context, macroiteration (-1 where the
# site does not have it in scope), and the vectors themselves.
_QED_DUMP_CI_VECTORS = os.environ.get("QED_DUMP_CI_VECTORS", "")
_QED_CI_INTERNAL_FACTOR = float(os.environ.get("QED_CI_INTERNAL_FACTOR", "0"))
_QED_CI_INTERNAL_FLOOR = float(os.environ.get("QED_CI_INTERNAL_FLOOR", "1e-8"))
_QED_CI_INTERNAL_CEIL = float(os.environ.get("QED_CI_INTERNAL_CEIL", "1e-4"))
_QED_INTERNAL_MAXITER = int(os.environ.get("QED_INTERNAL_MAXITER", "20"))
#
#   QED_FRESH_HDIAG_SORT : re-argsort H_diag3 each microiteration (default 0)
#
# self.H_diag3 IS rebuilt from the current orbitals every microiteration
# (microiteration_optimization6, :15910-15940), but self.index_Hdiag -- its
# argsort -- is assigned ONLY during setup (:3693 and :3900) and is never
# refreshed afterwards.
# The DETERMINANT SELECTION does not care: ci_solver.c:2329-2340 scans Hdiag2
# (a fresh copy of the current H_diag3) for its minima and never touches
# index_Hdiag.  The SPIN-COUPLING CLOSURE does:
#     while(indim < H_dim &&
#           Hdiag[index_Hdiag[indim]] - Hdiag[index_Hdiag[indim-1]] < 1e-10)
# (ci_solver.c:2242) walks the CURRENT diagonal in the STALE sort order, so
# after the orbitals rotate the two elements it differences are no longer
# adjacent in sorted order and the degeneracy test is meaningless.  Per that
# loop's own comment, a set cut mid-degenerate-group leaves S^2 non-block-
# diagonal over the selected determinants and contaminates the spin-filtered
# guess.
# ONLY THE COLD PATH IS AFFECTED: casscf == true takes indim = nroots and never
# reaches the closure loop, so no warm/production run touches index_Hdiag.  The
# exposure is exactly the initial CASCI (where the setup argsort is still
# current, hence correct) and QED_COLD_START_MICRO (where it is stale).
# Default off so it is a measurable A/B rather than a silent behaviour change;
# the cost is one argsort of H_dim per microiteration.
_QED_FRESH_HDIAG_SORT = int(os.environ.get("QED_FRESH_HDIAG_SORT", "0"))
_QED_SKIP_GKL = int(os.environ.get("QED_SKIP_GKL", "0"))          # see the note at :1183
_QED_LIBTRANS_OO = int(os.environ.get("QED_LIBTRANS_OO", "0"))    # see the note at :1210
# ---------------------------------------------------------------------------
# QED_JK_H2E_DISK : spill the packed self.twoeint to a file and let the JK
#                   kernel stream it back a row-block at a time  (default "" =
#                   off, tensor stays resident -- an exact no-op)
#
# STEP 1(b) OF THE SLW99 RESTRUCTURE.  self.twoeint is the packed (npair,npair)
# MO tensor -- nmo^4/4 doubles, and it is stored SQUARE though it is symmetric:
# 2.0 GiB at nmo 180, 15.60 GiB at nmo 302, 53.9 GiB at nmo 412.  It is the
# single largest allocation in the run and the reason nmo 302 needs a 31.5 GB
# peak while Molpro and pyscf both stream out of core.
#
# Once QED_SKIP_GKL is on, the tensor's ONLY consumer is the per-macro J/K
# build (the note at :1203 establishes that), and inside orbital.c the ONLY
# consumer is the S2 quarter transform, which step 1(a) already blocked over
# rows.  So the array need not be resident at all: write it once, stream it
# per macro.  The C side is orbital.c's `qed_jk_set_h2e_file`; step 1(a)'s
# QED_JK_BLOCK / QED_JK_BLOCK_MB choose the block, and validation/
# h2e_file_invariance.py proves the streamed J and K are BITWISE identical.
#
# Set it to "1" for a file in $TMPDIR, or to a DIRECTORY to place the file
# there (put it on a fast local disc, not NFS -- the whole tensor is re-read
# once per macroiteration, which is the cost this route trades RSS for).
# The file is removed at interpreter exit and on the next spill.
#
# NOT COMPATIBLE WITH QED_LIBTRANS_OO=2, which is a check mode that reads the
# resident tensor on purpose; that combination is refused at spill time.
_QED_JK_H2E_DISK = os.environ.get("QED_JK_H2E_DISK", "").strip()
# Spilled files are unlinked at exit through a MODULE-level registry, not
# through atexit.register(self._drop_h2e_file): a bound method would keep the
# whole PFHamiltonianGenerator (and its J, K, CI vectors) alive for the life of
# the process, which turns a scan script that builds one object per geometry
# into a leak.
_H2E_SPILL_FILES = set()


# QED_JK_H2E_STREAM (default 1, but only consulted when QED_JK_H2E_DISK is
# armed, which is itself default off).  1 = read the libtrans buffer a row-block
# at a time straight into the spill file, so the packed tensor is never
# materialised; 0 = the pre-2026-09-02 behaviour, build it whole and write it
# out afterwards.  0 exists so the two can be A/B'd -- item 354B measured the
# whole-matrix readback's transient at 30.86 GiB against a 15.6 GiB tensor.
_QED_JK_H2E_STREAM = int(os.environ.get("QED_JK_H2E_STREAM", "1"))
# Rows per streamed block are sized from a MiB budget, like QED_JK_BLOCK_MB on
# the C side; both are independent of each other.
_QED_JK_STREAM_MB = int(os.environ.get("QED_JK_STREAM_MB", "256"))
# QED_LIBTRANS_MEM_MB: cap the memory grant handed to psi4's libtrans
# (0 = the historical max(psi4 memory, 8*npair^2), i.e. an exact no-op).
# See the note at the set_memory call site; with the streamed readback this
# grant, not our tensor, is what sets the run's peak.
_QED_LIBTRANS_MEM_MB = int(os.environ.get("QED_LIBTRANS_MEM_MB", "0"))


def _jk_stream_armed(shim):
    """True when the streamed readback should be used for this build."""
    if not _QED_JK_H2E_DISK or _QED_JK_H2E_DISK == "0":
        return False
    if not _QED_JK_H2E_STREAM:
        return False
    for _f in ("qed_dpd_stream_open", "qed_dpd_stream_read",
               "qed_dpd_stream_close"):
        if not hasattr(shim, _f):
            # Loud, not silent: the point of the route is the memory, and a
            # quiet fall back to the whole-matrix reader would look like a
            # merely-slow run while the transient it exists to remove is still
            # there.
            raise RuntimeError(
                "QED_JK_H2E_DISK with QED_JK_H2E_STREAM=1 needs a "
                "psi4_dpd_shim.so that exports %s -- rebuild it (the build "
                "line is in psi4_dpd_shim.cpp's header), or set "
                "QED_JK_H2E_STREAM=0 to use the whole-matrix readback" % _f)
    return True


def _h2e_spill_cleanup():
    for _p in list(_H2E_SPILL_FILES):
        try:
            os.unlink(_p)
        except OSError:
            pass
        _H2E_SPILL_FILES.discard(_p)


atexit.register(_h2e_spill_cleanup)
#
#   QED_DISABLE_QN : never activate quasi-Newton / coupled CI  (default 0 = off)
#
# QN activation (microiteration_optimization6, the `step_norm < 0.05` trigger on
# an accepted step) is what flips the micro CI solve from the Kreplin UNCOUPLED
# cap (constint[8] = _QED_CI_UNCOUPLED_MAXITER, default 5) to COUPLED
# (constint[8] = _QED_CI_COUPLED_MAXITER, default 5 since 2026-08-25, run to a
# 0.1*||g_orb||-scaled threshold -- whichever of budget or threshold fires
# first). Setting this to
# 1 keeps the whole run uncoupled, which is the apples-to-apples setting for
# comparing against an UNCOUPLED-CI reference (e.g. the Molpro c10h8 benchmark in
# examples/c10h8.out) and also matches the cpp_casscf port, whose QN/BFGS path is
# deliberately not wired up. Gates the LIVE site only (microiteration_optimization6
# ~12328).  (microiteration_optimization5 carried a byte-identical block; that
# routine was REMOVED 2026-09-04, so this is now the only copy.)
_QED_DISABLE_QN = int(os.environ.get("QED_DISABLE_QN", "0"))
#
#   QED_USE_MINRES : solve Hx=-g with preconditioned MINRES  (default 1 = ON)
#
# Unlike the knobs above this one is NOT default-no-op: minres_solve replaces
# linear_equation_solve/LinearRMSolver at all four "gradient is small, use
# Newton step" / hard_case==2 call sites (internal, microiteration_optimization6
# ~10586,10634 and the live microiteration_optimization6 ~12263,12305).  See
# minres_solve's docstring for why: LinearRMSolver solves the normal equations
# S^T S a = -S^T g, squaring the condition number of a Hessian that is
# near-singular precisely when this branch fires.  Set to 0 to restore the old
# path for an A/B comparison; the scipy-minres fallback that follows a
# non-converged solve is untouched either way.
_QED_USE_MINRES = int(os.environ.get("QED_USE_MINRES", "1"))
#
#   QED_PROFILE_INTERMEDIATES : per-section timing of build_intermediates
#                               (default 0 = off, no-op)
#
# build_intermediates is the single largest orbital-side cost in the c10h8
# profile (64.8 s / 102 calls) and is rebuilt every MICROiteration. This splits
# it into its ~17 constituent contractions so the density-INDEPENDENT parts
# (notably `self.L = 4K - K^T - J`, which only changes when J/K are retransformed
# at a macroiteration boundary) can be separated from the ones that genuinely
# must track self.D_tu_avg. See the aggregation one-liner in the function body.
_QED_PROFILE_INTERMEDIATES = int(os.environ.get("QED_PROFILE_INTERMEDIATES", "0"))
#
#   QED_VERIFY_L_CACHE : rebuild L on every cache HIT and compare  (default 0)
#
# Turns build_intermediates' L/temp1 cache into a checked cache: any missed
# invalidation raises instead of silently producing a wrong orbital Hessian.
# Costs the very savings the cache buys, so it is a correctness harness, not a
# production setting -- run it once after touching anything that writes J/K.
_QED_VERIFY_L_CACHE = int(os.environ.get("QED_VERIFY_L_CACHE", "0"))
#
#   QED_SOLVER_STATS : count trust-region solver work                (default 0)
#
# Records, per orbital-optimization solve: which solver ran, how many outer
# iterations it took, how many Hessian-vector products (orbital_sigma3 / mv2)
# and dense bordered eigensolves (projection_step2) it consumed, and which
# termination branch it exited through.  Answers "are we paying for solver
# refinement or for building the Davidson subspace?" -- we take ~15.5 Hessian
# applications per orbital step where Molpro takes ~6.9, and the two have very
# different fixes (looser exit tolerance vs. a better preconditioner).
# Prints an aggregate table at the end of the run.  Zero effect when unset.
_QED_SOLVER_STATS = int(os.environ.get("QED_SOLVER_STATS", "0"))
#
#   QED_MINRES_TRACE : print a per-iteration MINRES residual        (default 1)
#
# The scipy-MINRES fallback in the small-gradient Newton branch reports its own
# residual from a callback.  scipy does not hand the callback a residual, only
# xk, so producing that line costs a FULL EXTRA HESSIAN-VECTOR PRODUCT per
# MINRES iteration -- on top of the one the algorithm itself needs.  Set to 0 to
# keep the iteration count but skip the residual line and its matvec.  Default 1
# preserves the existing output exactly.
_QED_MINRES_TRACE = int(os.environ.get("QED_MINRES_TRACE", "1"))
#
#   QED_GLTR_TOL_SCALE : multiplier on the GLTR exit tolerance     (default 1.0)
#
# solve_gltr_trust_region is called with tol=1e-7 at all three live sites; it
# exits when the Lanczos residual be*|y[-1]| drops below tol.  GLTR is 30.2% of
# c10h8's Hessian-vector products (770 / 2549) and the QED_SOLVER_STATS trace
# says 52 of 68 GLTR solves terminate "Interior (Newton)" -- i.e. strictly
# inside the trust region, where tol controls linear-solve accuracy only, not
# whether the step is on the right boundary point.  This knob scales tol so
# that claim can be A/B'd on a real system; 1.0 reproduces current behavior
# exactly.
_QED_GLTR_TOL_SCALE = float(os.environ.get("QED_GLTR_TOL_SCALE", "1.0"))
#
#   QED_NEWTON_MAXITER : iteration cap on the Newton-step solve     (default 20)
#
# Both live "gradient is small, use Newton step" sites (microiteration_
# optimization6 ~12690 and the hard_case==2 site ~12641) call minres_solve with
# max_iter=20; failing to reach conv_thresh there hands the system to scipy
# MINRES, which restarts from x0=0, runs UNPRECONDITIONED, and stops on a
# relative test rather than the absolute 1e-6 the caller wanted -- i.e. the
# fallback is strictly more expensive than continuing the preconditioned solve.
# c10h8 never exceeds 12 iterations so this knob is a no-op there; it exists to
# test whether raising the cap lets MgH+ (56 of 147 solves fall back) converge
# without scipy at all.
_QED_NEWTON_MAXITER = int(os.environ.get("QED_NEWTON_MAXITER", "20"))
#
#   QED_NEWTON_FORCING : make minres_solve's exit tolerance RELATIVE to ||g||
#                        (default 0 = OFF, byte-for-byte no-op)
#
# THE DEFECT THIS ADDRESSES.  minres_solve is called with conv_thresh=1e-6 and
# tests it against ||Hx + g||, an ABSOLUTE quantity.  The residual at x = 0 is
# exactly ||g||.  Measured on MgH+ CAS(8,12) lam=0 (64 solves at the live site):
# ||g|| is min 1.46e-07, MEDIAN 2.14e-07, max 7.39e-04, so in 43 of 64 solves
# the threshold is LARGER THAN THE RESIDUAL WE START WITH and the convergence
# test demands nothing at all (relative accuracy demanded: median 4.7, i.e.
# 470%).  Inexact-Newton theory wants the forcing term to SHRINK as ||g|| does
# (Eisenstat & Walker 1996) to keep the outer iteration superlinear; ours grows.
# What stops those solves today is not the accuracy rule but the conservative
# bound_factor (~10x) and the max_iter cap -- i.e. an accident.
# NOT an "x = 0 returned" bug: the only early exit is for an exactly-zero rhs.
#   > 0  : conv_thresh = QED_NEWTON_FORCING * ||g||   (eta = 1e-2 is a
#          reasonable first choice: 1% relative residual)
#   < 0  : Eisenstat-Walker choice 1 constant form,
#          conv_thresh = min(0.1, sqrt(||g||)) * ||g||
#   = 0  : OFF.  conv_thresh is whatever the caller passed (1e-6).
# NOTE the interaction with QED_NEWTON_MAXITER: a genuinely relative tolerance
# needs more iterations, so a forcing run that leaves the cap at 20 measures the
# cap, not the tolerance.  Raise both together; QED_NEWTON_MAXITER=100 alone is
# a measured no-op on this case (33 -> 32), so it is a clean control.
_QED_NEWTON_FORCING = float(os.environ.get("QED_NEWTON_FORCING", "0"))
#
#   QED_NEWTON_RANDOM_START : start minres_solve from a random x0 when ||g|| is
#                             small                  (default 0 = OFF, no-op)
#
# DIAGNOSTIC, not a proposed default.  linear_equation_solve (the
# QED_USE_MINRES=0 path) opens with: if ||g|| < 1e-3, draw a random unit vector
# trial_0 and build the first search direction from H*trial_0 + g instead of
# from g.  The author's stated intent was to BREAK STAGNATION; the small-||g||
# interaction was not intended, and since ||g|| here is ~2e-07 the branch fires
# on essentially every call.  This knob asks whether that random direction is
# what makes the MINRES=0 path cheaper on MgH+ (12 vs 33 macroiterations).
# DEVIATION, FLAGGED RATHER THAN HIDDEN: the two are not the same operation.
# LinearRMSolver never puts trial_0 into its solution -- it only uses it to
# generate the first trial direction, and returns the least-squares minimizer
# over the subspace it then accumulates.  MINRES has no equivalent seam: the
# only consistent way to start it off g is a genuine warm start x0, for which
# r0 = b - H*x0.  So this sets x0 = QED_NEWTON_RANDOM_START * (random unit
# vector), costing ONE extra mv2 to form r0.  A value of 1.0 matches
# LinearRMSolver's unit magnitude, which is ~100x the typical solution norm
# (~0.008) and therefore a deliberately POOR warm start -- that is the point of
# measuring rather than assuming.  Consumes the global RNG, like the routine it
# imitates and like solve_gltr_trust_region's noise draw.
_QED_NEWTON_RANDOM_START = float(os.environ.get("QED_NEWTON_RANDOM_START", "0"))
#
#   QED_NEWTON_DUMP : directory to dump the Newton subproblems (H, g, diag) to
#                     (default "" = OFF, no-op)
#   QED_NEWTON_DUMP_MAX    : stop after this many dumps      (default 12)
#   QED_NEWTON_DUMP_STRIDE : dump every Nth solve            (default 1)
#
# WHY.  The question "can the Newton problem always be converged by something
# faster than scipy MINRES?" cannot be answered from macroiteration counts:
# those are chaotic (the count moves 5x under bit-level FP noise) and they
# confound the linear solver with the outer optimizer.  The clean measurement is
# OFFLINE -- take the actual (H, g) pairs this branch sees and benchmark
# candidate solvers on them in isolation, counting Hessian-vector products to a
# RELATIVE residual target.  The reduced Hessian is only index_map_size square
# (150 for MgH+ CAS(8,12)), so a dense dump is cheap and an EXACT
# eigendecomposition is free -- which also settles whether these systems are
# consistent at all (if g has a component in the near-null space of H, no
# Krylov method converges and MINRES-QLP, not MINRES, is the right tool).
# COST: index_map_size extra Hessian applications per dump, via the same
# batched orbital_sigma3 path _exact_dense_trust_region_step uses.  Hence the
# MAX cap: 12 dumps x 150 = 1800 mv2, a few percent of a run.
#
#   QED_NEWTON_SCIPY : solve the Newton system with scipy's UNPRECONDITIONED
#                      MINRES at BOTH live call sites, never touching
#                      minres_solve or linear_equation_solve
#                      (value = rtol; default 0 = OFF, no-op)
#
# WHY THIS LEG EXISTS.  scipy MINRES is the REFERENCE the whole exercise is
# measured against -- item 208(B), "can I always converge the Newton problem
# with something FASTER THAN SCIPY MINRES?" -- and yet it has never been run as
# the primary solver.  It appears today only as the fallback after one of the
# other two fails, so its end-to-end behaviour is unmeasured.  (The `rstart` leg
# of item 210 approximated it by accident, 38 of 38 solves falling back, and
# came in at raw 20 against 33 and 12 -- but with a random x0 corrupting it, so
# it does not count.)
# WHAT IT ISOLATES.  With QED_NEWTON_FORCING the two Krylov solvers differ in
# exactly TWO things: the stopping rule and the PRECONDITIONER.  Forcing fixes
# the stopping rule.  So running scipy (unpreconditioned) at rtol=1e-6 against
# minres_solve (|diag|-preconditioned) at the SAME relative tolerance isolates
# the preconditioner itself -- two iterates that both satisfy ||Hx+g|| <= tol
# still differ, and on a near-singular H they differ precisely in the near-null
# components that this whole thread is about.
# CONTRACT: same signature and (solution, converged) return as its two
# siblings, so the call sites dispatch without any other change.  `conv_thresh`
# is IGNORED -- scipy's rtol is relative to ||b||, which is the point.  maxiter
# is left at scipy's default (5n), matching how the existing fallback calls it.
# Gates the two LIVE sites in microiteration_optimization6 only; the
# byte-identical pair in microiteration_optimization5 was REMOVED with it, the
# same convention QED_DISABLE_QN uses.
_QED_NEWTON_SCIPY = float(os.environ.get("QED_NEWTON_SCIPY", "0"))
_QED_NEWTON_DUMP = os.environ.get("QED_NEWTON_DUMP", "")
#   QED_LSTRS_HARD_DUMP : directory to dump the LSTRS subproblem at the moment
#                         the beta-bisection ends with BOTH roots still
#                         unnormalizable (default "" = off, exact no-op).
#
# WHY.  microiteration_optimization6 :16776 reaches a state the code does not
# implement -- both roots hard -- and calls a bare exit().  The question that
# decides how to fix it is whether the near-degenerate pair the Davidson keeps
# returning is a REAL property of the bordered matrix or an artefact of the
# iterative solver restarting from a frozen search space.  That is only
# answerable against a dense eigendecomposition, and the reduced space here is
# tiny (30 for LiH CAS(4,6)), so an exact one costs 30 Hessian applications.
_QED_LSTRS_HARD_DUMP = os.environ.get("QED_LSTRS_HARD_DUMP", "")
#
#   QED_TRS_HARD_FALLBACK : take a real step when the LSTRS beta-bisection ends
#                           with BOTH roots unnormalizable, instead of the bare
#                           exit() that site used to call    (default 1 = ON)
#
# WHY THIS IS DEFAULT ON, unlike every other knob here.  The state it handles is
# a GENUINE LSTRS hard case, verified by dense eigendecomposition of a dumped
# subproblem (QED_LSTRS_HARD_DUMP): the orbital Hessian's two lowest eigenvalues
# were degenerate to 5.7e-17 and the gradient was orthogonal to BOTH to machine
# zero (|u_i^T g|/||g|| = 5.9e-15 and 1.5e-16).  The old code's response was
# `exit()` -- SystemExit(None), i.e. the process dies MID-OPTIMISATION WITH
# STATUS 0 and no traceback, which any runner scoring a return code reads as
# success.  There is no configuration in which that is preferable, so the
# fallback is on and the knob exists only to reproduce the old behaviour.
#
# NOT A BEHAVIOUR CHANGE FOR ANY HEALTHY RUN: it is reachable only where the old
# code called exit(), so every run that completes today is untouched.
#
# WHAT IT DOES.  Dense More-Sorensen (_more_sorensen_dense, via
# _exact_dense_trust_region_step) when the reduced space is small enough --
# that solver has an EXPLICIT hard-case branch (pseudo-inverse solution plus a
# multiple of a lowest eigenvector, sign chosen to minimise the model), which is
# exactly the step the bisection cannot reach.  On the dumped subproblem it
# gives model -2.7215e-04 against -2.6587e-04 for the interior pseudo-inverse
# point alone.  Above the size cap it falls back to GLTR, which cannot fail but
# has no hard-case branch, so the step is a descent step rather than the
# optimum -- reported, not hidden.
_QED_TRS_HARD_FALLBACK = int(os.environ.get("QED_TRS_HARD_FALLBACK", "1"))
# QED_LSTRS_MAX_ITER (2026-09-26): the LSTRS beta-bisection's iteration cap, a literal 50 until now (default unchanged).  The
# cap exit used to leave `step` unbound (lstrs_trust_region.py); it now takes the exact trust-region step.  A knob so that
# tests/test_trs_hard_case.py can force the cap path (value 1) on a small deck.
_QED_LSTRS_MAX_ITER = int(os.environ.get("QED_LSTRS_MAX_ITER", "50"))
# Cost of the dense route is n = index_map_size Hessian-vector products, so the
# cap is a wall-time guard, not a correctness one.
_QED_TRS_FALLBACK_DENSE_MAX = int(
    os.environ.get("QED_TRS_FALLBACK_DENSE_MAX", "4096"))
#
#   QED_LSTRS_HARD_PATIENCE : stop the beta-bisection after this many
#                             CONSECUTIVE both-roots-hard iterations and take
#                             the fallback above           (default 25)
#
# The bisection cannot escape the hard case by bisecting: on the dumped case the
# Davidson returned BIT-IDENTICAL eigenvalues for the last ten iterations while
# alpha kept halving, so the loop only ended when the bracket collapsed after 40
# iterations.  A healthy subproblem leaves this state in a handful of iterations
# (the same case with a 1e-14-different mean field escaped on iteration 18), so
# 25 is slack.  0 disables the cap and restores the bracket-collapse-only exit.
_QED_LSTRS_HARD_PATIENCE = int(os.environ.get("QED_LSTRS_HARD_PATIENCE", "25"))
#
#   QED_INTERNAL_EXACT_TRS : in internal_optimization3, solve the trust-region
#                            subproblem by direct eigendecomposition instead of
#                            the LSTRS beta-bisection        (default 0 = off)
#
# The internal site is the one place where this is free: `hessian_tilde_ai` is
# ALREADY a dense (n_act_orb*n_in_a) matrix, and projection_step2 builds and
# diagonalises the full bordered matrix on EVERY bisection step.  So the loop
# pays many dense eigendecompositions to approach a solution that one call to
# _more_sorensen_dense returns exactly, hard case included, with no
# normalizability question and no iteration count.
#
# DEFAULT OFF because it changes the steps taken (energies then agree to the
# convergence threshold, not to machine precision) -- not because it is
# suspect.  It is strictly more robust than what it replaces.
_QED_INTERNAL_EXACT_TRS = int(os.environ.get("QED_INTERNAL_EXACT_TRS", "0"))
_QED_NEWTON_DUMP_MAX = int(os.environ.get("QED_NEWTON_DUMP_MAX", "12"))
_QED_NEWTON_DUMP_STRIDE = int(os.environ.get("QED_NEWTON_DUMP_STRIDE", "1"))
#
#   QED_LSTRS_NORM_TOL : LSTRS boundary tolerance |(||s||-D)/D|   (default 1e-3)
#
# The LSTRS alpha-bisection keeps iterating until the step norm matches the
# trust radius to this relative accuracy.  Gates the two LIVE sites --
# internal_optimization3 (~7912, dense/bordered) and microiteration_
# optimization6 (~12415, matrix-free Davidson) -- not the micro_opt5 or
# ah_orbital_optimization twins (that routine was REMOVED 2026-09-04; it never
# fired in the measured runs).
# Worth sweeping because the trust radius it is being matched against is
# itself only updated in coarse 0.7x / 1.2x / 0.5x jumps, so 0.1% accuracy
# against a quantity known to ~30% is not obviously earning its iterations:
# c10h8 spends 3.67 LSTRS iterations/solve at ~23 matvecs each (771 total,
# 30% of the run), so one saved iteration per solve is ~200 matvecs.
_QED_LSTRS_NORM_TOL = float(os.environ.get("QED_LSTRS_NORM_TOL", "1e-3"))
#
#   QED_STEP_CONTROL : step-norm-aware trust-radius update      (default 0 = off)
#
# Fixes two independent trust-region defects, both measured on c10h8:
#
# (a) Delta GREW on steps that Delta never constrained.  step_control multiplies
#     by 1.2 whenever ratio > 0.75, but 43 of the 46 growth events in the run
#     were on GLTR "Interior (Newton)" steps -- a good ratio there says the
#     model held over a SHORT step, not that a longer one would be safe.  Delta
#     therefore ratcheted to the 0.75 cap during easy steps and was far too
#     large when a hard one arrived (boundary reject rate 46.7% vs 11.3%
#     interior).  With the knob on, growth requires ||s|| >= 0.8*Delta.
#
# (b) The REJECT branch shrank Delta relative to ITSELF (0.5*Delta), ignoring
#     the rejected step.  For an interior step that changes nothing -- the
#     solver returns the identical vector -- so the same Newton step was
#     computed and rejected three times in a row before Delta finally fell
#     below its norm (0.5 -> 0.25 -> 0.125 -> 0.0625, step 0.079 throughout).
#     With the knob on, Delta_new = 0.5*||s_rejected||, the standard rule
#     (Nocedal & Wright Alg. 4.1) chosen precisely because s may be interior.
#     Since ||s|| <= Delta always, this is never weaker than the old rule and
#     is IDENTICAL for boundary steps -- only interior rejections change.
#
# Deliberately NOT changed: the ratio<=0.25 shrink (0.7*Delta) on ACCEPTED
# steps, which has the same step-norm-blindness but was not implicated in any
# measured rejection cascade.
_QED_STEP_CONTROL = int(os.environ.get("QED_STEP_CONTROL", "0"))
#
#   QED_LSTRS_EPS_HC : quasi-optimality tolerance for x_tilde   (default 1e-6)
#
# The LSTRS quasi-optimal exit (hard_case=3) accepts x_tilde when
#   (mu_2 - mu_1) * tau2^2 * (1 + Delta^2)  <  -2 * eta * psi_tilde,
# eta = eps_hc/(1 - eps_hc).  eps_hc IS the relative suboptimality you are
# willing to accept in the model decrease, so 1e-6 demands x_tilde be within
# ~one part in a million of the exact optimum -- and it shows: on c10h8 the
# x_tilde construction runs on 54 of 73 LSTRS iterations and the test accepts
# it ZERO times (terminations are 9 hard_case=0 and 19 hard_case=2, no
# hard_case=3 at all).  Raising this is the direct lever on "exit earlier with
# a provably near-optimal step" -- unlike QED_LSTRS_NORM_TOL, which was
# measured NOT to be the binding exit condition.  Gates the two LIVE sites,
# internal_optimization3 (~7974) and microiteration_optimization6 (~12486).
_QED_LSTRS_EPS_HC = float(os.environ.get("QED_LSTRS_EPS_HC", "1e-6"))
#
#   QED_TRUST_PERSIST : keep the trust radius across microiterations  (default 0)
#
# microiteration_optimization6 resets trust_radius = 0.5 at the top of EVERY
# microiteration (~11689), discarding whatever the previous one learned.  With
# N_orbital_optimization_steps normally 1, the inner accept/reject loop takes a
# single accepted step and exits, so across microiterations this is not a trust
# region at all -- it is "try 0.5, halve until the energy goes down".  Measured
# on c10h8: 13 of 14 rejections are on orbital step 0 of a microiteration, every
# cascade starts at Delta = 0.5, and the accepted radii are 0.031-0.25.
# NOTE the open question this knob is meant to answer: the rejections cluster at
# MICROITERATION 1, i.e. straight after a macroiteration's CI solve and integral
# transformation, so the carried radius is one that was learned on the PREVIOUS
# macroiteration's surface.  Whether it still predicts anything there is exactly
# what has to be measured -- do not assume it does.
_QED_TRUST_PERSIST = int(os.environ.get("QED_TRUST_PERSIST", "0"))
#
#   QED_TR_MAX / QED_TR_INIT : the trust-region CAP and the per-microiteration
# STARTING radius.  Defaults 0.75 / 0.5 reproduce current behaviour exactly.
#
# WHY THESE EXIST.  On MgH+ CAS(8,12) the trust region is INERT -- it is not
# adapting anything, it is a hard-coded step-length cap.  Measured on the
# base arm of the adaptive_cas812 wave (base_s3/s4/s7 logs):
#   - ZERO rejections in the entire run (grep -ic reject = 0 in all three);
#   - the radius walks 0.5 -> 0.6 -> 0.72 -> 0.75 (the 1.2x growth of
#     step_control) and then PINS at the 0.75 cap -- 80 of ~122 printed radii
#     are exactly 0.75 in s3;
#   - it is then RESET to 0.5 at the top of the next microiteration and
#     re-climbs the identical ladder.
# Combine that with the conditioning report's 13/15 subproblems terminating on
# the BOUNDARY and the conclusion is forced: THE STEP LENGTH IS SET BY THE
# CONSTANT 0.75, NOT BY THE MODEL.  ratio > 0.75 at every step, so the
# algorithm requests a larger region at every step and the cap denies it.
#
# THE ABOVE DIAGNOSIS IS CONFIRMED.  What follows is a record of the efficacy
# wave that tested it as a FIX, and FALSIFIED it.  Keep both halves: the
# measurement is sound and worth not re-deriving; the inference from it was not.
#
# RESULT (trmax_cas812, 2026-08-11, MgH+ CAS(8,12), 14 seeds x
# {base, QED_TR_MAX=1.0, QED_TR_MAX=3.0}, 42/42 converged, worst |dE| 5.7e-9;
# the base arm reproduced the adaptive_cas812 base EXACTLY per seed):
#         arm  | MAX  p90  | mean    sd   min | matvecs
#         base |  21  19.0 | 14.86  4.28    8 |   -
#         cap1 |  20  18.7 | 15.64  3.15    9 | +3.6% (p=0.73, 6/14 better)
#         cap3 |  20  18.7 | 15.64  3.15    9 | identical to cap1
# Macro-count p = 0.59.  The pre-registered rule was "wins only if MAX and p90
# come down"; they moved by 1 and 0.3, which over n=14 is a RE-ROLL, not a win.
# It is the same signature as the six earlier mechanisms: sd falls ONLY by
# lifting the MINIMUM (8 -> 9) while the upper tail sits still and the mean
# RISES.  No mechanism has ever moved the upper tail.
#
# THE MECHANISM DID FIRE -- that is what makes this informative rather than
# inconclusive.  Raising the cap woke the trust region up exactly as intended:
# rejections per run went 0.1 -> 0.8, and the radius ADAPTS instead of pinning
# (cap3 climbs 0.864/1.037/1.244/1.493/1.792/2.150/2.580).  The outer loop did
# not care.  So the arbitrary throttle was REAL, and was NOT what limits outer
# progress.  Outer progress is limited by MODEL ERROR (QED_ADAPTIVE_TOL's real
# finding), not by step quality (0/5, QED_EXACT_TR) and not by step LENGTH.
#
# THE COUNTER-ARGUMENT IS SETTLED: 0.75 is NOT load-bearing.  cap1 and cap3
# differ in radius BOOKKEEPING but their energy trajectories are BIT-IDENTICAL
# on every seed spot-checked -- above ~1.0 the steps go INTERIOR and the cap
# stops binding, exactly as on CAS(6,6).  A wider cap sweep would measure
# nothing.  CAVEAT FOR ANY FUTURE READER OF THE LOGS: the largest radius
# REACHED is not evidence the knob changed the step -- a radius can grow
# without the step ever using it, which is precisely what cap3 does.
#
# Both knobs stay at their default-no-op values (knob-off verified bit-exact,
# CAS(6,6) seed 3 = macro 10 / mv 10889 / E -199.53179254052137).  The CAS(6,6)
# smoke test had shown cap >= 1.0 giving macro 8 vs 10; it did NOT transfer.
# CAS(8,12) is the discriminator -- never score a candidate on CAS(6,6) alone.
_QED_TR_MAX = float(os.environ.get("QED_TR_MAX", "0.75"))
_QED_TR_INIT = float(os.environ.get("QED_TR_INIT", "0.5"))
#
# QED_TRUE_ENERGY_RATIO: judge an orbital step by the TRUE energy E(U) instead
# of by the second-order model E(T).  Default 0/OFF = byte-for-byte unchanged.
#
# THE DEFECT THIS ADDRESSES.  microiteration_optimization6 accepts a step on
#     energy_change = zero_energy + microiteration_exact_energy(U3,A,G) - current_energy
# (:16158) and then scores the trust region with
#     ratio = energy_change / predicted_energy2                        (:16233)
# Both sides of that ratio are the SAME quadratic model E(T) = 2A.T + T.G.T --
# they differ only in higher-order terms of the U parameterisation -- so the
# ratio is ~1.0002 at every step (item 320), step_control never shrinks, and no
# step is ever rejected.  But E(T) is a QUADRATIC FORM: for indefinite G it is
# unbounded below, so the microiteration rides it down without limit and the
# CI at the resulting orbitals comes back HIGHER.  Measured on the developer's
# ethylene conical-intersection case (aug-cc-pVDZ, CAS(4,7), lambda_x = 0.4,
# `jobs/qed_ci/ethylene/cavity/aug_cc_pvdz/040/test/c2h4_040.py`), where the
# model reaches -294.54 Eh while the true energy is -72.69:
#     micro  1: model -77.176642   true -77.081193   gap  -0.0954  rho_true -1.559
#     micro  4: model -77.281003   true -74.129989   gap  -3.1510  rho_true -8.777
#     micro 20: model -91.315717   true -72.248581   gap -19.0671  rho_true  0.116
#     micro 40: model -212.153479  true -63.382798   gap -148.77   rho_true  0.011
# rho_true is NEGATIVE from the first microiteration -- a textbook trust region
# would have rejected that step and shrunk Delta.  Molpro has the same E(T)
# below E(U) gap at the same magnitude on Cr2 (-4.23 vs our -4.71), but it
# computes and prints ENERGY(VAR) against ENERGY(PROJ) every macroiteration and
# drives the gap monotonically to zero; ours re-opens.  So the model is not the
# bug -- not measuring it is.
#
# WHAT IT COSTS, AND THE CHEAP ROUTE THAT DOES NOT WORK.  E(U) needs the ERIs
# genuinely retransformed at the rotated orbitals -- one
# c_full_transformation_macroiteration out of self.twoeint per evaluation.
# Measured mean cost per call, against the run it is added to:
#     c2h4   0.050 s   (176 candidate steps; the run wastes far more than that)
#     Cr2    0.140 s   (271 steps = 38 s on 1695 s, +2%)
#     c10h8  2.334 s   (45 steps = 105 s on 55 s, +189%)
# c10h8 is the case where the knob should stay OFF: its model is accurate, so
# there is nothing to catch.  The pathological cases are the cheap ones.
#
# THE CHEAP ROUTE IS A TRAP, AND IT COST THIS IMPLEMENTATION A REVISION.
# microiteration_ci_integrals_transform (:12072) is 12-46x cheaper and already
# runs once per accepted step, so it looks like a free E(U).  IT IS NOT A
# TRANSFORMATION.  It updates the active integrals from self.J[act,act,:,:] and
# self.fock_core with terms explicitly first and second order in T = U - I, so
# an energy assembled from its output IS the second-order model, by another
# route.  Measured on c2h4 when it was tried: `true energy change` agreed with
# `model energy change` to 12 DIGITS at every step while the macroiteration
# energy still rose 8.3 Eh.  That is two second-order expressions agreeing with
# each other, not a PROJ-VAR gap.  If a future reader is tempted by the same
# shortcut: the tell is a gap of ~1e-13 where the trajectory says it should be
# Eh-sized.
#
# MODES
#   0  off (default).  Byte-for-byte the original.
#   1  MEASURE AND STEER.  Compute E(U) at the trial point, print the
#      PROJ/VAR gap and rho_true, and hand rho_true to step_control instead of
#      the model-over-model ratio.  The ACCEPTANCE test is left alone, so the
#      trajectory changes only through the trust radius.
#   2  ENFORCE.  As 1, and additionally accept a step only when the TRUE
#      energy decreases (by more than QED_TRUE_ENERGY_TOL).  This is the
#      textbook trust-region construction and is what makes the method
#      globally convergent on the fixed-RDM surface regardless of how bad the
#      model is: the step is bounded by Delta and Delta shrinks until the true
#      energy falls.
#
# Because the CI re-solve at fixed orbitals is variational over the averaged
# subspace (equal weights -- Ky Fan; item 285), mode 2 gives
#     E_CI(new) <= E_U(new) <= E_CI(old)
# i.e. the MACROiteration energy becomes monotone as well.  The c2h4
# oscillation is exactly the failure of the first inequality, which nothing in
# the shipping path currently enforces.
_QED_TRUE_ENERGY_RATIO = int(os.environ.get("QED_TRUE_ENERGY_RATIO", "0"))
# Acceptance tolerance on the TRUE energy change in mode 2.  0.0 means "must
# strictly decrease".  A small positive value permits a step that is flat to
# within numerical noise.
_QED_TRUE_ENERGY_TOL = float(os.environ.get("QED_TRUE_ENERGY_TOL", "0.0"))
# Mode 2 retry bound.  The shipping reject path (`Reject step, restart`) halves
# the radius and loops with NO bound on consecutive rejections.  That is benign
# under the model test, where acceptance is near-certain, but the TRUE-energy
# test can legitimately reject every step -- at a stationary point of E(U) no
# step lowers it -- and would then spin with Delta -> 0.  On hitting either
# bound the microiteration gives up rather than looping: no step is applied, the
# CI is re-solved, and the next microiteration builds a fresh model.
_QED_TRUE_ENERGY_MAX_REJECT = int(os.environ.get("QED_TRUE_ENERGY_MAX_REJECT", "8"))
_QED_TRUE_ENERGY_MIN_RADIUS = float(
    os.environ.get("QED_TRUE_ENERGY_MIN_RADIUS", "1e-6")
)
#
# QED_TRUE_ENERGY_REEXPAND: on an ACCEPTED step, MOVE THE MODEL'S EXPANSION
# POINT to the new orbitals instead of accumulating T against a model expanded
# once at macroiteration entry.  Requires QED_TRUE_ENERGY_RATIO >= 1 (it reuses
# that mode's transform).  Default 0 = the shipping accumulate behaviour.
#
# WHY.  Inside microiteration_optimization6, self.J / self.K / self.H_spatial2 /
# self.d_cmo are NEVER rebound (every such assignment in the function is
# commented out), self.U2 accumulates for the whole macroiteration, and
# U3 = U2 . U_delta is the TOTAL displacement from macro entry.  build_gradient
# and build_hessian_diagonal re-EVALUATE the fixed A, G at that growing T; they
# do not rebuild them.  So the quadratic stays expanded at the macro-entry
# orbitals while the iterate walks away from it, and eventually no step from
# that stale model descends at any radius -- which is exactly the give-up
# cascade QED_TRUE_ENERGY_MAX_REJECT exists to escape (11 of them on c2h4).
#
# THIS IS THE DEVELOPER'S ah_orbital_optimization DESIGN.  That routine (REMOVED
# 2026-09-04, see git history) set self.U2 = eye once and never wrote it again, so every
# build_gradient(self.U2, ...) / mv2(self.U2, ...) is called at U = I, T = 0;
# its accept branch fully retransforms J/K/h1/d_cmo and rebuilds A, G from the
# NEW integrals, i.e. every accepted step is effectively a new macroiteration.
# It cannot be revived directly: its subproblem solver
# Davidson_augmented_hessian_solve3 has been deleted, and the surviving
# Davidson_augmented_hessian_solve6 has a different parameterization
# (G_blocks/alpha/alpha_range/trust_radius vs dense G1/beta), so substituting it
# would confound the very comparison this knob exists to make.  This knob tests
# the SAME design question inside the live path with the solver held identical.
#
# IT IS ALSO NEARLY FREE.  QED_TRUE_ENERGY_RATIO already does a full
# c_full_transformation_macroiteration at every candidate step and then keeps
# only the scalar energy.  On an accepted step this installs the J/K/h1/d_cmo it
# already produced, folds the rotation into U_total, resets U2/U3 to the
# identity and rebuilds A, G -- one build_intermediates (0.0098 s on c2h4) on
# top of a transform that was already paid for.
#
# QN IS FORCED OFF when this is on: the quasi-Newton branch caches U_zero /
# A_tilde_zero / G_blocks_zero / reduced_hessian_diagonal_zero in the OLD basis
# at qn_count == 1, and re-expansion invalidates all of them.  Disentangling
# that is a separate job; for now the knob refuses to combine with QN rather
# than silently using stale history.
_QED_TRUE_ENERGY_REEXPAND = int(os.environ.get("QED_TRUE_ENERGY_REEXPAND", "0"))
if _QED_TRUE_ENERGY_REEXPAND and _QED_TRUE_ENERGY_RATIO < 2:
    # It installs the transform QED_TRUE_ENERGY_RATIO produces, and only
    # on a step accepted on the TRUE energy.  Without mode 2 there is no
    # transform to install, so promote rather than fail at the call site.
    print("QED_TRUE_ENERGY_REEXPAND=1 requires QED_TRUE_ENERGY_RATIO>=2;"
          " promoting QED_TRUE_ENERGY_RATIO to 2", flush=True)
    _QED_TRUE_ENERGY_RATIO = 2
#
# QED_TRUE_ENERGY_ADAPTIVE: engage the true-energy machinery (BOTH the
# acceptance test and, if enabled, the re-expansion) ONLY while the trial
# rotation has walked far enough from the model's expansion point that the
# model can no longer be trusted.  Default 0 = the unconditional behaviour of
# QED_TRUE_ENERGY_RATIO / _REEXPAND, so this knob is an exact no-op when off.
#
# WHY IT EXISTS -- ITEM 330.  Measured on c10h8 CAS(10,10), a HEALTHY system
# whose shipping control reproduces Molpro to 1.1e-9:
#     control     10 macros  152.0 s   E = -383.5044493793   (= Molpro)
#     mode2        9 macros 1213.9 s   E = -383.4973645803   (+7.085 mEh)
#     reexpand     6 macros  259.5 s   E = -383.4973645891   (+7.085 mEh)
# Both knob legs CONVERGE, monotonically, to a DIFFERENT AND WORSE stationary
# point.  It is NOT premature stopping: reexpand reaches a SMALLER gradient
# norm (2.53e-6) than the control (4.45e-6) at the HIGHER energy.  And since
# mode2 carries no re-expansion at all yet lands on the same wrong solution,
# the cause is THE ACCEPTANCE TEST, not the re-expansion.
#
# THE MECHANISM (and the developer's reading of it).  Werner-Meyer-Knowles is
# valued for a LARGE RADIUS OF CONVERGENCE: the model-based test will accept an
# adventurous step in T, including one that raises the true energy, and that
# wandering is HOW a better basin gets reached.  Enforcing strict monotone
# descent in E(U) makes the optimiser greedy and parks it in the nearest basin.
# So "monotone = good" is FALSE in general -- it is good exactly where the model
# is untrustworthy (c2h4 at lambda 0.4, h4 at 24000 A, Cr2's +2.38 Eh macro 1)
# and harmful where the model is sound (c10h8).
#
# THE TRIGGER.  Kreplin's algorithm (item 264A) rebuilds the exact Hessian when
# k == 0, or |D^k - D^(k-1)| > eps_c = 0.025, or the energy rose -- i.e. Molpro
# re-expands CONDITIONALLY, sitting between our never-re-expand shipping path
# and _REEXPAND's always.  The quantity that actually makes OUR model stale is
# the displacement of the trial rotation from the point the model was expanded
# at, so the trigger here is ||U2_trial - I||_F > eps_c.  It costs one Frobenius
# norm of an nmo x nmo matrix and, crucially, is computable WITHOUT the E(U)
# transform -- so when it does not fire the expensive transform is never run and
# the shipping path is followed exactly.
#
# *** THE DEFAULT 0.025 IS A PLACEHOLDER, NOT A CALIBRATED VALUE, AND MUST NOT
# BE QUOTED AS KREPLIN'S. ***  The developer raised this and is right: 0.025 is
# an EMPIRICAL constant Kreplin fitted for |D^k - D^(k-1)|, a DENSITY change.
# The trigger here is a ROTATION norm -- a different quantity, different units,
# different scale -- so the number carries no authority once moved across, and
# an earlier draft of this comment that called it "Kreplin's own 0.025" was
# wrong.  Two honest routes, in order:
#   (1) MEASURE FIRST.  true_energy_should_engage prints ||U2_trial - I||_F on
#       every call.  Run each system with QED_TRUE_ENERGY_EPS_C=0 -- which
#       engages unconditionally and so reproduces the plain knob exactly -- and
#       read the distribution.  If c10h8's displacements do NOT separate from
#       c2h4 / h4 / Cr2's, NO single threshold can work and this trigger is the
#       wrong quantity; that is a decisive, cheap refutation.
#   (2) If they do separate, SWEEP eps_c across the gap and pick from the
#       measurement, scoring c10h8's ENERGY as well as its cost (item 330E).
# A third option if (1) fails: use Kreplin's ACTUAL quantity, the change in the
# one-particle density, for which 0.025 IS calibrated -- computable without the
# 4-index transform, so the cost argument survives.
#
# EXPECTED, AND THE GATE THIS MUST PASS.  On c10h8 the steps are small and the
# model is accurate, so this should rarely fire and the run should collapse back
# onto the control -- recovering BOTH the +70.7% cost AND the 7.085 mEh.  On
# c2h4 / h4 / Cr2 the rotations are large and it should stay engaged.
# *** ITS GATE MUST SCORE c10h8's ENERGY, NOT ONLY ITS COST (item 330E). ***
# QED_TRUE_ENERGY_PROBE: the SECOND adaptive strategy, and the principled one.
# Instead of guessing model quality from a geometric proxy, MEASURE it: on the
# FIRST candidate step of each macroiteration always evaluate E(U), form the
# relative model error
#       rel = |true_change - model_change| / |model_change|
# and engage the true-energy machinery for the REST of that macroiteration only
# if rel > QED_TRUE_ENERGY_PROBE_TOL.  Cost is bounded at ONE extra transform
# per macroiteration when it declines -- on c10h8 that is ~10 transforms (~23 s)
# against the 68 (160.6 s) the unconditional knob pays (item 330A).
#
# WHY THIS RATHER THAN THE DISPLACEMENT TRIGGER.  The displacement trigger
# (QED_TRUE_ENERGY_ADAPTIVE) keys off ||U2_trial - I||, a PROXY.  Measured, its
# medians on the normalised scale are 6.18e-4 (h4) / 2.54e-4 (c2h4) / 6.47e-4
# (c10h8) -- h4 and c10h8 are indistinguishable at the median, and only the
# upper tail separates.  The probe keys off the quantity the whole thread is
# about (the PROJ-VAR gap, items 322/325), so it needs no per-system tuning.
# QED_TRUE_ENERGY_REACTIVE: the THIRD strategy, and the only one that survived
# c10h8.  Do not PREDICT whether the model can be trusted -- wait until it
# demonstrably fails, then switch the true-energy machinery on for the rest of
# the run.  Default 0 = off.
#
# WHY THE PREDICTIVE STRATEGIES CANNOT WORK (item 331D).  Both
# QED_TRUE_ENERGY_ADAPTIVE (displacement) and QED_TRUE_ENERGY_PROBE (measured
# model error) leave c10h8 on a stationary point 7.085 mEh ABOVE the control.
# The reason is not tuning: on c10h8 the model error at macro 0 is genuinely
# large (rel = 0.158, and 0.996 at macro 2) and FOLLOWING IT ANYWAY is what
# finds the better basin -- the control sits nearly flat for five macros and
# then drops 0.0116 Eh at macro 7.  The beneficial exploration IS the model
# being wrong, so model quality cannot decide whether to enforce descent.
#
# THE REACTIVE SIGNAL SEPARATES THE TEST SET PERFECTLY, WITH NO THRESHOLD.
# Macro-energy RISES in each control: c10h8 0, Cr2 1, c2h4 216, h4 406.  So
# "has a macroiteration energy gone UP" is exactly the discriminator, it is
# dimensionless, and -- unlike every norm in this file -- it is INDEPENDENT OF
# SYSTEM SIZE (item 331B).  It is also Kreplin's own third condition: item
# 264A's rule is "exact Hessian if k == 0 OR |D^k - D^(k-1)| > eps_c OR THE
# ENERGY ROSE".  The clause that works needs no constant at all.
#
# COST WHEN IT NEVER FIRES: exactly zero -- no E(U) transform is ever run, and
# the run is the shipping path bit-for-bit.  Cost when it fires: the one
# macroiteration that rose.  This version arms from the NEXT macroiteration
# onward and does NOT roll back the macro that rose; on h4 the first rise is
# already large (macro 2, -2.2665 -> -0.533), so if arming that late proves too
# late, the refinement is to restore the previous macro's orbitals before
# re-running it.  That rollback is deliberately NOT attempted here.
_QED_TRUE_ENERGY_REACTIVE = int(os.environ.get("QED_TRUE_ENERGY_REACTIVE", "0"))
# A rise must clear this to count, so numerical noise at convergence cannot arm
# the knob.  1e-10 matches the macro convergence test at :5318.
_QED_TRUE_ENERGY_REACTIVE_TOL = float(
    os.environ.get("QED_TRUE_ENERGY_REACTIVE_TOL", "1e-10")
)
_QED_TRUE_ENERGY_PROBE = int(os.environ.get("QED_TRUE_ENERGY_PROBE", "0"))
# Provisional: the model is "good" if the true and predicted energy changes
# agree to 10%.  Item 320 measured the shipping ratio at 1.0002 on a healthy
# system, so a sound model sits far below this; c2h4's rho_true was NEGATIVE
# from the first microiteration (item 325 header).  NOT yet swept.
_QED_TRUE_ENERGY_PROBE_TOL = float(
    os.environ.get("QED_TRUE_ENERGY_PROBE_TOL", "0.1")
)
_QED_TRUE_ENERGY_ADAPTIVE = int(os.environ.get("QED_TRUE_ENERGY_ADAPTIVE", "0"))
# Default 5e-3 on the NORMALISED scale -- chosen from the measured
# distributions (h4 38.9%% / c2h4 30.8%% / c10h8 14.7%% engagement), NOT
# inherited from Kreplin.  Still provisional until the c10h8 ENERGY test.
_QED_TRUE_ENERGY_EPS_C = float(os.environ.get("QED_TRUE_ENERGY_EPS_C", "5e-3"))
#
# QED_DEGEN_CANON: pin the ARBITRARY GAUGE inside near-degenerate orbital
# clusters.  Default 0/OFF, so every existing run is byte-for-byte unchanged.
#
# THE MEASUREMENT THIS EXISTS FOR (MgH+ CAS(8,12) cc-pVDZ, lambda 0, seed 1,
# identical in every other respect):
#     OMP_NUM_THREADS=1  ->  33, 33, 33 macroiterations, energies BIT-IDENTICAL
#     OMP_NUM_THREADS=4  ->   6, 28, 26          (and 6, 32, 30 with MKL_CBWR)
# The energy is robust throughout (-199.543974671 +- 3e-9); only the COUNT
# moves, by 5.3x, with nothing physical varied.
#
# MECHANISM.  Eigenvectors WITHIN a degenerate subspace are arbitrary, so a
# last-bit difference from threaded MKL rotates them freely.  Measured directly:
# the same CQED-RHF calculation run at 1 and at 4 threads gives IDENTICAL
# spaces (principal angle 0.0000 deg in the inactive, active AND virtual
# blocks) but different mixing inside the degenerate pi pair.  The CASCI energy
# is invariant to that mixing -- verified to 1e-12 on all three roots -- but
# the OPTIMIZER'S TRAJECTORY IS NOT: the preconditioner M = |diag(H)| + 1e-12
# is built in the orbital-PAIR parameter basis, so rotating within a block
# changes diag(H) even though the energy does not change, which changes the
# Krylov space and hence the path length.  The energy is invariant; the route
# to it is not.
#
# RELATION TO THE EXISTING SIGN FIX.  The QED_RANDOM_ORBITAL_SEED block below
# already pins column SIGNS, for the same underlying reason (it records
# 9/12/11 macroiterations across identical repeats before it was added).  Two
# gaps: it runs ONLY on the random-rotation path, never on the canonical-guess
# path, and signs are the weaker case -- on MgH+ the two runs are not related
# by any sign pattern (|C| itself differs), i.e. genuine degenerate mixing.
_QED_DEGEN_CANON = int(os.environ.get("QED_DEGEN_CANON", "0"))
_QED_DEGEN_TOL = float(os.environ.get("QED_DEGEN_TOL", "1e-6"))
#
# (RETIRED 2026-07-29: QED_SCHEDULE_NATURAL.)  It scheduled
# N_microiterations / N_orbital_optimization_steps off the FIRST step computed
# at (microiteration 0, orbital step 0) rather than the first ACCEPTED one, on
# the argument that a rejection cascade makes the accept branch read a
# post-shrink norm.  Removed because the accepted norm is the right quantity
# after all.  Require the scheduling step to be BOTH unconstrained and actually
# energy-lowering and the two coincide in every case: an interior step accepted
# first try IS the accepted step; an interior step that is REJECTED leads to a
# cascade whose accepted step sits exactly on the shrunken boundary (measured on
# c10h8: accepted = first / 2^rejections in all 7 cascades, since step_control
# shrinks to 0.5*||s_rejected||); and a bound first step has ||s|| == Delta,
# i.e. the initial radius 0.5 carrying no information (7 of 19 c10h8 calls).
# The only three calls where the knob changed a decision all latched a REJECTED
# Newton step -- including the run's worst step, model ratio -74.6 -- and the
# run cost more (2379 vs 2223 matvecs).  A rejected step's length is evidence
# the model was wrong, not evidence of the natural step scale.
#
#   QED_GLTR_REUSE : reuse the GLTR Lanczos basis on a retry   (default 0 = off)
#
# When a step is rejected, the reject branch (~13037) shrinks the trust radius
# and changes NOTHING else: self.U2 is updated only in the accept branch, self.U3
# is re-seeded from it at ~12148, and the gradient / Hessian diagonal are rebuilt
# only on accept.  The GLTR Lanczos recurrence depends on (H, M_abs, r) alone --
# the radius enters only at the CHECK A/B subproblem -- so the retry rebuilds a
# bit-identical Krylov space at full matvec cost.  Measured c10h8: rows 37-40 of
# one cascade rebuild the same basis 4 times (0.4923 -> 0.25 -> 0.125 -> 0.0625).
# With this on, only the tridiagonal subproblem is re-solved.  The accepted step
# is unchanged, so unlike every trust-radius change tried this session the
# trajectory is IDENTICAL -- there is no convergence risk to trade off.
# Wired to solve_gltr_trust_region only; solve_gltr_with_operator (the QN/BFGS
# twin) is untouched and does not fire with QED_DISABLE_QN=1.
_QED_GLTR_REUSE = int(os.environ.get("QED_GLTR_REUSE", "0"))
#
#   QED_MV_BENCH : one-shot matvec cost-vs-batch-size curve   (default 0 = off)
#
# Prints a timing table at the first microiteration and changes nothing else.
# See the benchmark block in microiteration_optimization6 for what it separates.
_QED_MV_BENCH = int(os.environ.get("QED_MV_BENCH", "0"))
#
#   QED_JD_MAXITER / QED_JD_TOL : inner PCG budget in inner_solve_pcg
#                                 (defaults 5 / 0.1 = the original values)
#
# inner_solve_pcg solves the Jacobi-Davidson correction equation
# (I-Q)(H-theta I)(I-Q) v = -r with PCG.  Every inner iteration is one sigma
# build, so these two knobs trade inner matvecs against OUTER Davidson
# iterations -- each of which also costs a sigma build, plus orthogonalization
# and the projected eigensolve.  Solving the correction equation harder is only
# worth it if it removes more outer iterations than the inner ones it adds.
_QED_JD_MAXITER = int(os.environ.get("QED_JD_MAXITER", "5"))
_QED_JD_TOL = float(os.environ.get("QED_JD_TOL", "0.1"))
#
#   QED_G_SYMV : exploit the symmetry of G_ij in the matvec   (default 0 = off)
#
# The Hessian-vector product is memory-bandwidth bound at MACHINE PEAK, not
# compute bound: at c10h8 dims (nmo=180, n_in_a=29, n_act_orb=10) the four GEMMs
# at :9397-9400 stream 0.394 GB for ~54 MFLOP -- 0.25 flop/byte -- and measure
# 48 GB/s, which is all this box has.  So the only way to make it faster is to
# read fewer bytes, and the numbers say exactly where:
#     G_ij  218 MB  read once   4.48 ms   <- 59% of the whole matvec
#     G_ti   75 MB  read twice  1.74 ms      (fits in the 36 MB L3 on the
#     G_tu   26 MB  read once   0.30 ms       second pass, so it is cheap)
#     non-GEMM code in build_sigma_reduced7   0.59 ms (7.2%)
# G_ij is SYMMETRIC to 1 ulp (verify with QED_G_SYM_CHECK; see the note at the
# G_blocks build site), so a symmetric product reads one 109 MB triangle instead
# of the full 218 MB square.
#
# HOW BIG THE WIN IS DEPENDS ON THE PROCESS, SO USE THE IN-SITU NUMBER: 2.40x
# (4.49 -> 1.87 ms) in a bare process, but only ~1.5x (6.00 -> 3.80 ms) once psi4
# is imported, which is what every real driver does.  1.5x is the number that
# matters and it is self-consistent: 0.59 * (1 - 1/1.5) = 19.7% predicted against
# a MEASURED 19.0% cut in c10h8 matvec wall time.
#
# Thread count does not change any of this, checked 1/2/4/8/16/32 plus default:
# results identical to 1.6e-12 at every setting and the ratio stays in 1.34-1.61x,
# because in a psi4-loaded process NEITHER path scales -- numba's np.dot sits at
# ~36 GB/s and dsymv at ~24-29 GB/s no matter what MKL_NUM_THREADS says.  (numba's
# np.dot is NOT numpy's: nopython code binds scipy's cython_blas, and numpy's
# dgemv DOES thread in the same process, 4.48 ms vs numba's 6.00 ms.  numba dsymv
# at 3.80 ms still beats all four variants, so this is the right call here -- but
# the numba/numpy BLAS gap is a separate ~1.3x lead on every other big product.)
# There is no numba-prange/MKL-OpenMP nesting to worry about: the call sites are
# jitted parallel=False, so MKL's threads are the only ones in play.
#
# numba's np.dot always lowers to dgemm/dgemv and has no symmetric form, so
# dsymv is bound by hand below.  Gated on num_states == 1: a batched call would
# have to stream the triangle once PER VECTOR (dsymv is a matrix-VECTOR product),
# which for the 41-vector Davidson cold-build guess would be 41 x 109 MB against
# one 218 MB dgemm pass -- a large loss.  2083 of the 2085 real matvecs in a
# c10h8 run are single-vector, so the gate costs almost nothing.  (A blocked
# dsymm would cover both cases and is the natural extension.)
#
# DEFAULT ON since 2026-07-30.  The knob had been held at 0 for one reason only:
# the ctypes binding cost 5.6 s of numba recompile per process, which cancelled
# the -6.6 s it saves on matvec (c10h8 A/B: matvec wall 36.04 -> 29.47 s, -18.2%,
# but `optimization took` moved only 255.23 -> 254.27 s).  That penalty is gone
# now that dsymv is bound BY SYMBOL NAME -- see the _symv_accumulate note below --
# so the matvec cut reaches the wall clock instead of paying for a recompile.
# The switch to MKL's dsymv_ that the symbol binding implies was verified
# BIT-IDENTICAL to the previous scipy/cython_blas path (max|d| exactly 0.0).
# _QED_G_SYMV's VALUE is set in orbital_sigma.py (imported below), because the
# no-MKL fallback there has to be able to force it off.  Documented here, with
# the other knobs, because that is where a reader looks for it.
#
#   QED_G_TILED : blocked build of the G_ij/G_ti/G_tu blocks   (default 0 = off)
#
# "transpose matrix G" is 15.7 s of a c10h8 run, second only to build
# intermediates, and it is ONE numpy statement: G.transpose(3,1,2,0) sliced and
# reshaped into the three blocks (~12070).  G is stored (k,l,r,s) with s
# contiguous, and that permutation drags s all the way to OUTERMOST, so the copy
# moves 8-byte scattered elements: measured 0.151 s per microiteration for 638 MB
# of traffic = 4.2 GB/s, against 43.0 GB/s for a memcpy of the same volume.  The
# 0.151 s x 103 microiterations reproduces the observed 15.7 s exactly.
#
# The permutation is a plain transpose in disguise -- for fixed (k,l,r) the source
# is contiguous in s, and for fixed (s,l,r) the destination is contiguous in k --
# so blocking over (s,k) lets both tiles sit in cache.  Measured at c10h8 dims:
#     numpy, as shipped          0.1511 s    4.2 GB/s   1.00x
#     numba tiled, tile 32       0.0216 s   29.6 GB/s   7.01x
#     (pure memcpy floor)        0.0148 s   43.0 GB/s  10.19x
# Output is BIT-EXACT against the numpy path (verified at tile 8/16/32/64), and
# the index convention is unchanged, so no consumer sees any difference -- this
# is purely how the same bytes get moved.  Tile 32 is used; 16 and 64 are within
# 30% of it, so the choice is not delicate.  Set QED_G_TILED_CHECK=1 to re-verify
# that bit-exactness in situ, at the real dimensions (~12125).
#
# The buffers are allocated once and reused across microiterations, which is part
# of the win: the numpy path allocates 319 MB of fresh pages every microiteration
# and eats the first-touch faults.  Safe because the kernel writes every element
# of each buffer on every call (the loops span the full s/l/r/k ranges, so nothing
# stale can survive), and the only consumers that outlive a microiteration take a
# copy.deepcopy (~12366, ~13402).
#
# DEFAULT ON since the c10h8 A/B of 2026-07-30.  Three legs back to back on an
# otherwise idle box, over an identical 103 calls to this section:
#     transpose matrix G   20.47 s -> 5.73 s   (3.57x)
#     optimization took   278.73 s -> 255.23 s (-8.4%)
# and it composes with QED_G_SYMV (5.64 s with both on; the matvec's -18% is
# undisturbed).  The 3.57x here against 7.01x in the standalone probe is the
# expected direction -- the real process holds 33 GB resident and a colder memory
# system.  NOTE that the A/B cannot itself prove bit-exactness: this run is
# nondeterministic well upstream of CASSCF (two legs with identical settings
# diverge in the CQED-RHF SCF at ~1e-11, from threaded-BLAS reduction order), so
# the legs' final energies differ either way.  QED_G_TILED_CHECK is what proves
# it; set QED_G_TILED=0 to fall back to the numpy path.
_QED_G_TILED = int(os.environ.get("QED_G_TILED", "1"))
#
#   QED_BI_FASTPATH : the two uniform hotspots of build_intermediates (default 1)
#
# QED_PROFILE_INTERMEDIATES on c10h8 (103 calls, 28.27 s, and the BI_PROF marks
# sum to the printed total exactly, so nothing hides between them):
#     G_ij_plus_2L    10.03 s   0.0974 s/call   min 0.0939 max 0.1192
#     G_ti_D_times_L   6.93 s   0.0673 s/call   min 0.0635 max 0.1185
#     L_build          5.59 s   but only 19 of 103 calls rebuild (cache works)
# The first two are paid on EVERY call -- 17.0 s of a 244 s optimization -- and
# both were running an order of magnitude under the memory system:
#
#   :6813  G[:ni,:ni,:,:] = 2.0 * self.L[:ni,:,:,:]
#          The `2.0 *` materializes a full 218 MB temporary, which is then copied
#          into a STRIDED destination (G's axis 1 has extent n_occupied=39, the
#          slice takes n_in_a=29).  So four passes over 218 MB instead of two:
#          measured 6.0 GB/s against this box's 43 GB/s memcpy floor.
#          np.multiply(..., out=...) writes straight through.  Multiplying by 2.0
#          is EXACT in binary floating point, so this variant is bit-identical --
#          measured dev exactly 0.0, not merely small.
#
#   :6824  G[ni:no,:ni,:,:] = einsum("tv,vjrs->tjrs", D_tu_avg, L[ni:])
#          A (10,10) matrix against a 75 MB tensor.  L[ni:] is contiguous, so
#          this is a clean GEMM once reshaped to (n_act, n_in*rot*rot) -- no
#          transpose needed -- and einsum was getting 2.4 GB/s.
#
# Standalone at c10h8 dims: 4.26x and 4.07x respectively.  Note that np.dot into
# a FRESH array is only 1.67x while np.dot into a PERSISTENT buffer is 4.07x --
# allocating and first-touching 75 MB every call costs more than the BLAS does,
# the same effect that makes QED_G_TILED reuse its buffers.  np.dot requires a
# C-contiguous out=, and G's destination slice is strided, hence buffer + copy.
#
# NOT bit-exact on the second site: BLAS and einsum sum in different orders, so
# expect ~1e-16 relative there (measured 3.5e-16).  Same class of change as
# QED_G_SYMV, which moved the final c10h8 energy by 5.7e-12.
#
# DEFAULT ON since the c10h8 A/B of 2026-07-30, both legs back to back with the
# profiler on and the baseline re-measured in-session:
#     G_ij_plus_2L         10.06 -> 2.21 s    4.55x
#     G_ti_D_times_L        6.89 -> 1.85 s    3.72x
#     build intermediates  28.82 -> 15.59 s   -46%
#     optimization took   251.06 -> 229.49 s  -8.6%
#     10 macros both; energy -383.5044493792576 vs ...574 (2e-13)
# Every other BI_PROF section came in at 1.00-1.06x, which is what confines the
# effect to the two edited statements.  Set QED_BI_FASTPATH=0 to fall back.
#
# NOTE that the standalone probe UNDERSTATED this (4.26x -> 4.55x in situ),
# the reverse of QED_G_TILED (7.01x standalone -> 3.57x in situ).  The rule is
# not "standalone is always optimistic": removing ALLOCATIONS pays MORE under the
# real run's memory pressure, while merely reordering ACCESS pays less.  Classify
# which kind of fix it is before trusting a standalone number.
_QED_BI_FASTPATH = int(os.environ.get("QED_BI_FASTPATH", "1"))

#
#   QED_PROFILE_MICRO : resolve the UNTIMED python glue inside the
#                       microiteration (default 0 -- prints nothing, changes
#                       nothing)
#
# Item 239(G): the timed children of `microiteration` sum to 90.41 s of 111.17 s
# on the canonical c10h8 leg, so 20.76 s -- 10.5% of the whole macro loop, more
# than `build intermediates` itself -- is unattributed.  These marks bracket the
# regions the existing timers step over.  Emitted as "MICROPROF <label> took
# <s>" so they cannot collide with the unlabelled `... took` prints the log
# parsers already key on.  Rule 15: this exists because the cost has to be timed
# at the real call site, not predicted from a synthetic memset.
# NAME COLLISION, 2026-08-20: there is ALREADY a QED_PROFILE_MICRO at the
# bottom of this file (:21855) that wraps the trust-region solver METHODS and
# emits "MICRO_PROF <method> <s>".  These marks are a different thing (the
# untimed glue INSIDE microiteration_optimization6), so they get their own
# knob.  Setting both is fine and useful -- they are complementary.
_QED_PROFILE_GLUE = int(os.environ.get("QED_PROFILE_GLUE", "0"))


def _mp_start():
    return timer() if _QED_PROFILE_GLUE else 0.0


def _mp_end(label, t0):
    if _QED_PROFILE_GLUE:
        print("MICROPROF", label, "took", timer() - t0, flush=True)


#
#   QED_GRAD_CONV : add a GRADIENT criterion to the macroiteration stopping test
#                   (DEFAULT 1e-6 since item 247; set 0.0 to restore the
#                   pre-2026-08-20 energy-only behaviour exactly)
#
# Item 244.  The ONLY macro convergence test is :4857, |dE_avg| < 1e-10, energy
# only.  On MgH+ CAS(8,12) that costs 25 of 27 macroiterations: the answer is
# reached at macro 2 (3.8e-07 from the converged value) and macros 3-27 move the
# energy by 3.8e-07 in total, while dE crawls asymptotically down onto the 1e-10
# gate (1.43e-10 at macro 25).  Molpro stops the same calculation at 5 because
# it tests GRAD(0) against 1e-4 -- but item 216 established that Molpro's
# orbital gradient is a DIFFERENT QUANTITY from ours, so its threshold does not
# transfer.  Calibrated on OURS, from mgh_roots_vs_molpro/mgh_lam0.log:
#     tol 1e-4 -> stop after macro  3, energy error 1.6e-07
#     tol 1e-5 -> stop after macro  6, energy error 3.4e-08
#     tol 1e-6 -> stop after macro 11, energy error 3.0e-09
#     tol 1e-7 -> stop after macro 26 -- our |g| PLATEAUS near 1e-7 and never
#                 gets below 9.6e-08, so anything tighter never fires.
# THE QUANTITY TESTED is the gradient at the START of the macro's orbital work
# (the FIRST ||reduced_gradient|| of the macro, captured at the live print site
# in microiteration_optimization6).  NOT the last: the last-of-macro gradient is
# non-monotone -- 7.7e-07 at macro 2 rising back to 2.4e-06 at macro 7, the
# rebound of item 215 -- and would declare convergence by luck.
# IT IS OR-ed WITH THE ENERGY TEST, WHICH IS WHY IT ADDS NO NEW RISK: the code
# already stops on the energy test alone, so a second sufficient condition can
# only ever stop EARLIER, never later, and cannot admit a false convergence that
# :4857 does not already admit.
# SCORED ON BOTH SYSTEMS WE HAVE (items 246-247), which is why 1e-6 is the
# default and 1e-5/1e-4 are not:
#   MgH+ CAS(8,12)  1e-6 -> 27 macros become 11, 816.6 -> 572.7 s (-29.9%), for
#                           2.99e-09 Ha.  (1e-5 would give 7 macros / -48.1% for
#                           1.79e-08 Ha -- safe on both systems measured, but 6x
#                           the error, so it is left opt-in.)
#   c10h8 CAS(10,10) 1e-6 -> NO-OP, verified by running it: macro N 10 = 10,
#                           energy to 4.0e-12, criterion fired 0 times.  Its
#                           gradient DIPS to 4.4e-04 at macro 4 and rebounds to
#                           6.8e-03, which is what rules 1e-4 out (item 245E
#                           measured that dip at 1.06e-04 -- a 1.06x margin).
#                           Smallest norm the whole run sees is 4.45e-06, at the
#                           FINAL macro; the margin against the last macro that
#                           still had work to do (9.5e-04) is 949x.
_QED_GRAD_CONV = 1e-06  # resolved per calculation by qed_ci.runtime

# QED_MACRO_ENERGY_CONV: the macroiteration ENERGY convergence cut, which was
# hardcoded at 1e-10.  DEFAULT 1e-10 = exactly the previous behaviour; 0
# disables the energy criterion so the run stops on the GRADIENT alone.
#
# WHY IT NEEDS A KNOB.  The energy test is evaluated BEFORE the gradient test
# and sets convergence on its own, so it SHORT-CIRCUITS QED_GRAD_CONV.  Near a
# stationary point dE ~ |g|^2 / 2k, so a 1e-10 energy cut corresponds to
# |g| ~ 1.4e-05: the energy criterion structurally caps the reachable gradient
# at about 1e-5 and NO SETTING OF QED_GRAD_CONV CAN GO BELOW IT.  Measured on
# C2H4 distorted to C1, CAS(4,7)/cc-pVDZ, with QED_GRAD_CONV=1e-8: the run
# stopped at |g| = 2.55e-06 after a step of dE = -6.5e-11, printing
# "OPTIMIZATION CONVERGED" without ever satisfying the gradient criterion.
# Molpro reaches 1.3e-12 on the same system.
#
# That matters beyond tidiness: E_SA is stationary at the solution but the
# INDIVIDUAL state energies are not, so stopping 2.6e-06 short splits them by
# +-5.3e-07 (equal and opposite, average still exact to 1e-12) and moves the
# state gradients by 3e-06 -- which is what made that system unusable as a
# regression test against Molpro.
_QED_MACRO_ENERGY_CONV = 1e-10  # resolved per calculation by qed_ci.runtime

# QED_MICRO_GRAD_FLOOR / QED_MICRO_ENERGY_FLOOR: the two ABSOLUTE backstops
# inside microiteration_optimization6.  Defaults 1e-7 and 1e-10 reproduce the
# previous hardcoded behaviour exactly.
#
# The microiteration has an ADAPTIVE test (0.1 * convergence_threshold for the
# gradient, 0.01 * convergence_threshold for the energy) and, underneath it, an
# absolute floor.  The gradient floor is unconditional:
#
#     if gradient_norm < 1e-7: convergence = 1; break
#
# so the inner orbital optimisation cannot drive the reduced gradient below
# 1e-7 no matter what the outer loop asks for -- and therefore neither can the
# MACRO gradient.  That is the ~1e-7 floor measured on C2H4 distorted to C1
# (9.58e-08 at davidson_threshold 1e-9, 9.89e-08 at 1e-11, 9.83e-08 at 1e-13 --
# invariant in the CI accuracy, and every value sitting just under this cutoff),
# against Molpro's 1.3e-12 on the same system.
_QED_MICRO_GRAD_FLOOR = 1e-07  # resolved per calculation by qed_ci.runtime
_QED_MICRO_ENERGY_FLOOR = 1e-10  # resolved per calculation by qed_ci.runtime

# QED_MICRO_GRAD_SCALE / QED_MICRO_ENERGY_SCALE: the Kreplin-style RELATIVE part
# of the same two microiteration tests -- the criterion is a fraction of the
# convergence_threshold carried in from the macro loop (i.e. of the scaled
# initial gradient), with the absolute floors above underneath it:
#
#     gradient:  gradient_norm < QED_MICRO_GRAD_SCALE * convergence_threshold
#     energy:    |dE| < max(QED_MICRO_ENERGY_SCALE * convergence_threshold,
#                           QED_MICRO_ENERGY_FLOOR)
#
# Defaults 0.1 and 0.01 are the values that were hardcoded.  They are exposed
# for the same reason as the floors: the absolute cutoffs existed because the
# augmented-Hessian subproblem was unreliable at very small gradients, and that
# constraint is weaker now that the small-gradient case takes a Newton step --
# so how tightly the microiteration should be driven is a choice worth being
# able to make per run rather than one baked into the source.
_QED_MICRO_GRAD_SCALE = 0.1  # resolved per calculation by qed_ci.runtime
_QED_MICRO_ENERGY_SCALE = 0.01  # resolved per calculation by qed_ci.runtime


#
#   QED_GRAD_BLOCKS : build_gradient's contraction off the G blocks (default 0)
#
# build_gradient:7288 is
#     B[:, :no] = A[:, :no] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:, :no])
# with NO optimize= -- so it is numpy's naive C einsum loop over the whole
# 394 MB G, measured at 11 GB/s against this box's 43 GB/s memcpy floor.
# c10h8: 249 calls, 8.92 s, 0.0358 s/call, the largest of the three G consumers
# after the tiled copy itself.
#
# That contraction is ALREADY SOLVED three functions away.
# microiteration_exact_energy:10024-10026 carries the identical statement as its
# commented reference and evaluates it off the blocks:
#     B_i = 2A_i + np.dot(T_i, G_ij) + np.dot(T_a, G_ti)
#     B_a = 2A_a + np.dot(T_i, G_ti.T) + np.dot(T_a, G_tu)
# (build_gradient wants 1.0*A, not 2.0*A -- that is the only difference.)  The
# blocks are already built at :12281, three lines before the call site, so the
# fast path computes nothing new; it only reads less and reads it through BLAS.
#
# G_ij and G_tu are symmetric to 1 ulp (QED_G_SYM_CHECK, :12333), so those two
# products go through _symv_accumulate and touch one triangle each.  Traffic:
#     einsum   394 MB at 11 GB/s      (naive loop, whole G including G_it)
#     blocked  109 MB triangle (G_ij) + 2x75 MB (G_ti, G_ti.T) + 13 MB triangle
#              (G_tu) = 272 MB, at the 55/36 GB/s dsymv/dgemv rates of :306
#
# NOT bit-exact in general: BLAS and einsum sum in different orders.  Same class
# of change as QED_BI_FASTPATH's second site and QED_G_SYMV.  In practice
# QED_GRAD_BLOCKS_CHECK has returned max|d| EXACTLY 0.0 at both h2o and real
# c10h8 dims -- but that is one call per run, not a proof for every call, so the
# claim here is "~1e-13 or better", not "bit-exact".  G itself is still built and
# still dumped, so the cpp_casscf dump contract and the C++ port are untouched;
# G_blocks arrives as an OPTIONAL argument, so every caller that does not pass it
# keeps the einsum path.
#
# DEFAULT ON since the c10h8 A/B of 2026-07-30 (both legs back to back, baseline
# re-measured in-session, all other knobs at their defaults):
#     build gradient        9.46 -> 2.96 s   3.20x
#     optimization took   236.13 -> 230.29   -2.5%
#     build intermediates  15.59 / 15.92     transpose G   5.46 / 5.27
#     hessian diagonal     12.27 / 12.41     full JK      43.49 / 42.94
# i.e. every other section flat, which is what confines the effect to :7288.
# 10 macros / 103 microiterations both legs; final energy 4.9e-12, which is the
# run's known upstream nondeterminism (threaded-BLAS reduction order in the
# CQED-RHF SCF), not this change.  Set QED_GRAD_BLOCKS=0 to fall back.
_QED_GRAD_BLOCKS = int(os.environ.get("QED_GRAD_BLOCKS", "1"))
#
#   NTHREAD : threads for psi4 AND for the C backend       (default unset = 1)
#
# THIS IS NOT A SOLVER KNOB -- it is the fix for a process-wide default that
# silently serialised the entire C backend.  `import psi4` calls
# psi4.set_num_threads(1) as part of its own initialisation, and psi4's
# implementation of that sets the OMP and MKL thread counts for the WHOLE
# process.  It therefore overrides an OMP_NUM_THREADS exported in the shell,
# and because helper_PFCI imports psi4 at :18, ci_solver.c / orbital.c ran on
# ONE thread in every driver that did not explicitly ask for more -- their omp
# pragmas were live but empty.  Measured on c10h8 (24 threads vs the default):
#     wall               593.0 -> 476.8 s
#     optimization took  227.1 -> 163.0 s
#     full JK             45.1 ->  10.2 s
#     CI step             34.0 ->  14.8 s
# i.e. this one line is worth more than every algorithmic knob above combined.
#
# DELIBERATELY OPT-IN AND DELIBERATELY NOT HARDCODED: small systems REGRESS.
# h2o (6-31g, 2 states) goes 0.10 -> 0.27 s at 24 threads, because the C
# backend's arrays are far too small to amortise the fork/join, and numpy's own
# threaded BLAS competes for the same cores (`build hessian diagonal` is +44% at
# 24 threads on c10h8 even though everything around it improves).  So the
# correct N is a property of the SYSTEM, not of the machine, and there is no
# default that is right for both -- leaving this unset preserves today's exact
# behaviour for every existing driver.
#
# Set it in the driver script's environment (`NTHREAD=24 python c10h8.py`).
# OMP_NUM_THREADS is honoured as a fallback so an already-exported value is not
# silently discarded by psi4 -- which is the whole defect this knob exists for.
_QED_NTHREAD = os.environ.get("NTHREAD") or os.environ.get("OMP_NUM_THREADS")
if _QED_NTHREAD:
    psi4.set_num_threads(int(_QED_NTHREAD))
#
# ITS COMPANION LIVES IN helper_cqed_rhf.py: QED_PSI4_MEM.  Same class of
# defect -- psi4's 500 MiB default silently demotes a `scf_type pk` deck to
# DiskJK, which re-reads the AO integrals from disk on every SCF iteration
# (c10h8: psi4.energy("scf") 35.9 s -> 14.0 s at 8 GB).  Also opt-in.  Set both
# together in the driver's environment:
#     NTHREAD=24 QED_PSI4_MEM=8e9 python c10h8.py
# helper_cqed_rhf is imported at :22, so its knob is applied before anything
# here runs.
#
# ---------------------------------------------------------------------------
# QED_SKIP_GKL : skip building self.gkl, which is DEAD CODE.        (default 0)
#
# self.gkl is the nmo x nmo modified one-electron integral of the OLD
# full-Hamiltonian CASCI path.  Its ONLY consumer is build_sigma's
# `elif ci_level == "cas"` branch, and build_sigma's ONLY live call site sits
# inside `elif ci_level == "cis"` -- while build_sigma itself handles only
# "fci" and "cas".  So the branch that reads self.gkl is unreachable from
# every live path.  (The ACTIVE-space gkl2, built from active_fock_core and
# active_twoeint, is the one the CAS algorithm actually uses; it is unrelated
# and untouched.)
#
# PROVEN, NOT ASSUMED: filling self.gkl with NaN and running the full suite
# gives 127 passed / 2 failed -- bit-identical to the known baseline, whose two
# failures are the hardcoded-absolute-path ones documented in CLAUDE.md.
#
# WHY IT IS WORTH A KNOB.  Two reasons, and the second is the real one:
#   1. it costs ~5 s at nmo 302 and ~13 s at nmo 412 (the build is a Python
#      triple loop; measured 1.8e-7 s per nmo^3 iteration);
#   2. it is the ONLY consumer of the full packed self.twoeint that is not a
#      J/K transformation, so skipping it is a PREREQUISITE for QED_LIBTRANS_OO
#      below -- with gkl gone, twoeint exists solely to produce J and K.
# DEFAULT OFF (i.e. still built) because removing a quantity on the strength of
# a reachability argument deserves an explicit opt-in first.
# ---------------------------------------------------------------------------
#   QED_LIBTRANS : build self.twoeint through psi4's libtrans   (DEFAULT 1
#                  since item 247; set 0 for the mo_eri path)
#
# Replaces the mints.mo_eri(Ca,Ca,Ca,Ca) at :5362 -- the single largest item in
# the c10h8 job -- with mints.integrals() + IntegralTransform + a readback
# through psi4_dpd_shim.so.  Measured at c10h8 dims, 24 threads:
#     mo_eri(Ca,Ca,Ca,Ca)                              134.75 s
#     integrals 17.24 + presort 3.64 + transform 11.56 + readback 1.23
#                                                    =  33.67 s   => -101 s
# Correctness: the packed array agrees with mo_eri to 2.3e-12 at c10h8 dims and
# 6.2e-15 at h2o/6-31g (validation/dpd_readback_check.py).
#
# WHY THIS IS THE ONLY ROUTE THAT WINS.  mo_eri's cost is almost entirely the
# AO integrals, which it rebuilds internally on every call.  Measured
# separately: mints.ao_eri() 125.65 s vs mints.integrals() 17.24 s for THE SAME
# integrals -- the difference is purely materialising the dense nbf^4 array
# (8.40 GB, 8x redundant, unscreened).  And NOTHING in psi4's AO path threads:
# ao_eri is 125.32 s at 1 thread and 125.65 s at 24, integrals() 16.98/17.24,
# presort 3.66/3.64.  So "read the AO integrals and transform them ourselves"
# costs 125.65 + 19.09 = 144.7 s, i.e. WORSE than mo_eri; the only cheap AO
# build psi4 has writes to disk, where numpy cannot reach it.
#
# NOT molpro's design, deliberately.  Molpro keeps AO integrals on disk and
# re-transforms only the blocks it needs per macroiteration.  At 1 thread that
# beat us; at 24 threads it does NOT -- our per-macro rotate is ~1.0 s against
# libtrans's 7.55 s re-transform, so over 10 macros the molpro route is 96.2 s
# against 42.0 s for "swap the one-time build and keep everything else".  This
# knob therefore changes ONE line and no architecture.
#
# SECOND, UNPLANNED WIN: libtrans stores the buffer ALREADY PACKED as
# (npair x npair) with 8-fold folding, which is exactly what build2DSO packs
# self.twoeint into (:4578-4590).  So with this on, the dense nmo^4 array never
# exists -- 8.40 GB and its einsum temporaries were the run's measured 33.1 GB
# peak (see the comment at :4554).
#
# DEFAULT ON SINCE ITEM 247.  It was default-off because it needs
# psi4_dpd_shim.so built next to helper_PFCI.py (see the build line in
# psi4_dpd_shim.cpp) and that .so is ABI-coupled to the psi4 build it was
# compiled against.  But default-off meant NO ORDINARY USER EVER GOT the -30.2%
# of total wall (items 230/235, both of which end "make it the default"); every
# benchmark that ever measured it set it explicitly.  The shim being missing is
# a RECOVERABLE condition -- the code falls back to mo_eri and the run is
# correct, only ~100 s slower -- so the right default is ON with a LOUD
# fallback, which is what _libtrans_fallback() below now provides.
# Set QED_LIBTRANS=0 to force the mo_eri path (for an A/B, or if the shim is
# ABI-mismatched and the banner is unwanted).
_QED_LIBTRANS = int(os.environ.get("QED_LIBTRANS", "1"))


def _libtrans_fallback(reason):
    """Announce a QED_LIBTRANS fallback on BOTH streams, then return None.

    Item 247.  With the knob default-ON, a missing or ABI-mismatched shim no
    longer reads as a deliberate opt-out -- it reads as a normal run that
    happens to be ~100 s slower and to allocate 8.40 GB it did not need to.
    The banner goes to stderr as well as stdout because stdout is where psi4's
    own output lands, and a single line is lost in it.

    The literal substring "falling back" is LOAD-BEARING: bench_c10h8_1core.sh
    greps for it to VOID a timing leg (and so does dpd_readback_check.py).
    Keep it in any message passed here.
    """
    banner = (
        "\n" + "!" * 78 + "\n"
        "!! QED_LIBTRANS is ON but UNAVAILABLE -- falling back to mints.mo_eri.\n"
        f"!! Reason: {reason}\n"
        "!! The run is CORRECT but will be ~100 s slower at c10h8 dims and will\n"
        "!! allocate the dense nmo^4 array (8.40 GB) that libtrans avoids.\n"
        "!! Fix: rebuild psi4_dpd_shim.so with the g++ line in psi4_dpd_shim.cpp\n"
        "!! (it is ABI-coupled to psi4 -- rebuild it after changing psi4/conda\n"
        "!! envs).  Set QED_LIBTRANS=0 to silence this and use mo_eri on purpose.\n"
        + "!" * 78
    )
    print(banner, flush=True)
    print(banner, file=sys.stderr, flush=True)
    return None
#
#   QED_OTR_TR : solve the trust-region subproblem with otr_trust_region.py
#                                                               (default 0)
#
# Replaces the ENTIRE solver dispatch inside microiteration_optimization6
# (:13227 onward -- GLTR, the LSTRS/Davidson beta bisection, and the
# gradient-small Newton branch) with one call to
# otr_trust_region.solve_trust_region_subproblem, a faithful port of
# OpenTrustRegion's level_shifted_davidson.  Nothing else changes: the trust
# radius, the energy checks, the step acceptance and the microiteration
# structure all stay exactly as they are, and with the knob off not one line of
# the existing path is touched.
#
# WHY.  Our beta bisection sits OUTSIDE the Davidson solve, so every trial beta
# re-enters a full Davidson and pays fresh Hessian-vector products.  Measured on
# c10h8 at true single core (QED_SOLVER_STATS=1): 9 cold builds (302 matvecs) +
# 36 warm restarts (414) = 716 of lstrs_davidson's 749 matvecs are spent
# re-deriving a search space for a new beta, and newton_small_gradient spends a
# further 728 solving Hx=-g matrix-free at the full 5789 dimension.  Totals:
# ours 2430 Hessian-vector products against molpro's 741 for the same molecule.
#
# The ported design keeps ONE search space (V, HV) and runs the level-shift
# search inside the projected (n_red+1) augmented Hessian, where changing the
# shift costs one small eigh and no matvec at all; the Newton step becomes a
# dense solve in that same reduced space.  This is the Nonlinear Arnoldi
# acceleration of LSTRS (Lampe, Rojas, Sorensen, Voss, IMM-TR-2009-09), which
# also states outright that the strategy the current code uses -- seeding the
# next eigensolve with the previous eigenvector -- "usually does not
# significantly speed convergence".  Our 36 warm restarts still costing 11.5
# matvecs each is that sentence, measured.
#
# SCOPE.  *** NARROWED 2026-08-04: this now covers the TRUST-REGION subproblem
# only -- gltr (770 matvecs) and lstrs_davidson (749) = 1519 of c10h8's 2430
# Hessian-vector products.  newton_small_gradient (a further 728) KEEPS its
# MINRES solve: it is a linear solve, not a trust-region subproblem.  Earlier
# QED_OTR_TR measurements replaced that branch too and are NOT comparable. ***
# The lstrs_internal site at :8698 (19 solves, 69 matvecs, in the
# internal-optimization path) is NOT covered and still runs the original
# solver.
#
# DEFAULT OFF: unvalidated on production systems.  Turning it on changes the
# steps taken, so energies agree only to the convergence threshold, not to
# machine precision, and the macroiteration/step counts will differ.
_QED_OTR_TR = int(os.environ.get("QED_OTR_TR", "0"))
# Sub-knobs, live only when QED_OTR_TR=1.  Defaults are OpenTrustRegion's own
# (opentrustregion.f90:174-179); QED_OTR_SUBSOLVER="jacobi-davidson" switches on
# the JD correction branch, which in this port uses scipy's MINRES rather than
# OpenTrustRegion's (a documented deviation -- see otr_trust_region.py).
_QED_OTR_MAXITER = int(os.environ.get("QED_OTR_MAXITER", "50"))
_QED_OTR_RANDOM_VECS = int(os.environ.get("QED_OTR_RANDOM_VECS", "1"))
_QED_OTR_SUBSOLVER = os.environ.get("QED_OTR_SUBSOLVER", "davidson")
#   QED_OTR_RED_FACTOR : OpenTrustRegion's residual reduction factor, i.e. the
# micro loop exits at ||g + Hx - mu x|| < max(factor*|g|, 1e-12).  Its own
# default is 1e-3, which is what this knob preserves and therefore what every
# earlier QED_OTR_TR measurement on c10h8 was taken at.  1e-3 was measured 400x
# less accurate than LSTRS's quasi-optimal exit on real CASSCF data (and again
# on a dumped hard case, where only 1e-5 recovered the exact optimum), so any
# fair comparison against another subproblem solver should set this to 1e-5.
_QED_OTR_RED_FACTOR = float(os.environ.get("QED_OTR_RED_FACTOR", "1e-3"))
# Imported when EITHER OTR knob is on.  QED_OTR_TR_QN is read from the
# environment directly here rather than by name because it is defined further
# down, with the other quasi-Newton knobs; without this the QN sites raise
# NameError: _OtrSettings when QED_OTR_TR_QN=1 and QED_OTR_TR=0.
if _QED_OTR_TR or int(os.environ.get("QED_OTR_TR_QN", "0")):
    from otr_trust_region import (
        TrustRegionSettings as _OtrSettings,
        solve_trust_region_subproblem as _otr_solve,
    )

#   QED_AH_KRYLOV : solve the trust-region subproblem with the Krylov-seeded
#                   bordered augmented Hessian                    (default 0)
#
# A second leading branch in the same microiteration_optimization6 dispatch as
# QED_OTR_TR, ported from casscf1's Davidson_augmented_hessian_solve3.  Same
# family as OpenTrustRegion -- bordered pencil with the corner pinned at 0 and
# the gradient border scaled by alpha, step recovered as y[1:]/y0/alpha, the
# alpha search run INSIDE the projected problem at zero matvec cost, and the
# space grown by a level-shifted diagonally preconditioned residual -- but it
# keeps three things OpenTrustRegion does not:
#   1. when the trust radius is unreachable in the current subspace it takes
#      the largest well-defined step and EXPANDS (OTR returns a zero step and
#      makes the caller shrink the trust radius);
#   2. it scores every iterate by the model value q = g.x + x.Hx/2 and returns
#      the best, because the boundary step is recovered as the limit y0 -> 0+
#      and a subspace that goes on to contain the critical direction exactly
#      LOSES it;
#   3. no random trial vector, so it is deterministic.
# Measured on H2O cc-pVTZ CAS(8,6) it matches OTR's Hessian-vector product
# count exactly (419) and beats it on wall time; on the LiH 6-31G CAS(2,3) hard
# case that hangs the original prototype it takes 718 against OTR's 778.
# DEFAULT OFF, same caveat as QED_OTR_TR: turning it on changes the steps
# taken, so energies agree to the convergence threshold, not to machine
# precision.
_QED_AH_KRYLOV = int(os.environ.get("QED_AH_KRYLOV", "0"))
#   QED_AH_KRYLOV_QN : same solver at the two QUASI-NEWTON trust-region sites
# (gltr_qn, on the frozen U_zero/A_tilde_zero operators, and gltr_qn_bfgs, on
# the BFGS-updated operator via get_bfgs_mv).  Separate knob from
# QED_AH_KRYLOV because these are a different subproblem: gltr_qn_bfgs solves
# an APPROXIMATE Hessian exactly, which is why that site forces hard_case = 0.
# They were the two largest remaining solver sites after QED_AH_KRYLOV covered
# the main dispatch (83 + 173 of c10h8's remaining 557 matvecs).
#   MEASURED, DEFAULT STAYS OFF: on MgH+ this LOSES on both active spaces --
# CAS(6,6) +4.6% total matvecs (better on 4/14, p=0.16), CAS(8,12) +14.9%
# (better on 0/14, p=0.0001) with QN-site matvecs themselves up 25.3%.  There is
# no active space where it has been shown to win.
_QED_AH_KRYLOV_QN = int(os.environ.get("QED_AH_KRYLOV_QN", "0"))
#   QED_OTR_TR_QN : OTR at those same two quasi-Newton sites.  The gap this
# fills: QED_OTR_TR only ever covered the MAIN dispatch, so the QN sites were a
# strict two-way choice between GLTR (default) and AH-Krylov.  That is backwards
# relative to the evidence -- at the main site OTR beats AH-Krylov by 40.6% in
# matvecs (paired p=2.2e-10, 36/40 seeds) -- so the better solver was the one
# that could not run there.  Separate knob from QED_OTR_TR for the same reason
# QED_AH_KRYLOV_QN is separate: the qn-bfgs site solves an APPROXIMATE Hessian.
#   MEASURED, AND THE RESULT SPLITS BY ACTIVE SPACE -- DEFAULT STAYS OFF.
# The knob does what it was built to do: it cuts matvecs AT THE QN SITES on both
# spaces (CAS(8,12) -34.4%).  Whether that survives to the total does not follow.
#   CAS(6,6), 14 seeds: total matvecs -9.1%, better on 11/14, p=0.0049 (and
# -13.0% vs QED_AH_KRYLOV_QN, better on 13/14).  A real win.
#   CAS(8,12), 14 seeds: total matvecs +21.0%, better on only 2/14, p=0.0151,
# and MACROITERATIONS BLOW UP 14.86 -> 123.21 (sd 4.28 -> 21.31, range 65-150).
# All 14 converged -- this is not an iteration cap -- so the QN-site saving is
# not merely handed back elsewhere, it is handed back with interest: cheaper QN
# steps are worse steps, and the outer loop pays 8.3x for them.  Worst |dE| vs
# base is 3.34e-8, ~3x the ~1e-8 bar used elsewhere; with 123 macroiterations
# that is consistent with accumulated tolerance rather than a different
# stationary point, but it was not verified to be.
#   The CAS(8,12) hard case (item 149) is the discriminator, and it is the case
# that matters: do NOT promote this knob on the CAS(6,6) number alone.
_QED_OTR_TR_QN = int(os.environ.get("QED_OTR_TR_QN", "0"))
#   QED_EXACT_TR : materialize the MAIN-SITE orbital Hessian densely (n =
# index_map_size unit-vector products, batched) and solve the trust-region
# subproblem EXACTLY by More-Sorensen -- eigendecomposition plus bisection on
# the secular equation, with an explicit hard-case branch.  DEFAULT 0/OFF.
#   WHY.  Every solver tried so far at this site -- GLTR (default), AH-Krylov,
# OTR -- is a Krylov method seeded by g, and on MgH+ CAS(8,12) the Krylov space
# built from g NEVER contains v_min: measured overlap of K_m(H, g) with v_min is
# 0.000 at m=5, m=20 AND at m=60 of n=150, in 11 of the 15 dumped main-site
# subproblems.  That is one structural blind spot, not four independent
# failures, and it is why those three knobs all landed in the same place.
# QED_AH_NEG_CURV (default 1) is the existing remedy and only half-covers it:
# its seed e_argmin(h_diag) is a UNIT VECTOR, near-perfect where v_min is one
# coordinate (participation ratio 1.0 -> overlap 0.99) and ORTHOGONAL where
# v_min is delocalized (PR 4.5-8.7 -> overlap <= 0.021, 7 of 15, and those are
# the high-kappa ones).  Seeding with random-start Lanczos instead was measured
# and does NOT work -- Rayleigh-Ritz overlap stalls at 0.02-0.25 even at m=30,
# the low spectrum is too clustered.  A dense exact solve sidesteps the whole
# question: it never builds a Krylov space.
#   SCOPE IS DELIBERATELY THE MAIN SITE ONLY.  Of the 187 subproblem solves in
# one dumped CAS(8,12) run, only the 15 main-site ones are hard: they are 80%
# indefinite, 87% boundary-active, median |ghat.vmin| 2.8e-6, and the shipped
# GLTR step sits a median 16.2% (worst 49.6%) above the exact model optimum.
# The 89 qn_gltr solves are 2% indefinite / 3% boundary with |ghat.vmin| 0.44
# and GLTR already solves them to a 0.0% median gap; internal_lstrs (n=24) is
# 0% indefinite.  Making those dense would only add cost.
#   *** PRE-REGISTERED PREDICTION: THIS KNOB MAY WELL LOSE, AND THE EVIDENCE
# ALREADY POINTS THAT WAY.  Model-optimality is ANTI-correlated with outer-loop
# performance at this site: GLTR ships a 16.2%-suboptimal step and takes the
# FEWEST macroiterations, while QED_OTR_TR_QN solved its subproblems better and
# drove macroiterations 14.86 -> 123.21.  An exact solve is the logical endpoint
# of "solve the subproblem harder", which is the family that has already failed
# four times.  The reason to run it anyway is that it is the DECISIVE
# discriminator: it separates "the hard case is what hurts" from "chasing the
# model optimum is itself what hurts", and no cheaper experiment does that.
# Score on CAS(8,12) FIRST, on macroiteration count and scatter, not on
# subproblem quality. ***
#   Steps change, so energies agree to the convergence threshold, NOT to machine
# precision -- same caveat as QED_OTR_TR.
_QED_EXACT_TR = int(os.environ.get("QED_EXACT_TR", "0"))
#   QED_EXACT_TR_BATCH : how many unit vectors to push through orbital_sigma3
# per call when materializing.  The matvec COUNT is n either way; this is a wall
# time knob only, and QED_MV_BENCH shows per-call cost is nearly flat in the
# batch size, so the dense build costs far less wall time than its count implies.
_QED_EXACT_TR_BATCH = int(os.environ.get("QED_EXACT_TR_BATCH", "32"))
#   QED_ADAPTIVE_TOL : tie the GLTR exit tolerance to the observed model
# fidelity rho = (actual energy change)/(predicted energy change).
#   WHAT tol ACTUALLY IS, because the name "tolerance" oversells it.  GLTR exits
# when  be * |y[-1]| < tol  (:18424 interior, :18533 boundary), and by the
# Lanczos identity that quantity IS the residual norm of the linear system being
# solved in the subspace:  ||(H + lambda I) s + g||  -- exactly, not estimated
# (verified numerically against the dumped Hessians: ratio 1.000000 at every
# iteration).  lambda is 0 on the interior branch, the trust-region multiplier
# on the boundary branch.
#   THREE THINGS tol DOES NOT CONTROL, and they matter more than the one it
# does: (i) the accuracy of lambda itself, which comes from a separate inner
# bisection on the tridiagonal problem; (ii) the KRYLOV SUBSPACE RESTRICTION --
# the gap between the best step in the subspace and the true full-space
# trust-region solution, which is the DOMINANT error here (median 16.2% of model
# value); (iii) anything about the hard case, where the Krylov space built from
# g never contains v_min at all (measured overlap 0.000 even at m=60 of n=150),
# so no value of tol recovers that direction.
#   THE ONE INDIRECT EFFECT THAT DOES DRIVE THE RESULTS: because the test
# decides when Lanczos STOPS, tol controls the SUBSPACE DIMENSION.  Tightening
# it therefore does improve the step in the full-space sense -- by building a
# bigger Krylov space -- up to the ceiling of what that space can span.  That is
# the mechanism behind the floor sweep below, where a lower floor cut
# macroiterations monotonically while raising matvecs per solve 87.7 -> 108.0.
#   NOTE this knob has no effect under QED_EXACT_TR: _more_sorensen_dense is
# exact and takes no tolerance.
# DEFAULT 0/OFF.  0 = off (tol stays the hard-coded 1e-7 everywhere, path
# unchanged); 1 = adapt at the MAIN site only; 2 = also adapt at the two QN
# sites.  This is the DYNAMIC counterpart of QED_GLTR_TOL_SCALE, which applies a
# single static multiplier; the two compose (solve_gltr_trust_region does
# tol = tol * _QED_GLTR_TOL_SCALE internally).
#   *** SCORED ON MgH+ CAS(8,12) AND IT LOSES.  THIS BLOCK IS NOW A RECORD OF A
# NEGATIVE RESULT, NOT ADVOCACY.  DEFAULT STAYS 0/OFF. ***
# 14 seeds x {base, 1, 2}, 42/42 converged, worst |dE| 1.2e-9 (same stationary
# point, item 139 holds).  MACROITERATIONS: base 14.86 +-4.28 | =1 15.21 +-3.81
# (p=0.46) | =2 15.57 +-4.86 (p=0.22) -- FLAT, no win in either direction.
# MATVECS: +3.6% (1/14 better, p=0.0007) and +9.2% (2/14, p=0.0099) -- a small
# but statistically solid LOSS.  Base arm reproduced the QED_EXACT_TR wave's
# base BIT-EXACTLY on all 14 seeds, so the comparison is sound.
#   WHY IT FAILED, AND IT IS THE INTERESTING PART: rho IS THE WRONG CONTROLLER
# INPUT.  Across 4085 adaptation events rho sat at 0.63-1.13 (overwhelmingly
# 0.8-1.0) on the system whose outer loop is the WORST behaved in the whole
# study (sd 4.28, bistable subproblems, |Pg|/|g| ~1e-6).  The rule therefore
# read "model is trustworthy" almost everywhere and TIGHTENED monotonically to
# the 1e-11 floor: at =1 the loosening branch fired ONCE in 782 events (31% of
# events pinned at the floor), so =1 is in practice just "tighten to the floor"
# -- which won on CAS(6,6) and loses here.  At =2 it thrashes between both
# bounds: 65% at the floor, 13% at the 1e-4 ceiling, 78% AT A BOUND, i.e. THE
# BOUNDS ARE MAKING THE DECISION, NOT rho -- the exact failure signature that
# was pre-registered for this wave.
#   THE TRANSFERABLE LESSON: rho ~ 1 does not mean the model is good, it means
# the step was small enough that model and reality agree locally.  It cannot see
# the error that actually limits this problem -- the Krylov subspace restriction
# (iii) above, median 16.2% of model value, which is invisible to rho because
# both the model and the realized energy are evaluated at the SAME restricted
# step.  ANY FUTURE CONTROLLER MUST KEY ON SOMETHING THAT SEES SUBSPACE
# QUALITY (exit reason, Krylov dimension, boundary-vs-interior), NOT ON rho.
#   The design reasoning below is retained because it remains correct about the
# mechanism and about why "solve harder" fails; it was only wrong about rho
# being a usable proxy for model quality.
#   THE ORIGINAL ARGUMENT.  Five attempts to
# make the outer loop faster by solving the subproblem BETTER have now failed
# (act-virt shift; OTR/AH at the main site; OTR/AH at the QN sites; and the
# dense exact solve, which on CAS(6,6) reached the SAME energy in 13
# macroiterations instead of 10).  The reason is that outer progress is limited
# by MODEL ERROR, not by subproblem error:
#   - trust-region global convergence needs only a FRACTION of Cauchy decrease,
#     which GLTR satisfies trivially, so exactness buys nothing there;
#   - superlinear local convergence needs the step to approach the Newton step,
#     which both GLTR and an exact solve reach once the region goes inactive;
#   - in between, the quadratic model is only good to about +-20% here
#     (rho p10-p90 = 0.79-1.04 on CAS(6,6)), while GLTR's step sits a median
#     16.2% above the exact model optimum.  THE GAP BEING CLOSED IS SMALLER
#     THAN THE MODEL'S OWN ERROR, so closing it cannot pay -- and measurably
#     does not: the exact solver's steps have WORSE fidelity (median rho
#     0.967 -> 0.915, fraction with rho < 0 rising 0.01 -> 0.05, and 3x more
#     uphill microiteration trials).
#   So the lever is not "solve harder", it is "solve exactly as hard as the
# model deserves".  When rho sits near 1 the model is trustworthy and extra
# subproblem accuracy converts into real energy decrease, so tighten.  When rho
# is poor the model is not worth resolving; loosen and let step_control shrink
# the radius instead -- that is the mechanism that actually fixes a bad model.
#   Deliberately conservative: multiplicative moves with a hysteresis band, hard
# bounds, and an immediate loosen on any rho < 0 (a step that moved the energy
# the WRONG WAY is the strongest evidence the model is not resolvable).
_QED_ADAPTIVE_TOL = int(os.environ.get("QED_ADAPTIVE_TOL", "0"))
#   Reference/starting tolerance; 1e-7 reproduces the hard-coded value.
_QED_ADAPTIVE_TOL_BASE = float(os.environ.get("QED_ADAPTIVE_TOL_BASE", "1e-7"))
#   Hard bounds.  MIN stops it chasing accuracy the arithmetic cannot deliver;
# MAX stops it loosening so far that the step stops being a descent direction.
#   MIN WAS 1e-9 AND THAT WAS TOO HIGH -- it BOUND 65-78% of calls, i.e. the
# bound was making the decision instead of rho.  Swept on CAS(6,6), 4 seeds
# (3/5/7/11), macroiteration mean / matvec mean vs the fixed-1e-7 base:
#     1e-9    12.50  -0.2%   (floor binding 34-78% of calls)
#     1e-11   12.00  -2.6%   (binding 12-63%)
#     1e-13   11.25  -3.4%   (binding  2-54%)
#   base      12.75     -
# **Macroiterations fall MONOTONICALLY in every seed as the floor drops**
# (s3 10->9->9->9, s5 14->14->13->12, s7 13->13->13->12, s11 14->14->13->12),
# which is the trend worth trusting; the matvec column is noisy at n=4 (s11
# -9.4% vs s3 +7.0%) and is NOT statistically supported.  MECHANISM, measured on
# s3's main site: the solve COUNT is 55 at every floor while matvecs per solve
# rise 87.7 -> 98.0 -> 104.1 -> 108.0, so a lower floor buys better steps and
# pays in per-solve cost; it wins exactly when the macroiteration saving covers
# that.  SAFE AT ALL FLOORS TESTED: no solve at any floor terminated on the
# iteration cap (every one exited Boundary / Interior (Newton) / hard_case).
#   1e-11 chosen over the better-on-average 1e-13 because n=4 does not justify
# the aggressive setting: 1e-13's worst seed is +7.0% vs 1e-11's +3.7%.
#   THE LARGER WAVE DID NOT CONFIRM THE TREND: on CAS(8,12) (14 seeds, above)
# the =1 arm spends 31% of its calls pinned AT this 1e-11 floor and still loses
# +3.6% matvecs at flat macroiterations, so lowering the floor to 1e-13 is NOT
# indicated -- the CAS(6,6) sweep does not generalize.  Do not re-run that sweep
# expecting a different answer; the binding issue is the controller, not the
# floor.
_QED_ADAPTIVE_TOL_MIN = float(os.environ.get("QED_ADAPTIVE_TOL_MIN", "1e-11"))
_QED_ADAPTIVE_TOL_MAX = float(os.environ.get("QED_ADAPTIVE_TOL_MAX", "1e-4"))
#   How many recent rho values the median is taken over.  Short enough to react
# within one macroiteration (the main site is dispatched ~21x per macro).
_QED_ADAPTIVE_TOL_WINDOW = int(os.environ.get("QED_ADAPTIVE_TOL_WINDOW", "5"))
#   Hysteresis band on |1 - median(rho)|: below TIGHT -> tighten, above LOOSE ->
# loosen, in between -> hold.  TIGHT=0.05 and LOOSE=0.25 bracket the measured
# rho spread (p10-p90 = 0.79-1.04, i.e. |1-rho| ~ 0.21 at the p10 edge).
_QED_ADAPTIVE_TOL_TIGHT = float(os.environ.get("QED_ADAPTIVE_TOL_TIGHT", "0.05"))
_QED_ADAPTIVE_TOL_LOOSE = float(os.environ.get("QED_ADAPTIVE_TOL_LOOSE", "0.25"))
#   Multiplicative step per adaptation.
_QED_ADAPTIVE_TOL_FACTOR = float(os.environ.get("QED_ADAPTIVE_TOL_FACTOR", "3.0"))
#
#   QED_SUBSPACE_TOL : tie the GLTR exit tolerance to the observed SUBPROBLEM
#                      EXIT REASON                              (default 0 = off)
#
# WHY THIS EXISTS, AND WHY IT IS NOT QED_ADAPTIVE_TOL AGAIN.  QED_ADAPTIVE_TOL
# keyed the same tolerance on rho and LOST on MgH+ CAS(8,12) (macroiterations
# flat, matvecs +3.6% / +9.2%).  The post-mortem is the reason for this knob:
# rho ~ 1 does NOT mean the model is good, it means the STEP WAS SHORT ENOUGH
# that model and reality agree locally.  rho is structurally blind to the error
# that actually limits this problem -- the KRYLOV SUBSPACE RESTRICTION -- because
# the model value and the realized energy are evaluated at the SAME restricted
# step.  Over 4085 adaptation events rho sat at 0.63-1.13 on the worst-behaved
# outer loop in the study, so the rule read "trustworthy" nearly everywhere and
# just walked to the floor.  ANY CONTROLLER ON THIS TOLERANCE MUST KEY ON
# SOMETHING THAT SEES SUBSPACE/BOUNDARY STRUCTURE.  The GLTR exit reason is
# exactly that, and it is already computed and returned -- it costs nothing.
#
# THE RULE, IN ONE LINE: LOOSEN WHERE THE TRUST-REGION BOUNDARY IS NOT BINDING.
# On an "Interior (Newton)" exit GLTR stopped strictly inside the trust region,
# so tol governs only how accurately the unconstrained Newton step is resolved
# inside an already-chosen subspace -- not where the step is placed.  On a
# "Boundary" exit tol interacts with the lambda/boundary placement, so the same
# loosening is NOT free.  The controller therefore loosens while a site's recent
# exits are interior-dominated and snaps back toward base as soon as boundary
# exits reappear.
#
# THE OBSERVABLE IS STABLE UNDER THE ACTION, WHICH IS THE PROPERTY rho LACKED.
# Measured on c10h8 across QED_GLTR_TOL_SCALE = 1/10/100/1000: the per-site exit
# distribution is BIT-IDENTICAL in all four arms (gltr 1B/3I, gltr_qn 1B/10I,
# gltr_qn_bfgs 8B/9I).  So the controller cannot chase its own tail -- it reads
# the same signal after acting on it.  rho, by contrast, is a function of the
# step the tolerance produced.
#
# WHAT IT DISCRIMINATES (this is the whole point, and it is measured, not
# assumed).  Fraction of "Interior (Newton)" exits at the MAIN gltr site:
#     c10h8 CAS(10,10)      3 interior /  1 boundary  = 0.75
#     MgH+ CAS(8,12)     8-15 interior / 44-47 bndry  = 0.15-0.25  (5 base seeds)
# i.e. it loosens on the system where loosening was measured to be free and
# stays at base on the system where the boundary is doing the work.  Note the QN
# sites on MgH+ are 100% interior with ZERO boundary exits in every seed
# inspected -- an independent corroboration that the trust region is inert
# there.
#
#   0 = off; _subproblem_tol returns the hard-coded 1e-7 and every call site is
#       byte-for-byte unchanged.
#   1 = adapt the MAIN gltr site only.  Conservative: on MgH+ this is close to a
#       no-op because the main site is boundary-dominated.  On c10h8 it captures
#       only the main site's 43 matvecs of the 295 the GLTR family spends.
#   2 = adapt the main site AND both QN sites (gltr_qn, gltr_qn_bfgs).  This is
#       the full mechanism: on c10h8 the QN sites are 252 of those 295 matvecs.
# Mirrors QED_ADAPTIVE_TOL's 1/2 site split so the two are directly comparable.
_QED_SUBSPACE_TOL = int(os.environ.get("QED_SUBSPACE_TOL", "0"))
#   Tolerance the controller starts each site from; the 1e-7 the call sites
# hard-code, so the first solve at every site is unchanged.
_QED_SUBSPACE_TOL_BASE = float(os.environ.get("QED_SUBSPACE_TOL_BASE", "1e-7"))
#   Bounds.  THE FLOOR DEFAULTS TO BASE, i.e. THE CONTROLLER CAN ONLY LOOSEN OR
# RETURN TO BASE, NEVER TIGHTEN BELOW IT.  Deliberate: "tighten the subproblem"
# is the closed mechanism family (0 for 5 on CAS(8,12), and QED_ADAPTIVE_TOL's
# adap1 was in practice just "tighten to the floor" and lost).  There is no
# evidence tightening below 1e-7 ever helps, so the default design cannot try.
# Set _MIN below 1e-7 explicitly if you want to re-open that question.
# The ceiling is 1e-4, which is the loosest setting measured on c10h8
# (QED_GLTR_TOL_SCALE=1000 x 1e-7) -- past that is unmeasured territory: the
# failure mode when the tolerance is too loose has NEVER been observed.
_QED_SUBSPACE_TOL_MIN = float(os.environ.get("QED_SUBSPACE_TOL_MIN", "1e-7"))
_QED_SUBSPACE_TOL_MAX = float(os.environ.get("QED_SUBSPACE_TOL_MAX", "1e-4"))
#   How many of a site's most recent exits the fraction is taken over.  Short
# because the sites are solve-poor on easy systems: c10h8 runs only 4 / 11 / 17
# solves at gltr / gltr_qn / gltr_qn_bfgs for the WHOLE optimization, so a long
# window would leave the run over before the controller ever moved.
_QED_SUBSPACE_TOL_WINDOW = int(os.environ.get("QED_SUBSPACE_TOL_WINDOW", "4"))
#   Hysteresis band on the interior fraction: at or above HI -> loosen, at or
# below LO -> snap back toward base, in between -> hold.  HI=0.75 and LO=0.25
# bracket the measured separation above (c10h8 0.75 vs MgH+ 0.15-0.25) -- they
# are the edges of that gap, NOT tuned inside it.
_QED_SUBSPACE_TOL_HI = float(os.environ.get("QED_SUBSPACE_TOL_HI", "0.75"))
_QED_SUBSPACE_TOL_LO = float(os.environ.get("QED_SUBSPACE_TOL_LO", "0.25"))
#   Multiplicative step per adaptation.  10, not QED_ADAPTIVE_TOL's 3: the
# useful range here is the three decades 1e-7 -> 1e-4 and the sites see only
# 4-17 solves, so a factor of 3 could not cross it before the run ended.  This
# is a consequence of the solve counts above, not a free parameter.
_QED_SUBSPACE_TOL_FACTOR = float(os.environ.get("QED_SUBSPACE_TOL_FACTOR", "10.0"))
#   A single boundary exit in the last _RECENT solves blocks loosening outright,
# whatever the windowed fraction says.  The interior/boundary distinction is a
# property of the NEXT solve, which the fraction only predicts; this makes the
# controller react to a boundary exit immediately instead of waiting for it to
# dominate the window.  Set 0 to disable the veto and use the fraction alone.
_QED_SUBSPACE_TOL_RECENT = int(os.environ.get("QED_SUBSPACE_TOL_RECENT", "2"))
#
#   QED_EXIT_CONTROL : tie the TRUST-RADIUS UPDATE to the observed subproblem
#                      EXIT REASON                          (default 0 = off)
#
# SAME OBSERVABLE AS QED_SUBSPACE_TOL, DIFFERENT ACTION -- and the action is the
# point.  QED_SUBSPACE_TOL loosens the exit TOLERANCE where the boundary is not
# binding, which by its own logic has nothing to offer where the boundary IS
# binding.  On MgH+ CAS(8,12) the main gltr site is 44-47 boundary vs 8-15
# interior exits, i.e. the boundary is binding almost everywhere, and that is
# exactly where the tail pathology lives (items 135-141: near-empty active
# orbitals -> flat virt-act directions -> indefinite, near-hard-case
# subproblems).  So the tolerance controller is structurally the wrong tool
# there and this one takes the only action left.
#
# READ THIS BEFORE EXTENDING IT -- MOST OF THE ACTION SPACE IS ALREADY DEAD.
# Three actions can be keyed on the exit reason.  Two are already falsified:
#   * STEP ACCEPTANCE: there is nothing to key on.  MgH+ CAS(8,12) base runs
#     have ZERO rejections (grep -ic reject == 0 in every base log inspected),
#     so an acceptance rule could only manufacture rejections that do not exist.
#   * KRYLOV DIMENSION: QED_EXACT_TR is the endpoint of that axis -- the full
#     space, solved exactly -- and it LOST (item 179: 14 seeds, macroiterations
#     dead flat, and the steps were PROVABLY better, KKT residual <= 7e-14 and
#     model value <= GLTR's on 15/15 dumped subproblems).  "More subspace" is
#     falsified directly, not by proxy.
#   * TRUST RADIUS: falsified in the GROWTH direction only.  QED_TR_MAX raised
#     the cap to 1.0/3.0, the radius genuinely woke up (rejections/run 0.1 ->
#     0.8, radius climbing 0.86/1.04/1.24/1.49/1.79/2.15/2.58 instead of pinning
#     at 0.75) and the outer loop DID NOT CARE.
# What is left, and all this knob does, is the SHRINK direction -- untried.
#
# THE MECHANISM CLAIM, STATED SO IT CAN BE FALSIFIED.  A boundary exit means the
# model's own minimizer lies OUTSIDE the region: the step was placed by the
# constraint, not by the model.  When that happens persistently the step
# direction is dominated by the flat/indefinite directions the quadratic model
# cannot vouch for -- and rho CANNOT SEE IT, because rho is evaluated on the
# same constrained step (item 182: rho sat at 0.63-1.13 over 4085 events on the
# worst-behaved loop in the study, and on MgH+ ratio > 0.75 at essentially every
# step with zero rejections).  So: SHRINK THE REGION UNTIL THE MODEL'S OWN
# MINIMIZER IS INTERIOR, i.e. drive the loop toward exits where the step is the
# model's genuine unconstrained minimizer rather than a point on a fence.
# The rule self-terminates: once exits go interior the damping stops firing and
# the ordinary 1.2x growth resumes.
#
# HONEST EXPECTATION.  This makes steps SHORTER, and item 184 already showed the
# outer loop is indifferent to step LENGTH in the growth direction.  The reason
# to run it anyway is that shrinking is not the same experiment as growing: the
# claim is about step DIRECTION QUALITY (interior vs constrained), with length
# only the lever available to change it.  If it fails, the "key the trust region
# on subspace-quality observables" family is closed in both directions and the
# honest conclusion is that the CAS(8,12) tail is a property of the badly-posed
# active space (item 135), not of the orbital-optimization solver.
#
#   0 = off; step_control is byte-for-byte the original function.
#   1 = DAMP on persistent boundary exits (the untried direction).
#   2 = GATE GROWTH on the exit reason: never widen the region after an
#       Interior exit, because an interior step tells you nothing about whether
#       a larger Delta would still be trustworthy.  This is the exit-reason-
#       native form of QED_STEP_CONTROL (a), which implements the same idea via
#       the step-norm proxy ||s|| >= 0.8*Delta.  On c10h8 that proxy and this
#       are near-equivalent (43 of 46 growth events were on Interior steps); on
#       MgH+ they are NOT, because boundary steps have ||s|| == Delta and so
#       always pass the proxy.  Mode 2 is therefore close to inert on MgH+ by
#       construction -- it is here for c10h8 and as a control.
#   3 = both.
_QED_EXIT_CONTROL = int(os.environ.get("QED_EXIT_CONTROL", "0"))
#   Multiplicative shrink per damping event.  0.7 deliberately reuses the factor
# the existing ratio<=0.25 branch already uses, so the damping is not a new
# magnitude, only a new TRIGGER.
_QED_EXIT_CONTROL_DAMP = float(os.environ.get("QED_EXIT_CONTROL_DAMP", "0.7"))
#   Floor under the damping.  Without it a persistently boundary-terminating
# site would drive the radius to zero and stall the outer loop; 0.05 is an
# order of magnitude below the smallest accepted radius measured on c10h8
# (0.031-0.25), so it bounds the damage without being reachable in normal use.
_QED_EXIT_CONTROL_MIN = float(os.environ.get("QED_EXIT_CONTROL_MIN", "0.05"))
#   Window and threshold for "persistently boundary-terminating".  Same window
# and same 0.25 as QED_SUBSPACE_TOL, and for the same reason: 0.25 is the low
# edge of the MEASURED separation between the two systems (MgH+ main site
# 0.15-0.25 interior, c10h8 main site 0.75), not a fitted value.
_QED_EXIT_CONTROL_WINDOW = int(os.environ.get("QED_EXIT_CONTROL_WINDOW", "4"))
_QED_EXIT_CONTROL_LO = float(os.environ.get("QED_EXIT_CONTROL_LO", "0.25"))
#   QED_AH_RESID_TOL : RELATIVE exit tolerance on the optimality residual,
# ||(H - theta)x + g|| / |g|.  NOT comparable to the 1e-7 that
# solve_gltr_trust_region is called with: GLTR tests be*|y[-1]| < tol
# ABSOLUTELY, never scaled by |g|.
#   QED_AH_RESID_FLOOR : absolute floor under that relative test, matching
# OpenTrustRegion's RESIDUAL_NORM_FLOOR.  It matters because |g| spans some
# five orders of magnitude over an optimization, so a bare relative tolerance
# eventually demands an absolute residual below what the arithmetic can
# deliver, and the solve can then only exit on expansion collapse.
#   QED_AH_NEWTON : take OpenTrustRegion's reduced-space Newton step (mu = 0)
# when the projected Hessian is positive definite and that step is interior,
# instead of bisecting for a level shift.
# On H2O cc-pVTZ the floor alone is worth 555 -> 476 matvecs, the Newton
# shortcut alone 555 -> 489, and the two together 555 -> 419.
_AH_RESID_TOL = float(os.environ.get("QED_AH_RESID_TOL", "1e-5"))
_AH_RESID_FLOOR = float(os.environ.get("QED_AH_RESID_FLOOR", "1e-12"))
_AH_NEWTON = int(os.environ.get("QED_AH_NEWTON", "1"))
_AH_NEWTON_THRESH = float(os.environ.get("QED_AH_NEWTON_THRESH", "-1e-5"))
#   QED_AH_NEG_CURV : add e_argmin(h_diag) to the seed when that diagonal is
# negative -- the negative-curvature trial vector OpenTrustRegion also uses
# (otr_trust_region.py:276-282).  DEFAULT 1, and it is NOT optional in spirit:
# without it the seed spans only the Krylov space of g, so when the gradient is
# nearly orthogonal to the lowest eigenvector -- the HARD CASE -- the subspace
# can never represent the tau*v_min part of the solution, and the solver
# converges on its optimality residual to a step with theta > lambda_min, i.e.
# a NON-GLOBAL stationary point, while reporting success.  Measured on 15
# dumped MgH+ subproblems (n=150): without it, 2 of 15 land 45.0%% and 8.2%%
# above the exact More-Sorensen model value; with it, 0 of 15 exceed 0.002%%
# AND the total Hessian-vector count DROPS 581 -> 503, because the near-hard
# cases stop grinding (80 -> 44 and 72 -> 41 matvecs).  Random trial vectors do
# NOT substitute: 1 or even 3 of them leave both failures bit-unchanged, since
# the residual test is met on the wrong branch before they can be amplified.
_AH_NEG_CURV = int(os.environ.get("QED_AH_NEG_CURV", "1"))
_AH_Y0_TOL = float(os.environ.get("QED_AH_Y0_TOL", "1e-8"))
_AH_MAX_ITER = int(os.environ.get("QED_AH_MAX_ITER", "200"))
_AH_MAX_DIM = int(os.environ.get("QED_AH_MAX_DIM", "400"))
_AH_PRECOND_FLOOR = float(os.environ.get("QED_AH_PRECOND_FLOOR", "1e-12"))
_AH_BRACKET_STEPS = int(os.environ.get("QED_AH_BRACKET_STEPS", "14"))
_AH_BISECT_STEPS = int(os.environ.get("QED_AH_BISECT_STEPS", "60"))


# ---------------------------------------------------------------------------
# Occupation-scaled level shift on the ACT-VIRT rotation block.
#
#   QED_ACTVIRT_SHIFT       : 0/1, DEFAULT 0 (off).
#   QED_ACTVIRT_SHIFT_EPS   : numerator eps, default 1e-3.
#   QED_ACTVIRT_SHIFT_DELTA : occupation floor delta, default 1e-2.
#
# sigma_(t,a) = eps / (n_t + delta), added to the diagonal of every active(t) ->
# virtual(a) rotation, where n_t is that active orbital's occupation
# diag(D_tu_avg).  Zero on the act-inact and virt-inact blocks.
#
# WHY.  The orbital gradient for an act-virt rotation carries a factor of the
# active occupation through the generalized Fock matrix, so as n_t -> 0 BOTH the
# gradient and the corresponding Hessian block vanish together.  That direction
# is flat but NOT exactly redundant, so index_map (:3468-3481) cannot excise it
# the way it excises inact-inact / act-act / virt-virt.  Measured on MgH+
# CAS(8,12): 99.8%% of the weight of the flattest orbital-Hessian eigenvectors
# lands in the act-virt block and localizes on the active orbitals with
# occupations 0.004/0.001/0.000, giving kappa up to 9.3e7, 12 of 15 subproblems
# indefinite, 13 of 15 boundary-constrained, |Pg|/|g| on lambda_min ~ 1e-6, and
# 4 of 15 subproblems BISTABLE under a 1e-8 perturbation.
#
# The shift lifts exactly those directions away from zero (n_t = 0 -> eps/delta
# = 0.1 against a spectrum whose largest eigenvalue is ~100) while leaving
# genuinely correlated orbitals essentially untouched (n_t = 1.3 -> 7.6e-4).
#
# *** IT CANNOT MOVE THE ANSWER. *** The shift is added to the HESSIAN only,
# never to the gradient, so every stationary point of the energy is still a
# stationary point.  It changes the STEP, i.e. the path and the iteration count,
# not the converged solution.  test_actvirt_shift.py asserts exactly that.
#
# Applied in orbital_sigma3 (the single chokepoint every solver's Hessian-vector
# product goes through -- GLTR, LSTRS, PCG, AH-Krylov and the QN sites all route
# there) and in build_hessian_diagonal (so the preconditioner sees the SAME
# operator the matvec applies).
#
# *** MEASURED RESULT: THE SHIFT FAILS ITS EFFICACY TEST.  KEEP IT OFF. ***
# 14 seeds x {off, on} on the pathological case itself (MgH+ CAS(8,12), all 28
# arms converged, none maxiter-capped):
#
#       leg   mean macroiterations    sd     range
#       off          14.86           4.28     8-21
#       on           30.43          20.58     7-64
#
# The pre-registered rule was "earns its keep only if the SCATTER drops" (a
# single seed's count is nearly uninformative here).  sd went UP 4.8x, mean
# +15.6, paired t = +2.85 (df 13, p ~ 0.014), 9 seeds worse against 5 better.
# Correctness held exactly as argued above -- all 28 arms landed on the same
# stationary point, worst |dE| 2.4e-8 -- so the shift is SOUND but HARMFUL.
#
# LEADING HYPOTHESIS for why, NOT separately confirmed: with the default eps and
# delta an EMPTY active orbital gets sigma = 0.1, roughly 1e7 x the true local
# curvature (~1e-8) in exactly the directions the trust-region solver was using
# to escape.  Damping the negative curvature out of the flattest block replaces
# a boundary-length escape step with an interior step of order g/0.1, and the
# optimizer crawls.  Consistent with the blow-ups concentrating in the seeds
# that were boundary-constrained (+46, +52, +34, +32, +25).  If that is right
# the defect is directional, not a matter of tuning eps down, because a shift
# small enough to spare the negative curvature is also too small to lift the
# conditioning.  There is no cheap way to test a repair: CAS(8,12) costs hours
# per run and the well-conditioned small spaces cannot show efficacy at all.
#
# NOT IMPLEMENTED, AND NOW DISCOURAGED: the mirror pathology on the ACT-INACT
# block from nearly-DOUBLY-occupied active orbitals (n_t -> 2), which is what
# makes MgH+ CAS(4,4) worse than CAS(6,6).  Would want sigma ~ eps / (2 - n_t +
# delta) there.  Originally left out only to avoid confounding the two blocks in
# one experiment; after the result above, a mirror of a construction that lost
# 15 macroiterations on average is not worth building without a new argument.
_QED_ACTVIRT_SHIFT = int(os.environ.get("QED_ACTVIRT_SHIFT", "0"))
_QED_ACTVIRT_SHIFT_EPS = float(os.environ.get("QED_ACTVIRT_SHIFT_EPS", "1e-3"))
_QED_ACTVIRT_SHIFT_DELTA = float(os.environ.get("QED_ACTVIRT_SHIFT_DELTA", "1e-2"))


# ---------------------------------------------------------------------------
# Packed two-electron integrals.  self.twoeint is stored as (npair, npair) with
# npair = nmo(nmo+1)/2, folding BOTH the bra and the ket pair onto p >= q.
# 8.40 GB -> 2.12 GB at c10h8 dims.  Measured (unpack_probe.c, real shapes):
# unpacking a row into a thread-local nmo x nmo scratch and running the same
# dgemm is ~1.4x FASTER than the contiguous full-row version, because
# orbital.c's first transformation step is bandwidth-bound and this halves the
# DRAM read -- the unpack itself lands in L2 and costs nothing.
#
# The pair ordering is the one index_map_pq already uses (:3221-3224): the
# SMALLER index runs in the outer loop,
#     for q in range(nmo): for p in range(q, nmo): ...
# so packed row `pq_up` is exactly the pair index_map_pq[pq_up] names.  Both
# the Python unpackers and orbital.c's must enumerate ket pairs in this same
# order; note this is NOT np.tril_indices' order, which is larger-index-outer.
# ---------------------------------------------------------------------------
def _pair_index_table(nmo):
    """(nmo, nmo) int32 table: tbl[a, b] = packed index of the unordered pair."""
    tbl = np.zeros((nmo, nmo), dtype=np.int32)
    idx = 0
    for q in range(nmo):
        for p in range(q, nmo):
            tbl[p, q] = idx
            tbl[q, p] = idx
            idx += 1
    return tbl


def _pair_lists(nmo):
    """(pi, qi) with pi >= qi, the packed pair order; pair PQ is (pi[PQ], qi[PQ])."""
    qi = np.concatenate([np.full(nmo - q, q, dtype=np.intp) for q in range(nmo)])
    pi = np.concatenate([np.arange(q, nmo, dtype=np.intp) for q in range(nmo)])
    return pi, qi


import numba as nb
from scipy.linalg import eigh, cho_factor, cho_solve, solve_triangular
# The dsymv symbol binding and _symv_accumulate MOVED to orbital_sigma.py on
# 2026-09-04, together with the build_sigma_reduced7/_8 kernels that are their
# only reason to exist -- those are `nopython` and must resolve the helper as a
# compile-time global, which an import inside a function cannot provide.
# Imported back here for build_gradient, the one caller outside that module.
# _QED_G_SYMV is defined there too (the no-MKL fallback has to force it off);
# its documentation stays in the knob block above.
from orbital_sigma import (
    _symv_accumulate,
    _QED_G_SYMV,
    orbital_sigma3 as _os_orbital_sigma3,
    mv2 as _os_mv2,
    build_orbital_hessian_guess as _os_build_orbital_hessian_guess,
)


# QED_G_TILED's kernel.  out[s, l, r, k] = G[k0 + k, l0 + l, r, s], i.e. exactly
# one quadrant of G.transpose(3, 1, 2, 0), blocked over (s, k) -- the two axes
# that trade places -- so a tile of each fits in cache.  Loop order inside a tile
# is (l, r, s, k): r is the axis both sides step contiguously over, and it is
# innermost-but-two on purpose, because making it innermost costs the (s, k)
# blocking that the whole point of this is.  See the QED_G_TILED note above for
# the measurements.  No ctypes here, so cache=True is safe.
@nb.jit(
    "void(float64[:,:,:,::1], float64[:,:,:,::1], int64, int64, int64)",
    nopython=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def _transpose_G_block(G, out, l0, k0, TS):
    nmo = out.shape[0]
    nl = out.shape[1]
    nk = out.shape[3]
    n_s_tiles = (nmo + TS - 1) // TS
    n_k_tiles = (nk + TS - 1) // TS
    for s_tile in nb.prange(n_s_tiles):
        s_lo = s_tile * TS
        s_hi = min(s_lo + TS, nmo)
        for k_tile in range(n_k_tiles):
            k_lo = k_tile * TS
            k_hi = min(k_lo + TS, nk)
            for l in range(nl):
                for r in range(nmo):
                    for s in range(s_lo, s_hi):
                        for k in range(k_lo, k_hi):
                            out[s, l, r, k] = G[k0 + k, l0 + l, r, s]

# --- Optional validation instrumentation for the cpp_casscf port -----------
# When CPP_CASSCF_VALIDATION_DIR is set, a handful of trust-region solver
# call sites dump their real (Hessian, gradient, trust_radius, step)
# instances to disk so the C++ port (qed-ci/src/cpp_casscf) can be replayed
# against them and checked for an exact step match. No effect on behavior
# when the env var is unset.
_CPP_CASSCF_VALIDATION_DIR = os.environ.get("CPP_CASSCF_VALIDATION_DIR")
_cpp_casscf_validation_counters = {}


# --- Optional trust-region solver statistics (QED_SOLVER_STATS) -------------
class _SolverStats:
    """Counts solver work so it can be attributed per solve.

    ``matvecs`` counts Hessian-vector products (orbital_sigma3 counts
    ``num_states`` of them per call, since it builds that many at once; mv2
    builds one).  ``bordered`` counts dense bordered eigensolves
    (projection_step2), which is what an LSTRS iteration costs on the small
    internal block where no matrix-free product is involved.
    """

    class _Span:
        __slots__ = ("site", "mv0", "bd0", "tr0", "sec0")

        def __init__(self, site, mv0, bd0, tr0, sec0):
            self.site, self.mv0, self.bd0, self.tr0 = site, mv0, bd0, tr0
            self.sec0 = sec0

    def __init__(self):
        self.matvecs = 0
        self.bordered = 0
        self.records = []
        self.trace = []
        # Wall time actually spent building Hessian-vector products, and the
        # same split by BATCH SIZE (how many vectors one call built).  The
        # matvec COUNT alone cannot say whether a matvec percentage is a wall
        # percentage -- c10h8's wall clock is integral-transformation bound --
        # and a per-call fixed overhead would make 507 one-vector GLTR calls
        # cost far more than the same vectors built in batches.
        self.matvec_secs = 0.0
        self.mv_hist = {}
        # Block width available to the Jacobi-Davidson correction solve, i.e.
        # how many roots it could solve for in one batched sigma build.
        self.jd_widths = {}

    def jd_width(self, w):
        self.jd_widths[w] = self.jd_widths.get(w, 0) + 1

    def matvec(self, n=1, secs=0.0):
        self.matvecs += n
        self.matvec_secs += secs
        h = self.mv_hist.setdefault(n, [0, 0, 0.0])
        h[0] += 1
        h[1] += n
        h[2] += secs

    def bordered_solve(self):
        self.bordered += 1

    def trace_iter(self, resid, step_norm, tol):
        self.trace.append((resid, step_norm, tol))

    def begin(self, site):
        return self._Span(
            site, self.matvecs, self.bordered, len(self.trace), self.matvec_secs
        )

    def end(self, span, iters=None, status=None):
        if span is None:
            return
        if iters is None and isinstance(status, str):
            # GLTR reports its Lanczos count inside the status string, e.g.
            # "Boundary iter 7" / "Interior (Newton) iter 3".
            tail = status.rsplit(" ", 1)[-1]
            if tail.isdigit():
                iters = int(tail)
                status = status.rsplit(" ", 1)[0].rsplit(" iter", 1)[0]
        self.records.append(
            {
                "site": span.site,
                "iters": iters,
                "status": status,
                "matvecs": self.matvecs - span.mv0,
                "mv_secs": self.matvec_secs - span.sec0,
                "bordered": self.bordered - span.bd0,
                "trace": self.trace[span.tr0 :],
            }
        )
        del self.trace[span.tr0 :]

    def report(self, extra=None):
        if not self.records:
            print("\n[solver-stats] no solves recorded")
            return
        by = {}
        for r in self.records:
            b = by.setdefault(r["site"], [])
            b.append(r)
        print("\n" + "=" * 92)
        print("SOLVER STATISTICS (QED_SOLVER_STATS=1)")
        print("=" * 92)
        print(
            "%-26s %7s %10s %10s %12s %10s %10s %10s"
            % (
                "site",
                "solves",
                "iters",
                "it/solve",
                "matvecs",
                "mv/solve",
                "mv/iter",
                "mv wall s",
            )
        )
        print("-" * 92)
        # These spans sit INSIDE lstrs_davidson, so their matvecs are already
        # counted there -- list them separately or the totals double-count.
        nested = {"davidson_ah_warmstart", "davidson_ah_coldbuild", "newton_in_lstrs"}
        tot_mv = tot_solves = 0
        for site in sorted(s for s in by if s not in nested):
            rs = by[site]
            n = len(rs)
            it = sum(r["iters"] or 0 for r in rs)
            mv = sum(r["matvecs"] for r in rs)
            bd = sum(r["bordered"] for r in rs)
            sec = sum(r.get("mv_secs", 0.0) for r in rs)
            unit = mv if mv else bd
            tag = site + ("" if mv else " (bordered)")
            print(
                "%-26s %7d %10d %10.2f %12d %10.2f %10s %10.2f"
                % (
                    tag,
                    n,
                    it,
                    it / n if n else 0.0,
                    unit,
                    unit / n if n else 0.0,
                    ("%.2f" % (unit / it)) if it else "-",
                    sec,
                )
            )
            tot_mv += mv
            tot_solves += n
        print("-" * 92)
        print(
            "%-26s %7d %10s %10s %12d"
            % ("TOTAL", tot_solves, "", "", self.matvecs)
        )
        print("  (total Hessian-vector products over the whole run: %d)" % self.matvecs)
        print("  (total dense bordered eigensolves:                 %d)" % self.bordered)
        print(
            "  (total WALL TIME inside those products:            %.2f s)"
            % self.matvec_secs
        )

        # Cost per matvec as a function of how many vectors one call built.  If
        # s/vector falls with batch size there is a fixed per-call overhead, and
        # the one-vector-at-a-time solvers (GLTR, MINRES) are paying it on every
        # single product while the Davidson paths amortize it.
        if self.mv_hist:
            print("\nmatvec cost by batch size (vectors built per orbital_sigma3 call):")
            print(
                "  %-12s %8s %10s %10s %12s"
                % ("batch size", "calls", "vectors", "wall s", "s/vector")
            )
            for k in sorted(self.mv_hist):
                calls, vecs, sec = self.mv_hist[k]
                print(
                    "  %-12d %8d %10d %10.2f %12.5f"
                    % (k, calls, vecs, sec, sec / vecs if vecs else 0.0)
                )

        inner = sorted(s for s in by if s in nested)
        if inner:
            print("\nnested inside lstrs_davidson (already counted above):")
            print(
                "  %-24s %7s %10s %12s %10s"
                % ("site", "solves", "iters", "matvecs", "mv/solve")
            )
            for site in inner:
                rs = by[site]
                n = len(rs)
                it = sum(r["iters"] or 0 for r in rs)
                mv = sum(r["matvecs"] for r in rs)
                print(
                    "  %-24s %7d %10d %12d %10.2f"
                    % (site, n, it, mv, mv / n if n else 0.0)
                )

        # How much of the iteration count is spent squeezing the last digits?
        # For every solve that left a per-iteration convergence trace (GLTR),
        # find the first iteration whose residual was already within 10x / 100x
        # of the tolerance actually used.  The gap is what a looser exit would
        # have saved -- directly, not by extrapolation.
        traced = [r for r in self.records if r.get("trace")]
        if traced:
            print("\nwould a looser exit tolerance have stopped earlier?")
            print(
                "  %-22s %8s %10s %10s %10s"
                % ("site", "solves", "iters", "at 10x tol", "at 100x tol")
            )
            bysite = {}
            for r in traced:
                bysite.setdefault(r["site"], []).append(r)
            for site in sorted(bysite):
                rs = bysite[site]
                tot = e10 = e100 = 0
                for r in rs:
                    tr = r["trace"]
                    n = r["iters"] or len(tr)
                    tot += n
                    for mult, acc in ((10.0, "e10"), (100.0, "e100")):
                        hit = next(
                            (
                                i + 1
                                for i, (res, _sn, tol) in enumerate(tr)
                                if res < mult * tol
                            ),
                            n,
                        )
                        if acc == "e10":
                            e10 += min(hit, n)
                        else:
                            e100 += min(hit, n)
                print(
                    "  %-22s %8d %10d %10d %10d"
                    % (site, len(rs), tot, e10, e100)
                )
            print(
                "  (an 'at Nx tol' column far below 'iters' means the solver is"
                " grinding past\n   the point where the step stopped changing"
                " -- i.e. a looser exit is nearly free)"
            )

        if self.jd_widths:
            print("\nJacobi-Davidson correction solve, roots unconverged per entry:")
            print("  (block width a batched solve could use; 1 means nothing to block)")
            for w in sorted(self.jd_widths):
                print("  width %d: %d entries" % (w, self.jd_widths[w]))

        # termination-branch distribution, per site
        print("\ntermination branch (hard_case / GLTR status) by site:")
        for site in sorted(by):
            dist = {}
            for r in by[site]:
                dist[r["status"]] = dist.get(r["status"], 0) + 1
            if len(dist) == 1 and None in dist:
                continue
            print(
                "  %-24s %s"
                % (
                    site,
                    ", ".join(
                        "%s:%d" % (k, v) for k, v in sorted(dist.items(), key=str)
                    ),
                )
            )
        if extra:
            for line in extra:
                print("  " + line)
        print("=" * 92)


_solver_stats = _SolverStats()


def _stats_begin(site):
    return _solver_stats.begin(site) if _QED_SOLVER_STATS else None


def _stats_end(span, iters=None, status=None):
    if _QED_SOLVER_STATS:
        _solver_stats.end(span, iters=iters, status=status)


if _QED_SOLVER_STATS:
    import atexit

    atexit.register(_solver_stats.report)


def _dump_cpp_casscf_validation_case(kind, **arrays):
    if not _CPP_CASSCF_VALIDATION_DIR:
        return
    idx = _cpp_casscf_validation_counters.get(kind, 0)
    _cpp_casscf_validation_counters[kind] = idx + 1
    case_dir = os.path.join(_CPP_CASSCF_VALIDATION_DIR, f"{kind}_{idx:03d}")
    os.makedirs(case_dir, exist_ok=True)
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim <= 2:
            np.savetxt(os.path.join(case_dir, f"{name}.txt"), np.atleast_1d(arr))
        else:
            # savetxt only handles 1D/2D -- flatten (C-order, matching this
            # array's own in-memory layout) plus a sidecar shape file so the
            # C++ side can reshape it back. Only used by the intermediates
            # validation dumps (rank-3/4 tensors); the trust-region solver
            # dumps never exceed rank 2.
            np.savetxt(os.path.join(case_dir, f"{name}.txt"), arr.ravel(order="C"))
            with open(os.path.join(case_dir, f"{name}.shape.txt"), "w") as f:
                f.write(" ".join(str(d) for d in arr.shape) + "\n")


def _degenerate_clusters(eps, bounds, tol):
    """Maximal runs of near-degenerate orbitals, never crossing a block edge.

    `eps` are orbital energies in index order.  `bounds` are the inactive|active
    and active|virtual boundaries: a cluster must NEVER span one, because mixing
    across it would change the inactive/active/virtual SPACES and therefore the
    energy.  Only rotations strictly inside a block are redundant, and only
    those are safe to touch.

    Single-linkage on consecutive gaps, so a chain of pairwise-close orbitals
    forms one cluster.  Returns [(start, stop), ...] with stop - start >= 2;
    singletons carry no gauge freedom and are omitted.
    """
    n = len(eps)
    edges = set(int(b) for b in bounds)
    clusters = []
    start = 0
    for i in range(1, n + 1):
        if i == n or i in edges or abs(eps[i] - eps[i - 1]) > tol:
            if i - start >= 2:
                clusters.append((start, i))
            start = i
    return clusters


def _canonicalize_degenerate(C, T_ao, eps, bounds, tol):
    """Impose a deterministic gauge inside each near-degenerate orbital cluster.

    See the QED_DEGEN_CANON block at the top of this file for the measurement
    that motivates this.  In short: within a degenerate set the eigenvectors
    LAPACK returns are arbitrary, threaded runs return different ones, the
    energy does not care and the optimizer's trajectory does.

    THE CONSTRUCTION IS GAUGE-INVARIANT BY CONSTRUCTION, which is the whole
    point -- it does not merely pick *a* convention, it picks the SAME one from
    any starting gauge.  If a cluster is spanned by C1 and equally by
    C2 = C1 Q for any orthogonal Q, then the fixed operator T represented in
    the two bases satisfies T2 = C2^T T C2 = Q^T T1 Q, so its eigenvectors obey
    V2 = Q^T V1 and the canonicalized orbitals are

        C2 V2 = C1 Q Q^T V1 = C1 V1,

    identical whichever gauge came in.  The only requirement is that T's
    eigenvalues be distinct WITHIN the cluster, which is why T must not share
    the molecular symmetry that caused the degeneracy in the first place.
    """
    C = np.array(C, dtype=float, copy=True)
    for (a, b) in _degenerate_clusters(eps, bounds, tol):
        blk = C[:, a:b]
        T = blk.T @ T_ao @ blk
        T = 0.5 * (T + T.T)  # kill asymmetry from rounding before eigh
        _w, V = np.linalg.eigh(T)
        C[:, a:b] = blk @ V
    return C


def _pin_orbital_signs(C):
    """Deterministic column-sign convention: largest-|coefficient| made positive.

    np.round before argmax is essential, not cosmetic -- see the
    QED_RANDOM_ORBITAL_SEED block for the measured tie-breaking failure it
    prevents (symmetry-equivalent atoms give EXACTLY tied largest coefficients
    that differ only by ~1e-15 of threading noise, so a raw argmax picks between
    them arbitrarily and the sign rule itself stops being reproducible).
    """
    C = np.array(C, dtype=float, copy=True)
    for j in range(C.shape[1]):
        k = int(np.argmax(np.round(np.abs(C[:, j]), 10)))
        if C[k, j] < 0.0:
            C[:, j] *= -1.0
    return C


def _gauge_tiebreaker(nao):
    """The fixed operator T used to resolve degenerate clusters.

    Requirements: deterministic, symmetric, and NOT invariant under the spatial
    symmetry that produced the degeneracy (otherwise it inherits the same
    degeneracy and resolves nothing).  A diagonal weighting by AO index meets
    all three: the permutation of AO functions that maps p_x onto p_y does not
    leave diag(1..nao) invariant, so it generically splits a pi pair.

    It is deliberately arbitrary -- ANY fixed choice removes the arbitrariness,
    and the point is that it is the same choice on every run and every thread
    count.  It is NOT physically meaningful and nothing downstream should read
    it as an orbital energy.
    """
    return np.diag(np.arange(1.0, float(nao) + 1.0))


# _more_sorensen_dense MOVED to lstrs_trust_region.py -- see that file's
# module docstring for why the LSTRS solvers are being split out.  Imported
# under its old private name so every call site (the QED_EXACT_TR path and
# both hard-case fallbacks) reads exactly as before; the function itself is
# byte-for-byte unchanged, which validation/lstrs_refactor_gate.sh checks.
from lstrs_trust_region import (
    more_sorensen_dense as _more_sorensen_dense,
    solve_lstrs_microiteration,
    solve_lstrs_internal,
    exact_internal_trs,
)

# The trust-region subproblem solvers MOVED to trust_region_solvers.py on
# 2026-09-04 -- GLTR, AH-Krylov, the OTR step, the three Newton solvers, the
# Jacobi-Davidson correction, and the bordered eigensolve/matvec/preconditioner
# they share.  Each survives below as a delegator so every call site and every
# cpp_casscf citation still resolves.
from trust_region_solvers import (
    _ah_krylov_trust_region_step as _trs__ah_krylov_trust_region_step,
    _otr_trust_region_step as _trs__otr_trust_region_step,
    aug_matvec as _trs_aug_matvec,
    build_hessian_diagonal as _trs_build_hessian_diagonal,
    inner_solve_pcg as _trs_inner_solve_pcg,
    linear_equation_solve as _trs_linear_equation_solve,
    minres_solve as _trs_minres_solve,
    projection_step2 as _trs_projection_step2,
    scipy_minres_solve as _trs_scipy_minres_solve,
    solve_gltr_trust_region as _trs_solve_gltr_trust_region,
    solve_gltr_with_operator as _trs_solve_gltr_with_operator,
)

# Davidson_augmented_hessian_solve6's BODY MOVED to davidson_augmented_hessian.py
# -- the inner augmented-Hessian Davidson that the LSTRS bisection above calls
# once per beta.  The method survives on PFHamiltonianGenerator as a delegator,
# so every call site and the cpp_casscf line citations still resolve.
from davidson_augmented_hessian import davidson_augmented_hessian_solve


def _adaptive_tol_from_rho(
    rho_history,
    current_tol,
    base=None,
    lo=None,
    hi=None,
    window=None,
    tight=None,
    loose=None,
    factor=None,
):
    """Next subproblem exit tolerance, given the recent model-fidelity history.

    rho = (actual energy change) / (predicted energy change); rho == 1 means the
    quadratic model predicted the step's effect exactly.

    The rule, in one line: SOLVE THE SUBPROBLEM EXACTLY AS HARD AS THE MODEL
    DESERVES.  |1 - median(rho)| estimates the model's own error.  Below `tight`
    the model is trustworthy, so extra subproblem accuracy converts into real
    energy decrease -> tighten.  Above `loose` the model is not worth resolving
    -> loosen, and leave the fixing to step_control shrinking the radius.  In
    between, hold (hysteresis, so it cannot oscillate every iteration).

    A single rho < 0 in the window -- a step that moved the energy the WRONG WAY
    -- loosens immediately regardless of the median: that is the strongest
    available evidence that resolving the model further is wasted work.

    Pure function of its arguments so it can be tested without a CASSCF object.
    Defaults come from the QED_ADAPTIVE_TOL_* knobs.
    """
    base = _QED_ADAPTIVE_TOL_BASE if base is None else base
    lo = _QED_ADAPTIVE_TOL_MIN if lo is None else lo
    hi = _QED_ADAPTIVE_TOL_MAX if hi is None else hi
    window = _QED_ADAPTIVE_TOL_WINDOW if window is None else window
    tight = _QED_ADAPTIVE_TOL_TIGHT if tight is None else tight
    loose = _QED_ADAPTIVE_TOL_LOOSE if loose is None else loose
    factor = _QED_ADAPTIVE_TOL_FACTOR if factor is None else factor

    if current_tol is None:
        current_tol = base
    # Ignore non-finite rho: predicted_energy2 can underflow to ~0 near
    # convergence, and inf/nan must not be read as "model is perfect".
    w = [r for r in rho_history[-window:] if np.isfinite(r)]
    if not w:
        return float(min(max(current_tol, lo), hi))

    if min(w) < 0.0:
        return float(min(current_tol * factor, hi))

    dev = abs(1.0 - float(np.median(w)))
    # Compare with a relative slack: these are user-set thresholds, and a
    # boundary case must not flip on floating-point dust (|1 - 0.95| evaluates
    # to 0.050000000000000044, which would silently miss a `<= 0.05` test).
    slack = 1.0 + 1e-9
    if dev <= tight * slack:
        return float(max(current_tol / factor, lo))
    if dev * slack >= loose:
        return float(min(current_tol * factor, hi))
    return float(min(max(current_tol, lo), hi))


# GLTR reports its termination as a string, e.g. "Interior (Newton) iter 3" or
# "Boundary iter 7"; the other two returns are "success_zero_gradient" (no
# solve happened) and "Failure" (ran out of Lanczos iterations).  Only the first
# word matters to the controller, so classify once, here, rather than string-
# matching at every use.
def _gltr_exit_is_interior(status):
    """True iff this GLTR status is an interior (unconstrained Newton) exit.

    Anything that is not recognisably interior is treated as NOT interior --
    "Failure" and "success_zero_gradient" included.  That is the conservative
    direction: an unrecognised exit can only push the controller back toward
    base, never loosen it.
    """
    return isinstance(status, str) and status.startswith("Interior")


def _subspace_tol_from_exits(
    exit_history,
    current_tol,
    base=None,
    lo=None,
    hi=None,
    window=None,
    interior_hi=None,
    interior_lo=None,
    factor=None,
    recent=None,
):
    """Next GLTR exit tolerance for one site, given that site's recent exits.

    `exit_history` is a list of GLTR status strings, oldest first, for THIS site
    only (the three GLTR sites have very different exit mixes -- see the
    QED_SUBSPACE_TOL knob block -- so pooling them would average away exactly
    the signal the controller runs on).

    The rule, in one line: LOOSEN WHERE THE TRUST-REGION BOUNDARY IS NOT
    BINDING.  An "Interior (Newton)" exit means GLTR stopped strictly inside the
    trust region, where tol buys only linear-solve accuracy on a step whose
    PLACEMENT it does not control; a "Boundary" exit means tol interacts with
    where on the boundary the step lands, and loosening is no longer free.

    Two guards, both deliberately asymmetric toward safety:
      * a boundary exit among the last `recent` solves BLOCKS loosening outright,
        so the controller reacts to one immediately instead of waiting for it to
        win the window;
      * `lo` defaults to `base`, so the controller can only loosen or return to
        base -- it can never walk below the hard-coded 1e-7 the call sites use.

    Pure function of its arguments so it can be tested without a CASSCF object.
    Defaults come from the QED_SUBSPACE_TOL_* knobs.
    """
    base = _QED_SUBSPACE_TOL_BASE if base is None else base
    lo = _QED_SUBSPACE_TOL_MIN if lo is None else lo
    hi = _QED_SUBSPACE_TOL_MAX if hi is None else hi
    window = _QED_SUBSPACE_TOL_WINDOW if window is None else window
    interior_hi = _QED_SUBSPACE_TOL_HI if interior_hi is None else interior_hi
    interior_lo = _QED_SUBSPACE_TOL_LO if interior_lo is None else interior_lo
    factor = _QED_SUBSPACE_TOL_FACTOR if factor is None else factor
    recent = _QED_SUBSPACE_TOL_RECENT if recent is None else recent

    if current_tol is None:
        current_tol = base
    clamp = lambda t: float(min(max(t, lo), hi))

    w = list(exit_history[-window:]) if window > 0 else list(exit_history)
    if not w:
        # No solve has been observed at this site yet: run the first one at the
        # tolerance the call site would have used anyway.
        return clamp(current_tol)

    interior = [_gltr_exit_is_interior(s) for s in w]
    frac = sum(interior) / float(len(interior))

    # Compare with a relative slack for the same reason _adaptive_tol_from_rho
    # does: these are user-set thresholds and a boundary case must not flip on
    # floating-point dust (3/4 is exact, but 0.75 read from an env var need not
    # be, and interior fractions like 2/3 vs a 0.667 knob certainly are not).
    slack = 1.0 + 1e-9

    if frac <= interior_lo * slack:
        # Boundary-dominated: this site's steps are being placed by the trust
        # region, so pull back toward base.  Slack on the THRESHOLD so the test
        # is permissive, matching the documented "at or below LO" semantics --
        # see the note in _exit_boundary_dominated on why exactly 0.25 is a
        # common value and must be included.
        return clamp(current_tol / factor)

    if frac * slack >= interior_hi:
        if recent > 0 and not all(interior[-recent:]):
            # Interior-dominated over the window, but a boundary exit landed
            # very recently -- hold rather than loosen into it.
            return clamp(current_tol)
        return clamp(current_tol * factor)

    return clamp(current_tol)


def _exit_last_is_interior(exit_history):
    """True iff the most recent solve at this site exited interior.

    Mode 2's growth gate: an interior step says the model held over a step the
    region never constrained, which is no evidence that a LARGER region would
    still be trustworthy.  An empty history returns False, i.e. it does not gate
    -- with nothing observed the original 1.2x growth stands.
    """
    return bool(exit_history) and _gltr_exit_is_interior(exit_history[-1])


def _exit_boundary_dominated(exit_history, window=None, interior_lo=None):
    """True iff this site's recent exits are persistently BOUNDARY-terminating.

    Mode 1's damping trigger.  Persistently, not once: a single boundary exit is
    ordinary trust-region behaviour and damping on it would fight the algorithm
    rather than the pathology.  The threshold is the low edge of the measured
    c10h8-vs-MgH+ separation (see the QED_EXIT_CONTROL knob block).

    Pure function of its arguments so it can be tested without a CASSCF object.
    """
    window = _QED_EXIT_CONTROL_WINDOW if window is None else window
    interior_lo = _QED_EXIT_CONTROL_LO if interior_lo is None else interior_lo
    w = list(exit_history[-window:]) if window > 0 else list(exit_history)
    if not w:
        return False
    frac = sum(_gltr_exit_is_interior(s) for s in w) / float(len(w))
    # Relative slack for the same floating-point reason as the tolerance
    # controller: a boundary case must not flip on dust.  The slack goes on the
    # THRESHOLD, making the test permissive -- the documented semantics are "at
    # or below LO", and with the default window of 4 the achievable fractions
    # are 0, 0.25, 0.5, 0.75, 1.0, so exactly 0.25 is a COMMON value and an
    # accidentally-exclusive test would leave this knob nearly inert on the one
    # system it exists for.
    return frac <= interior_lo * (1.0 + 1e-9)


def _exit_control_radius(trust_radius, exit_history, damp=None, lo=None,
                         window=None, interior_lo=None):
    """Damped trust radius for a persistently boundary-terminating site.

    Returns the radius unchanged when the site is not boundary-dominated, so the
    caller can apply this unconditionally and let the predicate decide.
    """
    damp = _QED_EXIT_CONTROL_DAMP if damp is None else damp
    lo = _QED_EXIT_CONTROL_MIN if lo is None else lo
    if not _exit_boundary_dominated(exit_history, window=window,
                                    interior_lo=interior_lo):
        return float(trust_radius)
    return float(max(trust_radius * damp, lo))


script_dir = os.path.abspath(os.path.dirname(__file__))
from qed_ci.backend import backend_path, load_backend
lib_path = str(backend_path())

# import shared lib
cfunctions = load_backend()


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

cfunctions.build_sigma_2.argtypes = [
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
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
]
cfunctions.build_two_rdm.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_bool,
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
cfunctions.build_active_rdm_sa.argtypes = [
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
cfunctions.build_active_rdm_z.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
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

cfunctions.build_active_photon_electron_one_rdm_z.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
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
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.gram_schmidt_add.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]

# QED_JK_H2E_DISK (step 1b): point the JK kernel's packed-h2e source at a file.
# NULL restores the resident source.  Guarded by hasattr so an OLD cfunctions.so
# -- one built before orbital.c grew this symbol -- still imports; the spill
# refuses to arm in that case rather than silently falling back to resident.
_HAS_JK_H2E_FILE = hasattr(cfunctions, "qed_jk_set_h2e_file")
if _HAS_JK_H2E_FILE:
    cfunctions.qed_jk_set_h2e_file.argtypes = [ctypes.c_char_p]
    cfunctions.qed_jk_set_h2e_file.restype = None
cfunctions.full_transformation_macroiteration.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.full_transformation_internal_optimization.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
]
cfunctions.build_sigma_reduced.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_int32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(ctypes.c_double, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]


def c_H_diag_cas(
    h1e,
    h2e,
    H_diag,
    N_p,
    num_alpha,
    nmo,
    n_act_a,
    n_act_orb,
    n_in_a,
    E_core,
    omega,
    Enuc,
    dc,
    Y,
):
    cfunctions.build_H_diag_cas(
        h1e,
        h2e,
        H_diag,
        N_p,
        num_alpha,
        nmo,
        n_act_a,
        n_act_orb,
        n_in_a,
        E_core,
        omega,
        Enuc,
        dc,
        Y,
    )


def c_H_diag_cas_spin(
    h1e,
    h2e,
    H_diag,
    N_p,
    num_alpha,
    nmo,
    n_act_a,
    n_act_orb,
    n_in_a,
    E_core,
    omega,
    Enuc,
    dc,
    Y,
    target_spin,
):
    cfunctions.build_H_diag_cas_spin(
        h1e,
        h2e,
        H_diag,
        N_p,
        num_alpha,
        nmo,
        n_act_a,
        n_act_orb,
        n_in_a,
        E_core,
        omega,
        Enuc,
        dc,
        Y,
        target_spin,
    )


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
    target_spin,
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
        target_spin,
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

def c_sigma_2(
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
    omega1,
    omega2,
    d_exp,
    E_core,
    break_degeneracy,
):
    cfunctions.build_sigma_2(
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
        omega1,
        omega2,
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
    z_vector, eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2, z_state
):
    cfunctions.build_one_rdm(
       z_vector, eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2, z_state
    )


def c_build_two_rdm(
    z_vector, eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2, z_state
):
    cfunctions.build_two_rdm(
       z_vector, eigvec, D, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2, z_state
    )


def c_build_active_rdm(
    eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_rdm(
        eigvec,
        D_tu,
        D_tuvw,
        table,
        N_ac,
        n_o_ac,
        num_photon,
        state_p1,
        state_p2,
        weight,
    )

def c_build_active_rdm_sa(
    eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    """CASSCF-only bra==ket specialisation of c_build_active_rdm.

    ONLY for call sites where the bra and the ket are the same slice of the
    same array.  nuclear_grad.py must keep using c_build_active_rdm /
    c_build_active_rdm_z: it builds transition quantities where the two
    differ -- and at nuclear_grad.py:1024-1045 they differ by ARRAY while the
    state indices are equal (i, i), so the state indices alone do not identify
    the case.  The C side defers to the general routine if the states differ,
    but that check cannot see a second array, which is why the routing is done
    here by call site rather than by a runtime test.
    """
    cfunctions.build_active_rdm_sa(
        eigvec,
        D_tu,
        D_tuvw,
        table,
        N_ac,
        n_o_ac,
        num_photon,
        state_p1,
        state_p2,
        weight,
    )
def c_build_active_rdm_z(
    z_vector, eigvec, D_tu, D_tuvw, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_rdm_z(
        z_vector,
        eigvec,
        D_tu,
        D_tuvw,
        table,
        N_ac,
        n_o_ac,
        num_photon,
        state_p1,
        state_p2,
        weight,
    )



def c_build_photon_electron_one_rdm(
    z_vector, eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
):
    cfunctions.build_photon_electron_one_rdm(
       z_vector, eigvec, Dpe, table, N_ac, n_o_ac, n_o_in, num_photon, state_p1, state_p2
    )


def c_build_active_photon_electron_one_rdm(
    eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_photon_electron_one_rdm(
        eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
    )

def c_build_active_photon_electron_one_rdm_z(
    z_vector, eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
):
    cfunctions.build_active_photon_electron_one_rdm_z(
       z_vector, eigvec, Dpe_tu, table, N_ac, n_o_ac, num_photon, state_p1, state_p2, weight
    )



def c_gram_schmidt_orthogonalization(Q, rows, cols):
    cfunctions.gram_schmidt_orthogonalization(Q, rows, cols)


def c_gram_schmidt_add(Q, rows, cols, rows2):
    cfunctions.gram_schmidt_add(Q, rows, cols, rows2)


# Bumped by every C routine below that writes J/K.  Both of them take J/K as
# OUTPUT buffers and fill them IN PLACE (the internal-optimization one passes
# them as input *and* output), so the Python object identity is unchanged by a
# transformation -- an identity/weakref test cannot detect it.  Anything caching
# a quantity derived from J/K (see build_intermediates' L/temp1 cache) must key
# on this epoch.  Bumping here rather than at the call sites means new call
# sites are covered automatically; bumping unconditionally (even when the arrays
# passed are not self.J/self.K) is deliberate -- over-invalidating only costs a
# rebuild, whereas under-invalidating silently corrupts the orbital Hessian.
_JK_WRITE_EPOCH = 0


def c_full_transformation_macroiteration(
    U, h2e, J, K, index_map_pq, index_map_kl, nmo, n_occupied
):
    global _JK_WRITE_EPOCH
    _JK_WRITE_EPOCH += 1
    cfunctions.full_transformation_macroiteration(
        U, h2e, J, K, index_map_pq, index_map_kl, nmo, n_occupied
    )


def c_full_transformation_internal_optimization(
    U, J, K, h1, d_cmo, Jt, Kt, h1t, d_cmot, index_map_ab, index_map_kl, nmo, n_occupied
):
    global _JK_WRITE_EPOCH
    _JK_WRITE_EPOCH += 1
    cfunctions.full_transformation_internal_optimization(
        U,
        J,
        K,
        h1,
        d_cmo,
        Jt,
        Kt,
        h1t,
        d_cmot,
        index_map_ab,
        index_map_kl,
        nmo,
        n_occupied,
    )


def c_build_sigma_reduced(
    U,
    A_tilde,
    index_map,
    G,
    R_reduced,
    sigma_reduced,
    num_states,
    pointer,
    nmo,
    index_map_size,
    n_occupied,
):
    cfunctions.build_sigma_reduced(
        U,
        A_tilde,
        index_map,
        G,
        R_reduced,
        sigma_reduced,
        num_states,
        pointer,
        nmo,
        index_map_size,
        n_occupied,
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

        return (
            Determinant.countNumOrbitalsInBits(self.alphaObtBits),
            Determinant.countNumOrbitalsInBits(self.betaObtBits),
        )

    def getOrbitalIndexLists(self):
        """
        Return lists of orbital index
        """

        return (
            Determinant.obtBits2ObtIndexList(self.alphaObtBits),
            Determinant.obtBits2ObtIndexList(self.betaObtBits),
        )

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

        alphaList, betaList = (
            Determinant.obtBits2ObtIndexList(alphaBits),
            Determinant.obtBits2ObtIndexList(betaBits),
        )
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

        return (
            Determinant.getOrbitalPositions(self.alphaObtBits, alphaIndexList),
            Determinant.getOrbitalPositions(self.betaObtBits, betaIndexList),
        )

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

        return (
            Determinant.countNumOrbitalsInBits(
                self.alphaObtBits & another.alphaObtBits
            ),
            Determinant.countNumOrbitalsInBits(self.betaObtBits & another.betaObtBits),
        )

    def getCommonOrbitalsInLists(self, another):
        """Return common orbitals between this determinant and another determinant in lists"""
        return (
            Determinant.obtBits2ObtIndexList(self.alphaObtBits & another.alphaObtBits),
            Determinant.obtBits2ObtIndexList(self.betaObtBits & another.betaObtBits),
        )

    def getCommonOrbitalsInMixedSpinIndexList(self, another):
        alphaList, betaList = self.getCommonOrbitalsInLists(another)
        return Determinant.mixIndexList(alphaList, betaList)

    def numberOfDiffOrbitals(self, another):
        """
        Return the number of different alpha and beta orbitals between this determinant and another determinant
        """

        diffAlpha, diffBeta = (
            Determinant.countNumOrbitalsInBits(
                self.alphaObtBits ^ another.alphaObtBits
            ),
            Determinant.countNumOrbitalsInBits(self.betaObtBits ^ another.betaObtBits),
        )
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

        diffAlpha, diffBeta = (
            Determinant.countNumOrbitalsInBitsUpTo4(
                self.alphaObtBits ^ another.alphaObtBits
            ),
            Determinant.countNumOrbitalsInBitsUpTo4(
                self.betaObtBits ^ another.betaObtBits
            ),
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
        return (
            Determinant.obtBits2ObtIndexList(bits1),
            Determinant.obtBits2ObtIndexList(bits2),
        )

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
        sign1, sign2 = (
            self.getSignToMoveOrbitalsToTheFront(alphaList1, betaList1),
            another.getSignToMoveOrbitalsToTheFront(alphaList2, betaList2),
        )
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
    # Set for real in buildArraysInOrbitalBasis.  Declared here so the DF and
    # mo_eri paths, which never reach that assignment, still answer False.
    _oo_skip_tensor = False
    _oo_Ca_ref = None


    def __init__(self, molecule_string, psi4_options_dict=None, cavity_options=None, *, config=None):
        from qed_ci.config import CQEDConfig
        from qed_ci.runtime import calculation_scope
        if isinstance(psi4_options_dict, CQEDConfig):
            if config is not None or cavity_options is not None:
                raise ValueError("Pass either a configuration or legacy dictionaries, not both")
            config, psi4_options_dict = psi4_options_dict, None
        if config is not None:
            if psi4_options_dict is not None or cavity_options is not None:
                raise ValueError("Pass config= or legacy dictionaries, not both")
            if not isinstance(config, CQEDConfig):
                raise TypeError("config must be a CQEDConfig")
            self.config = config
        else:
            self.config = CQEDConfig.from_legacy(psi4_options_dict or {}, cavity_options or {})
        self.diagnostics = list(self.config.diagnostics)
        self._generated_files = {}
        from qed_ci.output import engine_output
        from qed_ci.results import build_result
        from qed_ci.report import report
        self._macro_diagnostics = dict(applicable=False, converged=None,
                                       reason='not_requested', iterations=0, history=[])
        self._initial_ci_diagnostics = dict(converged=None, reason='unavailable')
        self._final_ci_diagnostics = dict(converged=None, reason='unavailable')
        with calculation_scope(self, cfunctions):
            if self.config.verbosity != "normal":
                for message in self.diagnostics:
                    print("QED-CI input notice:", message)
            molecule = psi4.geometry(molecule_string)
            if molecule.multiplicity() != self.config.multiplicity:
                raise ValueError("Geometry multiplicity conflicts with the restricted reference configuration")
            if self.config.charge is not None and molecule.molecular_charge() != self.config.charge:
                raise ValueError("Geometry charge conflicts with CQEDConfig.charge")
            molecule.update_geometry()
            self._result_geometry = dict(symbols=[molecule.symbol(i) for i in range(molecule.natom())],
                                         coordinates=np.asarray(molecule.geometry()).tolist(),
                                         charge=molecule.molecular_charge(), multiplicity=molecule.multiplicity())
            started = time.perf_counter()
            import warnings
            with warnings.catch_warnings(record=True) as caught, engine_output(self.config, cfunctions):
                self._run_calculation(molecule_string, self.config.psi4_options,
                                      self.config.to_legacy_options())
            self.diagnostics.extend(str(w.message) for w in caught)
            self.result = build_result(self, time.perf_counter() - started)
            report(self.result)

    def output_context(self):
        """Route additional property/helper calls to this calculation's output.

        Subsequent calls append to an output file instead of truncating it.
        The constructor and nuclear_grad.compute_grad enter this automatically.
        """
        from qed_ci.runtime import calculation_scope
        return calculation_scope(self, cfunctions, append=True)

    def _run_calculation(self, molecule_string, psi4_options_dict, cavity_options):
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

        # QED_ORBITAL_SYMMETRY (default off => this block does nothing).  Project
        # the starting orbitals onto pure axial-symmetry species BEFORE any
        # integral is built from them, so everything downstream is consistent.
        # The species labels are assigned once, here, and held fixed for the whole
        # run -- the same choice Molpro makes when it reads starting orbitals.
        self._sym_labels = None
        if _QED_ORBITAL_SYMMETRY:
            import orbital_symmetry as _orbsym

            _bs = psi4_wfn_o.basisset()
            _S = np.asarray(psi4.core.MintsHelper(_bs).ao_overlap())
            _species = _orbsym.ao_species(_bs)
            _orbsym.validate_axis(_S, _species)
            _lab0, _pur0, _, _ = _orbsym.classify(self.C, _S, _species)
            self.C = _orbsym.symmetrize(self.C, _S, _species, _lab0)
            self._sym_labels, _, _, _ = _orbsym.classify(self.C, _S, _species)
            self._sym_report = (_lab0, _pur0)
            self.Ca = psi4.core.Matrix.from_array(self.C)

        # build arrays in orbital basis from last step
        self.buildArraysInOrbitalBasis(psi4_wfn_o, psi4_options_dict)

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
                self.H1temp = copy.deepcopy(self.H_spatial2)
                self.d_cmo_temp = copy.deepcopy(self.d_cmo)
                if "df_basis_scf" in psi4_options_dict:
                    self.J, self.K = self.build_JK_from_df(
                        self.B_mo,  # Your DF 3-index tensor
                        self.d_cmo,  # Your dipole matrix
                        self.n_occupied,
                        ignore_dse_terms=self.ignore_dse_terms
                    )
                    self.J = np.ascontiguousarray(self.J)
                    self.K = np.ascontiguousarray(self.K)
                elif _QED_LIBTRANS_OO == 1:
                    self.build_JK_oo()
                else:
                    self.build_JK()
                self.occupied_J = copy.deepcopy(
                    self.J[:, :, : self.n_occupied, : self.n_occupied]
                )
                self.occupied_K = copy.deepcopy(
                    self.K[:, :, : self.n_occupied, : self.n_occupied]
                )
                self.occupied_h1 = copy.deepcopy(
                    self.H_spatial2[: self.n_occupied, : self.n_occupied]
                )
                self.occupied_d_cmo = copy.deepcopy(
                    self.d_cmo[: self.n_occupied, : self.n_occupied]
                )
                self.occupied_J3 = self.occupied_J.reshape(
                    self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
                )

                # build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                # for k in range(self.nmo):
                #    for l in range(self.nmo):
                #        kl = k * self.nmo + l
                #        self.fock_core[k][l] = self.H_spatial2[k][l]
                #        for j in range(self.n_in_a):
                #            jj = j * self.nmo + j
                #            kj = k * self.nmo + j
                #            jl = j * self.nmo + l
                #            self.fock_core[k][l] += 2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]

                self.fock_core = copy.deepcopy(self.H_spatial2)
                self.fock_core += 2.0 * np.einsum(
                    "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :]
                )
                self.fock_core -= np.einsum(
                    "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :]
                )

                self.occupied_fock_core = copy.deepcopy(
                    self.fock_core[: self.n_occupied, : self.n_occupied]
                )

                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                # for k in range(self.n_act_orb):
                #    for l in range(self.n_act_orb):
                #        self.gkl2[k][l] = self.fock_core[k+self.n_in_a][l+self.n_in_a]
                #        for j in range(self.n_act_orb):
                #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                #            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]
                self.gkl2 = copy.deepcopy(
                    self.fock_core[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ]
                )
                self.gkl2 -= 0.5 * np.einsum(
                    "kjjl->kl",
                    self.J[
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                    ],
                )

                self.E_core = 0.0
                # for i in range(self.n_in_a):
                #    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i]
                self.E_core += np.einsum(
                    "jj->", self.occupied_h1[: self.n_in_a, : self.n_in_a]
                )
                self.E_core += np.einsum(
                    "jj->", self.fock_core[: self.n_in_a, : self.n_in_a]
                )

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
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                    dtype=np.int32,
                )
                c_graph(self.n_act_a, self.n_act_orb, self.Y)

                # self.H_diag3 = np.zeros(H_dim)
                # c_H_diag_cas(
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
                # print(self.H_diag3)

                # self.target_spin = 0.0
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
                    self.target_spin,
                )
                # print(self.H_diag3, flush = True)
                self.index_Hdiag = np.asarray(self.H_diag3.argsort(), dtype=np.int32)
                # np.savetxt("H_diag.out", self.H_diag3)
                # print(np.sort(self.H_diag3))
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
                    self.target_spin,
                )
                # print(self.H_diag, flush = True)
                # print(self.H_diag - self.H_diag3, flush = True)
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
                c_s_diag(
                    self.S_diag_projection,
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
                self.n_occupied = self.n_act_orb + self.n_in_a

                self.num_alpha = math.comb(
                    self.nmo, self.ndocc
                )  # number of alpha strings
                self.num_det = self.num_alpha * self.num_alpha  # number of determinants
                self.FCInumDets = self.num_det
                H_dim = self.FCInumDets * np1
                self.H_diag = np.zeros(H_dim)
                if "df_basis_scf" in psi4_options_dict:
                    self.J, self.K = self.build_JK_from_df(
                        self.B_mo,  # Your DF 3-index tensor
                        self.d_cmo,  # Your dipole matrix
                        self.n_occupied,
                        ignore_dse_terms=self.ignore_dse_terms
                    )
                    self.J = np.ascontiguousarray(self.J)
                    self.K = np.ascontiguousarray(self.K)
                elif _QED_LIBTRANS_OO == 1:
                    self.build_JK_oo()
                else:
                    self.build_JK()
                self.occupied_K = copy.deepcopy(
                    self.K[:, :, : self.n_occupied, : self.n_occupied]
                )
                self.occupied_J = copy.deepcopy(
                    self.J[:, :, : self.n_occupied, : self.n_occupied]
                )
                self.occupied_h1 = copy.deepcopy(
                    self.H_spatial2[: self.n_occupied, : self.n_occupied]
                )
                self.occupied_d_cmo = copy.deepcopy(
                    self.d_cmo[: self.n_occupied, : self.n_occupied]
                )
                self.occupied_J3 = self.occupied_J.reshape(
                    self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
                )

                # build core Fock matrix
                self.fock_core = np.zeros((self.nmo, self.nmo))
                # for k in range(self.nmo):
                #    for l in range(self.nmo):
                #        kl = k * self.nmo + l
                #        self.fock_core[k][l] = self.H_spatial2[k][l]
                #        for j in range(self.n_in_a):
                #            jj = j * self.nmo + j
                #            kj = k * self.nmo + j
                #            jl = j * self.nmo + l
                #            self.fock_core[k][l] += 2.0 * self.twoeint[kl][jj] - self.twoeint[kj][jl]
                self.fock_core = copy.deepcopy(self.H_spatial2)
                self.fock_core += 2.0 * np.einsum(
                    "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :]
                )
                self.fock_core -= np.einsum(
                    "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :]
                )

                self.occupied_fock_core = copy.deepcopy(
                    self.fock_core[: self.n_occupied, : self.n_occupied]
                )

                self.gkl2 = np.zeros((self.n_act_orb, self.n_act_orb))
                # for k in range(self.n_act_orb):
                #    for l in range(self.n_act_orb):
                #        self.gkl2[k][l] = self.fock_core[k+self.n_in_a][l+self.n_in_a]
                #        for j in range(self.n_act_orb):
                #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
                #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
                #            self.gkl2[k][l] -= 0.5 * self.twoeint[kj][jl]
                self.gkl2 = copy.deepcopy(
                    self.fock_core[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ]
                )
                self.gkl2 -= 0.5 * np.einsum(
                    "kjjl->kl",
                    self.J[
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                    ],
                )

                self.E_core = 0.0
                # for i in range(self.n_in_a):
                #    self.E_core += self.H_spatial2[i][i] + self.fock_core[i][i]
                self.E_core += np.einsum(
                    "jj->", self.occupied_h1[: self.n_in_a, : self.n_in_a]
                )
                self.E_core += np.einsum(
                    "jj->", self.fock_core[: self.n_in_a, : self.n_in_a]
                )

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
                    self.n_act_a * (self.n_act_orb - self.n_act_a + 1) * 3,
                    dtype=np.int32,
                )
                c_graph(self.n_act_a, self.n_act_orb, self.Y)

                # self.target_spin = 0.0
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
                    self.target_spin,
                )
                # print(self.H_diag3, flush = True)
                self.index_Hdiag = np.asarray(self.H_diag3.argsort(), dtype=np.int32)
                # np.savetxt("H_diag.out", self.H_diag3)
                # print(np.sort(self.H_diag3))
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
                    self.target_spin,
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
                c_s_diag(
                    self.S_diag_projection,
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

        # CIS IS ALWAYS SOLVED BY FULL DIAGONALIZATION.  Its H_PF is built
        # densely and stored unconditionally above (the `if self.ci_level ==
        # "cis"` branch at :4681 calls generatePFHMatrix with no direct-CI
        # alternative), so the matrix eigh needs is already resident and this
        # costs one O(N^3) solve, not new storage.
        #
        # The direct branch below no longer has a CIS path.  It used to call
        # the Python `Davidson`, which for CIS fell through to a DENSE
        # `S = np.einsum("ip,pq->iq", Q, H)` against that same H_PF -- i.e. a
        # subspace iteration converging to what eigh returns exactly.  The two
        # were measured equal to 2.3e-13 on all 8 asserted roots of both suite
        # decks BEFORE the branch was removed; rerun
        # validation/cis_full_diagonalization_check.py to reproduce that.
        # NOTE the shape change this implies for CIS callers: CIeigs goes from
        # (davidson_roots,) to (H_dim,) and CIvecs from (H_dim, davidson_roots)
        # to (H_dim, H_dim).  Both stay column-major in the eigenvector index,
        # so CIeigs[i] / CIvecs[:, i] are unchanged for every i that existed.
        #
        # `or self.ci_level == "cis"` rather than forcing
        # self.full_diagonalization True: the same idiom already guards :6546,
        # :6978, :7806 and :8058, and this way the attribute keeps reporting
        # what the user actually set.
        if self.full_diagonalization or self.ci_level == "cis":
            # call full diagonalization
            self.CIeigs, self.CIvecs = np.linalg.eigh(self.H_PF)
            self._initial_ci_diagnostics = dict(converged=True, reason='dense_eigensolver',
                                                iterations=None, residuals=None)
            self._final_ci_diagnostics = dict(self._initial_ci_diagnostics)

        # if doing direct method, call Davidson routine
        else:
            indim = self.davidson_indim * self.davidson_roots
            maxdim = self.davidson_maxdim * self.davidson_roots
            # print(H_dim, self.n_act_a,self.nmo)
            if indim > H_dim or maxdim > H_dim:
                # WAS: print(...) + sys.exit().  That is a SILENT SUCCESS -- the
                # process died mid-CASSCF with status 0 and no traceback, so any
                # runner scoring a return code read it as a completed job.  It
                # also fired on perfectly reasonable input: a small active space
                # (CAS(2,2) singlet spans 3 CSFs) cannot host the DEFAULT
                # subspace, and there is nothing wrong with such a request.
                #
                # A subspace larger than the space itself is not an error, it is
                # a request that has one obvious meaning: use the whole space.
                # So clamp, say so once, and carry on.  Note the alternative a
                # user is likely to reach for -- full_diagonalization=True -- is
                # NOT equivalent: it is CASCI, it skips the SA-CASSCF orbital
                # optimisation entirely and leaves self.eigenvals unset, so
                # nuclear_grad then dies with AttributeError.
                if H_dim < self.davidson_roots:
                    raise ValueError(
                        "CI space has dimension %d but %d roots were requested; "
                        "no solver can return more roots than the space has. "
                        "Reduce davidson_roots or enlarge the active space."
                        % (H_dim, self.davidson_roots)
                    )
                new_indim = min(indim, H_dim)
                new_maxdim = min(maxdim, H_dim)
                print(
                    "Davidson subspace clamped to the CI space: indim %d -> %d, "
                    "maxdim %d -> %d (H_dim %d, %d roots). The requested "
                    "subspace was larger than the space itself."
                    % (indim, new_indim, maxdim, new_maxdim, H_dim,
                       self.davidson_roots),
                    flush=True,
                )
                indim, maxdim = new_indim, new_maxdim
            if maxdim <= self.davidson_roots:
                # A subspace that cannot grow beyond the number of roots cannot
                # iterate, so there is no honest way to continue on this path.
                # Raise instead of dying quietly.
                #
                # NOTE for anyone tempted to "just diagonalise it here": on this
                # branch self.H_PF is the DUMMY np.eye(2) set at :4642 (the
                # direct solver never materialises the CI Hamiltonian), so a
                # dense eigh here would silently return garbage.  Making very
                # small active spaces work through this path means building the
                # real H, which is a separate change.
                raise ValueError(
                    "CI space has dimension %d, which is too small to run an "
                    "iterative solver for %d roots (the subspace cannot grow "
                    "beyond the roots). Enlarge the active space, ask for "
                    "fewer roots, or use a code path that builds the CI "
                    "Hamiltonian explicitly. NOTE full_diagonalization=True is "
                    "NOT a workaround: it is CASCI and skips the SA-CASSCF "
                    "orbital optimisation, leaving self.eigenvals unset."
                    % (H_dim, self.davidson_roots)
                )
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

                from qed_ci.results import ci_diagnostics
                self._initial_ci_diagnostics = ci_diagnostics(cfunctions, self.constint, self.constdouble, self.target_spin)
                self._initial_ci_diagnostics['threshold'] = self.davidson_threshold
                self._final_ci_diagnostics = dict(self._initial_ci_diagnostics)
                self.CIeigs = eigenvals
                self.CIvecs = eigenvecs

                print(
                    "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS",
                    flush=True,
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

                    excitation_rank = 0

                    # count the ground state only for now!
                    if i==0:
                        a_ref = np.array(alphalist)
                        b_ref = np.array(betalist)
                    
                    #for j in range(min(H_dim, 10)):
                    for j in range(H_dim):
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

                        c_i = eigenvecs[i][index[eigenvecs.shape[1] - j - 1]]

                        if i==0:
                            a_curr = np.array(alphalist)
                            b_curr = np.array(betalist)
                            
                            a_diff_count = np.sum(a_ref != a_curr)
                            b_diff_count = np.sum(b_ref != b_curr)
                            
                            # sum of alpha and beta difference instances is the excitation rank
                            excitation_rank = a_diff_count + b_diff_count
                            
                            # increment count of configurations with this excitation rank
                            self.casci_config_count_by_rank[excitation_rank] += 1
                            
                            # accumulate sum of squared weights for this excitation rank
                            self.casci_sum_squared_weight_by_rank[excitation_rank] += c_i ** 2


                        inactive_list = list(x for x in range(self.n_in_a))
                        alphalist2 = [x + self.n_in_a for x in alphalist]
                        # alphalist2[0:0] = inactive_list
                        betalist2 = [x + self.n_in_a for x in betalist]
                        # betalist2[0:0] = inactive_list
                        
                        # only print first 10
                        if j <= 10:
                            print(
                                "%20.12lf"
                                % c_i,
                                "%9.3d" % (index[eigenvecs.shape[1] - j - 1]),
                                "alpha",
                                alphalist2,
                                "   beta",
                                betalist2,
                                "%4.1d" % (photon_p),
                                "photon",
                                "excitation ranke",
                                excitation_rank
                            )

                print(" GOING TO COMPUTE 1-E PROPERTIES!", flush=True)
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
                            eigenvecs,
                            one_rdm,
                            self.table,
                            self.n_act_a,
                            self.n_act_orb,
                            self.n_in_a,
                            np1,
                            i,
                            j,
                            False
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
                            flush=True,
                        )
                        if i == j:
                            one_rdm = np.reshape(
                                one_rdm, (self.n_occupied, self.n_occupied)
                            )
                            rdm_eig = np.linalg.eigvalsh(one_rdm)
                            self.nat_obt_number[i, :] = rdm_eig[np.argsort(-rdm_eig)][:]
                        self.electronic_dipole_array[i, j, 0] = dipole_x
                        self.electronic_dipole_array[i, j, 1] = dipole_y
                        self.electronic_dipole_array[i, j, 2] = dipole_z

                # combine nuclear and electronic parts for the total dipole array
                self.dipole_array = (
                    self.electronic_dipole_array + self.nuclear_dipole_array
                )
                print(self.dipole_array)
                
                print("check total energy using active rdms")
                print(
                    "{:10s}".format("state"),
                    "{:20s}".format("active_one"),
                    "{:20s}".format("active_two"),
                    "{:20s}".format("eigenvalues"),
                    "{:20s}".format("total energies"),
                    "error",
                )
                active_twoeint = self.J[
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
                    c_build_active_rdm_sa(
                        eigenvecs,
                        self.D_tu,
                        self.D_tuvw,
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
                        self.Dpe_tu,
                        self.table,
                        self.n_act_a,
                        self.n_act_orb,
                        np1,
                        i,
                        i,
                        1.0,
                    )
                    active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu)
                    active_two_e_energy = 0.5 * np.dot(
                        active_twoeint.flatten(), self.D_tuvw
                    )
                    active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                        self.d_cmo[
                            self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
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
                        "{:20.12f}".format(active_one_e_energy),
                        "{:20.12f}".format(active_two_e_energy),
                        "{:20.12f}".format(eigenvals[i]),
                        "{:20.12f}".format(sum_energy),
                        "{:20.12f}".format(eigenvals[i] - sum_energy),
                    )
                
                self.build_state_average_rdms(eigenvecs)
                # test average energy
                print("test average energy")
                avg_energy = 0.0
                for i in range(self.davidson_roots):
                    avg_energy += self.weight[i] * eigenvals[i]
                    # print(self.weight[i], eigenvals[i], self.weight[i] * eigenvals[i])
                active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                active_two_e_energy = 0.5 * np.dot(
                    active_twoeint.flatten(), self.D_tuvw_avg
                )
                active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                    self.d_cmo[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ].flatten(),
                    self.Dpe_tu_avg,
                )
                ci_dependent_energy = self.calculate_ci_dependent_energy(
                    eigenvecs, self.d_cmo
                )
                sum_energy = (
                    active_one_e_energy
                    + active_two_e_energy
                    + active_one_pe_energy
                    + self.E_core
                    + self.Enuc
                    + self.d_c
                    + ci_dependent_energy
                )
                print(
                    "average energy zero" "{:20.12f}".format(sum_energy),
                    "{:20.12f}".format(avg_energy),
                )
                # print(
                #    "{:20.12f}".format(sum_energy),
                #    "{:20.12f}".format(active_one_e_energy),
                #    "{:20.12f}".format(active_two_e_energy),
                #    "{:20.12f}".format(active_one_pe_energy),
                #    "{:20.12f}".format(self.E_core),
                #    "{:20.12f}".format(self.Enuc),
                #    "{:20.12f}".format(self.d_c),
                #    "{:20.12f}".format(ci_dependent_energy),
                # )

                self.n_virtual = self.nmo - self.n_occupied
                n_ai = self.n_in_a * self.n_act_orb
                n_vi = self.n_in_a * self.n_virtual
                n_va = self.n_act_orb * self.n_virtual
                # build index map
                self.index_map_size = n_ai + n_vi + n_va
                self.index_map = np.zeros((self.index_map_size, 2), dtype=int)
                index_count = 0
                for r in range(self.nmo):
                    for s in range(r + 1, self.nmo):
                        if r < self.n_in_a and s < self.n_in_a:
                            continue
                        if (
                            self.n_in_a <= r < self.n_occupied
                            and self.n_in_a <= s < self.n_occupied
                        ):
                            continue
                        if r >= self.n_occupied and s >= self.n_occupied:
                            continue
                        self.index_map[index_count][0] = s
                        self.index_map[index_count][1] = r
                        index_count += 1

                # QED_ORBITAL_SYMMETRY: drop the rotations that would mix two
                # different symmetry species.  This is the constraint itself; the
                # matching filter appears at every other site that re-enumerates
                # this same list (they do not read index_map, they rebuild it).
                if _QED_ORBITAL_SYMMETRY and self._sym_labels is not None:
                    import orbital_symmetry as _orbsym

                    _n_full = self.index_map_size
                    self.index_map = _orbsym.filtered_index_map(
                        self._sym_labels, self.n_in_a, self.n_occupied, self.nmo
                    )
                    self.index_map_size = len(self.index_map)
                    _orbsym.report(
                        self._sym_labels, self._sym_report[0], self._sym_report[1],
                        self.n_in_a, self.n_occupied, _n_full, self.index_map_size,
                    )

                self.index_map1 = self.index_map.astype("int32")

                self.index_map_pq = np.zeros(
                    (self.nmo * (self.nmo + 1) // 2, 2), dtype=np.int32
                )
                index_count = 0
                for r in range(self.nmo):
                    for s in range(r, self.nmo):
                        self.index_map_pq[index_count][0] = s
                        self.index_map_pq[index_count][1] = r
                        index_count += 1
                self.index_map_kl = np.zeros(
                    (self.n_occupied * (self.n_occupied + 1) // 2, 2), dtype=np.int32
                )
                index_count = 0
                for r in range(self.n_occupied):
                    for s in range(r, self.n_occupied):
                        self.index_map_kl[index_count][0] = s
                        self.index_map_kl[index_count][1] = r
                        index_count += 1
                self.index_map_ab = np.zeros(
                    (self.n_virtual * (self.n_virtual + 1) // 2, 2), dtype=np.int32
                )
                index_count = 0
                for r in range(self.n_virtual):
                    for s in range(r, self.n_virtual):
                        self.index_map_ab[index_count][0] = s
                        self.index_map_ab[index_count][1] = r
                        index_count += 1
                self.H1 = copy.deepcopy(self.H_spatial2)
                self.d_cmo1 = copy.deepcopy(self.d_cmo)
                self.eigenvecs = eigenvecs
                # if self.n_in_a == 0 and self.n_act_orb == self.nmo:
                #    pass
                # else:
                #    self.ah_orbital_optimization(eigenvecs, c_get_roots)
                # self.bfgs_orbital_optimization(eigenvecs, c_get_roots)
                
                if self.casscf_optimization == True:
                    ####back up wmk algorithm
                    if self.n_in_a == 0 and self.n_act_orb == self.nmo:
                        print(
                            "The requested casci level is equivalent to fci so no orbital optimization will be performed"
                        )
                        pass
                    else:
                        start = timer()
                        macroiteration = 0
                        self.U_total = np.eye(self.nmo)
                        old_avg_energy = 0
                        new_avg_energy = avg_energy
                        convergence = 0
                        _dump_cpp_casscf_validation_case(
                            "macroiteration_bootstrap",
                            H_spatial2=self.H_spatial2,
                            d_cmo=self.d_cmo,
                            J=self.J,
                            K=self.K,
                            twoeint=self.twoeint,
                            eigenvecs=eigenvecs,
                            avg_energy=np.array([avg_energy]),
                            E_core=np.array([self.E_core]),
                            D_tu_avg=self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                            D_tuvw_avg=self.D_tuvw_avg.reshape(
                                (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
                            ),
                            Dpe_tu_avg=self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                            weight=self.weight,
                            dims=np.array(
                                [self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]
                            ),
                            config_int=np.array(
                                [
                                    self.n_act_a,
                                    self.N_p,
                                    self.num_det,
                                    self.davidson_roots,
                                    self.davidson_indim,
                                    self.davidson_maxdim,
                                    self.davidson_maxiter,
                                    1 if self.ignore_dse_terms else 0,
                                ]
                            ),
                            config_double=np.array(
                                [
                                    self.omega,
                                    self.Enuc,
                                    self.d_c,
                                    self.d_exp,
                                    self.davidson_threshold,
                                    self.target_spin,
                                ]
                            ),
                        )
                        self._macro_diagnostics.update(applicable=True, converged=False, reason='iteration_limit',
                                                       thresholds={'energy_Eh': _QED_MACRO_ENERGY_CONV,
                                                                   'gradient': _QED_GRAD_CONV})
                        while macroiteration < _QED_MACRO_MAXITER:
                            if macroiteration > 0:
                                # print("U total")
                                # self.printA(self.U_total)
                                occupied_J = np.zeros(
                                    (
                                        self.n_occupied,
                                        self.n_occupied,
                                        self.n_occupied,
                                        self.n_occupied,
                                    )
                                )
                                occupied_J[
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
                                active_twoeint = self.J[
                                    self.n_in_a : self.n_occupied,
                                    self.n_in_a : self.n_occupied,
                                    self.n_in_a : self.n_occupied,
                                    self.n_in_a : self.n_occupied,
                                ]
                                self.H_diag3 = np.zeros(H_dim)
                                fock_core = copy.deepcopy(self.H_spatial2)
                                fock_core += 2.0 * np.einsum(
                                    "jjrs->rs",
                                    self.J[: self.n_in_a, : self.n_in_a, :, :],
                                )
                                fock_core -= np.einsum(
                                    "jjrs->rs",
                                    self.K[: self.n_in_a, : self.n_in_a, :, :],
                                )

                                active_fock_core = copy.deepcopy(
                                    fock_core[
                                        self.n_in_a : self.n_occupied,
                                        self.n_in_a : self.n_occupied,
                                    ]
                                )
                                occupied_fock_core = np.zeros(
                                    (self.n_occupied, self.n_occupied)
                                )
                                occupied_fock_core[
                                    self.n_in_a : self.n_occupied,
                                    self.n_in_a : self.n_occupied,
                                ] = copy.deepcopy(active_fock_core)
                                occupied_d_cmo = np.zeros(
                                    (self.n_occupied, self.n_occupied)
                                )
                                occupied_d_cmo = copy.deepcopy(
                                    self.d_cmo[: self.n_occupied, : self.n_occupied]
                                )
                                gkl2 = copy.deepcopy(active_fock_core)
                                gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint)

                                occupied_J = occupied_J.reshape(
                                    self.n_occupied * self.n_occupied,
                                    self.n_occupied * self.n_occupied,
                                )
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
                                    self.E_core,
                                    self.omega,
                                    self.Enuc,
                                    self.d_c,
                                    self.Y,
                                    self.target_spin,
                                )
                                d_diag = 2.0 * np.einsum(
                                    "ii->", self.d_cmo[: self.n_in_a, : self.n_in_a]
                                )
                                self.constdouble[3] = self.d_exp - d_diag
                                # self.constdouble[4] = 1e-5
                                self.constdouble[4] = self.davidson_threshold
                                self.constdouble[5] = self.E_core
                                self.constint[8] = self.davidson_maxiter
                                eigenvals = np.zeros((self.davidson_roots))
                                # eigenvecs = np.zeros((self.davidson_roots, H_dim))
                                # eigenvecs[:,:] = 0.0
                                # print("heyhey5", eigenvecs)
                                # --- cold-start-between-macroiterations experiment (QED_COLD_START_CI) ---
                                # Default off => warm start (casscf=True, Q = previous macro's CI
                                # vector, which is a stale eigenvector of the previous macro's
                                # Hamiltonian).  When on => cold start (casscf=False): a fresh
                                # Algorithm-3 spin-adapted guess each macro, with a configurable
                                # large guess subspace (QED_COLD_START_INDIM / _MAXDIM).  This is now
                                # cheap thanks to the direct Slater-Condon guess-block build in
                                # ci_solver.c.  constint[5]/[6] (indim/maxdim) are set only for this
                                # cold solve and restored immediately after.
                                _cold = int(os.environ.get("QED_COLD_START_CI", "0"))
                                if _cold:
                                    _sv_indim = int(self.constint[5])
                                    _sv_maxdim = int(self.constint[6])
                                    self.constint[5] = int(os.environ.get("QED_COLD_START_INDIM", str(_sv_indim)))
                                    self.constint[6] = int(os.environ.get("QED_COLD_START_MAXDIM", str(_sv_maxdim)))
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
                                    (not _cold),
                                    self.target_spin,
                                )
                                self._final_ci_diagnostics = ci_diagnostics(cfunctions, self.constint, self.constdouble, self.target_spin)
                                self._final_ci_diagnostics['threshold'] = self.davidson_threshold
                                if _cold:
                                    self.constint[5] = _sv_indim
                                    self.constint[6] = _sv_maxdim
                                self._dump_ci_vectors(
                                    eigenvecs, "betweenmacro", macroiteration
                                )
                                old_avg_energy = new_avg_energy
                                avg_energy = 0.0
                                for i in range(self.davidson_roots):
                                    avg_energy += self.weight[i] * eigenvals[i]
                                print("avg energy", macroiteration, avg_energy)
                                new_avg_energy = avg_energy
                                print(
                                    "average energy at the start of macroiteration",
                                    avg_energy,
                                    flush = True
                                )
                                self.build_state_average_rdms(eigenvecs)

                            print("Macroiteration", macroiteration,
                                    "old CI energy", old_avg_energy,
                                    "new CI energy", new_avg_energy,
                                    flush = True)
                            # QED_TRUE_ENERGY_REACTIVE: arm on the first real
                            # macro-energy RISE.  macroiteration > 0 because
                            # old_avg_energy is 0.0 before the first CI solve.
                            if (_QED_TRUE_ENERGY_REACTIVE and macroiteration > 0
                                    and not getattr(self, "_true_energy_armed", False)
                                    and new_avg_energy >
                                    old_avg_energy + _QED_TRUE_ENERGY_REACTIVE_TOL):
                                self._true_energy_armed = True
                                print("REACTIVE: macro energy ROSE by",
                                      new_avg_energy - old_avg_energy,
                                      "-- arming the true-energy test from here on",
                                      flush=True)
                            if (_QED_MACRO_ENERGY_CONV > 0.0
                                    and np.abs(new_avg_energy - old_avg_energy)
                                        < _QED_MACRO_ENERGY_CONV):
                                convergence = 1
                                self._macro_trigger = dict(reason='energy', iteration=macroiteration)
                            # QED_GRAD_CONV (item 244): OR-ed gradient criterion.
                            # _macro_entry_gnorm is the gradient at the start of
                            # the PREVIOUS macro's orbital work, so the test lags
                            # by one macro -- deliberately conservative.
                            _gn = getattr(self, "_macro_entry_gnorm", None)
                            if _QED_GRAD_CONV > 0.0 and _gn is not None:
                                print("macro gradient norm", _gn,
                                      "threshold", _QED_GRAD_CONV, flush=True)
                                if _gn < _QED_GRAD_CONV:
                                    convergence = 1
                                    _trigger = getattr(self, '_macro_trigger', {})
                                    _both = (_trigger.get('iteration') == macroiteration and _trigger.get('reason') == 'energy')
                                    self._macro_trigger = dict(reason='energy_and_gradient' if _both else 'gradient',
                                                               iteration=macroiteration)
                                    print("CONVERGED ON GRADIENT CRITERION", flush=True)
                            # Re-arm the capture AFTER the test: the value the test
                            # reads must be the first gradient of the PREVIOUS
                            # macro's orbital work (the one-macro lag this criterion
                            # is calibrated for).  Resetting at the TOP of the loop
                            # instead leaves _gn None at every test and the
                            # criterion silently never fires -- measured, item 245.
                            self._macro_diagnostics['history'].append(dict(
                                iteration=macroiteration, energy=float(new_avg_energy),
                                energy_change=float(new_avg_energy-old_avg_energy),
                                gradient_norm=None if _gn is None else float(_gn)))
                            self._macro_diagnostics.update(iterations=macroiteration+1,
                                energy_change=float(new_avg_energy-old_avg_energy),
                                gradient_norm=None if _gn is None else float(_gn))
                            if macroiteration > 0 and convergence == 1:
                                self._macro_diagnostics.update(converged=True,
                                    reason=self._macro_trigger['reason'],
                                    trigger_iteration=self._macro_trigger['iteration'])
                            self._macro_entry_gnorm = None   # QED_GRAD_CONV
                            if macroiteration >= 1000:
                                self.casscf_converged = False
                                print(f"WARNING: SA-CASSCF did NOT converge after 30 iterations")
 
                            if macroiteration > 0 and convergence == 1:
                                self.casscf_converged = True
                                self.CASSCFeigs = eigenvals
                                self.CASSCFvecs = eigenvecs
                                _dump_cpp_casscf_validation_case(
                                    "macroiteration_convergence",
                                    avg_energy=np.array([new_avg_energy]),
                                    macroiteration=np.array([macroiteration]),
                                )

                                print(
                                    "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
                                )
                                Y = np.zeros(
                                    self.n_act_a
                                    * (self.n_act_orb - self.n_act_a + 1)
                                    * 3,
                                    dtype=np.int32,
                                )
                                c_graph(self.n_act_a, self.n_act_orb, Y)
                                np1 = self.N_p + 1
                                singlet_count = 0
                                triplet_count = 0
                                for i in range(eigenvecs.shape[0]):
                                    total_spin = self.check_total_spin(
                                        eigenvecs[i : (i + 1), :]
                                    )
                                    print(
                                        "STATE",
                                        i,
                                        "ENERGY =",
                                        eigenvals[i],
                                        "<S^2>=",
                                        total_spin,
                                        "WEIGHT =",
                                        self.weight[i],
                                        end="",
                                    )
                                    if np.abs(total_spin) < 1e-5:
                                        singlet_count += 1
                                        print("\tSINGLET", singlet_count)
                                    elif np.abs(total_spin - 2.0) < 1e-5:
                                        triplet_count += 1
                                        print("\tTRIPLET", triplet_count)
                                    elif np.abs(total_spin - 6.0) < 1e-5:
                                        print("\tQUINTET")

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
                                    a0 = c_index_to_string(
                                        Ia0, self.n_act_a, self.n_act_orb, Y
                                    )
                                    b0 = c_index_to_string(
                                        Ib0, self.n_act_a, self.n_act_orb, Y
                                    )

                                    alphalist = Determinant.obtBits2ObtIndexList(a0)
                                    betalist = Determinant.obtBits2ObtIndexList(b0)

                                    # only if i==0:
                                    if i == 0:
                                        a_ref = np.array(alphalist)
                                        b_ref = np.array(betalist)

                                    #for j in range(min(H_dim, 10)):
                                    for j in range(H_dim):
                                        Idet = (
                                            index[eigenvecs.shape[1] - j - 1]
                                            % self.num_det
                                        )
                                        photon_p = (
                                            index[eigenvecs.shape[1] - j - 1] - Idet
                                        ) // self.num_det
                                        Ib = Idet % self.num_alpha
                                        Ia = Idet // self.num_alpha
                                        a = c_index_to_string(
                                            Ia, self.n_act_a, self.n_act_orb, Y
                                        )
                                        b = c_index_to_string(
                                            Ib, self.n_act_a, self.n_act_orb, Y
                                        )

                                        alphalist = Determinant.obtBits2ObtIndexList(a)
                                        betalist = Determinant.obtBits2ObtIndexList(b)
                                        
                                        # get weight of current configh
                                        c_i = eigenvecs[i][index[eigenvecs.shape[1] - j - 1]]
                                        
                                        if i == 0:
                                            a_curr = np.array(alphalist)
                                            b_curr = np.array(betalist)

                                            #sum up all instances where the occupation of the current determinant differs from the reference for alpha and beta strings
                                            a_diff_count = np.sum(a_ref != a_curr)
                                            b_diff_count = np.sum(b_ref != b_curr)
                                            
                                            # sum of alpha and beta difference instances is the excitation rank
                                            excitation_rank = a_diff_count + b_diff_count
                                            # increment count of configurations with this excitation rank
                                            self.casscf_config_count_by_rank[excitation_rank] += 1
                                        
                                            # accumulate sum of squared weights for this excitation rank
                                            self.casscf_sum_squared_weight_by_rank[excitation_rank] += c_i ** 2
                                    

                                        inactive_list = list(
                                            x for x in range(self.n_in_a)
                                        )
                                        alphalist2 = [
                                            x + self.n_in_a for x in alphalist
                                        ]
                                        # alphalist2[0:0] = inactive_list
                                        betalist2 = [x + self.n_in_a for x in betalist]
                                        # betalist2[0:0] = inactive_list

                                        # only print first 10
                                        if j <= 10:
                                            print(
                                                "%20.12lf"
                                                % (
                                                    c_i
                                                ),
                                                "%9.3d"
                                                % (index[eigenvecs.shape[1] - j - 1]),
                                                "alpha",
                                                alphalist2,
                                                "   beta",
                                                betalist2,
                                                "%4.1d" % (photon_p),
                                                "photon",
                                                "excitation rank",
                                                excitation_rank
                                            )

                                print("OPTIMIZATION CONVERGED", flush=True)
                                print("avg energy final", macroiteration, avg_energy)
                                self.opt_C = np.einsum("pq,qr->pr", self.C, self.U_total)
                                if self.save_orbital == True:
                                    # print(new_C)
                                    if self.config.orbital_output_file:
                                        np.savetxt(self.config.orbital_output_file, self.opt_C)
                                        self._generated_files["orbital_output_file"] = self.config.orbital_output_file
                                    self.build_state_average_rdms(eigenvecs)
                                    active_rdm = self.D_tu_avg.reshape(self.n_act_orb, self.n_act_orb)

                                    # Optional but safe: enforce symmetry numerically before diagonalizing
                                    active_rdm = 0.5 * (active_rdm + active_rdm.T)

                                    _eig_act, _vec_act = np.linalg.eigh(active_rdm)
                                    _idx_act = _eig_act.argsort()[::-1]

                                    # Build full transformation: identity except in the active block
                                    no_vec = np.eye(self.nmo)
                                    no_vec[self.n_in_a:self.n_occupied, self.n_in_a:self.n_occupied] = _vec_act[:, _idx_act]

                                    self.nat_orbs = np.dot(self.opt_C, no_vec)
                                    self.noocs = np.concatenate([
                                        2.0 * np.ones(self.n_in_a),
                                        _eig_act[_idx_act],
                                        np.zeros(self.nmo - self.n_occupied)
                                    ])

                                    if self.config.occupation_output_file:
                                        np.savetxt(self.config.occupation_output_file, self.noocs)
                                        self._generated_files["occupation_output_file"] = self.config.occupation_output_file
                                    if self.config.natural_orbital_output_file:
                                        np.savetxt(self.config.natural_orbital_output_file, self.nat_orbs)
                                        self._generated_files["natural_orbital_output_file"] = self.config.natural_orbital_output_file
 
                              
                                _mu_x_spin = np.einsum(
                                    "uj,vi,uv",
                                    self.opt_C[:, : self.n_occupied],
                                    self.opt_C[:, : self.n_occupied],
                                    self.mu_x_ao,
                                )
                                _mu_y_spin = np.einsum(
                                    "uj,vi,uv",
                                    self.opt_C[:, : self.n_occupied],
                                    self.opt_C[:, : self.n_occupied],
                                    self.mu_y_ao,
                                )
                                _mu_z_spin = np.einsum(
                                    "uj,vi,uv",
                                    self.opt_C[:, : self.n_occupied],
                                    self.opt_C[:, : self.n_occupied],
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
                                            eigenvecs,
                                            one_rdm,
                                            self.table,
                                            self.n_act_a,
                                            self.n_act_orb,
                                            self.n_in_a,
                                            np1,
                                            i,
                                            j,
                                            False
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
                                            flush=True,
                                        )
                                        if i == j:
                                            one_rdm = np.reshape(
                                                one_rdm, (self.n_occupied, self.n_occupied)
                                            )
                                            rdm_eig = np.linalg.eigvalsh(one_rdm)
                                            self.nat_obt_number[i, :] = rdm_eig[np.argsort(-rdm_eig)][:]
                                        self.electronic_dipole_array[i, j, 0] = dipole_x
                                        self.electronic_dipole_array[i, j, 1] = dipole_y
                                        self.electronic_dipole_array[i, j, 2] = dipole_z

                                # combine nuclear and electronic parts for the total dipole array
                                self.dipole_array = (
                                    self.electronic_dipole_array + self.nuclear_dipole_array
                                )
                                print(self.dipole_array)
                                break
                            self.avg_energy = avg_energy
                            if macroiteration > 0 and self.n_in_a > 0:
                                start1 = timer()
                                # self.internal_optimization2(avg_energy, eigenvecs)
                                self.internal_optimization3(avg_energy, eigenvecs)
                                end1 = timer()
                                print("internal optimization took", end1 - start1)

                            ###print("LETS TEST THE TWO WAYS TO CALCULATE SECOND ORDER ENERGY AGAIN")
                            ###A = np.zeros((rot_dim, rot_dim))
                            ###G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                            ###print("initial e_core", self.E_core)
                            ###self.build_intermediates(eigenvecs, A, G, True)
                            ###A2 = np.zeros((rot_dim, rot_dim))
                            ###G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
                            ###self.build_intermediates2(eigenvecs, A2, G2, True)
                            ###print("test second order energy and ci updated integrals")
                            ###exact_t_energy = self.microiteration_exact_energy(self.U_delta, A, G)
                            ###print("exact_energy from second order expansion", exact_t_energy + avg_energy)
                            ###active_twoeint = np.zeros((self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb))
                            ###active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
                            ###d_cmo = np.zeros((self.nmo, self.nmo))
                            ###self.microiteration_ci_integrals_transform(self.U_delta, eigenvecs, d_cmo, active_fock_core, active_twoeint)
                            ###active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                            ###active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                            ###active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
                            ###ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
                            ###sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core +
                            ###        self.Enuc + self.d_c + ci_dependent_energy)
                            ###print("gfhgy",
                            ###    "{:20.12f}".format(sum_energy),
                            ###    "{:20.12f}".format(active_one_e_energy),
                            ###    "{:20.12f}".format(active_two_e_energy),
                            ###    "{:20.12f}".format(self.E_core),
                            ###    "{:20.12f}".format(active_one_pe_energy),
                            ###    "{:20.12f}".format(self.Enuc),
                            ###)

                            np.set_printoptions(precision=14)
                            if macroiteration == 0:
                                convergence_threshold = 1e-3
                            else:
                                convergence_threshold = 1e-4
                            print("avg energy", macroiteration, self.avg_energy)
                            U0 = np.eye(self.nmo)
                            # print("heyhey3",eigenvecs)
                            start1 = timer()
                            self.microiteration_optimization6(
                                U0, eigenvecs, c_get_roots, convergence_threshold
                            )
                            end1 = timer()
                            print("microiteration took", end1 - start1)
                            # print("heyhey",eigenvecs)
                            # print("u2",self.U2)
                            print("full transformation test")

                            R = 0.5 * (self.U2 - self.U2.T)
                            Rai = np.zeros((self.n_act_orb, self.n_in_a))
                            Rai[:, :] = R[self.n_in_a : self.n_occupied, : self.n_in_a]
                            print(
                                "norm of internal step", np.linalg.norm(Rai), flush=True
                            )
                            # EXPERIMENT (env-gated, default-no-op): repeat the
                            # internal-rotation correction up to _max_inner times
                            # instead of exactly once. Werner's WMK paper allows
                            # either "repeat until R_ij ~ 0" or the cheaper
                            # "once per macroiteration" approximation; this code
                            # implements once, and QED_INTERNAL_RESTART_MAX=1
                            # (default) reproduces that exactly. Set >1 to loop.
                            # Each pass: (1) extract internal Rai from the
                            # current R, (2) absorb it EXACTLY into the basis via
                            # the internal integral transform (all orders, not
                            # the 2nd-order model -- this is why it can beat the
                            # x^4-wrong micro), (3) restart the micro from the
                            # external part only, (4) recompute R.
                            _max_inner = int(os.environ.get("QED_INTERNAL_RESTART_MAX", "1"))
                            _inner = 0
                            while np.linalg.norm(Rai) > 1e-4 and _inner < _max_inner:
                                print(
                                    "RESTART MICROITERATION TO CORRECT INTERNAL ROTATION",
                                    _inner, flush=True,
                                )
                                # (1)+(2): internal-only unitary from the current
                                # Rai, absorbed exactly into J/K/H/d_cmo.
                                Rai_int = np.zeros((self.n_act_orb, self.n_in_a))
                                Rai_int[:, :] = R[self.n_in_a : self.n_occupied, : self.n_in_a]
                                Rvi = np.zeros((self.n_virtual, self.n_in_a))
                                Rva = np.zeros((self.n_virtual, self.n_act_orb))
                                self.build_unitary_matrix(Rai_int, Rvi, Rva)
                                self.K = np.ascontiguousarray(self.K)
                                start1 = timer()
                                c_full_transformation_internal_optimization(
                                    self.U_delta,
                                    self.J,
                                    self.K,
                                    self.H_spatial2,
                                    self.d_cmo,
                                    self.J,
                                    self.K,
                                    self.H_spatial2,
                                    self.d_cmo,
                                    self.index_map_ab,
                                    self.index_map_kl,
                                    self.nmo,
                                    self.n_occupied,
                                )
                                end1 = timer()
                                print(
                                    "full internal transformation took", end1 - start1
                                )
                                self.U_total = np.einsum(
                                    "pq,qs->ps", self.U_total, self.U_delta
                                )
                                # (3): restart micro from the external part of
                                # the CURRENT R (recomputed each pass, not held
                                # at the first pass's value).
                                Rai0 = np.zeros((self.n_act_orb, self.n_in_a))
                                Rvi[:, :] = R[self.n_occupied :, : self.n_in_a]
                                Rva[:, :] = R[
                                    self.n_occupied :, self.n_in_a : self.n_occupied
                                ]
                                self.build_unitary_matrix(Rai0, Rvi, Rva)
                                start1 = timer()
                                self.microiteration_optimization6(
                                    self.U_delta,
                                    eigenvecs,
                                    c_get_roots,
                                    convergence_threshold,
                                )
                                end1 = timer()
                                print("microiteration took", end1 - start1)
                                # (4): recompute the full R and its internal part.
                                R = 0.5 * (self.U2 - self.U2.T)
                                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                                Rai[:, :] = R[
                                    self.n_in_a : self.n_occupied, : self.n_in_a
                                ]
                                _inner += 1
                                print(
                                    "norm of internal step after", _inner,
                                    np.linalg.norm(Rai), flush=True,
                                )
                            print(
                                "norm of internal step after",
                                np.linalg.norm(Rai),
                                flush=True,
                            )
                            # print("u2 after", self.U2,flush = True)

                            temp8 = np.zeros((self.nmo, self.nmo))
                            temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                            self.H_spatial2[:, :] = np.einsum(
                                "ps,pr->rs", temp8, self.U2
                            )
                            d_cmo0 = copy.deepcopy(self.d_cmo)
                            temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                            self.d_cmo[:, :] = np.einsum("ps,pr->rs", temp8, self.U2)

                            self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)

                            # print(self.U_total)

                            #JJ = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                            #KK = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
                            #self.full_transformation_macroiteration(self.U_total, JJ, KK)
                            if self.density_fitting == False: 
                                start1 = timer()
                                self.K = np.ascontiguousarray(self.K)
                                self._macro_JK(self.U_total, self.J, self.K)
                                # self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                                end1 = timer()
                                print("full JK transformation took", end1 - start1, flush = True)
                            #print("tvhj", np.allclose(KK,self.K, rtol=1e-10,atol=1e-10))
                            #print("oins", np.allclose(JJ,self.J, rtol=1e-10,atol=1e-10))
                            else:
                                start1 = timer()
                                self.J, self.K = self.transform_JK_with_df(
                                    self.B_mo,
                                    self.U_total,
                                    self.n_occupied,
                                    ignore_dse_terms=self.ignore_dse_terms
                                )
                                end1 = timer()
                                #print("tvhj", np.allclose(KK,self.KKK, rtol=1e-10,atol=1e-10))
                                #print("oins", np.allclose(JJ,self.JJJ, rtol=1e-10,atol=1e-10))
                                print("full JK transformation took", end1 - start1, flush = True)
                            ##print("tc5k", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
                            ##print("p0ba", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
                            active_twoeint = self.J[
                                self.n_in_a : self.n_occupied,
                                self.n_in_a : self.n_occupied,
                                self.n_in_a : self.n_occupied,
                                self.n_in_a : self.n_occupied,
                            ]
                            self.fock_core = copy.deepcopy(self.H_spatial2)
                            self.fock_core += 2.0 * np.einsum(
                                "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :]
                            )
                            self.fock_core -= np.einsum(
                                "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :]
                            )

                            self.E_core = 0.0
                            self.E_core += np.einsum(
                                "jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a]
                            )
                            self.E_core += np.einsum(
                                "jj->", self.fock_core[: self.n_in_a, : self.n_in_a]
                            )
                            print("ahi", flush = True)
                            # print(eigenvecs)
                            active_fock_core = np.zeros(
                                (self.n_act_orb, self.n_act_orb)
                            )
                            active_fock_core[:, :] = self.fock_core[
                                self.n_in_a : self.n_occupied,
                                self.n_in_a : self.n_occupied,
                            ]
                            active_one_e_energy = np.dot(
                                active_fock_core.flatten(), self.D_tu_avg
                            )
                            active_two_e_energy = 0.5 * np.dot(
                                active_twoeint.flatten(), self.D_tuvw_avg
                            )
                            active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                                self.d_cmo[
                                    self.n_in_a : self.n_occupied,
                                    self.n_in_a : self.n_occupied,
                                ].flatten(),
                                self.Dpe_tu_avg,
                            )
                            ci_dependent_energy = self.calculate_ci_dependent_energy(
                                eigenvecs, self.d_cmo
                            )
                            sum_energy = (
                                active_one_e_energy
                                + active_two_e_energy
                                + active_one_pe_energy
                                + self.E_core
                                + self.Enuc
                                + self.d_c
                                + ci_dependent_energy
                            )
                            print(
                                "gfhgy end macroiteration",
                                "{:20.12f}".format(sum_energy),
                                "{:20.12f}".format(active_one_e_energy),
                                "{:20.12f}".format(active_two_e_energy),
                                "{:20.12f}".format(self.E_core),
                                "{:20.12f}".format(active_one_pe_energy),
                                "{:20.12f}".format(self.Enuc),
                            )
                            print("ahi", flush = True)
                            print("end one macroiteration")
                            #avg_energy = sum_energy
                            self.occupied_K = copy.deepcopy(
                                self.K[:, :, : self.n_occupied, : self.n_occupied]
                            )
                            self.occupied_J = copy.deepcopy(
                                self.J[:, :, : self.n_occupied, : self.n_occupied]
                            )
                            self.occupied_h1 = copy.deepcopy(
                                self.H_spatial2[: self.n_occupied, : self.n_occupied]
                            )
                            self.occupied_d_cmo = copy.deepcopy(
                                self.d_cmo[: self.n_occupied, : self.n_occupied]
                            )
                            self.occupied_fock_core = copy.deepcopy(
                                self.fock_core[: self.n_occupied, : self.n_occupied]
                            )
                            #new_avg_energy = avg_energy

                            macroiteration += 1
                        end = timer()
                        self.eigenvecs = eigenvecs
                        self.eigenvals = eigenvals

                        print("optimization took", end - start)

            # The `elif self.ci_level == "cis"` branch that stood here called
            # the Python `Davidson` on self.H_PF.  It is UNREACHABLE as of
            # 2026-09-04: CIS is routed to full diagonalization at :4754, so
            # control never enters this else.  `Davidson` moved to
            # legacy_python_ci.py with the rest of the retired pure-Python CI
            # kernels -- see that module for the reachability evidence.
            t_dav_end = time.time()
            # print(f" Completed Davidson iterations in {t_dav_end - t_H_build} seconds", flush = True)

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

        if "casscf_optimization" in cavity_dictionary:
            self.casscf_optimization = cavity_dictionary["casscf_optimization"]
        else:
            self.casscf_optimization = True

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

            # initialize arrays that will store the number of configurations by excitation rank for CASCI and CASSCF
            self.casscf_config_count_by_rank = np.zeros(self.n_act_el+1)
            self.casci_config_count_by_rank = np.zeros(self.n_act_el+1)

            # initialize arrays that will store the sum of squared weights by excitation rank for CASCI and CASSCF
            self.casscf_sum_squared_weight_by_rank = np.zeros(self.n_act_el+1)
            self.casci_sum_squared_weight_by_rank = np.zeros(self.n_act_el+1)

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
                # "no" is the same unadapted path as omitting the key entirely
                # (target_spin < 0 routes get_roots to the plain davidson).
                # Without this branch, passing it explicitly left target_spin
                # unset and died later with an AttributeError at its first use.
                if self.spin_adaptation == "no":
                    self.target_spin = -1.0
                if not hasattr(self, "target_spin"):
                    print(
                        f"unrecognized spin_adaptation '{self.spin_adaptation}';"
                        " expected one of: singlet, triplet, quintet, no"
                    )
                    exit()
            else:
                self.spin_adaptation = "no"
                self.target_spin = -1.0
                # print(self.target_spin)

        if "casscf_weight" in cavity_dictionary:
            self.weight = cavity_dictionary["casscf_weight"]
        else:
            self.weight = np.full(self.davidson_roots, 1.0)
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
        self.weight = self.weight / np.sum(self.weight)
        print("weight", self.weight)

    def parseArrays(self, cqed_rhf_dict):
        # grab quantities from cqed_rhf_dict that apply to both number state and coherent state bases
        self._scf_diagnostics = cqed_rhf_dict.get('SCF DIAGNOSTICS', dict(converged=None, reason='unavailable'))
        wfn = cqed_rhf_dict["PSI4 WFN"]
        self.resolved_options["dimensions"] = self.config.validate_dimensions(wfn.nmo(), wfn.nalpha(), wfn.nbeta())
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

        self.d_hf = copy.deepcopy(self.d_cmo)

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

        self.C_hf = copy.deepcopy(self.C)

        # collect rhf wfn object as dictionary
        wfn_dict = psi4.core.Wavefunction.to_file(wfn)
        wfn_dict["matrix"]["Ca"] = self.C
        wfn_dict["matrix"]["Cb"] = self.C
        wfn = psi4.core.Wavefunction.from_file(wfn_dict)
        self.Ca_hf = wfn.Ca()

        ##random orbital guess
        # Validation tooling: a seeded random unitary rotation of the initial
        # orbitals, replacing the commented-out unseeded block that used to
        # sit here. Gated behind an env var, so it is a no-op on every normal
        # run and on the whole C++ validation path -- set
        # QED_RANDOM_ORBITAL_SEED=<int> to enable.
        #
        # Why this is kept: canonical-guess CASSCF converges in ~8
        # macroiterations on the small test systems, which is too easy to
        # discriminate between convergence strategies. The random-guess regime
        # is what reproduces the hard >10-macroiteration cases, and this is
        # the only reproducer for them. cpp_casscf/validation/run_mgh_case.py
        # and run_qn_case.py both depend on it. See cpp_casscf/README.md, "The
        # QN/BFGS convergence-variant experiment", for what it was used for
        # and what that experiment concluded.
        _rand_orb_seed = os.environ.get("QED_RANDOM_ORBITAL_SEED")
        if _rand_orb_seed is not None:
            # Fix the MO sign convention BEFORE rotating. psi4's threaded SCF
            # returns the same orbitals every run (occupied subspaces agree to
            # ~1e-14) but with run-to-run-varying column SIGNS -- measured at
            # OMP_NUM_THREADS=16: 5 columns identical, 8 sign-flipped, 0
            # genuinely different. Those signs are unphysical and harmless for
            # a canonical-orbital run, but `C @ U` mixes all columns, so a
            # flipped sign makes the ROTATED guess a genuinely different
            # orbital set -- which made this experiment non-reproducible
            # (9/12/11 macroiterations across identical repeats) until the
            # signs were pinned here. Deterministic rule: make the
            # largest-magnitude coefficient of every column positive.
            # np.round before argmax is essential, not cosmetic: in a molecule
            # with symmetry-equivalent atoms (water's two hydrogens) an MO has
            # EXACTLY tied largest coefficients, and the two differ only by
            # ~1e-15 of threading noise -- so a raw argmax picks between them
            # arbitrarily and the sign rule itself becomes non-reproducible
            # (measured: 3 of 13 columns fragile, margins 1.5e-15 to 6e-14).
            # Rounding collapses the tie so argmax's first-occurrence rule
            # resolves it deterministically by lowest index.
            for _j in range(self.C.shape[1]):
                _k = int(np.argmax(np.round(np.abs(self.C[:, _j]), 10)))
                if self.C[_k, _j] < 0.0:
                    self.C[:, _j] *= -1.0
            if os.environ.get("QED_DUMP_PREROT") is not None:
                np.savetxt(os.environ["QED_DUMP_PREROT"], self.C)
            U = ortho_group.rvs(wfn.nmo(), random_state=int(_rand_orb_seed))
            new_C = np.einsum("pq,qr->pr", self.C, U)
            self.C[:, :] = new_C[:, :]
            # update d_cmo
            self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)

        # QED_DEGEN_CANON: pin the arbitrary gauge inside degenerate clusters
        # before ANYTHING downstream sees the orbitals -- deliberately placed
        # above the orbital2.out checkpoint so the file records what the run
        # actually optimizes, and above the use_orbital_guess load below so a
        # seeded guess is canonicalized on the same footing as a fresh one.
        # PURELY ADDITIVE: with the knob off not one line here executes.
        if _QED_DEGEN_CANON:
            _fock_ao = getattr(self, "fock_alpha", None)
            if _fock_ao is None:
                print(
                    "  degen_canon: no CQED-RHF Fock available, skipping",
                    flush=True,
                )
            else:
                # Block edges from the wfn, not from self.n_in_a -- that
                # attribute is assigned in __init__ from self.ndocc, which is
                # itself only set further down in THIS method, so it cannot be
                # relied on here.  doccpi() is available and gives the same
                # 2 / 12 / 9 split for MgH+ CAS(8,12).
                _n_in = int(wfn.doccpi()[0]) - self.n_act_el // 2
                _bounds = [_n_in, _n_in + self.n_act_orb]
                _eps = np.einsum("pi,pq,qi->i", self.C, _fock_ao, self.C)
                _C_new = _canonicalize_degenerate(
                    self.C,
                    _gauge_tiebreaker(self.C.shape[0]),
                    _eps,
                    _bounds,
                    _QED_DEGEN_TOL,
                )
                _C_new = _pin_orbital_signs(_C_new)
                if _QED_SOLVER_STATS:
                    _cl = _degenerate_clusters(_eps, _bounds, _QED_DEGEN_TOL)
                    print(
                        f"  degen_canon: blocks {_n_in}/{self.n_act_orb}/"
                        f"{self.C.shape[1] - _n_in - self.n_act_orb}, "
                        f"clusters {_cl}, max|dC| "
                        f"{float(np.abs(_C_new - self.C).max()):.3e}",
                        flush=True,
                    )
                self.C = _C_new
                # d_cmo is built from C, so it must be rebuilt here -- the
                # random-rotation branch above does the same after it rotates.
                self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)

        if self.config.initial_orbital_output_file:
            np.savetxt(self.config.initial_orbital_output_file, self.C)
            self._generated_files["initial_orbital_output_file"] = self.config.initial_orbital_output_file
        elif self.config.legacy:
            np.savetxt("orbital2.out", self.C)
            self._generated_files["orbital2.out"] = os.path.abspath("orbital2.out")
        # print("Unitary matrix")
        # print(U)
        # update wfn_dict with orbitals from CQED-RHF
        wfn_dict["matrix"]["Ca"] = self.C
        wfn_dict["matrix"]["Cb"] = self.C
        # mints = psi4.core.MintsHelper(wfn.basisset())
        # overlap_matrix = mints.ao_overlap()
        # overlap_matrix = np.asarray(overlap_matrix)
        # print(np.shape(overlap_matrix))
        # temp = np.einsum("pq, qr, rs->ps", self.C.T, overlap_matrix, self.C)
        # print("thrbf", np.allclose(temp,np.eye(wfn.nmo()), rtol=1e-14,atol=1e-14))
        # print(self.fock_alpha)

        if self.use_orbital_guess == True:
            if self.photon_number_basis == False:
                wfn_dict["matrix"]["Fa"] = self.fock_alpha
                wfn_dict["matrix"]["Fb"] = self.fock_alpha
            print("load file")
            guess = np.loadtxt(self.config.orbital_guess_file)
            expected_shape = np.asarray(wfn.Ca()).shape
            if guess.shape != expected_shape or not np.isfinite(guess).all():
                raise ValueError(f"Orbital guess must be finite with shape {expected_shape}; got {guess.shape}")
            wfn_dict["matrix"]["Ca"] = guess
            wfn_dict["matrix"]["Cb"] = guess.copy()
            # update wfn object
            wfn = psi4.core.Wavefunction.from_file(wfn_dict)
            wfn.Ca().copy(ortho_orbs(wfn, wfn))
            wfn.Cb().copy(wfn.Ca())
            self.C = np.asarray(wfn.Ca())
            # update d_cmo
            self.d_cmo = np.dot(self.C.T, self.d_ao).dot(self.C)
            # mints = psi4.core.MintsHelper(wfn.basisset())
            # overlap_matrix = mints.ao_overlap()
            # overlap_matrix = np.asarray(overlap_matrix)
            # print(np.shape(overlap_matrix))
            # temp = np.einsum("pq, qr, rs->ps", self.C.T, overlap_matrix, self.C)
            # print("thrbf", np.allclose(temp,np.eye(wfn.nmo()), rtol=1e-14,atol=1e-14))
            # print(temp)

        # update wfn object
        wfn = psi4.core.Wavefunction.from_file(wfn_dict)

        # Grab data from wavfunction class
        self.Ca = wfn.Ca()
        self.ndocc = wfn.doccpi()[0]
        self.nmo = wfn.nmo()
        self.nso = 2 * self.nmo
        self.nvirt = self.nmo - self.ndocc

        # this is a bad hack for now
        if self.ci_level == "fci" or self.ci_level == "FCI":
            # initialize arrays that will store the number of configurations by excitation rank for CASCI and CASSCF
            self.casscf_config_count_by_rank = np.zeros(2 * self.ndocc + 1)
            self.casci_config_count_by_rank = np.zeros(2 * self.ndocc + 1)

            # initialize arrays that will store the sum of squared weights by excitation rank for CASCI and CASSCF
            self.casscf_sum_squared_weight_by_rank = np.zeros(2 * self.ndocc + 1)
            self.casci_sum_squared_weight_by_rank = np.zeros(2 * self.ndocc + 1)

        self.docc_list = [i for i in range(self.ndocc)]
        self.wfn = wfn

        # print("mo coefficient")
        # for k in range(self.nmo):
        #    for l in range(self.nmo):
        #        print("{:20.16f}".format(self.C[k,l]), k, l, flush = True)

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
        _H_spin_hf = np.einsum("uj,vi,uv", self.C_hf, self.C_hf, self.H_1e_ao)
        self.H_spatial = _H_spin
        self.H_spatial2 = np.ascontiguousarray(_H_spin)
        self.H_hf = np.ascontiguousarray(_H_spin_hf)
        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            _H_spin = np.repeat(_H_spin, 2, axis=0)
            _H_spin = np.repeat(_H_spin, 2, axis=1)
            # spin part of 1-e integrals
            spin_ind = np.arange(_H_spin.shape[0], dtype=int) % 2
            # product of spatial and spin parts
            self.Hspin = _H_spin * (spin_ind.reshape(-1, 1) == spin_ind)
        #self.H_hf = copy.deepcopy(self.H_spatial2)

    # =====================================================================
    # QED_LIBTRANS_OO -- psi4's CONV strategy: keep NO nmo^4 MO tensor, and
    # re-transform the two blocks J and K actually need at the CURRENT
    # orbitals, once per macroiteration.
    #
    # WHY THIS IS EQUIVALENT.  build_JK documents the conventions:
    #       J[k,l,r,s] = (kl|rs)     bra pair both occupied, ket pair general
    #       K[k,l,r,s] = (kr|ls)     bra pair (k,r), ket pair (l,s)
    # so J is a (oo|gg) block and K is a (og|og) block.  Neither can be
    # obtained by ROTATING the other -- which is exactly why the stored
    # reference tensor exists today, and exactly why dropping it forces a
    # per-macro re-transform.  Validated against mints.mo_eri with sliced
    # coefficient matrices to 1.4e-15 (J) and 5.2e-16 (K); the check is kept
    # as validation/oo_transform_check.py.
    #
    # THE ONE NON-OBVIOUS PIECE: MOSpace.occ() means psi4's *docc*, not our
    # n_occupied = inactive + active.  A CUSTOM MOSpace(label, aOrbs, aIndex)
    # silently yields a ZERO-row pair space (the built-in spaces carry empty
    # aOrbs/aIndex -- they are sentinels libtrans resolves internally), so the
    # lever is wfn.force_occpi(), which redefines occ() to our count.  It is
    # force_occpi, not force_doccpi; the latter does not exist.
    #
    # BASIS BOOKKEEPING, the thing most likely to go wrong here.  U_total is
    # measured from the REFERENCE orbitals that self.twoeint was built in, and
    # self.Ca / self.d_cmo are MUTATED during optimisation -- so neither may be
    # read at call time.  _oo_setup snapshots both, and every transform uses
    #       C(U) = Ca_ref @ U          d(U) = U^T d_ref U
    # The DSE is added analytically rather than folded into a tensor:
    #       J_dse[k,l,r,s] = d[k,l] d[r,s]     K_dse[k,l,r,s] = d[k,r] d[l,s]
    # which is the same rank-1 pair update build2DSO applies, expressed in the
    # unpacked blocks (and cheaper -- it never touches an nmo^4 array).
    # =====================================================================
    def _macro_JK(self, U_total, J, K):
        """Per-macro J/K build: the one dispatch point for QED_LIBTRANS_OO.

        0 (default) -- rotate the stored packed nmo^4 tensor, as before.
        1           -- re-transform the (oo|gg) and (og|og) blocks at the
                       current orbitals; the tensor is not read at all.
        2           -- CHECK MODE: do BOTH and report max|difference|.  Slower
                       than either, and it needs self.twoeint, so it is a
                       validation mode rather than a production one.
        """
        if not _QED_LIBTRANS_OO:
            # Under QED_JK_H2E_DISK self.twoeint is the (1,1) placeholder and
            # the kernel streams the real rows from the spilled file; the call
            # is otherwise unchanged, which is the point of routing the path
            # through qed_jk_set_h2e_file rather than through this signature.
            c_full_transformation_macroiteration(
                U_total, self.twoeint, J, K,
                self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied,
            )
            return
        if _QED_LIBTRANS_OO >= 2:
            J_ref = np.empty_like(J)
            K_ref = np.ascontiguousarray(np.empty_like(K))
            c_full_transformation_macroiteration(
                U_total, self.twoeint, J_ref, K_ref,
                self.index_map_pq, self.index_map_kl, self.nmo, self.n_occupied,
            )
            self._oo_transform_JK(U_total, J, K)
            print(
                "QED_LIBTRANS_OO check: max|dJ| %.3e  max|dK| %.3e  "
                "(scale %.3e)"
                % (np.max(np.abs(J - J_ref)), np.max(np.abs(K - K_ref)),
                   max(np.max(np.abs(J_ref)), np.max(np.abs(K_ref)))),
                flush=True,
            )
            return
        self._oo_transform_JK(U_total, J, K)

    def _oo_setup(self):
        """One-time libtrans/shim setup for the QED_LIBTRANS_OO route.

        Everything orbital-INDEPENDENT happens once: the AO integrals, the
        SO presort, the DPD ids and buffer labels, and the packed-pair
        unpacking maps.  Only set_orbitals + transform_tei recur per macro,
        which is what makes the per-macro cost a transform rather than a
        transform plus a presort.
        """
        if getattr(self, "_oo_ready", False):
            return True
        shim_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "psi4_dpd_shim.so"
        )
        if not os.path.exists(shim_path):
            return _libtrans_fallback(f"{shim_path} not built (QED_LIBTRANS_OO)")
        ctypes.CDLL(psi4.core.__file__, mode=ctypes.RTLD_GLOBAL)
        shim = ctypes.CDLL(shim_path)
        if not hasattr(shim, "qed_dpd_read_buf4_rect"):
            return _libtrans_fallback(
                "psi4_dpd_shim.so predates qed_dpd_read_buf4_rect; rebuild it"
            )
        shim.qed_dpd_read_buf4_rect.restype = ctypes.c_int
        shim.qed_dpd_read_buf4_rect.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
        ]

        no, nmo = self.n_occupied, self.nmo
        # needed before the IntegralTransform exists, to size its memory
        self._oo_nJr_pre, self._oo_nJc_pre = no * (no + 1) // 2, nmo * (nmo + 1) // 2
        self._oo_nKr_pre = no * nmo
        wfn = self.wfn
        wfn.force_occpi(psi4.core.Dimension([no]), psi4.core.Dimension([0]))

        IT = psi4.core.IntegralTransform
        alln = psi4.core.MOSpace.all()
        occ = psi4.core.MOSpace.occ()
        mints = psi4.core.MintsHelper(wfn.basisset())
        _t0 = time.time()
        mints.integrals()  # AO -> disk, or presort dies with PSIO_ERROR 21
        it = IT(
            wfn, [occ, alln],
            IT.TransformationType.Restricted, IT.OutputType.DPDOnly,
            IT.MOOrdering.PitzerOrder, getattr(IT.FrozenOrbitals, "None"),
            False,
        )
        it.set_keep_dpd_so_ints(True)
        # SIZE THE TRANSFORM AGAINST THE BUFFERS IT MUST PRODUCE, not against
        # psi4's default.  The drivers here never call psi4.set_memory(), so
        # psi4.core.get_memory() is 500 MiB -- _build_twoeint_libtrans already
        # overrides it for exactly this reason ("sizing an 8.4 GB transform
        # against 500 MiB just forces needless multipassing", :6726).  At
        # nmo 302 the K buffer alone is 1.11 GB, and a 500 MiB budget took the
        # aug-cc-pVDZ leg down right after setup with no Python traceback.
        # Ask for both buffers plus the same again as working room; that is
        # still an order of magnitude under the tensor this route replaces.
        _need = 8 * (self._oo_nJr_pre * self._oo_nJc_pre
                     + self._oo_nKr_pre * self._oo_nKr_pre) * 2
        it.set_memory(max(int(psi4.core.get_memory()), _need))
        it.initialize()
        _t1 = time.time()
        it.presort_so_tei()
        it.set_tei_already_presorted(True)  # orbital-independent: sort ONCE
        _t2 = time.time()

        def _packmap(n):
            # DPD stores [p>=q]+ with the LARGER index in the outer loop.
            t = np.zeros((n, n), dtype=np.int64)
            k = 0
            for p in range(n):
                for q in range(p + 1):
                    t[p, q] = t[q, p] = k
                    k += 1
            return t

        self._oo_shim = shim
        self._oo_it = it
        self._oo_occ, self._oo_all = occ, alln
        self._oo_pqJ = it.DPD_ID(occ, occ, IT.SpinType.Alpha, True)
        self._oo_rsJ = it.DPD_ID(alln, alln, IT.SpinType.Alpha, True)
        self._oo_pqK = it.DPD_ID(occ, alln, IT.SpinType.Alpha, False)
        self._oo_nJr, self._oo_nJc = no * (no + 1) // 2, nmo * (nmo + 1) // 2
        self._oo_nKr = no * nmo
        self._oo_omap, self._oo_amap = _packmap(no), _packmap(nmo)
        self._oo_Jbuf = np.empty(self._oo_nJr * self._oo_nJc)
        self._oo_Kbuf = np.empty(self._oo_nKr * self._oo_nKr)
        # The reference basis was snapshotted in buildArraysInOrbitalBasis, at
        # the point twoeint became final.  Falling back to self.Ca here would
        # be wrong whenever setup happens after the first rotation.
        if getattr(self, "_oo_Ca_ref", None) is None:
            raise RuntimeError(
                "QED_LIBTRANS_OO: no reference-basis snapshot -- _oo_setup ran "
                "before buildArraysInOrbitalBasis"
            )
        self._oo_ready = True
        # This setup is the counterpart of the `ERI Build` phase the route
        # deletes, and it lands OUTSIDE both that timer and `optimization took`
        # (the first call comes from build_JK_oo, before the macro loop starts).
        # Print it or the A/B accounting silently loses ~30 s.
        print(
            "QED_LIBTRANS_OO: ready (J %d x %d, K %d x %d, %.2f GB of buffer "
            "against %.2f GB for the packed nmo^4 tensor); "
            "setup: ao_integrals %.2f s, presort %.2f s, libtrans budget %.2f GB"
            % (self._oo_nJr, self._oo_nJc, self._oo_nKr, self._oo_nKr,
               (self._oo_Jbuf.nbytes + self._oo_Kbuf.nbytes) / 1024 ** 3,
               (nmo * (nmo + 1) // 2) ** 2 * 8 / 1024 ** 3,
               _t1 - _t0, _t2 - _t1, _need / 1024 ** 3),
            flush=True,
        )
        return True

    def _oo_transform_JK(self, U_total, J, K):
        """Fill J and K in the basis Ca_ref @ U_total.  Drop-in replacement for
        c_full_transformation_macroiteration(U_total, self.twoeint, J, K, ...).
        """
        if not self._oo_setup():
            raise RuntimeError("QED_LIBTRANS_OO requested but unavailable")
        no, nmo = self.n_occupied, self.nmo
        U = np.asarray(U_total)
        it, shim = self._oo_it, self._oo_shim
        # TIMED SEPARATELY ON PURPOSE.  The J side is (oo|gg) -- 780 x 45753,
        # PACKED on both axes -- while the K side is (og|og), 11778 x 11778,
        # unpacked on both, i.e. 3.9x the elements.  Reporting one combined
        # number hides which of the two the cost actually lives in, and that
        # decides whether a reformulation should attack K's layout or the
        # shared first half-transformation.
        _t0 = time.time()
        it.set_orbitals(psi4.core.Matrix.from_array(self._oo_Ca_ref @ U))
        _t1 = time.time()
        it.transform_tei(self._oo_occ, self._oo_occ, self._oo_all, self._oo_all)
        _t2 = time.time()
        it.transform_tei(self._oo_occ, self._oo_all, self._oo_occ, self._oo_all)
        _t3 = time.time()
        shim.qed_dpd_set_default(it.get_dpd_id())

        rc = shim.qed_dpd_read_buf4_rect(
            61, self._oo_pqJ, self._oo_rsJ, b"MO Ints (OO|AA)",
            self._oo_Jbuf, self._oo_nJr, self._oo_nJc, None, None)
        if rc != 0:
            raise RuntimeError(f"QED_LIBTRANS_OO: J buffer read failed rc={rc}")
        rc = shim.qed_dpd_read_buf4_rect(
            61, self._oo_pqK, self._oo_pqK, b"MO Ints (OA|OA)",
            self._oo_Kbuf, self._oo_nKr, self._oo_nKr, None, None)
        if rc != 0:
            raise RuntimeError(f"QED_LIBTRANS_OO: K buffer read failed rc={rc}")

        _t4 = time.time()
        print("QED_LIBTRANS_OO: set_orbitals %.2f s | J transform %.2f s | "
              "K transform %.2f s | readback %.2f s"
              % (_t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3), flush=True)
        Jb = self._oo_Jbuf.reshape(self._oo_nJr, self._oo_nJc)
        Kb = self._oo_Kbuf.reshape(no, nmo, no, nmo)
        J[:] = Jb[self._oo_omap][:, :, self._oo_amap]
        K[:] = Kb.transpose(0, 2, 1, 3)
        if not self.ignore_dse_terms:
            d = U.T @ self._oo_d_ref @ U
            J += np.einsum("kl,rs->klrs", d[:no, :no], d, optimize=True)
            K += np.einsum("kr,ls->klrs", d[:no, :], d[:no, :], optimize=True)

    def _build_twoeint_libtrans(self, p4_wfn, mints):
        """Build the packed MO two-electron integrals through psi4's libtrans.

        Returns the (npair, npair) packed array in helper_PFCI's OWN pair order,
        or None if the route is unavailable -- in which case the caller falls
        back to mo_eri.  See the QED_LIBTRANS note at the top of this file for
        the measurements and for why this beats every alternative.

        Four things here are load-bearing and each of them fails SILENTLY (zeros
        or a segfault, not an exception) if got wrong; see psi4_dpd_shim.cpp.
        """
        try:
            shim_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "psi4_dpd_shim.so"
            )
            if not os.path.exists(shim_path):
                return _libtrans_fallback(f"{shim_path} not built")

            # (1) psi4 is one monolithic core.cpython-*.so that Python dlopens
            # RTLD_LOCAL, so its symbols are invisible to psi4_dpd_shim.so.
            # Re-dlopening it with RTLD_GLOBAL promotes them, and it works even
            # though psi4 was already imported normally at :18 -- which is what
            # lets this be a library-side change with no driver script touched.
            ctypes.CDLL(psi4.core.__file__, mode=ctypes.RTLD_GLOBAL)
            shim = ctypes.CDLL(shim_path)
            shim.qed_dpd_read_buf4.restype = ctypes.c_int
            shim.qed_dpd_read_buf4.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.c_long,
                np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ]

            npair = self.nmo * (self.nmo + 1) // 2

            t0 = time.time()
            mints.integrals()  # AO -> disk.  MUST precede presort or it dies
            t1 = time.time()  # with PSIO_ERROR 21 (file 33 not opened).

            IT = psi4.core.IntegralTransform
            alln = psi4.core.MOSpace.all()
            it = IT(
                p4_wfn,
                [alln],
                IT.TransformationType.Restricted,
                IT.OutputType.DPDOnly,
                # PitzerOrder, not QTOrder: we need the columns to stay in
                # self.Ca's order, not be re-sorted into docc/socc/virt blocks.
                IT.MOOrdering.PitzerOrder,
                getattr(IT.FrozenOrbitals, "None"),  # "None" is a keyword
                False,  # initialize=False, else the ctor initialises too early
            )
            # libtrans would otherwise transform with the WAVEFUNCTION's Ca,
            # but the integrals this class needs are in the CQED-RHF basis
            # (self.Ca), which is not what SCF left behind.  set_orbitals is
            # used rather than mutating p4_wfn.Ca() in place, because Ca()
            # hands back the wavefunction's own SharedMatrix and overwriting it
            # would silently change the basis for every later user of p4_wfn.
            it.set_orbitals(psi4.core.Matrix.from_array(np.asarray(self.Ca)))
            it.set_keep_dpd_so_ints(True)
            # libtrans otherwise inherits psi4's memory setting, which is 500
            # MiB unless the driver called psi4.set_memory() -- and none of the
            # drivers here do.  Sizing an 8.4 GB transform against 500 MiB just
            # forces needless multipassing.  We are about to hold the packed
            # npair^2 array in numpy anyway, so granting libtrans the same
            # budget is proportionate and self-scaling; take psi4's setting if
            # the driver did raise it above that.
            # QED_LIBTRANS_MEM_MB (item 356C).  The default below hands libtrans
            # 8*npair^2 bytes -- 15.6 GiB at nmo 302 -- and the reason written
            # here was "we are about to hold the packed npair^2 array in numpy
            # anyway, so granting libtrans the same budget is proportionate".
            # WITH QED_JK_H2E_DISK + STREAMING THAT PREMISE IS GONE: the array
            # is never held, and measurement showed this grant is then the
            # RUN'S PEAK -- a 15.8 GiB plateau spanning transform_tei, larger
            # than everything the optimization itself needs (8-14 GiB).
            # Setting this caps the grant instead; the cost is transform
            # multipassing, so it is a real trade and is left explicit.
            if _QED_LIBTRANS_MEM_MB > 0:
                it.set_memory(int(_QED_LIBTRANS_MEM_MB) * 1048576)
            else:
                it.set_memory(max(int(psi4.core.get_memory()),
                                  8 * npair * npair))
            it.initialize()
            it.presort_so_tei()
            it.set_tei_already_presorted(True)  # sort once, not per transform
            t2 = time.time()
            it.transform_tei(alln, alln, alln, alln)
            t3 = time.time()

            # (2) IntegralTransform allocates its OWN DPD instance; global_dpd_
            # follows whichever is current, and buf4_init against the wrong one
            # segfaults.
            shim.qed_dpd_set_default(it.get_dpd_id())
            # (3) the buffer is PACKED as [A>=A]+, not unpacked [A,A].  Asking
            # for the unpacked pair type yields rc=0 and an all-zero array.
            pq = it.DPD_ID(alln, alln, IT.SpinType.Alpha, True)

            # (4) DPD's [p>=q]+ runs the LARGER index outer (for p: for q<=p);
            # index_map_pq (:3221-3224) runs the SMALLER one outer.  Build the
            # gather that takes DPD order to ours and let the shim apply it on
            # both axes, which avoids a second 2.12 GB pass in numpy.
            _pi, _qi = _pair_lists(self.nmo)
            dpd_pos = np.empty((self.nmo, self.nmo), dtype=np.int32)
            _k = 0
            for _p in range(self.nmo):
                for _q in range(_p + 1):
                    dpd_pos[_p, _q] = _k
                    _k += 1
            perm = np.ascontiguousarray(dpd_pos[_pi, _qi], dtype=np.int32)

            PSIF_LIBTRANS_DPD = 61  # NOT 115
            if _jk_stream_armed(shim):
                # STREAMED READBACK (item 354E).  qed_dpd_read_buf4 calls
                # buf4_mat_irrep_rd, which materialises psi4's ENTIRE irrep
                # matrix and then copies it into `packed` -- two full tensors
                # alive at once, measured at nmo 302 as a 30.86 GiB transient
                # against a 15.6 GiB tensor, and that transient is what sets
                # the run's memory requirement (item 354B).  Here the buffer is
                # read a row-block at a time and written straight to the spill
                # file, so neither copy ever exists.
                #
                # The DSE fold is deliberately NOT done here: the caller's
                # existing chunked rank-1 loop runs afterwards and works
                # unchanged on the writable memmap this returns, 256 rows at a
                # time.  Reusing that loop keeps one implementation of the fold.
                packed_mm = self._stream_twoeint_to_file(
                    shim, PSIF_LIBTRANS_DPD, pq, npair, perm)
                t4 = time.time()
                print(
                    f" QED_LIBTRANS: integrals {t1-t0:.2f} s, presort "
                    f"{t2-t1:.2f} s, transform {t3-t2:.2f} s, readback "
                    f"{t4-t3:.2f} s (STREAMED), total {t4-t0:.2f} s",
                    flush=True,
                )
                return packed_mm
            packed = np.empty(npair * npair, dtype=np.float64)
            rc = shim.qed_dpd_read_buf4(
                PSIF_LIBTRANS_DPD, pq, pq, b"MO Ints (AA|AA)", packed,
                npair, perm,
            )
            t4 = time.time()

            # CLEAN UP THE SCRATCH UNITS, OR THE NEXT psi4 JOB IN THIS PROCESS
            # DIES.  mints.integrals() does not only write the AO two-electron
            # integrals: it writes PSIF_SO_TEI (33), the ONE-electron integrals
            # to PSIF_OEI (35), and presort_so_tei writes PSIF_SO_PRESORT (41).
            # psi4's own SCF uses those same units, and the entry layout is
            # sized for THIS molecule's basis -- so a later, larger job writing
            # into the leftovers fails with
            #     PSIO_ERROR: 18 (Incorrect block end address)
            #     Attempt to write into next entry: 35, SO-basis Potential
            #     Energy Ints
            # in Matrix::save during form_H.  That is how test_h2o_qed_rhf and
            # test_mgf_qed_rhf came to fail with this knob on even though
            # neither constructs a PFHamiltonianGenerator at all -- they run
            # cqed_rhf() after some earlier test had used this path.  35 is the
            # one that actually bites; 33/41/61 are cleaned for the same reason.
            # Targeted deletes rather than psi4.core.clean(), which would also
            # remove files the live wavefunction still owns.  Opening a unit
            # that does not exist and closing it with keep=0 is a harmless
            # no-op, so this needs no existence check.
            del it
            _io = psi4.core.IO.shared_object()
            for _unit in (33, 34, 35, 41, 61):
                try:
                    if not _io.open_check(_unit):
                        _io.open(_unit, 1)  # PSIO_OPEN_OLD
                    _io.close(_unit, 0)  # keep=0 -> delete
                except Exception:  # noqa: BLE001
                    pass

            if rc != 0:
                return _libtrans_fallback(f"readback failed (rc={rc})")
            print(
                f" QED_LIBTRANS: integrals {t1-t0:.2f} s, presort {t2-t1:.2f} "
                f"s, transform {t3-t2:.2f} s, readback {t4-t3:.2f} s, total "
                f"{t4-t0:.2f} s",
                flush=True,
            )
            return packed.reshape(npair, npair)
        except Exception as exc:  # noqa: BLE001 -- never fail the whole run
            return _libtrans_fallback(f"{exc!r}")

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
        # The DSE term used to be added out of place:
        #     self.twoeint = self.twoeint1 + np.einsum("ij,kl->ijkl", d, d)
        # which holds THREE full nmo^4 arrays at once -- twoeint1, the einsum
        # temporary, and the sum.  At c10h8 dims that is 3 x 8.4 GB, and since
        # build2DSO is called from inside buildArraysInOrbitalBasis (:5246), the
        # AO-basis ERI local there is live too: 4 x 8.4 GB = 33.6 GB, against a
        # measured maxrss of 33.1 GB.  That is the whole peak of the run.
        #
        # twoeint1 is dropped immediately below and nothing else references it,
        # so its buffer can simply be taken over and updated in place.  Flattened
        # to (nmo^2, nmo^2) the einsum is exactly outer(d_flat, d_flat) --
        # element (i,j,k,l) sits at row i*nmo+j, column k*nmo+l with value
        # d[i,j]*d[k,l] -- i.e. a rank-1 update, applied here in row chunks so
        # the temporary stays at chunk_size x nmo^2 instead of nmo^4.
        # BIT-IDENTICAL: same single multiply per element, same addition, so
        # neither the products nor the rounding change.
        # PACKED STORAGE.  self.twoeint is (npair, npair), not (nmo^2, nmo^2):
        # both the bra and the ket pair are folded onto p >= q using the 8-fold
        # permutational symmetry, which the DSE term preserves exactly (d_cmo is
        # symmetric, so outer(d,d) carries all 8 folds -- measured 1.7e-16).
        #     8.40 GB -> 2.12 GB at c10h8 dims (nmo=180, npair=16290)
        # The pair ordering is index_map_pq's (:3221-3224) -- smaller index in
        # the OUTER loop -- so packed row `pq_up` IS the pair index_map_pq names
        # and orbital.c needs no bra-side lookup at all.
        self.npair = self.nmo * (self.nmo + 1) // 2
        self.pair_index = _pair_index_table(self.nmo)
        # _pi/_qi and _chunk are needed by the DSE rank-1 update below on BOTH
        # paths, so they cannot live inside the fold branch.
        _pi, _qi = _pair_lists(self.nmo)
        _chunk = 256
        if getattr(self, "_twoeint_packed", None) is not None:
            # QED_LIBTRANS already produced the packed array in exactly this
            # layout and pair order, so there is nothing to fold: the dense
            # nmo^4 twoeint1 was never built at all.
            self.twoeint = self._twoeint_packed
            self._twoeint_packed = None
            self.twoeint1 = None
        else:
            _lin = (_pi * self.nmo + _qi).astype(np.intp)
            _g2 = self.twoeint1.reshape(self.nmo * self.nmo, self.nmo * self.nmo)
            _packed = np.empty((self.npair, self.npair))
            for _i0 in range(0, self.npair, _chunk):
                _i1 = min(_i0 + _chunk, self.npair)
                _packed[_i0:_i1] = _g2[_lin[_i0:_i1][:, None], _lin[None, :]]
            self.twoeint1 = None
            del self.twoeint1
            self.twoeint = _packed
        # self.contracted_twoeint = -0.5 * np.einsum("illj->ij", self.twoeint)
        if not self.ignore_dse_terms:
            # The DSE term on the packed form is a rank-1 update by the PAIR
            # vector d_pair[PQ] = d_cmo[p, q] -- 2.12 GB of work instead of the
            # 8.40 GB the full-tensor version touched, and no nmo^4 temporary.
            _d_pair = np.ascontiguousarray(self.d_cmo[_pi, _qi])
            for _i0 in range(0, self.npair, _chunk):
                _i1 = min(_i0 + _chunk, self.npair)
                self.twoeint[_i0:_i1] += _d_pair[_i0:_i1, None] * _d_pair
        # self.twoeint is FINAL here (packed, DSE folded).  This is the only
        # point where spilling it is safe -- see _spill_twoeint_to_disc.
        self._spill_twoeint_to_disc()

    def _stream_twoeint_to_file(self, shim, filenum, pq, npair, perm):
        """Read the libtrans buf4 block-wise into the spill file (item 354E).

        Returns a WRITABLE memmap of the file, so the caller's existing chunked
        DSE rank-1 loop runs on it unchanged.  _spill_twoeint_to_disc then sees
        the tensor is already on disc and only has to arm the C reader.

        WHY THE ROWS ARE SCATTERED.  `perm[i]` is the DPD row holding OUR row i;
        DPD can only deliver a CONTIGUOUS row range, so a block of DPD rows is a
        scattered set of ours.  Reading in DPD order and placing by the inverse
        permutation keeps the READ sequential (one pass over the buffer) at the
        cost of scattered writes, which the page cache absorbs.  The other way
        round would be npair random reads.
        """
        shim.qed_dpd_stream_open.restype = ctypes.c_int
        shim.qed_dpd_stream_open.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_long, ctypes.c_long, ctypes.c_long]
        shim.qed_dpd_stream_read.restype = ctypes.c_int
        shim.qed_dpd_stream_read.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_long, ctypes.c_long,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
        shim.qed_dpd_stream_close.restype = ctypes.c_int

        self._drop_h2e_file()
        _dir = None if _QED_JK_H2E_DISK == "1" else _QED_JK_H2E_DISK
        _fd, _path = tempfile.mkstemp(prefix="qedci_h2e_", suffix=".bin", dir=_dir)
        _nbytes = npair * npair * 8
        os.ftruncate(_fd, _nbytes)

        _rows = max(1, (_QED_JK_STREAM_MB * 1048576) // (npair * 8))
        _rows = min(_rows, npair)
        _buf = np.empty((_rows, npair), dtype=np.float64)
        _inv = np.empty(npair, dtype=np.intp)
        _inv[np.asarray(perm, dtype=np.intp)] = np.arange(npair)

        _t0 = time.time()
        rc = shim.qed_dpd_stream_open(filenum, pq, pq, b"MO Ints (AA|AA)",
                                      npair, npair, _rows)
        if rc != 0:
            os.close(_fd)
            os.unlink(_path)
            raise RuntimeError("qed_dpd_stream_open failed rc=%d" % rc)
        try:
            for _r0 in range(0, npair, _rows):
                _n = min(_rows, npair - _r0)
                rc = shim.qed_dpd_stream_read(_buf, _r0, _n, perm)
                if rc != 0:
                    raise RuntimeError("qed_dpd_stream_read failed rc=%d" % rc)
                _dest = _inv[_r0:_r0 + _n]
                for _k in range(_n):
                    os.pwrite(_fd, _buf[_k].tobytes(),
                              int(_dest[_k]) * npair * 8)
        finally:
            shim.qed_dpd_stream_close()
            os.close(_fd)

        self._h2e_path = _path
        _H2E_SPILL_FILES.add(_path)
        print("QED_JK_H2E_DISK: streamed %.2f GiB of packed twoeint into %s in "
              "%.2f s (%d rows/block); the tensor was never resident"
              % (_nbytes / 1073741824.0, _path, time.time() - _t0, _rows),
              flush=True)
        return np.memmap(_path, dtype=np.float64, mode="r+",
                         shape=(npair, npair))

    def _spill_twoeint_to_disc(self):
        """QED_JK_H2E_DISK (SLW99 step 1b): write the packed tensor out, drop it.

        Called once, at the point self.twoeint becomes final.  Afterwards
        self.twoeint is a (1,1) placeholder that exists only to satisfy the
        ndim=2 ctypes argtype of full_transformation_macroiteration, which no
        longer dereferences it: orbital.c's S2 reads its row-blocks from the
        file instead (`qed_jk_set_h2e_file`, proven bitwise in
        validation/h2e_file_invariance.py).

        Every precondition is checked LOUDLY rather than silently degrading --
        a silent fallback here would look like a merely-slow run while the
        memory saving it exists for never happened.
        """
        if not _QED_JK_H2E_DISK or _QED_JK_H2E_DISK == "0":
            return
        if not _HAS_JK_H2E_FILE:
            raise RuntimeError(
                "QED_JK_H2E_DISK: this cfunctions.so has no qed_jk_set_h2e_file "
                "-- rebuild it from the current orbital.c (see CLAUDE.md)")
        if _QED_LIBTRANS_OO >= 2:
            raise RuntimeError(
                "QED_JK_H2E_DISK is incompatible with QED_LIBTRANS_OO=2: that "
                "check mode rotates the RESIDENT tensor on purpose")
        if _CPP_CASSCF_VALIDATION_DIR:
            raise RuntimeError(
                "QED_JK_H2E_DISK is incompatible with the cpp_casscf validation "
                "dumps: the dump hooks record self.twoeint, which would be the "
                "placeholder rather than the tensor")
        if not _QED_SKIP_GKL:
            raise RuntimeError(
                "QED_JK_H2E_DISK needs QED_SKIP_GKL=1: self.gkl is the one "
                "consumer of the packed tensor that is not the J/K build "
                "(see the note at the top of this file)")
        _already = getattr(self, "_h2e_path", None)
        if _already and isinstance(self.twoeint, np.memmap):
            # The streamed readback already wrote this file and handed back a
            # WRITABLE memmap, which the caller's DSE rank-1 loop has just
            # updated in place.  There is nothing to write: flush the dirty
            # pages, arm the C reader, and downgrade the mapping to read-only
            # so nothing downstream can modify the tensor by accident.
            _shape = self.twoeint.shape
            self.twoeint.flush()
            self.twoeint = None
            cfunctions.qed_jk_set_h2e_file(_already.encode())
            self.twoeint = np.memmap(_already, dtype=np.float64, mode="r",
                                     shape=_shape)
            print("QED_JK_H2E_DISK: tensor already on disc (streamed readback);"
                  " armed the C reader on %s" % _already, flush=True)
            return
        self._drop_h2e_file()
        _dir = None if _QED_JK_H2E_DISK == "1" else _QED_JK_H2E_DISK
        _fd, _path = tempfile.mkstemp(prefix="qedci_h2e_", suffix=".bin", dir=_dir)
        os.close(_fd)
        _t0 = time.time()
        _arr = self.twoeint
        if _arr.flags["C_CONTIGUOUS"]:
            _arr.tofile(_path)
        else:                      # never seen; write in chunks rather than
            with open(_path, "wb") as _fh:      # materialise a second copy
                for _i0 in range(0, _arr.shape[0], 256):
                    _fh.write(np.ascontiguousarray(_arr[_i0:_i0 + 256]).tobytes())
        _gib = _arr.nbytes / 1073741824.0
        _shape = _arr.shape
        # Release the anonymous allocation and re-expose the SAME numbers as a
        # read-only memmap.  Two consumers need two different things:
        #   * build_JK (:9072) reads whole ROWS of the packed tensor in python
        #     to form the initial reference-basis J and K.  It is untouched by
        #     the C route, so it needs a real array-like -- the memmap serves
        #     its row gathers out of page cache, which is evictable, instead of
        #     out of a resident anonymous 15.60 GiB.
        #   * the per-macro C kernel takes its rows through the FILE source
        #     with a bounded pread buffer, and never dereferences the pointer
        #     it is handed.  Passing the memmap is therefore free -- and it
        #     also means a mis-armed file source degrades to "mapped and
        #     correct", never to a wrong answer.
        self.twoeint = None
        self.twoeint = np.memmap(_path, dtype=np.float64, mode="r", shape=_shape)
        cfunctions.qed_jk_set_h2e_file(_path.encode())
        self._h2e_path = _path
        _H2E_SPILL_FILES.add(_path)
        print("QED_JK_H2E_DISK: spilled %.2f GiB of packed twoeint to %s in "
              "%.2f s; it is now a read-only memmap and the C kernel streams "
              "it" % (_gib, _path, time.time() - _t0), flush=True)

    def _drop_h2e_file(self):
        """Remove the spilled tensor and unpoint the kernel.  Idempotent."""
        _path = getattr(self, "_h2e_path", None)
        if not _path:
            return
        self._h2e_path = None
        if _HAS_JK_H2E_FILE:
            cfunctions.qed_jk_set_h2e_file(None)
        # Drop the mapping before unlinking so the space is actually returned
        # rather than held by an open mapping until interpreter exit.
        if isinstance(getattr(self, "twoeint", None), np.memmap):
            self.twoeint = None
        _H2E_SPILL_FILES.discard(_path)
        try:
            os.unlink(_path)
        except OSError:
            pass

    def build_gkl_from_df_optimized(self, B_array, d_cmo, H_spatial):
        """
        Build gkl intermediate efficiently using DF tensors
        
        Memory: O(nmo^3) instead of O(nmo^4)
        """
        nmo = B_array.shape[1]
        naux = B_array.shape[0]
        
        print(f"Building gkl from DF tensors: {naux} auxiliary, {nmo} MO")
        
        # Start with one-electron part
        gkl = H_spatial.copy()
        
        # ============================================
        # Term 1: -sum_{j<k} (kj|jl)
        # ============================================
        # Compute all (kj|jl) at once
        # Coulomb: sum_Q B[Q,k,j] * B[Q,j,l]
        print("Computing Coulomb contribution...")
        I_kjjl = np.einsum('Qkj,Qjl->kjl', B_array, B_array, optimize=True)
        
        # Add DSE contribution: d[k,j] * d[j,l]
        if not np.allclose(d_cmo, 0):
            print("Adding DSE contribution...")
            I_kjjl += np.einsum('kj,jl->kjl', d_cmo, d_cmo)
        
        # Sum only over j < k
        for k in range(nmo):
            # Vectorized over l: sum over j=0 to k-1
            if k > 0:
                gkl[k, :] -= np.sum(I_kjjl[k, :k, :], axis=0)
        
        # ============================================
        # Term 2: -(k>=l) * (kk|kl) / (1 + delta_{kl})
        # ============================================
        # Extract B[Q,k,k] for all k
        B_diag = B_array[np.arange(naux)[:, None], 
                         np.arange(nmo), 
                         np.arange(nmo)]  # (naux, nmo)
        
        # Compute (kk|kl) = sum_Q B[Q,k,k] * B[Q,k,l]
        I_kkkl = np.einsum('Qk,Qkl->kl', B_diag, B_array, optimize=True)
        
        # Add DSE: d[k,k] * d[k,l]
        if not np.allclose(d_cmo, 0):
            d_diag = np.diag(d_cmo)
            I_kkkl += d_diag[:, np.newaxis] * d_cmo
        
        # Apply only for k >= l with normalization
        for k in range(nmo):
            # Lower triangle: l <= k
            gkl[k, :k+1] -= I_kkkl[k, :k+1] / np.where(
                np.arange(k+1) == k, 2.0, 1.0
            )
        
        return gkl

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
        #grab the number of atoms and nuclear gradient 
        mol = psi4.geometry(molecule_string)
        self.mol = psi4.geometry(molecule_string)

        self.n_atoms = mol.natom()
        self.nuclear_energy_gradient =  np.asarray(mol.nuclear_repulsion_energy_deriv1()).flatten()

        # Parse output of cqed-rhf calculation
        psi4_wfn = self.parseArrays(cqed_rhf_dict)

        if self.natural_orbitals:  # <== need to run CIS
            self.buildArraysInOrbitalBasis(psi4_wfn, psi4_options_dict)
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

    def buildArraysInOrbitalBasis(self, p4_wfn, psi4_options_dict):
        # build 1H in orbital basis
        t_1H_start = time.time()
        self.build1HSO()
        t_1H_end = time.time()
        print(f" Completed 1HSO Build in {t_1H_end - t_1H_start} seconds")

        # build 2eInt in cqed-rhf basis
        print("number of MO", self.nmo)
        mints = psi4.core.MintsHelper(p4_wfn.basisset())
        
        self.mints = psi4.core.MintsHelper(p4_wfn.basisset())
        if self.full_diagonalization or self.test_mode or self.ci_level == "cis":
            self.eri_so = np.asarray(mints.mo_spin_eri(self.Ca, self.Ca))
        
        if "df_basis_scf" in psi4_options_dict:
            df_basis_name = psi4_options_dict["df_basis_scf"]
            self.df_basis_name = psi4_options_dict["df_basis_scf"]
            self.density_fitting = True
            primary_basis = p4_wfn.basisset()
            mol = p4_wfn.molecule()
            nbf = primary_basis.nbf()
            use_spherical = primary_basis.has_puream()

            df_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", df_basis_name, puream=use_spherical)
            naux = df_basis.nbf()
            self.wfn.set_basisset("DF_BASIS_SCF", df_basis)
            print(f"DF reconstruction: {nbf} primary, {naux} auxiliary")

            ## Get DF tensor
            #dfh = psi4.core.DFHelper(primary_basis, df_basis)
            #dfh.initialize()

            #identity = psi4.core.Matrix("Identity", nbf, nbf)
            #identity.identity()

            #dfh.add_space("AO", identity)
            #dfh.add_transformation("B", "AO", "AO")
            #dfh.transform()

            #B_tensor = dfh.get_tensor("B")
            #B_array = np.asarray(B_tensor)
            #print(f"DF tensor shape: {B_array.shape}")
            
            # Get raw (pq|A) integrals
            zero = psi4.core.BasisSet.zero_ao_basis_set()
            orb_basis = p4_wfn.basisset()
            
            # FIX: Just use df_basis that you already built above!
            self.aux_basis = df_basis  # <-- This is the fix!
            aux_basis = df_basis  # <-- This is the fix!
            
            raw_3c = mints.ao_eri(aux_basis, zero, orb_basis, orb_basis)
            # Convert to numpy and reshape
            naux = aux_basis.nbf()
            nbf = orb_basis.nbf()
            raw_3c_np = np.asarray(raw_3c).reshape(naux, nbf, nbf)
            self.raw_3c_np = np.copy(raw_3c_np)
        
           
            #######metric_obj = psi4.core.FittingMetric(aux_basis, True)
            #######metric_obj.form_eig_inverse(1.0e-14)
            #######metric_raw = np.array(metric_obj.get_metric())
            #######eigs = np.linalg.eigvalsh(metric_raw)
            #######cond = np.max(eigs) / np.min(eigs)
            #######print(np.max(eigs))
            #######print(np.max(eigs)* 1e-16)
            #######print(f"Condition Number for {self.df_basis_name}: {cond:.2e}")
            #######metric_obj.form_eig_inverse(np.max(eigs) * 1e-18)
            ########metric_obj.form_eig_inverse(1.0e-10)
           
            #######J_inv_half = np.asarray(metric_obj.get_metric())
            
            J = mints.ao_eri(aux_basis, zero, aux_basis, zero)
            J_np = np.asarray(J).reshape(naux, naux)
            #####max_eig = np.linalg.eigvalsh(J_np).max()
            #####print(max_eig)
            #####thresh = max_eig * 1e-12 # Relative threshold is safest
            ###### 3. Derive Gradient term (J^-1)
            #####J_inv_mat = J.clone()
            #####J_inv_mat.power(-1.0, thresh)
            #####self.J_inv = np.array(J_inv_mat).squeeze() 

            ########self.J_chol  = cho_factor(J_np)            # exact factorization, no threshold
            ########self.L_lower = np.linalg.cholesky(J_np)   # for energy only: J = L @ L^T

            ######### Apply metric: (pq|Q) = J^{-1/2} @ (pq|A)
            #########Qpq_reconstructed = np.einsum('QA,Amn->Qmn', J_inv_half, raw_3c_np)
            ########Qpq_reconstructed = solve_triangular(
            ########    self.L_lower,
            ########    raw_3c_np.reshape(naux, -1),
            ########    lower=True
            ########).reshape(naux, nbf, nbf)
            
            eigvals, eigvecs = eigh(J_np)            # eigvals ascending, eigvecs columns
            max_eig = eigvals[-1]
            #thresh  = max_eig * 1e-11               # single threshold, used everywhere
            thresh  = 1e-8                # single threshold, used everywhere
            
            mask       = eigvals > thresh            # True for kept eigenvectors
            inv_vals   = np.where(mask, 1.0 / eigvals,         0.0)
            inv_half   = np.where(mask, 1.0 / np.sqrt(eigvals), 0.0)
            
            # Store compact (truncated) factors — only the kept subspace
            # eigvecs_k: (naux, nkeep),  shape avoids naux x naux intermediates
            nkeep = mask.sum()
            eigvecs_k   = eigvecs[:, mask]           # (naux, nkeep)
            inv_vals_k  = inv_vals[mask]             # (nkeep,)
            inv_half_k  = inv_half[mask]             # (nkeep,)
            
            # Convenience: pseudo-solve  J^{-1} @ X  for X of shape (naux, ...)
            def pseudo_solve(X, inv_k=inv_vals_k, V=eigvecs_k):
                """Solves J @ result = X in the non-null subspace."""
                shape = X.shape
                X_flat = X.reshape(naux, -1)
                coeffs = V.T @ X_flat              # (nkeep, ncols)
                coeffs *= inv_k[:, None]           # divide by eigenvalues
                return (V @ coeffs).reshape(shape)
            
            self.pseudo_solve = pseudo_solve
            self.eigvecs_k    = eigvecs_k
            self.inv_half_k   = inv_half_k
            print(f"J condition: {max_eig:.3e},  kept {nkeep}/{naux} eigenvectors")
            B_flat = raw_3c_np.reshape(naux, -1)
            coeffs = self.eigvecs_k.T @ B_flat                  # (nkeep, nbf*nbf)
            coeffs *= self.inv_half_k[:, None]
            Qpq_reconstructed = (self.eigvecs_k @ coeffs).reshape(naux, nbf, nbf)


            #diff = np.max(np.abs(Qpq_reconstructed - B_array))
            #print(f"Difference: {diff:.2e}")  # Should be ~1e-10 or smaller
            B_array = np.copy(Qpq_reconstructed)
            self.B_array = np.copy(Qpq_reconstructed)

            self.B_mo = np.einsum("Qmn,mp,nq->Qpq", Qpq_reconstructed, self.C, self.C)            
            self.B_mo_hf = np.einsum("Qmn,mp,nq->Qpq", Qpq_reconstructed, self.C_hf, self.C_hf)            
            #Correct contraction for (naux, nbf, nbf) format
            #if B_array.shape == (naux, nbf, nbf):
            #    print("Using Einstein summation: (μν|λσ) = Σ_P B_P^μν * B_P^λσ")
            #    I = np.einsum('Pmn,Pls->mnls', B_array, B_array)
            # Reconstruct 4-index AO integrals
            ######I_ao_4d = np.einsum('Pmn,Pls->mnls', B_array, B_array)
            ######print(f"Reconstructed AO integrals shape: {I_ao_4d.shape}")  # (nbf, nbf, nbf, nbf)
            ######
            ####### Method 1: Reshape and use mo_transform
            ####### Reshape from (nbf, nbf, nbf, nbf) to (nbf*nbf, nbf*nbf)
            ######I_ao_2d = I_ao_4d.reshape(nbf*nbf, nbf*nbf)
            ######
            ####### Convert to Psi4 Matrix
            ######I_ao_mat = psi4.core.Matrix.from_array(I_ao_2d)
            ######I_mo_mat = mints.mo_transform(I_ao_mat, self.Ca, self.Ca, self.Ca, self.Ca)
            ######I_mo = np.asarray(I_mo_mat).reshape(nbf, nbf, nbf, nbf)
            ######self.I_mo = np.copy(I_mo)
            ######print(f"MO integrals shape: {I_mo.shape}")
            ######print("\n=== Verification ===")

            ######dfh_mo = psi4.core.DFHelper(primary_basis, df_basis)
            ######dfh_mo.initialize()
            ######dfh_mo.add_space("MO", self.Ca)
            ######dfh_mo.add_transformation("B_MO", "MO", "MO")
            ######dfh_mo.transform()
            ######self.B_mo = np.asarray(dfh_mo.get_tensor("B_MO"))
            ######self.B_mo = np.einsum("Qmn,mp,nq->Qpq", Qpq_reconstructed, self.C, self.C)            
            ####### Reconstruct using DF in MO basis
            ######I_mo_df = np.einsum('Qpq,Qrs->pqrs', self.B_mo, self.B_mo)
            ######
            ####### Compare
            ######diff = np.max(np.abs(I_mo - I_mo_df))
            ######print(f"Difference: {diff:.2e}")
            ######self.twoeint1 = np.copy(I_mo_df)
            if self.ignore_dse_terms:
                d_cmo_effective = np.zeros_like(self.d_cmo)
            else:
                d_cmo_effective = self.d_cmo 
            if _QED_SKIP_GKL:
                self.gkl = np.zeros((self.nmo, self.nmo))
            else:
                self.gkl = self.build_gkl_from_df_optimized(self.B_mo, d_cmo_effective, self.H_spatial)
            #for k in range(self.nmo):
            #    for l in range(self.nmo):
            #        print(self.gkl[k,l], self.gkl_new[k,l])
            #print(np.allclose(self.gkl, self.gkl_new)) 
            #self.twoeint1 = np.copy(I_mo_df)
            #self.build2DSO()
        else:
            self.density_fitting = False
            # The AO-basis ERI tensor used to be built here as a bare local `I`
            # (nmo^4, 8.4 GB at c10h8 dims) that no code in this file ever read.
            # It stayed live for the rest of buildArraysInOrbitalBasis -- which
            # calls build2DSO at :5317 -- so it was part of the run's peak.
            # nuclear_grad.py builds its own AO/MO integrals from Ca_hf and does
            # not use this one.  Restore as `self.twoeint_ao` if AO integrals are
            # ever actually needed; a local cannot be reached from anywhere.
            # QED_LIBTRANS builds the PACKED array directly (build2DSO picks it
            # up from self._twoeint_packed and skips the fold), so on that path
            # the dense nmo^4 twoeint1 is never allocated at all.  Any failure
            # returns None and falls through to mo_eri below.
            self._twoeint_packed = None
            # QED_LIBTRANS_OO == 1 is the whole point of the route: the packed
            # nmo^4 tensor is never built, because J and K are re-transformed
            # from the AO integrals each macro instead.  Mode 2 is the
            # cross-check and DOES need the tensor, so it falls through.
            self._oo_skip_tensor = _QED_LIBTRANS_OO == 1
            if self._oo_skip_tensor and (
                self.full_diagonalization or self.test_mode or self.ci_level != "cas"
            ):
                raise RuntimeError(
                    "QED_LIBTRANS_OO=1 replaces self.twoeint by a per-macro J/K "
                    "transform, which covers the CAS orbital-optimisation path "
                    "only.  full_diagonalization / test_mode / ci_level != 'cas' "
                    "read the tensor directly (build2DSO's d_spatial/d_spin "
                    "branch, build_sigma, Davidson).  Use QED_LIBTRANS_OO=2 to "
                    "cross-check, or leave the knob off."
                )
            if self._oo_skip_tensor:
                self.twoeint = None
            elif _QED_LIBTRANS:
                self._twoeint_packed = self._build_twoeint_libtrans(p4_wfn, mints)
            if not self._oo_skip_tensor and self._twoeint_packed is None:
                self.twoeint1 = np.asarray(
                    mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca)
                )

            #self.twoeint1 = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca))
            t_eri_end = time.time()
            print(f" Completed ERI Build in {t_eri_end - t_1H_end} seconds ")

            # form the 2H in spin orbital basis
            if not self._oo_skip_tensor:
                self.build2DSO()
            t_2d_end = time.time()
            print(f" Completed 2D build in {t_2d_end - t_eri_end} seconds")

            print(psutil.Process().memory_info().rss / (1024 * 1024))
            self.gkl = np.zeros((self.nmo, self.nmo))
            for k in range(0 if (_QED_SKIP_GKL or self._oo_skip_tensor) else self.nmo):
                for l in range(self.nmo):
                    self.gkl[k][l] = self.H_spatial[k][l]
                    for j in range(self.nmo):
                        if j >= k:
                            continue
                        else:
                            # packed: twoeint[kj][jl] -> [pair(k,j), pair(j,l)]
                            self.gkl[k][l] -= self.twoeint[
                                self.pair_index[k, j], self.pair_index[j, l]
                            ]
                    if k >= l:
                        self.gkl[k][l] -= self.twoeint[
                            self.pair_index[k, k], self.pair_index[k, l]
                        ] / (1 + (k == l))
            if self.twoeint is not None:
                print("mem_twoeint", self.twoeint.size * self.twoeint.itemsize / 1024 / 1024)
        print("density_fitting",self.density_fitting)
        print(psutil.Process().memory_info().rss / (1024 * 1024))
        # REFERENCE-BASIS SNAPSHOT for QED_LIBTRANS_OO.  Taken here because this
        # is the point at which self.twoeint is final, and because self.Ca
        # (:4117) and self.d_cmo (:7249) are BOTH reassigned later during
        # orbital optimisation -- reading them at macro time would silently
        # transform in the wrong basis.  Two nmo^2 arrays; negligible.
        self._oo_Ca_ref = np.array(self.Ca, copy=True)
        self._oo_d_ref = np.array(self.d_cmo, copy=True)
        print("mem_gkl", self.gkl.size * self.gkl.itemsize / 1024 / 1024)
        print("mem_d_cmo", self.d_cmo.size * self.d_cmo.itemsize / 1024 / 1024)

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

    # ---------------------------------------------------------------
    # The pure-Python CI kernels that stood here MOVED to
    # legacy_python_ci.py on 2026-09-04, having been proved unreachable
    # (static call graph + a call counter over the full suite AND both
    # SA-QED-CASSCF drivers -- zero calls everywhere).  ci_solver.c has
    # an equivalent or better implementation of each; that module maps
    # them one to one.  check_total_spin stays below: it is LIVE and is
    # already six lines of glue around c_sigma_s_square.
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # The pure-Python CI kernels that stood here MOVED to
    # legacy_python_ci.py on 2026-09-04, having been proved unreachable
    # (static call graph + a call counter over the full suite AND both
    # SA-QED-CASSCF drivers -- zero calls everywhere).  ci_solver.c has
    # an equivalent or better implementation of each; that module maps
    # them one to one.  check_total_spin stays below: it is LIVE and is
    # already six lines of glue around c_sigma_s_square.
    # ---------------------------------------------------------------

    def build_JK_oo(self):
        """Initial J,K for QED_LIBTRANS_OO -- the same restricted transform at
        U = I.  build_JK cannot be used here: it reads self.twoeint, which on
        this path was never built.
        """
        no, nmo = self.n_occupied, self.nmo
        self.J = np.empty((no, no, nmo, nmo))
        self.K = np.empty((no, no, nmo, nmo))
        self._oo_transform_JK(np.eye(nmo), self.J, self.K)

    def build_JK(self):
        # naive build of J_klrs and K_klrs from the full integrals
        # self.J = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.K = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.J = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:self.n_occupied,:self.n_occupied,:,:]
        # self.K = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:,:self.n_occupied,:,:self.n_occupied]
        # PACKED SOURCE.  self.twoeint is (npair, npair) now, so neither J nor K
        # can be a reshape+slice of it any more; both are unpacked explicitly.
        # Everything downstream (occupied_J at :2275, occupied_J3, the CI
        # solves) is built from J/K rather than from twoeint, so these two
        # arrays are the whole compatibility surface.
        #     J[k,l,r,s] = (kl|rs) = twoeint[pair(k,l), pair(r,s)]
        # tbl[r, s] maps the ket pair, so row[tbl] unpacks a whole (nmo, nmo)
        # block in one gather.  J is symmetric under k<->l.
        tbl = self.pair_index
        no = self.n_occupied
        self.J = np.empty((no, no, self.nmo, self.nmo))
        for k in range(no):
            for l in range(k, no):
                blk = self.twoeint[tbl[k, l]][tbl]
                self.J[k, l] = blk
                self.J[l, k] = blk
        #self.J_temp = copy.deepcopy(
        #    self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[
        #        : self.n_occupied, : self.n_occupied, :, :
        #    ]
        #)
        # The old code sliced (p,l,q,s) -> (pl|qs) and transposed (1,3,0,2), i.e.
        #     K[k,l,r,s] = (rk|sl) = (kr|ls)
        # so the BRA pair is (k,r) and the KET pair is (l,s).  Held (k,r) fixed
        # that is one packed row, and tbl[:no, :] gathers the (l,s) block, so
        # the loop runs over the n_occupied*nmo bra pairs rather than gathering
        # element by element.
        ket_occ = tbl[:no, :]
        self.K = np.empty((no, no, self.nmo, self.nmo))
        for k in range(no):
            for r in range(self.nmo):
                self.K[k, :, r, :] = self.twoeint[tbl[k, r]][ket_occ]
        #self.K_temp = self.K_temp.transpose(1, 3, 0, 2)
        # print (np.shape(self.K))
    def build_JK_from_df(self, B_mo, d_cmo, n_occupied, ignore_dse_terms=False):
        """
        Build J and K integrals from DF 3-index tensors
        
        Parameters:
        -----------
        B_mo : np.ndarray, shape (naux, nmo, nmo)
            DF 3-index tensor in MO basis: B[Q,p,q]
        d_cmo : np.ndarray, shape (nmo, nmo)
            Dipole matrix in MO basis for QED terms
        n_occupied : int
            Number of occupied orbitals
        ignore_dse_terms : bool
            If True, ignore dipole self-energy (DSE) contributions
        
        Returns:
        --------
        J : np.ndarray, shape (n_occ, n_occ, nmo, nmo)
            J[i,j,k,l] = (ij|kl) for i,j occupied
        K : np.ndarray, shape (n_occ, n_occ, nmo, nmo)
            K[j,l,p,q] = (pj|ql) for j,l occupied (after transpose)
        """
        naux, nmo, _ = B_mo.shape
        
        print(f"Building J and K from DF: {naux} aux, {nmo} MO, {n_occupied} occupied")
        
        # ================================================
        # Build J: (ij|kl) for i,j < n_occupied, all k,l
        # ================================================
        # Extract occupied-occupied block of B
        B_oo = B_mo[:, :n_occupied, :n_occupied]  # (naux, n_occ, n_occ)
        
        # Coulomb contribution: sum_Q B[Q,i,j] * B[Q,k,l]
        J = np.einsum('Qij,Qkl->ijkl', B_oo, B_mo, optimize=True)
        
        # Add DSE contribution: d[i,j] * d[k,l]
        if not ignore_dse_terms:
            d_oo = d_cmo[:n_occupied, :n_occupied]
            J += np.einsum('ij,kl->ijkl', d_oo, d_cmo)
        
        print(f"J shape: {J.shape}")
        
        # ================================================
        # Build K: (pj|ql) then transpose to (j,l,p,q)
        # ================================================
        # Extract all_orbitals × occupied block
        B_mo_occ = B_mo[:, :, :n_occupied]  # (naux, nmo, n_occ)
        
        # Coulomb contribution: sum_Q B[Q,p,j] * B[Q,q,l]
        # We want final shape (j,l,p,q), so use einsum directly
        K = np.einsum('Qpj,Qql->jlpq', B_mo_occ, B_mo_occ, optimize=True)
        
        # Add DSE contribution: d[p,j] * d[q,l] -> d[j,l,p,q] after transpose
        if not ignore_dse_terms:
            d_mo_occ = d_cmo[:, :n_occupied]  # (nmo, n_occ)
            K += np.einsum('pj,ql->jlpq', d_mo_occ, d_mo_occ)
        
        print(f"K shape: {K.shape}")
        
        return J, K

    def build_intermediates_internal(
        self,
        eigenvecs,
        A,
        G,
        occupied_fock_core,
        occupied_d_cmo,
        occupied_J,
        occupied_K,
    ):
        rot_dim = self.n_occupied
        L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        fock_general = np.zeros((rot_dim, rot_dim))
        # print (np.shape(self.L))
        L = (
            4.0 * occupied_K[:, : self.n_in_a, :rot_dim, :rot_dim]
            - occupied_K.transpose(0, 1, 3, 2)[:, : self.n_in_a, :rot_dim, :rot_dim]
            - occupied_J[:, : self.n_in_a, :rot_dim, :rot_dim]
        )
        fock_general += occupied_fock_core[:rot_dim, :rot_dim] + np.einsum(
            "tu,turs->rs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            occupied_J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
        )
        fock_general -= 0.5 * np.einsum(
            "tu,turs->rs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            occupied_K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
        )

        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)
        A[:, : self.n_in_a] = 2.0 * (
            fock_general[:, : self.n_in_a]
            + occupied_d_cmo[:rot_dim, : self.n_in_a] * off_diagonal_constant
        )

        A[:, self.n_in_a : self.n_occupied] = np.einsum(
            "rt,tu->ru",
            occupied_fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )
        twoeint_rtvw = occupied_J[
            :,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
        ]
        # print(np.shape(self.active_twoeint))
        A[:, self.n_in_a : self.n_occupied] += np.einsum(
            "rtvw,tuvw->ru",
            twoeint_rtvw[:rot_dim, :, :, :],
            self.D_tuvw_avg.reshape(
                (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
            ),
        )
        A[:, self.n_in_a : self.n_occupied] += -np.sqrt(self.omega / 2) * np.einsum(
            "rt,tu->ru",
            occupied_d_cmo[:rot_dim, self.n_in_a : self.n_occupied],
            self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )
        # print(self.A-self.A.transpose())

        # if full_space == True:
        # self.A2 = np.zeros((self.nmo, self.nmo))
        # for r in range(self.nmo):
        #    for i in range(self.n_in_a):
        #        self.A2[r][i] = 2.0 * self.fock_general[r][i]
        #        self.A2[r][i] += 2.0 * off_diagonal_constant * self.d_cmo[r][i]

        # for r in range(self.nmo):
        #    for u in range(self.n_act_orb):
        #        for t in range(self.n_act_orb):
        #            self.A2[r][u+self.n_in_a] += self.fock_core[r][t+self.n_in_a] * self.D_tu_avg[t*self.n_act_orb + u]
        #            self.A2[r][u+self.n_in_a] += -np.sqrt(self.omega/2) * self.d_cmo[r][t+self.n_in_a] * self.Dpe_tu_avg[t*self.n_act_orb + u]
        #            for v in range(self.n_act_orb):
        #                for w in range(self.n_act_orb):
        #                    tu = t * self.n_act_orb + u
        #                    vw = v * self.n_act_orb + w
        #                    self.A2[r][u+self.n_in_a] += self.twoeint_rtvw[r][t][v][w] * self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw]
        # print("uue", np.allclose(self.A2,self.A, rtol=1e-14,atol=1e-14))

        G[: self.n_in_a, : self.n_in_a, :, :] = 2.0 * np.einsum(
            "rs,ij->ijrs", fock_general[:rot_dim, :rot_dim], np.eye(self.n_in_a)
        )
        G[: self.n_in_a, : self.n_in_a, :, :] += 2.0 * L[: self.n_in_a, :, :, :]
        G[: self.n_in_a, : self.n_in_a, :, :] += (
            2.0
            * off_diagonal_constant
            * np.einsum(
                "rs,ij->ijrs", occupied_d_cmo[:rot_dim, :rot_dim], np.eye(self.n_in_a)
            )
        )

        G[self.n_in_a : self.n_occupied, : self.n_in_a, :, :] = np.einsum(
            "tv,vjrs->tjrs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            L[self.n_in_a :, :, :, :],
        )
        G[: self.n_in_a, self.n_in_a : self.n_occupied, :, :] = G[
            self.n_in_a : self.n_occupied, : self.n_in_a, :, :
        ].transpose(1, 0, 3, 2)

        G[self.n_in_a :, self.n_in_a :, :, :] = np.einsum(
            "rs,tu->turs",
            occupied_fock_core[:rot_dim, :rot_dim],
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )
        G[self.n_in_a :, self.n_in_a :, :, :] += np.einsum(
            "vwrs,tuvw->turs",
            occupied_J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
            self.D_tuvw_avg.reshape(
                (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
            ),
        )
        G[self.n_in_a :, self.n_in_a :, :, :] += 2.0 * np.einsum(
            "vwrs,tvuw->turs",
            occupied_K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
            self.D_tuvw_avg.reshape(
                (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
            ),
        )
        G[self.n_in_a :, self.n_in_a :, :, :] += -np.sqrt(self.omega / 2) * np.einsum(
            "rs,tu->turs",
            occupied_d_cmo[:rot_dim, :rot_dim],
            self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )
        # for r in range(self.nmo):
        #    for s in range(self.nmo):
        #        for v in range(self.n_act_orb):
        #            for w in range(self.n_act_orb):
        #                dum = self.K[v][w][r][s]-self.K[w][v][s][r]
        #                if dum > 1e-12: print('{:.5f}'.format(dum))

        print(
            "n_act_orb,n_in_orb, nmo,nvir",
            self.n_act_orb,
            self.n_in_a,
            self.nmo,
            self.n_virtual,
        )
        # print("ooq", np.allclose(self.G2,G, rtol=1e-14,atol=1e-14))
        # self.G3 = G.transpose(0,2,1,3)
        # self.G3 = self.G3.reshape(self.n_occupied * rot_dim,self.n_occupied * rot_dim)
        # print("symmetric test", np.allclose(self.G3.T,self.G3, rtol=1e-14,atol=1e-14))
        ##print(self.G3)
        # for r in range(rot_dim):
        #    for s in range(rot_dim):
        #        for k in range(self.n_occupied):
        #            for l in range(self.n_occupied):
        #                dum = self.G3.T[k*rot_dim+r][l*rot_dim+s]-self.G3[k*rot_dim+r][l*rot_dim+s]
        #                #if dum > 1e-14: print('{:.14f}'.format(dum), k, l, r, s)
        #                #if dum > 1e-14: print('{:.16f}'.format(self.G3.T[k*self.nmo+r][l*self.nmo+s]),'{:.16f}'.format(self.G3[k*self.nmo+r][l*self.nmo+s]) , 2.0 * self.fock_general[r][s],k, l, r, s)
        #                if dum > 1e-14: print('{:.16f}'.format(self.G3[l*rot_dim+s][k*rot_dim+r]),'{:.16f}'.format(self.G3[k*rot_dim+r][l*rot_dim+s]) ,
        #                        #2.0 * self.fock_general[r][s],2.0 * self.fock_general[s][r],k, l, r, s)
        #                        2.0 * self.fock_core[r][s],2.0 * self.fock_core[s][r],k, l, r, s)

        # print("trq", self.G3.T-self.G3)
        # print(self.G2-self.G)

    def build_intermediates(self, eigenvecs, A, G, full_space):
        if full_space == True:
            rot_dim = self.nmo
        else:
            rot_dim = self.n_occupied

        # Per-section profiling, gated on QED_PROFILE_INTERMEDIATES (default
        # 0 = off).  Replaces the hand-commented "build intermediate step N"
        # timers that were scattered through this function; the labels below
        # keep the same section boundaries.  Emits one `BI_PROF <label> <secs>`
        # line per section per call -- aggregate over a run with:
        #   grep '^BI_PROF ' log | awk '{s[$2]+=$3; n[$2]++}
        #     END{for(k in s) printf "%-22s %8.2f %6d %8.4f\n", k, s[k], n[k], s[k]/n[k]}' | sort -k2 -rn
        _prof_t = timer() if _QED_PROFILE_INTERMEDIATES else 0.0

        def _bi_mark(label):
            nonlocal _prof_t
            if _QED_PROFILE_INTERMEDIATES:
                _now = timer()
                print(f"BI_PROF {label} {_now - _prof_t:.6f}")
                _prof_t = _now

        # start = timer()
        self.fock_core = copy.deepcopy(self.H_spatial2)
        self.fock_core += 2.0 * np.einsum(
            "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        self.fock_core -= np.einsum(
            "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        D_tu_avg = self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb))
        Dpe_tu_avg = self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb))
        D_tuvw_avg = self.D_tuvw_avg.reshape(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        self.E_core = 0.0
        self.E_core += np.einsum("jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a])
        self.E_core += np.einsum("jj->", self.fock_core[: self.n_in_a, : self.n_in_a])
        _bi_mark("fock_core_Ecore")
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
        _bi_mark("active_twoeint_copy")
        self.fock_general = np.zeros((rot_dim, rot_dim))
        # `self.L = 4K - K^T - J` is a PURE FUNCTION of self.J/self.K, which are
        # only rebound when the integrals are retransformed at a macroiteration
        # boundary (the full_transformation_macroiteration / transform_JK_with_df
        # branches around line 3145).  It does NOT depend on the RDMs.
        #
        # The rest of this routine legitimately rebuilds every microiteration:
        # that mirrors Kreplin's Uncoupled(CI) ("the one- and two-particle RDMs
        # are recalculated, and the orbital optimization is restarted"), and our
        # measured rate -- 9.2 microiterations per macroiteration on c10h8 -- sits
        # inside their N_micro range of 5/7/20.  L simply has no business being
        # inside that density-tracking rebuild: it was being built 101 times on a
        # c10h8 run that needed it 11 times, at 0.27 s and ~1.2 GB of traffic each
        # (44% of build_intermediates; see QED_PROFILE_INTERMEDIATES).
        #
        # INVALIDATION -- read this before touching it.  J/K change in TWO
        # different ways and the cache must catch both:
        #   (a) REBOUND to new arrays  (build_JK_from_df at ~1484/~1710,
        #       transform_JK_with_df at ~3145)  -> caught by the weakref identity
        #       test below;
        #   (b) OVERWRITTEN IN PLACE by the C backend -- c_full_transformation_
        #       macroiteration (~3127) and c_full_transformation_internal_
        #       optimization (~3051 and ~8131, the latter inside
        #       internal_optimization3).  These pass J/K as output buffers, so
        #       the Python object identity does NOT change.  An identity or
        #       weakref test CANNOT see this.  Caught by _JK_WRITE_EPOCH, which
        #       those two wrappers bump.
        # An earlier version of this cache used the weakref test ALONE.  It
        # silently served a stale L after every integral transformation; c10h8
        # then diverged (macro 2 came out at -382.69 vs the correct -383.48).
        # Do not drop the epoch check.
        #
        # `temp1` (J_tt - 0.5 K_tt) shares this cache for the same reason.
        #
        # NOTE self.L is also read outside this routine (helper_PFCI.py:8987 and
        # :9041, both inside the second-order integral transformation), so the
        # cached array must be -- and is -- bit-identical to a fresh build.
        # Checked: no consumer, here or there, writes into self.L or temp1 in
        # place (8987 is `local_temp1 += self.L[...]`, which mutates the local),
        # which is what makes it safe to hold these across microiterations.
        _L_ref_J = getattr(self, "_L_cache_J", None)
        _L_ref_K = getattr(self, "_L_cache_K", None)
        _L_is_current = (
            _L_ref_J is not None
            and _L_ref_K is not None
            and _L_ref_J() is self.J
            and _L_ref_K() is self.K
            and getattr(self, "_L_cache_rot_dim", None) == rot_dim
            and getattr(self, "_L_cache_epoch", None) == _JK_WRITE_EPOCH
        )
        if _L_is_current and _QED_VERIFY_L_CACHE:
            # Paranoia mode: rebuild and compare, so a missed invalidation shows
            # up as a loud mismatch instead of a silently wrong Hessian.
            _L_check = (
                4.0 * self.K[:, : self.n_in_a, :rot_dim, :rot_dim]
                - self.K.transpose(0, 1, 3, 2)[:, : self.n_in_a, :rot_dim, :rot_dim]
                - self.J[:, : self.n_in_a, :rot_dim, :rot_dim]
            )
            if not np.array_equal(_L_check, self.L):
                raise RuntimeError(
                    "L cache is STALE: J/K changed without bumping "
                    "_JK_WRITE_EPOCH or rebinding. max|dL| = "
                    f"{np.abs(_L_check - self.L).max():.6e}"
                )
            del _L_check
        if not _L_is_current:
            # (the `self.L = np.zeros(...)` that used to sit here was dead -- a
            #  293 MB memset discarded by the very next statement.)
            self.L = (
                4.0 * self.K[:, : self.n_in_a, :rot_dim, :rot_dim]
                - self.K.transpose(0, 1, 3, 2)[:, : self.n_in_a, :rot_dim, :rot_dim]
                - self.J[:, : self.n_in_a, :rot_dim, :rot_dim]
            )
            # temp1 = J_tt,rs - 0.5 K_tt,rs is likewise a pure function of J/K
            # (26 MB at c10h8 dims), so it shares this cache rather than getting
            # a second, identical weakref guard.  Its single consumer is the
            # fock_general einsum immediately below.
            self._temp1_cache = (
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
            self._L_cache_J = weakref.ref(self.J)
            self._L_cache_K = weakref.ref(self.K)
            self._L_cache_rot_dim = rot_dim
            self._L_cache_epoch = _JK_WRITE_EPOCH
        _bi_mark("L_build")

        temp1 = self._temp1_cache
        _bi_mark("temp1_build")
        # start = timer()
        self.fock_general += self.fock_core[:rot_dim, :rot_dim] + np.einsum(
            "tu,turs->rs", D_tu_avg, temp1, optimize="optimal"
        )
        _bi_mark("fock_general")
        
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)
        A[:, : self.n_in_a] = 2.0 * (
            self.fock_general[:, : self.n_in_a]
            + self.d_cmo[:rot_dim, : self.n_in_a] * off_diagonal_constant
        )
        _bi_mark("A_inactive")

       
        A[:, self.n_in_a : self.n_occupied] = np.einsum(
            "rt,tu->ru",
            self.fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            D_tu_avg,
            optimize="optimal",
        )
        _bi_mark("A_act_fock")
        # end   = timer()
        # print("build intermediate step 4", end - start)
        # print(np.shape(self.active_twoeint))
        # start = timer()
        A[:, self.n_in_a : self.n_occupied] += np.einsum(
            "vwrt,tuvw->ru",
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                self.n_in_a : self.n_occupied,
            ],
            D_tuvw_avg,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 5", end - start)
        _bi_mark("A_act_2rdm")
        # start = timer()
        A[:, self.n_in_a : self.n_occupied] += -np.sqrt(self.omega / 2) * np.einsum(
            "rt,tu->ru",
            self.d_cmo[:rot_dim, self.n_in_a : self.n_occupied],
            Dpe_tu_avg,
            optimize="optimal",
        )
        temp2 = (
            self.fock_general[:rot_dim, :rot_dim]
            + off_diagonal_constant * self.d_cmo[:rot_dim, :rot_dim]
        )
        _bi_mark("A_act_dipole_temp2")
        # end1   = timer()
        # print("build intermediate step 7_0", end1 - start1)
        # start1 = timer()
        # Was:
        #   G[:ni,:ni,:,:]  = 2.0 * einsum("rs,ij->ijrs", temp2, eye(ni))
        #   G[:ni,:ni,:,:] += 2.0 * L[:ni]
        # The einsum materialized delta_ij (x) temp2 as a full (ni,ni,rot,rot)
        # array -- 218 MB at c10h8 dims -- of which only the ni diagonal (i==j)
        # pairs are nonzero: 29 of 841, so 96.6% of what it wrote was zeros that
        # the `+= 2L` on the next line then overwrote.  Measured at 14.0 s of the
        # 61.1 s build_intermediates total (23%).
        # Now: assign the 2L part first, then add temp2 on the i==j diagonal
        # only.  Algebraically identical (fancy-indexing (i,i) pairs are unique,
        # so the += is a well-defined gather/scatter), 7.5 MB of adds instead of
        # a 218 MB einsum.
        if _QED_BI_FASTPATH:
            # Same values, bit-for-bit (x2.0 is exact); the temporary is gone.
            # See the QED_BI_FASTPATH note at the top of the file.
            np.multiply(
                self.L[: self.n_in_a, :, :, :],
                2.0,
                out=G[: self.n_in_a, : self.n_in_a, :, :],
            )
        else:
            G[: self.n_in_a, : self.n_in_a, :, :] = 2.0 * self.L[: self.n_in_a, :, :, :]
        _bi_mark("G_ij_plus_2L")
        _ii = np.arange(self.n_in_a)
        G[_ii, _ii, :, :] += 2.0 * temp2
        _bi_mark("G_ij_diag_temp2")
        # end1   = timer()
        # print("build intermediate step 7_2", end1 - start1)
        # end   = timer()
        # print("build intermediate step 7", end - start)

        # start = timer()
        if _QED_BI_FASTPATH:
            # "tv,vjrs->tjrs" with v contracted and t free is a plain GEMM once
            # the (v, j, r, s) block is flattened: self.L[n_in_a:] slices only
            # axis 0 of a C-contiguous array, so the reshape below is a free
            # view and no transpose is needed.  np.dot demands a C-contiguous
            # out=, and G's destination is a strided slice, so the product lands
            # in a buffer that is allocated once and reused -- writing into a
            # fresh 75 MB array every call costs more than the GEMM (1.67x vs
            # 4.07x standalone).  See the QED_BI_FASTPATH note at the top.
            _La = self.L[self.n_in_a :, :, :, :]
            _buf = getattr(self, "_bi_gti_buf", None)
            if _buf is None or _buf.shape != _La.shape:
                _buf = np.zeros(_La.shape)
                self._bi_gti_buf = _buf
            _nv = _La.shape[0]
            np.dot(
                D_tu_avg,
                _La.reshape(_nv, -1),
                out=_buf.reshape(_nv, -1),
            )
            G[self.n_in_a : self.n_occupied, : self.n_in_a, :, :] = _buf
        else:
            G[self.n_in_a : self.n_occupied, : self.n_in_a, :, :] = np.einsum(
                "tv,vjrs->tjrs",
                D_tu_avg,
                self.L[self.n_in_a :, :, :, :],
                optimize="optimal",
            )
        _bi_mark("G_ti_D_times_L")
        # end   = timer()
        # print("build intermediate step 8", end - start)
        # start = timer()
        G[: self.n_in_a, self.n_in_a : self.n_occupied, :, :] = G[
            self.n_in_a : self.n_occupied, : self.n_in_a, :, :
        ].transpose(1, 0, 3, 2)
        _bi_mark("G_it_transpose")

        # end   = timer()
        # print("build intermediate step 9", end - start)
        # start = timer()
        G[self.n_in_a :, self.n_in_a :, :, :] = np.einsum(
            "rs,tu->turs",
            self.fock_core[:rot_dim, :rot_dim],
            D_tu_avg,
            optimize="optimal",
        )
        _bi_mark("G_tu_fock_outer")
        # end   = timer()
        # print("build intermediate step 10", end - start)
        # start = timer()
        G[self.n_in_a :, self.n_in_a :, :, :] += np.einsum(
            "vwrs,tuvw->turs",
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
            D_tuvw_avg,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 11", end - start)
        _bi_mark("G_tu_J_2rdm")
        # start = timer()
        G[self.n_in_a :, self.n_in_a :, :, :] += 2.0 * np.einsum(
            "vwrs,tvuw->turs",
            self.K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
            D_tuvw_avg,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 12", end - start)
        _bi_mark("G_tu_K_2rdm")
        # start = timer()
        G[self.n_in_a :, self.n_in_a :, :, :] += -np.sqrt(self.omega / 2) * np.einsum(
            "rs,tu->turs",
            self.d_cmo[:rot_dim, :rot_dim],
            Dpe_tu_avg,
            optimize="optimal",
        )
        _bi_mark("G_tu_dipole")
        
    

    def build_intermediates2(self, eigenvecs, A, G, full_space):
        if full_space == True:
            rot_dim = self.nmo
        else:
            rot_dim = self.n_occupied
        self.fock_core = copy.deepcopy(self.H_spatial2)
        self.fock_core += 2.0 * np.einsum(
            "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :]
        )
        self.fock_core -= np.einsum(
            "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :]
        )

        self.E_core = 0.0
        self.E_core += np.einsum("jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a])
        self.E_core += np.einsum("jj->", self.fock_core[: self.n_in_a, : self.n_in_a])

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
        self.L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        self.fock_general = np.zeros((rot_dim, rot_dim))
        print(np.shape(self.L))
        self.L = (
            4.0 * self.K[:, : self.n_in_a, :rot_dim, :rot_dim]
            - self.K.transpose(0, 1, 3, 2)[:, : self.n_in_a, :rot_dim, :rot_dim]
            - self.J[:, : self.n_in_a, :rot_dim, :rot_dim]
        )

        # self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)),
        self.fock_general += np.einsum(
            "tu,turs->rs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
        )
        self.fock_general -= 0.5 * np.einsum(
            "tu,turs->rs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            self.K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ],
        )

        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(eigenvecs)

        # A[:,:self.n_in_a] = 2.0 * (self.fock_general[:,:self.n_in_a] + self.d_cmo[:rot_dim,:self.n_in_a] * off_diagonal_constant)
        A[:, : self.n_in_a] = 2.0 * (self.fock_general[:, : self.n_in_a])

        A[:, self.n_in_a : self.n_occupied] = np.einsum(
            "rt,tu->ru",
            self.fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )
        # print(np.shape(self.active_twoeint))
        # A[:,self.n_in_a:self.n_occupied] += np.einsum("vwrt,tuvw->ru", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,self.n_in_a:self.n_occupied],
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        # A[:,self.n_in_a:self.n_occupied] += -np.sqrt(self.omega/2) * np.einsum("rt,tu->ru", self.d_cmo[:rot_dim,self.n_in_a:self.n_occupied],
        #        self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))
        # print(self.A-self.A.transpose())

        G[: self.n_in_a, : self.n_in_a, :, :] = 2.0 * np.einsum(
            "rs,ij->ijrs", self.fock_general[:rot_dim, :rot_dim], np.eye(self.n_in_a)
        )
        # G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * self.L[:self.n_in_a,:,:,:]
        # G[:self.n_in_a,:self.n_in_a,:,:] += 2.0 * off_diagonal_constant * np.einsum("rs,ij->ijrs", self.d_cmo[:rot_dim,:rot_dim],
        #   np.eye(self.n_in_a))

        G[self.n_in_a : self.n_occupied, : self.n_in_a, :, :] = np.einsum(
            "tv,vjrs->tjrs",
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
            self.L[self.n_in_a :, :, :, :],
        )
        G[: self.n_in_a, self.n_in_a : self.n_occupied, :, :] = G[
            self.n_in_a : self.n_occupied, : self.n_in_a, :, :
        ].transpose(1, 0, 3, 2)

        G[self.n_in_a :, self.n_in_a :, :, :] = np.einsum(
            "rs,tu->turs",
            self.fock_core[:rot_dim, :rot_dim],
            self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
        )

        # G[self.n_in_a:,self.n_in_a:,:,:] += np.einsum("vwrs,tuvw->turs", self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        # G[self.n_in_a:,self.n_in_a:,:,:] += 2.0 * np.einsum("vwrs,tvuw->turs", self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim],
        #   self.D_tuvw_avg.reshape((self.n_act_orb,self.n_act_orb,self.n_act_orb,self.n_act_orb)))
        # G[self.n_in_a:,self.n_in_a:,:,:] += -np.sqrt(self.omega/2) * np.einsum("rs,tu->turs", self.d_cmo[:rot_dim,:rot_dim],
        #        self.Dpe_tu_avg.reshape((self.n_act_orb,self.n_act_orb)))

    def calculate_ci_dependent_energy(self, eigenvecs, occupied_d_cmo):
        off_diagonal_constant_energy = 0.0
        photon_energy = 0.0
        d_diag = 0.0
        # for i in range(self.n_in_a):
        #    d_diag += 2.0 * self.d_cmo[i][i]
        d_diag = 2.0 * np.einsum("ii->", occupied_d_cmo[: self.n_in_a, : self.n_in_a])
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            # print("weight", self.weight[i])
            eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
            eigenvecs2 = eigenvecs2.transpose(1, 0)
            for m in range(np1):
                if self.N_p == 0:
                    continue
                if m > 0 and m < self.N_p:
                    off_diagonal_constant_energy += (
                        self.weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * (self.d_exp - d_diag)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                    off_diagonal_constant_energy += (
                        self.weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * (self.d_exp - d_diag)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )
                elif m == self.N_p:
                    off_diagonal_constant_energy += (
                        self.weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * (self.d_exp - d_diag)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                else:
                    off_diagonal_constant_energy += (
                        self.weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * (self.d_exp - d_diag)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )
                photon_energy += (
                    self.weight[i]
                    * m
                    * self.omega
                    * np.dot(
                        eigenvecs2[:, m : (m + 1)].flatten(),
                        eigenvecs2[:, m : (m + 1)].flatten(),
                    )
                )

        # ci_dependent_energy = (off_diagonal_constant_energy + photon_energy) / self.davidson_roots
        ci_dependent_energy = off_diagonal_constant_energy + photon_energy
        return ci_dependent_energy

    def calculate_off_diagonal_photon_constant(self, eigenvecs):
        off_diagonal_constant = 0.0
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            # print("weight", self.weight[i])
            eigenvecs2 = eigenvecs[i].reshape((np1, self.num_det))
            eigenvecs2 = eigenvecs2.transpose(1, 0)
            for m in range(np1):
                if self.N_p == 0:
                    continue
                if m > 0 and m < self.N_p:
                    off_diagonal_constant += (
                        -self.weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                    off_diagonal_constant += (
                        -self.weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )
                elif m == self.N_p:
                    off_diagonal_constant += (
                        -self.weight[i]
                        * np.sqrt(m * self.omega / 2)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m - 1) : m].flatten(),
                        )
                    )
                else:
                    off_diagonal_constant += (
                        -self.weight[i]
                        * np.sqrt((m + 1) * self.omega / 2)
                        * np.dot(
                            eigenvecs2[:, m : (m + 1)].flatten(),
                            eigenvecs2[:, (m + 1) : (m + 2)].flatten(),
                        )
                    )

        return off_diagonal_constant

    def _d2h_ao_operations(self):
        """Signed AO matrices for the 8 D2h operations, built once and cached.

        Centre permutation from the overlap between the original basis and one
        placed at the transformed geometry, times the angular sign factor per
        AO component.  Validated on construction (R^T S R = S); returns None if
        the geometry is not D2h at 1e-6, so the projector silently disables
        itself on a geometry where it does not apply.
        """
        if getattr(self, "_d2h_cache", "miss") != "miss":
            return self._d2h_cache
        self._d2h_cache = None
        try:
            mol = self.wfn.molecule()
            bohr = 0.52917721067
            P = np.array(mol.geometry()) * bohr
            sym = [mol.symbol(i) for i in range(mol.natom())]
            ci = [i for i, t in enumerate(sym) if t.upper() == "C"]
            P = P - (P[ci].mean(axis=0) if ci else P.mean(axis=0))
            bs = self.wfn.basisset()
            S = np.asarray(psi4.core.MintsHelper(bs).ao_overlap())
            Sinv = np.linalg.inv(S)
            PAR = {0: [(0, 0, 0)],
                   1: [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
                   2: [(0, 0, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0), (1, 1, 0)]}
            expo = np.zeros((S.shape[0], 3), int)
            for si in range(bs.nshell()):
                sh = bs.shell(si)
                if sh.am not in PAR:
                    return self._d2h_cache
                off = bs.shell_to_basis_function(si)
                for k, e in enumerate(PAR[sh.am]):
                    expo[off + k] = e
            ops = []
            for sx, sy, sz in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
                               (-1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
                g = "\n".join("%s %.12f %.12f %.12f"
                               % (t, q[0] * sx, q[1] * sy, q[2] * sz)
                               for t, q in zip(sym, P))
                mt = psi4.geometry(g + "\nunits angstrom\nsymmetry c1"
                                       "\nno_reorient\nnocom")
                wt = psi4.core.Wavefunction.build(mt, bs.name())
                perm = Sinv @ np.asarray(
                    psi4.core.MintsHelper(bs).ao_overlap(bs, wt.basisset()))
                sgn = np.array([(sx ** e[0]) * (sy ** e[1]) * (sz ** e[2])
                                for e in expo], float)
                Rg = perm * sgn[None, :]
                if np.abs(Rg.T @ S @ Rg - S).max() > 1e-6:
                    print("QED_D2H_PROJECT: geometry is not D2h; projector OFF",
                          flush=True)
                    return self._d2h_cache
                ops.append(Rg)
            self._d2h_cache = (ops, S)
        except Exception as _e:                       # never break a normal run
            print("QED_D2H_PROJECT: setup failed (%s); projector OFF" % _e,
                  flush=True)
            self._d2h_cache = None
        return self._d2h_cache

    def build_unitary_matrix(self, Rai, Rvi, Rva):
        R = np.zeros((self.nmo, self.nmo))
        self.U_delta = np.zeros((self.nmo, self.nmo))

        R[self.n_in_a : self.n_occupied, : self.n_in_a] = Rai
        R[: self.n_in_a, self.n_in_a : self.n_occupied] = -Rai.T
        R[self.n_occupied :, : self.n_in_a] = Rvi
        R[: self.n_in_a, self.n_occupied :] = -Rvi.T
        R[self.n_occupied :, self.n_in_a : self.n_occupied] = Rva
        R[self.n_in_a : self.n_occupied, self.n_occupied :] = -Rva.T
        np.set_printoptions(precision=5)

        # QED_D2H_PROJECT: group-average the rotation GENERATOR over D2h so the
        # step stays inside the symmetric manifold.  R is the antisymmetric
        # kappa; exp of a group-invariant kappa preserves the symmetry of the
        # orbitals it acts on, which is why the projection belongs HERE and not
        # on U_total -- by the time U_total is updated the integrals have
        # already been transformed by the same step and would desynchronise.
        #
        # WHY IT IS NEEDED (measured 2026-09-21): purifying only the starting
        # orbitals and re-converging leaves the x-flipping partition leak at
        # 0.795 from the first lambda > 0 step.  Alternating purify/converge
        # from outside oscillates instead of converging -- the optimiser
        # recreates the same core/active mixing (eigenvalue 0.648) each pass.
        if _QED_D2H_PROJECT:
            _cache = self._d2h_ao_operations()
            if _cache is not None:
                _ops, _S = _cache
                _Ccur = np.dot(self.C, self.U_total)
                _M = [_Ccur.T @ _S @ _g @ _Ccur for _g in _ops]
                R = sum(_m @ R @ _m.T for _m in _M) / len(_M)
                R = 0.5 * (R - R.T)            # re-antisymmetrise against drift

        R1 = np.dot(-R, R)
        tau_square, W = np.linalg.eigh(R1)
        sine_product_array = np.zeros(self.nmo)
        cosine_array = np.zeros(self.nmo)
        for i in range(self.nmo):
            if tau_square[i] < 0.0:
                tau_square[i] = 0.0
            tau = np.sqrt(tau_square[i])
            cosine_array[i] = np.cos(tau)
            if np.sqrt(tau_square[i]) > 1e-15:
                sine_product_array[i] = np.sin(tau) / tau
            else:
                # print("dfgsg")
                sine_product_array[i] = 1 - pow(tau, 2) / 6 + pow(tau, 4) / 120
        # self.U_delta = np.einsum("pq,qs,st->pt", W, np.diag(cosine_array), W.T)
        # temp = np.einsum("pq,qs->ps", W.T, R)
        # self.U_delta += np.einsum("pq,qs,st->pt", W, np.diag(sine_product_array), temp)
        temp = np.dot(np.diag(cosine_array), W.T)
        self.U_delta = np.dot(W, temp)
        temp = np.dot(W.T, R)
        R1 = np.dot(np.diag(sine_product_array), temp)
        self.U_delta += np.dot(W, R1)

    def build_gradient(self, U, A, G, gradient_tilde, A_tilde, full_space,
                       G_blocks=None):
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            if _QED_GRAD_BLOCKS and G_blocks is not None:
                # QED_GRAD_BLOCKS: the same contraction off the blocks that are
                # already built at :12281.  Index convention is the one
                # build_sigma_reduced7 and microiteration_exact_energy:10009
                # assume -- rows (s,l), columns (r,k), flattened r*n_block + k --
                # so B_i / B_a below reshape straight back into B's columns.
                # See the QED_GRAD_BLOCKS note at the top of the file.
                G_ij, G_ti, G_tu = G_blocks
                ni = self.n_in_a
                na = self.n_act_orb
                T_i = np.ascontiguousarray(T[:, :ni]).reshape(self.nmo * ni)
                T_a = np.ascontiguousarray(
                    T[:, ni : self.n_occupied]
                ).reshape(self.nmo * na)
                # A is copied, not accumulated into, so the caller's A is safe.
                B_i = np.ascontiguousarray(A[:, :ni]).reshape(self.nmo * ni)
                B_a = np.ascontiguousarray(
                    A[:, ni : self.n_occupied]
                ).reshape(self.nmo * na)
                # symmetric blocks -> one triangle each; np.dot(T_i, G_ij) is
                # G_ij @ T_i because G_ij == G_ij.T (QED_G_SYM_CHECK, :12333)
                if _QED_G_SYMV:
                    _symv_accumulate(G_ij, T_i, B_i)
                    _symv_accumulate(G_tu, T_a, B_a)
                else:
                    B_i += np.dot(T_i, G_ij)
                    B_a += np.dot(T_a, G_tu)
                # G_it is G_ti.T, a free view -- never materialized
                B_i += np.dot(T_a, G_ti)
                B_a += np.dot(T_i, G_ti.T)
                B[:, :ni] = B_i.reshape(self.nmo, ni)
                B[:, ni : self.n_occupied] = B_a.reshape(self.nmo, na)
                # QED_GRAD_BLOCKS_CHECK: one-shot proof, in situ and at the real
                # dimensions, that the blocked contraction reproduces the einsum.
                # This is NOT bit-exact (BLAS vs einsum summation order), so the
                # test is a relative tolerance -- but it is a DIRECT array
                # comparison, which is the only valid evidence here: the c10h8
                # run is nondeterministic upstream of CASSCF (threaded-BLAS
                # reduction order in the CQED-RHF SCF), so two end-to-end legs
                # agreeing proves nothing about this statement.
                if int(os.environ.get("QED_GRAD_BLOCKS_CHECK", "0")) and not getattr(
                    self, "_grad_blocks_checked", False
                ):
                    self._grad_blocks_checked = True
                    _ref = A[:, : self.n_occupied] + np.einsum(
                        "klrs,sl->rk", G[:, :, :, :], T[:, : self.n_occupied]
                    )
                    _got = B[:, : self.n_occupied]
                    _s = np.abs(_ref).max()
                    _d = np.abs(_got - _ref).max()
                    print(
                        f"[GRAD_BLOCKS_CHECK] B{_got.shape} max|B|={_s:.6e} "
                        f"max|d|={_d:.6e} rel={_d/max(_s,1e-300):.3e}",
                        flush=True,
                    )
                    if _d > 1e-10 * max(_s, 1.0):
                        raise RuntimeError(
                            f"QED_GRAD_BLOCKS: B differs from the einsum path, "
                            f"max|d| = {_d:.6e}"
                        )
                    del _ref
            else:
                B[:, : self.n_occupied] = A[:, : self.n_occupied] + np.einsum(
                    "klrs,sl->rk", G[:, :, :, :], T[:, : self.n_occupied]
                )

            A_tilde[:, : self.n_occupied] = np.einsum(
                "rs,sk->rk", U.T, B[:, : self.n_occupied]
            )
            # A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)

            gradient_tilde[:, :] = (
                A_tilde[:, : self.n_occupied] - A_tilde.T[:, : self.n_occupied]
            )

        else:

            gradient_tilde[:, :] = A[:, :] - A.T[:, :]

    def build_gradient2(
        self, U, A, G, hessian_tilde, gradient_tilde, A_tilde, full_space
    ):
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            B[:, : self.n_occupied] = A[:, : self.n_occupied] + np.einsum(
                "klrs,sl->rk",
                G[:, :, :, :],
                T[:, : self.n_occupied],
                optimize="optimal",
            )

            A_tilde[:, : self.n_occupied] = np.einsum(
                "rs,sk->rk", U.T, B[:, : self.n_occupied], optimize="optimal"
            )
            # A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)

            gradient_tilde[:, :] = (
                A_tilde[:, : self.n_occupied] - A_tilde.T[:, : self.n_occupied]
            )
            A3_tilde = A_tilde + A_tilde.T

            # hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            hessian_tilde[:, :, :, :] = -0.5 * np.einsum(
                "kl,rs->klrs", np.eye(self.n_occupied), A3_tilde, optimize="optimal"
            )
            hessian_tilde[:, :, :, :] -= 0.5 * np.einsum(
                "rs,kl->klrs",
                np.eye(self.nmo),
                A3_tilde[: self.n_occupied, : self.n_occupied],
                optimize="optimal",
            )
            hessian_tilde[:, :, : self.n_occupied, :] += 0.5 * np.einsum(
                "rl,ks->klrs",
                np.eye(self.n_occupied),
                A3_tilde[: self.n_occupied, :],
                optimize="optimal",
            )
            hessian_tilde[:, :, :, : self.n_occupied] += 0.5 * np.einsum(
                "ks,rl->klrs",
                np.eye(self.n_occupied),
                A3_tilde[:, : self.n_occupied],
                optimize="optimal",
            )

        else:

            gradient_tilde[:, :] = A[:, :] - A.T[:, :]
            A3_tilde = A + A.T
            hessian_tilde[:, :, :, :] = -0.5 * np.einsum(
                "kl,rs->klrs", np.eye(self.n_occupied), A3_tilde, optimize="optimal"
            )
            hessian_tilde[:, :, :, :] -= 0.5 * np.einsum(
                "rs,kl->klrs",
                np.eye(self.n_occupied),
                A3_tilde[: self.n_occupied, : self.n_occupied],
                optimize="optimal",
            )
            hessian_tilde[:, :, : self.n_occupied, :] += 0.5 * np.einsum(
                "rl,ks->klrs",
                np.eye(self.n_occupied),
                A3_tilde[: self.n_occupied, :],
                optimize="optimal",
            )
            hessian_tilde[:, :, :, : self.n_occupied] += 0.5 * np.einsum(
                "ks,rl->klrs",
                np.eye(self.n_occupied),
                A3_tilde[:, : self.n_occupied],
                optimize="optimal",
            )

    def build_gradient_and_hessian(
        self, U, A, G, gradient_tilde, hessian_tilde, full_space
    ):
        print("omppm")
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
            B[:, : self.n_occupied] = A[:, : self.n_occupied] + np.einsum(
                "klrs,sl->rk", G[:, :, :, :], T[:, : self.n_occupied]
            )
            B2 = np.zeros((self.nmo, self.nmo))
            for r in range(self.nmo):
                for k in range(self.n_occupied):
                    B2[r][k] = A[r][k]
                    for s in range(self.nmo):
                        for l in range(self.n_occupied):
                            B2[r][k] += G[k][l][r][s] * T[s][l]
            print("sdgs", np.allclose(B2, B, rtol=1e-14, atol=1e-14))

            A_tilde = np.zeros((self.nmo, self.nmo))
            A2_tilde = np.zeros((self.nmo, self.nmo))
            # A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
            A_tilde = np.einsum("rs,sk->rk", U.T, B)
            A2_tilde = np.zeros((self.nmo, self.nmo))
            for r in range(self.nmo):
                for k in range(self.n_occupied):
                    for s in range(self.nmo):
                        A2_tilde[r][k] += U[s][r] * B[s][k]
            print("bxvc", np.allclose(A2_tilde, A_tilde, rtol=1e-14, atol=1e-14))

            G_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            G_tilde = np.einsum("pr,klpq,qs->klrs", U, G, U)
            G2_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            for k in range(self.n_occupied):
                for l in range(self.n_occupied):
                    for r in range(self.nmo):
                        for s in range(self.nmo):
                            for p in range(self.nmo):
                                for q in range(self.nmo):
                                    G2_tilde[k][l][r][s] += (
                                        U[p][r] * G[k][l][p][q] * U[q][s]
                                    )
            print("ngfn", np.allclose(G2_tilde, G_tilde, rtol=1e-14, atol=1e-14))

            gradient_tilde[:, :] = (
                A_tilde[:, : self.n_occupied] - A_tilde.T[:, : self.n_occupied]
            )
            gradient2_tilde = np.zeros((self.nmo, self.n_occupied))
            for k in range(self.n_occupied):
                for r in range(self.nmo):
                    gradient2_tilde[r][k] = A_tilde[r][k] - A_tilde[k][r]
            print(
                "ytuu",
                np.allclose(gradient2_tilde, gradient_tilde, rtol=1e-14, atol=1e-14),
            )
            gradient_norm = np.dot(gradient_tilde.flatten(), gradient_tilde.flatten())
            np.set_printoptions(precision=9, suppress=True)
            gradient_norm = np.sqrt(gradient_norm)
            # for k in range(self.n_occupied):
            #    for r in range(self.nmo):
            #        if(gradient_tilde[r][k]>1e-10):
            #            print("efgfb",gradient_tilde[r][k])
            # print(A_tilde)
            print(gradient_norm)
            A3_tilde = A_tilde + A_tilde.T
            # hessian_tilde = np.copy(G_tilde)
            hessian_tilde[:, :, :, :] = G_tilde[:, :, :, :]
            hessian_tilde[:, :, : self.n_occupied, :] -= G_tilde[
                :, :, : self.n_occupied, :
            ].transpose(2, 1, 0, 3)
            hessian_tilde[:, :, :, : self.n_occupied] -= G_tilde[
                :, :, :, : self.n_occupied
            ].transpose(0, 3, 2, 1)
            hessian_tilde[:, :, : self.n_occupied, : self.n_occupied] += G_tilde[
                :, :, : self.n_occupied, : self.n_occupied
            ].transpose(2, 3, 0, 1)
            hessian_tilde -= 0.5 * np.einsum(
                "kl,rs->klrs", np.eye(self.n_occupied), A3_tilde
            )
            hessian_tilde -= 0.5 * np.einsum(
                "rs,kl->klrs",
                np.eye(self.nmo),
                A3_tilde[: self.n_occupied, : self.n_occupied],
            )
            hessian_tilde[:, :, : self.n_occupied, :] += 0.5 * np.einsum(
                "rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[: self.n_occupied, :]
            )
            hessian_tilde[:, :, :, : self.n_occupied] += 0.5 * np.einsum(
                "ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:, : self.n_occupied]
            )
            hessian2_tilde = np.zeros(
                (self.n_occupied, self.n_occupied, self.nmo, self.nmo)
            )
            np.set_printoptions(suppress=True)
            for k in range(self.n_occupied):
                for l in range(self.n_occupied):
                    for r in range(self.nmo):
                        for s in range(self.nmo):
                            hessian2_tilde[k][l][r][s] = G_tilde[k][l][r][s]
                            hessian2_tilde[k][l][r][s] -= (
                                0.5 * (k == l) * (A_tilde[r][s] + A_tilde[s][r])
                            )
                            hessian2_tilde[k][l][r][s] += (
                                0.5 * (r == l) * (A_tilde[k][s] + A_tilde[s][k])
                            )
                            hessian2_tilde[k][l][r][s] += (
                                0.5 * (k == s) * (A_tilde[r][l] + A_tilde[l][r])
                            )
                            hessian2_tilde[k][l][r][s] -= (
                                0.5 * (r == s) * (A_tilde[k][l] + A_tilde[l][k])
                            )
                            if r < self.n_occupied:
                                hessian2_tilde[k][l][r][s] -= G_tilde[r][l][k][s]
                            if s < self.n_occupied:
                                hessian2_tilde[k][l][r][s] -= G_tilde[k][s][r][l]
                            if r < self.n_occupied and s < self.n_occupied:
                                hessian2_tilde[k][l][r][s] += G_tilde[r][s][k][l]

            # print(A_tilde)
            # print(B)
            print(
                "bfad",
                np.allclose(hessian2_tilde, hessian_tilde, rtol=1e-14, atol=1e-14),
            )
            # print(gradient_tilde)
            G3 = hessian_tilde.transpose(0, 2, 1, 3)
            G3 = G3.reshape(self.n_occupied * self.nmo, self.n_occupied * self.nmo)
            self.G7 = G3
            omega, eig_vecs = np.linalg.eigh(G3)
            print("nbjfgbnjtryrt", omega)
        else:

            gradient_tilde[:, :] = A[:, :] - A.T[:, :]

            hessian_tilde[:, :, :, :] = G[:, :, :, :]
            hessian_tilde[:, :, : self.n_occupied, :] -= G[
                :, :, : self.n_occupied, :
            ].transpose(2, 1, 0, 3)
            hessian_tilde[:, :, :, : self.n_occupied] -= G[
                :, :, :, : self.n_occupied
            ].transpose(0, 3, 2, 1)
            hessian_tilde[:, :, : self.n_occupied, : self.n_occupied] += G[
                :, :, : self.n_occupied, : self.n_occupied
            ].transpose(2, 3, 0, 1)
            A3_tilde = A + A.T
            hessian_tilde -= 0.5 * np.einsum(
                "kl,rs->klrs", np.eye(self.n_occupied), A3_tilde
            )
            hessian_tilde -= 0.5 * np.einsum(
                "rs,kl->klrs",
                np.eye(self.n_occupied),
                A3_tilde[: self.n_occupied, : self.n_occupied],
            )
            hessian_tilde[:, :, : self.n_occupied, :] += 0.5 * np.einsum(
                "rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[: self.n_occupied, :]
            )
            hessian_tilde[:, :, :, : self.n_occupied] += 0.5 * np.einsum(
                "ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:, : self.n_occupied]
            )

    def internal_transformation(self, U, h1, d_cmo1, J, K):
        # self.occupied_J2 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        # self.occupied_K2 = np.zeros((self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied))
        # for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for m_p in range(self.n_occupied):
        #            for n_p in range(self.n_occupied):
        #                for k in range(self.n_occupied):
        #                    for l in range(self.n_occupied):
        #                        for m in range(self.n_occupied):
        #                            for n in range(self.n_occupied):
        #                                self.occupied_J2[k_p][l_p][m_p][n_p] += U[k][k_p] * U[l][l_p] * U[m][m_p] * U[n][n_p] * self.J[k][l][m][n]
        #                                self.occupied_K2[k_p][l_p][m_p][n_p] += U[k][k_p] * U[l][l_p] * U[m][m_p] * U[n][n_p] * self.K[k][l][m][n]

        # self.occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
        # self.occupied_d_cmo1 = np.zeros((self.n_occupied, self.n_occupied))
        # for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for k in range(self.n_occupied):
        #            for l in range(self.n_occupied):
        #                self.occupied_h1[k_p][l_p] += U[k][k_p] * U[l][l_p] * self.H_spatial2[k][l]
        #                self.occupied_d_cmo1[k_p][l_p] += U[k][k_p] * U[l][l_p] * self.d_cmo[k][l]

        occupied_twoeint1 = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
        )
        occupied_twoeint2 = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
        )
        # occupied_twoeint1 = np.einsum("klmn,ns->klms", self.J[:,:,:self.n_occupied,:self.n_occupied], U[:self.n_occupied,:self.n_occupied])
        occupied_twoeint1 = np.einsum(
            "klmn,ns->klms",
            self.occupied_J[:, :, : self.n_occupied, : self.n_occupied],
            U[: self.n_occupied, : self.n_occupied],
        )
        occupied_twoeint2 = np.einsum(
            "klms,mr->klrs", occupied_twoeint1, U[: self.n_occupied, : self.n_occupied]
        )
        occupied_twoeint1 = np.einsum(
            "klrs,lq->kqrs", occupied_twoeint2, U[: self.n_occupied, : self.n_occupied]
        )
        occupied_twoeint2 = np.einsum(
            "kqrs,kp->pqrs", occupied_twoeint1, U[: self.n_occupied, : self.n_occupied]
        )
        J[:, :, : self.n_occupied, : self.n_occupied] = occupied_twoeint2[:, :, :, :]
        K[:, :, : self.n_occupied, : self.n_occupied] = occupied_twoeint2[
            :, :, :, :
        ].transpose(1, 3, 0, 2)

        occupied_oneeint1 = np.zeros((self.n_occupied, self.n_occupied))
        occupied_oneeint1 = np.einsum(
            "ij,jl->il",
            self.occupied_h1[: self.n_occupied, : self.n_occupied],
            U[: self.n_occupied, : self.n_occupied],
        )
        h1[: self.n_occupied, : self.n_occupied] = np.einsum(
            "il,ik->kl", occupied_oneeint1, U[: self.n_occupied, : self.n_occupied]
        )
        occupied_oneeint1 = np.einsum(
            "ij,jl->il",
            self.occupied_d_cmo[: self.n_occupied, : self.n_occupied],
            U[: self.n_occupied, : self.n_occupied],
        )
        d_cmo1[: self.n_occupied, : self.n_occupied] = np.einsum(
            "il,ik->kl", occupied_oneeint1, U[: self.n_occupied, : self.n_occupied]
        )
        # print("fdrx", np.allclose(self.occupied_K2,K[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        # print("lknx", np.allclose(self.occupied_J2,J[:,:,:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        # print("vcvx", np.allclose(self.occupied_h1,h1[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))
        # print("trgx", np.allclose(self.occupied_d_cmo1,d_cmo1[:self.n_occupied,:self.n_occupied], rtol=1e-14,atol=1e-14))

    # def internal_update_ci_integrals(self, E_core, occupied_fock_core, occupied_h1, occupied_d_cmo, occupied_J, occupied_K):
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
        # self.occupied_J2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.occupied_K2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.occupied_h1 = np.zeros((self.nmo, self.nmo))
        # self.occupied_d_cmo1 = np.zeros((self.nmo, self.nmo))
        # for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for r_p in range(self.nmo):
        #            for s_p in range(self.nmo):
        #                for k in range(self.n_occupied):
        #                    for l in range(self.n_occupied):
        #                        for r in range(self.nmo):
        #                            for s in range(self.nmo):
        #                                self.occupied_J2[k_p][l_p][r_p][s_p] += U[k][k_p] * U[l][l_p] * U[r][r_p] * U[s][s_p] * self.J[k][l][r][s]
        #                                self.occupied_K2[k_p][l_p][r_p][s_p] += U[k][k_p] * U[l][l_p] * U[r][r_p] * U[s][s_p] * self.K[k][l][r][s]
        # for r_p in range(self.nmo):
        #    for s_p in range(self.nmo):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                self.occupied_h1[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.H_spatial2[r][s]
        #                self.occupied_d_cmo1[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.d_cmo[r][s]

        twoeint1_ab = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_virtual, self.n_virtual)
        )
        # twoeint2_ab = np.zeros((self.n_occupied, self.n_occupied, self.n_virtual, self.n_virtual))

        twoeint1_ab = np.einsum(
            "klab,kp->plab",
            self.J[:, :, self.n_occupied :, self.n_occupied :],
            U[: self.n_occupied, : self.n_occupied],
        )
        twoeint1_ms = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.nmo)
        )
        twoeint2_ms = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.nmo)
        )
        twoeint1_ms = np.einsum(
            "klmn,kp->plmn",
            self.J[:, :, : self.n_occupied, :],
            U[: self.n_occupied, : self.n_occupied],
        )
        twoeint2_ms = np.einsum(
            "plmn,lq->pqmn", twoeint1_ms, U[: self.n_occupied, : self.n_occupied]
        )
        twoeint1_ms = np.einsum(
            "pqmn,mr->pqrn", twoeint2_ms, U[: self.n_occupied, : self.n_occupied]
        )
        # J^kl_ab
        J[:, :, self.n_occupied :, self.n_occupied :] = np.einsum(
            "plab,lq->pqab", twoeint1_ab, U[: self.n_occupied, : self.n_occupied]
        )
        # J^kl_mn
        J[:, :, : self.n_occupied, : self.n_occupied] = np.einsum(
            "pqrn,ns->pqrs",
            twoeint1_ms[:, :, :, : self.n_occupied],
            U[: self.n_occupied, : self.n_occupied],
        )
        # J^kl_ma
        J[:, :, : self.n_occupied, self.n_occupied :] = twoeint1_ms[
            :, :, :, self.n_occupied :
        ]
        # J^kl_am = J^kl_ma
        J[:, :, self.n_occupied :, : self.n_occupied] = twoeint1_ms[
            :, :, :, self.n_occupied :
        ].transpose(0, 1, 3, 2)
        # J = np.copy(J9)
        twoeint1_ab = np.einsum(
            "klab,kp->plab",
            self.K[:, :, self.n_occupied :, self.n_occupied :],
            U[: self.n_occupied, : self.n_occupied],
        )
        twoeint1_ms = np.einsum(
            "klmn,kp->plmn",
            self.K[:, :, : self.n_occupied, :],
            U[: self.n_occupied, : self.n_occupied],
        )
        twoeint2_ms = np.einsum(
            "plmn,lq->pqmn", twoeint1_ms, U[: self.n_occupied, : self.n_occupied]
        )
        twoeint1_ms = np.einsum(
            "pqmn,mr->pqrn", twoeint2_ms, U[: self.n_occupied, : self.n_occupied]
        )
        # K^kl_ab
        K[:, :, self.n_occupied :, self.n_occupied :] = np.einsum(
            "plab,lq->pqab", twoeint1_ab, U[: self.n_occupied, : self.n_occupied]
        )
        # K^kl_mn
        K[:, :, : self.n_occupied, : self.n_occupied] = np.einsum(
            "pqrn,ns->pqrs",
            twoeint1_ms[:, :, :, : self.n_occupied],
            U[: self.n_occupied, : self.n_occupied],
        )
        # K^kl_ma
        K[:, :, : self.n_occupied, self.n_occupied :] = twoeint1_ms[
            :, :, :, self.n_occupied :
        ]
        # K^lk_am = K^kl_ma
        K[:, :, self.n_occupied :, : self.n_occupied] = twoeint1_ms[
            :, :, :, self.n_occupied :
        ].transpose(1, 0, 3, 2)

        oneeint1 = np.zeros((self.nmo, self.nmo))
        oneeint1 = np.einsum("rs,sq->rq", self.H_spatial2, U)
        h1[:, :] = np.einsum("rq,rp->pq", oneeint1, U)
        oneeint1 = np.einsum("rs,sq->rq", self.d_cmo, U)
        d_cmo1[:, :] = np.einsum("rq,rp->pq", oneeint1, U)
        # print("vdwt", np.allclose(self.occupied_K2,K, rtol=1e-14,atol=1e-14))
        # print("ujhn", np.allclose(self.occupied_J2,J, rtol=1e-14,atol=1e-14))
        # print("nhfd", np.allclose(self.occupied_h1,h1, rtol=1e-14,atol=1e-14))
        # print("eerd", np.allclose(self.occupied_d_cmo1,d_cmo1, rtol=1e-14,atol=1e-14))

    def full_transformation_macroiteration(self, U, J, K):

        self.twoeint00 = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
        # self.J3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.K3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.h3 = np.zeros((self.nmo, self.nmo))
        # self.d_cmo3 = np.zeros((self.nmo, self.nmo))
        # for k_p in range(self.n_occupied):
        #    for l_p in range(self.n_occupied):
        #        for r_p in range(self.nmo):
        #            for s_p in range(self.nmo):
        #                for p in range(self.nmo):
        #                    for q in range(self.nmo):
        #                        for r in range(self.nmo):
        #                            for s in range(self.nmo):
        #                                self.J3[k_p][l_p][r_p][s_p] += U[p][k_p] * U[q][l_p] * U[r][r_p] * U[s][s_p] * self.twoeint[p][q][r][s]
        #                                self.K3[k_p][l_p][r_p][s_p] += U[r][r_p] * U[p][k_p] * U[s][s_p] * U[q][l_p] * self.twoeint[r][p][s][q]
        # for r_p in range(self.nmo):
        #    for s_p in range(self.nmo):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                self.h3[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.H1temp[r][s]
        #                self.d_cmo3[r_p][s_p] += U[r][r_p] * U[s][s_p] * self.d_cmo_temp[r][s]

        temp1 = np.zeros((self.n_occupied, self.nmo, self.nmo, self.nmo))
        temp2 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        temp3 = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        temp1 = np.einsum("pqrs,pk->kqrs", self.twoeint00, U[:, : self.n_occupied])
        temp2 = np.einsum("kqrs,ql->klrs", temp1, U[:, : self.n_occupied])
        temp3 = np.einsum("klrs,rm->klms", temp2, U)
        J[:, :, :, :] = np.einsum("klms,sn->klmn", temp3, U)

        temp1 = temp1.transpose(0, 2, 1, 3)
        temp2 = np.einsum("kqrs,ql->klrs", temp1, U[:, : self.n_occupied])
        temp3 = np.einsum("klrs,rm->klms", temp2, U)
        K[:, :, :, :] = np.einsum("klms,sn->klmn", temp3, U)
        # print("obne", np.allclose(self.K3,K, rtol=1e-14,atol=1e-14))
        # print("ztlv", np.allclose(self.J3,J, rtol=1e-14,atol=1e-14))
        # print("tdck", np.allclose(self.h3,self.H_spatial2, rtol=1e-14,atol=1e-14))
        # print("pgxq", np.allclose(self.d_cmo3,self.d_cmo, rtol=1e-14,atol=1e-14))
    def transform_JK_with_df(self, B_mo, U, n_occupied, ignore_dse_terms=False):
        """
        Transform J and K integrals using DF tensors

        Note: Symmetrization is NOT needed - it's automatic from the math!
        """
        naux, nmo_old, _ = B_mo.shape
        nmo = U.shape[1]

        print(f"Transforming J/K: {naux} aux, {nmo_old} -> {nmo} MO, {n_occupied} occupied")

        # Transform B to new basis
        # B_new is automatically symmetric because B_mo is symmetric
        B_temp = np.einsum('Qpq,pi->Qiq', B_mo, U, optimize=True)
        B_new = np.einsum('Qiq,qj->Qij', B_temp, U, optimize=True)

        ## Transform dipole (automatically symmetric if d_cmo is symmetric)
        #if not ignore_dse_terms:
        #    d_new = U.T @ d_cmo @ U
        #else:
        #    d_new = np.zeros((nmo, nmo))

        # Build J - automatically symmetric in k<->l and m<->n
        B_new_oo = B_new[:, :n_occupied, :n_occupied]
        J_new = np.einsum('Qkl,Qmn->klmn', B_new_oo, B_new, optimize=True)

        if not ignore_dse_terms:
            #d_new_oo = d_new[:n_occupied, :n_occupied]
            d_new_oo = self.d_cmo[:n_occupied, :n_occupied]
            J_new += np.einsum('kl,mn->klmn', d_new_oo, self.d_cmo)

        J_new = np.ascontiguousarray(J_new)

        # Build K - automatically has K[k,l,m,n] = K[l,k,n,m] symmetry
        B_new_occ = B_new[:, :n_occupied, :]
        K_new = np.einsum('Qkm,Qln->klmn', B_new_occ, B_new_occ, optimize=True)

        if not ignore_dse_terms:
            #d_new_occ = d_new[:n_occupied, :]
            d_new_occ = self.d_cmo[:n_occupied, :]
            K_new += np.einsum('km,ln->klmn', d_new_occ, d_new_occ)

        K_new = np.ascontiguousarray(K_new)

        return J_new, K_new

    def internal_optimization_exact_energy(
        self, E0, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K, hard_case
    ):
        # for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        for r in range(self.nmo):
        #            for s in range(self.nmo):
        #                  dum = self.J[k][l][r][s] - self.J_temp[k][l][r][s]
        #                  if np.abs(dum) == 0.0:
        #                      print("{:20.12f}".format(dum), k, l, r, s)
        #                  #print("{:20.12f}".format(self.J[k][l][r][s]), "{:20.12f}".format(self.J_temp[k][l][r][s]), k, l, r, s)
        # for r in range(self.nmo):
        #    for s in range(self.nmo):
        #          dum = self.H_spatial2[r][s] - self.H1temp[r][s]
        #          ##if np.abs(dum) == 0.0:
        #          ##    print("{:20.12f}".format(dum), r, s)
        #          print("{:20.12f}".format(self.H_spatial2[r][s]), "{:20.12f}".format(self.H1temp[r][s]), r, s)

        # occupied fock core
        occupied_fock_core = copy.deepcopy(
            occupied_h1[: self.n_occupied, : self.n_occupied]
        )
        occupied_fock_core += 2.0 * np.einsum(
            "jjrs->rs",
            occupied_J[
                : self.n_in_a, : self.n_in_a, : self.n_occupied, : self.n_occupied
            ],
        )
        occupied_fock_core -= np.einsum(
            "jjrs->rs",
            occupied_K[
                : self.n_in_a, : self.n_in_a, : self.n_occupied, : self.n_occupied
            ],
        )
        # occupied_fock_core1 = np.zeros((self.n_occupied, self.n_occupied))
        # for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        occupied_fock_core1[k][l] = self.H_spatial2[k][l]
        #        for j in range(self.n_in_a):
        #            occupied_fock_core1[k][l] += 2.0 * self.J[j][j][k][l] - self.K[j][j][k][l]
        # print("test occupied fock core")
        # for k in range(self.n_occupied):
        #    for l in range(self.n_occupied):
        #        print("{:20.12f}".format(occupied_fock_core[k][l] - occupied_fock_core1[k][l]), k, l)
        E_core = 0.0
        E_core += np.einsum("jj->", occupied_h1[: self.n_in_a, : self.n_in_a])
        E_core += np.einsum("jj->", occupied_fock_core[: self.n_in_a, : self.n_in_a])
        print("vtrhtr", E_core)
        active_one_e_energy = np.dot(
            occupied_fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ].flatten(),
            self.D_tu_avg,
        )
        active_two_e_energy = 0.5 * np.dot(
            occupied_J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ].flatten(),
            self.D_tuvw_avg,
        )
        active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
            occupied_d_cmo[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ].flatten(),
            self.Dpe_tu_avg,
        )
        ci_dependent_energy = self.calculate_ci_dependent_energy(
            eigenvecs, occupied_d_cmo
        )
        sum_energy = (
            active_one_e_energy
            + active_two_e_energy
            + active_one_pe_energy
            + E_core
            + self.Enuc
            + self.d_c
            + ci_dependent_energy
        )
        if sum_energy - E0 <= 0.0 or hard_case == 2:
            self.occupied_J[:, :, : self.n_occupied, : self.n_occupied] = occupied_J[
                :, :, :, :
            ]
            # self.twoeint = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))
            # self.twoeint[:self.n_occupied,:self.n_occupied,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
            # self.twoeint = self.twoeint.reshape((self.nmo * self.nmo, self.nmo * self.nmo))
            self.occupied_K[:, :, : self.n_occupied, : self.n_occupied] = occupied_K[
                :, :, :, :
            ]
            self.occupied_h1[: self.n_occupied, : self.n_occupied] = occupied_h1[:, :]
            self.occupied_d_cmo[: self.n_occupied, : self.n_occupied] = occupied_d_cmo[
                :, :
            ]
            self.occupied_fock_core[: self.n_occupied, : self.n_occupied] = (
                occupied_fock_core[:, :]
            )
            print("hard case internal optimization", hard_case)
            print("fsdb", self.E_core, E_core)
            self.E_core = E_core
            # self.gkl3 = np.zeros((self.n_act_orb, self.n_act_orb))
            # for k in range(self.n_act_orb):
            #    for l in range(self.n_act_orb):
            #        self.gkl3[k][l] = occupied_fock_core[k+self.n_in_a][l+self.n_in_a]
            #        for j in range(self.n_act_orb):
            #            kj = (k + self.n_in_a) * self.nmo + (j + self.n_in_a)
            #            jl = (j + self.n_in_a) * self.nmo + (l + self.n_in_a)
            #            self.gkl3[k][l] -= 0.5 * self.twoeint[kj][jl]
            self.gkl2 = copy.deepcopy(
                occupied_fock_core[
                    self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                ]
            )
            self.gkl2 -= 0.5 * np.einsum(
                "kjjl->kl",
                self.occupied_J[
                    self.n_in_a : self.n_occupied,
                    self.n_in_a : self.n_occupied,
                    self.n_in_a : self.n_occupied,
                    self.n_in_a : self.n_occupied,
                ],
            )

            # print("vffs", np.allclose(self.gkl3,self.gkl2, rtol=1e-14,atol=1e-14))

        # print("test exact energy")
        # print(
        #            "{:20.12f}".format(sum_energy),
        #            "{:20.12f}".format(active_one_e_energy),
        #            "{:20.12f}".format(active_two_e_energy),
        #            "{:20.12f}".format(active_one_pe_energy),
        #            "{:20.12f}".format(self.E_core),
        #            "{:20.12f}".format(self.Enuc),
        #            "{:20.12f}".format(self.d_c),
        #            "{:20.12f}".format(ci_dependent_energy),
        #        )
        return sum_energy - E0

    def internal_optimization_predicted_energy(self, gradient_ai, hessian_ai, Rai):
        print(np.shape(gradient_ai), np.shape(Rai))
        energy = 2.0 * np.einsum("a,a->", gradient_ai, Rai)
        energy += np.einsum("a,ab,b->", Rai, hessian_ai, Rai)
        return energy

    def _dump_ci_vectors(self, eigenvecs, ctx, macroiteration=-1):
        """QED_DUMP_CI_VECTORS -- see the knob comment at the top of this file.

        One .npz per converged CI solve, named by a monotone sequence number so
        the solves stay in execution order (the micro site does not have
        `macroiteration` in scope; the between-macro site does, so macro
        boundaries are recoverable from the interleaving).  No-op when the knob
        is unset, and it writes AFTER the solve, so it cannot perturb one.
        """
        if not _QED_DUMP_CI_VECTORS:
            return
        seq = getattr(self, "_ci_dump_seq", 0)
        self._ci_dump_seq = seq + 1
        os.makedirs(_QED_DUMP_CI_VECTORS, exist_ok=True)
        np.savez(
            os.path.join(_QED_DUMP_CI_VECTORS, f"ci_{seq:05d}_{ctx}.npz"),
            vecs=np.ascontiguousarray(eigenvecs),
            seq=seq,
            ctx=ctx,
            macroiteration=macroiteration,
        )

    def internal_optimization3(self, E0, eigenvecs):
        print("avg_energy", E0)
        current_residual = 1.0


        # test energy
        print("lets test the energy at the beginning of the internal optimization")
        occupied_fock_core = copy.deepcopy(
            self.H_spatial2[: self.n_occupied, : self.n_occupied]
        )
        occupied_fock_core += 2.0 * np.einsum(
            "jjrs->rs",
            self.J[
                : self.n_in_a,
                : self.n_in_a,
                : self.n_occupied,
                : self.n_occupied,
            ],
        )
        occupied_fock_core -= np.einsum(
            "jjrs->rs",
            self.K[
                : self.n_in_a,
                : self.n_in_a,
                : self.n_occupied,
                : self.n_occupied,
            ],
        )

        E_core = 0.0
        E_core += np.einsum(
            "jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a]
        )
        E_core += np.einsum(
            "jj->", occupied_fock_core[: self.n_in_a, : self.n_in_a]
        )
        active_one_e_energy = np.dot(
            occupied_fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ].flatten(),
            self.D_tu_avg,
        )
        active_two_e_energy = 0.5 * np.dot(
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ].flatten(),
            self.D_tuvw_avg,
        )
        active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
            self.d_cmo[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ].flatten(),
            self.Dpe_tu_avg,
        )
        ci_dependent_energy = self.calculate_ci_dependent_energy(
            eigenvecs, self.d_cmo
        )
        sum_energy = (
            active_one_e_energy
            + active_two_e_energy
            + active_one_pe_energy
            + E_core
            + self.Enuc
            + self.d_c
            + ci_dependent_energy
        )
        #self.avg_energy = sum_energy //just comment out on 04/11/25
        # print("heyhey1",eigenvecs)
        print(
            "imgf00000",
            sum_energy,
            E_core,
            active_one_e_energy,
            active_two_e_energy,
            active_one_pe_energy,
            ci_dependent_energy,
            self.Enuc,
            self.d_c,
        )







        self.U = np.eye(self.nmo)
        self.U1 = np.eye(self.nmo)
        #current_energy = E0
        current_energy = sum_energy
        trust_radius = _QED_TR_INIT  # QED_TR_INIT; default 0.5 = original literal
        trust_radius_hard_case = trust_radius
        rot_dim = self.n_occupied
        microiteration = 0
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1

        A1 = np.zeros((rot_dim, rot_dim))
        G1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        self.build_intermediates_internal(
            eigenvecs,
            A1,
            G1,
            self.occupied_fock_core,
            self.occupied_d_cmo,
            self.occupied_J,
            self.occupied_K,
        )
        if _CPP_CASSCF_VALIDATION_DIR:
            _internal_intermediates_dump = dict(
                occupied_fock_core=self.occupied_fock_core,
                occupied_d_cmo=self.occupied_d_cmo,
                occupied_J=self.occupied_J,
                occupied_K=self.occupied_K,
                D_tu_avg=self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                D_tuvw_avg=self.D_tuvw_avg.reshape(
                    (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
                ),
                Dpe_tu_avg=self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                off_diagonal_constant=self.calculate_off_diagonal_photon_constant(eigenvecs),
                omega=self.omega,
                dims=np.array([self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]),
                A1=A1,
                G1=G1,
            )

        occupied_J = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
        )
        occupied_K = np.zeros(
            (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
        )
        occupied_h1 = np.zeros((self.n_occupied, self.n_occupied))
        occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
        occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))

        gradient_tilde1 = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde1 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde_ai = np.zeros((self.n_act_orb, self.n_in_a))
        hessian_tilde_ai = np.zeros(
            (self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a)
        )

        self.build_gradient_and_hessian(
            self.U, A1, G1, gradient_tilde1, hessian_tilde1, False
        )
        if _CPP_CASSCF_VALIDATION_DIR:
            _dump_cpp_casscf_validation_case(
                "internal_intermediates",
                U=self.U,
                gradient_tilde1=gradient_tilde1,
                hessian_tilde1=hessian_tilde1,
                **_internal_intermediates_dump,
            )
        gradient_tilde_ai[:, :] = gradient_tilde1[
            self.n_in_a : self.n_occupied, : self.n_in_a
        ]
        hessian_tilde_ai[:, :, :, :] = hessian_tilde1.transpose(2, 0, 3, 1)[
            self.n_in_a : self.n_occupied,
            : self.n_in_a,
            self.n_in_a : self.n_occupied,
            : self.n_in_a,
        ]
        gradient_tilde_ai = gradient_tilde_ai.reshape(self.n_act_orb * self.n_in_a)
        hessian_tilde_ai = hessian_tilde_ai.reshape(
            self.n_act_orb * self.n_in_a, self.n_act_orb * self.n_in_a
        )
        # np.savetxt("internal_gradient.out", gradient_tilde_ai)
        # np.savetxt("internal_hessian.out", hessian_tilde_ai)
        # exit()

        restart = False
        count = 0
        hard_case = 0
        reduce_step = 0
        first_component = np.zeros(self.n_act_orb * self.n_in_a)
        second_component = np.zeros(self.n_act_orb * self.n_in_a)
        reduce_predicted_energy = 0
        # critical_step_norm = 0

        while True:
            print("trust radius", trust_radius)
            print("\n", current_energy)
            print("Internal optimization iteration", microiteration + 1)

            # print(hessian_tilde_ai)
            mu1, w1 = np.linalg.eigh(hessian_tilde_ai)
            # print("eigenvalue of active-inactive hessian", mu1)
            #print("eigenvalue of the reduced hessian", mu1)
            print(
                "dot product of gradient and first eigenvector of hessian",
                np.dot(gradient_tilde_ai, w1[:, 0]),
            )
            print(
                "dot product of gradient and second eigenvector of hessian",
                np.dot(gradient_tilde_ai, w1[:, 1]),
            )
            H_lambda = hessian_tilde_ai - mu1[0] * np.eye(self.n_act_orb * self.n_in_a)
            step_limit = -np.einsum(
                "pq,q->p", np.linalg.pinv(H_lambda), gradient_tilde_ai
            )
            print("norm of critical step", np.linalg.norm(step_limit))
            print("alpha critical", mu1[0] - np.dot(gradient_tilde_ai, step_limit))

            # print("dfbg", gradient_tilde1)
            # print(np.shape(gradient_tilde_ai))
            gradient_norm = np.dot(gradient_tilde_ai, gradient_tilde_ai.T)
            gradient_norm = np.sqrt(gradient_norm)
            print("gradient norm", gradient_norm)

            # new augmented hessian
            beta = 0
            dim00 = self.n_act_orb * self.n_in_a + 1
            # augmented_hessian9= np.zeros((dim00, dim00))
            w9 = np.zeros((dim00, dim00))
            step = np.zeros(self.n_act_orb * self.n_in_a)
            step10_norm = 0.0
            hessian_diagonal = np.diagonal(hessian_tilde_ai)
            if np.linalg.norm(gradient_tilde_ai) > 1e-4:
                if (
                    hard_case == 1
                    and reduce_step == 1
                    and np.linalg.norm(first_component) < trust_radius
                    and reduce_predicted_energy == 1
                ):
                    print("hard case and current step is smaller than trust radius")
                    print(np.linalg.norm(first_component))
                    xy_square = np.dot(first_component, second_component) * np.dot(
                        first_component, second_component
                    )
                    x_square = np.dot(first_component, first_component)
                    y_square = np.dot(second_component, second_component)
                    delta = 4 * xy_square - 4 * y_square * (
                        x_square - trust_radius * trust_radius
                    )
                    # print(delta)
                    t1 = (
                        -2 * np.dot(first_component, second_component) - np.sqrt(delta)
                    ) / (2 * y_square)
                    t2 = (
                        -2 * np.dot(first_component, second_component) + np.sqrt(delta)
                    ) / (2 * y_square)
                    print(
                        "x^2, xy, y^2, t",
                        x_square,
                        np.dot(first_component, second_component),
                        y_square,
                        t1,
                    )
                    adjusted_step = first_component + min(np.abs(t1), np.abs(t2)) * second_component
                    print("adjusted step norm", np.linalg.norm(adjusted_step))
                    step = adjusted_step
                else:
                    # QED_INTERNAL_EXACT_TRS: the dense Hessian is already in hand here,
                    # so the subproblem can be solved exactly instead of bisected.
                    if _QED_INTERNAL_EXACT_TRS:
                        step, hard_case = exact_internal_trs(
                            hessian_tilde_ai, gradient_tilde_ai, trust_radius)
                    else:
                        step, hard_case = solve_lstrs_internal(
                            self, dim00, gradient_tilde_ai, hessian_tilde_ai,
                            hessian_diagonal, trust_radius, w9, first_component,
                            second_component, hard_case, beta, step)

            elif np.linalg.norm(gradient_tilde_ai) <= 1e-4 and np.linalg.norm(gradient_tilde_ai) > 1e-8:
                print("gradient is small, use Newton step")
                x, exitCode = minres(hessian_tilde_ai, -gradient_tilde_ai, rtol=1e-5)
                print("exitcode", exitCode)
                hard_case = 2
                step = x
                # print(step)
            # w[:,0] = w[:,0]/scale
            else:
                step = np.zeros((self.n_act_orb * self.n_in_a))
            _dump_cpp_casscf_validation_case(
                "internal_lstrs",
                hessian=hessian_tilde_ai,
                gradient=gradient_tilde_ai,
                trust_radius=trust_radius,
                hard_case=hard_case,
                step=step,
            )
            Rai = step.reshape(self.n_act_orb, self.n_in_a)
            Rvi = np.zeros((self.n_virtual, self.n_in_a))
            Rva = np.zeros((self.n_virtual, self.n_act_orb))
            self.build_unitary_matrix(Rai, Rvi, Rva)

            start = timer()
            self.internal_transformation(
                self.U_delta, occupied_h1, occupied_d_cmo, occupied_J, occupied_K
            )
            end = timer()
            print("internal transformation took", end - start)
            # print("jnti",self.E_core)
            exact_energy = self.internal_optimization_exact_energy(
                current_energy,
                eigenvecs,
                occupied_h1,
                occupied_d_cmo,
                occupied_J,
                occupied_K,
                hard_case
            )
            predicted_energy = self.internal_optimization_predicted_energy(
                gradient_tilde_ai, hessian_tilde_ai, step
            )
            if predicted_energy <= 0:
                reduce_predicted_energy = 1
            else:
                reduce_predicted_energy = 0

            print(
                "internal exact energy",
                exact_energy,
                "internal predicted energy",
                predicted_energy,
                flush=True,
            )

            if exact_energy <= 0.0 or hard_case == 2:
                self.U1 = np.einsum("pq,qs->ps", self.U1, self.U_delta)
                self.occupied_J3 = self.occupied_J.reshape(
                    self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
                )
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
                    self.target_spin,
                )
                # self.J[:,:,:self.n_occupied, :self.n_occupied] = occupied_J[:,:,:,:]
                # self.K[:,:,:self.n_occupied, :self.n_occupied] = occupied_K[:,:,:,:]
                # self.H_spatial2[:self.n_occupied, :self.n_occupied] = occupied_h1[:,:]
                # self.d_cmo[:self.n_occupied, :self.n_occupied] = occupied_d_cmo[:,:]
                current_energy = current_energy + exact_energy
                print("wezn", current_energy, flush=True)
                d_diag = 2.0 * np.einsum(
                    "ii->", self.occupied_d_cmo[: self.n_in_a, : self.n_in_a]
                )

                # self.constdouble = np.zeros(6)
                # self.constdouble[0] = self.Enuc
                # if self.ignore_dse_terms:
                #    self.constdouble[1] = 0.0
                # else:
                #    self.constdouble[1] = self.d_c
                # self.constdouble[2] = self.omega
                self.constdouble[3] = self.d_exp - d_diag
                # QED_CI_INTERNAL_FACTOR -- see the knob comment at the top of
                # this file.  gradient_tilde_ai holds the CURRENT internal-
                # internal orbital gradient here: it is built before the loop
                # (:9800-9825) and rebuilt at the end of every iteration
                # (:10596-10611), so at this point it is the gradient the CI
                # solve about to run is being asked to keep up with.
                # Default 0 => the fixed 1e-5 this site has always used.
                self.constdouble[4] = 1e-5
                if _QED_CI_INTERNAL_FACTOR > 0.0:
                    _scaled = _QED_CI_INTERNAL_FACTOR * np.linalg.norm(gradient_tilde_ai)
                    self.constdouble[4] = min(
                        max(_scaled, _QED_CI_INTERNAL_FLOOR), _QED_CI_INTERNAL_CEIL
                    )
                    print(
                        f"[ci-internal] ||g_ii|| {np.linalg.norm(gradient_tilde_ai):.6e} "
                        f"-> CI threshold {self.constdouble[4]:.6e}",
                        flush=True,
                    )
                self.constdouble[5] = self.E_core
                # Same uncoupled cap knob as microiteration_optimization6's
                # (default 5) -- internal_optimization3 is the other live site
                # running the Kreplin uncoupled-CI strategy.
                self.constint[8] = _QED_CI_UNCOUPLED_MAXITER
                eigenvals = np.zeros((self.davidson_roots))
                # eigenvecs = np.zeros((self.davidson_roots, H_dim))
                # eigenvecs[:,:] = 0.0
                #self.occupied_J3 = self.occupied_J.reshape(
                #    self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
                #)
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
                print("CI convergence check", self.constint[8])
                avg_energy = 0.0
                for i in range(self.davidson_roots):
                    avg_energy += self.weight[i] * eigenvals[i]
                self.avg_energy = avg_energy
                #predicted_energy = self.internal_optimization_predicted_energy(
                #    gradient_tilde_ai, hessian_tilde_ai, step
                #)
                print(
                    "internal optimization iteration",
                    microiteration + 1,
                    exact_energy,
                    predicted_energy,
                    current_energy,
                    avg_energy,
                    #exact_energy / predicted_energy,
                    flush=True,
                )
                self.build_state_average_rdms(eigenvecs)
                if predicted_energy != 0:
                    ratio = exact_energy / predicted_energy
                    print("current ratio", ratio)
                    trust_radius = self.step_control(ratio, trust_radius)

                #update current energy using RDM energy
                E_core = 0.0
                E_core += np.einsum("jj->", self.occupied_h1[: self.n_in_a, : self.n_in_a])
                E_core += np.einsum("jj->", self.occupied_fock_core[: self.n_in_a, : self.n_in_a])
                active_one_e_energy = np.dot(
                    self.occupied_fock_core[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ].flatten(),
                    self.D_tu_avg,
                )
                active_two_e_energy = 0.5 * np.dot(
                    self.occupied_J[
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                    ].flatten(),
                    self.D_tuvw_avg,
                )
                active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                    self.occupied_d_cmo[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ].flatten(),
                    self.Dpe_tu_avg,
                )
                ci_dependent_energy = self.calculate_ci_dependent_energy(
                    eigenvecs, self.occupied_d_cmo
                )
                sum_energy = (
                    active_one_e_energy
                    + active_two_e_energy
                    + active_one_pe_energy
                    + E_core
                    + self.Enuc
                    + self.d_c
                    + ci_dependent_energy
                )
                print(
                    "new RDM energy",
                    sum_energy,
                    E_core,
                    active_one_e_energy,
                    active_two_e_energy,
                    active_one_pe_energy,
                    ci_dependent_energy,
                    self.Enuc,
                    self.d_c,
                )

                self.build_intermediates_internal(
                    eigenvecs,
                    A1,
                    G1,
                    self.occupied_fock_core,
                    self.occupied_d_cmo,
                    self.occupied_J,
                    self.occupied_K,
                )

                self.build_gradient_and_hessian(
                    self.U, A1, G1, gradient_tilde1, hessian_tilde1, False
                )
                gradient_tilde_ai = gradient_tilde_ai.reshape(
                    self.n_act_orb, self.n_in_a
                )
                hessian_tilde_ai = hessian_tilde_ai.reshape(
                    self.n_act_orb, self.n_in_a, self.n_act_orb, self.n_in_a
                )
                gradient_tilde_ai[:, :] = gradient_tilde1[
                    self.n_in_a : self.n_occupied, : self.n_in_a
                ]
                hessian_tilde_ai[:, :, :, :] = hessian_tilde1.transpose(2, 0, 3, 1)[
                    self.n_in_a : self.n_occupied,
                    : self.n_in_a,
                    self.n_in_a : self.n_occupied,
                    : self.n_in_a,
                ]
                gradient_tilde_ai = gradient_tilde_ai.reshape(
                    self.n_act_orb * self.n_in_a
                )
                hessian_tilde_ai = hessian_tilde_ai.reshape(
                    self.n_act_orb * self.n_in_a, self.n_act_orb * self.n_in_a
                )
                hard_case = 0

            else:
                reduce_step = 1
                trust_radius = 0.5 * trust_radius
                # if hard_case == 0:
                #    trust_radius = 0.7 * trust_radius
                # if hard_case == 1:
                #    trust_radius = 0.7 * trust_radius_hard_case
                print("Reject step, restart")
            gradient_norm = np.linalg.norm(gradient_tilde_ai)
            print(gradient_norm, current_residual)
            # QED_INTERNAL_MAXITER: the hard cap, default 20 = the value this
            # site has always used.  Molpro's equivalent cap is 4 with an early
            # exit (blocks run 4,4,4,4,4,4,4,2,1) -- see the knob comment at the
            # top of this file.
            if (
                #gradient_norm < 1e-4 and current_residual < 1e-5
                gradient_norm < 1e-4 and self.constint[8] == 0
            ) or microiteration == _QED_INTERNAL_MAXITER:
                print("internal rotation converged!")
                # print("qims", np.allclose(self.J, self.J_temp, rtol=1e-14,atol=1e-14))
                # print("ynss", np.allclose(self.K, self.K_temp, rtol=1e-14,atol=1e-14))
                # print("ibsp", np.allclose(self.H_spatial2, self.H1temp, rtol=1e-14,atol=1e-14))
                start = timer()

                c_full_transformation_internal_optimization(
                    self.U1,
                    self.J,
                    self.K,
                    self.H_spatial2,
                    self.d_cmo,
                    self.J,
                    self.K,
                    self.H_spatial2,
                    self.d_cmo,
                    self.index_map_ab,
                    self.index_map_kl,
                    self.nmo,
                    self.n_occupied,
                )
                # self.full_transformation_internal_optimization(self.U1, self.H_spatial2, self.d_cmo, self.J, self.K)
                end = timer()
                print("full internal transformation took", end - start)
                self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U1)
                # test energy
                occupied_fock_core = copy.deepcopy(
                    self.H_spatial2[: self.n_occupied, : self.n_occupied]
                )
                occupied_fock_core += 2.0 * np.einsum(
                    "jjrs->rs",
                    self.J[
                        : self.n_in_a,
                        : self.n_in_a,
                        : self.n_occupied,
                        : self.n_occupied,
                    ],
                )
                occupied_fock_core -= np.einsum(
                    "jjrs->rs",
                    self.K[
                        : self.n_in_a,
                        : self.n_in_a,
                        : self.n_occupied,
                        : self.n_occupied,
                    ],
                )

                E_core = 0.0
                E_core += np.einsum(
                    "jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a]
                )
                E_core += np.einsum(
                    "jj->", occupied_fock_core[: self.n_in_a, : self.n_in_a]
                )
                active_one_e_energy = np.dot(
                    occupied_fock_core[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ].flatten(),
                    self.D_tu_avg,
                )
                active_two_e_energy = 0.5 * np.dot(
                    self.J[
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                        self.n_in_a : self.n_occupied,
                    ].flatten(),
                    self.D_tuvw_avg,
                )
                active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
                    self.d_cmo[
                        self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                    ].flatten(),
                    self.Dpe_tu_avg,
                )
                ci_dependent_energy = self.calculate_ci_dependent_energy(
                    eigenvecs, self.d_cmo
                )
                sum_energy = (
                    active_one_e_energy
                    + active_two_e_energy
                    + active_one_pe_energy
                    + E_core
                    + self.Enuc
                    + self.d_c
                    + ci_dependent_energy
                )
                #self.avg_energy = sum_energy
                # print("heyhey1",eigenvecs)
                print(
                    "imgf",
                    sum_energy,
                    E_core,
                    active_one_e_energy,
                    active_two_e_energy,
                    active_one_pe_energy,
                    ci_dependent_energy,
                    self.Enuc,
                    self.d_c,
                )
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

        augmented_hessian[0, 1:] = gradient_tilde_ai
        augmented_hessian[1:, 0] = gradient_tilde_ai.T
        augmented_hessian[1:, 1:] = hessian_tilde_ai
        # print("hessian")
        # self.printA(hessian_tilde_ai)
        # print("augmented_hessian")
        # self.printA(augmented_hessian)

        # print(augmented_hessian)
        mu, w = np.linalg.eigh(augmented_hessian)
        # print("w",w)
        # print("first eigenvalues of the augmented hessian", mu[0], flush=True)
        scale = w[0][0]
        print(scale, flush=True)
        w[:, 0] = w[:, 0] / scale
        # print(w)
        step0 = w[1:, 0]
        # print(step0)
        step[:] = step0 / alpha
        # h_inverse = np.linalg.inv(hessian_tilde_ai)
        # step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai)
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        # print("step norm", step_norm, flush=True)
        return step_norm

    def internal_step2(
        self, gradient_tilde_ai, hessian_tilde_ai, step, mu, alpha, dim0
    ):
        gradient_tilde_ai = alpha * gradient_tilde_ai
        augmented_hessian = np.zeros((dim0, dim0))

        augmented_hessian[0, 1:] = gradient_tilde_ai
        augmented_hessian[1:, 0] = gradient_tilde_ai.T
        augmented_hessian[1:, 1:] = hessian_tilde_ai
        # print(augmented_hessian)
        mu[:], w = np.linalg.eigh(augmented_hessian)
        # print("w",w)
        # print("eigenvalues of the augmented hessian", mu, flush=True)
        scale = w[0][0]
        print(scale, flush=True)
        # w[:,0] = w[:,0]/scale
        # print(w)
        step0 = w[1:, 0]
        # print(step0)
        step[:] = step0 / alpha
        # h_inverse = np.linalg.inv(hessian_tilde_ai)
        # step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai)
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        # print("step norm", step_norm, flush=True)
        return (
            step_norm,
            scale,
        )

    def step_control(self, ratio, trust_radius, step_norm=None):
        # QED_EXIT_CONTROL: consume the pending GLTR exit (None if this step did
        # not come from a GLTR solve, in which case nothing below fires).  The
        # token is taken unconditionally so it can never go stale, but it is
        # only READ when the knob is on -- with the knob off this function is
        # byte-for-byte the original.
        _exit_hist = self._take_gltr_exit() if _QED_EXIT_CONTROL else None
        if _QED_EXIT_CONTROL in (1, 3) and _exit_hist is not None:
            # Mode 1: persistently boundary-terminating => the step is being
            # placed by the constraint rather than by the model.  Shrink until
            # the model's own minimizer is interior.  Deliberately checked
            # BEFORE the ratio branches: rho is blind to this (item 182), so on
            # MgH+ ratio > 0.75 at essentially every step and a rule gated on
            # the ratio would never fire.
            _damped = _exit_control_radius(trust_radius, _exit_hist)
            if _damped != trust_radius:
                if _QED_SOLVER_STATS:
                    print(
                        "  exit_control: boundary-dominated, radius"
                        f" {trust_radius:.4f} -> {_damped:.4f}"
                    )
                return _damped
        if ratio >= 0 and ratio <= 0.25:
            trust_radius = 0.7 * trust_radius
        if ratio > 0.75:
            # QED_STEP_CONTROL (a): only widen the region if the step actually
            # pressed against it.  An interior step tells us nothing about
            # whether a larger Delta would still be trustworthy.
            if _QED_STEP_CONTROL and step_norm is not None and step_norm < 0.8 * trust_radius:
                return trust_radius
            # QED_EXIT_CONTROL mode 2: the same idea keyed on the exit reason
            # itself rather than on the step-norm proxy above.
            if _QED_EXIT_CONTROL in (2, 3) and _exit_hist is not None:
                if _exit_last_is_interior(_exit_hist):
                    return trust_radius
            trust_radius = 1.2 * trust_radius
            # QED_TR_MAX: the cap.  This is the single chokepoint -- all four
            # step_control call sites (~9868, ~12821, ~15114, ~16559) route
            # through here, so the cap needs exactly this one edit.  Default
            # 0.75 is the original literal.
            if trust_radius > _QED_TR_MAX:
                trust_radius = _QED_TR_MAX
        return trust_radius

    def build_state_average_rdms(self, eigenvecs):
        self.D_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.Dpe_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.D_tuvw_avg = np.zeros(
            (self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb)
        )
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            c_build_active_rdm_sa(
                eigenvecs,
                self.D_tu_avg,
                self.D_tuvw_avg,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                self.weight[i],
            )
            c_build_active_photon_electron_one_rdm(
                eigenvecs,
                self.Dpe_tu_avg,
                self.table,
                self.n_act_a,
                self.n_act_orb,
                np1,
                i,
                i,
                self.weight[i],
            )

        ###symmetrize 2-rdm
        for t in range(self.n_act_orb):
            for u in range(t, self.n_act_orb):
                tu = t * self.n_act_orb + u
                ut = u * self.n_act_orb + t
                for vw in range(self.n_act_orb * self.n_act_orb):
                    dum = (
                        self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw]
                        + self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw]
                    )
                    # dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    self.D_tuvw_avg[tu * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )
                    self.D_tuvw_avg[ut * self.n_act_orb * self.n_act_orb + vw] = (
                        dum / 2.0
                    )

                    # dum = (self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw] +
                    # self.D_tuvw_avg2[ut * self.n_act_orb * self.n_act_orb + vw])
                    ##dum2 = self.D_tu_avg[tu] + self.D_tu_avg[tu]
                    # self.D_tuvw_avg2[tu * self.n_act_orb * self.n_act_orb + vw] = dum/2.0
                    # self.D_tuvw_avg2[ut * self.n_act_orb * self.n_act_orb + vw] = dum/2.0

                    # self.D_tu_avg[tu] = dum2/2.0
                    # self.D_tu_avg[ut] = dum2/2.0

    # ---------------------------------------------------------------
    # build_sigma_reduced / _2 / _3 / _4 and orbital_sigma -- earlier
    # generations of the orbital Hessian-vector product -- MOVED to
    # legacy_python_ci.py on 2026-09-04.  build_sigma_reduced7 (jitted,
    # block-G) reached through orbital_sigma3 is the live one.
    # ---------------------------------------------------------------
    def tiled_G_blocks(self, G):
        """QED_G_TILED: the (G_ij, G_ti, G_tu) blocks, built by a tiled copy.

        Drop-in replacement for the G.transpose(3,1,2,0) slice-and-reshape at
        ~12070, producing BIT-IDENTICAL arrays with the same index convention
        (rows (s,l), columns (r,k)) -- see the QED_G_TILED note at the top of the
        file for why the numpy version only reaches 4.2 GB/s and this reaches
        29.6.  Buffers are allocated on first use and reused thereafter.
        """
        ni = self.n_in_a
        na = self.n_act_orb
        nmo = self.nmo
        # (l-range start, k-range start, nl, nk) for each block, matching the
        # three quadrants the numpy path slices out.
        spec = (
            ("ij", 0, 0, ni, ni),
            ("ti", ni, 0, na, ni),
            ("tu", ni, ni, na, na),
        )
        bufs = getattr(self, "_g_tiled_bufs", None)
        if bufs is None or bufs["dims"] != (nmo, ni, na):
            bufs = {"dims": (nmo, ni, na)}
            for name, _l0, _k0, nl, nk in spec:
                bufs[name] = np.zeros((nmo, nl, nmo, nk))
            self._g_tiled_bufs = bufs
        out = []
        for name, l0, k0, nl, nk in spec:
            buf = bufs[name]
            _transpose_G_block(G, buf, l0, k0, 32)
            # free view: buf is C-contiguous, so this is a reshape, not a copy
            out.append(buf.reshape(nmo * nl, nmo * nk))
        return tuple(out)

    def _actvirt_shift_vector(self):
        """Occupation-scaled level shift, one entry per reduced rotation.

        Returns None when QED_ACTVIRT_SHIFT is off or the active RDM is not
        built yet (the very first sigma builds run before any CI solve), so
        every caller can treat None as "no shift" and skip the work entirely.

        sigma_(t,a) = eps / (n_t + delta) on act-virt, 0 elsewhere.  See the
        knob block at the top of this file for the justification and for the
        measurement that motivated it.
        """
        if not _QED_ACTVIRT_SHIFT:
            return None
        D_tu = getattr(self, "D_tu_avg", None)
        if D_tu is None:
            return None

        # index_map is fixed after __init__, so the block mask and the
        # active-orbital index of each parameter are built once and cached.
        # index_map[:, 0] is the LARGER MO index, [:, 1] the smaller
        # (:3468-3481), so act-virt is "smaller is active, larger is virtual".
        if getattr(self, "_actvirt_shift_mask", None) is None:
            s = self.index_map[:, 0]
            r = self.index_map[:, 1]
            self._actvirt_shift_mask = (
                (r >= self.n_in_a) & (r < self.n_occupied) & (s >= self.n_occupied)
            )
            self._actvirt_shift_act = r[self._actvirt_shift_mask] - self.n_in_a

        occ = np.diag(np.asarray(D_tu).reshape(self.n_act_orb, self.n_act_orb))
        shift = np.zeros(self.index_map_size)
        shift[self._actvirt_shift_mask] = _QED_ACTVIRT_SHIFT_EPS / (
            occ[self._actvirt_shift_act] + _QED_ACTVIRT_SHIFT_DELTA
        )
        return shift

    def orbital_sigma3(self, U, A_tilde, G_blocks, R_reduced, sigma_reduced, num_states, pointer):
        """The one chokepoint every solver's Hessian-vector product goes through.  MOVED to orbital_sigma.py (2026-09-04).

        Kept as a method because the name is this class's interface: the
        solver modules call it as ctx.orbital_sigma3(...).  See
        orbital_sigma.py for the kernel and for why that module is a leaf.
        """
        return _os_orbital_sigma3(
            self,
            U,
            A_tilde,
            G_blocks,
            R_reduced,
            sigma_reduced,
            num_states,
            pointer,
        )


    # ---------------------------------------------------------------
    # build_sigma_reduced / _2 / _3 / _4 and orbital_sigma -- earlier
    # generations of the orbital Hessian-vector product -- MOVED to
    # legacy_python_ci.py on 2026-09-04.  build_sigma_reduced7 (jitted,
    # block-G) reached through orbital_sigma3 is the live one.
    # ---------------------------------------------------------------
    # build_sigma_reduced7 MOVED to orbital_sigma.py (2026-09-04) -- module-private
    # there: its only caller is in that module.


    def microiteration_ci_integrals_transform(
        self, U, eigenvecs, d_cmo, active_fock_core, active_twoeint
    ):
        # print("test U2")
        # print(U)

        # print("E_core before second order transformation", self.E_core)
        T = U - np.eye(self.nmo)
        E_core = self.E_core
        E_core += 4.0 * np.einsum(
            "ir,ri->", self.fock_core[: self.n_in_a, :], T[:, : self.n_in_a]
        )
        # Was:
        #   temp1  = zeros((ni,ni,nmo,nmo))                  <- dead, rebound below
        #   temp1  = einsum("rs,ij->ijrs", fock_core, eye(ni))
        #   temp1 += L[:ni]
        #   temp2  = einsum("ijrs,sj->ri", temp1, T[:, :ni])
        # temp1 was a full (ni,ni,nmo,nmo) array -- 218 MB at c10h8 dims, of
        # which only the ni diagonal (i==j) pairs are nonzero, so 96.6% of it was
        # zeros -- materialized purely to be contracted away on the next line.
        # Same pattern as the G_ij one in build_intermediates, and this routine
        # ("second order integral transformation") is 38.2 s / 159 calls, 13% of
        # the c10h8 optimization.
        # Split the contraction instead:
        #   temp2[r,i] = sum_{j,s} (fock_core[r,s] delta_ij + L[i,j,r,s]) T[s,j]
        #              = sum_s fock_core[r,s] T[s,i]  +  sum_{j,s} L[i,j,r,s] T[s,j]
        # Algebraically exact.  Splitting one accumulation into two does reorder
        # the floating-point sum, so results move at the ~1e-16 relative level;
        # the L contraction below is left in exactly its original einsum form
        # (no `optimize=`) so that half is unchanged bit-for-bit.
        temp2 = self.fock_core @ T[:, : self.n_in_a]
        temp2 += np.einsum(
            "ijrs,sj->ri", self.L[: self.n_in_a, :, :, :], T[:, : self.n_in_a]
        )
        E_core += 2.0 * np.einsum("ri,ri->", T[:, : self.n_in_a], temp2)
        

        self.E_core2 = E_core

        active_fock_core[:, :] = 0.0
        # active_fock_core[:,:] = copy.deepcopy(self.active_fock_core)
        temp3 = np.zeros((self.nmo, self.n_act_orb))
        temp3 = np.einsum(
            "rs,su->ru", self.fock_core, U[:, self.n_in_a : self.n_occupied],
            optimize=True,
        )
        active_fock_core[:, :] = np.einsum(
            "rt,ru->tu", U[:, self.n_in_a : self.n_occupied], temp3
        )

        temp4 = np.zeros((self.n_act_orb, self.n_act_orb, self.nmo, self.n_in_a))
        temp4 = np.einsum(
            "turs,si->turi",
            self.J[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied, :, :],
            U[:, : self.n_in_a],
            optimize=True,
        )
        active_fock_core[:, :] += 2.0 * np.einsum(
            "ri,turi->tu", U[:, : self.n_in_a], temp4
        )
        active_fock_core[:, :] -= 2.0 * np.einsum(
            "tuii->tu",
            self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                : self.n_in_a,
                : self.n_in_a,
            ],
        )
        temp4 = np.einsum(
            "turs,si->turi",
            self.K[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied, :, :],
            U[:, : self.n_in_a],
            optimize=True,
        )
        active_fock_core[:, :] -= np.einsum("ri,turi->tu", U[:, : self.n_in_a], temp4)
        active_fock_core[:, :] += np.einsum(
            "tuii->tu",
            self.K[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                : self.n_in_a,
                : self.n_in_a,
            ],
        )

        temp3 = np.einsum(
            "uirs,si->ru",
            self.L[self.n_in_a : self.n_occupied, :, :, :],
            T[:, : self.n_in_a],
        )
        temp5 = np.zeros((self.n_act_orb, self.n_act_orb))
        temp5 = np.einsum("rt,ru->tu", T[:, self.n_in_a : self.n_occupied], temp3)
        active_fock_core[:, :] += temp5 + temp5.T

        active_twoeint[:, :, :, :] = -1.0 * self.active_twoeint
        temp6 = np.zeros((self.n_act_orb, self.n_act_orb, self.nmo, self.n_act_orb))
        temp7 = np.zeros(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        temp6 = np.einsum(
            "vwrs,su->vwru",
            self.J[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied, :, :],
            U[:, self.n_in_a : self.n_occupied],
            optimize=True,
        )
        temp7 = np.einsum("vwru,rt->vwtu", temp6, U[:, self.n_in_a : self.n_occupied], optimize=True)
        # print(np.shape(active_twoeint),np.shape(temp7) )

        active_twoeint[:, :, :, :] += temp7[:, :, :, :]
        active_twoeint[:, :, :, :] += temp7[:, :, :, :].transpose(2, 3, 0, 1)
        temp6 = np.einsum(
            "tvrs,sw->tvrw",
            self.K[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied, :, :],
            T[:, self.n_in_a : self.n_occupied],
            optimize=True,
        )
        temp7 = np.einsum("tvrw,ru->tvuw", temp6, T[:, self.n_in_a : self.n_occupied], optimize=True)
        # active_twoeint[:,:,:,:] = temp7[:,:,:,:]
        # active_twoeint[:,:,:,:] += temp7.transpose(2,1,0,3)
        # active_twoeint[:,:,:,:] += temp7.transpose(0,3,2,1)
        # active_twoeint[:,:,:,:] += temp7.transpose(2,3,0,1)
        temp7 = np.einsum("tvrw,ru->tuvw", temp6, T[:, self.n_in_a : self.n_occupied], optimize=True)
        active_twoeint[:, :, :, :] += temp7[:, :, :, :]
        active_twoeint[:, :, :, :] += temp7.transpose(1, 0, 2, 3)
        active_twoeint[:, :, :, :] += temp7.transpose(0, 1, 3, 2)
        active_twoeint[:, :, :, :] += temp7.transpose(1, 0, 3, 2)

        temp8 = np.einsum("pq,qs->ps", self.d_cmo, U, optimize=True)
        d_cmo[:, :] = np.einsum("ps,pr->rs", temp8, U, optimize=True)

       

    def true_energy_mode(self):
        """Effective QED_TRUE_ENERGY_RATIO at this moment.

        Without QED_TRUE_ENERGY_REACTIVE this is just the knob.  With it, the
        machinery stays OFF -- so the model keeps the large radius of
        convergence that finds the better basin (item 331D) -- until a
        macroiteration energy actually rises, after which it is armed for the
        rest of the run.
        """
        if not _QED_TRUE_ENERGY_REACTIVE:
            return _QED_TRUE_ENERGY_RATIO
        return 2 if getattr(self, "_true_energy_armed", False) else 0

    def true_energy_should_engage(self, U2_trial):
        """Should the true-energy machinery run at this trial rotation?

        QED_TRUE_ENERGY_ADAPTIVE only.  Returns True unconditionally when the
        knob is off, so the adaptive path is an exact no-op by default and the
        callers stay one-branch.

        The test is ||U2_trial - I||_F > QED_TRUE_ENERGY_EPS_C: how far the
        trial rotation has walked from the point the model was expanded at.
        Deliberately computable WITHOUT the E(U) transform -- that is the whole
        cost saving, and it is why the trigger is the rotation rather than
        Kreplin's |D^k - D^(k-1)| (which would need the rotated density).
        See the knob note at the top of this file for item 330's measurement.
        """
        if not _QED_TRUE_ENERGY_ADAPTIVE:
            return True
        # NORMALISED BY sqrt(index_map_size).  ||U - I||_F sums over every
        # rotation parameter, so for a typical per-parameter rotation theta it
        # grows like theta*sqrt(N) -- an EXTENSIVE quantity.  N differs 28-fold
        # across the test set (h4 208, c2h4 939, c10h8 5789), so the RAW norm is
        # not comparable between systems and a single threshold on it is
        # meaningless.  Measured medians: raw 0.0089 / 0.0078 / 0.0492 (c10h8
        # looks 6.3x "worse" purely because sqrt(5789/208) = 5.3); normalised
        # 6.18e-4 / 2.54e-4 / 6.47e-4.  The developer raised the system-size
        # point; an earlier draft here concluded from the RAW numbers that the
        # trigger was anti-correlated with need, and that conclusion is
        # WITHDRAWN -- it was measuring dimension, not conditioning.
        disp = np.linalg.norm(
            U2_trial - np.eye(U2_trial.shape[0])
        ) / np.sqrt(max(self.index_map_size, 1))
        engaged = disp > _QED_TRUE_ENERGY_EPS_C
        # Printed on EVERY adaptive call so the threshold can be CHOSEN from the
        # measured distribution instead of assumed.  Run with
        # QED_TRUE_ENERGY_EPS_C=0 to get the full distribution at behaviour
        # identical to the unconditional knob (disp > 0 is always true).
        print(
            "adaptive trigger ||U2_trial-I||_F/sqrt(N)", disp,
            "N", self.index_map_size,
            "eps_c", _QED_TRUE_ENERGY_EPS_C,
            "engaged", engaged,
            flush=True,
        )
        return engaged

    def true_energy_at_trial(self, U2_trial, eigenvecs):
        """EXACT energy E(U) at trial orbitals U2_trial, at FIXED RDMs.

        QED_TRUE_ENERGY_RATIO only -- nothing calls this when the knob is off.

        This is Molpro's ENERGY(VAR), and it is what
        microiteration_exact_energy's E(T) = 2A.T + T.G.T (ENERGY(PROJ)) is an
        approximation to.

        *** WHY THIS DOES THE FULL TRANSFORMATION, WHICH IS THE EXPENSIVE ONE. ***
        The obvious cheap route -- reuse microiteration_ci_integrals_transform,
        which already runs once per accepted orbital step and is 12-46x cheaper
        -- DOES NOT WORK, and the reason is worth recording because it is not
        visible from that routine's name.  It builds the updated active
        integrals from self.J[act,act,:,:] and self.fock_core with terms
        explicitly first and second order in T = U - I (`E_core += 4.0 *
        einsum(...T)`, `active_twoeint = -self.active_twoeint + temp7 +
        temp7.T + O(T^2)`).  It is a SECOND-ORDER INTEGRAL UPDATE, not a
        transformation.  An energy assembled from its output therefore
        reproduces E(T) rather than checking it: measured on the c2h4 case,
        `true energy change` matched `model energy change` to 12 digits at
        every step while the macroiteration energy still rose 8.3 Eh.  That is
        a consistency check on two second-order expressions, not a PROJ-VAR gap.

        The exact energy needs the ERIs genuinely retransformed at the rotated
        orbitals, i.e. c_full_transformation_macroiteration out of self.twoeint,
        exactly as ah_orbital_optimization did at its accept site (that routine
        was removed 2026-09-04; see git history) and
        as energy_function does for the BFGS path.  Measured cost per call:
        0.050 s (c2h4), 0.140 s (Cr2), 2.334 s (c10h8).

        BASIS BOOKKEEPING.  self.twoeint / self.H1 / self.d_cmo1 are in the
        ORIGINAL MO basis and are never rebound; self.U_total maps that basis to
        the current macroiteration's orbitals (:4953, :4972, :5629); self.U2 is
        the rotation accumulated inside this call to microiteration_
        optimization6.  So the total rotation to a trial point is
        self.U_total @ U2_trial -- the same product the macro loop forms after
        this routine returns.
        """
        U_tot_trial = np.einsum("pq,qs->ps", self.U_total, U2_trial)

        # (n_occupied, n_occupied, nmo, nmo) each -- 9 MB at c2h4 dims, 394 MB
        # at c10h8's.  Cached on self so a per-step call does not re-malloc
        # them; freed with the object.
        if getattr(self, "_true_energy_J", None) is None:
            self._true_energy_J = np.zeros(
                (self.n_occupied, self.n_occupied, self.nmo, self.nmo)
            )
            self._true_energy_K = np.ascontiguousarray(
                np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            )
        J_t = self._true_energy_J
        K_t = self._true_energy_K
        J_t[:, :, :, :] = 0.0
        K_t[:, :, :, :] = 0.0
        self._macro_JK(U_tot_trial, J_t, K_t)

        temp8 = np.einsum("pq,qs->ps", self.H1, U_tot_trial)
        h1_t = np.einsum("ps,pr->rs", temp8, U_tot_trial)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_tot_trial)
        d_cmo_t = np.einsum("ps,pr->rs", temp8, U_tot_trial)

        # QED_TRUE_ENERGY_REEXPAND keeps these so an ACCEPTED step can install
        # the transform instead of discarding it (J_t/K_t are already the cached
        # buffers; only these three are new, and they are nmo^2 each).
        if _QED_TRUE_ENERGY_REEXPAND:
            self._true_energy_h1 = h1_t
            self._true_energy_d = d_cmo_t
            self._true_energy_Utot = U_tot_trial

        return self.rdm_exact_energy(J_t, K_t, h1_t, d_cmo_t, eigenvecs)

    def rebuild_G_blocks(self, G):
        """The G_blocks tuple, rebuilt exactly as the microiteration top builds it.

        QED_TRUE_ENERGY_REEXPAND only.  Factored out because re-expansion has to
        redo it after build_intermediates writes a new G; the flattening
        convention is the one build_sigma_reduced7 / microiteration_exact_energy
        assume (rows (s,l), columns (r,k), flattened r*n_block + k).
        """
        if _QED_G_TILED:
            return self.tiled_G_blocks(G)
        G_t = G.transpose(3, 1, 2, 0)
        return (
            G_t[:, : self.n_in_a, :, : self.n_in_a].reshape(
                self.nmo * self.n_in_a, self.nmo * self.n_in_a
            ),
            G_t[:, self.n_in_a :, :, : self.n_in_a].reshape(
                self.nmo * self.n_act_orb, self.nmo * self.n_in_a
            ),
            G_t[:, self.n_in_a :, :, self.n_in_a :].reshape(
                self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
            ),
        )

    def reexpand_at_new_orbitals(self, A, G, eigenvecs):
        """Move the model's expansion point to the current trial orbitals.

        QED_TRUE_ENERGY_REEXPAND only.  Installs the J/K/h1/d_cmo that
        true_energy_at_trial has just produced, folds the rotation into
        U_total, resets U2/U3 to the identity, and rebuilds A and G from the NEW
        integrals -- so the next build_gradient(self.U2, ...) is evaluated at
        T = 0 and returns the gradient AT the new point, exactly as
        ah_orbital_optimization did, before it was removed 2026-09-04 (see the
        knob note at the top of this file).

        The caller's post-return bookkeeping stays correct: the macro loop folds
        self.U2 into self.U_total and rotates H_spatial2/d_cmo by it, and with
        U2 = I both are no-ops on an already-correct U_total.

        Returns the rebuilt G_blocks.
        """
        self.J[:, :, :, :] = self._true_energy_J
        self.K[:, :, :, :] = self._true_energy_K
        self.H_spatial2[:, :] = self._true_energy_h1
        self.d_cmo[:, :] = self._true_energy_d
        self.U_total = self._true_energy_Utot.copy()
        self.U2 = np.eye(self.nmo)
        self.U3 = np.eye(self.nmo)
        A[:, :] = 0.0
        G[:, :, :, :] = 0.0
        self.build_intermediates(eigenvecs, A, G, True)
        return self.rebuild_G_blocks(G)

    def microiteration_exact_energy(self, U, A, G_blocks):
        # print("test U")
        # print(U)

        # Blocked form of the former dense product over the full
        # (nmo*n_occupied) square G1.  G1 is no longer built: its four quadrants
        # are G_ij / G_ti / G_ti.T / G_tu, flattened with (r,k) -> r*n_block + k,
        # the same convention build_sigma_reduced7 assumes (helper_PFCI.py:8760).
        # Equivalent to the previous
        #     B = 2.0 * A_rk + np.dot(T_rk, G1);  E = np.dot(T_rk, B)
        # since G1 is symmetric and the blocks tile it exactly.
        G_ij, G_ti, G_tu = G_blocks
        T = U - np.eye(self.nmo)
        T_i = np.ascontiguousarray(T[:, : self.n_in_a]).reshape(
            self.nmo * self.n_in_a
        )
        T_a = np.ascontiguousarray(
            T[:, self.n_in_a : self.n_occupied]
        ).reshape(self.nmo * self.n_act_orb)
        A_i = np.ascontiguousarray(A[:, : self.n_in_a]).reshape(
            self.nmo * self.n_in_a
        )
        A_a = np.ascontiguousarray(
            A[:, self.n_in_a : self.n_occupied]
        ).reshape(self.nmo * self.n_act_orb)
        # B = np.zeros((self.nmo, self.n_occupied))
        # B = 2.0 * A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G, T[:,:self.n_occupied])
        B_i = 2.0 * A_i + np.dot(T_i, G_ij) + np.dot(T_a, G_ti)
        B_a = 2.0 * A_a + np.dot(T_i, G_ti.T) + np.dot(T_a, G_tu)
        # E = np.einsum("rk, rk->",T[:,:self.n_occupied], B)
        E = np.dot(T_i, B_i) + np.dot(T_a, B_a)
        # print("ccvcvc", B)
        # c= np.einsum("rk, rk->",T[:,:self.n_occupied], B)
        # c = np.dot(T_rk, B)   # dead recomputation of E; T_rk/B no longer built
        # d=np.einsum("rk, rk->",T[:,:self.n_in_a], B[:,:self.n_in_a])
        # E1 = E0 + np.einsum("rk, rk->",T[:,:self.n_in_a], B[:,:self.n_in_a])
        # print("uuunn", c)
        # B3 = np.zeros((self.nmo, self.n_in_a))
        # B3 = 2.0 * self.fock_core[:,:self.n_in_a]
        # temp = np.zeros((self.nmo, self.nmo, self.n_in_a, self.n_in_a))
        # temp = np.einsum("rs, ij->ijrs",self.fock_core, np.eye(self.n_in_a))
        # temp += self.L[:self.n_in_a,:,:,:]
        # B3 += np.einsum("ijrs,sj->ri", temp, T[:,:self.n_in_a])
        # E10 = 2.0 * np.einsum("ri,ri->", T[:,:self.n_in_a], B3)
        # print("rmmngghg", E0, E0+E10)
        return E

    def rdm_exact_energy(self, J, K, h1, d_cmo, eigenvecs):

        active_twoeint = J[
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
            self.n_in_a : self.n_occupied,
        ]
        fock_core = copy.deepcopy(h1)
        fock_core += 2.0 * np.einsum("jjrs->rs", J[: self.n_in_a, : self.n_in_a, :, :])
        fock_core -= np.einsum("jjrs->rs", K[: self.n_in_a, : self.n_in_a, :, :])

        E_core = 0.0
        E_core += np.einsum("jj->", h1[: self.n_in_a, : self.n_in_a])
        E_core += np.einsum("jj->", fock_core[: self.n_in_a, : self.n_in_a])

        # print(eigenvecs)
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        active_fock_core[:, :] = fock_core[
            self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
        ]
        active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
        active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
        active_one_pe_energy = -np.sqrt(self.omega / 2) * np.dot(
            d_cmo[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ].flatten(),
            self.Dpe_tu_avg,
        )
        ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
        sum_energy = (
            active_one_e_energy
            + active_two_e_energy
            + active_one_pe_energy
            + E_core
            + self.Enuc
            + self.d_c
            + ci_dependent_energy
        )
        # print("sum energy",
        #    "{:20.12f}".format(sum_energy),
        #    "{:20.12f}".format(active_one_e_energy),
        #    "{:20.12f}".format(active_two_e_energy),
        #    "{:20.12f}".format(E_core),
        #    "{:20.12f}".format(active_one_pe_energy),
        #    "{:20.12f}".format(self.Enuc),
        # )
        return sum_energy

    def energy_function(self, step):
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual, self.n_in_a))
        Rva = np.zeros((self.n_virtual, self.n_act_orb))
        for i in range(self.index_map_size):
            s = self.index_map[i][0]
            l = self.index_map[i][1]
            if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                Rai[s - self.n_in_a][l] = step[i]
            elif s >= self.n_occupied and l < self.n_in_a:
                Rvi[s - self.n_occupied][l] = step[i]
            else:
                Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

        self.build_unitary_matrix(Rai, Rvi, Rva)
        U_temp = self.U_delta
        J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        K_temp = np.ascontiguousarray(K_temp)
        self._macro_JK(U_temp, J_temp, K_temp)
        # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

        temp8 = np.zeros((self.nmo, self.nmo))
        temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
        h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
        d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

        sum_energy = self.rdm_exact_energy(
            J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs
        )

        return sum_energy

    def energy_function2(self, R):
        R1 = np.dot(-R, R)
        tau_square, W = np.linalg.eigh(R1)
        sine_product_array = np.zeros(self.nmo)
        cosine_array = np.zeros(self.nmo)
        for i in range(self.nmo):
            if tau_square[i] < 0.0:
                tau_square[i] = 0.0
            tau = np.sqrt(tau_square[i])
            cosine_array[i] = np.cos(tau)
            if np.sqrt(tau_square[i]) > 1e-15:
                sine_product_array[i] = np.sin(tau) / tau
            else:
                # print("dfgsg")
                sine_product_array[i] = 1 - pow(tau, 2) / 6 + pow(tau, 4) / 120
        # self.U_delta = np.einsum("pq,qs,st->pt", W, np.diag(cosine_array), W.T)
        # temp = np.einsum("pq,qs->ps", W.T, R)
        # self.U_delta += np.einsum("pq,qs,st->pt", W, np.diag(sine_product_array), temp)
        temp = np.dot(np.diag(cosine_array), W.T)
        self.U_delta = np.dot(W, temp)
        temp = np.dot(W.T, R)
        R1 = np.dot(np.diag(sine_product_array), temp)
        self.U_delta += np.dot(W, R1)

        U_temp = self.U_delta
        J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        K_temp = np.ascontiguousarray(K_temp)
        self._macro_JK(U_temp, J_temp, K_temp)
        # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

        temp8 = np.zeros((self.nmo, self.nmo))
        temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
        h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
        d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

        sum_energy = self.rdm_exact_energy(
            J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs
        )

        return sum_energy

    def numerical_orbital_gradient(self, step):
        h = 1e-5
        A = np.zeros((self.nmo, self.nmo))
        gradient = np.zeros(self.index_map_size)
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual, self.n_in_a))
        Rva = np.zeros((self.n_virtual, self.n_act_orb))
        for i in range(self.index_map_size):
            s = self.index_map[i][0]
            l = self.index_map[i][1]
            if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                Rai[s - self.n_in_a][l] = step[i]
            elif s >= self.n_occupied and l < self.n_in_a:
                Rvi[s - self.n_occupied][l] = step[i]
            else:
                Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

        R = np.zeros((self.nmo, self.nmo))

        R[self.n_in_a : self.n_occupied, : self.n_in_a] = Rai
        R[: self.n_in_a, self.n_in_a : self.n_occupied] = -Rai.T
        R[self.n_occupied :, : self.n_in_a] = Rvi
        R[: self.n_in_a, self.n_occupied :] = -Rvi.T
        R[self.n_occupied :, self.n_in_a : self.n_occupied] = Rva
        R[self.n_in_a : self.n_occupied, self.n_occupied :] = -Rva.T
        temp_en = self.energy_function2(R)
        print("temp_en", temp_en)
        print(R)
        R1 = np.dot(-R, R)
        tau_square, W = np.linalg.eigh(R1)
        sine_product_array = np.zeros(self.nmo)
        cosine_array = np.zeros(self.nmo)
        for i in range(self.nmo):
            if tau_square[i] < 0.0:
                tau_square[i] = 0.0
            tau = np.sqrt(tau_square[i])
            cosine_array[i] = np.cos(tau)
            if np.sqrt(tau_square[i]) > 1e-15:
                sine_product_array[i] = np.sin(tau) / tau
            else:
                # print("dfgsg")
                sine_product_array[i] = 1 - pow(tau, 2) / 6 + pow(tau, 4) / 120
        # self.U_delta = np.einsum("pq,qs,st->pt", W, np.diag(cosine_array), W.T)
        # temp = np.einsum("pq,qs->ps", W.T, R)
        # self.U_delta += np.einsum("pq,qs,st->pt", W, np.diag(sine_product_array), temp)
        temp = np.dot(np.diag(cosine_array), W.T)
        self.U_delta = np.dot(W, temp)
        temp = np.dot(W.T, R)
        R1 = np.dot(np.diag(sine_product_array), temp)
        self.U_delta += np.dot(W, R1)
        self.U_ttt = copy.deepcopy(self.U_delta)

        for i in range(self.nmo * self.nmo):
            q = i % self.nmo
            p = i // self.nmo
            save = R[p][q]
            R[p][q] = save + h
            p1 = self.energy_function2(R)
            R[p][q] = save + 2 * h
            p2 = self.energy_function2(R)
            R[p][q] = save - h
            m1 = self.energy_function2(R)
            R[p][q] = save - 2 * h
            m2 = self.energy_function2(R)
            R[p][q] = save
            A[p][q] = (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / 12.0 / h

        # for i in range(self.index_map_size):
        #    s = self.index_map[i][0]
        #    l = self.index_map[i][1]
        #    save = step[i]
        #    step[i] = save + h
        #    p1 = self.energy_function(step)
        #    step[i] = save + 2.0 * h
        #    p2 = self.energy_function(step)
        #    step[i] = save - h
        #    m1 = self.energy_function(step)
        #    step[i] = save - 2.0 * h
        #    m2 = self.energy_function(step)
        #    step[i] = save
        #    A[s][l] = (-p2 + 8.0 * p1 - 8.0 * m1 + m2)/12.0/h
        #    A[l][s] = -(-p2 + 8.0 * p1 - 8.0 * m1 + m2)/12.0/h
        # print(A)

        self.A1111 = copy.deepcopy(A)
        rot_dim = self.nmo
        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        gradient_tilde[:, :] = 0.5 * (
            A[:, : self.n_occupied] - A.T[:, : self.n_occupied]
        )
        # gradient_tilde[:,:] = 0.5 * A[:,:self.n_occupied]
        reduced_gradient = np.zeros(self.index_map_size)
        index_count1 = 0
        for k in range(self.n_occupied):
            for r in range(k + 1, self.nmo):
                if k < self.n_in_a and r < self.n_in_a:
                    continue
                if (
                    self.n_in_a <= k < self.n_occupied
                    and self.n_in_a <= r < self.n_occupied
                ):
                    continue
                # QED_ORBITAL_SYMMETRY: skip rotations that mix symmetry species.
                if (
                    _QED_ORBITAL_SYMMETRY
                    and self._sym_labels is not None
                    and self._sym_labels[r] != self._sym_labels[k]
                ):
                    continue
                reduced_gradient[index_count1] = gradient_tilde[r][k]
                # print(r,k,index_count1)
                # index_count2 = 0
                # for l in range(self.n_occupied):
                #    for s in range(l+1,self.nmo):
                #        if (l < self.n_in_a and s < self.n_in_a): continue
                #        if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                #        #if (k >= self.n_occupied and r >= self.n_occupied): continue
                #        reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                #        #print(r,k,s,l,index_count1,index_count2)
                #        index_count2 += 1
                index_count1 += 1
        return reduced_gradient

    def energy_grad_finite_difference_element(self, step):
        """
        A function to compute the elements of the orbital gradient using centered finite differences
        """
        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual, self.n_in_a))
        Rva = np.zeros((self.n_virtual, self.n_act_orb))

        # initialize numerical gradient array
        rot_dim = self.nmo
        A_num = np.zeros((rot_dim, rot_dim))

        # define step size for orbitals - is this reasonable???
        _h = 0.001

        # perform outter-loop to select gradient element to compute
        for j in range(self.index_map_size):
            # these are the indices of the current gradient element
            gradS = self.index_map[j][0]
            gradL = self.index_map[j][1]

            # perform inner-loop to build R with forward displacement
            for i in range(self.index_map_size):
                s = self.index_map[i][0]
                l = self.index_map[i][1]

                # if this is the element to displace, do forward displacement
                if i == j:
                    step_val = step[i] + _h
                # otherwise, don't displace
                else:
                    step_val = step[i]

                # build appropriate active-inactive, virtual-inactive, or virtual-active
                # block as normal except for one element has a displacement
                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s - self.n_in_a][l] = step_val
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s - self.n_occupied][l] = step_val
                else:
                    Rva[s - self.n_occupied][l - self.n_in_a] = step_val

            # build unitary matrix with one element displaced - the resulting unitary matrix will be antisymmetrized
            # so the transpose element will also have a displacement
            self.build_unitary_matrix(Rai, Rvi, Rva)

            # store the displaced unitary matrix
            U_temp = np.copy(self.U_delta)
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)

            # transform 2ei, J, and K with displaced unitary matrix
            self._macro_JK(U_temp, J_temp, K_temp)

            # initialize temporary arrays and perform transformations on 1e arrays
            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

            # compute energy from forward displaced
            forward_energy = self.rdm_exact_energy(
                J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs
            )

            # perform inner-loop to build R with backward displacement
            for i in range(self.index_map_size):
                s = self.index_map[i][0]
                l = self.index_map[i][1]

                # if this is the element to displace, do backwared displacement
                if i == j:
                    step_val = step[i] - _h
                # otherwise, don't displace
                else:
                    step_val = step[i]

                # build appropriate active-inactive, virtual-inactive, or virtual-active
                # block as normal except for one element has a displacement
                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s - self.n_in_a][l] = step_val
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s - self.n_occupied][l] = step_val
                else:
                    Rva[s - self.n_occupied][l - self.n_in_a] = step_val

            # build unitary matrix with one element displaced - the resulting unitary matrix will be antisymmetrized
            self.build_unitary_matrix(Rai, Rvi, Rva)
            U_temp = np.copy(self.U_delta)
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)

            # transform with displaced
            self._macro_JK(U_temp, J_temp, K_temp)
            # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

            backward_energy = self.rdm_exact_energy(
                J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs
            )

            # assign gradient element A_sl
            A_num[gradS, gradL] = (forward_energy - backward_energy) / (4 * _h)

            # assign transpose element -A_ls
            A_num[gradL, gradS] = (backward_energy - forward_energy) / (4 * _h)

        return A_num

    def energy_grad(self, step):

        Rai = np.zeros((self.n_act_orb, self.n_in_a))
        Rvi = np.zeros((self.n_virtual, self.n_in_a))
        Rva = np.zeros((self.n_virtual, self.n_act_orb))
        for i in range(self.index_map_size):
            s = self.index_map[i][0]
            l = self.index_map[i][1]
            if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                Rai[s - self.n_in_a][l] = step[i]
            elif s >= self.n_occupied and l < self.n_in_a:
                Rvi[s - self.n_occupied][l] = step[i]
            else:
                Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

        self.build_unitary_matrix(Rai, Rvi, Rva)
        # R1 = np.zeros((self.nmo, self.nmo))

        U_temp = self.U_delta
        self.U_temp2 = copy.deepcopy(self.U_delta)
        J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        K_temp = np.ascontiguousarray(K_temp)
        self._macro_JK(U_temp, J_temp, K_temp)
        # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

        temp8 = np.zeros((self.nmo, self.nmo))
        temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
        h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
        temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
        d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

        sum_energy = self.rdm_exact_energy(
            J_temp, K_temp, h1_temp, d_cmo_temp, self.eigenvecs
        )
        #print("ssssum", sum_energy)

        rot_dim = self.nmo
        A = np.zeros((rot_dim, rot_dim))
        fock_core = copy.deepcopy(h1_temp)
        fock_core += 2.0 * np.einsum(
            "jjrs->rs", J_temp[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        fock_core -= np.einsum(
            "jjrs->rs", K_temp[: self.n_in_a, : self.n_in_a, :, :], optimize="optimal"
        )
        D_tu_avg = self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb))
        Dpe_tu_avg = self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb))
        D_tuvw_avg = self.D_tuvw_avg.reshape(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        E_core = 0.0
        E_core += np.einsum("jj->", h1_temp[: self.n_in_a, : self.n_in_a])
        E_core += np.einsum("jj->", fock_core[: self.n_in_a, : self.n_in_a])

        active_fock_core = copy.deepcopy(
            fock_core[self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied]
        )
        active_twoeint = copy.deepcopy(
            J_temp[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ]
        )
        L = np.zeros((self.n_occupied, self.n_in_a, rot_dim, rot_dim))
        fock_general = np.zeros((rot_dim, rot_dim))
        # print (np.shape(self.L))
        # start = timer()
        L = (
            4.0 * K_temp[:, : self.n_in_a, :rot_dim, :rot_dim]
            - K_temp.transpose(0, 1, 3, 2)[:, : self.n_in_a, :rot_dim, :rot_dim]
            - J_temp[:, : self.n_in_a, :rot_dim, :rot_dim]
        )
        # end   = timer()
        # print("build intermediate step 2", end - start)
        ###self.L2 = np.zeros((self.n_occupied, self.n_in_a, self.nmo, self.nmo))
        ###for k in range(self.n_occupied):
        ###    for j in range(self.n_in_a):
        ###        for r in range(self.nmo):
        ###            for s in range(self.nmo):
        ###                self.L2[k][j][r][s] = (4.0 * self.K[k][j][r][s] -
        ###                    self.K[k][j][s][r] - self.J[k][j][r][s])
        ###print((self.L==self.L2).all())

        # self.fock_general += self.fock_core[:rot_dim,:rot_dim] + np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)),
        #        self.J[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        # self.fock_general -= 0.5 * np.einsum("tu,turs->rs", self.D_tu_avg.reshape((self.n_act_orb,self.n_act_orb)),
        #        self.K[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied,:rot_dim,:rot_dim], optimize = "optimal")
        temp1 = (
            J_temp[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
            - 0.5
            * K_temp[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                :rot_dim,
            ]
        )
        # start = timer()
        fock_general += fock_core[:rot_dim, :rot_dim] + np.einsum(
            "tu,turs->rs", D_tu_avg, temp1, optimize="optimal"
        )
        # end   = timer()
        # print("build intermediate step 3", end - start)
        ###self.fock_general2 = np.zeros((self.nmo, self.nmo))
        ###for r in range(self.nmo):
        ###    for s in range(self.nmo):
        ###        self.fock_general2[r][s] = self.fock_core[r][s]
        ###        for t in range(self.n_act_orb):
        ###            for u in range(self.n_act_orb):
        ###                self.fock_general2[r][s] += self.D_tu_avg[t*self.n_act_orb +u] * self.J[t+self.n_in_a][u+self.n_in_a][r][s]
        ###                self.fock_general2[r][s] -= 0.5 * self.D_tu_avg[t*self.n_act_orb +u] * self.K[t+self.n_in_a][u+self.n_in_a][r][s]
        ###print("rqq", np.allclose(self.fock_general,self.fock_general2, rtol=1e-14,atol=1e-14))
        # start = timer()
        off_diagonal_constant = self.calculate_off_diagonal_photon_constant(
            self.eigenvecs
        )
        A[:, : self.n_in_a] = 2.0 * (
            fock_general[:, : self.n_in_a]
            + d_cmo_temp[:rot_dim, : self.n_in_a] * off_diagonal_constant
        )

        # off_diagonal_constant2 = 0.0
        # np1 = self.N_p + 1
        # for i in range(self.davidson_roots):
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
        # print("rtty", off_diagonal_constant2, off_diagonal_constant)

        A[:, self.n_in_a : self.n_occupied] = np.einsum(
            "rt,tu->ru",
            fock_core[:rot_dim, self.n_in_a : self.n_occupied],
            D_tu_avg,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 4", end - start)
        # print(np.shape(self.active_twoeint))
        # start = timer()
        A[:, self.n_in_a : self.n_occupied] += np.einsum(
            "vwrt,tuvw->ru",
            J_temp[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                :rot_dim,
                self.n_in_a : self.n_occupied,
            ],
            D_tuvw_avg,
            optimize="optimal",
        )
        # end   = timer()
        # print("build intermediate step 5", end - start)
        # start = timer()
        A[:, self.n_in_a : self.n_occupied] += -np.sqrt(self.omega / 2) * np.einsum(
            "rt,tu->ru",
            d_cmo_temp[:rot_dim, self.n_in_a : self.n_occupied],
            Dpe_tu_avg,
            optimize="optimal",
        )
        self.A2222 = copy.deepcopy(A)

        # B = np.zeros((self.nmo, self.nmo))
        # T = U - np.eye(self.nmo)
        # B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
        # A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
        ##A_tilde[:,:] = np.einsum("rs,sk->rk",U.T, B)
        # gradient_tilde[:,:] = A_tilde[:,:self.n_occupied] - A_tilde.T[:,:self.n_occupied]

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        gradient_tilde[:, :] = A[:, : self.n_occupied] - A.T[:, : self.n_occupied]
        reduced_gradient = np.zeros(self.index_map_size)
        index_count1 = 0
        for k in range(self.n_occupied):
            for r in range(k + 1, self.nmo):
                if k < self.n_in_a and r < self.n_in_a:
                    continue
                if (
                    self.n_in_a <= k < self.n_occupied
                    and self.n_in_a <= r < self.n_occupied
                ):
                    continue
                # QED_ORBITAL_SYMMETRY: skip rotations that mix symmetry species.
                if (
                    _QED_ORBITAL_SYMMETRY
                    and self._sym_labels is not None
                    and self._sym_labels[r] != self._sym_labels[k]
                ):
                    continue
                reduced_gradient[index_count1] = gradient_tilde[r][k]
                # print(r,k,index_count1)
                # index_count2 = 0
                # for l in range(self.n_occupied):
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

    def microiteration_predicted_energy2(
        self, U, reduced_gradient, A_tilde, G_blocks, step
    ):
        # energy = 2.0 * np.einsum("a,a->", reduced_gradient, step)
        energy = 2.0 * np.dot(reduced_gradient, step)

        step2 = step.reshape(1, self.index_map_size)
        sigma_reduced = np.zeros((1, self.index_map_size))
        # self.build_sigma_reduced2(U, A_tilde, G, step2, sigma_reduced, 1, 0)
        # orbital_sigma3/build_sigma_reduced7 is the blocked equivalent of the
        # former orbital_sigma/build_sigma_reduced4: the two kernels are
        # identical apart from the single np.dot(R1, G1) becoming the four
        # blocked dots at helper_PFCI.py:8765-8768.
        self.orbital_sigma3(U, A_tilde, G_blocks, step2, sigma_reduced, 1, 0)

        # energy += np.einsum("ip,p->", sigma_reduced,step)
        sigma_reduced = sigma_reduced.reshape((self.index_map_size))
        energy += np.dot(sigma_reduced, step)
        return energy

    # microiteration_optimization5 REMOVED 2026-09-04 (developer's call).
    # It was the TEMPLATE for the quasi-Newton-off branch of
    # microiteration_optimization6, which has since absorbed it; it had no
    # callers and could not be revived as it stood.  1364 lines.  It also
    # carried its own copy of the LSTRS beta-bisection and of the
    # both-roots-unnormalizable `exit()` (item 372), so removing it deletes
    # one of the four duplicates rather than leaving it to be found again.
    # See git history for the original.

    def get_lbfgs_step(self, gradient, diag_hessian, trust_radius):
        """
        Computes L-BFGS direction. 
        Uses reduced_hessian_diagonal as the initial inverse Hessian approximation (Preconditioning).
        """
        q = gradient.copy()
        alpha_list = []
        
        # 1. Backward pass
        for s, y in reversed(list(zip(self.s_history, self.y_history))):
            rho = 1/np.dot(y, s)
            alpha = rho * np.dot(s, q)
            alpha_list.append(alpha)
            q -= alpha * y
        
        if self.s_history:
            last_s = self.s_history[-1]
            last_y = self.y_history[-1]
            # Scaling factor gamma
            gamma = np.dot(last_s, last_y) / np.dot(last_y, last_y)
        else:
            gamma = 1.0

        # 2. Apply initial Hessian approximation (H_0)
        # Using the inverse of the diagonal Hessian you already calculated
        # Safeguard against zero division
        print("diagonal", self.reduced_hessian_diagonal_zero)
        safe_diag = np.abs(self.reduced_hessian_diagonal_zero)
        safe_diag = np.where(safe_diag < 1e-6, 1.0, safe_diag)

        ## Simple diagonal preconditioning (no trust region here)
        z = q / safe_diag * gamma
        #z, shift_lambda, status = self.solve_gltr_trust_region(
        #                               self.U_zero,
        #                               self.A_tilde_zero,
        #                               self.G_blocks_zero,
        #                               -q,
        #                               self.reduced_hessian_diagonal_zero,
        #                               trust_radius,
        #                               max_iter=100,
        #                               tol=1e-10
        #                         )
        # 3. Forward pass
        for (s, y), alpha in zip(zip(self.s_history, self.y_history), reversed(alpha_list)):
            rho = 1/np.dot(y, s)
            beta = rho * np.dot(y, z)
            z += s * (alpha  -beta)
            
        return -z # The step is negative direction  
    

    def get_bfgs_mv(self, v, current_history=None):
        """
        Computes the product B * v using the recursive BFGS update (Eq. 63).

        Args:
            v: The vector to multiply.
            current_history: Optional list of history tuples. If None, uses self.bfgs_history.
        """
        if current_history is None:
            history = self.bfgs_history
        else:
            history = current_history

        # 1. Base Product: sigma = B_0 * v
        # We use the diagonal approximation for B_0
        # If reduced_hessian_diagonal has zeros, safeguard it
       
        n = self.index_map_size
        v_curr_reshaped = np.ascontiguousarray(v.reshape(1, n), dtype=np.float64)
        H_v = np.zeros_like(v_curr_reshaped)
        self.orbital_sigma3(self.U_zero, self.A_tilde_zero, self.G_blocks_zero, v_curr_reshaped, H_v, 1, 0)
        H_v = H_v.reshape(n)
        sigma = H_v


        #safe_diagonal = np.abs(self.reduced_hessian_diagonal_zero)
        #safe_diagonal = np.maximum(safe_diagonal, 1e-4) # Floor value

        #sigma = v * safe_diagonal

        # 2. Recursive Updates (Eq. 63)
        # We iterate from earliest step (k=0) to latest
        for (s, y, Bs, rho_y, rho_Bs) in history:
            # Terms needed:
            # y_dot_v = y^T * v
            # s_dot_sigma = s^T * sigma (equivalent to s^T * B_k * v)
            y_dot_v = np.dot(y, v)
            s_dot_sigma = np.dot(s, sigma) # Note: sigma here is B_k * v

            # Update formula:
            # sigma_new = sigma + (y * y_dot_v * rho_y) - (Bs * s_dot_sigma * rho_Bs)
            # where rho_y = 1 / (y^T s)
            # where rho_Bs = 1 / (s^T Bs)

            term1 = y * (y_dot_v * rho_y)
            term2 = Bs * (s_dot_sigma * rho_Bs)

            sigma = sigma + term1 - term2
         
        return sigma

    def _otr_trust_region_step(self, hess_x, h_diag, reduced_gradient, trust_radius, label='OTR'):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx._otr_trust_region_step, and the C++ port cites it).
        """
        return _trs__otr_trust_region_step(
            self,
            hess_x,
            h_diag,
            reduced_gradient,
            trust_radius,
            label,
        )

    def _orbital_hess_x(self, U, A_tilde, G_blocks):
        """``v -> H v`` through orbital_sigma3, for the AH-Krylov solver.

        A closure rather than a bound method so the same solver can serve the
        main trust-region site (U2 / A_tilde2 / G_blocks) and the QN site
        (U_zero / A_tilde_zero / G_blocks_zero), and so the QN-BFGS site can
        pass ``self.get_bfgs_mv`` instead of anything orbital_sigma3-shaped.
        """
        n = self.index_map_size

        def _hess_x(v):
            r = np.ascontiguousarray(np.asarray(v, dtype=float).reshape(1, n))
            out = np.zeros((1, n))
            self.orbital_sigma3(U, A_tilde, G_blocks, r, out, 1, 0)
            return out[0]

        return _hess_x

    def _ah_krylov_trust_region_step(self, hess_x, h_diag, reduced_gradient, trust_radius, label='AH-Krylov'):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx._ah_krylov_trust_region_step, and the C++ port cites it).
        """
        return _trs__ah_krylov_trust_region_step(
            self,
            hess_x,
            h_diag,
            reduced_gradient,
            trust_radius,
            label,
        )

    def _lstrs_hard_dump(self, U, A_tilde, G_blocks, reduced_gradient, alpha,
                         alpha_l, alpha_u, trust_radius, mu9, w9, v1, v2,
                         aa1, bb1, aa2, bb2, epsilon_v, aug_hessian_eigenvals):
        """Dump the LSTRS subproblem at a both-roots-hard termination.

        DIAGNOSTIC ONLY -- reached solely when QED_LSTRS_HARD_DUMP is set, and
        it changes nothing.  Builds the reduced Hessian densely by columns with
        the same batched orbital_sigma3 path the QED_NEWTON_DUMP block uses
        (:22138), so the near-degenerate pair the Davidson reports can be
        checked against an EXACT eigendecomposition of the bordered matrix
        B(alpha) = [[alpha, g^T], [g, H]] rather than against the iterative
        solver that produced it.
        """
        n = int(np.asarray(reduced_gradient).size)
        _batch = max(1, min(int(_QED_EXACT_TR_BATCH), n))
        H = np.zeros((n, n))
        col = 0
        while col < n:
            nb = min(_batch, n - col)
            R = np.zeros((nb, n))
            for k in range(nb):
                R[k, col + k] = 1.0
            R = np.ascontiguousarray(R)
            S = np.zeros((nb, n))
            self.orbital_sigma3(U, A_tilde, G_blocks, R, S, nb, 0)
            H[:, col:col + nb] = S.T
            col += nb
        seen = getattr(self, "_lstrs_hard_seen", 0)
        self._lstrs_hard_seen = seen + 1
        os.makedirs(_QED_LSTRS_HARD_DUMP, exist_ok=True)
        np.savez(
            os.path.join(_QED_LSTRS_HARD_DUMP, "lstrs_hard_%04d.npz" % seen),
            H=H,
            g=np.asarray(reduced_gradient, dtype=np.float64).ravel(),
            alpha=np.float64(alpha),
            alpha_l=np.float64(alpha_l),
            alpha_u=np.float64(alpha_u),
            trust_radius=np.float64(trust_radius),
            mu9=np.asarray(mu9, dtype=np.float64).ravel(),
            w9=np.asarray(w9, dtype=np.float64),
            v=np.array([v1, v2], dtype=np.float64),
            ab=np.array([aa1, bb1, aa2, bb2], dtype=np.float64),
            epsilon_v=np.float64(epsilon_v),
            aug_eigs=np.asarray(aug_hessian_eigenvals, dtype=np.float64).ravel(),
        )
        print("QED_LSTRS_HARD_DUMP: wrote lstrs_hard_%04d.npz (n=%d)"
              % (seen, n), flush=True)

    def _trs_hard_case_fallback(self, U, A_tilde, G_blocks, gradient,
                                trust_radius, site=""):
        """A real step for the LSTRS both-roots-unnormalizable state.

        Replaces the bare ``exit()`` those sites used to call
        (QED_TRS_HARD_FALLBACK).  Two routes, both of which always return a
        step:

        * dense More-Sorensen, when the reduced space is at or under
          QED_TRS_FALLBACK_DENSE_MAX.  ``_more_sorensen_dense`` has an explicit
          hard-case branch -- pseudo-inverse solution plus a multiple of a
          lowest eigenvector, sign chosen to minimise the model -- which is
          precisely the step a beta bisection cannot reach, because the
          quantity it bisects on (the first component of the eigenvector) is
          identically zero there.  Costs n Hessian-vector products.

        * GLTR above the cap.  Cannot fail, but has NO hard-case branch and its
          Krylov space is seeded by g, which in the hard case is orthogonal to
          the direction the optimum needs -- so it returns a valid DESCENT step,
          not the optimum.  Printed as such rather than passed off as exact.
        """
        n = int(np.asarray(gradient).size)
        if n <= _QED_TRS_FALLBACK_DENSE_MAX:
            step = self._exact_dense_trust_region_step(
                U, A_tilde, G_blocks, gradient, trust_radius)
            how = "dense More-Sorensen, hard case handled exactly"
        else:
            step, _lam, _status = self.solve_gltr_trust_region(
                U, A_tilde, G_blocks,
                np.asarray(gradient, dtype=float).ravel(),
                self.reduced_hessian_diagonal, trust_radius, verbose=False)
            how = ("GLTR (n=%d > QED_TRS_FALLBACK_DENSE_MAX=%d): a descent "
                   "step, NOT the hard-case optimum"
                   % (n, _QED_TRS_FALLBACK_DENSE_MAX))
        step = np.asarray(step, dtype=float).ravel()
        print("QED_TRS_HARD_FALLBACK%s: both LSTRS roots unnormalizable "
              "(genuine hard case) -> %s; |step| = %.6e, radius = %.6e"
              % (site, how, np.linalg.norm(step), trust_radius), flush=True)
        return step

    def microiteration_optimization6(
        self, U, eigenvecs, c_get_roots, convergence_threshold
    ):
        # print("E_core", self.E_core)
        self.U2 = copy.deepcopy(U)
        # QED_TRUE_ENERGY_PROBE.  One call to this routine IS one
        # macroiteration, so these are per-macro state: measure the model error
        # ONCE on the first candidate step of the macro, then reuse that verdict
        # for the rest of it.  See the knob note at the top of the file.
        probe_decided = False
        probe_engage = True
        # QED_TR_INIT.  THIS is the site QED_TRUST_PERSIST could not reach --
        # microiteration 1 takes its radius from function entry here, which is
        # why that knob measured as a confirmed no-op (item 20).
        trust_radius = _QED_TR_INIT
        rot_dim = self.nmo
        np1 = self.N_p + 1
        H_dim = self.num_alpha * self.num_alpha * np1
        A = np.zeros((rot_dim, rot_dim))
        G = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        G_ij = np.zeros((
               self.nmo * self.n_in_a, self.nmo * self.n_in_a    
           ))
        G_ti = np.zeros((
            self.nmo * self.n_act_orb, self.nmo * self.n_in_a    
        ))
        G_tu = np.zeros((
            self.nmo * self.n_act_orb, self.nmo * self.n_act_orb 
        ))
        G_blocks_zero = (G_ij, G_ti, G_tu)
        # A2 = np.zeros((rot_dim, rot_dim))
        # G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        # print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)
        self.s_history = []
        self.y_history = []
        self.bfgs_history = []
        self.m_history = 10
        self.density_norm_change = 0 
        self.predicted_energy = 10 

        active_twoeint = np.zeros(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        # E(U) at the current accumulated rotation.  Re-seeded from zero_energy
        # at the top of every microiteration and advanced on every accepted
        # step, so `true_change` below is always measured against the point the
        # step actually starts from.
        true_energy_current = 0.0

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))
        reduced_gradient = np.zeros(self.index_map_size)

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size + 1))
        # convergence_threshold = 1e-4
        convergence = 0
        current_residual = 1
        # while(True):
        current_energy = 0.0
        old_energy = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 20
        microiteration = 0
        qn_optimization = False
        qn_count = 0
        self.consecutive_skips = 0
        while microiteration < N_microiterations:
            print("\n")
            print("\n")
            print("MICROITERATION", microiteration + 1, flush=True)

            # while(microiteration < 2):
            # QED_TRUST_PERSIST: carry the trust radius across microiterations
            # instead of resetting it to 0.5 here.  Measured on c10h8: 13 of the
            # 14 rejections occur on the FIRST orbital step of a microiteration,
            # every cascade starting at Delta = 0.5, and the radii those cascades
            # finally accept are 0.031-0.25.  Since N_orbital_optimization_steps
            # is usually 1, the inner loop takes one accepted step and exits, so
            # this reset discards the step length essentially every time it is
            # learned.  Default 0 reproduces current behavior exactly.
            if not _QED_TRUST_PERSIST:
                trust_radius = _QED_TR_INIT  # QED_TR_INIT; default 0.5
            else:
                print("carried trust radius", trust_radius, flush=True)
            _mp0 = _mp_start()
            A[:, :] = 0.0
            G[:, :, :, :] = 0.0
            _mp_end("A_G_memset", _mp0)
            start = timer()
            self.build_intermediates(eigenvecs, A, G, True)
            #G_blocks = self.build_intermediates_with_blocks(eigenvecs, A, G, True)
            end = timer()
            print("build intermediates took", end - start)
            if _CPP_CASSCF_VALIDATION_DIR:
                _full_intermediates_dump = dict(
                    H_spatial2=self.H_spatial2,
                    d_cmo=self.d_cmo,
                    J=self.J,
                    K=self.K,
                    D_tu_avg=self.D_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                    D_tuvw_avg=self.D_tuvw_avg.reshape(
                        (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
                    ),
                    Dpe_tu_avg=self.Dpe_tu_avg.reshape((self.n_act_orb, self.n_act_orb)),
                    off_diagonal_constant=self.calculate_off_diagonal_photon_constant(eigenvecs),
                    omega=self.omega,
                    dims=np.array([self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]),
                    A=A,
                    G=G,
                )
                _dump_cpp_casscf_validation_case("full_intermediates", **_full_intermediates_dump)
            # A2[:,:] = 0.0
            # G2[:,:,:,:] = 0.0
            # self.build_intermediates2(eigenvecs, A2, G2, True)
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            #print(old_energy, current_energy)
            #print("initial convergence threshold", convergence_threshold)
            #if (
            #    np.abs(current_energy - old_energy)
            #    < max(0.01 * convergence_threshold, 1e-10)
            #) and microiteration >= 2:
            #    # if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
            #    print("microiteration converged (small energy change)")
            #    # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    break

            start = timer()
            start1 = timer()
            zero_energy = self.E_core
            zero_energy += np.dot(self.active_fock_core.flatten(), self.D_tu_avg)
            zero_energy += 0.5 * np.dot(self.active_twoeint.flatten(), self.D_tuvw_avg)
            zero_energy += -np.sqrt(self.omega / 2) * np.dot(
                self.d_cmo[
                    self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
                ].flatten(),
                self.Dpe_tu_avg,
            )
            end1 = timer()
            print("check zero energy took", end1 - start1)
            start1 = timer()
            zero_energy += self.calculate_ci_dependent_energy(eigenvecs, self.d_cmo)
            end1 = timer()
            print("check ci dependent energy took", end1 - start1)
            zero_energy += self.Enuc
            zero_energy += self.d_c
            print("zero energy", zero_energy, flush=True)
            # QED_TRUE_ENERGY_RATIO: establish this microiteration's ENERGY(VAR)
            # reference -- the exact energy at the orbitals the next step will
            # actually start from.
            #
            # CAREFUL, THIS IS THE SUBTLE PART.  `zero_energy` is NOT that
            # number except on the first microiteration.  self.U2 is set once at
            # entry (:14770) and then only ACCUMULATES -- it is never reset per
            # microiteration -- while self.J / self.K / self.H_spatial2 /
            # self.d_cmo are never rebound inside this routine at all.  So
            # build_intermediates re-derives self.E_core / self.active_* from the
            # MACROITERATION-ENTRY integrals every time, and `zero_energy` is the
            # exact energy at the MACRO-ENTRY orbitals contracted with the RDMs
            # from the CI solve that just ran.  The model E(T) is likewise a
            # second-order expansion about macro entry, evaluated at the full
            # accumulated U3 -- which is precisely why its error grows without
            # bound as U2 walks away from the expansion point.
            #
            # The step's true starting energy is therefore E(U) at self.U2 with
            # the CURRENT RDMs, which needs one explicit evaluation.  At
            # microiteration 1 self.U2 is the identity and this reproduces
            # zero_energy exactly; the cost is one second-order transform per
            # microiteration (0.0035 s c2h4 / 0.0114 s Cr2 / 0.050 s c10h8).
            if self.true_energy_mode():
                # Under QED_TRUE_ENERGY_REEXPAND the expansion point IS the
                # current orbitals, so self.U2 is the identity here and
                # zero_energy already is E(U) at that point -- skip a full
                # transform that would recompute exactly it.
                if _QED_TRUE_ENERGY_REEXPAND and np.array_equal(
                    self.U2, np.eye(self.nmo)
                ):
                    true_energy_current = zero_energy
                else:
                    true_energy_current = self.true_energy_at_trial(
                        self.U2, eigenvecs
                    )
                print(
                    "true energy at microiteration entry",
                    true_energy_current,
                    "(zero energy, at macro entry:",
                    zero_energy,
                    ")",
                    flush=True,
                )
            # QED_TRUE_ENERGY_RATIO mode 2 retry state.  Declared here, before
            # either branch, because the `count == 0` fixup below both branches
            # has to know whether the microiteration gave up.
            true_reject_streak = 0
            true_energy_gave_up = False
            start1 = timer()
            #G1 = G.transpose(3, 1, 2, 0).reshape(
            #    self.nmo * self.n_occupied, self.nmo * self.n_occupied
            #)
            # G_t is a free view (transpose only); the three reshapes below are
            # the copies that actually cost.  The full (nmo*n_occupied) square
            # G1 used to be materialized here too -- a second, larger copy --
            # but every consumer now takes the blocks, so it is gone.
            if _QED_G_TILED:
                # Same three arrays, same index convention, bit-for-bit; only the
                # copy loop differs.  See the QED_G_TILED note at the top.
                G_ij, G_ti, G_tu = self.tiled_G_blocks(G)
                # QED_G_TILED_CHECK: one-shot proof, in situ and at the real
                # dimensions, that the tiled copy reproduces the numpy path
                # EXACTLY (np.array_equal, not allclose -- a permuted copy moves
                # bytes and must not perturb a single ulp).  This exists because
                # the c10h8 A/B cannot demonstrate bit-exactness on its own: the
                # run is nondeterministic well upstream of here (two legs with
                # identical settings diverge in the CQED-RHF SCF, at the ~1e-11
                # level, from threaded-BLAS reduction order), so the two legs'
                # final energies differ whether or not this copy is faithful.
                if int(os.environ.get("QED_G_TILED_CHECK", "0")) and not getattr(
                    self, "_g_tiled_checked", False
                ):
                    self._g_tiled_checked = True
                    _Gt = G.transpose(3, 1, 2, 0)
                    _ref = (
                        _Gt[:, : self.n_in_a, :, : self.n_in_a].reshape(
                            self.nmo * self.n_in_a, self.nmo * self.n_in_a
                        ),
                        _Gt[:, self.n_in_a :, :, : self.n_in_a].reshape(
                            self.nmo * self.n_act_orb, self.nmo * self.n_in_a
                        ),
                        _Gt[:, self.n_in_a :, :, self.n_in_a :].reshape(
                            self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
                        ),
                    )
                    for _nm, _a, _b in zip(("G_ij", "G_ti", "G_tu"),
                                           (G_ij, G_ti, G_tu), _ref):
                        if not np.array_equal(_a, _b):
                            raise RuntimeError(
                                f"QED_G_TILED: {_nm} {_a.shape} differs from the "
                                f"numpy path, max|d| = {np.abs(_a - _b).max():.6e}"
                            )
                        print(f"[G_TILED_CHECK] {_nm} {_a.shape} bit-identical",
                              flush=True)
                    del _Gt, _ref
            else:
                G_t = G.transpose(3, 1, 2, 0)
                G_ij = G_t[:, :self.n_in_a, :, :self.n_in_a].reshape(
                    self.nmo * self.n_in_a, self.nmo * self.n_in_a
                )
                G_ti = G_t[:, self.n_in_a:, :, :self.n_in_a].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_in_a
                )
                G_tu = G_t[:, self.n_in_a:, :, self.n_in_a:].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
                )
            G_blocks = (G_ij, G_ti, G_tu)
            # QED_G_SYM_CHECK: one-shot verification that the two diagonal
            # orbital-Hessian blocks are symmetric.  G_ij[(a,b),(c,d)] = G[d,b,c,a],
            # so this is the statement G[p,q,r,s] == G[q,p,s,r]; the code already
            # relies on the off-diagonal half of it by writing G_it as G_ti.T at
            # :9400.  Measured on H2O 6-31g CAS(6,5), 3 roots, cavity on:
            # rel max|B-B.T| = 2.8e-16 for G_ij and 2.3e-16 for G_tu, i.e. 1 ulp.
            # This matters because the matvec is memory-bandwidth bound at machine
            # peak (48 GB/s on the 218 MB G_ij at c10h8 dims), so a symmetric
            # product that reads only one triangle is a measured 2.4x on the
            # dominant term -- see the QED_G_SYMV note at the top of the file.
            if int(os.environ.get("QED_G_SYM_CHECK", "0")) and not getattr(
                self, "_g_sym_checked", False
            ):
                self._g_sym_checked = True
                for _nm, _B in (("G_ij", G_ij), ("G_tu", G_tu)):
                    _s = np.abs(_B).max()
                    _d = np.abs(_B - _B.T).max()
                    print(
                        f"[G_SYM_CHECK] {_nm} {_B.shape} max|B|={_s:.6e} "
                        f"max|B-B.T|={_d:.6e} rel={_d/max(_s,1e-300):.3e}",
                        flush=True,
                    )
            end1 = timer()
            print("transpose matrix G took", end1 - start1)
            start1 = timer()
            initial_energy_change = self.microiteration_exact_energy(self.U2, A, G_blocks)
            end1 = timer()
            print("calculate initial exact energy took", end1 - start1)
            current_energy = zero_energy + initial_energy_change
            print(
                "current energy from zero energy + second order energy change",
                zero_energy + initial_energy_change,
                flush=True,
            )
            print(old_energy, current_energy)
            print("initial convergence threshold", convergence_threshold)
            if (
                np.abs(current_energy - old_energy)
                < max(_QED_MICRO_ENERGY_SCALE * convergence_threshold,
                      _QED_MICRO_ENERGY_FLOOR)
            ) and microiteration >= 2:
                # if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)")
                # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                break


            old_energy = current_energy
            end = timer()
            print("check initial energy took", end - start)

            #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            # print(np.shape(gradient_tilde), flush = True)
            old_reduced_gradient = copy.deepcopy(reduced_gradient)
            #print("old gradient", old_reduced_gradient, flush = True)
            start = timer()
            self.build_gradient(
                self.U2, A, G, gradient_tilde, A_tilde2, True, G_blocks=G_blocks
            )
            end = timer()
            # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)

            print("build gradient took", end - start)
            if _CPP_CASSCF_VALIDATION_DIR:
                _dump_cpp_casscf_validation_case(
                    "build_gradient",
                    U=self.U2, A=A, G=G, A_tilde=A_tilde2, gradient_tilde=gradient_tilde,
                    dims=np.array([self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]),
                )
            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end = timer()
            print("build hessian diagonal took", end - start)
            # QED_MV_BENCH: one-shot cost curve of the Hessian-vector product as a
            # function of how many vectors are built per call.  The question it
            # answers is what a BATCHED product would be worth: the four GEMMs at
            # 9373-9376 read the whole 0.319 GB of G regardless of the vector
            # count, so if their cost is flat in the batch size then any solver
            # that can ask for several products at once gets them nearly free.
            # Reports the MINIMUM over repetitions, not the mean: the first
            # version of this used the mean and was run while the machine was
            # loaded, which inflated the large-batch points and made batching look
            # 3x worse than it is.  Timings only; no state is touched, and the
            # sigma output goes to a scratch buffer.
            if _QED_MV_BENCH and not getattr(self, "_mv_bench_done", False):
                self._mv_bench_done = True
                _n = self.index_map_size
                _gb = 8.0 * sum(b.size for b in G_blocks) / 1e9
                print(f"\n--- MATVEC BATCH BENCHMARK (n={_n}, G blocks {_gb:.3f} GB) ---")
                print(f"{'batch':<8}{'reps':<6}{'best s/call':>13}{'s/vector':>12}{'vs batch1':>11}")
                _base = None
                for _bs in (1, 2, 4, 8, 16, 32):
                    _R = np.ascontiguousarray(np.random.randn(_bs, _n))
                    _S = np.zeros((_bs, _n))
                    _reps = 7
                    self.orbital_sigma3(self.U2, A_tilde2, G_blocks, _R, _S, _bs, 0)
                    _best = None
                    for _ in range(_reps):
                        _t0 = timer()
                        self.orbital_sigma3(self.U2, A_tilde2, G_blocks, _R, _S, _bs, 0)
                        _dt = timer() - _t0
                        _best = _dt if _best is None else min(_best, _dt)
                    if _base is None:
                        _base = _best
                    print(
                        f"{_bs:<8}{_reps:<6}{_best:>13.5f}{_best/_bs:>12.5f}"
                        f"{_best/_base:>10.2f}x"
                    )
                print("--- END MATVEC BATCH BENCHMARK ---\n", flush=True)
            if _CPP_CASSCF_VALIDATION_DIR:
                _dump_cpp_casscf_validation_case(
                    "build_hessian_diagonal",
                    U=self.U2, G=G, A_tilde=A_tilde2,
                    hessian_diagonal=self.hessian_diagonal,
                    reduced_hessian_diagonal=self.reduced_hessian_diagonal,
                    dims=np.array([self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]),
                )
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            _mp0 = _mp_start()
            index_count1 = 0
            for k in range(self.n_occupied):
                for r in range(k + 1, self.nmo):
                    if k < self.n_in_a and r < self.n_in_a:
                        continue
                    if (
                        self.n_in_a <= k < self.n_occupied
                        and self.n_in_a <= r < self.n_occupied
                    ):
                        continue
                    # QED_ORBITAL_SYMMETRY: skip rotations that mix symmetry species.
                    if (
                        _QED_ORBITAL_SYMMETRY
                        and self._sym_labels is not None
                        and self._sym_labels[r] != self._sym_labels[k]
                    ):
                        continue
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            new_reduced_gradient = copy.deepcopy(reduced_gradient)
            _mp_end("redgrad_loop_outer", _mp0)
           
            if qn_count > 1:
                s_vec = step  # The step we just took
                #print("step", step, flush = True)
                y_vec = new_reduced_gradient - old_reduced_gradient
                #print("y vec", y_vec, flush = True)
                curvature = np.dot(y_vec, s_vec)
                y = y_vec
                s = s_vec
                dot_ys = np.dot(y, s)
                norm_y = np.linalg.norm(y)
                norm_s = np.linalg.norm(s)
                
                cosine_angle = dot_ys / (norm_y * norm_s)
                
                print(f"  Step: {norm_s:.4e}, GradDiff: {norm_y:.4e}")
                print(f"  Curvature (y.s): {dot_ys:.4e} -> Rho: {1.0/dot_ys:.4e}")
                print(f"  Angle (Cos): {cosine_angle:.4f}")
                print("predicted energy", self.predicted_energy)
                
                Bs_vec = self.get_bfgs_mv(s_vec) 
                
                # 2. Calculate Curvature Scalars
                ys_dot = np.dot(y_vec, s_vec)      # Real curvature
                sBs_dot = np.dot(s_vec, Bs_vec)    # Model curvature
                
                # 3. Damping Logic
                sigma = 0.1 
                if ys_dot < sigma * sBs_dot:
                    print(f"  Curvature Violation ({ys_dot:.4e} < 0.2 * {sBs_dot:.4e}). Damping...")
                    theta = ((1.0 - sigma) * sBs_dot) / (sBs_dot - ys_dot)
                    y_bar = theta * y_vec + (1.0 - theta) * Bs_vec
                    
                    # Use damped vectors for update
                    final_y = y_bar
                    final_rho_y = 1.0 / (sigma * sBs_dot) # Guaranteed positive
                else:
                    # Standard BFGS
                    final_y = y_vec
                    final_rho_y = 1.0 / ys_dot
                
                # 4. Update History with SAFE vector
                # Store (s, y_bar, Bs_vec, rho_y_bar, rho_Bs)
                rho_Bs = 1.0 / sBs_dot
                self.bfgs_history.append((s_vec, final_y, Bs_vec, final_rho_y, rho_Bs))
                
                if len(self.bfgs_history) > self.m_history:
                    self.bfgs_history.pop(0)
                    
                print(f"  BFGS History Updated. Size: {len(self.bfgs_history)}")

            print("number of skips", self.consecutive_skips)

            if self.density_norm_change > 0.025 and qn_optimization == True or self.predicted_energy > 0 or qn_count == 1:
                self.A_tilde_zero = copy.deepcopy(A_tilde2)
                self.G_blocks_zero = copy.deepcopy(G_blocks)
                self.reduced_hessian_diagonal_zero = copy.deepcopy(self.reduced_hessian_diagonal)
                self.U_zero = copy.deepcopy(self.U2)
                self.s_history = []
                self.y_history = []
                self.bfgs_history = []
            print(f"  BFGS History Updated. Size: {len(self.bfgs_history)}")




            gradient_norm = np.linalg.norm(reduced_gradient)
            print("current gradient_norm and residual", gradient_norm, current_residual)
            print("current convergence_threshold", convergence_threshold)
            total_norm = np.sqrt(
                np.power(gradient_norm, 2) + np.power(current_residual, 2)
            )
            if total_norm < convergence_threshold:
                print("total norm", total_norm, flush=True)
                # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                # temp8 = np.zeros((self.nmo, self.nmo))
                # temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                # self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                # self.d_cmo[:,:] = d_cmo[:,:]
                # print(eigenvecs)
                # print("u2i",self.U2)
                print("microiteration converged! (small total norm)", flush=True)
                break
            
            if qn_optimization == True:
                
                print("--- Doing Direct BFGS Step (Kreplin Eq. 63) ---")
                
                if self.density_norm_change > 0.025 and qn_optimization == True or self.predicted_energy > 0 or qn_count == 1 or self.consecutive_skips >= 3:
                    print("solve step for original hessian")
                    _qn_gltr_dump_inputs = None
                    if _CPP_CASSCF_VALIDATION_DIR:
                        _n_qn = self.index_map_size
                        _H_dense = np.zeros((_n_qn, _n_qn))
                        for _i in range(_n_qn):
                            _e_i = np.zeros((1, _n_qn))
                            _e_i[0, _i] = 1.0
                            _Hv = np.zeros((1, _n_qn))
                            self.orbital_sigma3(self.U_zero, self.A_tilde_zero, self.G_blocks_zero, _e_i, _Hv, 1, 0)
                            _H_dense[:, _i] = _Hv[0, :]
                        _qn_gltr_dump_inputs = dict(
                            hessian=_H_dense,
                            m_diag=self.reduced_hessian_diagonal_zero,
                            trust_radius=trust_radius,
                        )
                    if _QED_OTR_TR_QN:
                        _stats_span = _stats_begin("otr_qn")
                        step = self._otr_trust_region_step(
                            self._orbital_hess_x(self.U_zero, self.A_tilde_zero,
                                                 self.G_blocks_zero),
                            self.reduced_hessian_diagonal_zero,
                            reduced_gradient, trust_radius, label="OTR(qn)",
                        )
                        _stats_end(_stats_span)
                        self.consecutive_skips = 0
                        _qn_gltr_dump_inputs = None
                    elif _QED_AH_KRYLOV_QN:
                        _stats_span = _stats_begin("ah_krylov_qn")
                        step = self._ah_krylov_trust_region_step(
                            self._orbital_hess_x(self.U_zero, self.A_tilde_zero,
                                                 self.G_blocks_zero),
                            self.reduced_hessian_diagonal_zero,
                            reduced_gradient, trust_radius, label="AH-Krylov(qn)",
                        )
                        _stats_end(_stats_span)
                        self.consecutive_skips = 0
                        _qn_gltr_dump_inputs = None
                    else:
                     _stats_span = _stats_begin("gltr_qn")
                     step_p, shift_lambda, status = self.solve_gltr_trust_region(
                                    self.U_zero,
                                    self.A_tilde_zero,
                                    self.G_blocks_zero,
                                    reduced_gradient,
                                    self.reduced_hessian_diagonal_zero,
                                    trust_radius,
                                    max_iter=1000,
                                    tol=self._subproblem_tol("qn", "gltr_qn")
                          )
                     _stats_end(_stats_span, status=status)
                     self._record_gltr_exit(status, "gltr_qn")
                     if _qn_gltr_dump_inputs is not None:
                         # gradient=the actual noised r solve_gltr_trust_region used,
                         # not reduced_gradient -- see that function's doc comment.
                         _dump_cpp_casscf_validation_case(
                             "qn_gltr", step=step_p,
                             gradient=self._cpp_casscf_last_noised_gradient,
                             **_qn_gltr_dump_inputs,
                         )
                     step = step_p
                     self.consecutive_skips = 0

                else:
                    print("solve bfgs for updated hessian")
                    # 1. Define the Linear Operator B
                    # This operator wraps our matrix-vector product function
                    B_op = LinearOperator(
                        (self.index_map_size, self.index_map_size),
                        matvec=self.get_bfgs_mv
                    )
                    _qn_bfgs_dump_inputs = None
                    if _CPP_CASSCF_VALIDATION_DIR:
                        _n_qn = self.index_map_size
                        _H_dense = np.zeros((_n_qn, _n_qn))
                        for _i in range(_n_qn):
                            _e_i = np.zeros(_n_qn)
                            _e_i[_i] = 1.0
                            _H_dense[:, _i] = self.get_bfgs_mv(_e_i)
                        _qn_bfgs_dump_inputs = dict(
                            hessian=_H_dense,
                            m_diag=self.reduced_hessian_diagonal_zero,
                            trust_radius=trust_radius,
                        )
                    # OR adapt your GLTR call:
                    # You effectively need to solve (B + lambda*I) step = -g
                    if _QED_OTR_TR_QN:
                        # get_bfgs_mv is the same symmetric operator B_op wraps;
                        # OTR, like AH-Krylov, needs only v -> B v plus a
                        # diagonal for the preconditioner.
                        _stats_span = _stats_begin("otr_qn_bfgs")
                        step = self._otr_trust_region_step(
                            self.get_bfgs_mv,
                            self.reduced_hessian_diagonal_zero,
                            reduced_gradient, trust_radius, label="OTR(qn-bfgs)",
                        )
                        _stats_end(_stats_span)
                    elif _QED_AH_KRYLOV_QN:
                        # get_bfgs_mv is the same symmetric operator B_op wraps;
                        # the solver only needs v -> B v and a diagonal for the
                        # preconditioner.
                        _stats_span = _stats_begin("ah_krylov_qn_bfgs")
                        step = self._ah_krylov_trust_region_step(
                            self.get_bfgs_mv,
                            self.reduced_hessian_diagonal_zero,
                            reduced_gradient, trust_radius,
                            label="AH-Krylov(qn-bfgs)",
                        )
                        _stats_end(_stats_span)
                    else:
                        _stats_span = _stats_begin("gltr_qn_bfgs")
                        step_p, shift_lambda, status= self.solve_gltr_with_operator(B_op, reduced_gradient,
                                                          self.reduced_hessian_diagonal_zero,
                                                                            trust_radius,
                                                                            max_iter=1000,
                                                                            tol=self._subproblem_tol("qn", "gltr_qn_bfgs"))
                        _stats_end(_stats_span, status=status)
                        self._record_gltr_exit(status, "gltr_qn_bfgs")
                        if _qn_bfgs_dump_inputs is not None:
                            # gradient=the actual noised r solve_gltr_with_operator used,
                            # not reduced_gradient -- see that function's doc comment.
                            _dump_cpp_casscf_validation_case(
                                "qn_bfgs", step=step_p,
                                gradient=self._cpp_casscf_last_noised_gradient,
                                **_qn_bfgs_dump_inputs,
                            )
                        step = step_p
                        print(status)
                    # 3. Trust Region Check
                step_norm = np.linalg.norm(step)
                #if step_norm > trust_radius:
                #    print(f"Step too large ({step_norm:.4f}), scaling to TR")
                #    step *= (trust_radius / step_norm)
                

                # Force hard_case = 0 for QN as we aren't solving the exact subproblem
                hard_case = 0

                step_norm = np.linalg.norm(step)

                print("step norm", step_norm)
                predicted_energy2 = self.microiteration_predicted_energy2(
                    self.U2, reduced_gradient, A_tilde2, G_blocks, step
                )
                #while predicted_energy2 >0:
                #    step = step * 0.5
                #    predicted_energy2 = self.microiteration_predicted_energy2(
                #       self.U2, reduced_gradient, A_tilde2, G1, step
                #    )
                #    step_norm = np.linalg.norm(step)
                #    print("step norm", step_norm)
                self.predicted_energy = predicted_energy2

                start = timer()
                Rai = np.zeros((self.n_act_orb, self.n_in_a))
                Rvi = np.zeros((self.n_virtual, self.n_in_a))
                Rva = np.zeros((self.n_virtual, self.n_act_orb))
                for i in range(self.index_map_size):
                    s = self.index_map[i][0]
                    l = self.index_map[i][1]
                    if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                        Rai[s - self.n_in_a][l] = step[i]
                    elif s >= self.n_occupied and l < self.n_in_a:
                        Rvi[s - self.n_occupied][l] = step[i]
                    else:
                        Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

                self.build_unitary_matrix(Rai, Rvi, Rva)
                # print("jnti",self.E_core)

                U3_before_step = self.U3
                self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta)
                second_order_energy_change = self.microiteration_exact_energy(
                    self.U3, A, G_blocks
                )
                # exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                # print("exact energy2", exact_energy2, flush = True)
                energy_change = (
                    zero_energy + second_order_energy_change - current_energy
                )
                print(
                    "old energy",
                    current_energy,
                    "energy change",
                    energy_change,
                    "second order energy change",
                    second_order_energy_change,
                    flush=True,
                )
                print("new energy", current_energy + energy_change, flush=True)

                # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                predicted_energy2 = self.microiteration_predicted_energy2(
                    self.U2, reduced_gradient, A_tilde2, G_blocks, step
                )
                self.predicted_energy = predicted_energy2
                if energy_change > 0:
                    print("qwrqwrq")
                else:
                    print("erewrw")
                print("microiteration predicted energy", predicted_energy2, flush=True)
                end = timer()
                print("build unitary matrix and recheck energy took", end - start)

                # QED_TRUE_ENERGY_RATIO -- the QN branch.  Unlike the main
                # branch this one has NO acceptance test, NO ratio and NO trust
                # radius: it applies its step unconditionally.  That is the
                # amplifier item 322 identified.  Measure E(U) here too, and in
                # mode 2 refuse a step that raises it, handing the microiteration
                # back to the main branch (which does have a trust region) by
                # clearing qn_optimization.
                qn_step_rejected = False
                qn_true_energy_engaged = False
                qn_te_mode = self.true_energy_mode()
                if qn_te_mode:
                    U2_trial = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                    qn_true_energy_engaged = self.true_energy_should_engage(U2_trial)
                if qn_true_energy_engaged:
                    start1 = timer()
                    true_energy_trial = self.true_energy_at_trial(U2_trial, eigenvecs)
                    end1 = timer()
                    print("true energy transformation took", end1 - start1)
                    true_change = true_energy_trial - true_energy_current
                    print(
                        "qn true energy E(U)",
                        true_energy_trial,
                        "model energy E(T)",
                        current_energy + energy_change,
                        "PROJ-VAR gap",
                        (current_energy + energy_change) - true_energy_trial,
                        flush=True,
                    )
                    print(
                        "qn true energy change",
                        true_change,
                        "model energy change",
                        energy_change,
                        flush=True,
                    )
                    if qn_te_mode >= 2 and not (
                        true_change < _QED_TRUE_ENERGY_TOL
                    ):
                        qn_step_rejected = True
                        print(
                            "qn step raises the true energy -- rejected,"
                            " leaving qn optimization",
                            flush=True,
                        )
                        self.U3 = U3_before_step
                        qn_optimization = False

                if qn_step_rejected:
                    hard_case = 0
                else:
                    qn_count +=1
                    self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)

                    if qn_true_energy_engaged:
                        true_energy_current = true_energy_trial
                    active_twoeint[:, :, :, :] = 0.0
                    active_fock_core[:, :] = 0.0
                    d_cmo[:, :] = 0.0
                    start1 = timer()
                    self.microiteration_ci_integrals_transform(
                        self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint
                    )
                    end1 = timer()
                    print("second order integral transformation took", end1 - start1)
                    count += 1
                    current_energy = zero_energy + second_order_energy_change
                    hard_case = 0


            else:
                # Analyze problem structure
                n_negative = np.sum(self.reduced_hessian_diagonal < 0)
                
                print(f"  Negative curvature directions: {n_negative}/{self.index_map_size}")


                orbital_optimization_step = 0
                restart = False
                count = 0
                hard_case = 0
                reduce_step = 0
                first_component = np.zeros(self.index_map_size)
                second_component = np.zeros(self.index_map_size)
                critical_step_norm = 0
                reduce_predicted_energy = 0
                while orbital_optimization_step < N_orbital_optimization_steps:
                    self.U3 = copy.deepcopy(self.U2)
                    _mp0_sv = _mp_start()
                    print("\n", current_energy)
                    print(
                        "Microiteration",
                        microiteration + 1,
                        "orbital optimization step",
                        orbital_optimization_step + 1,
                        flush=True,
                    )
                    gradient_norm = np.linalg.norm(reduced_gradient)
                    # QED_GRAD_CONV: first gradient of this macro's orbital work.
                    if getattr(self, "_macro_entry_gnorm", None) is None:
                        self._macro_entry_gnorm = gradient_norm
                    print("gradient norm", gradient_norm, flush=True)
                    print("convergence_threshold", convergence_threshold, flush=True)

                    # if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                    # if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                    # if (gradient_norm < 1e-5 and microiteration > 0):
                    if (gradient_norm
                            < _QED_MICRO_GRAD_SCALE * convergence_threshold
                            and microiteration > 0):
                        convergence = 1
                        # print("Microiteration converged (small gradient norm)")
                        break
                    if (_QED_MICRO_GRAD_FLOOR > 0.0
                            and gradient_norm < _QED_MICRO_GRAD_FLOOR):
                        convergence = 1
                        break
                    #print("calling pcg for trust region") 
                    #step_p, oo = self.solve_pcg_trust_region(self.U2, A_tilde2, G1, reduced_gradient, self.reduced_hessian_diagonal, trust_radius, tol=1e-8, max_iter=1000)
                    #print(oo)
                    # new augmented hessian
                    beta = 0
                    dim00 = self.index_map_size + 1
                    _mp0_ah = _mp_start()
                    augmented_hessian9 = np.zeros((dim00, dim00))
                    w9 = np.zeros((dim00, dim00))
                    _mp_end("aug_hess_alloc", _mp0_ah)
                    step = np.zeros(self.index_map_size)
                    step10_norm = 0.0
                    
                    # QED_OTR_TR / QED_AH_KRYLOV: one call replaces the
                    # TRUST-REGION subproblem only -- GLTR when the projected
                    # Hessian is positive definite, the LSTRS/Davidson beta
                    # bisection when it is not.  The gradient-small branch at
                    # the bottom keeps its MINRES Newton solve under either
                    # knob: that is a linear solve, not a trust-region
                    # subproblem, and is out of scope for both ported solvers.
                    # NOTE this is a DELIBERATE NARROWING of QED_OTR_TR's
                    # earlier scope, which also replaced newton_small_gradient
                    # (728 of c10h8's 2430 matvecs).  Measurements taken before
                    # this change are not comparable to ones taken after.
                    # Written as leading branches so the original path stays
                    # byte-for-byte intact when both knobs are off.
                    # hard_case stays 0 under either: neither ported solver has
                    # a hard-case branch, they expand the search space instead
                    # (opentrustregion.f90:763-770).
                    if np.linalg.norm(reduced_gradient) > 1e-3:
                        if _QED_EXACT_TR == 2:
                            # QED_EXACT_TR=2: take the WHOLE main site, both the
                            # GLTR and the LSTRS branch, for a clean A/B of
                            # "main site solved exactly".  =1 (below) takes only
                            # the GLTR branch and leaves LSTRS alone, which is
                            # the default because LSTRS is used exactly where
                            # Kreplin's P-subspace guess works.
                            _stats_span = _stats_begin("exact_tr")
                            step = self._exact_dense_trust_region_step(
                                self.U2, A_tilde2, G_blocks,
                                reduced_gradient, trust_radius,
                            )
                            _stats_end(_stats_span)
                        elif _QED_OTR_TR:
                            # own QED_SOLVER_STATS site, so its matvecs are
                            # attributed exactly like gltr's and lstrs's
                            _stats_span = _stats_begin("otr_trust_region")
                            step = self._otr_trust_region_step(
                                self._orbital_hess_x(self.U2, A_tilde2, G_blocks),
                                self.reduced_hessian_diagonal,
                                reduced_gradient, trust_radius
                            )
                            _stats_end(_stats_span)
                        elif _QED_AH_KRYLOV:
                            _stats_span = _stats_begin("ah_krylov")
                            step = self._ah_krylov_trust_region_step(
                                self._orbital_hess_x(self.U2, A_tilde2, G_blocks),
                                self.reduced_hessian_diagonal,
                                reduced_gradient, trust_radius,
                            )
                            _stats_end(_stats_span)
                        elif _QED_EXACT_TR and n_negative == 0:
                            # Replaces GLTR ONLY.  This is the branch with the
                            # coverage gap: n_negative == 0 tests the DIAGONAL,
                            # and 4 of 7 such subproblems on MgH+ CAS(8,12) are
                            # indefinite anyway, with a delocalized v_min that
                            # neither the P-subspace guess nor e_argmin(h_diag)
                            # can point at.  The LSTRS branch below is left
                            # untouched -- it is used precisely where its guess
                            # is good.
                            _stats_span = _stats_begin("exact_tr")
                            step = self._exact_dense_trust_region_step(
                                self.U2, A_tilde2, G_blocks,
                                reduced_gradient, trust_radius,
                            )
                            _stats_end(_stats_span)
                        elif n_negative == 0:
                             _gltr_dump_inputs = None
                             if _CPP_CASSCF_VALIDATION_DIR:
                                 _n_gltr = self.index_map_size
                                 _H_dense = np.zeros((_n_gltr, _n_gltr))
                                 for _i in range(_n_gltr):
                                     _e_i = np.zeros((1, _n_gltr))
                                     _e_i[0, _i] = 1.0
                                     _Hv = np.zeros((1, _n_gltr))
                                     self.orbital_sigma3(self.U2, A_tilde2, G_blocks, _e_i, _Hv, 1, 0)
                                     _H_dense[:, _i] = _Hv[0, :]
                                 _gltr_dump_inputs = dict(
                                     hessian=_H_dense,
                                     m_diag=self.reduced_hessian_diagonal,
                                     trust_radius=trust_radius,
                                 )
                             _stats_span = _stats_begin("gltr")
                             step_p, shift_lambda, status = self.solve_gltr_trust_region(
                                   self.U2,
                                   A_tilde2,
                                   G_blocks,
                                   reduced_gradient,
                                   self.reduced_hessian_diagonal,
                                   trust_radius,
                                   max_iter=1000,
                                   tol=self._subproblem_tol("main", "gltr")
                             )
                             _stats_end(_stats_span, status=status)
                             self._record_gltr_exit(status, "gltr")
                             if _gltr_dump_inputs is not None:
                                 # gradient=the actual noised r solve_gltr_trust_region used,
                                 # not reduced_gradient -- see that function's doc comment.
                                 _dump_cpp_casscf_validation_case(
                                     "gltr", step=step_p,
                                     gradient=self._cpp_casscf_last_noised_gradient,
                                     **_gltr_dump_inputs,
                                 )
                             step = step_p
                        else:
                             if (
                                 hard_case == 1
                                 and reduce_step == 1
                                 and np.linalg.norm(first_component) < trust_radius
                                 and reduce_predicted_energy == 1
                             ):
                                 print("hard case and current step is smaller than trust radius")
                                 print(np.linalg.norm(first_component))
                                 xy_square = np.dot(first_component, second_component) * np.dot(
                                     first_component, second_component
                                 )
                                 x_square = np.dot(first_component, first_component)
                                 y_square = np.dot(second_component, second_component)
                                 delta = 4 * xy_square - 4 * y_square * (
                                     x_square - trust_radius * trust_radius
                                 )
                                 # print(delta)
                                 t1 = (
                                     -2 * np.dot(first_component, second_component)
                                     - np.sqrt(delta)
                                 ) / (2 * y_square)
                                 t2 = (
                                     -2 * np.dot(first_component, second_component)
                                     + np.sqrt(delta)
                                 ) / (2 * y_square)
                                 print(
                                     "x^2, xy, y^2, t",
                                     x_square,
                                     np.dot(first_component, second_component),
                                     y_square,
                                     t1,
                                 )
                                 adjusted_step = first_component + min(np.abs(t1), np.abs(t2)) * second_component
                                 print("adjusted step norm", np.linalg.norm(adjusted_step))
                                 step = adjusted_step
                             else:
                                 step, hard_case = solve_lstrs_microiteration(
                                     self, A_tilde2, G, G_blocks, dim00,
                                     reduced_gradient, trust_radius, w9, first_component,
                                     second_component, guess_vector, hard_case, beta)
                                 

                    else:
                        print("gradient is small, use Newton step")
                        _stats_span = _stats_begin("newton_small_gradient")
                        denom = self.reduced_hessian_diagonal
                        max_iter = _QED_NEWTON_MAXITER  # QED_NEWTON_MAXITER, default 20
                        # QED_NEWTON_SCIPY overrides QED_USE_MINRES; both default off.
                        if _QED_NEWTON_SCIPY:
                            _solve_newton = self.scipy_minres_solve
                        else:
                            _solve_newton = self.minres_solve if _QED_USE_MINRES else self.linear_equation_solve
                        solution,converged = _solve_newton(self.U2, A_tilde2, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6)
                        if not converged:
                            print("\n--- Falling back to MINRES solver ---")
                            iteration_count = [0]
                            def callback(xk):
                                """Callback function called at each iteration"""
                                iteration_count[0] += 1
                                if not _QED_MINRES_TRACE:
                                    return
                                # costs one extra mv2 -- see QED_MINRES_TRACE
                                absolute_residual = np.linalg.norm(H1_op.matvec(xk) + reduced_gradient)
                                relative_residual = absolute_residual / np.linalg.norm(reduced_gradient)
                                print(f"MINRES Iter: {iteration_count[0]:3d}   "
                                f"Abs Residual: {absolute_residual:.4e}   "
                                f"Rel Residual: {relative_residual:.4e}")
   
                            H1_op = LinearOperator(
                                (self.index_map_size, self.index_map_size),
                                matvec=lambda Q: self.mv2(self.U2, A_tilde2, G_blocks, Q, 1, 0, 0),
                            )
                            solution, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6, callback=callback)
                            print(f"MINRES exit code: {exitCode}")
                            
                            if exitCode == 0:
                                print("MINRES converged successfully")
                            else:
                                print(f"MINRES warning: exit code {exitCode}")
                        #print("step norm2", np.linalg.norm(solution))
                        _stats_end(_stats_span, status="newton")
                        hard_case = 2
                        step = solution
                        #print(step)
                        #print(solution)
                    # w[:,0] = w[:,0]/scale

                    if _CPP_CASSCF_VALIDATION_DIR and n_negative > 0 and np.linalg.norm(reduced_gradient) > 1e-3:
                        _n_dav = self.index_map_size
                        _H_dense = np.zeros((_n_dav, _n_dav))
                        for _i in range(_n_dav):
                            _e_i = np.zeros((1, _n_dav))
                            _e_i[0, _i] = 1.0
                            _Hv = np.zeros((1, _n_dav))
                            self.orbital_sigma3(self.U2, A_tilde2, G_blocks, _e_i, _Hv, 1, 0)
                            _H_dense[:, _i] = _Hv[0, :]
                        _dump_cpp_casscf_validation_case(
                            "davidson_lstrs",
                            hessian=_H_dense,
                            gradient=reduced_gradient,
                            m_diag=self.reduced_hessian_diagonal,
                            trust_radius=trust_radius,
                            hard_case=hard_case,
                            step=step,
                        )

                    step_norm = np.linalg.norm(step)

                    print("step norm", step_norm)
                    #print("step", step)
                    _mp_end("inner_solver_block", _mp0_sv)
                    start = timer()
                    Rai = np.zeros((self.n_act_orb, self.n_in_a))
                    Rvi = np.zeros((self.n_virtual, self.n_in_a))
                    Rva = np.zeros((self.n_virtual, self.n_act_orb))
                    for i in range(self.index_map_size):
                        s = self.index_map[i][0]
                        l = self.index_map[i][1]
                        if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                            Rai[s - self.n_in_a][l] = step[i]
                        elif s >= self.n_occupied and l < self.n_in_a:
                            Rvi[s - self.n_occupied][l] = step[i]
                        else:
                            Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

                    self.build_unitary_matrix(Rai, Rvi, Rva)
                    # print("jnti",self.E_core)

                    self.U3 = np.einsum("pq,qs->ps", self.U3, self.U_delta)
                    second_order_energy_change = self.microiteration_exact_energy(
                        self.U3, A, G_blocks
                    )
                    # exact_energy2 = self.microiteration_exact_energy(self.U3, A2, G2)
                    # print("exact energy2", exact_energy2, flush = True)
                    energy_change = (
                        zero_energy + second_order_energy_change - current_energy
                    )
                    print(
                        "old energy",
                        current_energy,
                        "energy change",
                        energy_change,
                        "second order energy change",
                        second_order_energy_change,
                        flush=True,
                    )
                    print("new energy", current_energy + energy_change, flush=True)

                    # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                    predicted_energy2 = self.microiteration_predicted_energy2(
                        self.U2, reduced_gradient, A_tilde2, G_blocks, step
                    )
                    self.predicted_energy = predicted_energy2
                    if predicted_energy2 < 0:
                        reduce_predicted_energy = 1
                    else:
                        reduce_predicted_energy = 0

                    print("microiteration predicted energy", predicted_energy2, flush=True)
                    end = timer()
                    print("build unitary matrix and recheck energy took", end - start)

                    # QED_TRUE_ENERGY_RATIO.  `energy_change` above is a change in
                    # the MODEL E(T); on an indefinite G the model is unbounded
                    # below, so `energy_change < 0.0` is not evidence the step is
                    # any good.  Evaluate the TRUE energy E(U) at the trial
                    # rotation and judge the step by that instead.  See the knob
                    # note at the top of this file for the cost argument -- on an
                    # accepted step this transform REPLACES the one further down
                    # rather than adding to it.
                    accept_by_true_energy = None
                    true_change = 0.0
                    # QED_TRUE_ENERGY_ADAPTIVE (item 330): decide BEFORE paying
                    # for the transform.  When it declines, nothing below runs
                    # and the shipping model test governs acceptance, so the
                    # step keeps WMK's large radius of convergence.
                    true_energy_engaged = False
                    te_mode = self.true_energy_mode()
                    if te_mode:
                        U2_trial = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                        if _QED_TRUE_ENERGY_PROBE:
                            # always measure once per macro, then reuse the verdict
                            true_energy_engaged = (not probe_decided) or probe_engage
                        else:
                            true_energy_engaged = self.true_energy_should_engage(
                                U2_trial
                            )
                    if true_energy_engaged:
                        start1 = timer()
                        true_energy_trial = self.true_energy_at_trial(
                            U2_trial, eigenvecs
                        )
                        end1 = timer()
                        print("true energy transformation took", end1 - start1)
                        true_change = true_energy_trial - true_energy_current
                        # PROJ vs VAR, the pair Molpro prints every macroiteration.
                        print(
                            "true energy E(U)",
                            true_energy_trial,
                            "model energy E(T)",
                            current_energy + energy_change,
                            "PROJ-VAR gap",
                            (current_energy + energy_change) - true_energy_trial,
                            flush=True,
                        )
                        print(
                            "true energy change",
                            true_change,
                            "model energy change",
                            energy_change,
                            flush=True,
                        )
                        accept_by_true_energy = true_change < _QED_TRUE_ENERGY_TOL
                        if _QED_TRUE_ENERGY_PROBE and not probe_decided:
                            denom = abs(energy_change)
                            rel = (abs(true_change - energy_change) / denom
                                   if denom > 1e-14 else float("inf"))
                            probe_engage = rel > _QED_TRUE_ENERGY_PROBE_TOL
                            probe_decided = True
                            print(
                                "probe: rel model error", rel,
                                "tol", _QED_TRUE_ENERGY_PROBE_TOL,
                                "-> engage for this macroiteration", probe_engage,
                                flush=True,
                            )

                    if microiteration == 0 and orbital_optimization_step == 0:
                        convergence_threshold = min(
                            0.01 * gradient_norm, np.power(gradient_norm, 2)
                        )
                    # Mode 2 ENFORCES the true-energy test; mode 1 measures it and
                    # steers the radius with it but leaves acceptance alone, so a
                    # mode-1 run differs from the shipping default only through
                    # step_control.
                    # `and true_energy_engaged` is REQUIRED, not cosmetic: when
                    # the adaptive trigger declines, accept_by_true_energy is
                    # still None, and `None or hard_case == 2` would reject
                    # every step that is not a hard case.
                    if te_mode >= 2 and true_energy_engaged:
                        step_accepted = accept_by_true_energy or hard_case == 2
                    else:
                        step_accepted = energy_change < 0.0 or hard_case == 2
                    if step_accepted:
                        print("step norm", np.linalg.norm(step))
                        # QED_TRUE_ENERGY_REEXPAND forces QN off: the QN branch
                        # caches U_zero / A_tilde_zero / G_blocks_zero /
                        # reduced_hessian_diagonal_zero in the OLD basis at
                        # qn_count == 1, and re-expansion invalidates all four.
                        if (step_norm < 0.05 and not _QED_DISABLE_QN
                                and not _QED_TRUE_ENERGY_REEXPAND):
                            print("activate qn optimization")
                            qn_optimization = True
                            qn_count +=1
                            N_microiterations = 20  
                            N_orbital_optimization_steps= 1
                            print("number of microiteration left",N_microiterations, "number of orbital optimization", N_orbital_optimization_steps, "qn_count", qn_count)
                        if qn_count == 1:
                            self.U_zero = copy.deepcopy(self.U2) 
                            self.A_tilde_zero = copy.deepcopy(A_tilde2)
                            self.G_blocks_zero = copy.deepcopy(G_blocks)
                            self.reduced_hessian_diagonal_zero = copy.deepcopy(self.reduced_hessian_diagonal)
                        #else:
                        #    qn_optimization = False
                        # restart = False
                        # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                        # predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                        # print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                        self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                        # QED_TRUE_ENERGY_REEXPAND: move the expansion point to
                        # the orbitals just accepted, using the transform
                        # true_energy_at_trial already ran at exactly this
                        # rotation.  After this self.U2 is the identity, so the
                        # build_gradient / build_hessian_diagonal below evaluate
                        # the NEW A, G at T = 0, and the model increment for the
                        # next step is measured from here rather than from macro
                        # entry.  zero_energy / second_order_energy_change are
                        # reset to match, so the `current_energy = zero_energy +
                        # second_order_energy_change` at the bottom of this
                        # branch lands on the exact energy at the new point.
                        # true_energy_engaged: re-expansion INSTALLS the
                        # transform true_energy_at_trial produced, so it can
                        # only run on a step where that transform actually ran.
                        # te_mode >= 2 as well: under REACTIVE the machinery is
                        # dormant until armed, and re-expansion installs the
                        # transform only that path produces.
                        if (_QED_TRUE_ENERGY_REEXPAND and true_energy_engaged
                                and te_mode >= 2):
                            _t_re = timer()
                            G_blocks = self.reexpand_at_new_orbitals(A, G, eigenvecs)
                            zero_energy = true_energy_trial
                            second_order_energy_change = 0.0
                            true_energy_current = true_energy_trial
                            print(
                                "reexpanded model at accepted orbitals, E =",
                                true_energy_trial,
                                "took", timer() - _t_re,
                                flush=True,
                            )
                        if microiteration == 0 and orbital_optimization_step == 0:
                            # convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                            # convergence_threshold = 0.01 * gradient_norm
                            # Schedules N_microiterations / N_orbital_optimization_
                            # steps off the norm of the first ACCEPTED step -- the
                            # step that actually lowered the exact energy.  See the
                            # retired-knob note at the top of this file for why the
                            # first COMPUTED step is not a better measure.
                            schedule_norm = step_norm
                            if schedule_norm > 0.1:
                                N_microiterations = 5
                                N_orbital_optimization_steps = 4
                            elif schedule_norm <= 0.1 and schedule_norm > 0.05:
                                N_microiterations = 7
                                N_orbital_optimization_steps = 3
                            print("number of microiteration", N_microiterations, flush=True)
                            print(
                                "number of optimization steps",
                                N_orbital_optimization_steps,
                                flush=True,
                            )
                        orbital_optimization_step += 1
                        ratio = energy_change / predicted_energy2
                        print("compare model with actual energy change", ratio)
                        # QED_TRUE_ENERGY_RATIO: replace the model-over-model
                        # ratio with the textbook one, ACTUAL over PREDICTED
                        # reduction.  The quantity being divided is the only
                        # thing that changes -- step_control, _record_rho and
                        # every threshold in them are untouched.  Guard against
                        # a zero denominator, which the original could not hit
                        # (predicted_energy2 < 0 whenever a step was accepted)
                        # but which a true-energy accept can reach on a flat
                        # step.
                        # `and true_energy_engaged` IS A CORRECTNESS FIX, not a
                        # style change: true_change / true_energy_trial are only
                        # assigned inside the engaged branch, so under the
                        # ADAPTIVE / PROBE / REACTIVE paths this would otherwise
                        # read a STALE value from an earlier step and both
                        # mis-report the ratio and corrupt true_energy_current.
                        if te_mode and true_energy_engaged:
                            if predicted_energy2 != 0.0:
                                ratio = true_change / predicted_energy2
                            else:
                                ratio = 0.0
                            print("compare model with TRUE energy change", ratio)
                            true_energy_current = true_energy_trial
                            true_reject_streak = 0
                        # QED_ADAPTIVE_TOL: same number step_control already
                        # uses to size the radius now also sizes the subproblem
                        # exit tolerance.  No extra work, no extra matvecs.
                        self._record_rho(ratio)
                        trust_radius = self.step_control(ratio, trust_radius, step_norm)

                        #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                        # print(np.shape(gradient_tilde), flush = True)
                        if qn_count == 0:
                            start = timer()
                            self.build_gradient(
                                self.U2, A, G, gradient_tilde, A_tilde2, True,
                                G_blocks=G_blocks,
                            )
                            end = timer()
                            # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
                            #G1 = G.transpose(3, 1, 2, 0).reshape(
                            #    self.nmo * self.n_occupied, self.nmo * self.n_occupied
                            #)
                            print("build gradient took", end - start)

                            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
                            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

                            # hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
                            # reduced_hessian_diagonal = np.zeros(self.index_map_size)
                            # index_count1 = 0
                            # for k in range(self.n_occupied):
                            #    for r in range(k+1,self.nmo):
                            #        if (k < self.n_in_a and r < self.n_in_a): continue
                            #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
                            #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
                            #        index_count1 += 1
                            start = timer()
                            self.build_hessian_diagonal(self.U2, G, A_tilde2)
                            end = timer()
                            print("build hessian diagonal took", end - start)
                            # print("diagonal elements of the reduced hessian")
                            # for i in range(self.index_map_size):
                            #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
                            #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
                            # reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
                            _mp0 = _mp_start()
                            reduced_gradient = np.zeros(self.index_map_size)
                            index_count1 = 0
                            np.set_printoptions(precision=14)
                            for k in range(self.n_occupied):
                                for r in range(k + 1, self.nmo):
                                    if k < self.n_in_a and r < self.n_in_a:
                                        continue
                                    if (
                                        self.n_in_a <= k < self.n_occupied
                                        and self.n_in_a <= r < self.n_occupied
                                    ):
                                        continue
                                    # QED_ORBITAL_SYMMETRY: skip rotations that mix symmetry species.
                                    if (
                                        _QED_ORBITAL_SYMMETRY
                                        and self._sym_labels is not None
                                        and self._sym_labels[r] != self._sym_labels[k]
                                    ):
                                        continue
                                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                                    
                                    index_count1 += 1
                            _mp_end("redgrad_loop_inner", _mp0)
                            _mp0 = _mp_start()
                            print("reduced_gradient",reduced_gradient, flush = True)
                            _mp_end("redgrad_debug_print", _mp0)

                            

                        # NOT reusable from true_energy_at_trial: that returns
                        # an ENERGY from a full transformation, while the CI
                        # needs the second-order-updated ACTIVE integrals this
                        # call produces.  Two different objects; both are needed.
                        active_twoeint[:, :, :, :] = 0.0
                        active_fock_core[:, :] = 0.0
                        d_cmo[:, :] = 0.0
                        start1 = timer()
                        self.microiteration_ci_integrals_transform(
                            self.U2, eigenvecs, d_cmo, active_fock_core, active_twoeint
                        )
                        end1 = timer()
                        print("second order integral transformation took", end1 - start1)
                        count += 1
                        
                        current_energy = zero_energy + second_order_energy_change
                        hard_case = 0

                    else:
                        # restart = False
                        # guess_vector = trial_vector
                        reduce_step = 1
                        # QED_STEP_CONTROL (b): tie the shrink to the REJECTED STEP, not to
                        # Delta itself -- otherwise an interior step is returned unchanged.
                        if _QED_STEP_CONTROL:
                            trust_radius = 0.5 * step_norm
                        else:
                            trust_radius = 0.5 * trust_radius
                        print("Reject step, restart", flush=True)
                        print("new trust radius", trust_radius, flush=True)
                        # QED_TRUE_ENERGY_RATIO mode 2: bound the retry.  See
                        # the knob note at the top of this file -- under the
                        # true-energy test a stationary point rejects every
                        # step, and the shipping loop has no way out.
                        if te_mode >= 2 and true_energy_engaged:
                            true_reject_streak += 1
                            if (
                                true_reject_streak >= _QED_TRUE_ENERGY_MAX_REJECT
                                or trust_radius < _QED_TRUE_ENERGY_MIN_RADIUS
                            ):
                                print(
                                    "true-energy test rejected",
                                    true_reject_streak,
                                    "consecutive steps at radius",
                                    trust_radius,
                                    "-- giving up this microiteration",
                                    flush=True,
                                )
                                true_energy_gave_up = True
                                break
                # if convergence == 1:
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

            if (convergence == 1 or true_energy_gave_up) and count == 0:
                # `or true_energy_gave_up`: a mode-2 microiteration that applied
                # no step leaves the local integrals belonging to the CURRENT
                # self.U2 (or, in the first microiteration, still zero), so the
                # same restore the convergence exit uses is what the CI needs.
                active_twoeint[:, :, :, :] = self.active_twoeint[:, :, :, :]
                active_fock_core[:, :] = self.active_fock_core[:, :]
                d_cmo[:, :] = self.d_cmo[:, :]
                self.E_core2 = self.E_core
                

            start = timer()
            _mp0_ci = _mp_start()
            occupied_J = np.zeros(
                (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
            )
            occupied_J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ] = copy.deepcopy(active_twoeint)
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ] = copy.deepcopy(active_fock_core)
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied, : self.n_occupied])
            gkl2 = copy.deepcopy(active_fock_core)
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint)

            occupied_J = occupied_J.reshape(
                self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
            )

            _mp_end("ci_prep_arrays", _mp0_ci)
            _mp0_hd = _mp_start()
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
                self.target_spin,
            )
            _mp_end("ci_prep_Hdiag", _mp0_hd)
            # QED_FRESH_HDIAG_SORT -- see the knob comment at the top of this
            # file.  H_diag3 has just been rebuilt from the CURRENT orbitals;
            # without this, index_Hdiag still holds the SETUP argsort (:3693 /
            # :3900) and the spin-coupling closure at ci_solver.c:2242 walks the
            # current diagonal in a stale order.  Only the cold path reads it
            # (casscf == true takes indim = nroots and never gets there), so
            # this is a no-op for warm/production runs.
            # This is the LIVE site (microiteration_optimization6); the
            # byte-identical block in microiteration_optimization5 was dead code
            # (no callers) and is deliberately left untouched, same convention
            # as the constint[8] block below.
            if _QED_FRESH_HDIAG_SORT:
                _fresh = np.asarray(self.H_diag3.argsort(), dtype=np.int32)
                if _QED_FRESH_HDIAG_SORT > 1:
                    _n = min(4000, len(_fresh))
                    _ov = np.intersect1d(_fresh[:_n], self.index_Hdiag[:_n]).size
                    _pos = int((_fresh[:_n] == self.index_Hdiag[:_n]).sum())
                    print(f"[hdiag-sort] lowest-{_n}: stale-vs-fresh set overlap "
                          f"{_ov}/{_n} ({100.0*_ov/_n:.1f}%), identical positions {_pos}",
                          flush=True)
                self.index_Hdiag = _fresh
            print("qn status", qn_optimization, "qn count", qn_count, flush = True)
            print("gradient norm", np.linalg.norm(reduced_gradient), flush = True)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[: self.n_in_a, : self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9
            self.constdouble[5] = self.E_core2
            # Experiment knobs, default-no-op -- see the definitions at the
            # top of this file. This is the LIVE coupled-CI site
            # (microiteration_optimization6); the byte-identical block in
            # microiteration_optimization5 was dead code (no callers) and was
            # deliberately left untouched.
            self.constint[8] = _QED_CI_UNCOUPLED_MAXITER
            if qn_count > 0:
                self.constdouble[4] = _QED_CI_COUPLED_FACTOR * np.linalg.norm(reduced_gradient)
                # QED_CI_COUPLED_MAXITER, default 10000 = the literal constant
                # that stood here before, so the default path is unchanged.
                # Molpro's equivalent budget is 2 (see the knob comment at the
                # top of this file); threshold and budget race, first wins.
                self.constint[8] = _QED_CI_COUPLED_MAXITER
            print("number of CI iteration", self.constint[8])
            eigenvals = np.zeros((self.davidson_roots))
            # eigenvecs = np.zeros((self.davidson_roots, H_dim))
            # eigenvecs[:,:] = 0.0
            # print("heyhey5", eigenvecs)
            # --- cold-start-microiteration experiment (QED_COLD_START_MICRO) ---
            # Default off => warm start (casscf=True, Q = previous micro's CI
            # vector).  When on => cold start (casscf=False): a fresh Algorithm-3
            # spin-adapted guess with a configurable large subspace
            # (QED_COLD_START_MICRO_INDIM / _MAXDIM), then the (still constint[8]-
            # capped, e.g. 5) Davidson refinement -- tests whether a fresh
            # large-subspace guess gives a smaller residual after the fixed cap
            # than the stale warm start.  constint[5]/[6] restored right after.
            # Independent of the macro-level QED_COLD_START_CI toggle.
            _cold_micro = int(os.environ.get("QED_COLD_START_MICRO", "0"))
            if _cold_micro:
                _svm_indim = int(self.constint[5])
                _svm_maxdim = int(self.constint[6])
                self.constint[5] = int(os.environ.get("QED_COLD_START_MICRO_INDIM", str(_svm_indim)))
                self.constint[6] = int(os.environ.get("QED_COLD_START_MICRO_MAXDIM", str(_svm_maxdim)))
            _mp_end("ci_prep", _mp0_ci)
            _mp0_gr = _mp_start()
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
                (not _cold_micro),
                self.target_spin,
            )
            _mp_end("c_get_roots", _mp0_gr)
            if _cold_micro:
                self.constint[5] = _svm_indim
                self.constint[6] = _svm_maxdim
            self._dump_ci_vectors(eigenvecs, "micro")
            end = timer()
            print("CI step took", end - start)

            # print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print(
                "microiteration",
                microiteration + 1,
                "current average energy",
                avg_energy,
                flush=True,
            )
            #current_energy = avg_energy

            start = timer()
            self.D_tu_avg_old = copy.deepcopy(self.D_tu_avg)
            self.build_state_average_rdms(eigenvecs)
            self.density_norm_change = np.linalg.norm(self.D_tu_avg - self.D_tu_avg_old)
            print("density norm change", self.density_norm_change)
            if qn_count >0:
                if qn_count ==1 or self.density_norm_change <=0.025 or energy_change < 0:
                    self.U222 = self.U_zero              
                    print("ahihi")
            end = timer()
            print("building RDM took", end - start)
            

            microiteration += 1











    # ah_orbital_optimization REMOVED 2026-09-04 (developer's call).
    # It was the PROTOTYPE for the reactive true-energy strategy
    # (QED_TRUE_ENERGY_REACTIVE), which has since been adapted into
    # microiteration_optimization6 -- see the h4 24000 A case, item 375,
    # where the reactive knob converges the deck in 24 macroiterations.
    # Its only call site was already commented out, and its subproblem
    # solver Davidson_augmented_hessian_solve3 had been deleted earlier, so
    # it could not run at all.  1638 lines, including a fourth copy of the
    # LSTRS beta-bisection and its `exit()`.  See git history.

    def bfgs_orbital_optimization(self, eigenvecs, c_get_roots):
        self.eigenvecs = eigenvecs
        self.H1 = copy.deepcopy(self.H_spatial2)
        self.d_cmo1 = copy.deepcopy(self.d_cmo)
        # print("E_core", self.E_core)
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
        # A2 = np.zeros((rot_dim, rot_dim))
        # G2 = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        # print(eigenvecs)
        self.reduced_hessian_diagonal = np.zeros(self.index_map_size)

        active_twoeint = np.zeros(
            (self.n_act_orb, self.n_act_orb, self.n_act_orb, self.n_act_orb)
        )
        active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
        d_cmo = np.zeros((self.nmo, self.nmo))

        gradient_tilde = np.zeros((rot_dim, self.n_occupied))
        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, rot_dim, rot_dim))
        gradient_tilde2 = np.zeros((rot_dim, self.n_occupied))
        A_tilde2 = np.zeros((rot_dim, rot_dim))

        davidson_step = np.zeros((1, self.index_map_size))
        guess_vector = np.zeros((1, self.index_map_size + 1))
        # convergence_threshold = 1e-4
        convergence = 0
        # while(True):
        macroiteration_energy_initial = 0.0
        macroiteration_energy_current = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 5000
        microiteration = 0
        x00 = np.zeros(self.index_map_size)
        while microiteration < N_microiterations:
            print("\n")
            print("\n")
            print("MACROITERATION", microiteration + 1, flush=True)

            # while(microiteration < 2):
            # trust_radius = 0.35
            # A[:,:] = 0.0
            # G[:,:,:,:] = 0.0
            # start = timer()
            # self.build_intermediates(eigenvecs, A, G, True)
            # end   = timer()
            # print("build intermediates took", end - start,flush = True)

            start = timer()
            start1 = timer()
            macroiteration_energy_initial = macroiteration_energy_current

            macroiteration_energy_current = self.rdm_exact_energy(
                self.J, self.K, self.H_spatial2, self.d_cmo, self.eigenvecs
            )

            end = timer()
            print("check initial energy took", end - start, flush=True)

            # self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)

            # start = timer()
            # self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            # G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            # end   = timer()
            ##self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            #

            #
            # print("build gradient took", end - start,flush = True)
            # hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            # hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            ##hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            # reduced_hessian_diagonal = np.zeros(self.index_map_size)
            # start = timer()
            # self.build_hessian_diagonal(self.U2, G, A_tilde2)
            # end   = timer()
            # print("build hessian diagonal took", end - start,flush = True)
            # reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            # reduced_gradient = np.zeros(self.index_map_size)
            # print(np.shape(reduced_gradient), np.shape(gradient_tilde))
            # index_count1 = 0
            # for k in range(self.n_occupied):
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
            # print("reduced_gradient",reduced_gradient, flush = True)
            ####np.set_printoptions(precision = 14)
            ####mu1, w1 = np.linalg.eigh(reduced_hessian)
            ####print("eigenvalue of the reduced hessian", mu1,flush = True)
            ####print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]),flush = True)
            ####print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]),flush = True)
            ####H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            ####step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            ####print("norm of critical step", np.linalg.norm(step_limit),flush = True)
            ####print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit),flush = True)
            # gradient_norm = np.linalg.norm(reduced_gradient)
            # print("gradient norm", gradient_norm,flush = True)
            ##if microiteration ==0:
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            # if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break
            print(
                "LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION", flush=True
            )
            print(
                macroiteration_energy_initial, macroiteration_energy_current, flush=True
            )
            print("initial convergence threshold", convergence_threshold, flush=True)
            if (
                np.abs(macroiteration_energy_current - macroiteration_energy_initial)
                < 1e-9
            ) and microiteration >= 2:
                # if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)", flush=True)
                # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                break

            # print("eigenvec", self.eigenvecs,flush = True)
            x0 = np.zeros(self.index_map_size)
            x0[:] = x00
            res = scipy.optimize.minimize(
                self.energy_function,
                x0,
                method="BFGS",
                jac=self.energy_grad,
                options={"disp": True},
            )
            step = res.x
            # print(res.x,flush = True)
            step_norm = np.linalg.norm(step)
            x00 = res.x

            start = timer()
            Rai = np.zeros((self.n_act_orb, self.n_in_a))
            Rvi = np.zeros((self.n_virtual, self.n_in_a))
            Rva = np.zeros((self.n_virtual, self.n_act_orb))
            for i in range(self.index_map_size):
                s = self.index_map[i][0]
                l = self.index_map[i][1]
                if s >= self.n_in_a and s < self.n_occupied and l < self.n_in_a:
                    Rai[s - self.n_in_a][l] = step[i]
                elif s >= self.n_occupied and l < self.n_in_a:
                    Rvi[s - self.n_occupied][l] = step[i]
                else:
                    Rva[s - self.n_occupied][l - self.n_in_a] = step[i]

            self.build_unitary_matrix(Rai, Rvi, Rva)
            # print("jnti",self.E_core)

            U_temp = self.U_delta
            J_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
            K_temp = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

            K_temp = np.ascontiguousarray(K_temp)
            self._macro_JK(U_temp, J_temp, K_temp)
            # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

            temp8 = np.zeros((self.nmo, self.nmo))
            temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
            h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
            temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
            d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

            new_energy = self.rdm_exact_energy(
                J_temp, K_temp, h1_temp, d_cmo_temp, eigenvecs
            )

            end = timer()
            print(
                "build unitary matrix and recheck energy took", end - start, flush=True
            )
            self.J[:, :, :, :] = J_temp
            self.K[:, :, :, :] = K_temp
            self.H_spatial2[:, :] = h1_temp
            self.d_cmo[:, :] = d_cmo_temp
            # self.full_transformation_macroiteration(self.U_total, self.J, self.K)
            # temp_energy = self.rdm_exact_energy(self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs)
            # print("check energy again",temp_energy,flush = True)
            A[:, :] = 0
            A_tilde2[:, :] = 0
            # G[:,:,:,:] = 0
            # gradient_tilde[:] = 0
            # hessian_tilde[:,:,:,:] = 0
            ###self.build_intermediates(eigenvecs, A, G, True)
            ###self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
            ###self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            ###end   = timer()
            ####self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
            ###G1 = G.transpose(3,1,2,0).reshape(self.nmo*self.n_occupied,self.nmo*self.n_occupied)
            ###print("build gradient took", end - start,flush = True)

            ###hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            ###hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            # hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            # reduced_hessian_diagonal = np.zeros(self.index_map_size)
            # index_count1 = 0
            # for k in range(self.n_occupied):
            #    for r in range(k+1,self.nmo):
            #        if (k < self.n_in_a and r < self.n_in_a): continue
            #        if (self.n_in_a <= k < self.n_occupied and self.n_in_a <= r < self.n_occupied): continue
            #        reduced_hessian_diagonal[index_count1] = hessian_diagonal3[r][k]
            #        index_count1 += 1
            # start = timer()
            # self.build_hessian_diagonal(self.U2, G, A_tilde2)
            # end   = timer()
            # print("build hessian diagonal took", end - start,flush = True)
            # print("diagonal elements of the reduced hessian")
            # for i in range(self.index_map_size):
            #    aa = self.reduced_hessian_diagonal[i] - reduced_hessian_diagonal[i]
            #    if np.abs(aa) > 1e-12: print("ERROR TOO LARGE")
            # reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
            # reduced_gradient[:] = 0
            # reduced_hessian[:,:] = 0
            # index_count1 = 0
            # np.set_printoptions(precision = 14)
            # for k in range(self.n_occupied):
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
            # mu1, w1 = np.linalg.eigh(reduced_hessian)
            # print("eigenvalue of the reduced hessian", mu1,flush = True)
            # print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]),flush = True)
            # print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]),flush = True)
            # H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            # step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            # print("norm of critical step", np.linalg.norm(step_limit),flush = True)
            # print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit),flush = True)
            # print("gradient norm", np.linalg.norm(reduced_gradient),flush = True)

            start = timer()
            active_twoeint = self.J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ]
            fock_core = copy.deepcopy(self.H_spatial2)
            fock_core += 2.0 * np.einsum(
                "jjrs->rs", self.J[: self.n_in_a, : self.n_in_a, :, :]
            )
            fock_core -= np.einsum(
                "jjrs->rs", self.K[: self.n_in_a, : self.n_in_a, :, :]
            )

            E_core = 0.0
            E_core += np.einsum("jj->", self.H_spatial2[: self.n_in_a, : self.n_in_a])
            E_core += np.einsum("jj->", fock_core[: self.n_in_a, : self.n_in_a])

            active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
            active_fock_core[:, :] = fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ]
            d_cmo = self.d_cmo
            occupied_J = np.zeros(
                (self.n_occupied, self.n_occupied, self.n_occupied, self.n_occupied)
            )
            occupied_J[
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
                self.n_in_a : self.n_occupied,
            ] = copy.deepcopy(active_twoeint)
            self.H_diag3 = np.zeros(H_dim)
            occupied_fock_core = np.zeros((self.n_occupied, self.n_occupied))
            occupied_fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ] = copy.deepcopy(active_fock_core)
            occupied_d_cmo = np.zeros((self.n_occupied, self.n_occupied))
            occupied_d_cmo = copy.deepcopy(d_cmo[: self.n_occupied, : self.n_occupied])
            gkl2 = copy.deepcopy(active_fock_core)
            gkl2 -= 0.5 * np.einsum("kjjl->kl", active_twoeint)
            # print("recheck energy", flush = True)
            # active_one_e_energy = np.dot(occupied_fock_core[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tu_avg)
            # active_two_e_energy = 0.5 * np.dot(occupied_J[self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied,self.n_in_a: self.n_occupied].flatten(), self.D_tuvw_avg)
            # active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(occupied_d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            # ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            # sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
            #        self.Enuc + self.d_c + ci_dependent_energy)
            # print("sum_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            # print("gfhgy",
            #    "{:20.12f}".format(sum_energy),
            #    "{:20.12f}".format(active_one_e_energy),
            #    "{:20.12f}".format(active_two_e_energy),
            #    "{:20.12f}".format(self.E_core2),
            #    "{:20.12f}".format(active_one_pe_energy),
            #    "{:20.12f}".format(ci_dependent_energy),
            #    "{:20.12f}".format(self.Enuc),
            #    flush = True
            # )
            occupied_J = occupied_J.reshape(
                self.n_occupied * self.n_occupied, self.n_occupied * self.n_occupied
            )

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
                self.target_spin,
            )
            # print(self.H_diag3,flush = True)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[: self.n_in_a, : self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9
            self.constdouble[5] = E_core
            self.constint[8] = 4
            eigenvals = np.zeros((self.davidson_roots))
            # eigenvecs = np.zeros((self.davidson_roots, H_dim))
            # eigenvecs[:,:] = 0.0
            # print("heyhey5", eigenvecs)
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
            end = timer()
            print("CI step took", end - start, flush=True)

            # print("current residual", self.constdouble[4])
            current_residual = self.constdouble[4]
            avg_energy = 0.0
            for i in range(self.davidson_roots):
                avg_energy += self.weight[i] * eigenvals[i]
            print(
                "microiteration",
                microiteration + 1,
                "current average energy",
                avg_energy,
                flush=True,
            )
            current_energy = avg_energy

            start = timer()
            self.build_state_average_rdms(eigenvecs)
            end = timer()
            print("building RDM took", end - start, flush=True)

            current_energy2 = self.rdm_exact_energy(
                self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs
            )

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

            print(
                "ci energy", current_energy, "rdm_energy", current_energy2, flush=True
            )
            ###print("current gradient_norm and residual", gradient_norm, current_residual,flush = True)
            ###print("current convergence_threshold", convergence_threshold,flush = True)
            ###total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2))
            ###self.eigenvecs = eigenvecs
            ###print("eigenvec", self.eigenvecs,flush = True)
            # macroiteration_energy_current = current_energy2
            # if total_norm < convergence_threshold:
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
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.build_hessian_diagonal, and the C++ port cites it).
        """
        return _trs_build_hessian_diagonal(
            self,
            U,
            G,
            A_tilde,
        )
        # end = timer()
        # print("hessian diagonal step 5 took", end - start)
    
    def _get_preconditioned_w(self, w_unconv, theta_unconv, H_diag, full_eigvecs, converged_indices, U, A_tilde, G_blocks, reduced_gradient, alpha, reduced_hessian_diagonal):
        
        # 1. Run diagnostics to decide on strategy
        n_negative = np.sum(reduced_hessian_diagonal < 0)
        min_diag_pos = reduced_hessian_diagonal[reduced_hessian_diagonal > 0]
        min_diag = np.min(min_diag_pos) if len(min_diag_pos) > 0 else 1.0
        max_diag = np.max(reduced_hessian_diagonal)
        condition_est = max_diag / abs(min_diag) if abs(min_diag) > 1e-10 else 1e12
        ILL_CONDITION_THRESHOLD = 50000 
        
        preconditioned_w = np.zeros_like(w_unconv)
        H_dim = H_diag.shape[0] # Get full H_dim
        
        if n_negative > 0 or condition_est > ILL_CONDITION_THRESHOLD:
            # --- CASE 1: ILL-CONDITIONED ---
            # Use the fast, cheap, and effective DIAGONAL preconditioner
            # print("LSTRS: (Precond) Ill-conditioned, using DIAGONAL.") # Optional debug
            precon_denom = theta_unconv.reshape(
                len(theta_unconv), 1
            ) - H_diag.reshape(1, H_dim)
            
            preconditioned_w = np.divide(
                w_unconv,
                precon_denom,
                out=np.zeros_like(w_unconv),
                where=precon_denom != 0,
            )
            
        else:
            # --- CASE 2: WELL-CONDITIONED (LSTRS HARD CASE) ---
            # Use the robust, but more expensive, JACOBI-DAVIDSON (PCG) solver
            # This will solve the 30+ iteration stall.
            # print("LSTRS: (Precond) Well-conditioned, using JACOBI-DAVIDSON.") # Optional debug
            
            Q_conv = full_eigvecs[converged_indices, :]
            
            for i in range(len(theta_unconv)):
                current_residual = w_unconv[i, :]
                current_theta = theta_unconv[i]
                
                P_diag = H_diag - current_theta
                # Add safety for zero diagonals
                P_diag[np.abs(P_diag) < 1e-12] = 1.0
            
                v_corr = self.inner_solve_pcg(
                    current_residual,
                    current_theta,
                    Q_conv,
                    P_diag,
                    U, A_tilde, G_blocks, reduced_gradient, alpha
                )
                preconditioned_w[i, :] = v_corr
                
        return preconditioned_w   



    def inner_solve_pcg(self, residual_k, theta_k, Q_conv, P_diag, U, A_tilde, G_blocks, reduced_gradient, alpha):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.inner_solve_pcg, and the C++ port cites it).
        """
        return _trs_inner_solve_pcg(
            self,
            residual_k,
            theta_k,
            Q_conv,
            P_diag,
            U,
            A_tilde,
            G_blocks,
            reduced_gradient,
            alpha,
        )


    def check_root_0(self, aug_hessian_eigenvecs, aug_hessian_eigenvals, reduced_gradient, trust_radius):
        epsilon_v = 1e-4
        mu = aug_hessian_eigenvals
        w = np.copy(aug_hessian_eigenvecs.T)
        v1 = w[0, 0] / np.linalg.norm(w[:, 0])
        u1 = w[1:, 0] / np.linalg.norm(w[:, 0])
        aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
        bb1 = np.sqrt(1 - v1 * v1)
        if (mu > -1e-8 and np.linalg.norm(
            u1
        ) < trust_radius * np.abs(v1)) or aa1 > epsilon_v * bb1:
            print("use first root")
            return True
        else:
            print("use second root")
            return False


    

    def aug_matvec(self, U, A_tilde, G_blocks, reduced_gradient, vector, alpha, H_dim):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.aug_matvec, and the C++ port cites it).
        """
        return _trs_aug_matvec(
            self,
            U,
            A_tilde,
            G_blocks,
            reduced_gradient,
            vector,
            alpha,
            H_dim,
        )
    def modified_gram_schmidt(self, V, new_vec):
        """
        Orthonormalize new_vec against basis V.
        """
        for i in range(len(V)):
            proj = np.dot(V[i], new_vec)
            new_vec -= proj * V[i]
        
        norm = np.linalg.norm(new_vec)
        if norm < 1e-12:
            return None # Linear dependence
        return new_vec / norm
    
    def generate_krylov_guess(self, U, A_tilde, G_blocks, reduced_gradient, alpha, H_dim, k=5):
        """
        Generates a guess subspace based on the Krylov sequence K_l(B_alpha, e1).
        
        Rationale: If H_pp is positive (convex), the shifted Augmented Hessian 
        contains the most relevant curvature information.
        """
        # Start with e1 (which corresponds to index 0 in Augmented Matrix 0-based indexing)
        start_vec = np.zeros(H_dim)
        start_vec[0] = 1.0 
        
        guess_basis = np.array([start_vec])
 
        current_v = start_vec.copy()
        
        # Generate k-1 additional vectors
        for i in range(k - 1):
            # Apply B(alpha)
            # This uses the specific structure: | alpha g^T |
            #                                 | g     H   |
            w = self.aug_matvec(U, A_tilde, G_blocks, reduced_gradient, current_v, alpha, H_dim)
            
            # Orthonormalize against existing guess basis
            w_ortho = self.modified_gram_schmidt(guess_basis, w)
            
            if w_ortho is None:
                # Subspace exhausted early
                break
            guess_basis = np.vstack((guess_basis, w_ortho))        
            current_v = w_ortho
            
        return guess_basis
    
    def generate_gradient_guess(self, U, A_tilde, G_blocks, reduced_gradient, hessian_diagonal, alpha, H_dim):
        """
        Vectors:
        1. e0: Unit vector in augmentation dimension (1, 0...)
        2. b1: (0, g/|g|) - Normalized gradient direction
        3. b2: (0, -g - H*(g/|g|)) - Curvature/Residual direction
        4. b3: (0, e_min) - Unit vector at the lowest Hessian diagonal index
        """
        print("Generating robust 4-vector guess (Hoyvik et al.)...")

        # 1. First Vector: e0 (Augmentation unit)
        v0 = np.zeros(H_dim)
        v0[0] = 1.0

        # Initialize basis matrix
        guess_basis = np.array([v0])

        # Prepare for Vector 2
        g_norm = np.linalg.norm(reduced_gradient)
        if g_norm < 1e-12:
            # If gradient is zero, we are at a stationary point.
            # Just return e0 and e_min_diag.
            print("Gradient is zero, skipping gradient-based guess vectors.")
        else:
            g_unit = reduced_gradient/ g_norm

            # 2. Second Vector: (0, g_unit)
            v1 = np.zeros(H_dim)
            v1[1:] = g_unit

            # Orthogonalize and add
            v1_ortho = self.modified_gram_schmidt(guess_basis, v1)
            if v1_ortho is not None:
                guess_basis = np.vstack((guess_basis, v1_ortho))

            # 3. Third Vector: (0, -g - H*g_unit)
            # This captures the immediate curvature along the gradient
            g_unit = g_unit.reshape(1, H_dim -1)
            Hg_unit = np.zeros_like(g_unit)
            self.orbital_sigma3(U, A_tilde, G_blocks, g_unit, Hg_unit, 1, 0) # Assumes 'self' is available
            residual_core = -reduced_gradient - Hg_unit

            v2 = np.zeros(H_dim)
            v2[1:] = residual_core

            # Orthogonalize and add
            v2_ortho = self.modified_gram_schmidt(guess_basis, v2)
            if v2_ortho is not None:
                guess_basis = np.vstack((guess_basis, v2_ortho))

        # 4. Fourth Vector: (0, e_min)
        # Unit vector corresponding to the lowest diagonal element of H
        # This is critical for finding negative curvature not aligned with gradient.
        min_idx = np.argmin(hessian_diagonal)
        v3 = np.zeros(H_dim)
        v3[min_idx + 1] = 1.0 # +1 because index 0 is augmentation dim

        # Orthogonalize and add
        v3_ortho = self.modified_gram_schmidt(guess_basis, v3)
        if v3_ortho is not None:
            guess_basis = np.vstack((guess_basis, v3_ortho))
        
        v4 = np.zeros(H_dim)
        v4[1:] = np.random.rand(H_dim-1)

        # Orthogonalize and add
        v4_ortho = self.modified_gram_schmidt(guess_basis, v4)
        if v4_ortho is not None:
            guess_basis = np.vstack((guess_basis, v4_ortho))
        
        v5 = np.zeros(H_dim)
        v5[1:] = np.random.rand(H_dim-1)

        # Orthogonalize and add
        v5_ortho = self.modified_gram_schmidt(guess_basis, v5)
        if v5_ortho is not None:
            guess_basis = np.vstack((guess_basis, v5_ortho))



        return guess_basis
        
    def generate_bagel_guess(self, reduced_gradient, hessian_diagonal, H_dim, shift=1e-3):
        """
        Generates the initial guess used in BAGEL (cassecond.cc).
        
        Strategy:
        1. e0: Unit vector in augmentation dimension.
        2. v1: Preconditioned gradient v = g / (h_diag + shift).
        """
        print(f"Generating BAGEL-style guess (Preconditioned Gradient, shift={shift})...")
        
        # 1. First Vector: e0 (Augmentation unit)
        v0 = np.zeros(H_dim)
        v0[0] = 1.0
        
        guess_basis = np.array([v0])
        
        # 2. Second Vector: Preconditioned Gradient
        # v_i = g_i / (D_ii + shift)
        
        # Avoid division by zero or extremely small numbers
        denom = hessian_diagonal + shift
        
        # Handle small denominators (optional safety, BAGEL does a check)
        # In python, we can just divide; if self.h_diag is reasonable, it's fine.
        # We assume h_diag are the diagonal elements.
        
        v1_part = np.zeros_like(reduced_gradient)
        
        # Element-wise division with safety check (mimicking apply_denom logic somewhat)
        mask = np.abs(denom) > 1e-12
        v1_part[mask] = reduced_gradient[mask] / denom[mask]
        # For small denominators, BAGEL leaves it as gradient (conceptually), 
        # but here we leave as 0 if denom is 0, or just use gradient if needed.
        # Let's fallback to gradient if denom is problematic, or just 0.
        v1_part[~mask] = reduced_gradient[~mask] 
        
        v1 = np.zeros(H_dim)
        v1[1:] = v1_part
        
        # Orthonormalize and add
        v1_ortho = self.modified_gram_schmidt(guess_basis, v1)
        if v1_ortho is not None:
            guess_basis = np.vstack((guess_basis, v1_ortho))
            
        return guess_basis




    # ---------------------------------------------------------------
    # print_matrix_nice and the Steihaug-Toint solvers
    # solve_pcg_trust_region / solve_pcg_with_operator MOVED to
    # legacy_python_ci.py on 2026-09-04 -- no live caller, only
    # commented-out references.  Kept rather than deleted because the
    # C++ port cites them as the source of PcgTrustRegionSolver.
    # ---------------------------------------------------------------

    def solve_gltr_with_operator(self, B_operator, g, M_diag, radius, max_iter=100, tol=0.0001, verbose=True):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.solve_gltr_with_operator, and the C++ port cites it).
        """
        return _trs_solve_gltr_with_operator(
            self,
            B_operator,
            g,
            M_diag,
            radius,
            max_iter,
            tol,
            verbose,
        )



    def _record_rho(self, ratio):
        """Log one model-fidelity observation for QED_ADAPTIVE_TOL.

        Called right where `ratio` is already computed for step_control, so it
        adds no work and no matvecs.  Kept unconditional and side-effect free
        when the knob is off (the list is simply never read).
        """
        if not hasattr(self, "_rho_history"):
            self._rho_history = []
        try:
            self._rho_history.append(float(ratio))
        except (TypeError, ValueError):
            return

    def _record_gltr_exit(self, status, gltr_site):
        """Log one GLTR termination for QED_SUBSPACE_TOL.

        Called immediately after each GLTR solve returns, with the status string
        that solve already produced -- no work, no matvecs.  Kept unconditional
        and side-effect free when the knob is off (the lists are never read).
        Histories are PER SITE: the three GLTR sites have very different exit
        mixes and pooling them would average away the signal.
        """
        if not hasattr(self, "_gltr_exit_history"):
            self._gltr_exit_history = {}
        self._gltr_exit_history.setdefault(gltr_site, []).append(status)
        # QED_EXIT_CONTROL freshness token.  step_control has FOUR call sites and
        # the main site can dispatch to LSTRS / OTR / AH instead of GLTR, so a
        # radius rule that simply read the last recorded exit would act on a
        # STALE one whenever a non-GLTR solver produced the step.  The token is
        # consumed by _take_gltr_exit(), so each exit can steer exactly one
        # radius update and a non-GLTR step steers none.
        self._gltr_exit_pending = gltr_site

    def _take_gltr_exit(self):
        """Consume the pending GLTR exit, if the current step came from one.

        Returns that site's exit history, or None when the step did not come
        from a GLTR solve -- in which case the caller must not adapt.
        """
        site = getattr(self, "_gltr_exit_pending", None)
        if site is None:
            return None
        self._gltr_exit_pending = None
        return getattr(self, "_gltr_exit_history", {}).get(site)

    def _subproblem_tol(self, site, gltr_site=None):
        """Exit tolerance for a trust-region subproblem solve at `site`.

        Returns the hard-coded 1e-7 unless QED_ADAPTIVE_TOL or QED_SUBSPACE_TOL
        selects this site, so with both knobs off every call site is
        byte-for-byte unchanged.  `site` is "main" or "qn" and selects by mode
        (1 adapts "main" only, 2 adapts both) for either controller.
        `gltr_site` is the FINE site name ("gltr" / "gltr_qn" / "gltr_qn_bfgs")
        and is what QED_SUBSPACE_TOL keys its per-site history on; it is
        optional so the signature stays compatible with any other caller.

        The two controllers are mutually exclusive by construction -- they drive
        the same number from contradictory inputs (rho vs exit reason), so
        running both at once would measure neither.  QED_SUBSPACE_TOL wins if
        both are set, and says so.
        """
        if _QED_SUBSPACE_TOL:
            if _QED_ADAPTIVE_TOL and not getattr(self, "_subspace_tol_warned", False):
                self._subspace_tol_warned = True
                print(
                    "  [QED_SUBSPACE_TOL] both tolerance controllers are enabled;"
                    " QED_ADAPTIVE_TOL is IGNORED for this run."
                )
            if site == "qn" and _QED_SUBSPACE_TOL < 2:
                return 1e-7
            key = gltr_site or site
            if not hasattr(self, "_subspace_tol_cur"):
                self._subspace_tol_cur = {}
            hist = getattr(self, "_gltr_exit_history", {}).get(key, [])
            cur = _subspace_tol_from_exits(hist, self._subspace_tol_cur.get(key))
            self._subspace_tol_cur[key] = cur
            if _QED_SOLVER_STATS:
                w = hist[-_QED_SUBSPACE_TOL_WINDOW:]
                frac = (
                    sum(_gltr_exit_is_interior(s) for s in w) / float(len(w))
                    if w
                    else float("nan")
                )
                print(
                    f"  subspace_tol[{key}]: n={len(hist)} frac_interior={frac:.3f}"
                    f" tol={cur:.3e}"
                )
            return cur
        if not _QED_ADAPTIVE_TOL:
            return 1e-7
        if site == "qn" and _QED_ADAPTIVE_TOL < 2:
            return 1e-7
        cur = getattr(self, "_adaptive_tol_cur", None)
        cur = _adaptive_tol_from_rho(getattr(self, "_rho_history", []), cur)
        self._adaptive_tol_cur = cur
        if _QED_SOLVER_STATS:
            h = getattr(self, "_rho_history", [])[-_QED_ADAPTIVE_TOL_WINDOW:]
            med = float(np.median([r for r in h if np.isfinite(r)])) if h else float("nan")
            print(f"  adaptive_tol[{site}]: median_rho={med:.4f} tol={cur:.3e}")
        return cur

    def _exact_dense_trust_region_step(self, U, A_tilde, G_blocks, gradient, trust_radius):
        """Materialize the reduced orbital Hessian densely, then solve the
        trust-region subproblem exactly (QED_EXACT_TR).

        Costs exactly n = index_map_size Hessian-vector products, pushed through
        orbital_sigma3 in batches of QED_EXACT_TR_BATCH.  The matvec COUNT is n
        regardless of batch size; batching only buys wall time, and QED_MV_BENCH
        shows per-call cost is nearly flat in the batch size.

        THE GAP THIS FILLS.  The main-site dispatch picks GLTR when the Hessian
        DIAGONAL has no negative entry (n_negative == 0) and the LSTRS/Davidson
        beta bisection otherwise.  That split is deliberate: the LSTRS initial
        guess is Kreplin's P-subspace construction, which keys off NEGATIVE
        diagonal entries, so with an all-positive diagonal its Davidson would
        need many more iterations -- LSTRS is used exactly where it has a guess.
        But a positive diagonal does not imply a positive definite Hessian.  On
        the MgH+ CAS(8,12) dumps, 4 of the 7 all-positive-diagonal subproblems
        are INDEFINITE (up to 5 negative eigenvalues, lam_min ~ -3.6e-4), and in
        those the lowest eigenvector is DELOCALIZED (participation ratio
        4.5-5.6) -- so there is no negative diagonal entry to seed from, and
        QED_AH_NEG_CURV's e_argmin(h_diag) seed is orthogonal to it (overlap
        <= 0.001).  Those subproblems reach GLTR, which has no hard-case branch,
        and the Krylov space built from g never contains v_min.  A dense exact
        solve needs no initial guess and no Krylov space at all, which is the
        only reason to prefer it here.

        Deliberately NOT a replacement for the LSTRS branch by default: where
        the diagonal does carry negative entries the lowest eigenvector is
        localized on a single coordinate (participation ratio 1.0), which is the
        configuration Kreplin's guess handles well.  QED_EXACT_TR=2 overrides
        that and takes the whole main site, for A/B purposes.
        """
        n = self.index_map_size
        batch = max(1, min(int(_QED_EXACT_TR_BATCH), n))
        H = np.zeros((n, n))
        col = 0
        while col < n:
            nb = min(batch, n - col)
            R = np.zeros((nb, n))
            for k in range(nb):
                R[k, col + k] = 1.0
            R = np.ascontiguousarray(R)
            S = np.zeros((nb, n))
            self.orbital_sigma3(U, A_tilde, G_blocks, R, S, nb, 0)
            # S[k, :] is H @ e_{col+k}, i.e. column col+k of H.
            H[:, col:col + nb] = S.T
            col += nb

        g = np.asarray(gradient, dtype=float).ravel()
        step, lam_shift, kind = _more_sorensen_dense(H, g, trust_radius)
        if _QED_SOLVER_STATS:
            asym = float(np.abs(H - H.T).max())
            print(
                f"  exact_tr: n={n} kind={kind} lambda={lam_shift:.6e} "
                f"|step|={np.linalg.norm(step):.6e} radius={trust_radius:.6e} "
                f"asym={asym:.2e}"
            )
        return step

    def solve_gltr_trust_region(self, U, A_tilde, G_blocks, g, M_diag, radius, max_iter=100, tol=0.0001, verbose=True):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.solve_gltr_trust_region, and the C++ port cites it).
        """
        return _trs_solve_gltr_trust_region(
            self,
            U,
            A_tilde,
            G_blocks,
            g,
            M_diag,
            radius,
            max_iter,
            tol,
            verbose,
        )




    def Davidson_augmented_hessian_solve6(
        self,
        U,
        A_tilde,
        G,
        G_blocks,
        reduced_hessian_diagonal,
        reduced_gradient,
        alpha,
        alpha_range,
        trust_radius,
        aug_hessian_eigenvecs,
        aug_hessian_eigenvals,
        guess_vector,
        restart
    ):
        """Augmented-Hessian Davidson.  MOVED to davidson_augmented_hessian.py.

        The 861-line body left this file on 2026-09-04; see that module for what
        the solver does, why it takes ``ctx`` rather than otr_trust_region.py's
        ``hess_x`` closure, and which generator attributes it carries across
        calls.  Nothing about the arithmetic changed -- the move is gated
        bit-exactly by validation/lstrs_refactor_gate.sh.

        The METHOD stays because the name is part of this class's interface:
        ``lstrs_trust_region.solve_lstrs_microiteration`` calls
        ``ctx.Davidson_augmented_hessian_solve6`` at three sites, the
        micro-profiler list at the bottom of this file patches it by name, and
        the C++ port cites it as the source of ``DavidsonAugmentedHessianSolver``
        (cpp_casscf/README.md).  Deleting it would break all three for no gain.
        """
        return davidson_augmented_hessian_solve(
            self,
            U,
            A_tilde,
            G,
            G_blocks,
            reduced_hessian_diagonal,
            reduced_gradient,
            alpha,
            alpha_range,
            trust_radius,
            aug_hessian_eigenvecs,
            aug_hessian_eigenvals,
            guess_vector,
            restart,
        )

    def scipy_minres_solve(self, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-06):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.scipy_minres_solve, and the C++ port cites it).
        """
        return _trs_scipy_minres_solve(
            self,
            U,
            A_tilde,
            G_blocks,
            reduced_gradient,
            denom,
            max_iter,
            conv_thresh,
        )

    def minres_solve(self, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-06):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.minres_solve, and the C++ port cites it).
        """
        return _trs_minres_solve(
            self,
            U,
            A_tilde,
            G_blocks,
            reduced_gradient,
            denom,
            max_iter,
            conv_thresh,
        )


    def linear_equation_solve(self, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-06):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.linear_equation_solve, and the C++ port cites it).
        """
        return _trs_linear_equation_solve(
            self,
            U,
            A_tilde,
            G_blocks,
            reduced_gradient,
            denom,
            max_iter,
            conv_thresh,
        )


    # ---------------------------------------------------------------
    # print_matrix_nice and the Steihaug-Toint solvers
    # solve_pcg_trust_region / solve_pcg_with_operator MOVED to
    # legacy_python_ci.py on 2026-09-04 -- no live caller, only
    # commented-out references.  Kept rather than deleted because the
    # C++ port cites them as the source of PcgTrustRegionSolver.
    # ---------------------------------------------------------------


    def build_orbital_hessian_element(self, U, sym_A_tilde, G, r, k, s, l):
        temp1 = np.copy(G[k, l, :, :])
        temp2 = np.copy(U[:, s])
        temp3 = np.copy(U[:, r])
        temp4 = np.zeros((self.nmo))
        temp4 = np.einsum("pq, q ->p", temp1, temp2)
        a = np.einsum("p, p ->", temp3, temp4)
        a -= 0.5 * (k == l) * (sym_A_tilde[r][s])
        a -= 0.5 * (r == s) * (sym_A_tilde[k][l])
        # print(np.shape(temp1))
        if r < self.n_occupied:
            temp1 = np.copy(G[r, l, :, :])
            temp3 = np.copy(U[:, k])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a -= np.einsum("p, p ->", temp3, temp4)
            a += 0.5 * (r == l) * (sym_A_tilde[k][s])
        if s < self.n_occupied:
            temp1 = np.copy(G[k, s, :, :])
            temp2 = np.copy(U[:, l])
            temp3 = np.copy(U[:, r])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a -= np.einsum("p, p ->", temp3, temp4)
            a += 0.5 * (k == s) * (sym_A_tilde[r][l])
        if r < self.n_occupied and s < self.n_occupied:
            temp1 = np.copy(G[r, s, :, :])
            temp2 = np.copy(U[:, l])
            temp3 = np.copy(U[:, k])
            temp4 = np.einsum("pq, q ->p", temp1, temp2)
            a += np.einsum("p, p ->", temp3, temp4)
        return a

    def build_orbital_hessian_guess(self, U, sym_A_tilde, reduced_gradient, G, guess_hessian, guess_gradient, dim1, idx):
        """Guess-subspace Hessian/gradient for the AH Davidson.  MOVED to orbital_sigma.py (2026-09-04).

        Kept as a method because the name is this class's interface: the
        solver modules call it as ctx.build_orbital_hessian_guess(...).  See
        orbital_sigma.py for the kernel and for why that module is a leaf.
        """
        return _os_build_orbital_hessian_guess(
            self,
            U,
            sym_A_tilde,
            reduced_gradient,
            G,
            guess_hessian,
            guess_gradient,
            dim1,
            idx,
        )

    # hessian_guess MOVED to orbital_sigma.py (2026-09-04) -- module-private
    # there: its only caller is in that module.
                # print(b, hessian_tilde[r*self.n_occupied+k][s*self.n_occupied+l])

    def microiteration_step(self, gradient_tilde, hessian_tilde, step, alpha, dim0):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))
        # print("gradient_tilde", gradient_tilde, flush = True)
        augmented_hessian[0, 1:] = gradient_tilde
        augmented_hessian[1:, 0] = gradient_tilde.T
        augmented_hessian[1:, 1:] = hessian_tilde
        # print("hessian")
        # self.printA(hessian_tilde)
        # print("augmented_hessian")
        # self.printA(augmented_hessian)
        # print(" ", flush = True)
        # U_a = np.zeros((dim0, dim0))
        # U_a[0][0] = 1
        # U_a[1:,1:] = self.w1
        # A_b = np.einsum("rp,rs,sq->pq", U_a, augmented_hessian, U_a)
        # mu2, w2 = np.linalg.eigh(A_b)
        # print("eigenvalues of transformed augmented hessian", mu2)
        # print("eigenvectors of transformed augmented hessian",w2)

        # print("augmented_hessian", augmented_hessian, flush = True)
        # print("transformed augmented hesssian", A_b)
        mu, w = np.linalg.eigh(augmented_hessian)

        idx = mu.argsort()[:dim0]
        # print(idx)
        np.set_printoptions(precision=14)
        print("check eigenvalues and eigenvectors of the augmented hessian", flush=True)
        print(mu[0], mu[1], flush=True)
        # print("w",w, flush = True)
        print("alpha", alpha, flush=True)
        scale = w[0][0]
        w[:, 0] = w[:, 0] / scale
        # print(w)
        step0 = w[1:, 0]
        # print(step0)
        step[:] = step0 / alpha
        # h_inverse = np.linalg.inv(hessian_tilde_ai)
        # step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai)
        step_norm = np.linalg.norm(step)
        # print("step", step)
        print("step norm", step_norm, flush=True)
        return step_norm

    def microiteration_step2(self, gradient_tilde, hessian_tilde, step, alpha, dim0):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))

        augmented_hessian[0, 1:] = gradient_tilde
        augmented_hessian[1:, 0] = gradient_tilde.T
        augmented_hessian[1:, 1:] = hessian_tilde
        # U_a = np.zeros((dim0, dim0))
        # U_a[0][0] = 1
        # U_a[1:,1:] = self.w1
        # A_b = np.einsum("rp,rs,sq->pq", U_a, augmented_hessian, U_a)
        # mu2, w2 = np.linalg.eigh(A_b)
        # print("eigenvalues of transformed augmented hessian", mu2)
        # print("eigenvectors of transformed augmented hessian",w2)

        # print("augmented_hessian", augmented_hessian)
        # print("transformed augmented hesssian", A_b)
        mu, w = np.linalg.eigh(augmented_hessian)
        idx = mu.argsort()[:dim0]
        # print(idx)

        np.set_printoptions(precision=14)
        print("check eigenvalues and eigenvectors of the augmented hessian")
        # print("w",w)
        print(mu, flush=True)
        scale = w[0][0]
        # w[:,0] = w[:,0]/scale
        # print(w)
        step0 = w[1:, 0]
        product = np.dot(gradient_tilde, step0.flatten())
        print("product of the gradient and the step from augmented hessian", product)
        # print(step0)
        step[:] = step0 / alpha
        # h_inverse = np.linalg.inv(hessian_tilde_ai)
        # step = - np.einsum("pq,q->p", h_inverse, gradient_tilde_ai)
        step_norm = np.dot(step.T, step)
        step_norm = np.sqrt(step_norm)
        # print("step norm", step_norm, flush=True)
        return step_norm, scale

    def projection_step(
        self, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0
    ):
        gradient_tilde = alpha * gradient_tilde
        augmented_hessian = np.zeros((dim0, dim0))
        # print("projected_gradient", gradient_tilde, flush = True)
        augmented_hessian[0, 1:] = gradient_tilde
        augmented_hessian[1:, 0] = gradient_tilde.T
        augmented_hessian[1:, 1:] = hessian_tilde
        # print("projected_hessian")
        # self.printA(hessian_tilde)
        # print("projected_augmented_hessian")
        # self.printA(augmented_hessian)
        # print(" ", flush = True)

        mu, aug_eigenvecs[:, :] = np.linalg.eigh(augmented_hessian)
        # np.set_printoptions(precision = 14)
        # print("check eigenvalues and eigenvectors of the projected augmented hessian", flush = True)
        # print("eigenvecs",aug_eigenvecs, flush = True)
        # print("alpha",alpha, flush = True)
        # print(mu, flush=True)
        return mu

    def projection_step2(self, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0):
        """MOVED to trust_region_solvers.py (2026-09-04).

        Delegator: the name is part of this class's interface (the solver
        modules call it as ctx.projection_step2, and the C++ port cites it).
        """
        return _trs_projection_step2(
            self,
            gradient_tilde,
            hessian_tilde,
            aug_eigenvecs,
            alpha,
            dim0,
        )

    def mv(self, U, A_tilde, G, R_reduced, num_states, pointer, eigval):
        R_reduced = R_reduced.reshape((1, self.index_map_size))
        sigma_reduced = np.zeros_like(R_reduced)
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw", num_states)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0]
                k = self.index_map[j][1]
                R_total[i][r][k] = R_reduced[i][j + pointer]

        A3_tilde = A_tilde + A_tilde.T

        hessian_tilde = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))

        hessian_tilde -= 0.5 * np.einsum(
            "kl,rs->klrs", np.eye(self.n_occupied), A3_tilde
        )
        hessian_tilde -= 0.5 * np.einsum(
            "rs,kl->klrs",
            np.eye(self.nmo),
            A3_tilde[: self.n_occupied, : self.n_occupied],
        )
        hessian_tilde[:, :, : self.n_occupied, :] += 0.5 * np.einsum(
            "rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[: self.n_occupied, :]
        )
        hessian_tilde[:, :, :, : self.n_occupied] += 0.5 * np.einsum(
            "ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:, : self.n_occupied]
        )

        sigma_total = np.zeros((num_states, self.nmo, self.n_occupied))

        temp1 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp2 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,isl->iql", U, R_total)
        temp2 = np.einsum("klpq,iql->ipk", G, temp1)
        sigma_total = np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:, : self.n_occupied, :] -= np.einsum(
            "ipr,pk->irk", temp2, U[:, : self.n_occupied]
        )

        temp1 = np.einsum(
            "ql,isl->iqs", U[:, : self.n_occupied], R_total[:, : self.n_occupied, :]
        )
        temp2 = np.einsum("kspq,iqs->ipk", G, temp1)
        sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
        sigma_total[:, : self.n_occupied, :] += np.einsum(
            "ipr,pk->irk", temp2, U[:, : self.n_occupied]
        )
        sigma_total += np.einsum("klrs,isl->irk", hessian_tilde, R_total)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0]
                k = self.index_map[j][1]
                sigma_reduced[i][j + pointer] = sigma_total[i][r][k]
        sigma_reduced -= eigval * R_reduced
        sigma_reduced = sigma_reduced.flatten()
        return sigma_reduced

    def mv2(self, U, A_tilde, G_blocks, R_reduced, num_states, pointer, eigval):
        """Single-vector (H - eigval*I) product, for the LinearOperator sites.  MOVED to orbital_sigma.py (2026-09-04).

        Kept as a method because the name is this class's interface: the
        solver modules call it as ctx.mv2(...).  See
        orbital_sigma.py for the kernel and for why that module is a leaf.
        """
        return _os_mv2(
            self,
            U,
            A_tilde,
            G_blocks,
            R_reduced,
            num_states,
            pointer,
            eigval,
        )

    # build_sigma_reduced8 MOVED to orbital_sigma.py (2026-09-04) -- module-private
    # there: its only caller is in that module.

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


# --- Optional method-level timing for the microiteration solver machinery ----
#
#   QED_PROFILE_MICRO : time the trust-region solver entry points (default 0)
#
# Profiling the microiteration loop with the existing print-timers left ~31.6 s
# of a 290 s c10h8 run (11%) unattributed -- all of it inside the trust-region
# dispatch, where the ~740 MINRES iterations live.  This wraps the relevant
# methods ON THE CLASS rather than editing call sites: several of these are
# reached from three or more branches of microiteration_optimization6's
# dispatch, so wrapping the method covers every branch automatically and avoids
# mis-editing a deeply nested region.  Strict no-op when the knob is unset.
#
# Each call emits `MICRO_PROF <method> <secs>`.  Aggregate with:
#   grep '^MICRO_PROF ' log | awk '{s[$2]+=$3;n[$2]++}
#     END{for(k in s) printf "%-36s %8.2f %7d %9.5f\n",k,s[k],n[k],s[k]/n[k]}' | sort -k2 -rn
#
# NOTE THE TIMINGS NEST: mv2 is called from inside minres_solve /
# linear_equation_solve / solve_pcg_trust_region / orbital_sigma3, and
# orbital_sigma3 from inside the GLTR and Davidson solvers.  Summing all labels
# double-counts; subtract inner from outer when building a budget.
_QED_PROFILE_MICRO = int(os.environ.get("QED_PROFILE_MICRO", "0"))

if _QED_PROFILE_MICRO:

    def _install_micro_profilers(cls, names):
        for _name in names:
            _orig = getattr(cls, _name, None)
            if _orig is None:
                print(f"MICRO_PROF_WARNING no such method: {_name}")
                continue

            def _make(orig, name):
                def _wrapper(self, *args, **kwargs):
                    _t = timer()
                    try:
                        return orig(self, *args, **kwargs)
                    finally:
                        print(f"MICRO_PROF {name} {timer() - _t:.6f}")

                return _wrapper

            setattr(cls, _name, _make(_orig, _name))

    _install_micro_profilers(
        PFHamiltonianGenerator,
        (
            "solve_gltr_trust_region",
            "solve_gltr_with_operator",
            # "solve_pcg_trust_region" was listed here until 2026-09-04; it is
            # now in legacy_python_ci.py, and _install_micro_profilers would
            # print a MICRO_PROF_WARNING for it on every run with the knob on.
            "Davidson_augmented_hessian_solve6",
            "minres_solve",
            "linear_equation_solve",
            "mv2",
            "orbital_sigma3",
            "microiteration_exact_energy",
            "microiteration_predicted_energy2",
            # --- internal_optimization3 and everything it calls -------------
            # 27.5 s / 9 calls on c10h8, of which ~19.3 s is already attributed
            # to the two integral-transformation timers ("full internal
            # transformation" 12.66, "internal transformation" 6.59); these
            # cover the remaining ~8.3 s.  The LSTRS loop inside
            # internal_optimization3 is INLINE (not a method), so its cost shows
            # up as internal_optimization3 minus the sum of the children below.
            "internal_optimization3",
            "build_intermediates_internal",
            "build_gradient_and_hessian",
            "internal_optimization_predicted_energy",
            "internal_optimization_exact_energy",
            "calculate_ci_dependent_energy",
            "internal_transformation",
            "full_transformation_internal_optimization",
            "step_control",
            "build_state_average_rdms",
            "build_unitary_matrix",
            "calculate_off_diagonal_photon_constant",
        ),
    )
