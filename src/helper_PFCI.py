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
#
# The coupled threshold is tied to the orbital gradient, so it tightens as the
# optimization converges; measured cost of that tightening is mean iterations
# rising 23.9 -> 29.9 from the largest to the smallest gradient-norm quartile.
# Kreplin/Werner's precondition is qualitative ("CI residual substantially
# lower than the last orbital gradient"), so 0.1 is a choice, not a derived
# value -- which is what makes it worth sweeping.
_QED_CI_UNCOUPLED_MAXITER = int(os.environ.get("QED_CI_UNCOUPLED_MAXITER", "5"))
_QED_CI_COUPLED_FACTOR = float(os.environ.get("QED_CI_COUPLED_FACTOR", "0.1"))
#
#   QED_DISABLE_QN : never activate quasi-Newton / coupled CI  (default 0 = off)
#
# QN activation (microiteration_optimization6, the `step_norm < 0.05` trigger on
# an accepted step) is what flips the micro CI solve from the Kreplin UNCOUPLED
# cap (constint[8] = _QED_CI_UNCOUPLED_MAXITER, default 5) to COUPLED
# (constint[8] = 10000, run to a 0.1*||g_orb||-scaled threshold). Setting this to
# 1 keeps the whole run uncoupled, which is the apples-to-apples setting for
# comparing against an UNCOUPLED-CI reference (e.g. the Molpro c10h8 benchmark in
# examples/c10h8.out) and also matches the cpp_casscf port, whose QN/BFGS path is
# deliberately not wired up. Gates the LIVE site only (microiteration_optimization6
# ~12328); the byte-identical block in dead microiteration_optimization5 (~10639)
# is untouched.
_QED_DISABLE_QN = int(os.environ.get("QED_DISABLE_QN", "0"))
#
#   QED_USE_MINRES : solve Hx=-g with preconditioned MINRES  (default 1 = ON)
#
# Unlike the knobs above this one is NOT default-no-op: minres_solve replaces
# linear_equation_solve/LinearRMSolver at all four "gradient is small, use
# Newton step" / hard_case==2 call sites (internal microiteration_optimization5
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
#   QED_LSTRS_NORM_TOL : LSTRS boundary tolerance |(||s||-D)/D|   (default 1e-3)
#
# The LSTRS alpha-bisection keeps iterating until the step norm matches the
# trust radius to this relative accuracy.  Gates the two LIVE sites --
# internal_optimization3 (~7912, dense/bordered) and microiteration_
# optimization6 (~12415, matrix-free Davidson) -- not the micro_opt5 or
# ah_orbital_optimization twins, which do not fire in the measured runs.
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
_QED_G_SYMV = int(os.environ.get("QED_G_SYMV", "1"))
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
import numba as nb
from scipy.linalg import eigh, cho_factor, cho_solve, solve_triangular
import ctypes as _ctypes
import llvmlite.binding as _llvm
from numba import types as _nbtypes

# _symv_accumulate(A, x, y) computes y += A @ x for a symmetric, C-contiguous A.
#
# dsymv IS BOUND BY SYMBOL NAME, AND UNCONDITIONALLY.  Two bugs are designed out
# here; both were real and both are described in the QED_G_SYMV note above.
#
# (1) THE CACHE PENALTY.  This used to bind dsymv through a ctypes address
#     obtained from get_cython_function_address.  A nopython function holding a
#     ctypes address is a "dynamic global": numba refuses to write it to the
#     on-disk cache (it would bake an absolute address that ASLR invalidates in
#     the next process) and the refusal PROPAGATES to build_sigma_reduced7/_8,
#     which reference this helper -- a MEASURED 5.6 s of recompile per process,
#     comparable to everything the knob saved on one c10h8 run.  A symbol-name
#     binding puts no address in the IR, so cache=True survives; verified by the
#     artifacts appearing on disk and a cold second process loading them.
#
# (2) THE SILENTLY-INERT KNOB.  Numba's cache key is a function's own source, NOT
#     which helper the module bound, so gating the BINDING on the knob meant a
#     knob-on process could load an artifact compiled against the dgemv stand-in
#     and do nothing at all -- silently, since both variants compute the same
#     numbers.  That is why the decorators used to carry cache=(_QED_G_SYMV==0).
#     Binding dsymv ALWAYS removes the ambiguity at the root: there is now only
#     one _symv_accumulate, so build_sigma_reduced7/_8 have one cached artifact
#     that is correct for either setting, and QED_G_SYMV is a pure RUNTIME flag
#     (the `use_symv` argument, tested at the call site).  Nothing is lost by
#     binding when the knob is off: with use_symv == 0 the branch is never taken.
#
# WHICH LIBRARY.  scipy.linalg.cython_blas exports no ELF symbol for dsymv (it
# publishes Cython capsules), so binding by name necessarily means binding a real
# BLAS .so.  libmkl_rt is already resident in any process that imports psi4, is
# what ci_solver.c/orbital.c link against, and exports `dsymv_` in the LP64
# interface -- 32-bit int, which is exactly what the marshalling below already
# assumed.  (It also exports the ILP64 `dsymv_64_`; do NOT bind that one without
# widening the integers.)  Measured against the previous ctypes path at c10h8
# dims: BIT-IDENTICAL output, max|d| exactly 0.0, and the same speed (2.02 vs
# 1.90 ms, both ~55 GB/s on the triangle, against 4.98 ms for the full dgemv).
def _find_dsymv_library():
    """First libmkl_rt that loads; None if there is none."""
    for _cand in (
        os.path.join(sys.prefix, "lib", "libmkl_rt.so.2"),
        os.path.join(sys.prefix, "lib", "libmkl_rt.so.1"),
        os.path.join(sys.prefix, "lib", "libmkl_rt.so"),
        "libmkl_rt.so.2",
        "libmkl_rt.so",
    ):
        try:
            _ctypes.CDLL(_cand)
            return _cand
        except OSError:
            continue
    return None


_DSYMV_LIB = _find_dsymv_library()
if _DSYMV_LIB is not None:
    _llvm.load_library_permanently(_DSYMV_LIB)
    #   void dsymv(char *uplo, int *n, double *alpha, double *a, int *lda,
    #              double *x, int *incx, double *beta, double *y, int *incy)
    # Fortran BLAS takes everything by reference, so every argument goes over as
    # a void* and the scalars have to be handed over as one-element arrays.
    _dsymv = _nbtypes.ExternalFunction(
        "dsymv_", _nbtypes.void(*([_nbtypes.voidptr] * 10))
    )

    @nb.njit(fastmath=True, cache=True)
    def _symv_accumulate(A, x, y):
        """y += A @ x, reading one triangle of A only.

        BLAS is column-major and A here is row-major, but a symmetric matrix is
        its own transpose, so the same bytes serve both interpretations and no
        copy is needed -- 'L' and 'U' merely swap which physical triangle gets
        touched.  beta=1.0 accumulates into y in place, so nothing is allocated.
        """
        _n = np.empty(1, dtype=np.int32)
        _n[0] = A.shape[0]
        _inc = np.empty(1, dtype=np.int32)
        _inc[0] = 1
        _alpha = np.empty(1, dtype=np.float64)
        _alpha[0] = 1.0
        _beta = np.empty(1, dtype=np.float64)
        _beta[0] = 1.0
        _uplo = np.empty(1, dtype=np.uint8)
        _uplo[0] = 76  # ord('L')
        _dsymv(
            _uplo.ctypes.data,
            _n.ctypes.data,
            _alpha.ctypes.data,
            A.ctypes.data,
            _n.ctypes.data,   # lda == n: A is a contiguous square block
            x.ctypes.data,
            _inc.ctypes.data,
            _beta.ctypes.data,
            y.ctypes.data,
            _inc.ctypes.data,
        )
else:
    # No MKL: fall back to dgemv and force the knob off, so a machine without it
    # runs the documented default path rather than a symmetric product that is
    # silently reading the whole matrix anyway.
    _QED_G_SYMV = 0

    @nb.njit(fastmath=True, cache=True)
    def _symv_accumulate(A, x, y):
        """Stand-in with identical results via dgemv, reading all of A."""
        y += np.dot(x, A)


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
                    c_build_active_rdm(
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
                        while macroiteration < 1000:
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
                                if _cold:
                                    self.constint[5] = _sv_indim
                                    self.constint[6] = _sv_maxdim
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
                            if np.abs(new_avg_energy - old_avg_energy) < 1e-10:
                                convergence = 1
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
                                    np.savetxt("orbital.out", self.opt_C)
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

                                    np.savetxt("occupation_number.out", self.noocs)
                                    np.savetxt("natural_orbital.out", self.nat_orbs)
 
                              
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
                                c_full_transformation_macroiteration(
                                    self.U_total,
                                    self.twoeint,
                                    self.J,
                                    self.K,
                                    self.index_map_pq,
                                    self.index_map_kl,
                                    self.nmo,
                                    self.n_occupied,
                                )
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

        np.savetxt("orbital2.out", self.C)
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
            wfn_dict["matrix"]["Ca"] = np.loadtxt("orbital.out")
            wfn_dict["matrix"]["Cb"] = np.loadtxt("orbital.out")
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
            self.gkl = self.build_gkl_from_df_optimized(self.B_mo, d_cmo_effective, self.H_spatial)
            #for k in range(self.nmo):
            #    for l in range(self.nmo):
            #        print(self.gkl[k,l], self.gkl_new[k,l])
            #print(np.allclose(self.gkl, self.gkl_new)) 
            #self.twoeint1 = np.copy(I_mo_df)
            #self.build2DSO()
        else:
            self.density_fitting = False
            I = np.asarray(mints.ao_eri())
            self.twoeint1 = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca))

            #self.twoeint1 = np.asarray(mints.mo_eri(self.Ca, self.Ca, self.Ca, self.Ca))
            t_eri_end = time.time()
            print(f" Completed ERI Build in {t_eri_end - t_1H_end} seconds ")

            # form the 2H in spin orbital basis
            self.build2DSO()
            t_2d_end = time.time()
            print(f" Completed 2D build in {t_2d_end - t_eri_end} seconds")

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
            print("mem_twoeint", self.twoeint.size * self.twoeint.itemsize / 1024 / 1024)
        print("density_fitting",self.density_fitting)
        print(psutil.Process().memory_info().rss / (1024 * 1024))
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

        # an array to hold the excitation rank count and cumulative weight
        self.determinant_count_by_excitation_rank = np.zeros(self.n_act_el)
        self.determinant_cumulative_weight_by_excitation_rank = np.zeros(self.n_act_el)

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

                # store alphalist and betalist for the reference to a numpy array
                a_ref = np.array(alphalist)
                b_ref = np.array(betalist)

                #for j in range(min(H_dim, 10)):
                for j in range(H_dim):
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
        # naive build of J_klrs and K_klrs from the full integrals
        # self.J = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.K = np.zeros((self.n_occupied, self.n_occupied, self.nmo, self.nmo))
        # self.J = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:self.n_occupied,:self.n_occupied,:,:]
        # self.K = self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[:,:self.n_occupied,:,:self.n_occupied]
        self.J = copy.deepcopy(
            self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[
                : self.n_occupied, : self.n_occupied, :, :
            ]
        )
        #self.J_temp = copy.deepcopy(
        #    self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[
        #        : self.n_occupied, : self.n_occupied, :, :
        #    ]
        #)
        self.K = copy.deepcopy(
            self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[
                :, : self.n_occupied, :, : self.n_occupied
            ]
        )
        #self.K_temp = copy.deepcopy(
        #    self.twoeint.reshape((self.nmo, self.nmo, self.nmo, self.nmo))[
        #        :, : self.n_occupied, :, : self.n_occupied
        #    ]
        #)
        # print (np.shape(self.J),np.shape(self.K))
        self.K = self.K.transpose(1, 3, 0, 2)
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

    def build_gradient(self, U, A, G, gradient_tilde, A_tilde, full_space):
        if full_space == True:
            B = np.zeros((self.nmo, self.nmo))
            T = U - np.eye(self.nmo)
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
        trust_radius = 0.5
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
                    mu9 = np.zeros(dim00)
                    beta0 = 0
                    both_roots_normalization = 0
                    v1 = 0
                    v2 = 0
                    u1 = np.zeros(self.n_act_orb * self.n_in_a)
                    u2 = np.zeros(self.n_act_orb * self.n_in_a)
                    x1 = np.zeros(self.n_act_orb * self.n_in_a)
                    x2 = np.zeros(self.n_act_orb * self.n_in_a)
                    phi1 = 0
                    phi2 = 0

                    lambda1 = 0
                    lambda2 = 0
                    lambda_c = 0
                    idx = hessian_diagonal.argsort()
                    delta_u = hessian_diagonal[idx[0]]
                    delta_l = 0
                    alpha_u = delta_u + np.linalg.norm(gradient_tilde_ai) * trust_radius
                    alpha_l = 0
                    count10 = 0
                    # beta = min(0, alpha_u)
                    beta = alpha_u
                    # aug_hessian_eigenvecs = np.zeros((dim00,2))
                    # aug_hessian_eigenvals = np.zeros(2)
                    print("trust radius", trust_radius)
                    _stats_span = _stats_begin("lstrs_internal")
                    while True:
                        print("****************************")
                        print("lstrs iteration", count10, "beta", beta)
                        beta0 = beta
                        mu9[:] = self.projection_step2(
                            gradient_tilde_ai, hessian_tilde_ai, w9, beta, dim00
                        )
                        # print("two roots", mu9[0], mu9[1], mu9[2])
                        if count10 == 0:
                            delta_l = mu9[0]
                            alpha_l = (
                                delta_l
                                - np.linalg.norm(gradient_tilde_ai) / trust_radius
                            )
                        # else:
                        #    guess_vector = Q
                        #    Q = self.Davidson_augmented_hessian_solve3(self.U2, A_tilde2, G, G1, self.reduced_hessian_diagonal,
                        #            reduced_gradient, beta, aug_hessian_eigenvecs, aug_hessian_eigenvals, guess_vector, restart = True)
                        # print(aug_hessian_eigenvals)
                        # print(aug_hessian_eigenvecs)
                        # mu9[0] = aug_hessian_eigenvals[0]
                        # mu9[1] = aug_hessian_eigenvals[1]
                        # w9[:,0] = aug_hessian_eigenvecs[:,0]
                        # w9[:,1] = aug_hessian_eigenvecs[:,1]
                        print("ALPHA range", alpha_l, alpha_u)
                        # idx = mu9.argsort()[:dim00]
                        # print(idx)
                        # w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                        # w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                        v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                        v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                        u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                        u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                        aa1 = np.linalg.norm(gradient_tilde_ai) * np.abs(v1)
                        bb1 = np.sqrt(1 - v1 * v1)
                        aa2 = np.linalg.norm(gradient_tilde_ai) * np.abs(v2)
                        bb2 = np.sqrt(1 - v2 * v2)
                        print("aa1, bb1", aa1, bb1, "aa2, bb2", aa2, bb2)
                        if mu9[0] > -1e-8 and np.linalg.norm(
                            u1
                        ) < trust_radius * np.abs(v1):
                            hard_case = 2
                            print("use newton step")
                            break
                        delta_u2 = mu9[0] - v1 * np.dot(gradient_tilde_ai, u1) / (
                            1 - v1 * v1
                        )
                        delta_u = min(delta_u, delta_u2)
                        print("delta_u2", delta_u2)

                        epsilon_v = 1e-4
                        print(aa1 <= epsilon_v * bb1)
                        print(aa2 <= epsilon_v * bb2)
                        alpha = beta
                        while (
                            (aa1 <= epsilon_v * bb1)
                            and (aa2 <= epsilon_v * bb2)
                            and np.abs(alpha_u - alpha_l)
                            > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
                        ):
                            alpha_u = alpha
                            alpha = (alpha_l + alpha_u) / 2
                            w9[:, :] = 0.0
                            mu9[:] = self.projection_step2(
                                gradient_tilde_ai, hessian_tilde_ai, w9, alpha, dim00
                            )

                            v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                            v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                            u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                            u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                            aa1 = np.linalg.norm(gradient_tilde_ai) * np.abs(v1)
                            bb1 = np.sqrt(1 - v1 * v1)
                            aa2 = np.linalg.norm(gradient_tilde_ai) * np.abs(v2)
                            bb2 = np.sqrt(1 - v2 * v2)
                            print(
                                alpha,
                                aa1 <= epsilon_v * bb1,
                                aa2 <= epsilon_v * bb2,
                                mu9[0],
                                mu9[1],
                                bb1,
                                bb2,
                            )
                            delta_u2 = mu9[0] - v1 * np.dot(gradient_tilde_ai, u1) / (
                                1 - v1 * v1
                            )
                            delta_u = min(delta_u, delta_u2)
                            print("new delta_u", delta_u, delta_u2)

                        if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
                            exit()
                        elif (aa1 > epsilon_v * bb1) and (aa2 > epsilon_v * bb2):
                            both_roots_normalization = 1
                        else:
                            both_roots_normalization = 0
                        beta = alpha
                        # update delta_u
                        # temptemp = np.dot(reduced_hessian, u1)
                        # temptemp2 = np.dot(u1, temptemp)

                        # delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1)/np.dot(u1,u1)
                        # delta_u = min(delta_u, temptemp2/np.dot(u1,u1))
                        # print("new delta_u", delta_u)
                        ####scale9 = w9[0][0]
                        ####print("scale",scale9)
                        ####if np.abs(scale9) < 1e-15:
                        ####    break
                        # assign (k-1) and k-iteration values
                        lambda1 = lambda2
                        x1 = x2
                        phi1 = phi2
                        print("two roots", mu9[0], mu9[1])
                        if aa1 > epsilon_v * bb1:
                            step10 = u1 / v1
                            second_component[:] = u2
                            lambda2 = mu9[0]
                            print("norm from first root", np.linalg.norm(step10))
                            # if aa2 > epsilon_v * bb2:
                            #    second_component[:] = u2/v2
                            #    print("second root norm", np.linalg.norm(second_component))
                            if np.linalg.norm(step10) < trust_radius:
                                alpha_l = beta
                            if np.linalg.norm(step10) > trust_radius:
                                alpha_u = beta
                        else:
                            step10 = u2 / v2
                            second_component[:] = u1
                            lambda2 = mu9[1]
                            print("norm from second root", np.linalg.norm(step10))
                            alpha_u = beta
                        print("NEW ALPHA range", alpha_l, alpha_u)
                        step10_norm = np.linalg.norm(step10)
                        x2 = step10
                        if mu9[0] > -1e-8 and np.linalg.norm(
                            u1
                        ) < trust_radius * np.abs(v1):
                            hard_case = 2
                            print("use newton step")
                            break
                        phi2 = -np.dot(gradient_tilde_ai, x2)
                        # print(step10)
                        # print(x1)
                        # print(x2)
                        # if count10 == 0 and step10_norm < trust_radius: break
                        # print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                        # print(mu9, flush=True)
                        print(
                            "step10 norm", step10_norm, "current beta", beta, flush=True
                        )
                        x1_norm = np.linalg.norm(x1)
                        x2_norm = np.linalg.norm(x2)
                        phi2_p = np.dot(x2, x2)
                        phi1_p = np.dot(x1, x1)
                        print(
                            "phi1",
                            phi1,
                            "phi1_p",
                            phi1_p,
                            "phi2",
                            phi2,
                            "phi2_p",
                            phi2_p,
                        )
                        if (
                            np.abs((step10_norm - trust_radius) / trust_radius) <= _QED_LSTRS_NORM_TOL
                            and mu9[0] <= 0
                        ):
                            step = step10
                            hard_case = 0
                            break
                        print("check orthogonality of the first two roots")
                        vv1 = w9[0, 0]
                        vv2 = w9[0, 1]
                        uu1 = w9[1:, 0]
                        uu2 = w9[1:, 1]
                        # check if the quasi-optimal condition is satisfied
                        print(
                            "quasi-optimal condition",
                            (1 + trust_radius * trust_radius) * (vv1 * vv1 + vv2 * vv2),
                            flush=True,
                        )
                        qs = (1 + trust_radius * trust_radius) * (vv1 * vv1 + vv2 * vv2)
                        print(np.dot(w9[:, :2].T, w9[:, :2]))

                        # construct quasi-optimal step
                        epsilon_hc = _QED_LSTRS_EPS_HC  # QED_LSTRS_EPS_HC, default 1e-6
                        eta = epsilon_hc / (1 - epsilon_hc)
                        tau1 = 1
                        tau2 = 1
                        if qs > 1.0:
                            tau1 = (
                                (vv1 - vv2 * np.sqrt(qs - 1))
                                / (vv1 * vv1 + vv2 * vv2)
                                / np.sqrt(1 + trust_radius * trust_radius)
                            )
                            tau2 = (
                                (vv2 + vv1 * np.sqrt(qs - 1))
                                / (vv1 * vv1 + vv2 * vv2)
                                / np.sqrt(1 + trust_radius * trust_radius)
                            )
                        elif np.abs(qs - 1.0) < 2e-308:
                            tau1 = vv1 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                            tau2 = vv2 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                        x_tilde = (tau1 * uu1 + tau2 * uu2) / (tau1 * vv1 + tau2 * vv2)

                        lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                        psi_tilde = 0.5 * self.internal_optimization_predicted_energy(
                            gradient_tilde_ai, hessian_tilde_ai, x_tilde
                        )
                        if qs > 1.0 or np.abs(qs - 1.0) < 2e-308:
                            print(
                                "x_tilde norm1",
                                np.linalg.norm(x_tilde),
                                (mu9[1] - mu9[0])
                                * tau2
                                * tau2
                                * (1 + trust_radius * trust_radius),
                                -2.0 * eta * psi_tilde,
                            )
                            if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                1 + trust_radius * trust_radius
                            ) < -2.0 * eta * psi_tilde:
                                print("find quasi-optimal solution1", flush=True)
                                hard_case = 3
                                break
                            else:
                                if qs > 1.0:
                                    tau1 = (
                                        (vv1 + vv2 * np.sqrt(qs - 1))
                                        / (vv1 * vv1 + vv2 * vv2)
                                        / np.sqrt(1 + trust_radius * trust_radius)
                                    )
                                    tau2 = (
                                        (vv2 - vv1 * np.sqrt(qs - 1))
                                        / (vv1 * vv1 + vv2 * vv2)
                                        / np.sqrt(1 + trust_radius * trust_radius)
                                    )
                                    x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                        tau1 * vv1 + tau2 * vv2
                                    )
                                    lambda_tilde = (
                                        tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                                    )
                                    psi_tilde = (
                                        0.5
                                        * self.internal_optimization_predicted_energy(
                                            gradient_tilde_ai, hessian_tilde_ai, x_tilde
                                        )
                                    )
                                    print(
                                        "x_tilde norm2",
                                        np.linalg.norm(x_tilde),
                                        (mu9[1] - mu9[0])
                                        * tau2
                                        * tau2
                                        * (1 + trust_radius * trust_radius),
                                        -2.0 * eta * psi_tilde,
                                        flush=True,
                                    )
                                    if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                        1 + trust_radius * trust_radius
                                    ) < -2.0 * eta * psi_tilde:
                                        print(
                                            "find quasi-optimal solution2", flush=True
                                        )
                                        hard_case = 3
                                        break
                        if np.abs(alpha_u - alpha_l) <= 1e-8 * max(
                            np.abs(alpha_u), np.abs(alpha_l)
                        ):
                            print("interval too small")
                            if x2_norm < trust_radius and both_roots_normalization == 0:
                                hard_case = 1
                            if both_roots_normalization == 1:
                                hard_case = 4
                            break

                        if count10 == 0:
                            beta = beta + (beta - lambda2) / x2_norm * (
                                trust_radius - x2_norm
                            ) / trust_radius * (trust_radius + 1.0 / x2_norm)
                        else:
                            print("denominator", x2_norm - x1_norm)
                            denom = trust_radius * (x2_norm - x1_norm)

                            if np.abs(denom) > 2e-308:
                                lambda_c = (
                                    lambda1 * x1_norm * (x2_norm - trust_radius)
                                    + lambda2 * x2_norm * (trust_radius - x1_norm)
                                ) / denom
                            else:
                                lambda_c = delta_u
                            if lambda_c > delta_u:
                                print("need to safeguard delta_u", lambda_c, delta_u)
                                lambda_c = delta_u
                            # lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                            # print(lambda_c)
                            if np.abs(lambda2 - lambda1) <= 2e-308:
                                print("interpolate alpha by bisection")
                                beta = (alpha_l + alpha_u) / 2
                            else:
                                omega_k = (lambda2 - lambda_c) / (lambda2 - lambda1)
                                num = (
                                    x1_norm
                                    * x2_norm
                                    * (x2_norm - x1_norm)
                                    * (lambda1 - lambda_c)
                                    * (lambda2 - lambda_c)
                                )

                                denom = (
                                    omega_k * x2_norm + (1 - omega_k) * x1_norm
                                ) * (lambda2 - lambda1)
                                print(x1_norm, x2_norm)
                                print(lambda1, lambda2, lambda_c)
                                print(num, denom)
                                if np.abs(denom) <= 2e-308:
                                    print("interpolate alpha by bisection2")
                                    beta = (alpha_l + alpha_u) / 2
                                else:
                                    beta = (
                                        lambda_c
                                        + omega_k * phi1
                                        + (1 - omega_k) * phi2
                                        + num / denom
                                    )

                        # if count10 == 0 or (np.abs(x2_norm - x1_norm) <= 1e-15):
                        #    if (np.abs(x2_norm - x1_norm) <= 1e-15):
                        #        print("too small denominator")
                        #    beta = beta + (beta - lambda2)/x2_norm * (trust_radius - x2_norm)/trust_radius * (trust_radius + 1.0/x2_norm)
                        # else:
                        #    print("denominator", x2_norm-x1_norm)
                        #    if (np.abs(x2_norm - x1_norm) > 1e-15):
                        #        lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                        #    else:
                        #        lambda_c = delta_u
                        #    if lambda_c > delta_u:
                        #        print("need to safeguard delta_u", lambda_c, delta_u)
                        #        lambda_c = delta_u
                        #    #lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                        #    #print(lambda_c)
                        #    omega_k = (lambda2 - lambda_c)/(lambda2 - lambda1)
                        #    ratio1 = x1_norm * x2_norm * (x2_norm - x1_norm)/(omega_k * x2_norm + (1-omega_k) * x1_norm)
                        #    ratio2 = (lambda1 - lambda_c) * (lambda2 - lambda_c) / (lambda2 - lambda1)
                        #    beta = lambda_c + omega_k *phi1 + (1-omega_k) * phi2 + ratio1 * ratio2
                        print("beta after interpolation", beta)
                        # safeguard alpha
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
                                print("using bisection")
                                beta = (alpha_l + alpha_u) / 2
                        count10 += 1
                        # if np.abs(beta -beta0) <= 1e-12:
                        #    print("possible convergence")
                        #    print(beta, beta0)
                        #    print(x2_norm)

                        #    if x2_norm < trust_radius:
                        #        print("11111111")
                        #        hard_case = 1
                        #    elif x2_norm > trust_radius and both_roots_normalization == 1:
                        #        print("22222222")
                        #        hard_case = 1

                        #    break
                        if count10 == 50:
                            break
                    print("check hard case", hard_case)
                    _stats_end(
                        _stats_span, iters=count10, status="hard_case=%d" % hard_case
                    )
                    if hard_case == 1:
                        first_component[:] = x2[:]
                        if x2_norm > trust_radius:
                            print("some logic is wrong")
                            exit()
                            # temp_vector = np.zeros(self.index_map_size)
                            # temp_vector[:] = second_component[:]
                            # second_component[:] = first_component[:]
                            # first_component[:] = temp_vector[:]
                        # print(first_component)
                        # print(second_component)
                        print("norm of current root", np.linalg.norm(first_component))
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

                    if hard_case == 2:
                        Q = np.zeros((1, self.n_act_orb * self.n_in_a))
                        x, exitCode = minres(
                            hessian_tilde_ai, -gradient_tilde_ai, rtol=1e-5
                        )
                        print("exitcode", exitCode)
                        step = x

                    if hard_case == 3:
                        step = x_tilde
                    if hard_case == 4:
                        if x2_norm > trust_radius:
                            step = x2 / x2_norm * trust_radius
                        if x2_norm < trust_radius:
                            step = x2

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
                self.constdouble[4] = 1e-5
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
            if (
                #gradient_norm < 1e-4 and current_residual < 1e-5
                gradient_norm < 1e-4 and self.constint[8] == 0
            ) or microiteration == 20:
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
        if ratio >= 0 and ratio <= 0.25:
            trust_radius = 0.7 * trust_radius
        if ratio > 0.75:
            # QED_STEP_CONTROL (a): only widen the region if the step actually
            # pressed against it.  An interior step tells us nothing about
            # whether a larger Delta would still be trustworthy.
            if _QED_STEP_CONTROL and step_norm is not None and step_norm < 0.8 * trust_radius:
                return trust_radius
            trust_radius = 1.2 * trust_radius
            if trust_radius > 0.75:
                trust_radius = 0.75
        return trust_radius

    def build_state_average_rdms(self, eigenvecs):
        self.D_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.Dpe_tu_avg = np.zeros((self.n_act_orb * self.n_act_orb))
        self.D_tuvw_avg = np.zeros(
            (self.n_act_orb * self.n_act_orb * self.n_act_orb * self.n_act_orb)
        )
        np1 = self.N_p + 1
        for i in range(self.davidson_roots):
            c_build_active_rdm(
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

    def build_sigma_reduced(self, U, A, G, step):
        self.reduced_hessian2 = np.zeros((self.index_map_size, self.index_map_size))
        # build the component of tranformed hessian from gradient
        B = np.zeros((self.nmo, self.nmo))
        self.T = U - np.eye(self.nmo)
        B[:, : self.n_occupied] = A[:, : self.n_occupied] + np.einsum(
            "klrs,sl->rk", G[:, :, :, :], self.T[:, : self.n_occupied]
        )
        # B2 = np.zeros((self.nmo, self.nmo))
        # for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        B2[r][k] = A[r][k]
        #        for s in range(self.nmo):
        #            for l in range(self.n_occupied):
        #                B2[r][k] += G[k][l][r][s] * self.T[s][l]
        # print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))

        A_tilde = np.zeros((self.nmo, self.nmo))
        # A_tilde[:,:self.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:self.n_occupied])
        # A2_tilde = np.zeros((self.nmo, self.nmo))
        A_tilde = np.einsum("rs,sk->rk", U.T, B)
        # A2_tilde = np.zeros((self.nmo, self.nmo))
        # for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        for s in range(self.nmo):
        #            A2_tilde[r][k] += U[s][r] * B[s][k]
        # print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))

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
        self.sigma_total[: self.n_occupied, :] -= np.einsum(
            "pr,pk->rk", temp2, U[:, : self.n_occupied]
        )

        temp1 = np.einsum(
            "ql,sl->qs", U[:, : self.n_occupied], self.R_total[: self.n_occupied, :]
        )
        temp2 = np.einsum("kspq,qs->pk", G, temp1)
        self.sigma_total -= np.einsum("pk,pr->rk", temp2, U)
        self.sigma_total[: self.n_occupied, :] += np.einsum(
            "pr,pk->rk", temp2, U[:, : self.n_occupied]
        )
        self.sigma_total += np.einsum("klrs,sl->rk", hessian_tilde, self.R_total)

    def build_sigma_reduced2(
        self, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer
    ):
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw", num_states)
        # print(self.index_map)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0]
                k = self.index_map[j][1]
                R_total[i][r][k] = R_reduced[i][j + pointer]
        # print("R total")
        # for i in range(num_states):
        #    for r in range(self.nmo):
        #        for k in range(self.n_occupied):
        #            print("%20.12lf" %(R_total[i,r,k]))
        # print("\n")

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
        # A2_tilde = np.zeros((self.nmo, self.nmo))
        # for r in range(self.nmo):
        #    for k in range(self.n_occupied):
        #        for s in range(self.nmo):
        #            A2_tilde[r][k] += U[s][r] * B[s][k]
        # print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))

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

    def build_sigma_reduced3(
        self, U, hessian_tilde, G, R_reduced, sigma_reduced, num_states, pointer
    ):
        R_total = np.zeros((num_states, self.nmo, self.n_occupied))
        print("weqeqw", num_states)
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0]
                k = self.index_map[j][1]
                R_total[i][r][k] = R_reduced[i][j + pointer]

        sigma_total = np.zeros((num_states, self.nmo, self.n_occupied))

        temp1 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp2 = np.zeros((num_states, self.nmo, self.n_occupied))
        temp1 = np.einsum("qs,isl->iql", U, R_total, optimize="optimal")
        temp1 -= np.einsum(
            "ql,isl->iqs",
            U[:, : self.n_occupied],
            R_total[:, : self.n_occupied, :],
            optimize="optimal",
        )
        temp2 = np.einsum("klpq,iql->ipk", G, temp1, optimize="optimal")
        sigma_total = np.einsum("ipk,pr->irk", temp2, U, optimize="optimal")
        sigma_total[:, : self.n_occupied, :] -= np.einsum(
            "ipr,pk->irk", temp2, U[:, : self.n_occupied], optimize="optimal"
        )

        # temp1 = np.einsum("ql,isl->iqs", U[:,:self.n_occupied], R_total[:,:self.n_occupied,:])
        # temp2 = np.einsum("kspq,iqs->ipk", G, temp1, optimize = "optimal")
        # sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
        # sigma_total[:,:self.n_occupied,:] += np.einsum("ipr,pk->irk", temp2, U[:,:self.n_occupied])
        sigma_total += np.einsum(
            "klrs,isl->irk", hessian_tilde, R_total, optimize="optimal"
        )
        for i in range(num_states):
            for j in range(self.index_map_size):
                r = self.index_map[j][0]
                k = self.index_map[j][1]
                sigma_reduced[i][j + pointer] = sigma_total[i][r][k]
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

    def orbital_sigma3(
        self, U, A_tilde, G_blocks, R_reduced, sigma_reduced, num_states, pointer
    ):
        _mv_t0 = timer() if _QED_SOLVER_STATS else None
        nmo = self.nmo
        index_map = self.index_map
        index_map_size = self.index_map_size
        n_occupied = self.n_occupied
        n_inactive = self.n_in_a
        n_active = self.n_act_orb
        G_ij, G_ti, G_tu = G_blocks
        G_ij = np.ascontiguousarray(G_ij)
        G_ti = np.ascontiguousarray(G_ti)
        G_tu = np.ascontiguousarray(G_tu)
        # build_sigma_reduced7 is jitted with an explicit float64[:,::1] signature,
        # so R_reduced must be C-contiguous.  The un-jitted build_sigma_reduced4
        # this path replaced accepted any layout; microiteration_predicted_energy2
        # can hand us a strided step.  (sigma_reduced is an output -- do NOT copy.)
        R_reduced = np.ascontiguousarray(R_reduced)

        _ret = self.build_sigma_reduced7(
            U,
            A_tilde,
            index_map,
            G_ij,
            G_ti,
            G_tu,
            R_reduced,
            sigma_reduced,
            num_states,
            pointer,
            nmo,
            index_map_size,
            n_occupied,
            n_inactive,
            n_active,
            _QED_G_SYMV,
        )
        if _QED_SOLVER_STATS:
            # builds num_states Hessian-vector products in one call; the timed
            # region includes the ascontiguousarray prep above, which is part of
            # what a caller pays to get one matvec.
            _solver_stats.matvec(num_states, timer() - _mv_t0)
        return _ret


    def orbital_sigma(
        self, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer
    ):
        """Dense-G1 sigma build.  SUPERSEDED by orbital_sigma3/build_sigma_reduced7,
        which take the (G_ij, G_ti, G_tu) blocks instead.  No live caller remains:
        the full (nmo*n_occupied) square G1 this needs is no longer materialized
        anywhere in microiteration_optimization6.  Kept for reference only --
        build_sigma_reduced4 below is otherwise identical to build_sigma_reduced7.
        """
        nmo = self.nmo
        index_map = self.index_map
        index_map_size = self.index_map_size
        n_occupied = self.n_occupied
        return self.build_sigma_reduced4(
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

    @staticmethod
    # @nb.njit("""void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,:,:,::1], float64[:,::1], float64[:,::1],
    #        int64, int64, int64, int64, int64)""", fastmath = True, parallel = True)
    # def build_sigma_reduced6(U, A_tilde, index_map, G1, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
    #####@nb.jit(
    #####    """void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1],
    #####        int64, int64, int64, int64, int64)""",
    #####    nopython=True,
    #####    cache=True,
    #####    fastmath=True,
    #####    parallel=False,
    #####)
    def build_sigma_reduced4(
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

        assert U.shape == (nmo, nmo)
        assert G.shape == (nmo * n_occupied, nmo * n_occupied)
        assert A_tilde.shape == (nmo, nmo)
        R_total = np.zeros((num_states, nmo, n_occupied))
        print("num_states", num_states)
        #for j in nb.prange(index_map_size):
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            for i in range(num_states):
                R_total[i][r][k] = R_reduced[i][j + pointer]

        sigma_total = np.zeros((num_states, nmo, n_occupied))
        temp1 = np.zeros((num_states, nmo, n_occupied))
        temp2 = np.zeros((num_states, nmo, n_occupied))

        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(U, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :].transpose(2, 0, 1))
        # R_total1 = np.ascontiguousarray(R_total1)
        R1 = np.reshape(R1, (n_occupied, num_states * n_occupied))
        U1 = np.ascontiguousarray(U[:, :n_occupied])
        temp2 = np.dot(U1, R1)
        temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)

        ##sigma_total1 = np.zeros((num_states, nmo, n_occupied))
        ##temp11 = np.einsum("qs,isl->iql", U, R_total)
        # for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for l in range(n_occupied):
        #            #a = 0.0
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += U[q,s] * R_total[i,s,l]
        #                #temp1[i,q,l] += U[q,s] * R_total[i,s,l]
        #            temp1[i,q,l] = a
        #

        # for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for s in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a -= U[q,l] * R_total[i,s,l]
        #                #temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
        #            temp1[i,q,s] += a
        R1 = np.ascontiguousarray(temp1)
        R1 = np.reshape(R1, (num_states, nmo * n_occupied))
        
        # R2 = np.zeros((num_states, nmo * n_occupied))
        ##for i in nb.prange(num_states):
        ##    for r in range(nmo):
        ##        for k in range(n_occupied):
        ##            a = R1[i][r*n_occupied+k] - temp1[i,r,k]
        ##            if np.abs(a) > 1e-14: print("LARGE error")
        #G_tensor = G.reshape(nmo, n_occupied, nmo, n_occupied)

        temp1 = np.dot(R1, G)
        # temp2 = temp1.reshape(num_states, nmo, n_occupied)
        # for r in nb.prange(nmo):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for p in range(nmo):
        #                a += U[p,r] * temp2[i,p,k]
        #                #sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
        #            temp1[i,r,k] = a
        #            sigma_total[i,r,k] = a
        #

        # for r in nb.prange(n_occupied):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] -= temp1[i,k,r]

        #######for i in nb.prange(num_states):
        #######    R3 = np.dot(R1[i,:], G)
        #######    R2[i,:] = R3

        temp2 = temp1.reshape(num_states, nmo, n_occupied)
        #temp2 = oe.contract("irk,rksl->isl", R1, G_tensor, optimize = "optimal") 
        temp2 = temp2.transpose(0, 2, 1)
        temp2 = np.ascontiguousarray(temp2)
        temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
        temp1 = np.dot(temp2, U)
        temp2 = temp1.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] = temp2.transpose(0, 2, 1)[:, :, :]
        sigma_total[:, :n_occupied, :] -= temp2[:, :, :n_occupied]

        A3_tilde = A_tilde + A_tilde.T
        # temp1 = np.zeros((num_states, nmo, n_occupied))
        # temp2 = np.zeros((num_states, nmo, n_occupied))

        ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        ##hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += A3_tilde[r,s] * R_total[i,s,k]
        #            temp1[i,r,k] = a
        #            sigma_total[i,r,k] -= 0.5 * a
        #
        # for i in nb.prange(num_states):
        #    for r in range(n_occupied):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r]
        #
        ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[k,l] * R_total[i,r,l]
        #            sigma_total[i,r,k] -= 0.5 * a

        ##hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[r,l] * R_total[i,k,l]
        #            sigma_total[i,r,k] += 0.5 * a
        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(A3_tilde, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        sigma_total[:, :, :] -= 0.5 * temp1[:, :, :]
        sigma_total[:, :n_occupied, :] += (
            0.5 * temp1.transpose(0, 2, 1)[:, :, :n_occupied]
        )
        A3 = A3_tilde[:n_occupied, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total)
        R1 = np.reshape(R1, (num_states * nmo, n_occupied))
        temp2 = np.dot(R1, A3)
        sigma_total[:, :, :] -= (
            0.5 * temp2.reshape(num_states, nmo, n_occupied)[:, :, :]
        )
        A3 = A3_tilde[:, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :])
        R1 = np.reshape(R1, (num_states * n_occupied, n_occupied))
        temp2 = np.dot(R1, A3)
        temp1 = temp2.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] += 0.5 * temp1.transpose(0, 2, 1)[:, :, :]

        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            for i in range(num_states):
                sigma_reduced[i][j + pointer] = sigma_total[i][r][k]
    @staticmethod
    @nb.jit(
        """void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1],
            int64, int64, int64, int64, int64, int64, int64, int64)""",
        nopython=True,
        # cache=True unconditionally.  This used to be cache=(_QED_G_SYMV == 0),
        # because numba's cache key is the function's own source and does NOT see
        # which _symv_accumulate the module bound -- so a knob-on process could
        # load an artifact compiled against the dgemv stand-in and the knob would
        # do nothing, silently, both variants computing the same numbers.  That
        # is fixed at the root: dsymv is now bound by SYMBOL NAME and
        # UNCONDITIONALLY, so there is exactly one _symv_accumulate and one
        # correct artifact, and QED_G_SYMV only selects a branch at run time via
        # the use_symv argument.  See the _symv_accumulate note.
        cache=True,
        fastmath=True,
        parallel=False,
    )
    def build_sigma_reduced7(
        U,
        A_tilde,
        index_map,
        G_ij,
        G_ti,
        G_tu,
        R_reduced,
        sigma_reduced,
        num_states,
        pointer,
        nmo,
        index_map_size,
        n_occupied,
        n_inactive,
        n_active,
        use_symv,
    ):

        assert U.shape == (nmo, nmo)
        assert A_tilde.shape == (nmo, nmo)
        R_total = np.zeros((num_states, nmo, n_occupied))
        # A debug `print("num_states", num_states)` used to sit here.  In a
        # nopython function that is one GIL acquisition + one write per
        # Hessian-vector product: measured at 0.27 ms of an 8.18 ms call, i.e.
        # 0.6 s over the 2083 single-vector matvecs of a c10h8 run, and ~2000
        # lines of a 21 MB log.  (The same print remains in the retired
        # build_sigma_reduced4 above, which has no live caller.)
        #for j in nb.prange(index_map_size):
        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            for i in range(num_states):
                R_total[i][r][k] = R_reduced[i][j + pointer]

        sigma_total = np.zeros((num_states, nmo, n_occupied))
        temp1 = np.zeros((num_states, nmo, n_occupied))
        temp2 = np.zeros((num_states, nmo, n_occupied))

        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(U, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :].transpose(2, 0, 1))
        # R_total1 = np.ascontiguousarray(R_total1)
        R1 = np.reshape(R1, (n_occupied, num_states * n_occupied))
        U1 = np.ascontiguousarray(U[:, :n_occupied])
        temp2 = np.dot(U1, R1)
        temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)



        # 1. Create views of the (num_states, nmo, n_occupied) tensor 'temp1'.
        # Slicing on the last dimension is contiguous in memory and very fast (no copy).
        R_i_tensor = temp1[:, :, :n_inactive]
        R_a_tensor = temp1[:, :, n_inactive:]
        
        # 2. Reshape these tensors into 2D matrices for np.dot.
        # This is also a fast, copy-free "view" operation.
        # We assume the G-blocks are flattened with index (r,k) -> r * n_block_occ + k
        
        R1_i = R_i_tensor.copy().reshape(num_states, nmo * n_inactive)
        R1_a = R_a_tensor.copy().reshape(num_states, nmo * n_active)


        # 3. Perform the cache-friendly blocked matrix multiplication.
        # We use the G_ij, G_ti, G_tu blocks you passed in.
        # We use the symmetry G_it = G_ti.T

        # Calculate the inactive block of sigma: (num_states, nmo * n_inactive)
        # QED_G_SYMV arrives as the runtime argument use_symv rather than being
        # read from the module global, so both settings compile from identical
        # source and nothing here depends on numba's branch pruning.  What keeps
        # the on-disk cache honest is that dsymv is bound unconditionally and by
        # symbol name -- see the note there.  G_ij is symmetric, so dsymv reads one 109 MB
        # triangle where dgemv reads the whole 218 MB square: measured 2.40x on
        # this term at c10h8 dims.  See the module-level QED_G_SYMV note for why
        # this is gated on num_states == 1.
        if use_symv == 1 and num_states == 1:
            sigma_i = np.dot(R1_a, G_ti)
            _symv_accumulate(G_ij, R1_i[0], sigma_i[0])
        else:
            sigma_i = np.dot(R1_i, G_ij) + np.dot(R1_a, G_ti)

        # Calculate the active block of sigma: (num_states, nmo * n_active)
        sigma_a = np.dot(R1_i, G_ti.T) + np.dot(R1_a, G_tu)

        # 4. Assemble the result back into the 3D tensor 'temp2'.
        # We allocate the output tensor 'temp2' (which you did in the original code).
        temp2 = np.zeros((num_states, nmo, n_occupied)) # You already have this
        temp2[:, :, :n_inactive] = sigma_i.reshape(num_states, nmo, n_inactive)
        temp2[:, :, n_inactive:] = sigma_a.reshape(num_states, nmo, n_active)  



        #temp2 = oe.contract("irk,rksl->isl", R1, G_tensor, optimize = "optimal") 
        temp2 = temp2.transpose(0, 2, 1)
        temp2 = np.ascontiguousarray(temp2)
        temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
        temp1 = np.dot(temp2, U)
        temp2 = temp1.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] = temp2.transpose(0, 2, 1)[:, :, :]
        sigma_total[:, :n_occupied, :] -= temp2[:, :, :n_occupied]

        A3_tilde = A_tilde + A_tilde.T
        # temp1 = np.zeros((num_states, nmo, n_occupied))
        # temp2 = np.zeros((num_states, nmo, n_occupied))

        ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        ##hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += A3_tilde[r,s] * R_total[i,s,k]
        #            temp1[i,r,k] = a
        #            sigma_total[i,r,k] -= 0.5 * a
        #
        # for i in nb.prange(num_states):
        #    for r in range(n_occupied):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r]
        #
        ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[k,l] * R_total[i,r,l]
        #            sigma_total[i,r,k] -= 0.5 * a

        ##hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[r,l] * R_total[i,k,l]
        #            sigma_total[i,r,k] += 0.5 * a
        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(A3_tilde, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        sigma_total[:, :, :] -= 0.5 * temp1[:, :, :]
        sigma_total[:, :n_occupied, :] += (
            0.5 * temp1.transpose(0, 2, 1)[:, :, :n_occupied]
        )
        A3 = A3_tilde[:n_occupied, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total)
        R1 = np.reshape(R1, (num_states * nmo, n_occupied))
        temp2 = np.dot(R1, A3)
        sigma_total[:, :, :] -= (
            0.5 * temp2.reshape(num_states, nmo, n_occupied)[:, :, :]
        )
        A3 = A3_tilde[:, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :])
        R1 = np.reshape(R1, (num_states * n_occupied, n_occupied))
        temp2 = np.dot(R1, A3)
        temp1 = temp2.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] += 0.5 * temp1.transpose(0, 2, 1)[:, :, :]

        for j in range(index_map_size):
            r = index_map[j][0]
            k = index_map[j][1]
            for i in range(num_states):
                sigma_reduced[i][j + pointer] = sigma_total[i][r][k]


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
        c_full_transformation_macroiteration(
            U_temp,
            self.twoeint,
            J_temp,
            K_temp,
            self.index_map_pq,
            self.index_map_kl,
            self.nmo,
            self.n_occupied,
        )
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
        c_full_transformation_macroiteration(
            U_temp,
            self.twoeint,
            J_temp,
            K_temp,
            self.index_map_pq,
            self.index_map_kl,
            self.nmo,
            self.n_occupied,
        )
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
            c_full_transformation_macroiteration(
                U_temp,
                self.twoeint,
                J_temp,
                K_temp,
                self.index_map_pq,
                self.index_map_kl,
                self.nmo,
                self.n_occupied,
            )

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
            c_full_transformation_macroiteration(
                U_temp,
                self.twoeint,
                J_temp,
                K_temp,
                self.index_map_pq,
                self.index_map_kl,
                self.nmo,
                self.n_occupied,
            )
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
        c_full_transformation_macroiteration(
            U_temp,
            self.twoeint,
            J_temp,
            K_temp,
            self.index_map_pq,
            self.index_map_kl,
            self.nmo,
            self.n_occupied,
        )
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

    def microiteration_optimization5(
        self, U, eigenvecs, c_get_roots, convergence_threshold
    ):
        # print("E_core", self.E_core)
        self.U2 = copy.deepcopy(U)
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
        current_residual = 1
        # while(True):
        current_energy = 0.0
        old_energy = 0.0
        N_orbital_optimization_steps = 1
        N_microiterations = 20
        microiteration = 0
        qn_optimization = False
        qn_count = 0
        while microiteration < N_microiterations:
            print("\n")
            print("\n")
            print("MICROITERATION", microiteration + 1, flush=True)

            # while(microiteration < 2):
            trust_radius = 0.4
            A[:, :] = 0.0
            G[:, :, :, :] = 0.0
            start = timer()
            self.build_intermediates(eigenvecs, A, G, True)
            #G_blocks = self.build_intermediates_with_blocks(eigenvecs, A, G, True)
            end = timer()
            print("build intermediates took", end - start)
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
            start1 = timer()
            #G1 = G.transpose(3, 1, 2, 0).reshape(
            #    self.nmo * self.n_occupied, self.nmo * self.n_occupied
            #)
            # G_t is a free view (transpose only); the three reshapes below are
            # the copies that actually cost.  The full (nmo*n_occupied) square
            # G1 used to be materialized here too -- a second, larger copy --
            # but every consumer now takes the blocks, so it is gone.
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
                < max(0.01 * convergence_threshold, 1e-10)
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

            start = timer()
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            end = timer()
            # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)

            print("build gradient took", end - start)
            #hessian_tilde3 = hessian_tilde.transpose(2,0,3,1)
            #hessian_tilde3 = hessian_tilde3.reshape((self.n_occupied*self.nmo, self.n_occupied*self.nmo))

            #hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end = timer()
            print("build hessian diagonal took", end - start)
            #reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
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
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    #print(r,k,index_count1)
                    #index_count2 = 0
                    #for l in range(self.n_occupied):
                    #   for s in range(l+1,self.nmo):
                    #       if (l < self.n_in_a and s < self.n_in_a): continue
                    #       if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                    #       #if (k >= self.n_occupied and r >= self.n_occupied): continue
                    #       reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                    #       #print(r,k,s,l,index_count1,index_count2)
                    #       index_count2 += 1
                    index_count1 += 1
            #mu1, w1 = np.linalg.eigh(reduced_hessian)
            #print("eigenvalue of the reduced hessian", mu1)
            #print(mu1[0]>0)
            #print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]))
            #print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]))
            #H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            #step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
            #print("norm of critical step", np.linalg.norm(step_limit))
            #print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
            #print("gradient norm", np.linalg.norm(reduced_gradient))
            #if mu1[0] > 0:
            #    step_limit = -np.einsum("pq,q->p", np.linalg.pinv(reduced_hessian), reduced_gradient)
            #    print("norm of newton solution", np.linalg.norm(step_limit))
            ##if microiteration ==0:
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            # if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break

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
                print("\n", current_energy)
                print(
                    "Microiteration",
                    microiteration + 1,
                    "orbital optimization step",
                    orbital_optimization_step + 1,
                    flush=True,
                )
                gradient_norm = np.linalg.norm(reduced_gradient)
                print("gradient norm", gradient_norm, flush=True)
                print("convergence_threshold", convergence_threshold, flush=True)

                # if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                # if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                # if (gradient_norm < 1e-5 and microiteration > 0):
                if gradient_norm < 0.1 * convergence_threshold and microiteration > 0:
                    convergence = 1
                    # print("Microiteration converged (small gradient norm)")
                    break
                if gradient_norm < 1e-7:
                    convergence = 1
                    break
                print("calling pcg for trust region") 
                step_p, oo = self.solve_pcg_trust_region(self.U2, A_tilde2, G_blocks, reduced_gradient, self.reduced_hessian_diagonal, trust_radius, tol=1e-8, max_iter=1000)
                print(oo)
                # new augmented hessian
                beta = 0
                dim00 = self.index_map_size + 1
                augmented_hessian9 = np.zeros((dim00, dim00))
                w9 = np.zeros((dim00, dim00))
                step = np.zeros(self.index_map_size)
                step10_norm = 0.0
                # print("w",w)
                # count10 = 0
                # while np.abs((step10_norm - trust_radius)/trust_radius) > 1e-3:
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
                # step9 = w9[1:,0]
                if np.linalg.norm(reduced_gradient) > 1e-3:
                    if n_negative == 0:
                         step_p, shift_lambda, status = self.solve_gltr_trust_region(
                               self.U2,
                               A_tilde2,
                               G_blocks,
                               reduced_gradient,
                               self.reduced_hessian_diagonal,
                               trust_radius,
                               max_iter=1000,
                               tol=1e-7
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
                             alpha_u = (
                                 delta_u + np.linalg.norm(reduced_gradient) * trust_radius
                             )
                             alpha_l = 0
                             count10 = 0
                             # beta = min(0, alpha_u)
                             beta = alpha_u
                             aug_hessian_eigenvecs = np.zeros((dim00, 2))
                             aug_hessian_eigenvals = np.zeros(2)
                             print("trust radius", trust_radius)
                             self.collapse_subspace_check = False
                             while True:
                                 print("****************************")
                                 print("lstrs iteration", count10, "beta", beta)
                                 beta0 = beta
                                 # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, beta, dim00)
                                 # print("two roots", mu9[0], mu9[1], mu9[2])
                                 if count10 == 0:
                                     alpha_range = np.abs(alpha_u - alpha_l)
                                     Q = self.Davidson_augmented_hessian_solve6(
                                         self.U2,
                                         A_tilde2,
                                         G,
                                         G_blocks,
                                         self.reduced_hessian_diagonal,
                                         reduced_gradient,
                                         beta,
                                         alpha_range,
                                         trust_radius,
                                         aug_hessian_eigenvecs,
                                         aug_hessian_eigenvals,
                                         guess_vector,
                                         restart=False
                                     )
                                     delta_l = aug_hessian_eigenvals[0]
                                     alpha_l = (
                                         delta_l
                                         - np.linalg.norm(reduced_gradient) / trust_radius
                                     )
                                 else:
                                     alpha_range = np.abs(alpha_u - alpha_l)
                                     guess_vector = Q
                                     Q = self.Davidson_augmented_hessian_solve6(
                                         self.U2,
                                         A_tilde2,
                                         G,
                                         G_blocks,
                                         self.reduced_hessian_diagonal,
                                         reduced_gradient,
                                         beta,
                                         alpha_range,
                                         trust_radius,
                                         aug_hessian_eigenvecs,
                                         aug_hessian_eigenvals,
                                         guess_vector,
                                         restart=True
                                     )
                                 # print(aug_hessian_eigenvals)
                                 # print(aug_hessian_eigenvecs)
                                 mu9[0] = aug_hessian_eigenvals[0]
                                 mu9[1] = aug_hessian_eigenvals[1]
                                 w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                                 w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                                 print("ALPHA range", alpha_l, alpha_u)
                                 # idx = mu9.argsort()[:dim00]
                                 # print(idx)
                                 # w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                                 # w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                                 v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                                 v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                                 u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                                 u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                                 aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                 bb1 = np.sqrt(1 - v1 * v1)
                                 aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                                 bb2 = np.sqrt(1 - v2 * v2)
                                 print("aa1, bb1", aa1, bb1, "aa2, bb2", aa2, bb2)
                                 if mu9[0] > -1e-8 and np.linalg.norm(
                                     u1
                                 ) < trust_radius * np.abs(v1):
                                     hard_case = 2
                                     print("use newton step")
                                     break
                                 delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1) / (
                                     1 - v1 * v1
                                 )
                                 delta_u = min(delta_u, delta_u2)
                                 print("delta_u2", delta_u2)

                                 epsilon_v = 1e-4
                                 print(aa1 <= epsilon_v * bb1)
                                 print(aa2 <= epsilon_v * bb2)
                                 alpha = beta
                                 while (
                                     (aa1 <= epsilon_v * bb1)
                                     and (aa2 <= epsilon_v * bb2)
                                     and np.abs(alpha_u - alpha_l)
                                     > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
                                 ):
                                     alpha_u = alpha
                                     alpha = (alpha_l + alpha_u) / 2
                                     w9[:, :] = 0.0
                                     # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, alpha, dim00)
                                     # print("two roots", mu9[0], mu9[1], mu9[2])

                                     alpha_range = np.abs(alpha_u - alpha_l)
                                     Q = self.Davidson_augmented_hessian_solve6(
                                         self.U2,
                                         A_tilde2,
                                         G,
                                         G_blocks,
                                         self.reduced_hessian_diagonal,
                                         reduced_gradient,
                                         alpha,
                                         alpha_range,
                                         trust_radius,
                                         aug_hessian_eigenvecs,
                                         aug_hessian_eigenvals,
                                         Q,
                                         restart=True
                                     )
                                     mu9[0] = aug_hessian_eigenvals[0]
                                     mu9[1] = aug_hessian_eigenvals[1]
                                     w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                                     w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                                     v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                                     v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                                     u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                                     u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                                     aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                     bb1 = np.sqrt(1 - v1 * v1)
                                     aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                                     bb2 = np.sqrt(1 - v2 * v2)
                                     print(
                                         alpha,
                                         aa1 <= epsilon_v * bb1,
                                         aa2 <= epsilon_v * bb2,
                                         mu9[0],
                                         mu9[1],
                                         bb1,
                                         bb2,
                                     )
                                     delta_u2 = mu9[0] - v1 * np.dot(
                                         reduced_gradient, u1
                                     ) / (1 - v1 * v1)
                                     delta_u = min(delta_u, delta_u2)
                                     print("new delta_u", delta_u, delta_u2)

                                 if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
                                     exit()
                                 elif (aa1 > epsilon_v * bb1) and (aa2 > epsilon_v * bb2):
                                     both_roots_normalization = 1
                                 else:
                                     both_roots_normalization = 0
                                 beta = alpha
                                 # update delta_u
                                 # temptemp = np.dot(reduced_hessian, u1)
                                 # temptemp2 = np.dot(u1, temptemp)

                                 # delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1)/np.dot(u1,u1)
                                 # delta_u = min(delta_u, temptemp2/np.dot(u1,u1))
                                 # print("new delta_u", delta_u)
                                 ####scale9 = w9[0][0]
                                 ####print("scale",scale9)
                                 ####if np.abs(scale9) < 1e-15:
                                 ####    break
                                 # assign (k-1) and k-iteration values
                                 lambda1 = lambda2
                                 x1 = x2
                                 phi1 = phi2
                                 print("two roots", mu9[0], mu9[1])
                                 if aa1 > epsilon_v * bb1:
                                     step10 = u1 / v1
                                     second_component[:] = u2
                                     lambda2 = mu9[0]
                                     print("norm from first root", np.linalg.norm(step10))
                                     # if aa2 > epsilon_v * bb2:
                                     #    second_component[:] = u2/v2
                                     #    print("second root norm", np.linalg.norm(second_component))
                                     if np.linalg.norm(step10) < trust_radius:
                                         alpha_l = beta
                                     if np.linalg.norm(step10) > trust_radius:
                                         alpha_u = beta
                                 else:
                                     step10 = u2 / v2
                                     second_component[:] = u1
                                     lambda2 = mu9[1]
                                     print("norm from second root", np.linalg.norm(step10))
                                     alpha_u = beta
                                 print("NEW ALPHA range", alpha_l, alpha_u)
                                 step10_norm = np.linalg.norm(step10)
                                 x2 = step10
                                 if mu9[0] > -1e-8 and np.linalg.norm(
                                     u1
                                 ) < trust_radius * np.abs(v1):
                                     hard_case = 2
                                     print("use newton step")
                                     break
                                 phi2 = -np.dot(reduced_gradient, x2)
                                 # print(step10)
                                 # print(x1)
                                 # print(x2)
                                 # if count10 == 0 and step10_norm < trust_radius: break
                                 # print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                                 # print(mu9, flush=True)
                                 print(
                                     "step10 norm",
                                     step10_norm,
                                     "current beta",
                                     beta,
                                     flush=True,
                                 )
                                 x1_norm = np.linalg.norm(x1)
                                 x2_norm = np.linalg.norm(x2)
                                 phi2_p = np.dot(x2, x2)
                                 phi1_p = np.dot(x1, x1)
                                 print(
                                     "phi1",
                                     phi1,
                                     "phi1_p",
                                     phi1_p,
                                     "phi2",
                                     phi2,
                                     "phi2_p",
                                     phi2_p,
                                 )
                                 if (
                                     np.abs((step10_norm - trust_radius) / trust_radius)
                                     <= 1e-3
                                     and mu9[0] <= 0
                                 ):
                                     step = step10
                                     hard_case = 0
                                     break
                                 print("check orthogonality of the first two roots")
                                 vv1 = w9[0, 0]
                                 vv2 = w9[0, 1]
                                 uu1 = w9[1:, 0]
                                 uu2 = w9[1:, 1]
                                 # check if the quasi-optimal condition is satisfied
                                 print(
                                     "quasi-optimal condition",
                                     (1 + trust_radius * trust_radius)
                                     * (vv1 * vv1 + vv2 * vv2),
                                     flush=True,
                                 )
                                 qs = (1 + trust_radius * trust_radius) * (
                                     vv1 * vv1 + vv2 * vv2
                                 )
                                 print(np.dot(w9[:, :2].T, w9[:, :2]))

                                 # construct quasi-optimal step
                                 epsilon_hc = 1e-6 
                                 eta = epsilon_hc / (1 - epsilon_hc)
                                 tau1 = 1
                                 tau2 = 1
                                 if qs > 1.0:
                                     tau1 = (
                                         (vv1 - vv2 * np.sqrt(qs - 1))
                                         / (vv1 * vv1 + vv2 * vv2)
                                         / np.sqrt(1 + trust_radius * trust_radius)
                                     )
                                     tau2 = (
                                         (vv2 + vv1 * np.sqrt(qs - 1))
                                         / (vv1 * vv1 + vv2 * vv2)
                                         / np.sqrt(1 + trust_radius * trust_radius)
                                     )
                                 elif np.abs(qs - 1.0) < 2e-308:
                                     tau1 = vv1 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                                     tau2 = vv2 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                                 x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                     tau1 * vv1 + tau2 * vv2
                                 )

                                 lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                                 psi_tilde = 0.5 * self.microiteration_predicted_energy2(
                                     self.U2, reduced_gradient, A_tilde2, G_blocks, x_tilde
                                 )
                                 if qs > 1.0 or np.abs(qs - 1.0) < 2e-308:
                                     print(
                                         "x_tilde norm1",
                                         np.linalg.norm(x_tilde),
                                         (mu9[1] - mu9[0])
                                         * tau2
                                         * tau2
                                         * (1 + trust_radius * trust_radius),
                                         -2.0 * eta * psi_tilde,
                                     )
                                     if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                         1 + trust_radius * trust_radius
                                     ) < -2.0 * eta * psi_tilde:
                                         print("find quasi-optimal solution1", flush=True)
                                         hard_case = 3
                                         break
                                     else:
                                         if qs > 1.0:
                                             tau1 = (
                                                 (vv1 + vv2 * np.sqrt(qs - 1))
                                                 / (vv1 * vv1 + vv2 * vv2)
                                                 / np.sqrt(1 + trust_radius * trust_radius)
                                             )
                                             tau2 = (
                                                 (vv2 - vv1 * np.sqrt(qs - 1))
                                                 / (vv1 * vv1 + vv2 * vv2)
                                                 / np.sqrt(1 + trust_radius * trust_radius)
                                             )
                                             x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                                 tau1 * vv1 + tau2 * vv2
                                             )
                                             lambda_tilde = (
                                                 tau1 * tau1 * lambda1
                                                 + tau2 * tau2 * lambda2
                                             )
                                             psi_tilde = (
                                                 0.5
                                                 * self.microiteration_predicted_energy2(
                                                     self.U2,
                                                     reduced_gradient,
                                                     A_tilde2,
                                                     G_blocks,
                                                     x_tilde,
                                                 )
                                             )
                                             print(
                                                 "x_tilde norm2",
                                                 np.linalg.norm(x_tilde),
                                                 (mu9[1] - mu9[0])
                                                 * tau2
                                                 * tau2
                                                 * (1 + trust_radius * trust_radius),
                                                 -2.0 * eta * psi_tilde,
                                                 flush=True,
                                             )
                                             if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                                 1 + trust_radius * trust_radius
                                             ) < -2.0 * eta * psi_tilde:
                                                 print(
                                                     "find quasi-optimal solution2",
                                                     flush=True,
                                                 )
                                                 hard_case = 3
                                                 break
                                 if np.abs(alpha_u - alpha_l) <= 1e-8 * max(
                                     np.abs(alpha_u), np.abs(alpha_l)
                                 ):
                                     print("interval too small")
                                     if (
                                         x2_norm < trust_radius
                                         and both_roots_normalization == 0
                                     ):
                                         hard_case = 1
                                     if both_roots_normalization == 1:
                                         hard_case = 4
                                     break

                                 if count10 == 0:
                                     beta = beta + (beta - lambda2) / x2_norm * (
                                         trust_radius - x2_norm
                                     ) / trust_radius * (trust_radius + 1.0 / x2_norm)
                                 else:
                                     print("denominator", x2_norm - x1_norm)
                                     denom = trust_radius * (x2_norm - x1_norm)

                                     if np.abs(denom) > 2e-308:
                                         lambda_c = (
                                             lambda1 * x1_norm * (x2_norm - trust_radius)
                                             + lambda2 * x2_norm * (trust_radius - x1_norm)
                                         ) / denom
                                     else:
                                         lambda_c = delta_u
                                     if lambda_c > delta_u:
                                         print(
                                             "need to safeguard delta_u", lambda_c, delta_u
                                         )
                                         lambda_c = delta_u
                                     # lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                     # print(lambda_c)
                                     if np.abs(lambda2 - lambda1) <= 2e-308:
                                         print("interpolate alpha by bisection")
                                         beta = (alpha_l + alpha_u) / 2
                                     else:
                                         omega_k = (lambda2 - lambda_c) / (lambda2 - lambda1)
                                         num = (
                                             x1_norm
                                             * x2_norm
                                             * (x2_norm - x1_norm)
                                             * (lambda1 - lambda_c)
                                             * (lambda2 - lambda_c)
                                         )

                                         denom = (
                                             omega_k * x2_norm + (1 - omega_k) * x1_norm
                                         ) * (lambda2 - lambda1)
                                         print(x1_norm, x2_norm)
                                         print(lambda1, lambda2, lambda_c)
                                         print(num, denom)
                                         if np.abs(denom) <= 2e-308:
                                             print("interpolate alpha by bisection2")
                                             beta = (alpha_l + alpha_u) / 2
                                         else:
                                             beta = (
                                                 lambda_c
                                                 + omega_k * phi1
                                                 + (1 - omega_k) * phi2
                                                 + num / denom
                                             )

                                 # if count10 == 0 or (np.abs(x2_norm - x1_norm) <= 1e-15):
                                 #    if (np.abs(x2_norm - x1_norm) <= 1e-15):
                                 #        print("too small denominator")
                                 #    beta = beta + (beta - lambda2)/x2_norm * (trust_radius - x2_norm)/trust_radius * (trust_radius + 1.0/x2_norm)
                                 # else:
                                 #    print("denominator", x2_norm-x1_norm)
                                 #    if (np.abs(x2_norm - x1_norm) > 1e-15):
                                 #        lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                 #    else:
                                 #        lambda_c = delta_u
                                 #    if lambda_c > delta_u:
                                 #        print("need to safeguard delta_u", lambda_c, delta_u)
                                 #        lambda_c = delta_u
                                 #    #lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                 #    #print(lambda_c)
                                 #    omega_k = (lambda2 - lambda_c)/(lambda2 - lambda1)
                                 #    ratio1 = x1_norm * x2_norm * (x2_norm - x1_norm)/(omega_k * x2_norm + (1-omega_k) * x1_norm)
                                 #    ratio2 = (lambda1 - lambda_c) * (lambda2 - lambda_c) / (lambda2 - lambda1)
                                 #    beta = lambda_c + omega_k *phi1 + (1-omega_k) * phi2 + ratio1 * ratio2
                                 print("beta after interpolation", beta)
                                 # safeguard alpha
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
                                         print("using bisection")
                                         beta = (alpha_l + alpha_u) / 2
                                 count10 += 1
                                 # if np.abs(beta -beta0) <= 1e-12:
                                 #    print("possible convergence")
                                 #    print(beta, beta0)
                                 #    print(x2_norm)

                                 #    if x2_norm < trust_radius:
                                 #        print("11111111")
                                 #        hard_case = 1
                                 #    elif x2_norm > trust_radius and both_roots_normalization == 1:
                                 #        print("22222222")
                                 #        hard_case = 1

                                 #    break
                                 if count10 == 50:
                                     break
                             print("check hard case", hard_case)
                             if hard_case == 1:
                                 first_component[:] = x2[:]
                                 if x2_norm > trust_radius:
                                     print("some logic is wrong")
                                     exit()
                                     # temp_vector = np.zeros(self.index_map_size)
                                     # temp_vector[:] = second_component[:]
                                     # second_component[:] = first_component[:]
                                     # first_component[:] = temp_vector[:]
                                 # print(first_component)
                                 # print(second_component)
                                 print(
                                     "norm of current root", np.linalg.norm(first_component)
                                 )
                                 xy_square = np.dot(
                                     first_component, second_component
                                 ) * np.dot(first_component, second_component)
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
                                 adjusted_step = (
                                     first_component + min(np.abs(t1), np.abs(t2)) * second_component
                                 )
                                 print("adjusted step norm", np.linalg.norm(adjusted_step))
                                 step = adjusted_step

                             if hard_case == 2:
                                 #Q = np.zeros((1, self.index_map_size))
                                 #H1_op = LinearOperator(
                                 #    (self.index_map_size, self.index_map_size),
                                 #    matvec=lambda Q: self.mv2(
                                 #        self.U2, A_tilde2, G1, Q, 1, 0, 0
                                 #    ),
                                 #)
                                 #x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
                                 #print("exitcode", exitCode)
                                 #step = x
                                 denom = self.reduced_hessian_diagonal
                                 max_iter = 20  
                                 _solve_newton = self.minres_solve if _QED_USE_MINRES else self.linear_equation_solve
                                 solution, converged = _solve_newton(self.U2, A_tilde2, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6)
                                 if not converged:
                                     print("\n--- Falling back to MINRES solver ---")
                                     iteration_count = [0]
                                     def callback(xk):
                                         """Callback function called at each iteration"""
                                         iteration_count[0] += 1
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
                                 hard_case = 2
                                 step = solution


                             if hard_case == 3:
                                 step = x_tilde
                             if hard_case == 4:
                                 if x2_norm > trust_radius:
                                     step = x2 / x2_norm * trust_radius
                                 if x2_norm < trust_radius:
                                     step = x2
                             

                else:
                    print("gradient is small, use Newton step")
                    #Q = np.zeros((1, self.index_map_size))
                    #H1_op = LinearOperator(
                    #    (self.index_map_size, self.index_map_size),
                    #    matvec=lambda Q: self.mv2(self.U2, A_tilde2, G1, Q, 1, 0, 0),
                    #)
                    #x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
                    #print("exitcode", exitCode)
                    denom = self.reduced_hessian_diagonal
                    max_iter = 20  
                    _solve_newton = self.minres_solve if _QED_USE_MINRES else self.linear_equation_solve
                    solution,converged = _solve_newton(self.U2, A_tilde2, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6)
                    if not converged:
                        print("\n--- Falling back to MINRES solver ---")
                        iteration_count = [0]
                        def callback(xk):
                            """Callback function called at each iteration"""
                            iteration_count[0] += 1
                            absolute_residual = np.linalg.norm(H1_op.matvec(xk) + reduced_gradient)
                            residual = np.linalg.norm(H1_op.matvec(xk) + reduced_gradient)
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
                    hard_case = 2
                    step = solution
                    #print(step)
                    #print(solution)
                # w[:,0] = w[:,0]/scale

                step_norm = np.linalg.norm(step)

                print("step norm", step_norm)

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
                if predicted_energy2 < 0:
                    reduce_predicted_energy = 1
                else:
                    reduce_predicted_energy = 0

                print("microiteration predicted energy", predicted_energy2, flush=True)
                end = timer()
                print("build unitary matrix and recheck energy took", end - start)

                if microiteration == 0 and orbital_optimization_step == 0:
                    convergence_threshold = min(
                        0.01 * gradient_norm, np.power(gradient_norm, 2)
                    )
                if energy_change < 0.0 or hard_case == 2:
                    print("step norm", np.linalg.norm(step))
                    if step_norm < 0.05: 
                        print("activate qn optimization")
                        qn_optimization = True
                        qn_count +=1
                        N_microiterations = 20  
                        N_orbital_optimization_steps= 1
                        print("number of microiteration left",N_microiterations, "number of orbital optimization", N_orbital_optimization_steps, "qn_count", qn_count)
                    self.U_zero = self.U2 
                    #else:
                    #    qn_optimization = False
                    # restart = False
                    # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                    # predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                    # print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                    self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                    if microiteration == 0 and orbital_optimization_step == 0:
                        # convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                        # convergence_threshold = 0.01 * gradient_norm
                        if step_norm > 0.1:
                            N_microiterations = 5
                            N_orbital_optimization_steps = 4
                        elif step_norm <= 0.1 and step_norm > 0.05:
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
                    trust_radius = self.step_control(ratio, trust_radius)

                    #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                    # print(np.shape(gradient_tilde), flush = True)

                    start = timer()
                    self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
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
                            reduced_gradient[index_count1] = gradient_tilde[r][k]
                            ###print(r,k,index_count1)
                            ##index_count2 = 0
                            ##for l in range(self.n_occupied):
                            ##   for s in range(l+1,self.nmo):
                            ##       if (l < self.n_in_a and s < self.n_in_a): continue
                            ##       if (self.n_in_a <= l < self.n_occupied and self.n_in_a <= s < self.n_occupied): continue
                            ##       #if (k >= self.n_occupied and r >= self.n_occupied): continue
                            ##       reduced_hessian[index_count1][index_count2] = hessian_tilde3[r*self.n_occupied+k][s*self.n_occupied+l]
                            ##       #print(r,k,s,l,index_count1,index_count2)
                            ##       index_count2 += 1
                            index_count1 += 1
                    # print("reduced_gradient",reduced_gradient, flush = True)

                    #mu1, w1 = np.linalg.eigh(reduced_hessian)
                    #print("eigenvalue of the reduced hessian", mu1)
                    # print("dot product of gradient and first eigenvector of hessian", np.dot(reduced_gradient, w1[:,0]))
                    # print("dot product of gradient and second eigenvector of hessian", np.dot(reduced_gradient, w1[:,1]))
                    # H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                    # step_limit = -np.einsum("pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient)
                    # print("norm of critical step", np.linalg.norm(step_limit))
                    # print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
                    # print("gradient norm", np.linalg.norm(reduced_gradient))

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
                    # active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
                    # active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
                    # active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
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
                    current_energy = zero_energy + second_order_energy_change
                    hard_case = 0

                else:
                    # restart = False
                    # guess_vector = trial_vector
                    reduce_step = 1
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush=True)
                    print("new trust radius", trust_radius, flush=True)
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

            if convergence == 1 and count == 0:
                active_twoeint[:, :, :, :] = self.active_twoeint[:, :, :, :]
                active_fock_core[:, :] = self.active_fock_core[:, :]
                d_cmo[:, :] = self.d_cmo[:, :]

                # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
                # temp8 = np.zeros((self.nmo, self.nmo))
                # temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
                # self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                # temp8 = np.einsum("pq,qs->ps", self.d_cmo, self.U2)
                # self.d_cmo[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
                # break

            start = timer()
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
                self.E_core2,
                self.omega,
                self.Enuc,
                self.d_c,
                self.Y,
                self.target_spin,
            )
            print("qn status", qn_optimization, "qn count", qn_count, flush = True)
            print("gradient norm", np.linalg.norm(reduced_gradient), flush = True)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[: self.n_in_a, : self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9
            self.constdouble[5] = self.E_core2
            self.constint[8] = 5
            if qn_count > 0: 
                self.constdouble[4] = 0.1 * np.linalg.norm(reduced_gradient)
                self.constint[8] = 10000
            print("number of CI iteration", self.constint[8])
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
                if qn_count ==1 or self.density_norm_change <0.025 or energy_change < 0:
                    self.U222 = self.U_zero              
                    print("ahihi")
            end = timer()
            print("building RDM took", end - start)
            #active_one_e_energy = np.dot(active_fock_core.flatten(), self.D_tu_avg)
            #active_two_e_energy = 0.5 * np.dot(active_twoeint.flatten(), self.D_tuvw_avg)
            #active_one_pe_energy = -np.sqrt(self.omega/2) * np.dot(d_cmo[self.n_in_a:self.n_occupied,self.n_in_a:self.n_occupied].flatten(), self.Dpe_tu_avg)
            #ci_dependent_energy = self.calculate_ci_dependent_energy(eigenvecs, d_cmo)
            #sum_energy = (active_one_e_energy + active_two_e_energy + active_one_pe_energy + self.E_core2 +
            #       self.Enuc + self.d_c + ci_dependent_energy)
            #print("RDM_energy    active_one    active_two E_core active_pe_energy ci_dependent_energy E_nuc")
            #print("gfhgy",
            #   "{:20.12f}".format(sum_energy),
            #   "{:20.12f}".format(active_one_e_energy),
            #   "{:20.12f}".format(active_two_e_energy),
            #   "{:20.12f}".format(self.E_core2),
            #   "{:20.12f}".format(active_one_pe_energy),
            #   "{:20.12f}".format(ci_dependent_energy),
            #   "{:20.12f}".format(self.Enuc),
            #   flush = True
            #)
            #print("current gradient_norm and residual", gradient_norm, current_residual)
            #print("current convergence_threshold", convergence_threshold)
            #total_norm = np.sqrt(np.power(gradient_norm,2) + np.power(current_residual,2))
            #if total_norm < convergence_threshold:
            #   print("total norm", total_norm, flush = True)
            #   #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #   #temp8 = np.zeros((self.nmo, self.nmo))
            #   #temp8 = np.einsum("pq,qs->ps", self.H_spatial2, self.U2)
            #   #self.H_spatial2[:,:] = np.einsum("ps,pr->rs", temp8, self.U2)
            #   #self.d_cmo[:,:] = d_cmo[:,:]
            #   #print(eigenvecs)
            #   #print("u2i",self.U2)
            #   print("microiteration converged! (small total norm)", flush = True)
            #   break

            microiteration += 1

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

    def microiteration_optimization6(
        self, U, eigenvecs, c_get_roots, convergence_threshold
    ):
        # print("E_core", self.E_core)
        self.U2 = copy.deepcopy(U)
        trust_radius = 0.5
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
                trust_radius = 0.5
            else:
                print("carried trust radius", trust_radius, flush=True)
            A[:, :] = 0.0
            G[:, :, :, :] = 0.0
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
                < max(0.01 * convergence_threshold, 1e-10)
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
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
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
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    index_count1 += 1
            #print("reduced_gradient",reduced_gradient, flush = True)
            new_reduced_gradient = copy.deepcopy(reduced_gradient)
           
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
                    _stats_span = _stats_begin("gltr_qn")
                    step_p, shift_lambda, status = self.solve_gltr_trust_region(
                                   self.U_zero,
                                   self.A_tilde_zero,
                                   self.G_blocks_zero,
                                   reduced_gradient,
                                   self.reduced_hessian_diagonal_zero,
                                   trust_radius,
                                   max_iter=1000,
                                   tol=1e-7
                         )
                    _stats_end(_stats_span, status=status)
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
                    _stats_span = _stats_begin("gltr_qn_bfgs")
                    step_p, shift_lambda, status= self.solve_gltr_with_operator(B_op, reduced_gradient,
                                                          self.reduced_hessian_diagonal_zero,
                                                                            trust_radius,
                                                                            max_iter=1000,
                                                                            tol=1e-7)
                    _stats_end(_stats_span, status=status)
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
                qn_count +=1
                self.U2 = np.einsum("pq,qs->ps", self.U2, self.U_delta)
                
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
                    print("\n", current_energy)
                    print(
                        "Microiteration",
                        microiteration + 1,
                        "orbital optimization step",
                        orbital_optimization_step + 1,
                        flush=True,
                    )
                    gradient_norm = np.linalg.norm(reduced_gradient)
                    print("gradient norm", gradient_norm, flush=True)
                    print("convergence_threshold", convergence_threshold, flush=True)

                    # if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                    # if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                    # if (gradient_norm < 1e-5 and microiteration > 0):
                    if gradient_norm < 0.1 * convergence_threshold and microiteration > 0:
                        convergence = 1
                        # print("Microiteration converged (small gradient norm)")
                        break
                    if gradient_norm < 1e-7:
                        convergence = 1
                        break
                    #print("calling pcg for trust region") 
                    #step_p, oo = self.solve_pcg_trust_region(self.U2, A_tilde2, G1, reduced_gradient, self.reduced_hessian_diagonal, trust_radius, tol=1e-8, max_iter=1000)
                    #print(oo)
                    # new augmented hessian
                    beta = 0
                    dim00 = self.index_map_size + 1
                    augmented_hessian9 = np.zeros((dim00, dim00))
                    w9 = np.zeros((dim00, dim00))
                    step = np.zeros(self.index_map_size)
                    step10_norm = 0.0
                    
                    if np.linalg.norm(reduced_gradient) > 1e-3:
                        if n_negative == 0:
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
                                   tol=1e-7
                             )
                             _stats_end(_stats_span, status=status)
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
                                 alpha_u = (
                                     delta_u + np.linalg.norm(reduced_gradient) * trust_radius
                                 )
                                 alpha_l = 0
                                 count10 = 0
                                 # beta = min(0, alpha_u)
                                 beta = alpha_u
                                 aug_hessian_eigenvecs = np.zeros((dim00, 2))
                                 aug_hessian_eigenvals = np.zeros(2)
                                 print("trust radius", trust_radius)
                                 self.collapse_subspace_check = False
                                 _stats_span = _stats_begin("lstrs_davidson")
                                 while True:
                                     print("****************************")
                                     print("lstrs iteration", count10, "beta", beta)
                                     beta0 = beta
                                     # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, beta, dim00)
                                     # print("two roots", mu9[0], mu9[1], mu9[2])
                                     if count10 == 0:
                                         alpha_range = np.abs(alpha_u - alpha_l)
                                         Q = self.Davidson_augmented_hessian_solve6(
                                             self.U2,
                                             A_tilde2,
                                             G,
                                             G_blocks,
                                             self.reduced_hessian_diagonal,
                                             reduced_gradient,
                                             beta,
                                             alpha_range,
                                             trust_radius,
                                             aug_hessian_eigenvecs,
                                             aug_hessian_eigenvals,
                                             guess_vector,
                                             restart=False
                                         )
                                         delta_l = aug_hessian_eigenvals[0]
                                         alpha_l = (
                                             delta_l
                                             - np.linalg.norm(reduced_gradient) / trust_radius
                                         )
                                     else:
                                         alpha_range = np.abs(alpha_u - alpha_l)
                                         guess_vector = Q
                                         Q = self.Davidson_augmented_hessian_solve6(
                                             self.U2,
                                             A_tilde2,
                                             G,
                                             G_blocks,
                                             self.reduced_hessian_diagonal,
                                             reduced_gradient,
                                             beta,
                                             alpha_range,
                                             trust_radius,
                                             aug_hessian_eigenvecs,
                                             aug_hessian_eigenvals,
                                             guess_vector,
                                             restart=True
                                         )
                                     # print(aug_hessian_eigenvals)
                                     # print(aug_hessian_eigenvecs)
                                     mu9[0] = aug_hessian_eigenvals[0]
                                     mu9[1] = aug_hessian_eigenvals[1]
                                     w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                                     w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                                     print("ALPHA range", alpha_l, alpha_u)
                                     # idx = mu9.argsort()[:dim00]
                                     # print(idx)
                                     # w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                                     # w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                                     v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                                     v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                                     u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                                     u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                                     aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                     bb1 = np.sqrt(1 - v1 * v1)
                                     aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                                     bb2 = np.sqrt(1 - v2 * v2)
                                     print("aa1, bb1", aa1, bb1, "aa2, bb2", aa2, bb2)
                                     if mu9[0] > -1e-8 and np.linalg.norm(
                                         u1
                                     ) < trust_radius * np.abs(v1):
                                         hard_case = 2
                                         print("use newton step")
                                         break
                                     delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1) / (
                                         1 - v1 * v1
                                     )
                                     delta_u = min(delta_u, delta_u2)
                                     print("delta_u2", delta_u2)

                                     epsilon_v = 1e-4
                                     print(aa1 <= epsilon_v * bb1)
                                     print(aa2 <= epsilon_v * bb2)
                                     alpha = beta
                                     while (
                                         (aa1 <= epsilon_v * bb1)
                                         and (aa2 <= epsilon_v * bb2)
                                         and np.abs(alpha_u - alpha_l)
                                         > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
                                     ):
                                         alpha_u = alpha
                                         alpha = (alpha_l + alpha_u) / 2
                                         w9[:, :] = 0.0
                                         # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, alpha, dim00)
                                         # print("two roots", mu9[0], mu9[1], mu9[2])

                                         alpha_range = np.abs(alpha_u - alpha_l)
                                         Q = self.Davidson_augmented_hessian_solve6(
                                             self.U2,
                                             A_tilde2,
                                             G,
                                             G_blocks,
                                             self.reduced_hessian_diagonal,
                                             reduced_gradient,
                                             alpha,
                                             alpha_range,
                                             trust_radius,
                                             aug_hessian_eigenvecs,
                                             aug_hessian_eigenvals,
                                             Q,
                                             restart=True
                                         )
                                         mu9[0] = aug_hessian_eigenvals[0]
                                         mu9[1] = aug_hessian_eigenvals[1]
                                         w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                                         w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                                         v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                                         v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                                         u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                                         u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                                         aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                         bb1 = np.sqrt(1 - v1 * v1)
                                         aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                                         bb2 = np.sqrt(1 - v2 * v2)
                                         print(
                                             alpha,
                                             aa1 <= epsilon_v * bb1,
                                             aa2 <= epsilon_v * bb2,
                                             mu9[0],
                                             mu9[1],
                                             bb1,
                                             bb2,
                                         )
                                         delta_u2 = mu9[0] - v1 * np.dot(
                                             reduced_gradient, u1
                                         ) / (1 - v1 * v1)
                                         delta_u = min(delta_u, delta_u2)
                                         print("new delta_u", delta_u, delta_u2)

                                     if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
                                         exit()
                                     elif (aa1 > epsilon_v * bb1) and (aa2 > epsilon_v * bb2):
                                         both_roots_normalization = 1
                                     else:
                                         both_roots_normalization = 0
                                     beta = alpha
                                     # update delta_u
                                     # temptemp = np.dot(reduced_hessian, u1)
                                     # temptemp2 = np.dot(u1, temptemp)

                                     # delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1)/np.dot(u1,u1)
                                     # delta_u = min(delta_u, temptemp2/np.dot(u1,u1))
                                     # print("new delta_u", delta_u)
                                     ####scale9 = w9[0][0]
                                     ####print("scale",scale9)
                                     ####if np.abs(scale9) < 1e-15:
                                     ####    break
                                     # assign (k-1) and k-iteration values
                                     lambda1 = lambda2
                                     x1 = x2
                                     phi1 = phi2
                                     print("two roots", mu9[0], mu9[1])
                                     if aa1 > epsilon_v * bb1:
                                         step10 = u1 / v1
                                         second_component[:] = u2
                                         lambda2 = mu9[0]
                                         print("norm from first root", np.linalg.norm(step10))
                                         # if aa2 > epsilon_v * bb2:
                                         #    second_component[:] = u2/v2
                                         #    print("second root norm", np.linalg.norm(second_component))
                                         if np.linalg.norm(step10) < trust_radius:
                                             alpha_l = beta
                                         if np.linalg.norm(step10) > trust_radius:
                                             alpha_u = beta
                                     else:
                                         step10 = u2 / v2
                                         second_component[:] = u1
                                         lambda2 = mu9[1]
                                         print("norm from second root", np.linalg.norm(step10))
                                         alpha_u = beta
                                     print("NEW ALPHA range", alpha_l, alpha_u)
                                     step10_norm = np.linalg.norm(step10)
                                     x2 = step10
                                     if mu9[0] > -1e-8 and np.linalg.norm(
                                         u1
                                     ) < trust_radius * np.abs(v1):
                                         hard_case = 2
                                         print("use newton step")
                                         break
                                     phi2 = -np.dot(reduced_gradient, x2)
                                     # print(step10)
                                     # print(x1)
                                     # print(x2)
                                     # if count10 == 0 and step10_norm < trust_radius: break
                                     # print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                                     # print(mu9, flush=True)
                                     print(
                                         "step10 norm",
                                         step10_norm,
                                         "current beta",
                                         beta,
                                         flush=True,
                                     )
                                     x1_norm = np.linalg.norm(x1)
                                     x2_norm = np.linalg.norm(x2)
                                     phi2_p = np.dot(x2, x2)
                                     phi1_p = np.dot(x1, x1)
                                     print(
                                         "phi1",
                                         phi1,
                                         "phi1_p",
                                         phi1_p,
                                         "phi2",
                                         phi2,
                                         "phi2_p",
                                         phi2_p,
                                     )
                                     if (
                                         np.abs((step10_norm - trust_radius) / trust_radius)
                                         <= _QED_LSTRS_NORM_TOL
                                         and mu9[0] <= 0
                                     ):
                                         step = step10
                                         hard_case = 0
                                         break
                                     print("check orthogonality of the first two roots")
                                     vv1 = w9[0, 0]
                                     vv2 = w9[0, 1]
                                     uu1 = w9[1:, 0]
                                     uu2 = w9[1:, 1]
                                     # check if the quasi-optimal condition is satisfied
                                     print(
                                         "quasi-optimal condition",
                                         (1 + trust_radius * trust_radius)
                                         * (vv1 * vv1 + vv2 * vv2),
                                         flush=True,
                                     )
                                     qs = (1 + trust_radius * trust_radius) * (
                                         vv1 * vv1 + vv2 * vv2
                                     )
                                     print(np.dot(w9[:, :2].T, w9[:, :2]))

                                     # construct quasi-optimal step
                                     epsilon_hc = _QED_LSTRS_EPS_HC  # QED_LSTRS_EPS_HC, default 1e-6
                                     eta = epsilon_hc / (1 - epsilon_hc)
                                     tau1 = 1
                                     tau2 = 1
                                     if qs > 1.0:
                                         tau1 = (
                                             (vv1 - vv2 * np.sqrt(qs - 1))
                                             / (vv1 * vv1 + vv2 * vv2)
                                             / np.sqrt(1 + trust_radius * trust_radius)
                                         )
                                         tau2 = (
                                             (vv2 + vv1 * np.sqrt(qs - 1))
                                             / (vv1 * vv1 + vv2 * vv2)
                                             / np.sqrt(1 + trust_radius * trust_radius)
                                         )
                                     elif np.abs(qs - 1.0) < 2e-308:
                                         tau1 = vv1 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                                         tau2 = vv2 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                                     x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                         tau1 * vv1 + tau2 * vv2
                                     )

                                     lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                                     psi_tilde = 0.5 * self.microiteration_predicted_energy2(
                                         self.U2, reduced_gradient, A_tilde2, G_blocks, x_tilde
                                     )
                                     # The quasi-optimal test below reads mu9[1]/vv2/uu2, i.e. the
                                     # SECOND root -- but Davidson_augmented_hessian_solve6 only
                                     # converges it once alpha_range <= 1e-5 (nroots 1 -> 2, ~16350).
                                     # While the range is wide, root 1 is whatever the current
                                     # subspace happens to give, so accepting x_tilde on it would
                                     # act on an unconverged vector.  At the default eps_hc = 1e-6
                                     # the test never passes and this is moot (c10h8: 54 x_tilde
                                     # constructions, 0 hard_case=3 exits), but loosening eps_hc
                                     # removes that accidental protection -- so gate the exit on
                                     # the same condition that makes the second root real.
                                     _second_root_converged = alpha_range <= 1e-5
                                     if (qs > 1.0 or np.abs(qs - 1.0) < 2e-308) and _second_root_converged:
                                         print(
                                             "x_tilde norm1",
                                             np.linalg.norm(x_tilde),
                                             (mu9[1] - mu9[0])
                                             * tau2
                                             * tau2
                                             * (1 + trust_radius * trust_radius),
                                             -2.0 * eta * psi_tilde,
                                         )
                                         if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                             1 + trust_radius * trust_radius
                                         ) < -2.0 * eta * psi_tilde:
                                             print("find quasi-optimal solution1", flush=True)
                                             hard_case = 3
                                             break
                                         else:
                                             if qs > 1.0:
                                                 tau1 = (
                                                     (vv1 + vv2 * np.sqrt(qs - 1))
                                                     / (vv1 * vv1 + vv2 * vv2)
                                                     / np.sqrt(1 + trust_radius * trust_radius)
                                                 )
                                                 tau2 = (
                                                     (vv2 - vv1 * np.sqrt(qs - 1))
                                                     / (vv1 * vv1 + vv2 * vv2)
                                                     / np.sqrt(1 + trust_radius * trust_radius)
                                                 )
                                                 x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                                     tau1 * vv1 + tau2 * vv2
                                                 )
                                                 lambda_tilde = (
                                                     tau1 * tau1 * lambda1
                                                     + tau2 * tau2 * lambda2
                                                 )
                                                 psi_tilde = (
                                                     0.5
                                                     * self.microiteration_predicted_energy2(
                                                         self.U2,
                                                         reduced_gradient,
                                                         A_tilde2,
                                                         G_blocks,
                                                         x_tilde,
                                                     )
                                                 )
                                                 print(
                                                     "x_tilde norm2",
                                                     np.linalg.norm(x_tilde),
                                                     (mu9[1] - mu9[0])
                                                     * tau2
                                                     * tau2
                                                     * (1 + trust_radius * trust_radius),
                                                     -2.0 * eta * psi_tilde,
                                                     flush=True,
                                                 )
                                                 if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                                     1 + trust_radius * trust_radius
                                                 ) < -2.0 * eta * psi_tilde:
                                                     print(
                                                         "find quasi-optimal solution2",
                                                         flush=True,
                                                     )
                                                     hard_case = 3
                                                     break
                                     if np.abs(alpha_u - alpha_l) <= 1e-8 * max(
                                         np.abs(alpha_u), np.abs(alpha_l)
                                     ):
                                         print("interval too small")
                                         if (
                                             x2_norm < trust_radius
                                             and both_roots_normalization == 0
                                         ):
                                             hard_case = 1
                                         if both_roots_normalization == 1:
                                             hard_case = 4
                                         break

                                     if count10 == 0:
                                         beta = beta + (beta - lambda2) / x2_norm * (
                                             trust_radius - x2_norm
                                         ) / trust_radius * (trust_radius + 1.0 / x2_norm)
                                     else:
                                         print("denominator", x2_norm - x1_norm)
                                         denom = trust_radius * (x2_norm - x1_norm)

                                         if np.abs(denom) > 2e-308:
                                             lambda_c = (
                                                 lambda1 * x1_norm * (x2_norm - trust_radius)
                                                 + lambda2 * x2_norm * (trust_radius - x1_norm)
                                             ) / denom
                                         else:
                                             lambda_c = delta_u
                                         if lambda_c > delta_u:
                                             print(
                                                 "need to safeguard delta_u", lambda_c, delta_u
                                             )
                                             lambda_c = delta_u
                                         # lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                         # print(lambda_c)
                                         if np.abs(lambda2 - lambda1) <= 2e-308:
                                             print("interpolate alpha by bisection")
                                             beta = (alpha_l + alpha_u) / 2
                                         else:
                                             omega_k = (lambda2 - lambda_c) / (lambda2 - lambda1)
                                             num = (
                                                 x1_norm
                                                 * x2_norm
                                                 * (x2_norm - x1_norm)
                                                 * (lambda1 - lambda_c)
                                                 * (lambda2 - lambda_c)
                                             )

                                             denom = (
                                                 omega_k * x2_norm + (1 - omega_k) * x1_norm
                                             ) * (lambda2 - lambda1)
                                             print(x1_norm, x2_norm)
                                             print(lambda1, lambda2, lambda_c)
                                             print(num, denom)
                                             if np.abs(denom) <= 2e-308:
                                                 print("interpolate alpha by bisection2")
                                                 beta = (alpha_l + alpha_u) / 2
                                             else:
                                                 beta = (
                                                     lambda_c
                                                     + omega_k * phi1
                                                     + (1 - omega_k) * phi2
                                                     + num / denom
                                                 )

                                     
                                     print("beta after interpolation", beta)
                                     # safeguard alpha
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
                                             print("using bisection")
                                             beta = (alpha_l + alpha_u) / 2
                                     count10 += 1
                                     # if np.abs(beta -beta0) <= 1e-12:
                                     #    print("possible convergence")
                                     #    print(beta, beta0)
                                     #    print(x2_norm)

                                     #    if x2_norm < trust_radius:
                                     #        print("11111111")
                                     #        hard_case = 1
                                     #    elif x2_norm > trust_radius and both_roots_normalization == 1:
                                     #        print("22222222")
                                     #        hard_case = 1

                                     #    break
                                     if count10 == 50:
                                         break
                                 print("check hard case", hard_case)
                                 _stats_end(
                                     _stats_span,
                                     iters=count10,
                                     status="hard_case=%d" % hard_case,
                                 )
                                 if hard_case == 1:
                                     first_component[:] = x2[:]
                                     if x2_norm > trust_radius:
                                         print("some logic is wrong")
                                         exit()
                                         # temp_vector = np.zeros(self.index_map_size)
                                         # temp_vector[:] = second_component[:]
                                         # second_component[:] = first_component[:]
                                         # first_component[:] = temp_vector[:]
                                     # print(first_component)
                                     # print(second_component)
                                     print(
                                         "norm of current root", np.linalg.norm(first_component)
                                     )
                                     xy_square = np.dot(
                                         first_component, second_component
                                     ) * np.dot(first_component, second_component)
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
                                     adjusted_step = (
                                         first_component + min(np.abs(t1), np.abs(t2)) * second_component
                                     )
                                     print("adjusted step norm", np.linalg.norm(adjusted_step))
                                     step = adjusted_step

                                 if hard_case == 2:
                                     #Q = np.zeros((1, self.index_map_size))
                                     #H1_op = LinearOperator(
                                     #    (self.index_map_size, self.index_map_size),
                                     #    matvec=lambda Q: self.mv2(
                                     #        self.U2, A_tilde2, G1, Q, 1, 0, 0
                                     #    ),
                                     #)
                                     #x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
                                     #print("exitcode", exitCode)
                                     #step = x
                                     _stats_span_n = _stats_begin("newton_in_lstrs")
                                     denom = self.reduced_hessian_diagonal
                                     max_iter = _QED_NEWTON_MAXITER  # QED_NEWTON_MAXITER, default 20
                                     _solve_newton = self.minres_solve if _QED_USE_MINRES else self.linear_equation_solve
                                     solution, converged = _solve_newton(self.U2, A_tilde2, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6)
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
                                     _stats_end(_stats_span_n, status="newton")
                                     hard_case = 2
                                     step = solution


                                 if hard_case == 3:
                                     step = x_tilde
                                 if hard_case == 4:
                                     if x2_norm > trust_radius:
                                         step = x2 / x2_norm * trust_radius
                                     if x2_norm < trust_radius:
                                         step = x2
                                 

                    else:
                        print("gradient is small, use Newton step")
                        _stats_span = _stats_begin("newton_small_gradient")
                        denom = self.reduced_hessian_diagonal
                        max_iter = _QED_NEWTON_MAXITER  # QED_NEWTON_MAXITER, default 20
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

                    if microiteration == 0 and orbital_optimization_step == 0:
                        convergence_threshold = min(
                            0.01 * gradient_norm, np.power(gradient_norm, 2)
                        )
                    if energy_change < 0.0 or hard_case == 2:
                        print("step norm", np.linalg.norm(step))
                        if step_norm < 0.05 and not _QED_DISABLE_QN:
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
                        trust_radius = self.step_control(ratio, trust_radius, step_norm)

                        #self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                        # print(np.shape(gradient_tilde), flush = True)
                        if qn_count == 0:
                            start = timer()
                            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
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
                                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                                    
                                    index_count1 += 1
                            print("reduced_gradient",reduced_gradient, flush = True)

                            

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

            if convergence == 1 and count == 0:
                active_twoeint[:, :, :, :] = self.active_twoeint[:, :, :, :]
                active_fock_core[:, :] = self.active_fock_core[:, :]
                d_cmo[:, :] = self.d_cmo[:, :]
                self.E_core2 = self.E_core
                

            start = timer()
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
            print("qn status", qn_optimization, "qn count", qn_count, flush = True)
            print("gradient norm", np.linalg.norm(reduced_gradient), flush = True)
            d_diag = 2.0 * np.einsum("ii->", d_cmo[: self.n_in_a, : self.n_in_a])
            self.constdouble[3] = self.d_exp - d_diag
            self.constdouble[4] = 1e-9
            self.constdouble[5] = self.E_core2
            # Experiment knobs, default-no-op -- see the definitions at the
            # top of this file. This is the LIVE coupled-CI site
            # (microiteration_optimization6); the byte-identical block in
            # microiteration_optimization5 is dead code (no callers) and is
            # deliberately left untouched.
            self.constint[8] = _QED_CI_UNCOUPLED_MAXITER
            if qn_count > 0:
                self.constdouble[4] = _QED_CI_COUPLED_FACTOR * np.linalg.norm(reduced_gradient)
                self.constint[8] = 10000
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
            if _cold_micro:
                self.constint[5] = _svm_indim
                self.constint[6] = _svm_maxdim
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











    def ah_orbital_optimization(self, eigenvecs, c_get_roots):
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
        N_orbital_optimization_steps = 4
        N_microiterations = 25
        microiteration = 0
        while microiteration < N_microiterations:
            print("\n")
            print("\n")
            print("MICROITERATION", microiteration + 1, flush=True)

            # while(microiteration < 2):
            trust_radius = 0.35
            A[:, :] = 0.0
            G[:, :, :, :] = 0.0
            start = timer()
            self.build_intermediates(eigenvecs, A, G, True)
            end = timer()
            print("build intermediates took", end - start)
            # A2[:,:] = 0.0
            # G2[:,:,:,:] = 0.0
            # self.build_intermediates2(eigenvecs, A2, G2, True)
            # print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            # print(old_energy, current_energy)
            # print("initial convergence threshold", convergence_threshold)
            # if (np.abs(current_energy - old_energy) < 0.01 * convergence_threshold) and microiteration >=2:
            ##if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
            #    print("microiteration converged (small energy change)")
            #    #self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
            #    break

            start = timer()
            start1 = timer()
            macroiteration_energy_initial = macroiteration_energy_current
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

            # print(eigenvecs)
            active_fock_core = np.zeros((self.n_act_orb, self.n_act_orb))
            active_fock_core[:, :] = fock_core[
                self.n_in_a : self.n_occupied, self.n_in_a : self.n_occupied
            ]
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
            macroiteration_energy_current = (
                active_one_e_energy
                + active_two_e_energy
                + active_one_pe_energy
                + E_core
                + self.Enuc
                + self.d_c
                + ci_dependent_energy
            )
            print(
                "macroiteration_energy_current",
                "{:20.12f}".format(macroiteration_energy_current),
                "{:20.12f}".format(active_one_e_energy),
                "{:20.12f}".format(active_two_e_energy),
                "{:20.12f}".format(E_core),
                "{:20.12f}".format(active_one_pe_energy),
                "{:20.12f}".format(self.Enuc),
            )

            end = timer()
            print("check initial energy took", end - start)

            self.build_gradient_and_hessian(
                self.U2, A, G, gradient_tilde, hessian_tilde, True
            )

            start = timer()
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            G1 = G.transpose(3, 1, 2, 0).reshape(
                self.nmo * self.n_occupied, self.nmo * self.n_occupied
            )
            # Davidson_augmented_hessian_solve3 (this dead branch's solver) still
            # takes the dense G1, but microiteration_predicted_energy2/mv2 are
            # block-only now, so build the blocks here too.
            _G_t = G.transpose(3, 1, 2, 0)
            G_blocks = (
                _G_t[:, : self.n_in_a, :, : self.n_in_a].reshape(
                    self.nmo * self.n_in_a, self.nmo * self.n_in_a
                ),
                _G_t[:, self.n_in_a :, :, : self.n_in_a].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_in_a
                ),
                _G_t[:, self.n_in_a :, :, self.n_in_a :].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
                ),
            )
            end = timer()
            # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)

            print("build gradient took", end - start)
            hessian_tilde3 = hessian_tilde.transpose(2, 0, 3, 1)
            hessian_tilde3 = hessian_tilde3.reshape(
                (self.n_occupied * self.nmo, self.n_occupied * self.nmo)
            )

            # hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end = timer()
            print("build hessian diagonal took", end - start)
            reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
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
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    # print(r,k,index_count1)
                    index_count2 = 0
                    for l in range(self.n_occupied):
                        for s in range(l + 1, self.nmo):
                            if l < self.n_in_a and s < self.n_in_a:
                                continue
                            if (
                                self.n_in_a <= l < self.n_occupied
                                and self.n_in_a <= s < self.n_occupied
                            ):
                                continue
                            # if (k >= self.n_occupied and r >= self.n_occupied): continue
                            reduced_hessian[index_count1][index_count2] = (
                                hessian_tilde3[r * self.n_occupied + k][
                                    s * self.n_occupied + l
                                ]
                            )
                            # print(r,k,s,l,index_count1,index_count2)
                            index_count2 += 1
                    index_count1 += 1
            # print("reduced_gradient",reduced_gradient, flush = True)
            np.set_printoptions(precision=14)
            mu1, w1 = np.linalg.eigh(reduced_hessian)
            print("eigenvalue of the reduced hessian", mu1)
            print(
                "dot product of gradient and first eigenvector of hessian",
                np.dot(reduced_gradient, w1[:, 0]),
            )
            print(
                "dot product of gradient and second eigenvector of hessian",
                np.dot(reduced_gradient, w1[:, 1]),
            )
            H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            step_limit = -np.einsum(
                "pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient
            )
            print("norm of critical step", np.linalg.norm(step_limit))
            print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
            print("norm of reduced hessian", np.linalg.norm(reduced_hessian))
            gradient_norm = np.linalg.norm(reduced_gradient)
            print("gradient norm", gradient_norm)
            ##if microiteration ==0:
            ##    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
            # if gradient_norm < 1e-6:
            #    print("Microiteration converged (small gradient norm)")
            #    break
            print("LETS CHECK ENERGY AT THE BEGINING OF EACH MICROITERATION")
            print(macroiteration_energy_initial, macroiteration_energy_current)
            print("initial convergence threshold", convergence_threshold)
            if (
                np.abs(macroiteration_energy_current - macroiteration_energy_initial)
                < 0.0001 * convergence_threshold
            ) and microiteration >= 2:
                # if (np.abs(current_energy - old_energy) < 1e-15) and microiteration >=2:
                print("microiteration converged (small energy change)")
                # self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U2)
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
            reduce_predicted_energy = 0
            while orbital_optimization_step < N_orbital_optimization_steps:
                print("\n")
                old_energy = current_energy
                print(
                    "Microiteration",
                    microiteration + 1,
                    "orbital optimization step",
                    orbital_optimization_step + 1,
                    flush=True,
                )
                print("current energy", current_energy)
                print("reduced_gradient", reduced_gradient)
                gradient_norm = np.linalg.norm(reduced_gradient)
                print("gradient norm", gradient_norm, flush=True)
                print("convergence_threshold", convergence_threshold, flush=True)

                # if (gradient_norm < max(0.1 * convergence_threshold, 1e-8)):
                # if (gradient_norm < max(0.1 * convergence_threshold, 1e-6) and microiteration > 0):
                # if (gradient_norm < 1e-5 and microiteration > 0):
                if gradient_norm < 0.001 * convergence_threshold and microiteration > 0:
                    convergence = 1
                    # print("Microiteration converged (small gradient norm)")
                    break
                if gradient_norm < 1e-7:
                    convergence = 1
                    break
                # if microiteration == 1 and orbital_optimization_step == 1:
                #    np.savetxt("reduced_gradient.out", reduced_gradient)
                #    np.savetxt("reduced_hessian.out", reduced_hessian)

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

                # new augmented hessian
                beta = 0
                dim00 = self.index_map_size + 1
                # augmented_hessian9= np.zeros((dim00, dim00))
                w9 = np.zeros((dim00, dim00))
                step = np.zeros(self.index_map_size)
                step10_norm = 0.0
                # print("w",w)
                # count10 = 0
                # while np.abs((step10_norm - trust_radius)/trust_radius) > 1e-3:
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
                # step9 = w9[1:,0]
                if np.linalg.norm(reduced_gradient) > 1e-3:
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
                        adjusted_step = (
                            first_component
                            + min(np.abs(t1), np.abs(t2)) * second_component
                        )
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
                        alpha_u = (
                            delta_u + np.linalg.norm(reduced_gradient) * trust_radius
                        )
                        alpha_l = 0
                        count10 = 0
                        # beta = min(0, alpha_u)
                        beta = alpha_u
                        aug_hessian_eigenvecs = np.zeros((dim00, 2))
                        aug_hessian_eigenvals = np.zeros(2)
                        print("trust radius", trust_radius)
                        while True:
                            print("****************************")
                            print("lstrs iteration", count10, "beta", beta)
                            beta0 = beta
                            # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, beta, dim00)
                            # augmented_hessian9[:,:] = 0.0
                            # augmented_hessian9[0,0] = beta
                            # augmented_hessian9[0,1:] = reduced_gradient
                            # augmented_hessian9[1:,0] = reduced_gradient.T
                            # augmented_hessian9[1:,1:] = reduced_hessian
                            # mu9, w9 = np.linalg.eigh(augmented_hessian9)
                            # print("eig",mu9)
                            # print("two roots", mu9[0], mu9[1], mu9[2])
                            if count10 == 0:
                                Q = self.Davidson_augmented_hessian_solve3(
                                    self.U2,
                                    A_tilde2,
                                    G,
                                    G1,
                                    self.reduced_hessian_diagonal,
                                    reduced_gradient,
                                    beta,
                                    aug_hessian_eigenvecs,
                                    aug_hessian_eigenvals,
                                    guess_vector,
                                    restart=False,
                                )
                                delta_l = aug_hessian_eigenvals[0]
                                alpha_l = (
                                    delta_l
                                    - np.linalg.norm(reduced_gradient) / trust_radius
                                )
                            else:
                                guess_vector = Q
                                Q = self.Davidson_augmented_hessian_solve3(
                                    self.U2,
                                    A_tilde2,
                                    G,
                                    G1,
                                    self.reduced_hessian_diagonal,
                                    reduced_gradient,
                                    beta,
                                    aug_hessian_eigenvecs,
                                    aug_hessian_eigenvals,
                                    guess_vector,
                                    restart=True,
                                )
                            # print(aug_hessian_eigenvals)
                            # print(aug_hessian_eigenvecs)
                            mu9[0] = aug_hessian_eigenvals[0]
                            mu9[1] = aug_hessian_eigenvals[1]
                            w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                            w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                            print("ALPHA range", alpha_l, alpha_u)
                            # idx = mu9.argsort()[:dim00]
                            # print(idx)
                            # w9[:,0] = w9[:,0]/np.linalg.norm(w9[:,0])
                            # w9[:,1] = w9[:,1]/np.linalg.norm(w9[:,1])
                            v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                            v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                            u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                            u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                            aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                            bb1 = np.sqrt(1 - v1 * v1)
                            aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                            bb2 = np.sqrt(1 - v2 * v2)
                            print("aa1, bb1", aa1, bb1, "aa2, bb2", aa2, bb2)
                            if mu9[0] > -1e-8 and np.linalg.norm(
                                u1
                            ) < trust_radius * np.abs(v1):
                                hard_case = 2
                                print("use newton step")
                                break
                            delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1) / (
                                1 - v1 * v1
                            )
                            delta_u = min(delta_u, delta_u2)
                            print("delta_u2", delta_u2)

                            epsilon_v = 1e-4
                            print(aa1 <= epsilon_v * bb1)
                            print(aa2 <= epsilon_v * bb2)
                            alpha = beta
                            while (
                                (aa1 <= epsilon_v * bb1)
                                and (aa2 <= epsilon_v * bb2)
                                and np.abs(alpha_u - alpha_l)
                                > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
                            ):
                                alpha_u = alpha
                                alpha = (alpha_l + alpha_u) / 2
                                w9[:, :] = 0.0
                                # mu9[:] = self.projection_step2(reduced_gradient, reduced_hessian, w9, alpha, dim00)
                                # print("two roots", mu9[0], mu9[1], mu9[2])

                                Q = self.Davidson_augmented_hessian_solve3(
                                    self.U2,
                                    A_tilde2,
                                    G,
                                    G1,
                                    self.reduced_hessian_diagonal,
                                    reduced_gradient,
                                    alpha,
                                    aug_hessian_eigenvecs,
                                    aug_hessian_eigenvals,
                                    Q,
                                    restart=True,
                                )
                                mu9[0] = aug_hessian_eigenvals[0]
                                mu9[1] = aug_hessian_eigenvals[1]
                                w9[:, 0] = aug_hessian_eigenvecs[:, 0]
                                w9[:, 1] = aug_hessian_eigenvecs[:, 1]
                                v1 = w9[0, 0] / np.linalg.norm(w9[:, 0])
                                v2 = w9[0, 1] / np.linalg.norm(w9[:, 1])
                                u1[:] = w9[1:, 0] / np.linalg.norm(w9[:, 0])
                                u2[:] = w9[1:, 1] / np.linalg.norm(w9[:, 1])
                                aa1 = np.linalg.norm(reduced_gradient) * np.abs(v1)
                                bb1 = np.sqrt(1 - v1 * v1)
                                aa2 = np.linalg.norm(reduced_gradient) * np.abs(v2)
                                bb2 = np.sqrt(1 - v2 * v2)
                                print(
                                    alpha,
                                    aa1 <= epsilon_v * bb1,
                                    aa2 <= epsilon_v * bb2,
                                    mu9[0],
                                    mu9[1],
                                    bb1,
                                    bb2,
                                )
                                delta_u2 = mu9[0] - v1 * np.dot(
                                    reduced_gradient, u1
                                ) / (1 - v1 * v1)
                                delta_u = min(delta_u, delta_u2)
                                print("new delta_u", delta_u, delta_u2)

                            if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
                                exit()
                            elif (aa1 > epsilon_v * bb1) and (aa2 > epsilon_v * bb2):
                                both_roots_normalization = 1
                            else:
                                both_roots_normalization = 0
                            beta = alpha
                            # update delta_u
                            temptemp = np.dot(reduced_hessian, u1)
                            temptemp2 = np.dot(u1, temptemp)

                            # delta_u2 = mu9[0] - v1 * np.dot(reduced_gradient, u1)/np.dot(u1,u1)
                            delta_u = min(delta_u, temptemp2 / np.dot(u1, u1))
                            # print("new delta_u", delta_u)
                            ####scale9 = w9[0][0]
                            ####print("scale",scale9)
                            ####if np.abs(scale9) < 1e-15:
                            ####    break
                            # assign (k-1) and k-iteration values
                            lambda1 = lambda2
                            x1 = x2
                            phi1 = phi2
                            print("two roots", mu9[0], mu9[1])
                            if aa1 > epsilon_v * bb1:
                                step10 = u1 / v1
                                second_component[:] = u2
                                lambda2 = mu9[0]
                                print("norm from first root", np.linalg.norm(step10))
                                # if aa2 > epsilon_v * bb2:
                                #    second_component[:] = u2/v2
                                #    print("second root norm", np.linalg.norm(second_component))
                                if np.linalg.norm(step10) < trust_radius:
                                    alpha_l = beta
                                if np.linalg.norm(step10) > trust_radius:
                                    alpha_u = beta
                            else:
                                step10 = u2 / v2
                                second_component[:] = u1
                                lambda2 = mu9[1]
                                print("norm from second root", np.linalg.norm(step10))
                                alpha_u = beta
                            print("NEW ALPHA range", alpha_l, alpha_u)
                            step10_norm = np.linalg.norm(step10)
                            x2 = step10
                            if mu9[0] > -1e-8 and np.linalg.norm(
                                u1
                            ) < trust_radius * np.abs(v1):
                                hard_case = 2
                                print("use newton step")
                                break
                            phi2 = -np.dot(reduced_gradient, x2)
                            # print(step10)
                            # print(x1)
                            # print(x2)
                            # if count10 == 0 and step10_norm < trust_radius: break
                            # print("check eigenvalues and step norm from diagonalizing the bordering matrix")
                            # print(mu9, flush=True)
                            print(
                                "step10 norm",
                                step10_norm,
                                "current beta",
                                beta,
                                flush=True,
                            )
                            x1_norm = np.linalg.norm(x1)
                            x2_norm = np.linalg.norm(x2)
                            phi2_p = np.dot(x2, x2)
                            phi1_p = np.dot(x1, x1)
                            print(
                                "phi1",
                                phi1,
                                "phi1_p",
                                phi1_p,
                                "phi2",
                                phi2,
                                "phi2_p",
                                phi2_p,
                            )
                            if (
                                np.abs((step10_norm - trust_radius) / trust_radius)
                                <= 1e-3
                                and mu9[0] <= 0
                            ):
                                step = step10
                                hard_case = 0
                                break
                            print("check orthogonality of the first two roots")
                            vv1 = w9[0, 0]
                            vv2 = w9[0, 1]
                            uu1 = w9[1:, 0]
                            uu2 = w9[1:, 1]
                            # check if the quasi-optimal condition is satisfied
                            print(
                                "quasi-optimal condition",
                                (1 + trust_radius * trust_radius)
                                * (vv1 * vv1 + vv2 * vv2),
                                flush=True,
                            )
                            qs = (1 + trust_radius * trust_radius) * (
                                vv1 * vv1 + vv2 * vv2
                            )
                            print(np.dot(w9[:, :2].T, w9[:, :2]))

                            # construct quasi-optimal step
                            epsilon_hc = 1e-6
                            eta = epsilon_hc / (1 - epsilon_hc)
                            tau1 = 1
                            tau2 = 1
                            if qs > 1.0:
                                tau1 = (
                                    (vv1 - vv2 * np.sqrt(qs - 1))
                                    / (vv1 * vv1 + vv2 * vv2)
                                    / np.sqrt(1 + trust_radius * trust_radius)
                                )
                                tau2 = (
                                    (vv2 + vv1 * np.sqrt(qs - 1))
                                    / (vv1 * vv1 + vv2 * vv2)
                                    / np.sqrt(1 + trust_radius * trust_radius)
                                )
                            elif np.abs(qs - 1.0) < 2e-308:
                                tau1 = vv1 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                                tau2 = vv2 / np.sqrt(vv1 * vv1 + vv2 * vv2)
                            x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                tau1 * vv1 + tau2 * vv2
                            )

                            lambda_tilde = tau1 * tau1 * lambda1 + tau2 * tau2 * lambda2
                            psi_tilde = 0.5 * self.microiteration_predicted_energy2(
                                self.U2, reduced_gradient, A_tilde2, G_blocks, x_tilde
                            )
                            if qs > 1.0 or np.abs(qs - 1.0) < 2e-308:
                                print(
                                    "x_tilde norm1",
                                    np.linalg.norm(x_tilde),
                                    (mu9[1] - mu9[0])
                                    * tau2
                                    * tau2
                                    * (1 + trust_radius * trust_radius),
                                    -2.0 * eta * psi_tilde,
                                )
                                if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                    1 + trust_radius * trust_radius
                                ) < -2.0 * eta * psi_tilde:
                                    print("find quasi-optimal solution1", flush=True)
                                    hard_case = 3
                                    break
                                else:
                                    if qs > 1.0:
                                        tau1 = (
                                            (vv1 + vv2 * np.sqrt(qs - 1))
                                            / (vv1 * vv1 + vv2 * vv2)
                                            / np.sqrt(1 + trust_radius * trust_radius)
                                        )
                                        tau2 = (
                                            (vv2 - vv1 * np.sqrt(qs - 1))
                                            / (vv1 * vv1 + vv2 * vv2)
                                            / np.sqrt(1 + trust_radius * trust_radius)
                                        )
                                        x_tilde = (tau1 * uu1 + tau2 * uu2) / (
                                            tau1 * vv1 + tau2 * vv2
                                        )
                                        lambda_tilde = (
                                            tau1 * tau1 * lambda1
                                            + tau2 * tau2 * lambda2
                                        )
                                        psi_tilde = (
                                            0.5
                                            * self.microiteration_predicted_energy2(
                                                self.U2,
                                                reduced_gradient,
                                                A_tilde2,
                                                G_blocks,
                                                x_tilde,
                                            )
                                        )
                                        print(
                                            "x_tilde norm2",
                                            np.linalg.norm(x_tilde),
                                            (mu9[1] - mu9[0])
                                            * tau2
                                            * tau2
                                            * (1 + trust_radius * trust_radius),
                                            -2.0 * eta * psi_tilde,
                                            flush=True,
                                        )
                                        if (mu9[1] - mu9[0]) * tau2 * tau2 * (
                                            1 + trust_radius * trust_radius
                                        ) < -2.0 * eta * psi_tilde:
                                            print(
                                                "find quasi-optimal solution2",
                                                flush=True,
                                            )
                                            hard_case = 3
                                            break
                            if np.abs(alpha_u - alpha_l) <= 1e-8 * max(
                                np.abs(alpha_u), np.abs(alpha_l)
                            ):
                                print("interval too small")
                                if (
                                    x2_norm < trust_radius
                                    and both_roots_normalization == 0
                                ):
                                    hard_case = 1
                                if both_roots_normalization == 1:
                                    hard_case = 4
                                break

                            if count10 == 0:
                                beta = beta + (beta - lambda2) / x2_norm * (
                                    trust_radius - x2_norm
                                ) / trust_radius * (trust_radius + 1.0 / x2_norm)
                            else:
                                print("denominator", x2_norm - x1_norm)
                                denom = trust_radius * (x2_norm - x1_norm)

                                if np.abs(denom) > 2e-308:
                                    lambda_c = (
                                        lambda1 * x1_norm * (x2_norm - trust_radius)
                                        + lambda2 * x2_norm * (trust_radius - x1_norm)
                                    ) / denom
                                else:
                                    lambda_c = delta_u
                                if lambda_c > delta_u:
                                    print(
                                        "need to safeguard delta_u", lambda_c, delta_u
                                    )
                                    lambda_c = delta_u
                                # lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                                # print(lambda_c)
                                if np.abs(lambda2 - lambda1) <= 2e-308:
                                    print("interpolate alpha by bisection")
                                    beta = (alpha_l + alpha_u) / 2
                                else:
                                    omega_k = (lambda2 - lambda_c) / (lambda2 - lambda1)
                                    num = (
                                        x1_norm
                                        * x2_norm
                                        * (x2_norm - x1_norm)
                                        * (lambda1 - lambda_c)
                                        * (lambda2 - lambda_c)
                                    )

                                    denom = (
                                        omega_k * x2_norm + (1 - omega_k) * x1_norm
                                    ) * (lambda2 - lambda1)
                                    print(x1_norm, x2_norm)
                                    print(lambda1, lambda2, lambda_c)
                                    print(num, denom)
                                    if np.abs(denom) <= 2e-308:
                                        print("interpolate alpha by bisection2")
                                        beta = (alpha_l + alpha_u) / 2
                                    else:
                                        beta = (
                                            lambda_c
                                            + omega_k * phi1
                                            + (1 - omega_k) * phi2
                                            + num / denom
                                        )

                            # if count10 == 0 or (np.abs(x2_norm - x1_norm) <= 1e-15):
                            #    if (np.abs(x2_norm - x1_norm) <= 1e-15):
                            #        print("too small denominator")
                            #    beta = beta + (beta - lambda2)/x2_norm * (trust_radius - x2_norm)/trust_radius * (trust_radius + 1.0/x2_norm)
                            # else:
                            #    print("denominator", x2_norm-x1_norm)
                            #    if (np.abs(x2_norm - x1_norm) > 1e-15):
                            #        lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                            #    else:
                            #        lambda_c = delta_u
                            #    if lambda_c > delta_u:
                            #        print("need to safeguard delta_u", lambda_c, delta_u)
                            #        lambda_c = delta_u
                            #    #lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) + lambda2 * x2_norm * (trust_radius - x1_norm))/trust_radius/(x2_norm - x1_norm)
                            #    #print(lambda_c)
                            #    omega_k = (lambda2 - lambda_c)/(lambda2 - lambda1)
                            #    ratio1 = x1_norm * x2_norm * (x2_norm - x1_norm)/(omega_k * x2_norm + (1-omega_k) * x1_norm)
                            #    ratio2 = (lambda1 - lambda_c) * (lambda2 - lambda_c) / (lambda2 - lambda1)
                            #    beta = lambda_c + omega_k *phi1 + (1-omega_k) * phi2 + ratio1 * ratio2
                            print("beta after interpolation", beta)
                            # safeguard alpha
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
                                    print("using bisection")
                                    beta = (alpha_l + alpha_u) / 2
                            count10 += 1
                            # if np.abs(beta -beta0) <= 1e-12:
                            #    print("possible convergence")
                            #    print(beta, beta0)
                            #    print(x2_norm)

                            #    if x2_norm < trust_radius:
                            #        print("11111111")
                            #        hard_case = 1
                            #    elif x2_norm > trust_radius and both_roots_normalization == 1:
                            #        print("22222222")
                            #        hard_case = 1

                            #    break
                            if count10 == 50:
                                break
                        print("check hard case", hard_case)
                        if hard_case == 1:
                            first_component[:] = x2[:]
                            if x2_norm > trust_radius:
                                print("some logic is wrong")
                                exit()
                                # temp_vector = np.zeros(self.index_map_size)
                                # temp_vector[:] = second_component[:]
                                # second_component[:] = first_component[:]
                                # first_component[:] = temp_vector[:]
                            # print(first_component)
                            # print(second_component)
                            print(
                                "norm of current root", np.linalg.norm(first_component)
                            )
                            xy_square = np.dot(
                                first_component, second_component
                            ) * np.dot(first_component, second_component)
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
                            adjusted_step = (
                                first_component
                                + min(np.abs(t1), np.abs(t2)) * second_component
                            )
                            print("adjusted step norm", np.linalg.norm(adjusted_step))
                            step = adjusted_step

                        if hard_case == 2:
                            Q = np.zeros((1, self.index_map_size))
                            H1_op = LinearOperator(
                                (self.index_map_size, self.index_map_size),
                                matvec=lambda Q: self.mv2(
                                    self.U2, A_tilde2, G_blocks, Q, 1, 0, 0
                                ),
                            )
                            x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
                            print("exitcode", exitCode)
                            step = x

                        if hard_case == 3:
                            step = x_tilde
                        if hard_case == 4:
                            if x2_norm > trust_radius:
                                step = x2 / x2_norm * trust_radius
                            if x2_norm < trust_radius:
                                step = x2

                else:
                    print("gradient is small, use Newton step")
                    Q = np.zeros((1, self.index_map_size))
                    H1_op = LinearOperator(
                        (self.index_map_size, self.index_map_size),
                        matvec=lambda Q: self.mv2(self.U2, A_tilde2, G_blocks, Q, 1, 0, 0),
                    )
                    x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
                    print("exitcode", exitCode)
                    hard_case = 2
                    step = x
                    print(step)
                # w[:,0] = w[:,0]/scale
                # print(w)
                step9 = w9[1:, 0]

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
                ###############        adjusted_step = step + min(np.abs(t1), np.abs(t2)) * step2
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

                U_temp = np.einsum("pq,qs->ps", self.U_total, self.U_delta)
                J_temp = np.zeros(
                    (self.n_occupied, self.n_occupied, self.nmo, self.nmo)
                )
                K_temp = np.zeros(
                    (self.n_occupied, self.n_occupied, self.nmo, self.nmo)
                )

                K_temp = np.ascontiguousarray(K_temp)
                c_full_transformation_macroiteration(
                    U_temp,
                    self.twoeint,
                    J_temp,
                    K_temp,
                    self.index_map_pq,
                    self.index_map_kl,
                    self.nmo,
                    self.n_occupied,
                )
                # self.full_transformation_macroiteration(self.U_total, self.J, self.K)

                temp8 = np.zeros((self.nmo, self.nmo))
                temp8 = np.einsum("pq,qs->ps", self.H1, U_temp)
                h1_temp = np.einsum("ps,pr->rs", temp8, U_temp)
                temp8 = np.einsum("pq,qs->ps", self.d_cmo1, U_temp)
                d_cmo_temp = np.einsum("ps,pr->rs", temp8, U_temp)

                new_energy = self.rdm_exact_energy(
                    J_temp, K_temp, h1_temp, d_cmo_temp, eigenvecs
                )
                energy_change = new_energy - old_energy
                print(
                    "old energy",
                    old_energy,
                    "new energy",
                    new_energy,
                    "energy change",
                    energy_change,
                    flush=True,
                )

                # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                predicted_energy2 = self.microiteration_predicted_energy2(
                    self.U2, reduced_gradient, A_tilde2, G_blocks, step
                )
                print("microiteration predicted energy", predicted_energy2, flush=True)
                if predicted_energy2 < 0:
                    reduce_predicted_energy = 1
                else:
                    reduce_predicted_energy = 0
                end = timer()
                print("build unitary matrix and recheck energy took", end - start)

                if microiteration == 0 and orbital_optimization_step == 0:
                    convergence_threshold = min(
                        0.01 * gradient_norm, np.power(gradient_norm, 2)
                    )
                if energy_change < 0.0 or hard_case == 2:
                    # restart = False
                    # predicted_energy1 = self.microiteration_predicted_energy(reduced_gradient, reduced_hessian, step)
                    # predicted_energy2 = self.microiteration_predicted_energy2(self.U2, reduced_gradient, A_tilde2, G, step)
                    # print("microinteration predicted energy", predicted_energy1, predicted_energy2, flush = True)
                    self.U_total = np.einsum("pq,qs->ps", self.U_total, self.U_delta)
                    if microiteration == 0 and orbital_optimization_step == 0:
                        #    convergence_threshold = min(0.01 * gradient_norm, np.power(gradient_norm,2))
                        # convergence_threshold = 0.01 * gradient_norm
                        ###10/10/2024 comment out this part to use standard trust region method
                        # if step_norm > 0.1:
                        #    N_microiterations = 5
                        #    N_orbital_optimization_steps = 4
                        # elif step_norm <= 0.1 and step_norm > 0.01:
                        #    N_microiterations = 7
                        #    N_orbital_optimization_steps = 3
                        ###end comment
                        print("number of microiteration", N_microiterations, flush=True)
                        print(
                            "number of optimization steps",
                            N_orbital_optimization_steps,
                            flush=True,
                        )
                    orbital_optimization_step += 1
                    ratio = energy_change / predicted_energy2
                    print("compare model with actual energy change", ratio)
                    trust_radius = self.step_control(ratio, trust_radius)

                    # self.build_gradient_and_hessian(self.U2, A, G, gradient_tilde, hessian_tilde, True)
                    # print(np.shape(gradient_tilde), flush = True)

                    start = timer()
                    self.J[:, :, :, :] = J_temp
                    self.K[:, :, :, :] = K_temp
                    self.H_spatial2[:, :] = h1_temp
                    self.d_cmo[:, :] = d_cmo_temp
                    # self.full_transformation_macroiteration(self.U_total, self.J, self.K)
                    temp_energy = self.rdm_exact_energy(
                        self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs
                    )
                    print("check energy again", temp_energy)
                    A[:, :] = 0
                    A_tilde2[:, :] = 0
                    G[:, :, :, :] = 0
                    gradient_tilde[:] = 0
                    hessian_tilde[:, :, :, :] = 0
                    self.build_intermediates(eigenvecs, A, G, True)
                    self.build_gradient_and_hessian(
                        self.U2, A, G, gradient_tilde, hessian_tilde, True
                    )
                    self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
                    end = timer()
                    # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)
                    G1 = G.transpose(3, 1, 2, 0).reshape(
                        self.nmo * self.n_occupied, self.nmo * self.n_occupied
                    )
                    # Davidson_augmented_hessian_solve3 (this dead branch's solver) still
                    # takes the dense G1, but microiteration_predicted_energy2/mv2 are
                    # block-only now, so build the blocks here too.
                    _G_t = G.transpose(3, 1, 2, 0)
                    G_blocks = (
                        _G_t[:, : self.n_in_a, :, : self.n_in_a].reshape(
                            self.nmo * self.n_in_a, self.nmo * self.n_in_a
                        ),
                        _G_t[:, self.n_in_a :, :, : self.n_in_a].reshape(
                            self.nmo * self.n_act_orb, self.nmo * self.n_in_a
                        ),
                        _G_t[:, self.n_in_a :, :, self.n_in_a :].reshape(
                            self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
                        ),
                    )
                    print("build gradient took", end - start)

                    hessian_tilde3 = hessian_tilde.transpose(2, 0, 3, 1)
                    hessian_tilde3 = hessian_tilde3.reshape(
                        (self.n_occupied * self.nmo, self.n_occupied * self.nmo)
                    )

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
                    reduced_gradient[:] = 0
                    reduced_hessian[:, :] = 0
                    np.set_printoptions(precision=14)
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
                            reduced_gradient[index_count1] = gradient_tilde[r][k]
                            print(r, k, index_count1)
                            index_count2 = 0
                            for l in range(self.n_occupied):
                                for s in range(l + 1, self.nmo):
                                    if l < self.n_in_a and s < self.n_in_a:
                                        continue
                                    if (
                                        self.n_in_a <= l < self.n_occupied
                                        and self.n_in_a <= s < self.n_occupied
                                    ):
                                        continue
                                    # if (k >= self.n_occupied and r >= self.n_occupied): continue
                                    reduced_hessian[index_count1][index_count2] = (
                                        hessian_tilde3[r * self.n_occupied + k][
                                            s * self.n_occupied + l
                                        ]
                                    )
                                    # print(r,k,s,l,index_count1,index_count2)
                                    index_count2 += 1
                            index_count1 += 1
                    print("reduced_gradient", reduced_gradient, flush=True)
                    mu1, w1 = np.linalg.eigh(reduced_hessian)
                    print("eigenvalue of the reduced hessian", mu1)
                    print(
                        "dot product of gradient and first eigenvector of hessian",
                        np.dot(reduced_gradient, w1[:, 0]),
                    )
                    print(
                        "dot product of gradient and second eigenvector of hessian",
                        np.dot(reduced_gradient, w1[:, 1]),
                    )
                    H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
                    step_limit = -np.einsum(
                        "pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient
                    )
                    print("norm of critical step", np.linalg.norm(step_limit))
                    print(
                        "alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit)
                    )
                    print("gradient norm", np.linalg.norm(reduced_gradient))

                    current_energy = old_energy + energy_change
                    print("now accept step, check current energy", current_energy)
                    hard_case = 0
                else:
                    # restart = False
                    # guess_vector = trial_vector
                    reduce_step = 1
                    trust_radius = 0.5 * trust_radius
                    print("Reject step, restart", flush=True)
                    print("new trust radius", trust_radius, flush=True)
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
            print(self.H_diag3)
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
            print("building RDM took", end - start)

            current_energy2 = self.rdm_exact_energy(
                self.J, self.K, self.H_spatial2, self.d_cmo, eigenvecs
            )

            self.build_intermediates(eigenvecs, A, G, True)
            self.build_gradient_and_hessian(
                self.U2, A, G, gradient_tilde, hessian_tilde, True
            )
            # print(np.shape(gradient_tilde), flush = True)

            start = timer()
            self.build_gradient(self.U2, A, G, gradient_tilde, A_tilde2, True)
            G1 = G.transpose(3, 1, 2, 0).reshape(
                self.nmo * self.n_occupied, self.nmo * self.n_occupied
            )
            # Davidson_augmented_hessian_solve3 (this dead branch's solver) still
            # takes the dense G1, but microiteration_predicted_energy2/mv2 are
            # block-only now, so build the blocks here too.
            _G_t = G.transpose(3, 1, 2, 0)
            G_blocks = (
                _G_t[:, : self.n_in_a, :, : self.n_in_a].reshape(
                    self.nmo * self.n_in_a, self.nmo * self.n_in_a
                ),
                _G_t[:, self.n_in_a :, :, : self.n_in_a].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_in_a
                ),
                _G_t[:, self.n_in_a :, :, self.n_in_a :].reshape(
                    self.nmo * self.n_act_orb, self.nmo * self.n_act_orb
                ),
            )
            end = timer()
            # self.build_gradient2(self.U2, A, G, hessian_tilde, gradient_tilde, A_tilde2, True)

            print("build gradient took", end - start)
            hessian_tilde3 = hessian_tilde.transpose(2, 0, 3, 1)
            hessian_tilde3 = hessian_tilde3.reshape(
                (self.n_occupied * self.nmo, self.n_occupied * self.nmo)
            )

            # hessian_diagonal3 = np.diagonal(hessian_tilde3).reshape((self.nmo, self.n_occupied))
            reduced_hessian_diagonal = np.zeros(self.index_map_size)
            start = timer()
            self.build_hessian_diagonal(self.U2, G, A_tilde2)
            end = timer()
            print("build hessian diagonal took", end - start)
            reduced_hessian = np.zeros((self.index_map_size, self.index_map_size))
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
                    reduced_gradient[index_count1] = gradient_tilde[r][k]
                    # print(r,k,index_count1)
                    index_count2 = 0
                    for l in range(self.n_occupied):
                        for s in range(l + 1, self.nmo):
                            if l < self.n_in_a and s < self.n_in_a:
                                continue
                            if (
                                self.n_in_a <= l < self.n_occupied
                                and self.n_in_a <= s < self.n_occupied
                            ):
                                continue
                            # if (k >= self.n_occupied and r >= self.n_occupied): continue
                            reduced_hessian[index_count1][index_count2] = (
                                hessian_tilde3[r * self.n_occupied + k][
                                    s * self.n_occupied + l
                                ]
                            )
                            # print(r,k,s,l,index_count1,index_count2)
                            index_count2 += 1
                    index_count1 += 1
            # print("reduced_gradient",reduced_gradient, flush = True)
            np.set_printoptions(precision=14)
            mu1, w1 = np.linalg.eigh(reduced_hessian)
            print("eigenvalue of the reduced hessian", mu1)
            print(
                "dot product of gradient and first eigenvector of hessian",
                np.dot(reduced_gradient, w1[:, 0]),
            )
            print(
                "dot product of gradient and second eigenvector of hessian",
                np.dot(reduced_gradient, w1[:, 1]),
            )
            H_lambda = reduced_hessian - mu1[0] * np.eye(self.index_map_size)
            step_limit = -np.einsum(
                "pq,q->p", np.linalg.pinv(H_lambda), reduced_gradient
            )
            print("norm of critical step", np.linalg.norm(step_limit))
            print("alpha critical", mu1[0] - np.dot(reduced_gradient, step_limit))
            gradient_norm = np.linalg.norm(reduced_gradient)
            print("gradient norm", gradient_norm)
            alpha_critical = mu1[0] - np.dot(reduced_gradient, step_limit)
            augmented_hessian90 = np.zeros((dim00, dim00))
            augmented_hessian90[0, 0] = alpha_critical
            augmented_hessian90[0, 1:] = reduced_gradient
            augmented_hessian90[1:, 0] = reduced_gradient.T
            augmented_hessian90[1:, 1:] = reduced_hessian

            mu90, w90 = np.linalg.eigh(augmented_hessian90)
            print(mu90[0], mu90[1])
            if w90[0, 0] > 1e-15:
                zzzz = w90[1:, 0] / w90[0, 0]
                print(np.linalg.norm(zzzz))
            elif w90[0, 1] > 1e-15:
                yyyy = w90[1:, 1] / w90[0, 1]
                print("second case", np.linalg.norm(yyyy))
            print("ci energy", current_energy, "rdm_energy", current_energy2)
            print("current gradient_norm and residual", gradient_norm, current_residual)
            print("current convergence_threshold", convergence_threshold)
            total_norm = np.sqrt(
                np.power(gradient_norm, 2) + np.power(current_residual, 2)
            )
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
            c_full_transformation_macroiteration(
                U_temp,
                self.twoeint,
                J_temp,
                K_temp,
                self.index_map_pq,
                self.index_map_kl,
                self.nmo,
                self.n_occupied,
            )
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
        dim1 = self.n_occupied * self.nmo
        # start = timer()
        temp1 = np.zeros((self.n_occupied, self.nmo, self.nmo))
        temp1 = np.einsum(
            "kkrs->krs",
            G[: self.n_occupied, : self.n_occupied, :, :],
            optimize="optimal",
        )
        temp2 = np.zeros((self.n_occupied, self.nmo, self.n_virtual))
        temp2 = np.einsum(
            "krs,sa->kra", temp1, U[:, self.n_occupied : self.nmo], optimize="optimal"
        )
        self.hessian_diagonal = np.zeros((self.nmo, self.n_occupied))
        self.hessian_diagonal[self.n_occupied : self.nmo, :] = np.einsum(
            "kra,ra->ak", temp2, U[:, self.n_occupied : self.nmo], optimize="optimal"
        )

        # end = timer()
        # print("hessian diagonal step took", end - start)
        # B = np.zeros((self.nmo, self.nmo))
        # T = U - np.eye(self.nmo)
        # B[:,:self.n_occupied] = A[:,:self.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:self.n_occupied])
        # A_tilde = np.zeros((self.nmo, self.nmo))
        # A_tilde = np.einsum("rs,sk->rk",U.T, B)
        temp4 = np.zeros((self.n_occupied, self.n_occupied))

        # start = timer()
        o = np.full((self.n_occupied), 1)
        v = np.full((self.n_virtual), 1)
        self.hessian_diagonal[self.n_occupied : self.nmo, :] -= np.einsum(
            "k,a->ak",
            o,
            np.diagonal(
                A_tilde[self.n_occupied : self.nmo, self.n_occupied : self.nmo]
            ),
        )
        self.hessian_diagonal[self.n_occupied : self.nmo, :] -= np.einsum(
            "a,k->ak", v, np.diagonal(A_tilde[: self.n_occupied, : self.n_occupied])
        )
        # end = timer()
        # print("hessian diagonal step 1 took", end - start)
        # start = timer()
        temp3 = np.zeros((self.n_occupied, self.nmo, self.n_occupied))
        temp3 = np.einsum(
            "krs,sl->krl", temp1, U[:, : self.n_occupied], optimize="optimal"
        )
        temp4 = np.einsum(
            "krl,rl->lk", temp3, U[:, : self.n_occupied], optimize="optimal"
        )
        self.hessian_diagonal[: self.n_occupied, :] += 1.0 * temp4
        self.hessian_diagonal[: self.n_occupied, :] += 1.0 * temp4.T
        # end = timer()
        # print("hessian diagonal step 2 took", end - start)
        # start = timer()
        temp3 = np.einsum(
            "klrs,sk->krl", G, U[:, : self.n_occupied], optimize="optimal"
        )
        # end = timer()
        # print("hessian diagonal step 3 took", end - start)
        # start = timer()
        self.hessian_diagonal[: self.n_occupied, :] -= 2.0 * np.einsum(
            "krl,rl->kl", temp3, U[:, : self.n_occupied], optimize="optimal"
        )
        self.hessian_diagonal[: self.n_occupied, :] -= np.einsum(
            "k,l->kl", o, np.diagonal(A_tilde[: self.n_occupied, : self.n_occupied])
        )
        self.hessian_diagonal[: self.n_occupied, :] -= np.einsum(
            "l,k->kl", o, np.diagonal(A_tilde[: self.n_occupied, : self.n_occupied])
        )
        temp4 = (
            A_tilde[: self.n_occupied, : self.n_occupied]
            + A_tilde.T[: self.n_occupied, : self.n_occupied]
        )
        self.hessian_diagonal[: self.n_occupied, :] += np.einsum(
            "kl,kl->kl", np.eye(self.n_occupied), temp4, optimize="optimal"
        )
        # end = timer()
        # print("hessian diagonal step 4 took", end - start)
        # print("i7gt",hessian_diagonal)
        # start = timer()
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
                self.reduced_hessian_diagonal[index_count1] = self.hessian_diagonal[r][
                    k
                ]
                index_count1 += 1
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



    def inner_solve_pcg(self, residual_k, theta_k, Q_conv, P_diag,
                    U, A_tilde, G_blocks, reduced_gradient, alpha):

        H_dim = residual_k.shape[0]
        n_conv = Q_conv.shape[0]

        # Define a helper function to project a vector (or batch)
        # This makes it orthogonal to the converged roots.
        def project_out(vectors):
            if n_conv == 0:
                return vectors
            # Use np.einsum for batch dot product, it's fast
            # (N_vec, N_dim) @ (N_dim, N_conv) -> (N_vec, N_conv)
            coeffs = np.dot(vectors, Q_conv.T)
            # (N_vec, N_conv) @ (N_conv, N_dim) -> (N_vec, N_dim)
            return vectors - np.dot(coeffs, Q_conv)

        # Define a helper for the "A*p" matrix-vector product
        # This is (H - theta*I) * p
        def apply_H_minus_theta(vector, theta_k):
            # We need a (1, H_dim) vector for your sigma function
            p_vec = vector.reshape(1, H_dim)
            Hp = np.zeros_like(p_vec)

            # This is your expensive "H*p" product
            # You must adapt this to use your existing functions

            # 1. Apply the (H) part
            # This is equivalent to your self.orbital_sigma(...)
            # and the gradient/alpha parts
            #self.orbital_sigma(U, A_tilde, G1, p_vec, Hp, 1, 1) # Assumes 'self' is available
            #start = timer()
            self.orbital_sigma3(U, A_tilde, G_blocks, p_vec, Hp, 1, 1) # Assumes 'self' is available
            Hp[:, 1:] += np.einsum("p,q->qp", reduced_gradient, p_vec[:, 0])
            Hp[:, 0] = alpha * p_vec[:, 0] + np.dot(p_vec[:, 1:], reduced_gradient)
            #end = timer()
            #print("build orbital sigma for precondition took", end - start)


            # 2. Apply the (-theta*I) part
            Hp -= theta_k * p_vec

            return Hp.flatten() # Return a 1D vector

        # === PCG Algorithm ===
        # QED_JD_MAXITER: inner iteration cap.  Each inner iteration is one
        # sigma build, so this trades inner matvecs against OUTER Davidson
        # iterations (which each cost a matvec plus orthogonalization plus the
        # projected eigensolve).  Default 5 = the original value.
        max_inner_iter = _QED_JD_MAXITER
        _stats_span_jd = _stats_begin("jd_inner_pcg")
        _jd_iters = 0
        _jd_exit = "cap"
        v_corr = np.zeros(H_dim)    # Initial guess: 0
        r = -residual_k             # RHS of (H - theta*I)v = -r

        # 1. Apply preconditioner: z = P_inv * r
        z = r / P_diag
        # 2. Project!
        z = project_out(z)

        p = z                       # Initial search direction
        rho = np.dot(r, z)

        for _ in range(max_inner_iter):
            # 1. Matrix-vector product: q = A * p
            # Our "A" is the projected operator (I-Q_conv)(H-theta)(I-Q_conv)
            # We apply it in steps:
            p_proj = project_out(p) # This is (I-Q_conv)*p
            q = apply_H_minus_theta(p_proj, theta_k) # This is (H-theta)*(I-Q_conv)*p
            q = project_out(q) # This is (I-Q_conv)*(H-theta)*(I-Q_conv)*p
            _jd_iters += 1

            alpha = rho / np.dot(p_proj, q)

            v_corr += alpha * p_proj
            r -= alpha * q

            # Check for convergence (optional but good)
            # if np.linalg.norm(r) < 1e-3 * np.linalg.norm(residual_k):
            #     break
            # ⭐️ ADD THIS EARLY EXIT ⭐️
            # Check if the *inner* residual is 10x smaller than the *outer* residual
            if np.linalg.norm(r) < _QED_JD_TOL * np.linalg.norm(residual_k):
                _jd_exit = "tol"
                break
            # 2. Apply preconditioner: z = P_inv * r
            z = r / P_diag
            # 3. Project!
            z = project_out(z)

            rho_new = np.dot(r, z)
            beta = rho_new / rho
            rho = rho_new

            p = z + beta * p # p is implicitly projected in the next loop

        _stats_end(_stats_span_jd, iters=_jd_iters, status=_jd_exit)
        return v_corr


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
        p_vec = vector.reshape(1, H_dim)
        Hp = np.zeros_like(p_vec)

        # This is your expensive "H*p" product
        # You must adapt this to use your existing functions

        # 1. Apply the (H) part
        # This is equivalent to your self.orbital_sigma(...)
        # and the gradient/alpha parts
        #self.orbital_sigma(U, A_tilde, G1, p_vec, Hp, 1, 1) # Assumes 'self' is available
        #start = timer()
        self.orbital_sigma3(U, A_tilde, G_blocks, p_vec, Hp, 1, 1) # Assumes 'self' is available
        Hp[:, 1:] += np.einsum("p,q->qp", reduced_gradient, p_vec[:, 0])
        Hp[:, 0] = alpha * p_vec[:, 0] + np.dot(p_vec[:, 1:], reduced_gradient)
        #end = timer()
        #print("build orbital sigma for precondition took", end - start)

        return Hp.flatten() # Return a 1D vector
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

    def solve_gltr_with_operator(self, B_operator, g, M_diag, radius, max_iter=100, tol=1e-4, verbose=True):
        """
        Returns: (p, lambda, status)
        lambda is the shift such that (H + lambda*M) is positive semidefinite.
        """
        n = len(g)
        if max_iter is None or max_iter > n:
            max_iter = n

        # QED_GLTR_TOL_SCALE (see the knob block at the top of this file);
        # no-op at the default 1.0.
        tol = tol * _QED_GLTR_TOL_SCALE

        Q = np.zeros((max_iter + 1, n))
        alpha = np.zeros(max_iter)
        beta = np.zeros(max_iter)
        
        M_abs = np.abs(M_diag) + 1e-12 
        
        #r = g.copy()
        # NEW: Add microscopic noise to see hidden negative curvature
        noise_scale = 1e-6 * np.linalg.norm(g)
        noise = np.random.randn(n)
        r = g + noise_scale * noise
        # Validation instrumentation: capture the actual realized (post-noise)
        # gradient so the C++ port can be replayed against the literal vector
        # Python used (GltrConfig.add_noise = false, fed this r directly) --
        # rather than disabling the noise draw, which would remove the very
        # mechanism this hack exists to provide (steering away from exact
        # hard-case degeneracies). No effect when CPP_CASSCF_VALIDATION_DIR
        # is unset.
        if _CPP_CASSCF_VALIDATION_DIR:
            self._cpp_casscf_last_noised_gradient = r.copy()

        z = r / M_abs
        inner_prod = np.dot(r, z)
        
        if inner_prod <= 1e-20:
            return np.zeros(n), 0.0, "success_zero_gradient"
            
        beta_0 = np.sqrt(inner_prod)
        Q[0] = z / beta_0
        
        hist_len = 5
        hist_eig = deque(maxlen=hist_len)
        hist_norm = deque(maxlen=hist_len)
        
        if verbose:
            print(f"\n--- GLTR START (Rad={radius:.4f}, Dim={n}) ---")
            print(f"{'Iter':<5} | {'Min Eig':<12} | {'Step Norm':<12} | {'Residual':<12} | {'Status'}")
    
        for k in range(max_iter):
            v_curr = Q[k]
            
            H_v = B_operator(v_curr)
            al = np.dot(v_curr, H_v)
            alpha[k] = al
            
            w = H_v / M_abs
            if k == 0:
                r_next = w - al * v_curr
            else:
                r_next = w - al * v_curr - beta[k-1] * Q[k-1]
                
            for j in range(k + 1):
                overlap = np.dot(r_next, Q[j] * M_abs)
                r_next = r_next - overlap * Q[j]
    
            inner_prod = np.dot(r_next, r_next * M_abs)
            be = np.sqrt(max(0.0, inner_prod)) 
            beta[k] = be
            
            solve_final = False
            if be < 1e-12:
                solve_final = True
                curr_betas = beta[:k]
            else:
                if k < max_iter - 1:
                    Q[k+1] = r_next / be
                curr_betas = beta[:k]
    
            curr_alphas = alpha[:k+1]
            try:
                w_eig, v_eig = eigh_tridiagonal(curr_alphas, curr_betas)
            except ValueError:
                print("GLTR: Tridiagonal solver failed.")
                break
    
            min_eig = np.min(w_eig)
            g_inner = beta_0 * v_eig[0, :]
            
            p_current = None
            step_norm_val = 0.0
            resid_val = 0.0
            status_tag = ""
            final_lambda = 0.0 # Default for Newton step
    
            # --- CHECK A: Interior Newton Step ---
            if min_eig > 1e-12:
                z_newton = -g_inner / w_eig
                y_T_newton = v_eig @ z_newton
                
                p_newton = np.zeros(n)
                for i in range(k+1):
                    p_newton += y_T_newton[i] * Q[i]
                    
                norm_newton = np.linalg.norm(p_newton)
                resid = be * abs(y_T_newton[-1])
                
                p_current = p_newton
                step_norm_val = norm_newton
                resid_val = resid
                
                if norm_newton <= radius:
                    if resid < tol or solve_final:
                        if verbose: print(f"{k:<5} | {min_eig: .4e}   | {norm_newton:.4e}   | {resid:.4e}   | CONVERGED (Newton)")
                        return p_newton, 0.0, f"Interior (Newton) iter {k+1}"
                    status_tag = "Newton"
                else:
                    status_tag = "Newton(Out)"
            else:
                status_tag = "Indefinite"
    
            # --- CHECK B: Boundary / Hard Case ---
            should_solve_boundary = solve_final or (k == max_iter - 1) or (min_eig < 0) or (min_eig > 0 and step_norm_val > radius)
            
            if should_solve_boundary:
                # 1. Secular Function
                def get_norm_error(lam):
                    # Calculate norm of p(lam) - radius
                    z_local = -g_inner / (w_eig + lam)
                    coeffs = v_eig @ z_local
                    
                    # Fast reconstruction of norm (since vectors are not orthogonal in Euclidean metric)
                    p_temp = np.zeros(n)
                    for i in range(k+1):
                        p_temp += coeffs[i] * Q[i]
                    return np.linalg.norm(p_temp) - radius

                # 2. Establish Bounds
                # lam_min must be slightly above the singularity to avoid division by zero
                # singularity is at -min_eig (which is positive if min_eig is negative)
                lam_min = max(0, -min_eig) + 1e-6
                
                # --- CRITICAL CHECK FOR HARD CASE ---
                # Check the norm at the closest safe point to the singularity.
                # If the norm here is ALREADY smaller than radius, we are in the Hard Case.
                # The function ||p(lambda)|| is monotonically decreasing. 
                # If it starts below Delta, it will never hit Delta.
                
                err_low = get_norm_error(lam_min)
                
                if err_low < 0:
                    # HARD CASE: No root exists. 
                    # Stop Bisection immediately.
                    lam_final = lam_min
                    status_tag += "/Hard"
                    
                    # We will handle the eigenvector addition in Step 4 below.
                    
                else:
                    # STANDARD CASE: Root exists.
                    # We need an upper bound where ||p(lambda)|| < Delta (error < 0).
                    # Since ||p(lambda)|| -> 0 as lambda -> infinity, such an upper bound must exist.
                    
                    low = lam_min
                    high = lam_min + 1.0 # Initial guess for upper bound
                    
                    # Exponential search for valid upper bound
                    found_bracket = False
                    for _ in range(15):
                        if get_norm_error(high) < 0:
                            found_bracket = True
                            break
                        high = max(high * 2.0, high + 10.0)
                    
                    # Fallback if bracket not found (should be rare/impossible for valid inputs)
                    if not found_bracket:
                        lam_final = high 
                    else:
                        # Bisection / Root Finding
                        for _ in range(30):
                            mid = (low + high) / 2
                            err = get_norm_error(mid)
                            
                            if abs(err) < 1e-4 * radius:
                                low = mid
                                break
                            
                            if err > 0: 
                                # Norm too big -> Need larger lambda (to shrink step)
                                low = mid
                            else:
                                # Norm too small -> Need smaller lambda
                                high = mid
                        lam_final = low
                        status_tag += "/Bound"

                final_lambda = lam_final # Store for return
    
                z_boundary = -g_inner / (w_eig + lam_final)
                y_T_boundary = v_eig @ z_boundary
                
                p_boundary = np.zeros(n)
                for i in range(k+1):
                    p_boundary += y_T_boundary[i] * Q[i]
                    
                if err_low < 0:
                    idx_min = np.argmin(w_eig)
                    if abs(g_inner[idx_min]) < 1e-6:
                        curr_norm = np.linalg.norm(p_boundary)
                        if curr_norm < radius:
                            y_eig = v_eig[:, idx_min]
                            eig_vec = np.zeros(n)
                            for i in range(k+1):
                                eig_vec += y_eig[i] * Q[i]
                            tau = np.sqrt(max(0.0, radius**2 - curr_norm**2))
                            p_boundary += tau * (eig_vec / np.linalg.norm(eig_vec))
    
                p_current = p_boundary
                step_norm_val = np.linalg.norm(p_boundary)
                resid_val = be * abs(y_T_boundary[-1])
    
                if resid_val < tol or solve_final:
                    if verbose: print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | CONVERGED (Boundary)")
                    return p_boundary, final_lambda, f"Boundary iter {k+1}"
    
            ## --- CHECK C: Stagnation ---
            #hist_eig.append(min_eig)
            #hist_norm.append(step_norm_val)
            #
            #if len(hist_eig) == hist_len:
            #    eig_range = max(hist_eig) - min(hist_eig)
            #    norm_range = max(hist_norm) - min(hist_norm)
            #    if eig_range < 1e-5 and norm_range < 1e-5 * radius:
            #        if verbose: print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | CONVERGED (Stagnation)")
            #        return p_current, final_lambda, f"Stagnation iter {k+1}"
    
            if _QED_SOLVER_STATS:
                # per-Lanczos-iteration convergence trace: lets the report say how
                # many iterations earlier this solve could have exited at a looser
                # tolerance, without spending an extra matvec to find out.
                _solver_stats.trace_iter(resid_val, step_norm_val, tol)
            if verbose:
                print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | {status_tag}")

        return p_current if p_current is not None else np.zeros(n), 0.0, "Failure"



    def solve_gltr_trust_region(self, U, A_tilde, G_blocks, g, M_diag, radius, max_iter=100, tol=1e-4, verbose=True):
        """
        Returns: (p, lambda, status)
        lambda is the shift such that (H + lambda*M) is positive semidefinite.
        """
        n = len(g)
        if max_iter is None or max_iter > n:
            max_iter = n

        # QED_GLTR_TOL_SCALE (see the knob block at the top of this file);
        # no-op at the default 1.0.
        tol = tol * _QED_GLTR_TOL_SCALE

        Q = np.zeros((max_iter + 1, n))
        alpha = np.zeros(max_iter)
        beta = np.zeros(max_iter)

        M_abs = np.abs(M_diag) + 1e-12

        # QED_GLTR_REUSE: a rejected step re-enters this solver with the SAME
        # Hessian and gradient and only a smaller radius -- the reject branch
        # (~13037) touches nothing else, self.U3 is re-seeded from self.U2 at
        # ~12148, and only the ACCEPT branch updates U2 / rebuilds the gradient
        # and the Hessian diagonal.  The Lanczos recurrence below depends only on
        # (H, M_abs, r): the radius enters at CHECK A/B, never in the vectors.
        # So the cached basis is EXACT for the retry and the step is bit-identical
        # to what a cold rebuild produces -- the property GLTR is built on
        # (Gould/Lucidi/Roma/Toint: re-solution over a sequence of radii).
        # Keyed on the CONTENT of g and M_diag, so an accepted step invalidates
        # the cache automatically with no manual bookkeeping.
        _cache = getattr(self, "_gltr_cache", None)
        _n_cached = 0
        if (
            _QED_GLTR_REUSE
            and _cache is not None
            and _cache["n"] == n
            and radius < _cache["radius"]
            and np.array_equal(_cache["g"], g)
            and np.array_equal(_cache["M_diag"], M_diag)
        ):
            _n_cached = _cache["kmax"] + 1
            # Reuse the cached NOISED gradient too: the noise is redrawn per call,
            # and a different r generates a different Krylov space -- exactly what
            # the cache exists to avoid.
            r = _cache["r"].copy()
            alpha[:_n_cached] = _cache["alpha"][:_n_cached]
            beta[:_n_cached] = _cache["beta"][:_n_cached]
            _nq = min(_n_cached + 1, max_iter + 1)
            Q[:_nq] = _cache["Q"][:_nq]
            if verbose:
                print(
                    f"GLTR: reusing {_n_cached} cached Lanczos vectors "
                    f"(radius {_cache['radius']:.4f} -> {radius:.4f})"
                )
        else:
            #r = g.copy()
            # NEW: Add microscopic noise to see hidden negative curvature
            noise_scale = 1e-6 * np.linalg.norm(g)
            noise = np.random.randn(n)
            r = g + noise_scale * noise
        # Validation instrumentation: capture the actual realized (post-noise)
        # gradient so the C++ port can be replayed against the literal vector
        # Python used (GltrConfig.add_noise = false, fed this r directly) --
        # rather than disabling the noise draw, which would remove the very
        # mechanism this hack exists to provide (steering away from exact
        # hard-case degeneracies). No effect when CPP_CASSCF_VALIDATION_DIR
        # is unset.
        if _CPP_CASSCF_VALIDATION_DIR:
            self._cpp_casscf_last_noised_gradient = r.copy()

        z = r / M_abs
        inner_prod = np.dot(r, z)

        if inner_prod <= 1e-20:
            return np.zeros(n), 0.0, "success_zero_gradient"

        beta_0 = np.sqrt(inner_prod)
        Q[0] = z / beta_0

        hist_len = 5
        hist_eig = deque(maxlen=hist_len)
        hist_norm = deque(maxlen=hist_len)

        if verbose:
            print(f"\n--- GLTR START (Rad={radius:.4f}, Dim={n}) ---")
            print(f"{'Iter':<5} | {'Min Eig':<12} | {'Step Norm':<12} | {'Residual':<12} | {'Status'}")

        for k in range(max_iter):
            v_curr = Q[k]

            if k < _n_cached:
                # QED_GLTR_REUSE hit: alpha[k], beta[k] and Q[k+1] were produced by
                # the discarded solve from the identical (H, M_abs, r).  Skip the
                # Hessian-vector product -- that is the entire point of the cache.
                al = alpha[k]
                be = beta[k]
            else:
                v_curr_reshaped = v_curr.reshape(1, n)
                H_v = np.zeros_like(v_curr_reshaped)
                self.orbital_sigma3(U, A_tilde, G_blocks, v_curr_reshaped, H_v, 1, 0)
                H_v = H_v.reshape(n)

                al = np.dot(v_curr, H_v)
                alpha[k] = al

                w = H_v / M_abs
                if k == 0:
                    r_next = w - al * v_curr
                else:
                    r_next = w - al * v_curr - beta[k-1] * Q[k-1]

                for j in range(k + 1):
                    overlap = np.dot(r_next, Q[j] * M_abs)
                    r_next = r_next - overlap * Q[j]

                inner_prod = np.dot(r_next, r_next * M_abs)
                be = np.sqrt(max(0.0, inner_prod))
                beta[k] = be

                if be >= 1e-12 and k < max_iter - 1:
                    Q[k+1] = r_next / be

            solve_final = be < 1e-12
            curr_betas = beta[:k]

            if _QED_GLTR_REUSE:
                # Bank the basis as it grows, so ANY of the returns below leaves a
                # usable cache.  Q is trimmed to what has actually been filled
                # (the full array is max_iter+1 rows and would pin ~46 MB on c10h8).
                # Never shrink below what we inherited: a retry that converges in
                # fewer iterations than the cached solve must not truncate the
                # basis for the NEXT retry in the same cascade.
                _bank = max(k, _n_cached - 1)
                self._gltr_cache = {
                    "n": n,
                    "g": g.copy(),
                    "M_diag": M_diag.copy(),
                    "r": r.copy(),
                    "alpha": alpha[:_bank+1].copy(),
                    "beta": beta[:_bank+1].copy(),
                    "Q": Q[:min(_bank+2, max_iter+1)].copy(),
                    "kmax": _bank,
                    "radius": radius,
                }

            curr_alphas = alpha[:k+1]
            try:
                w_eig, v_eig = eigh_tridiagonal(curr_alphas, curr_betas)
            except ValueError:
                print("GLTR: Tridiagonal solver failed.")
                break
    
            min_eig = np.min(w_eig)
            g_inner = beta_0 * v_eig[0, :]
            
            p_current = None
            step_norm_val = 0.0
            resid_val = 0.0
            status_tag = ""
            final_lambda = 0.0 # Default for Newton step
    
            # --- CHECK A: Interior Newton Step ---
            if min_eig > 1e-12:
                z_newton = -g_inner / w_eig
                y_T_newton = v_eig @ z_newton
                
                p_newton = np.zeros(n)
                for i in range(k+1):
                    p_newton += y_T_newton[i] * Q[i]
                    
                norm_newton = np.linalg.norm(p_newton)
                resid = be * abs(y_T_newton[-1])
                
                p_current = p_newton
                step_norm_val = norm_newton
                resid_val = resid
                
                if norm_newton <= radius:
                    if resid < tol or solve_final:
                        if verbose: print(f"{k:<5} | {min_eig: .4e}   | {norm_newton:.4e}   | {resid:.4e}   | CONVERGED (Newton)")
                        return p_newton, 0.0, f"Interior (Newton) iter {k+1}"
                    status_tag = "Newton"
                else:
                    status_tag = "Newton(Out)"
            else:
                status_tag = "Indefinite"
    
            # --- CHECK B: Boundary / Hard Case ---
            should_solve_boundary = solve_final or (k == max_iter - 1) or (min_eig < 0) or (min_eig > 0 and step_norm_val > radius)
            
            if should_solve_boundary:
                # 1. Secular Function
                def get_norm_error(lam):
                    # Calculate norm of p(lam) - radius
                    z_local = -g_inner / (w_eig + lam)
                    coeffs = v_eig @ z_local
                    
                    # Fast reconstruction of norm (since vectors are not orthogonal in Euclidean metric)
                    p_temp = np.zeros(n)
                    for i in range(k+1):
                        p_temp += coeffs[i] * Q[i]
                    return np.linalg.norm(p_temp) - radius

                # 2. Establish Bounds
                # lam_min must be slightly above the singularity to avoid division by zero
                # singularity is at -min_eig (which is positive if min_eig is negative)
                lam_min = max(0, -min_eig) + 1e-6
                
                # --- CRITICAL CHECK FOR HARD CASE ---
                # Check the norm at the closest safe point to the singularity.
                # If the norm here is ALREADY smaller than radius, we are in the Hard Case.
                # The function ||p(lambda)|| is monotonically decreasing. 
                # If it starts below Delta, it will never hit Delta.
                
                err_low = get_norm_error(lam_min)
                
                if err_low < 0:
                    # HARD CASE: No root exists. 
                    # Stop Bisection immediately.
                    lam_final = lam_min
                    status_tag += "/Hard"
                    
                    # We will handle the eigenvector addition in Step 4 below.
                    
                else:
                    # STANDARD CASE: Root exists.
                    # We need an upper bound where ||p(lambda)|| < Delta (error < 0).
                    # Since ||p(lambda)|| -> 0 as lambda -> infinity, such an upper bound must exist.
                    
                    low = lam_min
                    high = lam_min + 1.0 # Initial guess for upper bound
                    
                    # Exponential search for valid upper bound
                    found_bracket = False
                    for _ in range(15):
                        if get_norm_error(high) < 0:
                            found_bracket = True
                            break
                        high = max(high * 2.0, high + 10.0)
                    
                    # Fallback if bracket not found (should be rare/impossible for valid inputs)
                    if not found_bracket:
                        lam_final = high 
                    else:
                        # Bisection / Root Finding
                        for _ in range(30):
                            mid = (low + high) / 2
                            err = get_norm_error(mid)
                            
                            if abs(err) < 1e-4 * radius:
                                low = mid
                                break
                            
                            if err > 0: 
                                # Norm too big -> Need larger lambda (to shrink step)
                                low = mid
                            else:
                                # Norm too small -> Need smaller lambda
                                high = mid
                        lam_final = low
                        status_tag += "/Bound"

                final_lambda = lam_final # Store for return
    
                z_boundary = -g_inner / (w_eig + lam_final)
                y_T_boundary = v_eig @ z_boundary
                
                p_boundary = np.zeros(n)
                for i in range(k+1):
                    p_boundary += y_T_boundary[i] * Q[i]
                    
                if err_low < 0:
                    idx_min = np.argmin(w_eig)
                    if abs(g_inner[idx_min]) < 1e-6:
                        curr_norm = np.linalg.norm(p_boundary)
                        if curr_norm < radius:
                            y_eig = v_eig[:, idx_min]
                            eig_vec = np.zeros(n)
                            for i in range(k+1):
                                eig_vec += y_eig[i] * Q[i]
                            tau = np.sqrt(max(0.0, radius**2 - curr_norm**2))
                            p_boundary += tau * (eig_vec / np.linalg.norm(eig_vec))
    
                p_current = p_boundary
                step_norm_val = np.linalg.norm(p_boundary)
                resid_val = be * abs(y_T_boundary[-1])
    
                if resid_val < tol or solve_final:
                    if verbose: print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | CONVERGED (Boundary)")
                    return p_boundary, final_lambda, f"Boundary iter {k+1}"
    
            ## --- CHECK C: Stagnation ---
            #hist_eig.append(min_eig)
            #hist_norm.append(step_norm_val)
            #
            #if len(hist_eig) == hist_len:
            #    eig_range = max(hist_eig) - min(hist_eig)
            #    norm_range = max(hist_norm) - min(hist_norm)
            #    if eig_range < 1e-5 and norm_range < 1e-5 * radius:
            #        if verbose: print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | CONVERGED (Stagnation)")
            #        return p_current, final_lambda, f"Stagnation iter {k+1}"
    
            if _QED_SOLVER_STATS:
                # per-Lanczos-iteration convergence trace: lets the report say how
                # many iterations earlier this solve could have exited at a looser
                # tolerance, without spending an extra matvec to find out.
                _solver_stats.trace_iter(resid_val, step_norm_val, tol)
            if verbose:
                print(f"{k:<5} | {min_eig: .4e}   | {step_norm_val:.4e}   | {resid_val:.4e}   | {status_tag}")

        return p_current if p_current is not None else np.zeros(n), 0.0, "Failure"




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
        # Attributes this Davidson solve's matvecs separately from the LSTRS
        # bisection that wraps it, split by how the subspace was seeded.
        # NOTE the parameter name reads backwards: restart=True means "restart
        # FROM the previous search space" (Q = deepcopy(guess_vector), the warm
        # start used for every alpha after the first), while restart=False
        # builds a fresh guess subspace from |g|/H_diag.  So restart=True is the
        # CHEAP path.  Labelled by what they do, not by the flag's name.
        _stats_span_d = _stats_begin(
            "davidson_ah_warmstart" if restart else "davidson_ah_coldbuild"
        )
        # Analyze problem structure
        n_negative = np.sum(reduced_hessian_diagonal < 0)
        min_diag = np.min(reduced_hessian_diagonal)
        max_diag = np.max(reduced_hessian_diagonal)
        grad_norm = np.linalg.norm(reduced_gradient)
        
        print(f"Problem structure:")
        print(f"  Negative curvature directions: {n_negative}/{self.index_map_size}")
        print(f"  H_diag range: [{min_diag:.2e}, {max_diag:.2e}]")
        print(f"  ||gradient||: {grad_norm:.2e}")
        print(f"  Condition estimate: {max_diag/abs(min_diag) if min_diag != 0 else 'inf'}") 
        print("collapse subspace check", self.collapse_subspace_check)
        threshold = 1e-7
        count = 0
        # start = timer()
        print("restart", restart, "alpha", alpha)
        for i in range(self.index_map_size):
            if reduced_hessian_diagonal[i] <= 1e-14:
                count += 1
        if self.index_map_size > 600:
            dim0 = 200
        else:
            dim0 = self.index_map_size // 2
            #dim0 = self.index_map_size 
        dim1 = max(count, dim0)
        # print("eepp",reduced_hessian_diagonal)

        dim2 = self.index_map_size + 1
        H_dim = dim2
        H_diag0 = copy.deepcopy(reduced_hessian_diagonal)
        # print(H_diag)
        H_diag = np.concatenate(([alpha], H_diag0))
        #print("H_diag", H_diag)
        indim = dim1 + 1
        # maxdim = dim1 + 3
        maxdim = min(H_dim, dim1 + min(dim1, 40))
        self.project_augmented_hessian00 = np.zeros((maxdim, maxdim))
        print("indim", indim, "maxdim", maxdim, "Hdim", H_dim)
        H_pp = np.zeros((indim, indim))
        unconverged_idx = []
        if alpha_range > 1e-5:
            nroots = 1
            roots_to_check = [0]
        else:
            nroots = 2
            roots_to_check = [0, 1]

        nroots_target = 2
        root_0_locked = False

        # end   = timer()
        # print("guess1 took", end - start)

        # start = timer()
        if restart == False:
            # start1 = timer()
            out = np.empty(self.index_map_size)
            out.fill(1e14)
            d = np.divide(
                    np.absolute(reduced_gradient),
                    reduced_hessian_diagonal,
                    out,
                    where=reduced_hessian_diagonal > 1e-5,
                    )
            

            self.idx_hessian = (-d).argsort()[:dim1]

            self.indim = indim


            #print(d)
            #print("order", self.idx_hessian)
            self.guess_hessian = np.zeros((dim1, dim1))
            self.guess_gradient = np.zeros(dim1)
            sym_A_tilde = A_tilde + A_tilde.T
            # end1   = timer()
            # print("building orbital guess1 took1", end1 - start1, flush = True)

            

            print("davidson iteration", 1)
            self.build_orbital_hessian_guess(
                    U,
                    sym_A_tilde,
                    reduced_gradient,
                    G,
                    self.guess_hessian,
                    self.guess_gradient,
                    dim1,
                    self.idx_hessian,
                    )
            if _CPP_CASSCF_VALIDATION_DIR:
                _dump_cpp_casscf_validation_case(
                    "hessian_guess",
                    U=U, sym_A_tilde=sym_A_tilde, reduced_gradient=reduced_gradient, G=G,
                    idx=np.asarray(self.idx_hessian, dtype=float),
                    guess_hessian=self.guess_hessian, guess_gradient=self.guess_gradient,
                    dims=np.array([self.n_in_a, self.n_act_orb, self.n_virtual, self.nmo, self.n_occupied]),
                )

            aug_eigvecs = np.zeros((dim1 + 1, dim1 + 1))
            Lmax = maxdim
            L = indim

            H_pp = np.zeros((indim, indim))
            H_pp[0, 0] = alpha
            H_pp[0, 1:] = self.guess_gradient
            H_pp[1:, 0] = self.guess_gradient.T
            H_pp[1:, 1:] = self.guess_hessian
            self.H_pp = copy.deepcopy(H_pp)
            self.H_qp = np.zeros((0, indim))
            self.sigma_total = np.zeros((maxdim, H_dim))
            Q = np.zeros((indim, H_dim))
            projected_step = np.zeros(dim1)
            theta = self.projection_step2(
                self.guess_gradient, self.guess_hessian, aug_eigvecs, alpha, dim1 + 1
            )

            # start1 = timer()
            for i in range(dim1 + 1):
                for j in range(dim1):
                    index1 = self.idx_hessian[j]
                    Q[i][0] = aug_eigvecs[0][i]
                    Q[i][index1 + 1] = aug_eigvecs[j + 1][i]
            
            
            num_to_build = len(roots_to_check)
            w_sigma = np.zeros((num_to_build, H_dim)) # Holds sigma vectors
            # Build sigma only for active roots (e.g., [0] or [0, 1])
            ##self.orbital_sigma(U, A_tilde, G1, Q, w, nroots, 1)
            start = timer()
            Q_slice = np.ascontiguousarray(Q[roots_to_check, :])
            self.orbital_sigma3(U, A_tilde, G_blocks, Q_slice, w_sigma, num_to_build, 1)
            end = timer()
            print("build orbital sigma for w took", end - start)


            w = np.zeros((nroots_target, H_dim)) # Full residual matrix
            residual_norm = np.full(nroots_target, np.inf)
            conv_status = [False] * nroots_target
            
            for i, root_idx in enumerate(roots_to_check):
                vec = Q[root_idx, :]
                w_vec = w_sigma[i, :] # Get the corresponding sigma
                w_vec[1:] += reduced_gradient * vec[0]
                w_vec[0] = alpha * vec[0] + np.dot(vec[1:], reduced_gradient)
                
                r_vec = w_vec - theta[root_idx] * vec
                w[root_idx, :] = r_vec # Store the *residual*
                residual_norm[root_idx] = np.linalg.norm(r_vec)
                
                if residual_norm[root_idx] < threshold:
                    conv_status[root_idx] = True 



            unconverged_idx = []
            exit_solver = False
            converged_indices = []
            # --- NEW LSTRS CONVERGENCE CHECK (Initial Guess) ---
            if nroots == 1:
                if conv_status[0]:
                    converged_indices.append(0)
                    # Root 0 is converged. Call the callback.
                    if self.check_root_0(Q, theta[0], reduced_gradient, trust_radius):
                        # === EASY CASE ===
                        print("LSTRS (Initial Guess): Root 0 converged (Easy Case). Exiting.")
                        aug_hessian_eigenvals[0] = theta[0]
                        aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                        aug_hessian_eigenvals[1] = 1e10 # Fake root 1
                        aug_hessian_eigenvecs[:, 1] = 1e-14
                        exit_solver = True
                    else:
                        # === HARD CASE ===
                        print("LSTRS (Initial Guess): Root 0 converged (Hard Case). Checking Root 1.")
                        root_0_locked = True
                        roots_to_check = [1] # Now we only care about Root 1
                        
                        # We must check Root 1 *now*
                        start1 = timer()
                        w1_sigma = np.zeros((1, H_dim))
                        self.orbital_sigma3(U, A_tilde, G_blocks, Q[1:2, :], w1_sigma, 1, 1)
                        #w1_sigma[0, 1:] += np.einsum("p,q->qp", reduced_gradient, Q[1, 0])
                        w1_sigma[0, 1:] += reduced_gradient * Q[1, 0] # Changed from einsum
                        w1_sigma[0, 0] = alpha * Q[1, 0] + np.dot(Q[1, 1:], reduced_gradient)
                        end1 = timer()
                        print("build orbital sigma for w took", end1 - start1)

           
                        r_1 = w1_sigma[0, :] - theta[1] * Q[1, :]
                        w[1, :] = r_1 # Store residual
                        residual_norm[1] = np.linalg.norm(r_1)
                        
                        if residual_norm[1] < threshold:
                            converged_indices.append(1)
                            conv_status[1] = True
                            print("LSTRS (Initial Guess): Root 1 also converged. Exiting.")
                            exit_solver = True
                            aug_hessian_eigenvals[0] = theta[0]
                            aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                            aug_hessian_eigenvals[1] = theta[1]
                            aug_hessian_eigenvecs[:, 1] = Q[1, :].T
                        else:
                            unconverged_idx = [1]

                        #if is_root_1_conv:
                        #    print("LSTRS (Initial Guess): Root 1 also converged. Exiting.")
                        #    aug_hessian_eigenvals[0] = theta[0]
                        #    aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                        #    aug_hessian_eigenvals[1] = theta[1]
                        #    aug_hessian_eigenvecs[:, 1] = Q[1, :].T
                        #    exit_solver = True
                        #else:
                        #    print("LSTRS (Initial Guess): Root 1 unconverged. Adding to search.")
                        #    unconverged_idx = [1]
                else:
                    # Root 0 is not converged
                    unconverged_idx = [0]
            elif nroots == 2:
                if conv_status[0] and conv_status[1]:
                    if conv_status[0]: converged_indices.append(0)
                    if conv_status[1]: converged_indices.append(1)
                    # Both are converged
                    print("LSTRS: Both roots converged. Exiting.")
                    aug_hessian_eigenvals[:] = theta[:nroots_target]
                    aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                    exit_solver = True
                else:
                    # Find whichever is not converged
                    if not conv_status[0]: unconverged_idx.append(0)
                    if not conv_status[1]: unconverged_idx.append(1)
                    #if not is_root_0_conv:
                    #    unconverged_idx.append(0)
                    #if not is_root_1_conv:
                    #    unconverged_idx.append(1)

            print(unconverged_idx)
            #print("root", "AH residual norm", "Eigenvalue", "Convergence")
            #for j in range(nroots_target):
            #    print(
            #        j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
            #    )
            print("root", "AH residual norm", "Eigenvalue", "Convergence")
            for j in range(nroots_target):
                print(
                    j + 1, residual_norm[j], theta[j], "Yes" if conv_status[j] else "No", flush=True
                )

            if exit_solver:
                # self.projected_augmented_hessian = np.block([[0, self.guess_gradient], [self.guess_gradient.T, self.guess_hessian]])
                self.projected_augmented_hessian = np.zeros((L, L))
                self.projected_augmented_hessian[0, 0] = 0
                self.projected_augmented_hessian[0, 1:] = self.guess_gradient
                self.projected_augmented_hessian[1:, 0] = self.guess_gradient.T
                self.projected_augmented_hessian[1:, 1:] = self.guess_hessian

                print("converged from the first iteration!", flush=True)

            # end1   = timer()
            # print("building orbital guess7 took", end1 - start1, flush = True)
            QQ = np.zeros((dim1, self.index_map_size))
            for i in range(dim1):
               index1 = self.idx_hessian[i]
               r = self.index_map[index1][0]
               k = self.index_map[index1][1]
               #guess_gradient2[i] = reduced_gradient[index1]
               QQ[i][index1] = 1.0
            Q = np.zeros((indim, H_dim))
            Q[0,0] = 1
            Q[1:,1:] = QQ

            preconditioned_w = np.zeros((0, H_dim)) # Initialize empty
            if len(unconverged_idx) > 0:
                w_unconv = w[unconverged_idx, :]
                precon_denom = theta[unconverged_idx].reshape(
                    len(unconverged_idx), 1
                ) - H_diag.reshape(1, H_dim)
                preconditioned_w = np.divide(
                    w_unconv, 
                    precon_denom,
                    out=np.zeros_like(w_unconv),
                    where=precon_denom != 0,
                )
           
            
            Q = np.concatenate((Q, preconditioned_w), axis=0)
            c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
            L = Q.shape[0]
            print("current dimension", L, flush = True)
            
        else:
            print("restart from previous search space")
            # if indim == maxdim:
            #    maxdim +=1
            #print(H_dim, indim)
            Q = copy.deepcopy(guess_vector)
            Lmax = maxdim
            L = Q.shape[0]
            indim = self.indim
            print("current dim", L)
            print("indim", indim)
            if self.collapse_subspace_check == False:
                if L == indim:
                    self.projected_augmented_hessian[0, 0] = alpha
                else:
                    print("indim", indim)
                    start1 = timer()
                    b_dim = L - indim
                    self.projected_augmented_hessian[0, 0] = alpha
                    H_qp = np.zeros((b_dim, indim))
                    Sq = np.zeros((b_dim, H_dim))
                    Sq[:, 0] += alpha * Q[indim:, 0]
                    self.sigma_total[indim:L,0] += Sq[:, 0]
                    if indim > 0:
                        for i in range(b_dim):
                            self.projected_augmented_hessian[indim + i][0] += Sq[i][0]
                            self.projected_augmented_hessian[0][indim + i] += Sq[i][0]
                            self.H_qp[i][0] += Sq[i][0]
                            # for j in range(dim1):
                            #    #index1 = self.idx_hessian[j]
                            #    #H_qp[i][0] = Sq[i][0]
                            #    #H_qp[i][j+1] = Sq[i][index1+1]
                        self.H_pp[0, 0] = alpha
                        H_qq = np.dot(Sq, Q[indim:, :].T)
                        self.projected_augmented_hessian[indim:, indim:] += H_qq
                    else:
                        self.projected_augmented_hessian[indim:, indim:] = np.dot(self.sigma_total[:L,:], Q[:L, :].T)
                        print("qqqq")
                    theta, aug_eigvecs = np.linalg.eigh(self.projected_augmented_hessian)
                    #print("iiii", theta)
                    end1 = timer()
                    print("restart took", end1 - start1)
            else:
                L_new = Q.shape[0]
                Sq = np.zeros((L_new, H_dim))
                Sq[:, 0] += alpha * Q[:, 0]
                self.sigma_total[:L_new,0] += Sq[:,0]
                self.projected_augmented_hessian = np.dot(self.sigma_total[:L_new,:], Q[:L_new, :].T)
                             
                # Set unconverged indices based on current LSTRS state
                if nroots == 1 and not root_0_locked:
                    unconverged_idx = [0]
                elif nroots == 1 and root_0_locked:
                    unconverged_idx = [1]
                elif nroots == 2:
                    unconverged_idx = [0, 1]
                # --- END MODIFIED RESTART=TRUE BLOCK ---
        if (len(unconverged_idx) > 0 and restart == False) or restart == True:
            num_iter = 10000
            # num_iter = 20
            L_old = indim
            collapse = True
            for davidson_iteration in range(1, num_iter):
                if davidson_iteration == num_iter - 1:
                    print(
                        "maximum iteration reaches but no converged roots have been found"
                    )
                    exit()
                print("\n")
                L_new = Q.shape[0]
                L = Q.shape[0]
                projected_step = np.zeros(L - 1)
                full_eigvecs = np.zeros((nroots, H_dim))
                full_eigvecs2 = np.zeros((nroots, H_dim))
                projected_augmented_hessian = np.zeros((1, 1))
                if restart == True:
                    print("davidson iteration", davidson_iteration)
                else:
                    print("davidson iteration", davidson_iteration + 1)
                if restart == True and davidson_iteration == 1:
                    projected_augmented_hessian = self.projected_augmented_hessian
                else:
                    if self.collapse_subspace_check == False:   
                        

                        start = timer()
                        c_dim = L_new - L_old
                        Sq1 = np.zeros((c_dim, H_dim))
                        # self.build_sigma_reduced2(U, A_tilde, G, Q[indim:,:], Sq0, b_dim, 1)
                        # c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q[indim:,:], Sq0, b_dim, 1, self.nmo, self.index_map_size, self.n_occupied)
                        #self.orbital_sigma(U, A_tilde, G1,  Q[L_old:, :], Sq1, c_dim, 1)
                        
                        self.orbital_sigma3(U, A_tilde, G_blocks,  Q[L_old:, :], Sq1, c_dim, 1)
                        
                        gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[L_old:, 0])
                        gradient2 = np.dot(Q[L_old:, 1:], reduced_gradient)
                        print("c dim, L old, L new", c_dim, L_old, L_new) 
                        end = timer()
                        print("build orbital sigma for Q space took", end - start)
                        Sq = copy.deepcopy(Sq1)
                        # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                        Sq[:, 1:] += gradient1
                        # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                        Sq[:, 0] = alpha * Q[L_old:, 0] + gradient2
                        self.sigma_total[L_old:L_new,:] = np.copy(Sq)
                        if indim > 0:
                            H_qp = np.zeros((c_dim, indim))
                            for i in range(c_dim):
                                for j in range(dim1):
                                    index1 = self.idx_hessian[j]
                                    H_qp[i][0] = Sq[i][0]
                                    H_qp[i][j + 1] = Sq[i][index1 + 1]
                            self.H_qp = np.vstack([self.H_qp, H_qp])
                            H_qq = np.dot(self.sigma_total[indim:L_new,:], Q[indim:, :].T)
                            print("dimension of sigma", np.shape(self.sigma_total))
                            H1 = np.concatenate((self.H_pp, self.H_qp), axis=0)
                            H2 = np.concatenate((self.H_qp.T, H_qq), axis=0)
                            self.projected_augmented_hessian00 = np.concatenate((H1, H2), axis=1)
                            projected_augmented_hessian = np.concatenate((H1, H2), axis=1)
                        else: 
                            # Pure Q-space case (Krylov guess start)
                            # The projected matrix is just the Q-space projection H_qq
                            H_qq = np.dot(self.sigma_total[:L_new,:], Q[:L_new, :].T)

                            # No H_pp, H_qp, or H_pq to concatenate
                            self.projected_augmented_hessian00 = H_qq
                            projected_augmented_hessian = H_qq
                            #self.print_matrix_nice(H_qq, precision=10, width=14, cols_per_line=6)
                    else:   
                        print("collapse", collapse)
                        if collapse == False:
                            L_new = Q.shape[0]
                            c_dim = L_new - L_old
                            Sq1 = np.zeros((c_dim, H_dim))
                            # self.build_sigma_reduced2(U, A_tilde, G, Q[indim:,:], Sq0, b_dim, 1)
                            # c_build_sigma_reduced(U, A_tilde, self.index_map1, G1, Q[indim:,:], Sq0, b_dim, 1, self.nmo, self.index_map_size, self.n_occupied)
                            start = timer()
                            #self.orbital_sigma(U, A_tilde, G1,  Q[L_old:, :], Sq1, c_dim, 1)
                            self.orbital_sigma3(U, A_tilde, G_blocks,  Q[L_old:, :], Sq1, c_dim, 1)
                            end = timer()
                            print("build orbital sigma for Q space took", end - start)

                            gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[L_old:, 0])
                            gradient2 = np.dot(Q[L_old:, 1:], reduced_gradient)
                            print("c dim, L old, L new", c_dim, L_old, L_new) 
                            Sq = copy.deepcopy(Sq1)
                            # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                            Sq[:, 1:] += gradient1
                            # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                            Sq[:, 0] = alpha * Q[L_old:, 0] + gradient2
                            self.sigma_total[L_old:L_new,:] = np.copy(Sq)
                            projected_augmented_hessian = np.dot(self.sigma_total[:L_new,:], Q[:L_new, :].T)
                            L_old = Q.shape[0]
                        else:
                            L_new = Q.shape[0]
                            self.sigma_total[:,:] = 0
                            c_dim = L_new
                            Sq1 = np.zeros((c_dim, H_dim))
                            start = timer()
                            #self.orbital_sigma(U, A_tilde, G1,  Q, Sq1, c_dim, 1)
                            self.orbital_sigma3(U, A_tilde, G_blocks,  Q, Sq1, c_dim, 1)
                            end = timer()
                            print("build orbital sigma for Q space took", end - start)

                            gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[:, 0])
                            gradient2 = np.dot(Q[:, 1:], reduced_gradient)
                            print("c dim, L old, L new", c_dim, L_old, L_new) 
                            Sq = copy.deepcopy(Sq1)
                            # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                            Sq[:, 1:] += gradient1
                            # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                            Sq[:, 0] = alpha * Q[:, 0] + gradient2
                            self.sigma_total[:L_new,:] = np.copy(Sq)
                            projected_augmented_hessian = np.dot(self.sigma_total[:L_new,:], Q[:L_new, :].T)

                theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
                #print("zopp",theta)
                # print(np.shape(aug_eigvecs))
                # print(aug_eigvecs.T[0,:])
                full_eigvecs = np.dot(aug_eigvecs.T[:nroots_target, :], Q[:,])
                full_eigvecs2 = copy.deepcopy(full_eigvecs)
                  


                start = timer()
                roots_to_check = []
                if nroots == 1 and not root_0_locked:
                    roots_to_check = [0]
                elif nroots == 1 and root_0_locked:
                    roots_to_check = [1]
                elif nroots == 2:
                    roots_to_check = [0, 1]
                
                num_to_build = len(roots_to_check)
                w = np.zeros((nroots_target, H_dim)) # Full residual matrix
                residual_norm = np.full(nroots_target, np.inf)
                conv_status = [False] * nroots_target
                
                if num_to_build > 0:
                    w_sigma = np.zeros((num_to_build, H_dim)) # For sigma vectors
                    
                    # --- CORE BUG FIX ---
                    if self.collapse_subspace_check == True:
                        # sigma_total *already* has gradient/alpha terms from the rebuild
                        w_sigma = np.dot(aug_eigvecs.T[roots_to_check, :], self.sigma_total[:L_new,:])
                    else:
                        # We must build sigma from scratch and add gradient/alpha terms
                        self.orbital_sigma3(U, A_tilde, G_blocks, full_eigvecs[roots_to_check, :], w_sigma, num_to_build, 1)
                        for i, root_idx in enumerate(roots_to_check):
                            vec = full_eigvecs[root_idx, :]
                            w_vec = w_sigma[i, :] # Get the corresponding sigma
                            w_vec[1:] += reduced_gradient * vec[0] 
                            w_vec[0] = alpha * vec[0] + np.dot(vec[1:], reduced_gradient)
                    # --- END FIX ---

                    # Now, calculate residuals using the correct w_sigma
                    for i, root_idx in enumerate(roots_to_check):
                        vec = full_eigvecs[root_idx, :]
                        w_vec = w_sigma[i, :] # This is (H*y + G*y)
                        r_vec = w_vec - theta[root_idx] * vec
                        w[root_idx, :] = r_vec # Store the *residual*
                        residual_norm[root_idx] = np.linalg.norm(r_vec)
                        
                        if residual_norm[root_idx] < threshold:
                            conv_status[root_idx] = True

                if root_0_locked:
                    conv_status[0] = True # It's locked, so it's "converged"
 
               

                

                unconverged_idx = []
                exit_solver = False
                # --- NEW LSTRS CONVERGENCE CHECK (Main Loop) ---
                if nroots == 1:
                    if conv_status[0] and not root_0_locked: # Root 0 *just* converged
                    #if is_root_0_conv:
                        if self.check_root_0(full_eigvecs, theta[0], reduced_gradient, trust_radius):
                            # === EASY CASE ===
                            print("LSTRS: Root 0 converged (Easy Case). Exiting.")
                            aug_hessian_eigenvals[0] = theta[0]
                            aug_hessian_eigenvecs[:, 0] = full_eigvecs[0, :].T
                            aug_hessian_eigenvals[1] = 1e10 # Fake root 1
                            aug_hessian_eigenvecs[:, 1] = 0.0
                            exit_solver = True
                        else:
                            # === HARD CASE ===
                            root_0_locked = True
                            print("LSTRS: Root 0 converged (Hard Case). Checking Root 1.")
                            # *Now* we must check Root 1 for the *first time*
                            start1 = timer()
                            w1_sigma = np.zeros((1, H_dim))
                            # --- CORE BUG FIX (Hard Case Check) ---
                            if self.collapse_subspace_check == True:
                                # sigma_total *already* has gradient/alpha terms
                                w1_sigma = np.dot(aug_eigvecs.T[1:2, :], self.sigma_total[:L_new,:])
                            else:
                                self.orbital_sigma3(U, A_tilde, G_blocks, full_eigvecs[1:2, :], w1_sigma, 1, 1) # <-- SLICE FIX
                                w1_sigma[0, 1:] += reduced_gradient * full_eigvecs[1, 0] # Changed from einsum
                                w1_sigma[0, 0] = alpha * full_eigvecs[1, 0] + np.dot(full_eigvecs[1, 1:], reduced_gradient)
                            # --- END FIX ---
                            
                            r_1 = w1_sigma[0, :] - theta[1] * full_eigvecs[1, :]
                            w[1, :] = r_1 # Store residual
                            residual_norm[1] = np.linalg.norm(r_1)
                            
                            #if is_root_1_conv:
                            if residual_norm[1] < threshold:
                                conv_status[1] = True
                                print("LSTRS: Root 1 also converged. Exiting.")
                                aug_hessian_eigenvals[:] = theta[:nroots_target]
                                aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                                exit_solver = True
                            else:
                                print("LSTRS: Root 1 unconverged. Adding to search.")
                                unconverged_idx = [1]
                    elif not conv_status[0]: # Root 0 still unconverged
                        unconverged_idx = [0]    
                    elif root_0_locked: # Root 0 is locked, check Root 1
                        if not conv_status[1]:
                            unconverged_idx = [1]
                    #else:
                    #    # Root 0 is not converged
                    #    unconverged_idx = [0]

                elif nroots == 2:
                    #if is_root_0_conv and is_root_1_conv:
                    #    # Both are converged
                    #    print("LSTRS: Both roots converged. Exiting.")
                    #    aug_hessian_eigenvals[:] = theta[:nroots_target]
                    #    aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                    #    exit_solver = True
                    #else:
                    # Find whichever is not converged
                    if not conv_status[0]: unconverged_idx.append(0)
                    if not conv_status[1]: unconverged_idx.append(1)
                    #if not is_root_0_conv:
                    #    unconverged_idx.append(0)
                    #if not is_root_1_conv:
                    #    unconverged_idx.append(1)
                # Check for final convergence
                if (nroots == 2 and not unconverged_idx) or \
                   (nroots == 1 and root_0_locked and not unconverged_idx):
                    print("LSTRS: All required roots converged.")
                    exit_solver = True
                    aug_hessian_eigenvals[:] = theta[:nroots_target]
                    aug_hessian_eigenvecs[:, :] = full_eigvecs.T
        
                # --- END NEW LOGIC ---
                #print(unconverged_idx)
                #print("root", "AH residual norm", "Eigenvalue", "Convergence")
                #for j in range(nroots_target):
                #    print(
                #        j + 1,
                #        residual_norm[j],
                #        theta[j],
                #        convergence_check[j],
                #        flush=True,
                #    )
                print(unconverged_idx)
                print("root", "AH residual norm", "Eigenvalue", "Convergence")
                for j in range(nroots_target):
                    print(
                        j + 1, residual_norm[j], theta[j], "Yes" if conv_status[j] else "No", flush=True
                    )
                 



                if exit_solver:
                    aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                    aug_hessian_eigenvals[:] = theta[:nroots_target]
                    if self.collapse_subspace_check == False:
                        Q1 = full_eigvecs
                        start1 = timer()
                        b_dim = L - indim
                        print("b_dim", b_dim)
                        print("indim", indim)

                        self.projected_augmented_hessian = copy.deepcopy(
                            projected_augmented_hessian
                        )
                        self.projected_augmented_hessian[0, 0] = alpha
                        if b_dim > 0:
                            Sq = np.zeros((b_dim, H_dim))
                            Sq[:, 0] = alpha * Q[indim:, 0]
                            self.sigma_total[indim:L,0] -= Sq[:, 0]
                            if indim > 0:
                                # H_qp = np.zeros((b_dim, indim))
                                for i in range(b_dim):
                                    self.projected_augmented_hessian[indim + i][0] -= Sq[i][0]
                                    self.projected_augmented_hessian[0][indim + i] -= Sq[i][0]
                                    self.H_qp[i][0] -= Sq[i][0]
                                    # for j in range(dim1):
                                    #    #index1 = self.idx_hessian[j]
                                    #    #H_qp[i][0] = Sq[i][0]
                                    #    #H_qp[i][j+1] = Sq[i][index1+1]
                                H_qq = np.dot(Sq, Q[indim:, :].T)
                                self.projected_augmented_hessian[indim:, indim:] -= H_qq
                            else:
                                self.projected_augmented_hessian[indim:, indim:] = 0.0 

                        end1 = timer()
                        print("initialize took", end1 - start1)
                    else:
                        Sq = np.zeros((L_new, H_dim))
                        Sq[:, 0] = alpha * Q[:, 0]
                        self.sigma_total[:L_new,0] -= Sq[:, 0]

                    print("converged!", flush=True)
                    break
                
                # --- NEW HYBRID LOGIC ---
                preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
                converged_indices = [i for i in range(nroots) if i not in unconverged_idx]
                
                if not converged_indices:
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


                    
                
                else:
                    # === PHASE 2: At least one root is converged ===
                    # NOW use the robust (but tuned-down) Jacobi-Davidson solver
                    # to prevent the 70-iteration stall for Root 1.
                    
                    Q_conv = full_eigvecs2[converged_indices, :]

                    if _QED_SOLVER_STATS:
                        # How wide could a BLOCKED correction solve be here?  This
                        # branch is guarded by "at least one root converged", and
                        # nroots is 1 or 2, so the answer is expected to be 1 --
                        # i.e. there is nothing to block across.  Counted rather
                        # than assumed.
                        _solver_stats.jd_width(len(unconverged_idx))

                    for i in range(len(unconverged_idx)):
                        idx = unconverged_idx[i]
                        current_residual = w[idx, :]
                        current_theta = theta[idx]
                        P_diag = H_diag - current_theta
                        # ... (handle P_diag safety) ...
                        
                        # Call the inner solver, but make sure its
                        # max_inner_iter is set to a low number, like 3 or 5.
                        v_corr = self.inner_solve_pcg(
                            current_residual,
                            current_theta,
                            Q_conv,
                            P_diag,
                            U, A_tilde, G_blocks, reduced_gradient, alpha # Need to pass these through
                        )
                        
                        preconditioned_w[i, :] = v_corr
                # --- END HYBRID LOGIC ---


                ## --- NEW JACOBI-DAVIDSON LOGIC ---
                #preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
                #converged_indices = [i for i in range(nroots) if i not in unconverged_idx]
                #Q_conv = full_eigvecs2[converged_indices, :] # Converged eigenvectors (N_conv, H_dim)
                #
                #for i in range(len(unconverged_idx)):
                #    idx = unconverged_idx[i] # The root we are solving for
                #    
                #    # Get inputs for the inner solver
                #    current_residual = w[idx, :]
                #    current_theta = theta[idx]
                #    
                #    # This is your *diagonal* preconditioner. We still use it
                #    # to precondition the *inner* solver.
                #    P_diag = H_diag - current_theta
                #    # (Handle divide by zero, e.g., set P_diag[P_diag == 0] = 1.0)
                #    
                #    # Call the inner solver to find the correction vector
                #    v_corr = self.inner_solve_pcg(
                #        current_residual,
                #        current_theta,
                #        Q_conv,
                #        P_diag,
                #        U, A_tilde, G1, reduced_gradient, alpha # Need to pass these through
                #    )
                #    
                #    preconditioned_w[i, :] = v_corr        


                
                
                ## preconditioned_w = np.zeros((len(unconverged_idx),H_dim))
                #if len(unconverged_idx) > 0:
                #    preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
                #    preconditioned_w = theta[unconverged_idx].reshape(
                #        len(unconverged_idx), 1
                #    ) - H_diag.reshape(1, H_dim)
                #    # print(np.shape(preconditioned_w))
                #    preconditioned_w = np.divide(
                #        w[unconverged_idx],
                #        preconditioned_w,
                #        out=np.zeros_like(w[unconverged_idx]),
                #        where=preconditioned_w != 0,
                #    )

                    
               
                end = timer()
                print("residual took", end - start)
                L_old = Q.shape[0]
                start = timer()

                if Lmax - L < len(unconverged_idx):
                    if Lmax > 40:
                        num_restart_vecs = 40
                    else:
                        num_restart_vecs = Lmax//2
                    
                    print(f"Subspace limit reached. Performing soft restart from {L} to {num_restart_vecs} vectors.")
                    t_collapsing_begin = time.time()
                    
                    # 1. Choose how many vectors to keep.
                    # This is a tunable parameter. Let's say 40.

                    # 2. 'aug_eigvecs' holds the (L, L) eigenvectors of the projected Hessian.
                    # 'theta' holds the eigenvalues. You already have these from np.linalg.eigh.
                    
                    # 3. Get the k "best" projected eigenvectors (the first k columns)
                    # Note: np.linalg.eigh already sorts them by eigenvalue.
                    best_projected_vecs = aug_eigvecs[:, :num_restart_vecs] # Shape (L, k)

                    # 4. Transform these k vectors back into the full H_dim space.
                    # Q is (L, H_dim). We need to project it down.
                    # (k, L) @ (L, H_dim) -> (k, H_dim)
                    Q_restarted = np.dot(best_projected_vecs.T, Q) # Shape (k, H_dim)
                    
                    # 5. Add your new correction vectors to this restarted subspace
                    Q = np.concatenate((Q_restarted, preconditioned_w), axis=0) # Shape (k + n_unconv, H_dim)

                    # 6. Re-orthogonalize the new, smaller subspace
                    c_gram_schmidt_orthogonalization(
                       Q, Q.shape[0], H_dim
                    )
                    
                    t_collapsing_end = time.time()
                    print(
                        "restart took", t_collapsing_end - t_collapsing_begin, "seconds"
                    )
                    
                    # 7. IMPORTANT: You MUST tell your code to rebuild the projected Hessian
                    # from this new Q, as it's not just an expansion.
                    # Your existing logic handles this perfectly.
                    self.collapse_subspace_check = True 
                    collapse = True
                    L_old = 0 # Tell the expansion logic the "old" size is 0 

                else:
                    t_expanding_begin = time.time()

                    Q = np.concatenate((Q, preconditioned_w), axis=0)
                    c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                    # gc.collect()
                    t_expanding_end = time.time()
                    print("expand took", t_expanding_end - t_expanding_begin, "seconds")
                    collapse = False
                    print("collapse subspace check", self.collapse_subspace_check)
                    # print(Q)
                end = timer()

        _stats_end(_stats_span_d, status="warm" if restart else "cold")
        return Q

    def minres_solve(self, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6):
        """
        Diagonally-preconditioned MINRES (Paige & Saunders) for the Newton
        equations  H x = -g,  where H is the reduced orbital Hessian applied
        matrix-free through self.mv2 and g is `reduced_gradient`.

        Drop-in replacement for linear_equation_solve: same argument list, same
        (solution, converged) return contract, so the call sites can dispatch
        between the two on _QED_USE_MINRES without any other change.

        Why replace linear_equation_solve here.  That routine drives
        LinearRMSolver, which minimizes ||H c + g||^2 over its stored subspace
        -- i.e. it forms and solves the NORMAL equations S^T S a = -S^T g with
        S = H C.  Two consequences: the effective condition number is
        cond(H)^2, which is exactly the wrong property near the singular
        Hessians this branch is called on; and every iteration rebuilds the
        size x size `mat`/`overlap` matrices from scratch and re-transforms the
        whole basis, i.e. O(k^2 n) work on top of the one mv2.  MINRES gets the
        same minimum-residual property from a three-term Lanczos recurrence:
        one mv2 and O(n) work per iteration, no subspace storage, and it
        operates on cond(H) rather than cond(H)^2.

        H is symmetric but INDEFINITE here (self.reduced_hessian_diagonal
        routinely has negative entries -- that count is what n_negative
        measures), which is why this is MINRES and not CG.

        Preconditioner: M = max(|denom|, 1e-6), elementwise.  MINRES requires
        an SPD preconditioner, so the absolute value is mandatory, not a
        safeguard -- `denom` is self.reduced_hessian_diagonal and is signed.
        This is the same construction solve_pcg_trust_region already uses
        (M_diag = np.abs(M_diag), then np.maximum against epsilon=1e-6).
        Note this differs from how linear_equation_solve used `denom`: it
        divided the residual by the SIGNED diagonal to form the next trial
        vector.

        Deviation from linear_equation_solve, flagged rather than silently
        dropped: that routine, when ||g|| < 1e-3, replaced the initial residual
        with one built from a random trial vector.  There is no analogue here
        and none is needed -- MINRES starts from x0 = 0, for which the residual
        is exactly -g, and a small ||g|| simply means a small solution.  The
        random start also made that routine non-reproducible run to run.
        """
        eps = np.finfo(np.float64).eps

        # --- Preconditioner M (SPD by construction), and psolve = r / M ---
        M_diag_raw = np.abs(np.asarray(denom, dtype=np.float64))
        M_diag = np.maximum(M_diag_raw, 1e-6)
        if _QED_SOLVER_STATS:
            # Diagnostic for "is the Jacobi preconditioner the bottleneck?".
            # The 1e-6 floor is ABSOLUTE, so any entry it catches gets
            # M^-1 = 1e6 -- a huge amplification applied to exactly the
            # near-null directions that make this branch fire in the first
            # place.  n_floored > 0 means the floor, not the diagonal, is
            # setting the preconditioned spectrum; the effective condition
            # number max(M)/min(M) bounds how much Jacobi can help at all.
            _n_floored = int(np.count_nonzero(M_diag_raw < 1e-6))
            print(
                "PMINRES precond: |diag| min %.3e max %.3e  floored %d/%d  "
                "cond(M) %.3e"
                % (M_diag_raw.min(), M_diag_raw.max(), _n_floored,
                   M_diag_raw.size, M_diag.max() / M_diag.min())
            )
        # ||r||_2 <= sqrt(max(M)) * ||r||_{M^-1}, since r^T M^-1 r >= ||r||^2 / max(M).
        # phibar below is exactly ||r_k||_{M^-1}, so this converts the free
        # recurrence quantity into a rigorous upper bound on the true residual
        # norm -- no extra mv2 per iteration to monitor convergence.
        bound_factor = np.sqrt(np.max(M_diag))

        b = -np.asarray(reduced_gradient, dtype=np.float64).ravel()
        n = b.shape[0]
        x = np.zeros(n)

        print("--------------------------------------------")
        print("--- Start solving Hx=-g equations (preconditioned MINRES) ---")
        print("--------------------------------------------")

        # --- Set up the first Lanczos vector ---
        r1 = b.copy()
        y = r1 / M_diag
        beta1 = np.dot(r1, y)
        if beta1 < 0.0:
            # Cannot happen mathematically for SPD M; clamp rather than raise so
            # a roundoff-level sign flip can't kill the whole optimization.
            beta1 = 0.0
        if beta1 == 0.0:
            print("PMINRES: rhs is zero, x = 0 is exact")
            return x, True
        beta1 = np.sqrt(beta1)

        oldb = 0.0
        beta = beta1
        dbar = 0.0
        epsln = 0.0
        phibar = beta1
        cs = -1.0
        sn = 0.0
        w = np.zeros(n)
        w2 = np.zeros(n)
        r2 = r1

        converged = False
        bound_converged = False  # exited on the rigorous ||Hx+g||_2 upper bound
        residual_bound = np.inf
        itn = 0
        for itn in range(1, max_iter + 1):
            # --- Lanczos step: v = M^-1-normalized next basis vector ---
            v = y / beta
            y = self.mv2(U, A_tilde, G_blocks, v, 1, 0, 0).ravel()
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alfa = np.dot(v, y)
            y = y - (alfa / beta) * r2
            r1 = r2
            r2 = y
            y = r2 / M_diag
            oldb = beta
            beta = np.dot(r2, y)
            if beta < 0.0:
                beta = 0.0
            beta = np.sqrt(beta)

            # --- Apply the previous plane rotation Q_{k-1} ---
            oldeps = epsln
            delta = cs * dbar + sn * alfa
            gbar = sn * dbar - cs * alfa
            epsln = sn * beta
            dbar = -cs * beta

            # --- Compute and apply the next plane rotation Q_k ---
            gamma = max(np.sqrt(gbar * gbar + beta * beta), eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar

            # --- Update the solution ---
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) / gamma
            x = x + phi * w

            residual_bound = bound_factor * phibar
            print(f"PMINRES Iter: {itn:3d}   Residual Norm (bound): {residual_bound:.4e}")

            if residual_bound < conv_thresh:
                converged = True
                bound_converged = True
                break
            if beta <= eps * max(1.0, oldb):
                # Krylov space exhausted: x already minimizes the residual over
                # the full invariant subspace, so verify and stop either way.
                print("PMINRES: Lanczos breakdown (invariant subspace reached)")
                break

        if bound_converged:
            # `residual_bound` is a RIGOROUS UPPER BOUND on ||Hx + g||_2 (see
            # bound_factor above: ||r||_2 <= sqrt(max(M)) * ||r||_{M^-1}, and
            # phibar IS ||r||_{M^-1}).  Having broken on it, ||Hx + g|| <
            # conv_thresh is already proved, so evaluating the true residual can
            # only re-confirm what we know -- at the cost of one full mv2.
            # Measured on c10h8: 97 of 97 solves exit here (none reaches the
            # iteration cap), so skipping it is 97 Hessian applications, 3.8% of
            # that run's 2548, for a bit-identical x and `converged` flag.
            # The bound runs a measured 3.3-6.8x above the true residual, so
            # this exit is conservative, never optimistic.
            print(
                f"PMINRES: residual bound {residual_bound:.4e} < {conv_thresh:.1e} "
                f"after {itn} iterations (bound is rigorous; no verification mv2)"
            )
            print("\n--- Convergence Achieved ---")
            print("total iteration", itn)
            return x, True

        # Reached only on the iteration cap or a Lanczos breakdown, where the
        # recurrence bound never certified convergence.  One true residual
        # evaluation (a single extra mv2 for the whole solve, not one per
        # iteration) so the caller's `converged` flag reflects ||Hx + g||, the
        # same quantity linear_equation_solve tested against -- the bound is
        # conservative, so a solve can still be converged in truth after
        # failing to prove it.
        true_residual = np.linalg.norm(
            self.mv2(U, A_tilde, G_blocks, x, 1, 0, 0).ravel() - b
        )
        converged = bool(true_residual < conv_thresh)
        print(f"PMINRES: true residual ||Hx+g|| = {true_residual:.4e} after {itn} iterations")
        if converged:
            print("\n--- Convergence Achieved ---")
            print("total iteration", itn)
        else:
            print("\n--- Solver did not converge within max iterations ---")
        return x, converged


    def linear_equation_solve(self, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6):
        """
        Main driver loop to solve the linear system Ax + b = 0.

        Superseded by minres_solve above (see its docstring for why); kept as
        the _QED_USE_MINRES=0 comparison path and as the TAMM/cpp_casscf port
        reference, whose LinearRMSolver mirrors this routine.
        """
        #self._reset()
        residual = reduced_gradient.copy()
        if np.linalg.norm(residual) < 1e-3:
            #random guess
            dim00 = self.index_map_size
            trial_0 = np.random.rand(dim00)

            norm = np.linalg.norm(trial_0)
            if norm > 1e-12:
                trial_0 /= norm

            sigma = self.mv2(U, A_tilde, G_blocks, trial_0, 1, 0, 0)
            residual = sigma + reduced_gradient

        #print("initial guess", residual)
        #residual = self.apply_preconditioner(A, G1, z)
        #residual /= np.linalg.norm(residual)
        solver_sym = LinearRMSolver(b_vector=reduced_gradient, max_subspace=max_iter)
        #solution_sym = solver_sym.solve(matvec_prod_sym, precond_sym, max_iter=100, conv_thresh=1e-7)
        #print(denom[:self.index_map_size])
        print("--------------------------------------------")
        print("--- Start solving Hx=-g equations ---")
        print("--------------------------------------------")
        for i in range(max_iter):
            #sigma0 = matvec_product(residual, A, G1)
            #residual0 = sigma0 + self.total_gradient
            #print("zzz", np.linalg.norm(residual0))
            residual_norm = np.linalg.norm(residual)
            print(f"Iter: {i+1:3d}   Residual Norm: {residual_norm:.4e}")

            if residual_norm < conv_thresh:
                print("\n--- Convergence Achieved ---")
                print("total iteration", i+1)
                return solver_sym.get_solution(), True

            trial_c = residual/denom
            
            #print("trial_c", trial_c)
            norm = np.linalg.norm(trial_c)
            if norm > 1e-12:
                trial_c /= norm

            sigma = self.mv2(U, A_tilde, G_blocks, trial_c, 1, 0, 0)
            #print("sigma", sigma)
            residual_old =copy.deepcopy(residual)
            residual = solver_sym.update_subspace_and_extrapolate(trial_c, sigma)
            residual_new = copy.deepcopy(residual)
            error = residual_new - residual_old
            if i > 0 and np.linalg.norm(error) < 1e-8:
                print("\n--- Convergence Achieved (solution becomes self-consistent)---")
                print("total iteration", i+1)
                return solver_sym.get_solution(), True


        print("\n--- Solver did not converge within max iterations ---")
        return solver_sym.get_solution(), False


    def solve_pcg_trust_region(self, U, A_tilde, G_blocks, g, M_diag, trust_radius, tol=1e-8, max_iter=1000):
        """
        Solves the trust-region subproblem using Preconditioned Conjugate Gradient (Steihaug-CG).
    
        This solver finds a step 'p' that minimizes the quadratic model:
            m(p) = g^T p + 0.5 * p^T H p
        subject to:
            ||p|| <= trust_radius
    
        It is "matrix-free" and handles the key cases:
        1. Finds the Newton step if H is positive definite and the step is inside the trust region.
        2. Stops if H is found to be non-positive-definite (negative curvature).
        3. Stops if the step exceeds the trust radius.
    
        Arguments:
        Hess_vec_product (callable): A function that takes a vector 'v' 
                                     and returns the Hessian-vector product H*v.
        g (np.ndarray):              The current gradient vector.
        M_diag (np.ndarray):         A vector of the diagonal entries of the preconditioner M.
        trust_radius (float):        The current trust radius (Delta).
        tol (float):                 Tolerance for convergence (based on residual norm).
        max_iter (int):              Maximum number of CG iterations.
    
        Returns:
        (np.ndarray, str): A tuple containing the computed step 'p' and a string 
                           indicating the reason for termination.
        """
        M_diag = np.abs(M_diag)
        # Prevent division by zero (or extremely large steps) for near-zero diagonals
        epsilon = 1e-6
        M_diag = np.maximum(M_diag, epsilon)
        # --- 1. Initialization ---
        p = np.zeros_like(g) # Step (starts at 0)
        r = -g               # Residual (r = -g - H*p, but p=0)
        
        # --- This is where the diagonal preconditioner is used ---
        # z = M_inv * r. For a diagonal M, M_inv * r is just r / M_diag
        z = r / M_diag
        
        d = z.copy()         # Search direction
        r_dot_z = np.dot(r, z)
        print(r)
        print(z)
        print("r_dot_z", r_dot_z) 
        # Check for a trivial solution (gradient is already zero)
        if np.sqrt(r_dot_z) < tol:
            return p, "success_gradient_zero"
    
        # --- 2. PCG Iteration Loop ---
        for j in range(max_iter):
            
            # --- This is the main matrix-vector product ---
            
            Hd = self.mv2(U, A_tilde, G_blocks, d, 1, 0, 0)
            # --- This is the Positive-Definite (Curvature) Check ---
            d_H_d = np.dot(d, Hd)
            print(d_H_d)
            if d_H_d <= 0:
                # The Hessian is not positive definite (negative or zero curvature).
                # We must find the step 'p' that extends to the trust boundary
                # along the current direction 'd'.
                # We need to find tau > 0 such that ||p + tau*d|| = trust_radius
                
                a = np.dot(d, d)
                b = 2 * np.dot(p, d)
                c = np.dot(p, p) - trust_radius**2
                
                # Solve a*tau^2 + b*tau + c = 0 for the positive root tau
                discriminant = b**2 - 4*a*c
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                return p + tau * d, "stop_negative_curvature"
            
            # --- Standard PCG Step ---
            alpha = r_dot_z / d_H_d
            p_new = p + alpha * d
            print("norm of p", np.linalg.norm(p_new)) 
            # --- This is the Trust-Radius Boundary Check ---
            if np.linalg.norm(p_new) > trust_radius:
                # The step has crossed the boundary. We must find the
                # step 'p' that stops exactly *on* the boundary.
                # We use the same quadratic formula as in the negative curvature case.
                
                a = np.dot(d, d)
                b = 2 * np.dot(p, d)
                c = np.dot(p, p) - trust_radius**2
                
                discriminant = b**2 - 4*a*c
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                return p + tau * d, "stop_trust_boundary"
            
            # --- Update for next iteration (no problems found) ---
            p = p_new
            r = r - alpha * Hd
            
            # --- Check for Convergence (Newton step found) ---
            print(j, np.linalg.norm(r))
            if np.linalg.norm(r) < tol:
                return p, "success_solution_found" # This is the internal Newton step
                
            # --- Apply preconditioner again ---
            z = r / M_diag
            
            r_dot_z_new = np.dot(r, z)
            
            # Beta calculation (Fletcher-Reeves)
            beta = r_dot_z_new / r_dot_z
            
            # Update search direction
            d = z + beta * d
            r_dot_z = r_dot_z_new
    
        # If the loop finishes, we ran out of iterations
        return p, "stop_max_iterations"   


    def solve_pcg_with_operator(self, B_op, g, M_diag, trust_radius, tol=1e-8, max_iter=1000):
        """
        Solves the trust-region subproblem using Preconditioned Conjugate Gradient (Steihaug-CG).
    
        This solver finds a step 'p' that minimizes the quadratic model:
            m(p) = g^T p + 0.5 * p^T H p
        subject to:
            ||p|| <= trust_radius
    
        It is "matrix-free" and handles the key cases:
        1. Finds the Newton step if H is positive definite and the step is inside the trust region.
        2. Stops if H is found to be non-positive-definite (negative curvature).
        3. Stops if the step exceeds the trust radius.
    
        Arguments:
        Hess_vec_product (callable): A function that takes a vector 'v' 
                                     and returns the Hessian-vector product H*v.
        g (np.ndarray):              The current gradient vector.
        M_diag (np.ndarray):         A vector of the diagonal entries of the preconditioner M.
        trust_radius (float):        The current trust radius (Delta).
        tol (float):                 Tolerance for convergence (based on residual norm).
        max_iter (int):              Maximum number of CG iterations.
    
        Returns:
        (np.ndarray, str): A tuple containing the computed step 'p' and a string 
                           indicating the reason for termination.
        """
        M_diag = np.abs(M_diag)
        # Prevent division by zero (or extremely large steps) for near-zero diagonals
        epsilon = 1e-6
        M_diag = np.maximum(M_diag, epsilon)
        # --- 1. Initialization ---
        p = np.zeros_like(g) # Step (starts at 0)
        r = -g               # Residual (r = -g - H*p, but p=0)
        
        # --- This is where the diagonal preconditioner is used ---
        # z = M_inv * r. For a diagonal M, M_inv * r is just r / M_diag
        z = r / M_diag
        
        d = z.copy()         # Search direction
        r_dot_z = np.dot(r, z)
        #print(r)
        #print(z)
        print("r_dot_z", r_dot_z) 
        # Check for a trivial solution (gradient is already zero)
        if np.sqrt(r_dot_z) < tol:
            return p, "success_gradient_zero"
    
        # --- 2. PCG Iteration Loop ---
        for j in range(max_iter):
            
            # --- This is the main matrix-vector product ---
            
            Hd = B_op(d) 
            # --- This is the Positive-Definite (Curvature) Check ---
            d_H_d = np.dot(d, Hd)
            print(d_H_d)
            if d_H_d <= 0:
                # The Hessian is not positive definite (negative or zero curvature).
                # We must find the step 'p' that extends to the trust boundary
                # along the current direction 'd'.
                # We need to find tau > 0 such that ||p + tau*d|| = trust_radius
                
                a = np.dot(d, d)
                b = 2 * np.dot(p, d)
                c = np.dot(p, p) - trust_radius**2
                
                # Solve a*tau^2 + b*tau + c = 0 for the positive root tau
                discriminant = b**2 - 4*a*c
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                return p + tau * d, "stop_negative_curvature"
            
            # --- Standard PCG Step ---
            alpha = r_dot_z / d_H_d
            p_new = p + alpha * d
            print("norm of p", np.linalg.norm(p_new)) 
            # --- This is the Trust-Radius Boundary Check ---
            if np.linalg.norm(p_new) > trust_radius:
                # The step has crossed the boundary. We must find the
                # step 'p' that stops exactly *on* the boundary.
                # We use the same quadratic formula as in the negative curvature case.
                
                a = np.dot(d, d)
                b = 2 * np.dot(p, d)
                c = np.dot(p, p) - trust_radius**2
                
                discriminant = b**2 - 4*a*c
                tau = (-b + np.sqrt(discriminant)) / (2 * a)
                
                return p + tau * d, "stop_trust_boundary"
            
            # --- Update for next iteration (no problems found) ---
            p = p_new
            r = r - alpha * Hd
            
            # --- Check for Convergence (Newton step found) ---
            print(j, np.linalg.norm(r))
            if np.linalg.norm(r) < tol:
                return p, "success_solution_found" # This is the internal Newton step
                
            # --- Apply preconditioner again ---
            z = r / M_diag
            
            r_dot_z_new = np.dot(r, z)
            
            # Beta calculation (Fletcher-Reeves)
            beta = r_dot_z_new / r_dot_z
            
            # Update search direction
            d = z + beta * d
            r_dot_z = r_dot_z_new
    
        # If the loop finishes, we ran out of iterations
        return p, "stop_max_iterations"   


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

    def build_orbital_hessian_guess(
        self,
        U,
        sym_A_tilde,
        reduced_gradient,
        G,
        guess_hessian,
        guess_gradient,
        dim1,
        idx,
    ):
        index_map = self.index_map
        nmo = self.nmo
        n_occupied = self.n_occupied
        return self.hessian_guess(
            U,
            sym_A_tilde,
            reduced_gradient,
            G,
            guess_hessian,
            guess_gradient,
            dim1,
            nmo,
            n_occupied,
            index_map,
            idx,
        )

    @staticmethod
    # @nb.njit("""void(float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,:,:,::1], float64[:,::1], float64[:,::1],
    #        int64, int64, int64, int64, int64)""", fastmath = True, parallel = True)
    # def build_sigma_reduced6(U, A_tilde, index_map, G1, G, R_reduced, sigma_reduced, num_states, pointer, nmo, index_map_size, n_occupied):
    @nb.njit(
        """void(float64[:,::1], float64[:,::1], float64[::1], float64[:,:,:,::1], float64[:,::1], float64[::1],
            int64, int64, int64, int64[:,::1], int64[::1])""",
        fastmath=True,
        parallel=True,
    )
    def hessian_guess(
        U,
        sym_A_tilde,
        reduced_gradient,
        G,
        guess_hessian,
        guess_gradient,
        dim1,
        nmo,
        n_occupied,
        index_map,
        idx,
    ):
        for i in nb.prange(dim1):
            index1 = idx[i]
            r = index_map[index1][0]
            k = index_map[index1][1]
            guess_gradient[i] = reduced_gradient[index1]
            for j in range(dim1):
                index2 = idx[j]
                s = index_map[index2][0]
                l = index_map[index2][1]
                # print(r,k,s,l)
                # b = self.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
                temp1 = np.copy(G[k, l, :, :])
                temp2 = np.copy(U[:, s])
                temp3 = np.copy(U[:, r])
                temp4 = np.zeros((nmo))
                # temp4 = np.einsum("pq, q ->p", temp1, temp2)
                temp4 = np.dot(temp1, temp2)
                # a = np.einsum("p, p ->", temp3, temp4)
                a = np.dot(temp3, temp4)
                a -= 0.5 * (k == l) * (sym_A_tilde[r][s])
                a -= 0.5 * (r == s) * (sym_A_tilde[k][l])
                # print(np.shape(temp1))
                if r < n_occupied:
                    temp1 = np.copy(G[r, l, :, :])
                    temp3 = np.copy(U[:, k])
                    # temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    # a -= np.einsum("p, p ->", temp3, temp4)
                    a -= np.dot(temp3, temp4)
                    a += 0.5 * (r == l) * (sym_A_tilde[k][s])
                if s < n_occupied:
                    temp1 = np.copy(G[k, s, :, :])
                    temp2 = np.copy(U[:, l])
                    temp3 = np.copy(U[:, r])
                    # temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    # a -= np.einsum("p, p ->", temp3, temp4)
                    a -= np.dot(temp3, temp4)
                    a += 0.5 * (k == s) * (sym_A_tilde[r][l])
                if r < n_occupied and s < n_occupied:
                    temp1 = np.copy(G[r, s, :, :])
                    temp2 = np.copy(U[:, l])
                    temp3 = np.copy(U[:, k])
                    # temp4 = np.einsum("pq, q ->p", temp1, temp2)
                    temp4 = np.dot(temp1, temp2)
                    # a += np.einsum("p, p ->", temp3, temp4)
                    a += np.dot(temp3, temp4)
                guess_hessian[i][j] = a
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

    def projection_step2(
        self, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0
    ):
        if _QED_SOLVER_STATS:
            _solver_stats.bordered_solve()
        augmented_hessian = np.zeros((dim0, dim0))
        # print("projected_gradient", gradient_tilde, flush = True)

        augmented_hessian[0, 0] = alpha
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
        print(mu, flush=True)
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
        _mv_t0 = timer() if _QED_SOLVER_STATS else None
        R_reduced = R_reduced.reshape((1, self.index_map_size)).astype(np.float64)
        nmo = self.nmo
        index_map = self.index_map
        index_map_size = self.index_map_size
        n_occupied = self.n_occupied
        n_inactive = self.n_in_a
        n_active = self.n_act_orb
        G_ij, G_ti, G_tu = G_blocks
        G_ij = np.ascontiguousarray(G_ij)
        G_ti = np.ascontiguousarray(G_ti)
        G_tu = np.ascontiguousarray(G_tu)
        # print("U dtype:", U.dtype, "shape:", U.shape)
        # print("A_tilde dtype:", A_tilde.dtype, "shape:", A_tilde.shape)
        # print("index_map dtype:", index_map.dtype, "shape:", index_map.shape)
        # print("G dtype:", G.dtype, "shape:", G.shape)
        # print("R_reduced dtype:", R_reduced.dtype, "shape:", R_reduced.shape)
        _ret = self.build_sigma_reduced8(
            U,
            A_tilde,
            index_map,
            G_ij,
            G_ti,
            G_tu,
            R_reduced,
            num_states,
            pointer,
            nmo,
            index_map_size,
            n_occupied,
            n_inactive,
            n_active,
            eigval,
            _QED_G_SYMV,
        )
        if _QED_SOLVER_STATS:
            _solver_stats.matvec(1, timer() - _mv_t0)
        return _ret

    @staticmethod
    @nb.jit(
        """float64[::1](float64[:,::1], float64[:,::1], int64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1], float64[:,::1],
            int64, int64, int64, int64, int64, int64, int64, float64, int64)""",
        nopython=True,
        cache=True,   # see build_sigma_reduced7 / _symv_accumulate for why
        fastmath=True,
        parallel=False,
    )
    def build_sigma_reduced8(
        U,
        A_tilde,
        index_map,
        G_ij,
        G_ti,
        G_tu,
        R_reduced,
        num_states,
        pointer,
        nmo,
        index_map_size,
        n_occupied,
        n_inactive,
        n_active,
        eigval,
        use_symv,
    ):
        assert U.shape == (nmo, nmo)
        assert A_tilde.shape == (nmo, nmo)
        sigma_reduced = np.zeros((1, index_map_size))
        R_total = np.zeros((num_states, nmo, n_occupied))
        #print("oivdpw", num_states)
        for i in range(num_states):
            for j in range(index_map_size):
                r = index_map[j][0]
                k = index_map[j][1]
                R_total[i][r][k] = R_reduced[i][j + pointer]

        sigma_total = np.zeros((num_states, nmo, n_occupied))

        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(U, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :].transpose(2, 0, 1))
        # R_total1 = np.ascontiguousarray(R_total1)
        R1 = np.reshape(R1, (n_occupied, num_states * n_occupied))
        U1 = np.ascontiguousarray(U[:, :n_occupied])
        temp2 = np.dot(U1, R1)
        temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)

        ##sigma_total1 = np.zeros((num_states, nmo, n_occupied))
        ##temp11 = np.einsum("qs,isl->iql", U, R_total)
        # for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for l in range(n_occupied):
        #            #a = 0.0
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += U[q,s] * R_total[i,s,l]
        #                #temp1[i,q,l] += U[q,s] * R_total[i,s,l]
        #            temp1[i,q,l] = a
        #

        # for q in nb.prange(nmo):
        #    for i in range(num_states):
        #        for s in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a -= U[q,l] * R_total[i,s,l]
        #                #temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
        #            temp1[i,q,s] += a
        # The flat R1/R2 staging that used to sit here fed the single dense
        # np.dot(R1, G1); the blocked product below consumes temp1 directly, so
        # it is gone (build_sigma_reduced7 has no equivalent either).
        ##for i in nb.prange(num_states):
        ##    for r in range(nmo):
        ##        for k in range(n_occupied):
        ##            a = R1[i][r*n_occupied+k] - temp1[i,r,k]
        ##            if np.abs(a) > 1e-14: print("LARGE error")

        # Blocked replacement for the former  temp1 = np.dot(R1, G1)  over the
        # full (nmo*n_occupied) square G1.  Identical to the block section of
        # build_sigma_reduced7 (helper_PFCI.py:8760-8775): the G blocks are
        # flattened with index (r,k) -> r*n_block_occ + k, and the (i,t)
        # quadrant is recovered as G_ti.T by the Hessian's pair-exchange
        # symmetry.  temp1 here is (num_states, nmo, n_occupied).
        R_i_tensor = temp1[:, :, :n_inactive]
        R_a_tensor = temp1[:, :, n_inactive:]
        R1_i = R_i_tensor.copy().reshape(num_states, nmo * n_inactive)
        R1_a = R_a_tensor.copy().reshape(num_states, nmo * n_active)

        # QED_G_SYMV, as in build_sigma_reduced7 -- see the note there and the
        # module-level block.  mv2 always passes num_states == 1, so the gate is
        # always taken when the knob is on; it is written the same way as the
        # twin so the two stay comparable.
        if use_symv == 1 and num_states == 1:
            sigma_i = np.dot(R1_a, G_ti)
            _symv_accumulate(G_ij, R1_i[0], sigma_i[0])
        else:
            sigma_i = np.dot(R1_i, G_ij) + np.dot(R1_a, G_ti)
        sigma_a = np.dot(R1_i, G_ti.T) + np.dot(R1_a, G_tu)

        temp2 = np.zeros((num_states, nmo, n_occupied))
        temp2[:, :, :n_inactive] = sigma_i.reshape(num_states, nmo, n_inactive)
        temp2[:, :, n_inactive:] = sigma_a.reshape(num_states, nmo, n_active)
        # for r in nb.prange(nmo):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for p in range(nmo):
        #                a += U[p,r] * temp2[i,p,k]
        #                #sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
        #            temp1[i,r,k] = a
        #            sigma_total[i,r,k] = a
        #

        # for r in nb.prange(n_occupied):
        #    for i in range(num_states):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] -= temp1[i,k,r]

        #######for i in nb.prange(num_states):
        #######    R3 = np.dot(R1[i,:], G)
        #######    R2[i,:] = R3

        temp2 = temp2.transpose(0, 2, 1)
        temp2 = np.ascontiguousarray(temp2)
        temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
        temp1 = np.dot(temp2, U)
        temp2 = temp1.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] = temp2.transpose(0, 2, 1)[:, :, :]
        sigma_total[:, :n_occupied, :] -= temp2[:, :, :n_occupied]

        A3_tilde = A_tilde + A_tilde.T
        # temp1 = np.zeros((num_states, nmo, n_occupied))
        # temp2 = np.zeros((num_states, nmo, n_occupied))

        ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(self.n_occupied), A3_tilde)
        ##hessian_tilde[:,:,:self.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(self.n_occupied), A3_tilde[:self.n_occupied,:])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for s in range(nmo):
        #                a += A3_tilde[r,s] * R_total[i,s,k]
        #            temp1[i,r,k] = a
        #            sigma_total[i,r,k] -= 0.5 * a
        #
        # for i in nb.prange(num_states):
        #    for r in range(n_occupied):
        #        for k in range(n_occupied):
        #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r]
        #
        ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(self.nmo), A3_tilde[:self.n_occupied,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[k,l] * R_total[i,r,l]
        #            sigma_total[i,r,k] -= 0.5 * a

        ##hessian_tilde[:,:,:,:self.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(self.n_occupied), A3_tilde[:,:self.n_occupied])
        # for i in nb.prange(num_states):
        #    for r in range(nmo):
        #        for k in range(n_occupied):
        #            a = np.float64(0)
        #            for l in range(n_occupied):
        #                a += A3_tilde[r,l] * R_total[i,k,l]
        #            sigma_total[i,r,k] += 0.5 * a
        R1 = R_total.transpose(1, 0, 2)
        R1 = np.ascontiguousarray(R1)
        R1 = np.reshape(R1, (nmo, num_states * n_occupied))
        temp2 = np.dot(A3_tilde, R1)
        temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
        sigma_total[:, :, :] -= 0.5 * temp1[:, :, :]
        sigma_total[:, :n_occupied, :] += (
            0.5 * temp1.transpose(0, 2, 1)[:, :, :n_occupied]
        )
        A3 = A3_tilde[:n_occupied, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total)
        R1 = np.reshape(R1, (num_states * nmo, n_occupied))
        temp2 = np.dot(R1, A3)
        sigma_total[:, :, :] -= (
            0.5 * temp2.reshape(num_states, nmo, n_occupied)[:, :, :]
        )
        A3 = A3_tilde[:, :n_occupied].T
        A3 = np.ascontiguousarray(A3)
        R1 = np.ascontiguousarray(R_total[:, :n_occupied, :])
        R1 = np.reshape(R1, (num_states * n_occupied, n_occupied))
        temp2 = np.dot(R1, A3)
        temp1 = temp2.reshape(num_states, n_occupied, nmo)
        sigma_total[:, :, :] += 0.5 * temp1.transpose(0, 2, 1)[:, :, :]

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
                sigma_reduced[i][j + pointer] = sigma_total[i][r][k]
                # sigma_reduced[i][j+pointer] -= eigval * R_reduced[i][j+pointer]
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
            "solve_pcg_trust_region",
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
