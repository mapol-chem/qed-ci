"""The orbital Hessian-vector product: kernels, BLAS binding, and wrappers.

THE HOT PATH.  Everything the trust-region solvers spend their time in is here.
One SA-QED-CASSCF driver run enters `orbital_sigma3` 685-837 times and `mv2`
93-228 times; `build_sigma_reduced7` under them is the single most executed
piece of numerics in the package.

A LEAF MODULE, ON PURPOSE.  Nothing here imports helper_PFCI at module scope,
and nothing on the default path imports it at all -- so the dependency runs one
way (helper_PFCI -> orbital_sigma) and the sigma kernel knows nothing about the
generator.  The `ctx` the wrappers take is duck-typed: they read
`ctx.nmo` / `ctx.index_map` / `ctx.index_map_size` / `ctx.n_occupied` /
`ctx.n_in_a` / `ctx.n_act_orb` and call `ctx._actvirt_shift_vector()`, nothing
more.  `_actvirt_shift_vector` stays a method on the generator because
`build_hessian_diagonal`, which is not part of this module, also calls it.

WHY THE dsymv BINDING LIVES HERE TOO.  `_symv_accumulate` is this kernel's
inner product; `build_sigma_reduced7`/`_8` are `nopython` and so must resolve it
as a compile-time global, which a deferred import cannot provide.  It moved with
them.  helper_PFCI imports it back for `build_gradient`, its one other caller,
and `_QED_G_SYMV` is defined here as well because the no-MKL fallback below has
to be able to force it off.  The knob's documentation stays in helper_PFCI's
knob block, where a reader looks for knobs; read the two together.

`_QED_SOLVER_STATS` is re-read from the environment here rather than imported,
so the default path touches helper_PFCI zero times.  The `_solver_stats`
ACCUMULATOR is genuinely shared, so that one is fetched through `_stats()` --
but only when the knob is on, which it is not by default.

MOVED VERBATIM (2026-09-04).  `self` became `ctx`, the `@staticmethod`
decorators went away with the class, and calls between these functions became
module-local; the arithmetic and the numba signatures are byte-for-byte
unchanged.  The move is gated bit-exactly by validation/lstrs_refactor_gate.sh.
Expect ONE cold recompile per jitted function afterwards: numba's on-disk cache
lives beside the source file, so relocating the source invalidates it once.
That is a startup cost, not a numerical change -- the gate proves the numbers.

WHAT IS STILL DUPLICATED, AND DELIBERATELY LEFT SO.  `build_sigma_reduced8` is
`build_sigma_reduced7` plus three things: it allocates its own output instead of
filling a caller's buffer, it subtracts `eigval * R_reduced` at the end, and it
returns flattened.  84 of their ~90 code lines are identical.  Collapsing them
into one kernel is a real simplification of the hottest code in the package, but
it is a BEHAVIOUR-PRESERVING PERFORMANCE change, not a move, and it belongs in
its own commit with its own before/after timing -- not smuggled into a
relocation whose whole value is being provably inert.
"""

import os
import sys

import numpy as np
import numba as nb

_QED_SOLVER_STATS = int(os.environ.get("QED_SOLVER_STATS", "0"))

# Use MKL's dsymv for the symmetric G products instead of a full dgemv.
# DEFAULT 1.  The value lives here rather than in helper_PFCI's knob block
# because the no-MKL fallback further down has to be able to force it off, and a
# knob whose value can be overridden from two files is a knob nobody can reason
# about.  The RATIONALE and the measurements stay in that knob block; this is
# only where the number is decided.  It is a pure RUNTIME flag -- it is passed
# to the kernels as `use_symv`, never baked into a compiled artifact.
_QED_G_SYMV = int(os.environ.get("QED_G_SYMV", "1"))


def _stats():
    """The shared solver-stats accumulator, fetched only when the knob is on.

    Deferred so that importing this module never pulls in helper_PFCI (and
    through it psi4 and the ctypes backend).  Callers guard every use with
    `if _QED_SOLVER_STATS`, which is 0 by default, so on the hot path this is
    never reached.
    """
    from helper_PFCI import _solver_stats
    return _solver_stats


from timeit import default_timer as timer

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
        # BLAS requires lda >= max(1, n).  An EMPTY block gives n = lda = 0,
        # which MKL rejects outright: "Parameter 5 was incorrect on entry to
        # DSYMV" (parameter 5 IS lda).  A 0-sized block contributes nothing, so
        # returning here is exactly what the call means -- the numbers are
        # unaffected, only the error spam goes away.
        # HIT WHENEVER THERE ARE NO CLOSED ORBITALS, since G_ij is then 0x0:
        # e.g. CAS(4,4) on H4 (4 electrons, 4 active, restricted_docc [0]),
        # which emitted 294,520 of these in one run.  The dgemv fallback below
        # never had the bug because numpy handles 0-sized operands.
        if A.shape[0] == 0:
            return
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


def orbital_sigma3(
    ctx, U, A_tilde, G_blocks, R_reduced, sigma_reduced, num_states, pointer
):
    _mv_t0 = timer() if _QED_SOLVER_STATS else None
    nmo = ctx.nmo
    index_map = ctx.index_map
    index_map_size = ctx.index_map_size
    n_occupied = ctx.n_occupied
    n_inactive = ctx.n_in_a
    n_active = ctx.n_act_orb
    G_ij, G_ti, G_tu = G_blocks
    G_ij = np.ascontiguousarray(G_ij)
    G_ti = np.ascontiguousarray(G_ti)
    G_tu = np.ascontiguousarray(G_tu)
    # build_sigma_reduced7 is jitted with an explicit float64[:,::1] signature,
    # so R_reduced must be C-contiguous.  The un-jitted build_sigma_reduced4
    # this path replaced accepted any layout; microiteration_predicted_energy2
    # can hand us a strided step.  (sigma_reduced is an output -- do NOT copy.)
    R_reduced = np.ascontiguousarray(R_reduced)

    _ret = build_sigma_reduced7(
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
    # Occupation-scaled level shift.  Applied HERE because this is the one
    # chokepoint every solver's Hessian-vector product goes through, so GLTR,
    # LSTRS, PCG, AH-Krylov and the QN sites all see the SAME shifted
    # operator without ~20 individual call sites needing to know about it.
    # build_sigma_reduced7 indexes BOTH R_reduced and sigma_reduced as
    # [i][j + pointer], so the same offset applies to the correction.
    _shift = ctx._actvirt_shift_vector()
    if _shift is not None:
        _sl = slice(pointer, pointer + index_map_size)
        sigma_reduced[:num_states, _sl] += _shift * R_reduced[:num_states, _sl]
    if _QED_SOLVER_STATS:
        # builds num_states Hessian-vector products in one call; the timed
        # region includes the ascontiguousarray prep above, which is part of
        # what a caller pays to get one matvec.
        _stats().matvec(num_states, timer() - _mv_t0)
    return _ret


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

    ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(ctx.n_occupied), A3_tilde)
    ##hessian_tilde[:,:,:ctx.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(ctx.n_occupied), A3_tilde[:ctx.n_occupied,:])
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
    ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(ctx.nmo), A3_tilde[:ctx.n_occupied,:ctx.n_occupied])
    # for i in nb.prange(num_states):
    #    for r in range(nmo):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for l in range(n_occupied):
    #                a += A3_tilde[k,l] * R_total[i,r,l]
    #            sigma_total[i,r,k] -= 0.5 * a

    ##hessian_tilde[:,:,:,:ctx.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(ctx.n_occupied), A3_tilde[:,:ctx.n_occupied])
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


def build_orbital_hessian_guess(
    ctx,
    U,
    sym_A_tilde,
    reduced_gradient,
    G,
    guess_hessian,
    guess_gradient,
    dim1,
    idx,
):
    index_map = ctx.index_map
    nmo = ctx.nmo
    n_occupied = ctx.n_occupied
    return hessian_guess(
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
            # b = ctx.build_orbital_hessian_element(U, sym_A_tilde, G, r,k,s,l)
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


def mv2(ctx, U, A_tilde, G_blocks, R_reduced, num_states, pointer, eigval):
    _mv_t0 = timer() if _QED_SOLVER_STATS else None
    R_reduced = R_reduced.reshape((1, ctx.index_map_size)).astype(np.float64)
    nmo = ctx.nmo
    index_map = ctx.index_map
    index_map_size = ctx.index_map_size
    n_occupied = ctx.n_occupied
    n_inactive = ctx.n_in_a
    n_active = ctx.n_act_orb
    G_ij, G_ti, G_tu = G_blocks
    G_ij = np.ascontiguousarray(G_ij)
    G_ti = np.ascontiguousarray(G_ti)
    G_tu = np.ascontiguousarray(G_tu)
    # print("U dtype:", U.dtype, "shape:", U.shape)
    # print("A_tilde dtype:", A_tilde.dtype, "shape:", A_tilde.shape)
    # print("index_map dtype:", index_map.dtype, "shape:", index_map.shape)
    # print("G dtype:", G.dtype, "shape:", G.shape)
    # print("R_reduced dtype:", R_reduced.dtype, "shape:", R_reduced.shape)
    _ret = build_sigma_reduced8(
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
    # Same occupation-scaled level shift orbital_sigma3 applies, so the
    # LinearOperator sites (MINRES / the shifted solves at :12274, :12325,
    # :14480, :14528, :15858) see the same operator as the trust-region
    # solvers.  build_sigma_reduced8 returns (H - eigval*I) v flattened; the
    # shift belongs to the H part, hence it is added, not scaled by eigval.
    # num_states == 1 is hard-coded by this method's own reshape above.
    _shift = ctx._actvirt_shift_vector()
    if _shift is not None:
        _sl = slice(pointer, pointer + index_map_size)
        _ret[_sl] += _shift * R_reduced[0, _sl]
    if _QED_SOLVER_STATS:
        _stats().matvec(1, timer() - _mv_t0)
    return _ret


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

    ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(ctx.n_occupied), A3_tilde)
    ##hessian_tilde[:,:,:ctx.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(ctx.n_occupied), A3_tilde[:ctx.n_occupied,:])
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
    ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(ctx.nmo), A3_tilde[:ctx.n_occupied,:ctx.n_occupied])
    # for i in nb.prange(num_states):
    #    for r in range(nmo):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for l in range(n_occupied):
    #                a += A3_tilde[k,l] * R_total[i,r,l]
    #            sigma_total[i,r,k] -= 0.5 * a

    ##hessian_tilde[:,:,:,:ctx.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(ctx.n_occupied), A3_tilde[:,:ctx.n_occupied])
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
