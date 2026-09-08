"""Trust-region subproblem solvers and their Krylov machinery.

Fourth and last file in the split that produced `otr_trust_region.py`,
`lstrs_trust_region.py` and `davidson_augmented_hessian.py`.  Those hold one
solver family each; this holds the rest of them, so that helper_PFCI.py can be
the SKELETON of the CI and CASSCF optimization rather than also being the
library of subproblem solvers it dispatches to.

WHAT IS HERE, and why these belong together -- every one of them answers the
same question, "given a gradient and a way to multiply by the Hessian, what step
should we take?", and differs only in how:

    solve_gltr_trust_region / solve_gltr_with_operator
        GLTR: Lanczos on the trust-region subproblem.  The two are the same
        algorithm differing only in how they obtain Hv -- see the README note
        in the C++ port, which collapses both into one class.
    _ah_krylov_trust_region_step
        Krylov-seeded bordered augmented Hessian (QED_AH_KRYLOV).
    _otr_trust_region_step
        OpenTrustRegion's step, delegating to otr_trust_region.py (QED_OTR_TR).
    minres_solve / scipy_minres_solve / linear_equation_solve
        The three Newton-step solvers behind the "gradient is small" fallback,
        selected by QED_USE_MINRES / QED_NEWTON_SCIPY.
    inner_solve_pcg
        The Jacobi-Davidson correction the AH Davidson uses once a root has
        converged.
    projection_step2, aug_matvec, build_hessian_diagonal
        Shared pieces: the bordered eigensolve, the bordered matvec, and the
        preconditioner every solver above uses.

`ctx` IS THE GENERATOR, not a closure bundle.  These read a good deal of state
off it (`ctx.U2`, `ctx.index_map`, `ctx.reduced_hessian_diagonal`, ...) and call
back into `ctx.orbital_sigma3` / `ctx.mv2` for the Hessian-vector product, so
the `hess_x`-closure interface `otr_trust_region.py` uses would be a redesign
rather than a move.  `lstrs_trust_region.py` and
`davidson_augmented_hessian.py` take `ctx` for the same reason.

MOVED VERBATIM (2026-09-04).  `self` became `ctx` -- including the bare
`getattr(self, "_gltr_cache", None)` in the GLTR reuse path -- calls between
these functions became module-local, and each function grew the function-local
`from helper_PFCI import ...` that `lstrs_trust_region.py` already uses to dodge
the circular import.  The arithmetic is byte-for-byte unchanged, and every
method here survives on `PFHamiltonianGenerator` as a delegator, so the ~30 call
sites and the C++ port's line citations still resolve.  Gated bit-exactly by
`validation/lstrs_refactor_gate.sh`.

KNOWN, PRE-EXISTING, NOT FIXED HERE: `solve_gltr_trust_region` injects unseeded
`np.random.randn` noise with no knob to disable it.  That is a real
reproducibility wart, but fixing it would change results, and this file's whole
claim is that it changed none.  Fix it separately.
"""

import copy
import os
from collections import deque

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, minres

from residual_minimization import LinearRMSolver


def _otr_trust_region_step(ctx, hess_x, h_diag, reduced_gradient, trust_radius,
                           label="OTR"):
    """Trust-region step from otr_trust_region.py.  Live under QED_OTR_TR=1
    (main site) or QED_OTR_TR_QN=1 (the two quasi-Newton sites).

    A drop-in for the solver dispatch in microiteration_optimization6: same
    inputs (gradient, Hessian diagonal, trust radius, and the same
    orbital_sigma3 matvec), same output (the step in reduced-rotation
    coordinates), so everything downstream -- the unitary build, the energy
    check, the trust-radius update -- is unchanged.

    The Hessian-vector products go through orbital_sigma3 exactly as the
    original solvers do, including the QED_SOLVER_STATS instrumentation, so
    the matvec counts stay directly comparable between the two paths.
    """
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _OtrSettings, _QED_OTR_MAXITER, _QED_OTR_RANDOM_VECS,
        _QED_OTR_RED_FACTOR, _QED_OTR_SUBSOLVER, _otr_solve,
    )
    # ``hess_x`` is any symmetric v -> H v and ``h_diag`` its diagonal (used
    # only as the expansion preconditioner) -- the same contract as
    # _ah_krylov_trust_region_step.  Nothing below assumes the operator is
    # the exact orbital Hessian, so the BFGS-updated operator at the
    # qn-bfgs site is equally valid input.  Before this was generalized the
    # matvec was built inline from ctx.U2/A_tilde2/G_blocks, which is the
    # only reason OTR could not run at the QN sites; the main site now
    # passes ctx._orbital_hess_x(ctx.U2, ...), an identical closure.
    settings = _OtrSettings(
        n_micro=_QED_OTR_MAXITER,
        n_random_trial_vectors=_QED_OTR_RANDOM_VECS,
        subsystem_solver=_QED_OTR_SUBSOLVER,
        # QED_OTR_RED_FACTOR; both are set so the exit tolerance does not
        # silently change when mu happens to be ~0 (otr:1982-1988).
        global_red_factor=_QED_OTR_RED_FACTOR,
        local_red_factor=_QED_OTR_RED_FACTOR,
    )
    step, info = _otr_solve(
        reduced_gradient,
        h_diag,
        hess_x,
        trust_radius,
        settings=settings,
    )
    print(
        "%s trust region: micro %d, hess_x %d, mu %.6e, |step| %.6e, converged %s"
        % (
            label,
            info["n_micro"],
            info["n_hess_x"],
            info["mu"],
            np.linalg.norm(step),
            info["micro_converged"],
        ),
        flush=True,
    )
    return step


def _ah_krylov_trust_region_step(ctx, hess_x, h_diag, reduced_gradient,
                                 trust_radius, label="AH-Krylov"):
    """Krylov-seeded bordered augmented-Hessian trust-region step.

    Live only under QED_AH_KRYLOV=1 (main site) or QED_AH_KRYLOV_QN=1 (the
    two quasi-Newton sites).  Ported from casscf1's
    ``Davidson_augmented_hessian_solve3``; a drop-in for the trust-region
    dispatch in ``microiteration_optimization6`` with the same output as
    ``_otr_trust_region_step``, so the unitary build, energy check and
    trust-radius update downstream are unchanged.

    ``hess_x`` is any symmetric ``v -> H v``; ``h_diag`` its diagonal, used
    only as the expansion preconditioner.  Nothing here assumes the
    operator is the exact orbital Hessian, so the BFGS-updated operator at
    the ``gltr_qn_bfgs`` site is equally valid input -- as at that site
    already, the step is then the exact solution of an approximate
    subproblem, which is why it forces ``hard_case = 0``.

    The subproblem is the bordered pencil

        B(alpha) = [[0, alpha g^T], [alpha g, H]],   x = y[1:] / y0 / alpha,

    projected onto a growing subspace, with alpha searched INSIDE the
    projection (one small ``eigh`` per trial, no matvec) and the space
    expanded by the level-shifted diagonally preconditioned residual.  Same
    family as OpenTrustRegion; see the QED_AH_KRYLOV knob block for the
    three places it deliberately differs.

    DEVIATION FROM casscf1's solve3: the seed there batches ``e_0`` into
    the same ``orbital_sigma`` call as ``g/|g|`` and so pays a Hessian-
    vector product on a vector whose parameter part is identically zero.
    Here only ``g/|g|`` is transformed, which is one matvec per solve
    cheaper and changes nothing else.
    """
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _AH_BISECT_STEPS, _AH_BRACKET_STEPS, _AH_MAX_DIM, _AH_MAX_ITER,
        _AH_NEG_CURV, _AH_NEWTON, _AH_NEWTON_THRESH, _AH_PRECOND_FLOOR,
        _AH_RESID_FLOOR, _AH_RESID_TOL, _AH_Y0_TOL,
    )
    n = ctx.index_map_size
    H_dim = n + 1
    g = np.ascontiguousarray(reduced_gradient, dtype=np.float64)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-14:
        return np.zeros(n)
    H_diag_b = np.concatenate(([0.0], np.asarray(h_diag, dtype=float)))
    _hess_x = hess_x

    # ---- seed: e_0, g/|g|, and the negative-curvature direction ---------
    h_diag_arr = np.asarray(h_diag, dtype=float)
    seeds = [g / g_norm]
    if _AH_NEG_CURV:
        i_min = int(np.argmin(h_diag_arr))
        if h_diag_arr[i_min] < 0.0:
            e_nc = np.zeros(n)
            e_nc[i_min] = 1.0
            seeds.append(e_nc)
    rows = [np.concatenate(([1.0], np.zeros(n)))]      # e_0
    for v in seeds:
        w = np.concatenate(([0.0], v))
        for r in rows:
            w = w - (r @ w) * r
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        rows.append(w / w_norm)
    Q = np.ascontiguousarray(np.array(rows))
    S0 = np.zeros((Q.shape[0], H_dim))   # S0[:,1:] = H @ Q[:,1:]
    for i in range(1, Q.shape[0]):
        S0[i, 1:] = _hess_x(Q[i, 1:])
    n_matvec = Q.shape[0] - 1

    def reduced(alpha, A0, q0, gq):
        """Q B(alpha) Q^T."""
        return A0 + alpha * (np.outer(q0, gq) + np.outer(gq, q0))

    best_step = None
    best_q = np.inf
    max_dim = min(H_dim, _AH_MAX_DIM)
    # Exit reason, for QED_SOLVER_STATS-style diagnosis: "residual" is the
    # only healthy one.  "expansion-collapse" means the preconditioned
    # residual had nothing left to add to the space and the solver stopped
    # short -- there is no restart, so that is a hard stop.
    reason = "max-iter"

    for _it in range(_AH_MAX_ITER):
        m = Q.shape[0]
        q0 = np.ascontiguousarray(Q[:, 0])
        gq = Q[:, 1:] @ g
        A0 = Q[:, 1:] @ S0[:, 1:].T
        A0 = 0.5 * (A0 + A0.T)

        def solve_at(alpha):
            """(theta, y, bordered Ritz vector, step or None)."""
            th, V = np.linalg.eigh(reduced(alpha, A0, q0, gq))
            y = V[:, 0]
            vec = y @ Q
            if abs(vec[0]) < _AH_Y0_TOL:
                return th[0], y, vec, None
            return th[0], y, vec, vec[1:] / vec[0] / alpha

        # ---- reduced-space Newton shortcut (QED_AH_NEWTON) --------------
        # Q[0] = e_0 and every expansion vector is orthogonalised against
        # it, so Q[1:, 1:] is an orthonormal basis of the PARAMETER part
        # and A0[1:, 1:] is exactly the projected Hessian OpenTrustRegion
        # forms.  When it is positive definite and its Newton step is
        # interior, take that step at mu = 0 (otr:1953-1962) rather than
        # bisecting alpha down 14 decades to reach the same place with a
        # nonzero level shift left in the preconditioner.
        newton = False
        step_n = y_n = None
        if _AH_NEWTON and m > 1:
            H_red = A0[1:, 1:]
            if np.linalg.eigvalsh(H_red)[0] > _AH_NEWTON_THRESH:
                try:
                    y_n = np.linalg.solve(H_red, -gq[1:])
                except np.linalg.LinAlgError:
                    y_n = None
                if y_n is not None:
                    step_n = y_n @ Q[1:, 1:]
                    newton = np.linalg.norm(step_n) < trust_radius

        if not newton:
            # ---- alpha search on the WELL-DEFINED branch only -----------
            # ||x(alpha)|| decreases monotonically in alpha wherever the
            # step is defined, so march-and-bisect brackets it.  A step
            # counts as defined only when |y0| > QED_AH_Y0_TOL: y[1:]/y0
            # with y0 at roundoff returns norms of ~1e12 that manufacture a
            # phantom bracket, which is what hung the original prototype.
            def norm_at(alpha):
                st = solve_at(alpha)[3]
                return None if st is None else np.linalg.norm(st)

            alpha_lo = alpha_hi = None     # lo: ||x|| > Delta, hi: <=
            best_defined_a, best_defined_n = None, -1.0
            a = 1.0
            n_at = norm_at(a)
            if n_at is not None and n_at > best_defined_n:
                best_defined_a, best_defined_n = a, n_at
            if n_at is not None and n_at > trust_radius:
                alpha_lo = a
                for _ in range(_AH_BRACKET_STEPS):    # march up: step shrinks
                    a *= 10.0
                    n_at = norm_at(a)
                    if n_at is None:
                        break
                    if n_at <= trust_radius:
                        alpha_hi = a
                        break
                    alpha_lo = a
            else:
                if n_at is not None:
                    alpha_hi = a
                for _ in range(_AH_BRACKET_STEPS):    # march down: step grows
                    a /= 10.0
                    n_at = norm_at(a)
                    if n_at is None:
                        break
                    if n_at > best_defined_n:
                        best_defined_a, best_defined_n = a, n_at
                    if n_at > trust_radius:
                        alpha_lo = a
                        break
                    alpha_hi = a

            alpha = None
            if alpha_lo is not None and alpha_hi is not None:
                lo, hi = alpha_lo, alpha_hi
                for _ in range(_AH_BISECT_STEPS):
                    mid = np.sqrt(lo * hi)
                    n_at = norm_at(mid)
                    if n_at is None:
                        break
                    if abs(n_at - trust_radius) <= 1e-12 * trust_radius:
                        lo = hi = mid
                        break
                    if n_at > trust_radius:
                        lo = mid
                    else:
                        hi = mid
                cand = np.sqrt(lo * hi)
                if norm_at(cand) is not None:
                    alpha = cand

            if alpha is not None:
                alpha_step = alpha_exp = alpha
            else:                          # Delta out of reach in this space
                if best_defined_a is None:
                    reason = "no-well-defined-step"
                    break
                # Report the largest well-defined step, but EXPAND from the
                # plain augmented Hessian alpha = 1: theta -> 0 as alpha
                # -> 0, which switches the level shift in the
                # preconditioner off entirely and stalls the expansion.
                alpha_step = best_defined_a
                alpha_exp = 1.0 if norm_at(1.0) is not None else alpha_step

            theta_s, y_s, vec_s, step = solve_at(alpha_step)
            if step is None:
                reason = "step-undefined"
                break
            theta, y, vec = solve_at(alpha_exp)[:3]

            # The BORDERED residual drives the expansion, but must NOT test
            # convergence: at small alpha the bordered lowest eigenvector
            # tends to e_0 with theta -> 0, so its residual is O(alpha*|g|)
            # and vanishes however bad the subspace is.
            Sv = np.empty(H_dim)
            Sv[0] = alpha_exp * (vec[1:] @ g)
            Sv[1:] = (y @ S0[:, 1:]) + alpha_exp * vec[0] * g
            resid = Sv - theta * vec
            Hx = (y_s @ S0[:, 1:]) / (vec_s[0] * alpha_step)
        else:
            # OTR's Newton branch: mu = 0, step straight from the reduced
            # solve, expansion residual g + Hx at zero level shift.  The
            # border component is 0 because e_0 is projected out of every
            # expansion vector anyway.
            theta_s = theta = 0.0
            step = step_n
            Hx = y_n @ S0[1:, 1:]
            resid = np.zeros(H_dim)
            resid[1:] = Hx + g

        # Convergence on the SCALE-FREE optimality residual: dividing the
        # bordered eigenproblem's second block row by alpha*y0 gives
        # exactly (H - theta I) x = -g, the trust-region condition.
        opt_resid = Hx + g - theta_s * step
        resid_abs = np.linalg.norm(opt_resid)

        # Score by model value and keep the BEST: the boundary step is the
        # limit y0 -> 0+, so a subspace that goes on to contain the
        # critical direction exactly LOSES a converged boundary solution.
        q_model = g @ step + 0.5 * step @ Hx
        if q_model < best_q:
            best_q, best_step = q_model, step.copy()

        if resid_abs < max(_AH_RESID_TOL * g_norm, _AH_RESID_FLOOR):
            reason = "residual"
            break
        if m >= max_dim:
            reason = "max-dim"
            break

        # ---- expand: level-shifted preconditioned residual --------------
        den = theta - H_diag_b
        pw = np.divide(resid, den, out=np.zeros_like(resid),
                       where=np.abs(den) > _AH_PRECOND_FLOOR)
        for _ in range(2):
            pw -= Q.T @ (Q @ pw)
        pw_norm = np.linalg.norm(pw)
        if pw_norm < 1e-12:
            reason = "expansion-collapse"
            break
        Q = np.ascontiguousarray(np.vstack([Q, pw / pw_norm]))
        new_S = np.zeros((1, H_dim))
        new_S[0, 1:] = _hess_x(Q[-1, 1:])
        n_matvec += 1
        S0 = np.ascontiguousarray(np.vstack([S0, new_S]))

    if best_step is None:
        best_step = np.zeros(n)
    print(
        "%s trust region: dim %d, hess_x %d, |step| %.6e, q %.6e, exit %s"
        % (label, Q.shape[0], n_matvec, np.linalg.norm(best_step), best_q,
           reason),
        flush=True,
    )
    return best_step


def inner_solve_pcg(ctx, residual_k, theta_k, Q_conv, P_diag,
                U, A_tilde, G_blocks, reduced_gradient, alpha):
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _QED_JD_MAXITER, _QED_JD_TOL, _stats_begin, _stats_end,
    )

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
        # This is equivalent to your ctx.orbital_sigma(...)
        # and the gradient/alpha parts
        #ctx.orbital_sigma(U, A_tilde, G1, p_vec, Hp, 1, 1) # Assumes 'ctx' is available
        #start = timer()
        ctx.orbital_sigma3(U, A_tilde, G_blocks, p_vec, Hp, 1, 1) # Assumes 'ctx' is available
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


def build_hessian_diagonal(ctx, U, G, A_tilde):
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _QED_ORBITAL_SYMMETRY,
    )
    dim1 = ctx.n_occupied * ctx.nmo
    # start = timer()
    temp1 = np.zeros((ctx.n_occupied, ctx.nmo, ctx.nmo))
    temp1 = np.einsum(
        "kkrs->krs",
        G[: ctx.n_occupied, : ctx.n_occupied, :, :],
        optimize="optimal",
    )
    temp2 = np.zeros((ctx.n_occupied, ctx.nmo, ctx.n_virtual))
    temp2 = np.einsum(
        "krs,sa->kra", temp1, U[:, ctx.n_occupied : ctx.nmo], optimize="optimal"
    )
    ctx.hessian_diagonal = np.zeros((ctx.nmo, ctx.n_occupied))
    ctx.hessian_diagonal[ctx.n_occupied : ctx.nmo, :] = np.einsum(
        "kra,ra->ak", temp2, U[:, ctx.n_occupied : ctx.nmo], optimize="optimal"
    )

    # end = timer()
    # print("hessian diagonal step took", end - start)
    # B = np.zeros((ctx.nmo, ctx.nmo))
    # T = U - np.eye(ctx.nmo)
    # B[:,:ctx.n_occupied] = A[:,:ctx.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:ctx.n_occupied])
    # A_tilde = np.zeros((ctx.nmo, ctx.nmo))
    # A_tilde = np.einsum("rs,sk->rk",U.T, B)
    temp4 = np.zeros((ctx.n_occupied, ctx.n_occupied))

    # start = timer()
    o = np.full((ctx.n_occupied), 1)
    v = np.full((ctx.n_virtual), 1)
    ctx.hessian_diagonal[ctx.n_occupied : ctx.nmo, :] -= np.einsum(
        "k,a->ak",
        o,
        np.diagonal(
            A_tilde[ctx.n_occupied : ctx.nmo, ctx.n_occupied : ctx.nmo]
        ),
    )
    ctx.hessian_diagonal[ctx.n_occupied : ctx.nmo, :] -= np.einsum(
        "a,k->ak", v, np.diagonal(A_tilde[: ctx.n_occupied, : ctx.n_occupied])
    )
    # end = timer()
    # print("hessian diagonal step 1 took", end - start)
    # start = timer()
    temp3 = np.zeros((ctx.n_occupied, ctx.nmo, ctx.n_occupied))
    temp3 = np.einsum(
        "krs,sl->krl", temp1, U[:, : ctx.n_occupied], optimize="optimal"
    )
    temp4 = np.einsum(
        "krl,rl->lk", temp3, U[:, : ctx.n_occupied], optimize="optimal"
    )
    ctx.hessian_diagonal[: ctx.n_occupied, :] += 1.0 * temp4
    ctx.hessian_diagonal[: ctx.n_occupied, :] += 1.0 * temp4.T
    # end = timer()
    # print("hessian diagonal step 2 took", end - start)
    # start = timer()
    temp3 = np.einsum(
        "klrs,sk->krl", G, U[:, : ctx.n_occupied], optimize="optimal"
    )
    # end = timer()
    # print("hessian diagonal step 3 took", end - start)
    # start = timer()
    ctx.hessian_diagonal[: ctx.n_occupied, :] -= 2.0 * np.einsum(
        "krl,rl->kl", temp3, U[:, : ctx.n_occupied], optimize="optimal"
    )
    ctx.hessian_diagonal[: ctx.n_occupied, :] -= np.einsum(
        "k,l->kl", o, np.diagonal(A_tilde[: ctx.n_occupied, : ctx.n_occupied])
    )
    ctx.hessian_diagonal[: ctx.n_occupied, :] -= np.einsum(
        "l,k->kl", o, np.diagonal(A_tilde[: ctx.n_occupied, : ctx.n_occupied])
    )
    temp4 = (
        A_tilde[: ctx.n_occupied, : ctx.n_occupied]
        + A_tilde.T[: ctx.n_occupied, : ctx.n_occupied]
    )
    ctx.hessian_diagonal[: ctx.n_occupied, :] += np.einsum(
        "kl,kl->kl", np.eye(ctx.n_occupied), temp4, optimize="optimal"
    )
    # end = timer()
    # print("hessian diagonal step 4 took", end - start)
    # print("i7gt",hessian_diagonal)
    # start = timer()
    index_count1 = 0
    for k in range(ctx.n_occupied):
        for r in range(k + 1, ctx.nmo):
            if k < ctx.n_in_a and r < ctx.n_in_a:
                continue
            if (
                ctx.n_in_a <= k < ctx.n_occupied
                and ctx.n_in_a <= r < ctx.n_occupied
            ):
                continue
            # QED_ORBITAL_SYMMETRY: skip rotations that mix symmetry species.
            if (
                _QED_ORBITAL_SYMMETRY
                and ctx._sym_labels is not None
                and ctx._sym_labels[r] != ctx._sym_labels[k]
            ):
                continue
            ctx.reduced_hessian_diagonal[index_count1] = ctx.hessian_diagonal[r][
                k
            ]
            index_count1 += 1
    # The preconditioner must describe the SAME operator orbital_sigma3/mv2
    # apply, otherwise the level shift would be invisible to every
    # diagonally-preconditioned solver and to the QED_AH_NEG_CURV seed
    # (which picks argmin of this diagonal).  The loop above walks index_map's
    # ordering, so the shift vector lines up entry for entry.
    _shift = ctx._actvirt_shift_vector()
    if _shift is not None:
        ctx.reduced_hessian_diagonal += _shift


def aug_matvec(ctx, U, A_tilde, G_blocks, reduced_gradient, vector, alpha, H_dim):
    p_vec = vector.reshape(1, H_dim)
    Hp = np.zeros_like(p_vec)

    # This is your expensive "H*p" product
    # You must adapt this to use your existing functions

    # 1. Apply the (H) part
    # This is equivalent to your ctx.orbital_sigma(...)
    # and the gradient/alpha parts
    #ctx.orbital_sigma(U, A_tilde, G1, p_vec, Hp, 1, 1) # Assumes 'ctx' is available
    #start = timer()
    ctx.orbital_sigma3(U, A_tilde, G_blocks, p_vec, Hp, 1, 1) # Assumes 'ctx' is available
    Hp[:, 1:] += np.einsum("p,q->qp", reduced_gradient, p_vec[:, 0])
    Hp[:, 0] = alpha * p_vec[:, 0] + np.dot(p_vec[:, 1:], reduced_gradient)
    #end = timer()
    #print("build orbital sigma for precondition took", end - start)

    return Hp.flatten() # Return a 1D vector


def solve_gltr_with_operator(ctx, B_operator, g, M_diag, radius, max_iter=100, tol=1e-4, verbose=True):
    """
    Returns: (p, lambda, status)
    lambda is the shift such that (H + lambda*M) is positive semidefinite.
    """
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _CPP_CASSCF_VALIDATION_DIR, _QED_GLTR_TOL_SCALE, _QED_SOLVER_STATS,
        _solver_stats,
    )
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
        ctx._cpp_casscf_last_noised_gradient = r.copy()

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


def solve_gltr_trust_region(ctx, U, A_tilde, G_blocks, g, M_diag, radius, max_iter=100, tol=1e-4, verbose=True):
    """
    Returns: (p, lambda, status)
    lambda is the shift such that (H + lambda*M) is positive semidefinite.
    """
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _CPP_CASSCF_VALIDATION_DIR, _QED_GLTR_REUSE, _QED_GLTR_TOL_SCALE,
        _QED_SOLVER_STATS, _solver_stats,
    )
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
    # (~13037) touches nothing else, ctx.U3 is re-seeded from ctx.U2 at
    # ~12148, and only the ACCEPT branch updates U2 / rebuilds the gradient
    # and the Hessian diagonal.  The Lanczos recurrence below depends only on
    # (H, M_abs, r): the radius enters at CHECK A/B, never in the vectors.
    # So the cached basis is EXACT for the retry and the step is bit-identical
    # to what a cold rebuild produces -- the property GLTR is built on
    # (Gould/Lucidi/Roma/Toint: re-solution over a sequence of radii).
    # Keyed on the CONTENT of g and M_diag, so an accepted step invalidates
    # the cache automatically with no manual bookkeeping.
    _cache = getattr(ctx, "_gltr_cache", None)
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
        ctx._cpp_casscf_last_noised_gradient = r.copy()

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
            ctx.orbital_sigma3(U, A_tilde, G_blocks, v_curr_reshaped, H_v, 1, 0)
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
            ctx._gltr_cache = {
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


def scipy_minres_solve(ctx, U, A_tilde, G_blocks, reduced_gradient, denom,
                       max_iter, conv_thresh=1e-6):
    """scipy's UNPRECONDITIONED MINRES as the primary Newton solver.

    Selected by QED_NEWTON_SCIPY (see the knob block at the top of this
    file for why this leg exists and what it isolates).  Drop-in for
    minres_solve / linear_equation_solve: same argument list, same
    (solution, converged) contract.

    `denom` and `conv_thresh` are accepted and IGNORED -- no preconditioner
    is the whole point, and scipy's rtol is relative to ||b||, which is the
    property being tested.  `max_iter` is likewise ignored so that this
    matches how the existing fallback invokes scipy (no maxiter, i.e.
    scipy's default 5n).
    """
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _QED_NEWTON_SCIPY,
    )
    b = -np.asarray(reduced_gradient, dtype=np.float64).ravel()
    n = b.shape[0]
    g_norm = float(np.linalg.norm(b))
    rtol = _QED_NEWTON_SCIPY if _QED_NEWTON_SCIPY > 0.0 else 1e-6

    _nmv = [0]

    def _mv(v):
        _nmv[0] += 1
        return ctx.mv2(U, A_tilde, G_blocks,
                        np.asarray(v).ravel(), 1, 0, 0).ravel()

    print("--------------------------------------------")
    print("--- Start solving Hx=-g equations (scipy MINRES, unpreconditioned) ---")
    print("--------------------------------------------")
    op = LinearOperator((n, n), matvec=_mv, dtype=float)
    x, info = minres(op, b, rtol=rtol)
    true_residual = float(np.linalg.norm(
        ctx.mv2(U, A_tilde, G_blocks, x, 1, 0, 0).ravel() - b))
    converged = bool(info == 0)
    print("total iteration", _nmv[0])
    # Same "quality" line the other solvers print, so the legs are directly
    # comparable: what relative accuracy did we actually achieve?
    print(
        "PMINRES quality: rel residual %.3e (true/||g||), ||g|| %.3e"
        % (true_residual / g_norm if g_norm > 0 else np.inf, g_norm)
    )
    print(f"scipy MINRES exit code: {info}  matvecs: {_nmv[0]}")
    if converged:
        print("\n--- Convergence Achieved ---")
    else:
        print("\n--- Solver did not converge within max iterations ---")
    return x, converged


def minres_solve(ctx, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6):
    """
    Diagonally-preconditioned MINRES (Paige & Saunders) for the Newton
    equations  H x = -g,  where H is the reduced orbital Hessian applied
    matrix-free through ctx.mv2 and g is `reduced_gradient`.

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

    H is symmetric but INDEFINITE here (ctx.reduced_hessian_diagonal
    routinely has negative entries -- that count is what n_negative
    measures), which is why this is MINRES and not CG.

    Preconditioner: M = max(|denom|, 1e-6), elementwise.  MINRES requires
    an SPD preconditioner, so the absolute value is mandatory, not a
    safeguard -- `denom` is ctx.reduced_hessian_diagonal and is signed.
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
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _QED_EXACT_TR_BATCH, _QED_NEWTON_DUMP, _QED_NEWTON_DUMP_MAX,
        _QED_NEWTON_DUMP_STRIDE, _QED_NEWTON_FORCING,
        _QED_NEWTON_RANDOM_START, _QED_SOLVER_STATS,
    )
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
    g_norm = float(np.linalg.norm(b))

    # --- QED_NEWTON_DUMP: capture this subproblem for the offline solver
    # benchmark (see the knob comment).  Purely observational -- nothing
    # below reads anything written here, and with the knob unset the whole
    # block is skipped, so it is a byte-for-byte no-op by default. ---
    if _QED_NEWTON_DUMP:
        _seen = getattr(ctx, "_newton_dump_seen", 0)
        _done = getattr(ctx, "_newton_dump_done", 0)
        ctx._newton_dump_seen = _seen + 1
        if (_done < _QED_NEWTON_DUMP_MAX
                and _seen % max(1, _QED_NEWTON_DUMP_STRIDE) == 0):
            os.makedirs(_QED_NEWTON_DUMP, exist_ok=True)
            # Dense H by columns, same batched orbital_sigma3 path
            # _exact_dense_trust_region_step uses.  Duplicated rather than
            # refactored so that routine stays byte-for-byte untouched.
            _batch = max(1, min(int(_QED_EXACT_TR_BATCH), n))
            _H = np.zeros((n, n))
            _col = 0
            while _col < n:
                _nb = min(_batch, n - _col)
                _R = np.zeros((_nb, n))
                for _k in range(_nb):
                    _R[_k, _col + _k] = 1.0
                _R = np.ascontiguousarray(_R)
                _S = np.zeros((_nb, n))
                ctx.orbital_sigma3(U, A_tilde, G_blocks, _R, _S, _nb, 0)
                _H[:, _col:_col + _nb] = _S.T
                _col += _nb
            _path = os.path.join(
                _QED_NEWTON_DUMP, "newton_%04d.npz" % ctx._newton_dump_seen)
            np.savez(
                _path,
                H=_H,
                g=np.asarray(reduced_gradient, dtype=np.float64).ravel(),
                denom=np.asarray(denom, dtype=np.float64).ravel(),
                conv_thresh=np.float64(conv_thresh),
                max_iter=np.int64(max_iter),
                solve_index=np.int64(ctx._newton_dump_seen),
            )
            ctx._newton_dump_done = _done + 1
            print(
                "NEWTON DUMP %s  n=%d ||g||=%.6e asym=%.2e (%d/%d)"
                % (_path, n, g_norm, float(np.abs(_H - _H.T).max()),
                   ctx._newton_dump_done, _QED_NEWTON_DUMP_MAX)
            )

    # --- QED_NEWTON_FORCING: relative exit tolerance.  Default 0 leaves
    # conv_thresh exactly as the caller passed it, so this block is a
    # byte-for-byte no-op when the knob is unset.  See the knob comment. ---
    if _QED_NEWTON_FORCING != 0.0 and g_norm > 0.0:
        if _QED_NEWTON_FORCING > 0.0:
            eta = _QED_NEWTON_FORCING
        else:
            # Eisenstat-Walker choice-1 constant form; the min() caps the
            # demand at 10% relative so an early, large-||g|| solve does not
            # get an absurdly loose tolerance.
            eta = min(0.1, np.sqrt(g_norm))
        conv_thresh = eta * g_norm
        print(
            "PMINRES forcing: ||g|| %.3e  eta %.3e  conv_thresh %.3e "
            "(relative)" % (g_norm, eta, conv_thresh)
        )
    elif _QED_SOLVER_STATS:
        # The number that shows whether the ABSOLUTE default is vacuous:
        # conv_thresh/||g|| > 1 means x = 0 already passes the test.
        print(
            "PMINRES tol: ||g|| %.3e  conv_thresh %.3e  demanded relative "
            "residual %.3e" % (g_norm, conv_thresh,
                               conv_thresh / g_norm if g_norm > 0 else np.inf)
        )

    print("--------------------------------------------")
    print("--- Start solving Hx=-g equations (preconditioned MINRES) ---")
    print("--------------------------------------------")

    # --- Set up the first Lanczos vector ---
    r1 = b.copy()

    # --- QED_NEWTON_RANDOM_START (diagnostic; see the knob comment for the
    # deviation from linear_equation_solve this necessarily involves).  The
    # ||g|| < 1e-3 gate is copied from linear_equation_solve:20327 so the
    # branch fires on the same solves.  Costs one extra mv2. ---
    if _QED_NEWTON_RANDOM_START != 0.0 and g_norm < 1e-3:
        trial_0 = np.random.rand(n)
        _t_nrm = np.linalg.norm(trial_0)
        if _t_nrm > 1e-12:
            trial_0 = trial_0 / _t_nrm
        x = _QED_NEWTON_RANDOM_START * trial_0
        r1 = b - ctx.mv2(U, A_tilde, G_blocks, x, 1, 0, 0).ravel()
        print(
            "PMINRES random start: ||x0|| %.3e  ||r0|| %.3e (vs ||g|| %.3e)"
            % (np.linalg.norm(x), np.linalg.norm(r1), g_norm)
        )
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
        y = ctx.mv2(U, A_tilde, G_blocks, v, 1, 0, 0).ravel()
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
        # SOLUTION QUALITY, the primary thing to score a solver change on:
        # what relative accuracy did we actually achieve?  Reported from the
        # bound, so it is an UPPER bound on the relative residual and costs
        # no mv2.  (bound_factor runs a measured 3.3-6.8x above truth.)
        print(
            "PMINRES quality: rel residual <= %.3e (bound/||g||), ||g|| %.3e"
            % (residual_bound / g_norm if g_norm > 0 else np.inf, g_norm)
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
        ctx.mv2(U, A_tilde, G_blocks, x, 1, 0, 0).ravel() - b
    )
    converged = bool(true_residual < conv_thresh)
    print(f"PMINRES: true residual ||Hx+g|| = {true_residual:.4e} after {itn} iterations")
    # SOLUTION QUALITY (exact here -- the true residual was just evaluated).
    print(
        "PMINRES quality: rel residual %.3e (true/||g||), ||g|| %.3e"
        % (true_residual / g_norm if g_norm > 0 else np.inf, g_norm)
    )
    if converged:
        print("\n--- Convergence Achieved ---")
        print("total iteration", itn)
    else:
        print("\n--- Solver did not converge within max iterations ---")
    return x, converged


def linear_equation_solve(ctx, U, A_tilde, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6):
    """
    Main driver loop to solve the linear system Ax + b = 0.

    Superseded by minres_solve above (see its docstring for why); kept as
    the _QED_USE_MINRES=0 comparison path and as the TAMM/cpp_casscf port
    reference, whose LinearRMSolver mirrors this routine.
    """
    #ctx._reset()
    residual = reduced_gradient.copy()
    if np.linalg.norm(residual) < 1e-3:
        #random guess
        dim00 = ctx.index_map_size
        trial_0 = np.random.rand(dim00)

        norm = np.linalg.norm(trial_0)
        if norm > 1e-12:
            trial_0 /= norm

        sigma = ctx.mv2(U, A_tilde, G_blocks, trial_0, 1, 0, 0)
        residual = sigma + reduced_gradient

    #print("initial guess", residual)
    #residual = ctx.apply_preconditioner(A, G1, z)
    #residual /= np.linalg.norm(residual)
    solver_sym = LinearRMSolver(b_vector=reduced_gradient, max_subspace=max_iter)
    #solution_sym = solver_sym.solve(matvec_prod_sym, precond_sym, max_iter=100, conv_thresh=1e-7)
    #print(denom[:ctx.index_map_size])
    print("--------------------------------------------")
    print("--- Start solving Hx=-g equations ---")
    print("--------------------------------------------")
    for i in range(max_iter):
        #sigma0 = matvec_product(residual, A, G1)
        #residual0 = sigma0 + ctx.total_gradient
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

        sigma = ctx.mv2(U, A_tilde, G_blocks, trial_c, 1, 0, 0)
        #print("sigma", sigma)
        residual_old =copy.deepcopy(residual)
        residual = solver_sym.update_subspace_and_extrapolate(trial_c, sigma)
        residual_new = copy.deepcopy(residual)
        error = residual_new - residual_old
        if i > 0 and np.linalg.norm(error) < 1e-8:
            print("\n--- Convergence Achieved (solution becomes ctx-consistent)---")
            print("total iteration", i+1)
            return solver_sym.get_solution(), True


    print("\n--- Solver did not converge within max iterations ---")
    return solver_sym.get_solution(), False


def projection_step2(
    ctx, gradient_tilde, hessian_tilde, aug_eigenvecs, alpha, dim0
):
    # Module constants fixed at import; helper_PFCI imports THIS module,
    # so a module-level import would be circular.  Same idiom as
    # lstrs_trust_region.py -- reading them per call is value-identical.
    from helper_PFCI import (
        _QED_SOLVER_STATS, _solver_stats,
    )
    if _QED_SOLVER_STATS:
        _solver_stats.bordered_solve()
    augmented_hessian = np.zeros((dim0, dim0))
    # print("projected_gradient", gradient_tilde, flush = True)

    augmented_hessian[0, 0] = alpha
    augmented_hessian[0, 1:] = gradient_tilde
    augmented_hessian[1:, 0] = gradient_tilde.T
    augmented_hessian[1:, 1:] = hessian_tilde
    # print("projected_hessian")
    # ctx.printA(hessian_tilde)
    # print("projected_augmented_hessian")
    # ctx.printA(augmented_hessian)
    # print(" ", flush = True)

    mu, aug_eigenvecs[:, :] = np.linalg.eigh(augmented_hessian)
    # np.set_printoptions(precision = 14)
    # print("check eigenvalues and eigenvectors of the projected augmented hessian", flush = True)
    # print("eigenvecs",aug_eigenvecs, flush = True)
    # print("alpha",alpha, flush = True)
    print(mu, flush=True)
    return mu