"""LSTRS-family trust-region subproblem solvers, split out of helper_PFCI.py.

Companion to ``otr_trust_region.py`` and deliberately the same shape: the
solvers here take a gradient, a Hessian DIAGONAL (preconditioner only) and a
matrix-free ``hess_x`` closure, never a PFHamiltonianGenerator, so they can
serve any site that can supply those three things.

WHY THIS FILE EXISTS.  The LSTRS beta-bisection was written inline in FOUR
places -- ``internal_optimization3``, ``microiteration_optimization6``, and the
now-removed ``microiteration_optimization5`` and ``ah_orbital_optimization`` --
as four copies of the same ~700 lines.  The latter two were deleted on
2026-09-04 (they were a template and a prototype, both since absorbed into
``microiteration_optimization6``), taking two of the four copies with them; the
other two live here.  That duplication is not cosmetic: the
both-roots-unnormalizable hard case was left unimplemented as a bare ``exit()``
in every one of them, so a single genuine hard case (item 371: two lowest
Hessian eigenvalues degenerate to 5.7e-17 with the gradient orthogonal to both
at machine zero) killed the process with STATUS 0 from whichever copy it
happened to reach.  One implementation is the fix that keeps being the fix.

MOVED VERBATIM, NOT REWRITTEN.  Everything here is byte-for-byte what it was in
helper_PFCI.py apart from the leading underscore being dropped from public
names.  The extraction is gated bit-exactly by
``validation/lstrs_refactor_gate.sh``, which compares full solver TRAJECTORIES
(not just converged energies) across eight runs, precisely because item 371
showed two mean fields agreeing to 12 dp in the energy while diverging into
different solver branches.
"""

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, minres


def more_sorensen_dense(H, g, delta, deg_rtol=1e-10, gnull_rtol=1e-10):
    """Exact trust-region subproblem solve for a DENSE symmetric H.

    Solves   min_s  g.s + 0.5 s.H.s   s.t.  ||s|| <= delta
    by the More-Sorensen characterization: s is optimal iff there is a
    lambda >= max(0, -lam_min) with (H + lambda I) positive semidefinite,
    (H + lambda I) s = -g, and lambda (||s|| - delta) = 0.

    Used only under QED_EXACT_TR, where n = index_map_size (150 on the MgH+
    CAS(8,12) case) so a full eigendecomposition is cheap relative to the
    Hessian-vector products already being spent on the same subproblem.

    Unlike the Krylov solvers at this site (GLTR / AH-Krylov / OTR) this never
    forms a Krylov space, so the hard case -- g orthogonal to the lowest
    eigenvector, which is where all three of those go blind -- is handled
    explicitly rather than approached iteratively.

    Returns (step, lam_shift, kind) with kind in
    {"interior", "boundary", "hard_case"}.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    g = np.asarray(g, dtype=float).ravel()
    n = g.shape[0]
    lam, V = np.linalg.eigh(H)
    gt = V.T @ g
    scale = max(1.0, float(np.abs(lam).max()) if n else 1.0)
    gnorm = float(np.linalg.norm(g))

    if gnorm == 0.0:
        # Pure negative-curvature step: no gradient to descend, so go as far as
        # the trust region allows along the most negative direction.
        if lam[0] < 0.0:
            return delta * V[:, 0], -lam[0], "hard_case"
        return np.zeros(n), 0.0, "interior"

    def s_of(l):
        return -(V @ (gt / (lam + l)))

    # Unconstrained Newton step, taken only if H > 0 and it stays inside.
    if lam[0] > deg_rtol * scale:
        s = s_of(0.0)
        if np.linalg.norm(s) <= delta:
            return s, 0.0, "interior"

    lo = -float(lam[0])
    deg = np.abs(lam - lam[0]) <= deg_rtol * scale
    safe = ~deg

    # HARD CASE: g has (numerically) no component in the lowest eigenspace, so
    # (H + lo I) s = -g cannot be pushed out to the boundary by raising lambda
    # -- the pseudo-inverse solution is already interior.  The optimizer is then
    # that solution plus a multiple of a lowest eigenvector.  This is exactly
    # the configuration a g-seeded Krylov method cannot represent.
    if np.linalg.norm(gt[deg]) <= gnull_rtol * gnorm:
        s_pinv = np.zeros(n)
        if safe.any():
            s_pinv = -(V[:, safe] @ (gt[safe] / (lam[safe] + lo)))
        if np.linalg.norm(s_pinv) <= delta:
            tau = math.sqrt(max(delta * delta - float(s_pinv @ s_pinv), 0.0))
            v = V[:, int(np.argmax(deg))]
            best, best_m = None, None
            for cand in (s_pinv + tau * v, s_pinv - tau * v):
                mval = float(g @ cand + 0.5 * cand @ (H @ cand))
                if best_m is None or mval < best_m:
                    best, best_m = cand, mval
            return best, lo, "hard_case"

    # BOUNDARY: ||s(l)|| is strictly decreasing on (lo, inf), so bracket then
    # bisect.  Bisection rather than the usual Newton iteration on
    # 1/||s|| - 1/delta because n is small and this cannot fail to converge.
    l = lo + max(1e-14, 1e-12 * scale)
    hi = l + 1.0
    for _ in range(200):
        if np.linalg.norm(s_of(hi)) <= delta:
            break
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (l + hi)
        if np.linalg.norm(s_of(mid)) > delta:
            l = mid
        else:
            hi = mid
    l_star = 0.5 * (l + hi)
    return s_of(l_star), l_star, "boundary"



def solve_lstrs_microiteration(
        ctx, A_tilde2, G, G_blocks, dim00, reduced_gradient, trust_radius, w9, first_component, second_component, guess_vector, hard_case, beta):
    """The LSTRS beta-bisection from ``microiteration_optimization6``.

    MOVED VERBATIM out of helper_PFCI.py (~630 lines inline inside an already
    very long method).  The ONLY edits are mechanical: ``self.`` -> ``ctx.``,
    the inherited names promoted to parameters, and ``return step, hard_case``.
    No arithmetic, no reordering, no renaming -- gated bit-exactly by
    ``validation/lstrs_refactor_gate.sh``.

    GETTING THE PARAMETER LIST RIGHT WAS THE WHOLE DIFFICULTY, and it is not a
    syntactic question.  Three names (``guess_vector``, ``hard_case``, ``beta``)
    are assigned INSIDE the block as well, so a "never assigned here" scan calls
    them local -- but each is read on a path that reaches before that assignment,
    so they are inherited.  ``guess_vector`` proved it with
    ``UnboundLocalError`` at the first attempt.  The opposite over-correction
    (pass everything the method mentions) fails the other way: ``delta``,
    ``t1``, ``max_iter`` and friends are block-local and NOT bound at the call
    site, so passing them raises UnboundLocalError in the CALLER.  The list
    below is the set that is both read-before-write here and unconditionally
    bound there.

    ``ctx`` is the PFHamiltonianGenerator.  Taking it whole is a deliberate
    FIRST step and a documented deviation from ``otr_trust_region.py``, which
    takes a ``hess_x`` closure and no generator: that is the right end state and
    this is not it yet.  Purifying the interface means threading 13 distinct
    ``ctx`` members through a callback struct -- worth doing SEPARATELY so each
    half stays independently bisectable if the gate fails.

    ``w9``, ``first_component`` and ``second_component`` are caller-owned
    buffers written in place, so they are passed rather than returned.

    Returns ``(step, hard_case)`` -- ``hard_case`` because the caller still
    tests it (``step_accepted = ... or hard_case == 2``).
    """
    # Deferred to call time: helper_PFCI imports THIS module (for
    # more_sorensen_dense), so a module-level import would be circular.  These
    # are module constants fixed at import, so reading them per call is
    # value-identical to reading them as globals.
    from helper_PFCI import (
        _QED_LSTRS_EPS_HC, _QED_LSTRS_HARD_DUMP, _QED_LSTRS_HARD_PATIENCE,
        _QED_LSTRS_NORM_TOL, _QED_MINRES_TRACE, _QED_NEWTON_MAXITER,
        _QED_NEWTON_SCIPY, _QED_TRS_HARD_FALLBACK, _QED_USE_MINRES,
        _mp_start, _mp_end, _stats_begin, _stats_end, _QED_LSTRS_MAX_ITER,
    )

    mu9 = np.zeros(dim00)
    beta0 = 0
    both_roots_normalization = 0
    v1 = 0
    v2 = 0
    u1 = np.zeros(ctx.index_map_size)
    u2 = np.zeros(ctx.index_map_size)
    x1 = np.zeros(ctx.index_map_size)
    x2 = np.zeros(ctx.index_map_size)
    phi1 = 0
    phi2 = 0

    lambda1 = 0
    lambda2 = 0
    lambda_c = 0
    idx = ctx.reduced_hessian_diagonal.argsort()
    delta_u = ctx.reduced_hessian_diagonal[idx[0]]
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
    ctx.collapse_subspace_check = False
    _stats_span = _stats_begin("lstrs_davidson")
    while True:
        print("****************************")
        print("lstrs iteration", count10, "beta", beta)
        beta0 = beta
        # mu9[:] = ctx.projection_step2(reduced_gradient, reduced_hessian, w9, beta, dim00)
        # print("two roots", mu9[0], mu9[1], mu9[2])
        if count10 == 0:
            alpha_range = np.abs(alpha_u - alpha_l)
            Q = ctx.Davidson_augmented_hessian_solve6(
                ctx.U2,
                A_tilde2,
                G,
                G_blocks,
                ctx.reduced_hessian_diagonal,
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
            Q = ctx.Davidson_augmented_hessian_solve6(
                ctx.U2,
                A_tilde2,
                G,
                G_blocks,
                ctx.reduced_hessian_diagonal,
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
        # QED_LSTRS_HARD_PATIENCE: bisecting cannot escape a genuine hard
        # case -- on the dumped subproblem the Davidson returned
        # BIT-IDENTICAL eigenvalues for the last ten iterations while
        # alpha kept halving, so the loop only ended when the bracket
        # collapsed after 40 of them.  Stop early and let the
        # hard-case fallback below take the step.
        _hard_spin = 0
        while (
            (aa1 <= epsilon_v * bb1)
            and (aa2 <= epsilon_v * bb2)
            and (_QED_LSTRS_HARD_PATIENCE <= 0
                 or _hard_spin < _QED_LSTRS_HARD_PATIENCE)
            and np.abs(alpha_u - alpha_l)
            > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
        ):
            _hard_spin += 1
            alpha_u = alpha
            alpha = (alpha_l + alpha_u) / 2
            _mp0_w9 = _mp_start()
            w9[:, :] = 0.0
            _mp_end("w9_memset", _mp0_w9)
            # mu9[:] = ctx.projection_step2(reduced_gradient, reduced_hessian, w9, alpha, dim00)
            # print("two roots", mu9[0], mu9[1], mu9[2])

            alpha_range = np.abs(alpha_u - alpha_l)
            Q = ctx.Davidson_augmented_hessian_solve6(
                ctx.U2,
                A_tilde2,
                G,
                G_blocks,
                ctx.reduced_hessian_diagonal,
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

        if _QED_LSTRS_HARD_DUMP and (
                aa1 <= epsilon_v * bb1) and (
                aa2 <= epsilon_v * bb2):
            # DIAGNOSTIC ONLY, default off, no
            # behaviour change: the beta-bisection
            # above ended with BOTH roots still
            # unnormalizable, which is the state
            # the bare exit() below cannot handle.
            # Dump the subproblem densely so the
            # degeneracy can be checked against an
            # exact eigendecomposition rather than
            # against the Davidson that produced
            # it.  Same dense-H-by-columns recipe
            # as the QED_NEWTON_DUMP block (:22138).
            ctx._lstrs_hard_dump(
                ctx.U2, A_tilde2, G_blocks,
                reduced_gradient, alpha, alpha_l,
                alpha_u, trust_radius, mu9, w9,
                v1, v2, aa1, bb1, aa2, bb2,
                epsilon_v, aug_hessian_eigenvals)
        if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
            # The beta bisection ended with BOTH
            # roots unnormalizable: a genuine hard
            # case, not a solver artefact (see the
            # QED_TRS_HARD_FALLBACK note).  This
            # used to be a bare exit().
            if not _QED_TRS_HARD_FALLBACK:
                exit()
            x_hard_fallback = (
                ctx._trs_hard_case_fallback(
                    ctx.U2, A_tilde2, G_blocks,
                    reduced_gradient,
                    trust_radius,
                    site=" [microiteration]"))
            hard_case = 5
            break
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
        psi_tilde = 0.5 * ctx.microiteration_predicted_energy2(
            ctx.U2, reduced_gradient, A_tilde2, G_blocks, x_tilde
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
                        * ctx.microiteration_predicted_energy2(
                            ctx.U2,
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
        if count10 == _QED_LSTRS_MAX_ITER:
            # 2026-09-26: the beta bisection hit its iteration cap (QED_LSTRS_MAX_ITER, default 50).  Nothing bound `step` on this exit: the inline
            # original (helper_PFCI microiteration_optimization6 / internal_optimization3, before d0c7dee) silently reused
            # the previous iteration's step; the extracted function raised UnboundLocalError (seen once, in the
            # composition search, c006).  Take the exact trust-region step, as for the both-roots-unnormalizable case.
            if not _QED_TRS_HARD_FALLBACK:
                exit()
            x_hard_fallback = ctx._trs_hard_case_fallback(ctx.U2, A_tilde2, G_blocks, reduced_gradient, trust_radius,
                                                          site=" [microiteration, iteration cap]")
            hard_case = 5
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
            # temp_vector = np.zeros(ctx.index_map_size)
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
        #Q = np.zeros((1, ctx.index_map_size))
        #H1_op = LinearOperator(
        #    (ctx.index_map_size, ctx.index_map_size),
        #    matvec=lambda Q: ctx.mv2(
        #        ctx.U2, A_tilde2, G1, Q, 1, 0, 0
        #    ),
        #)
        #x, exitCode = minres(H1_op, -reduced_gradient, rtol=1e-6)
        #print("exitcode", exitCode)
        #step = x
        _stats_span_n = _stats_begin("newton_in_lstrs")
        denom = ctx.reduced_hessian_diagonal
        max_iter = _QED_NEWTON_MAXITER  # QED_NEWTON_MAXITER, default 20
        # QED_NEWTON_SCIPY overrides QED_USE_MINRES; both default off.
        if _QED_NEWTON_SCIPY:
            _solve_newton = ctx.scipy_minres_solve
        else:
            _solve_newton = ctx.minres_solve if _QED_USE_MINRES else ctx.linear_equation_solve
        solution, converged = _solve_newton(ctx.U2, A_tilde2, G_blocks, reduced_gradient, denom, max_iter, conv_thresh=1e-6)
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
                (ctx.index_map_size, ctx.index_map_size),
                matvec=lambda Q: ctx.mv2(ctx.U2, A_tilde2, G_blocks, Q, 1, 0, 0),
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
    if hard_case == 5:
        # QED_TRS_HARD_FALLBACK: the step is already
        # complete (dense More-Sorensen, or GLTR above
        # the size cap).  Unlike cases 1 and 4 it needs
        # no boundary rescaling -- the solver placed it
        # on the boundary itctx.
        step = x_hard_fallback
    return step, hard_case


def exact_internal_trs(hessian_tilde_ai, gradient_tilde_ai, trust_radius):
    """Solve the INTERNAL trust-region subproblem exactly (QED_INTERNAL_EXACT_TRS).

    The internal site is the one place where the exact solve is essentially
    free.  ``hessian_tilde_ai`` is ALREADY a dense (n_act_orb * n_in_a) matrix,
    and the beta-bisection it replaces calls ``projection_step2`` -- which
    builds and diagonalises the full bordered matrix of that same size -- on
    EVERY iteration.  So the loop pays many dense eigendecompositions to
    approach, iteratively and with a hard case it cannot handle, the answer that
    one ``more_sorensen_dense`` returns exactly.

    Contrast the microiteration site, where the Hessian is matrix-free and the
    same solve would cost n Hessian-vector products plus an O(n^3)
    eigendecomposition -- impractical at c10h8's n = 5789, which is why it is
    only the rare hard-case FALLBACK there and capped by
    QED_TRS_FALLBACK_DENSE_MAX.

    Returns ``(step, hard_case)`` to match the bisection's contract.  hard_case
    is reported as 0: the caller uses it only to force step acceptance
    (``... or hard_case == 2``), and an exact solve needs no such override.
    """
    step, lam, kind = more_sorensen_dense(
        hessian_tilde_ai, gradient_tilde_ai, trust_radius)
    print("QED_INTERNAL_EXACT_TRS: %s solve, lambda %.6e, |step| %.6e, "
          "radius %.6e" % (kind, lam, np.linalg.norm(step), trust_radius),
          flush=True)
    return step, 0


def solve_lstrs_internal(ctx, dim00, gradient_tilde_ai, hessian_tilde_ai, hessian_diagonal, trust_radius, w9, first_component, second_component, hard_case, beta, step):
    """The LSTRS beta-bisection from ``internal_optimization3``.

    MOVED VERBATIM out of helper_PFCI.py, the same mechanical treatment as
    ``solve_lstrs_microiteration``: ``self.`` -> ``ctx.``, inherited names
    promoted to parameters, ``return step, hard_case``.  Gated bit-exactly by
    ``validation/lstrs_refactor_gate.sh``, whose ``i*`` cases exist precisely
    because NOTHING ELSE IN THE GATE REACHES THIS FUNCTION -- it runs only when
    ``macroiteration > 0 and n_in_a > 0`` (helper_PFCI.py:5790), and the other
    decks put every electron in the active space.

    Parameter list chosen the same way as in the microiteration move: the
    syntactically-free names, plus those assigned here as well but bound
    unconditionally by the caller before the block (``hard_case`` at method
    level, ``beta`` and ``step``).  Names bound only inside the SIBLING branch
    (``delta``, ``t1``, ``t2``, ``x_square`` ...) are deliberately NOT passed --
    they are not live at the call site, and passing them raises
    UnboundLocalError in the CALLER.
    """
    from helper_PFCI import (
        _QED_LSTRS_EPS_HC, _QED_LSTRS_HARD_PATIENCE, _QED_LSTRS_NORM_TOL,
        _QED_TRS_HARD_FALLBACK, _stats_begin, _stats_end, _QED_LSTRS_MAX_ITER,
    )
    _more_sorensen_dense = more_sorensen_dense

    mu9 = np.zeros(dim00)
    beta0 = 0
    both_roots_normalization = 0
    v1 = 0
    v2 = 0
    u1 = np.zeros(ctx.n_act_orb * ctx.n_in_a)
    u2 = np.zeros(ctx.n_act_orb * ctx.n_in_a)
    x1 = np.zeros(ctx.n_act_orb * ctx.n_in_a)
    x2 = np.zeros(ctx.n_act_orb * ctx.n_in_a)
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
        mu9[:] = ctx.projection_step2(
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
        #    Q = ctx.Davidson_augmented_hessian_solve3(ctx.U2, A_tilde2, G, G1, ctx.reduced_hessian_diagonal,
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
        # QED_LSTRS_HARD_PATIENCE: bisecting cannot escape a genuine hard
        # case -- on the dumped subproblem the Davidson returned
        # BIT-IDENTICAL eigenvalues for the last ten iterations while
        # alpha kept halving, so the loop only ended when the bracket
        # collapsed after 40 of them.  Stop early and let the
        # hard-case fallback below take the step.
        _hard_spin = 0
        while (
            (aa1 <= epsilon_v * bb1)
            and (aa2 <= epsilon_v * bb2)
            and (_QED_LSTRS_HARD_PATIENCE <= 0
                 or _hard_spin < _QED_LSTRS_HARD_PATIENCE)
            and np.abs(alpha_u - alpha_l)
            > 1e-8 * max(np.abs(alpha_u), np.abs(alpha_l))
        ):
            _hard_spin += 1
            alpha_u = alpha
            alpha = (alpha_l + alpha_u) / 2
            w9[:, :] = 0.0
            mu9[:] = ctx.projection_step2(
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
            # Both roots unnormalizable: a genuine hard case, and
            # this used to be a bare exit() (see the
            # QED_TRS_HARD_FALLBACK note).  Unlike the
            # microiteration site the Hessian here is ALREADY
            # dense, so the exact solve costs no Hessian-vector
            # products at all -- projection_step2 is already
            # diagonalising a matrix this size every iteration.
            if not _QED_TRS_HARD_FALLBACK:
                exit()
            step, _lam_hc, _kind_hc = _more_sorensen_dense(
                hessian_tilde_ai, gradient_tilde_ai, trust_radius)
            print("QED_TRS_HARD_FALLBACK [internal]: both LSTRS "
              "roots unnormalizable (genuine hard case) -> "
              "dense More-Sorensen [%s]; |step| = %.6e, "
              "radius = %.6e"
              % (_kind_hc, np.linalg.norm(step), trust_radius),
              flush=True)
            hard_case = 5
            break
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
        psi_tilde = 0.5 * ctx.internal_optimization_predicted_energy(
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
                        * ctx.internal_optimization_predicted_energy(
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
        if count10 == _QED_LSTRS_MAX_ITER:
            # 2026-09-26: the beta bisection hit its iteration cap (QED_LSTRS_MAX_ITER, default 50).  Nothing bound `step` on this exit: the inline
            # original (helper_PFCI microiteration_optimization6 / internal_optimization3, before d0c7dee) silently reused
            # the previous iteration's step; the extracted function raised UnboundLocalError (seen once, in the
            # composition search, c006).  Take the exact trust-region step, as for the both-roots-unnormalizable case.
            if not _QED_TRS_HARD_FALLBACK:
                exit()
            step, _lam_hc, _kind_hc = _more_sorensen_dense(hessian_tilde_ai, gradient_tilde_ai, trust_radius)
            print("QED_TRS_HARD_FALLBACK [internal, iteration cap] -> dense More-Sorensen [%s]; |step| = %.6e, radius = %.6e"
                  % (_kind_hc, np.linalg.norm(step), trust_radius), flush=True)
            hard_case = 5
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
            # temp_vector = np.zeros(ctx.index_map_size)
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
        Q = np.zeros((1, ctx.n_act_orb * ctx.n_in_a))
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
    return step, hard_case
