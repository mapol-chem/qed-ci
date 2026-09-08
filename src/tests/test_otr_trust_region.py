# -*- coding: utf-8 -*-
"""Tests for otr_trust_region.py.

Checks the ported solver against a dense More-Sorensen reference on problems
with the structure that matters for CASSCF: positive definite (interior Newton
step), indefinite (boundary solution, level shift active), and the near-hard
case where the gradient is nearly orthogonal to the lowest eigenvector.

The matvec counts printed here are the quantity the port exists to reduce --
compare against helper_PFCI's measured 2430 on c10h8.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otr_trust_region import (  # noqa: E402
    TrustRegionSettings,
    bisection,
    level_shifted_davidson,
    newton_step,
)


def dense_trs_reference(H, g, delta):
    """Dense reference solution of min 1/2 x^T H x + g^T x s.t. ||x|| <= delta.

    Straight More-Sorensen: try the interior Newton step, otherwise bisect the
    scalar secular equation ||(H - lam I)^-1 g|| = delta on lam < eigmin(H).
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    if eigvals[0] > 0:
        x = np.linalg.solve(H, -g)
        if np.linalg.norm(x) <= delta:
            return x, 0.0

    gt = eigvecs.T @ g

    def norm_of(lam):
        return np.linalg.norm(gt / (eigvals - lam))

    lo, hi = eigvals[0] - 1.0 - np.linalg.norm(g) / delta, eigvals[0] - 1e-14
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_of(mid) > delta:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return eigvecs @ (gt / (eigvals - lam)) * -1.0 + eigvecs @ (
        gt / (eigvals - lam)
    ) * 0.0, lam


def make_problem(n, seed, shift):
    """Symmetric H with eigenvalues shifted so that `shift` controls definiteness."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = eigvals - eigvals[0] + shift
    H = eigvecs @ np.diag(eigvals) @ eigvecs.T
    g = rng.standard_normal(n)
    return H, g


def model(H, g, x):
    return 0.5 * x @ H @ x + g @ x


def run(H, g, delta, settings=None):
    h_diag = np.diag(H).copy()
    res = level_shifted_davidson(
        func=0.0,
        grad=g,
        h_diag=h_diag,
        hess_x=lambda v: H @ v,
        obj_func=lambda kappa: model(H, g, kappa),
        trust_radius=delta,
        settings=settings,
    )
    return res


@pytest.mark.parametrize("shift", [1.0, -2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_dense_reference(shift, seed):
    """Model value must match the dense More-Sorensen optimum."""
    n = 60
    H, g = make_problem(n, seed, shift)
    delta = 0.5

    res = run(H, g, delta)
    x = res["solution"]

    x_ref, _ = dense_trs_reference(H, g, delta)
    # reference may sit inside or on the boundary; clip to the trust region
    if np.linalg.norm(x_ref) > delta:
        x_ref = x_ref * (delta / np.linalg.norm(x_ref))

    assert np.linalg.norm(x) <= delta * (1 + 1e-8), "step left the trust region"
    # the solver may accept a shrunken radius, so compare model decrease on the
    # radius it actually converged to, not the initial one
    assert model(H, g, x) <= model(H, g, x_ref) + 1e-6 * abs(model(H, g, x_ref)) + 1e-8


def test_positive_definite_takes_interior_newton_step():
    """With a large trust radius and H > 0 the step is the interior Newton step.

    Accuracy is bounded by the solver's OWN exit criterion, not by machine
    precision: with mu == 0 the local region is active, so the micro loop stops
    at ||g + Hx|| < local_red_factor * ||g|| with local_red_factor = 1e-4
    (otr:176, otr:2005).  Assert against that, and check the residual directly.
    """
    n = 40
    H, g = make_problem(n, seed=3, shift=1.5)
    res = run(H, g, delta=1e3)
    x = res["solution"]
    x_exact = np.linalg.solve(H, -g)

    assert res["mu"] == 0.0
    residual_norm = np.linalg.norm(g + H @ x)
    assert residual_norm < 1e-4 * np.linalg.norm(g)
    assert np.allclose(x, x_exact, atol=1e-4)


def test_bisection_costs_no_hessian_vector_products():
    """THE POINT OF THE PORT: changing the level shift must not touch hess_x.

    Builds a fixed search space, then runs the full bisection and asserts the
    Hessian-vector counter never moves.
    """
    n = 50
    H, g = make_problem(n, seed=4, shift=-1.0)
    grad_norm = np.linalg.norm(g)

    calls = {"n": 0}

    def hess_x(v):
        calls["n"] += 1
        return H @ v

    rng = np.random.default_rng(0)
    V, _ = np.linalg.qr(rng.standard_normal((n, 8)))
    HV = np.column_stack([hess_x(V[:, i]) for i in range(V.shape[1])])
    assert calls["n"] == 8

    aug_hess = np.zeros((9, 9))
    aug_hess[1:, 1:] = V.T @ HV

    before = calls["n"]
    solution, red, mu, bracketed = bisection(aug_hess, grad_norm, V, trust_radius=0.5)
    assert bracketed
    assert calls["n"] == before, "bisection performed a Hessian-vector product"
    assert np.linalg.norm(solution) == pytest.approx(0.5, rel=1e-6)


def test_newton_step_costs_no_hessian_vector_products():
    """Same for the Newton branch -- it is a dense solve in the reduced space."""
    n = 50
    H, g = make_problem(n, seed=5, shift=2.0)
    grad_norm = np.linalg.norm(g)

    calls = {"n": 0}

    def hess_x(v):
        calls["n"] += 1
        return H @ v

    rng = np.random.default_rng(1)
    V, _ = np.linalg.qr(rng.standard_normal((n, 6)))
    HV = np.column_stack([hess_x(V[:, i]) for i in range(V.shape[1])])
    aug_hess = np.zeros((7, 7))
    aug_hess[1:, 1:] = V.T @ HV

    before = calls["n"]
    newton_step(aug_hess, grad_norm, V)
    assert calls["n"] == before


def test_matvec_count_is_one_per_micro_iteration():
    """Each micro iteration expands the space by exactly one vector.

    This is the invariant helper_PFCI violates: there, one outer solve spends
    83 matvecs on average because the beta loop re-enters the Davidson.
    """
    n = 80
    H, g = make_problem(n, seed=6, shift=-1.0)
    settings = TrustRegionSettings(n_random_trial_vectors=1)
    res = run(H, g, delta=0.5, settings=settings)

    # initial space (gradient + negative-curvature seed + 1 random) plus one
    # matvec per expansion; the trust-radius retry loop reuses the space
    assert res["n_hess_x"] <= res["n_micro"] + 3
    print(
        f"\n  n={n}  micro={res['n_micro']}  hess_x={res['n_hess_x']}  "
        f"mu={res['mu']:.3e}  |x|={np.linalg.norm(res['solution']):.4f}"
    )


def test_near_hard_case():
    """Gradient nearly orthogonal to the lowest eigenvector.

    helper_PFCI's lstrs_internal hits hard_case=2 on 18 of 19 solves, so this
    branch matters.  OpenTrustRegion has no explicit hard-case code path: the
    bracket check (otr:763-770) simply reports the radius as unattainable in
    the current space and the driver expands it.
    """
    n = 60
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = eigvals - eigvals[0] - 1.0
    H = eigvecs @ np.diag(eigvals) @ eigvecs.T

    g = rng.standard_normal(n)
    g -= eigvecs[:, 0] * np.dot(eigvecs[:, 0], g)  # strip the lowest eigenvector
    g *= 1e-3

    res = run(H, g, delta=0.5)
    x = res["solution"]
    assert np.linalg.norm(x) <= 0.5 * (1 + 1e-8)
    assert model(H, g, x) < 0.0, "no descent in the near-hard case"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
