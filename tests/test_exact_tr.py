"""Unit tests for _more_sorensen_dense, the QED_EXACT_TR subproblem solver.

The solver is exercised directly (it is a pure function of H, g, delta), so
these run without psi4 and without building a CASSCF object.

Reference facts these pin down, all measured on the MgH+ CAS(8,12) main-site
dumps (see the QED_EXACT_TR doc block in helper_PFCI.py):
  - the shipped GLTR step is 9.3-49.6% above the exact model value on the 13
    boundary-active subproblems, and exactly equal on the 2 convex interior ones
  - an all-positive Hessian DIAGONAL does not imply a positive definite Hessian
"""

import numpy as np
import pytest

from helper_PFCI import _more_sorensen_dense


def _model(H, g, s):
    return float(g @ s + 0.5 * s @ (H @ s))


def _kkt_residual(H, g, s, lam):
    n = len(g)
    return np.linalg.norm((H + lam * np.eye(n)) @ s + g) / max(1.0, np.linalg.norm(g))


def test_interior_newton_step_when_positive_definite_and_inside():
    H = np.diag([1.0, 2.0, 3.0])
    g = np.array([0.1, 0.1, 0.1])
    s, lam, kind = _more_sorensen_dense(H, g, delta=10.0)
    assert kind == "interior"
    assert lam == 0.0
    np.testing.assert_allclose(s, -np.linalg.solve(H, g), rtol=1e-12)


def test_boundary_solution_is_on_the_boundary_and_satisfies_kkt():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((12, 12))
    H = 0.5 * (A + A.T)
    g = rng.standard_normal(12)
    delta = 0.3
    s, lam, kind = _more_sorensen_dense(H, g, delta)
    assert kind in ("boundary", "hard_case")
    assert np.linalg.norm(s) == pytest.approx(delta, rel=1e-8)
    assert lam >= max(0.0, -np.linalg.eigvalsh(H)[0]) - 1e-10
    assert _kkt_residual(H, g, s, lam) < 1e-9


def test_hard_case_gradient_orthogonal_to_lowest_eigenvector():
    # H indefinite, g living entirely in the positive-curvature subspace: the
    # textbook hard case, and the configuration a g-seeded Krylov method cannot
    # represent because v_min never enters the Krylov space.
    lam = np.array([-1.0, 0.5, 2.0, 3.0])
    V = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 4)))[0]
    H = V @ np.diag(lam) @ V.T
    g = V[:, 1] * 1e-3  # exactly orthogonal to v_min = V[:, 0]
    delta = 1.0
    s, l, kind = _more_sorensen_dense(H, g, delta)
    assert kind == "hard_case"
    assert np.linalg.norm(s) == pytest.approx(delta, rel=1e-8)
    # the step must pick up a real component along the lowest eigenvector
    assert abs(s @ V[:, 0]) > 0.5 * delta
    # and it must beat the step that ignores that direction
    s_no_vmin = -np.linalg.lstsq(H + 1.0 * np.eye(4), g, rcond=None)[0]
    if np.linalg.norm(s_no_vmin) > delta:
        s_no_vmin *= delta / np.linalg.norm(s_no_vmin)
    assert _model(H, g, s) < _model(H, g, s_no_vmin)


def test_krylov_from_g_cannot_see_vmin_but_exact_solve_can():
    """The blind spot QED_EXACT_TR exists to sidestep, in miniature."""
    lam = np.array([-1.0, 0.5, 2.0, 3.0])
    V = np.linalg.qr(np.random.default_rng(5).standard_normal((4, 4)))[0]
    H = V @ np.diag(lam) @ V.T
    g = V[:, 1] * 1e-3
    # full Krylov space K(H, g): g is an exact eigenvector, so it never grows
    q = g / np.linalg.norm(g)
    K = [q]
    for _ in range(4):
        w = H @ K[-1]
        for k in K:
            w -= (k @ w) * k
        if np.linalg.norm(w) < 1e-12:
            break
        K.append(w / np.linalg.norm(w))
    Km = np.array(K).T
    assert abs(Km.T @ V[:, 0]).max() < 1e-8  # Krylov is blind to v_min
    s, _, kind = _more_sorensen_dense(H, g, delta=1.0)
    assert kind == "hard_case" and abs(s @ V[:, 0]) > 0.5


def test_positive_diagonal_does_not_imply_positive_definite():
    # This is why n_negative == 0 (a DIAGONAL test) can route an indefinite
    # subproblem to GLTR, which has no hard-case branch.
    H = np.array([[1.0, 3.0], [3.0, 1.0]])
    assert (np.diag(H) > 0).all()
    assert np.linalg.eigvalsh(H)[0] < 0
    s, lam, kind = _more_sorensen_dense(H, np.array([1e-9, 1e-9]), delta=1.0)
    assert kind in ("boundary", "hard_case")
    assert np.linalg.norm(s) == pytest.approx(1.0, rel=1e-8)


def test_zero_gradient_at_indefinite_point_follows_negative_curvature():
    H = np.diag([-2.0, 1.0])
    s, lam, kind = _more_sorensen_dense(H, np.zeros(2), delta=0.7)
    assert kind == "hard_case"
    assert np.linalg.norm(s) == pytest.approx(0.7, rel=1e-12)
    assert _model(H, np.zeros(2), s) < 0.0


def test_zero_gradient_at_convex_point_is_the_zero_step():
    s, lam, kind = _more_sorensen_dense(np.diag([1.0, 2.0]), np.zeros(2), delta=0.7)
    assert kind == "interior"
    np.testing.assert_allclose(s, np.zeros(2), atol=0)


@pytest.mark.parametrize("seed", range(8))
def test_never_worse_than_the_scaled_cauchy_point(seed):
    """The exact solve must dominate the steepest-descent step, always."""
    rng = np.random.default_rng(seed)
    n = 10
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    g = rng.standard_normal(n)
    delta = 0.5
    s, _, _ = _more_sorensen_dense(H, g, delta)
    assert np.linalg.norm(s) <= delta * (1 + 1e-9)
    cauchy = -g / np.linalg.norm(g) * delta
    assert _model(H, g, s) <= _model(H, g, cauchy) + 1e-12
