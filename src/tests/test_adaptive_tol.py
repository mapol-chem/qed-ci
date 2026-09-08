"""Unit tests for _adaptive_tol_from_rho (QED_ADAPTIVE_TOL).

The rule: solve the subproblem exactly as hard as the model deserves.
|1 - median(rho)| estimates the model's own error; tighten below `tight`,
loosen above `loose`, hold in between, and loosen immediately on any rho < 0.
"""

import numpy as np
import pytest

from helper_PFCI import _adaptive_tol_from_rho as adapt

KW = dict(base=1e-7, lo=1e-9, hi=1e-4, window=5, tight=0.05, loose=0.25, factor=3.0)


def test_empty_history_returns_current():
    assert adapt([], 1e-7, **KW) == pytest.approx(1e-7)


def test_good_model_tightens():
    # rho ~ 1 => model trustworthy => solve harder
    assert adapt([1.0] * 5, 1e-7, **KW) == pytest.approx(1e-7 / 3.0)


def test_bad_model_loosens():
    # |1 - 0.5| = 0.5 >= loose
    assert adapt([0.5] * 5, 1e-7, **KW) == pytest.approx(1e-7 * 3.0)


def test_middle_band_holds():
    # |1 - 0.9| = 0.1, between tight (0.05) and loose (0.25)
    assert adapt([0.9] * 5, 1e-7, **KW) == pytest.approx(1e-7)


def test_negative_rho_loosens_even_if_median_is_perfect():
    # four perfect steps and one that moved the energy the WRONG way
    hist = [1.0, 1.0, 1.0, 1.0, -0.3]
    assert np.median(hist) == 1.0  # median alone would say "tighten"
    assert adapt(hist, 1e-7, **KW) == pytest.approx(1e-7 * 3.0)


def test_bounds_are_respected():
    assert adapt([1.0] * 5, 1e-9, **KW) == pytest.approx(1e-9)      # floor
    assert adapt([0.0] * 5, 1e-4, **KW) == pytest.approx(1e-4)      # ceiling
    # and a current tol already outside the bounds is pulled back in
    assert adapt([0.9] * 5, 1.0, **KW) == pytest.approx(1e-4)
    assert adapt([0.9] * 5, 1e-15, **KW) == pytest.approx(1e-9)


def test_only_the_window_is_used():
    # ancient bad history must not hold the tolerance loose forever
    hist = [-5.0] * 20 + [1.0] * 5
    assert adapt(hist, 1e-7, **KW) == pytest.approx(1e-7 / 3.0)


def test_non_finite_rho_is_ignored_not_treated_as_perfect():
    # predicted_energy2 can underflow near convergence -> inf/nan
    assert adapt([np.inf, np.nan], 1e-7, **KW) == pytest.approx(1e-7)
    # a finite bad value among non-finite ones still governs
    assert adapt([np.nan, 0.4, np.inf], 1e-7, **KW) == pytest.approx(1e-7 * 3.0)


def test_repeated_good_steps_walk_down_to_the_floor_and_stop():
    tol = 1e-7
    for _ in range(20):
        tol = adapt([1.0] * 5, tol, **KW)
    assert tol == pytest.approx(1e-9)


def test_repeated_bad_steps_walk_up_to_the_ceiling_and_stop():
    tol = 1e-7
    for _ in range(20):
        tol = adapt([0.2] * 5, tol, **KW)
    assert tol == pytest.approx(1e-4)


def test_hysteresis_prevents_oscillation():
    """Alternating tighten/loosen must not occur inside the hold band."""
    tol = 1e-7
    seen = []
    for rho in (0.92, 0.88, 0.93, 0.90, 0.91):
        tol = adapt([rho] * 5, tol, **KW)
        seen.append(tol)
    assert all(t == pytest.approx(1e-7) for t in seen)


@pytest.mark.parametrize("rho,expected", [
    (1.00, 1e-7 / 3),   # perfect
    (0.96, 1e-7 / 3),   # inside tight
    (0.95, 1e-7 / 3),   # exactly at tight -> tighten
    (0.80, 1e-7),       # hold band
    (0.75, 1e-7 * 3),   # exactly at loose -> loosen
    (1.30, 1e-7 * 3),   # overshoot is just as untrustworthy as undershoot
])
def test_measured_rho_values_map_to_expected_action(rho, expected):
    assert adapt([rho] * 5, 1e-7, **KW) == pytest.approx(expected)
