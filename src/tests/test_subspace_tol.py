"""Unit tests for _subspace_tol_from_exits (QED_SUBSPACE_TOL).

The rule: LOOSEN WHERE THE TRUST-REGION BOUNDARY IS NOT BINDING.  An
"Interior (Newton)" GLTR exit means the step was placed strictly inside the
trust region, so the exit tolerance bought only linear-solve accuracy; a
"Boundary" exit means the tolerance interacts with where on the boundary the
step lands, and loosening is no longer free.

This is the successor to QED_ADAPTIVE_TOL, which keyed the same tolerance on
rho and lost -- rho is structurally blind to the Krylov subspace restriction.
The exit reason is not.
"""

import pytest

from helper_PFCI import _gltr_exit_is_interior as is_interior
from helper_PFCI import _subspace_tol_from_exits as adapt

# Explicit defaults so the tests do not move if the knob defaults are retuned.
KW = dict(
    base=1e-7,
    lo=1e-7,
    hi=1e-4,
    window=4,
    interior_hi=0.75,
    interior_lo=0.25,
    factor=10.0,
    recent=2,
)

I = "Interior (Newton) iter 3"
B = "Boundary iter 7"


# --- the classifier -------------------------------------------------------

def test_classifier_recognises_the_two_real_statuses():
    assert is_interior(I)
    assert not is_interior(B)


def test_classifier_treats_unknown_exits_as_not_interior():
    # "Failure" (ran out of Lanczos iterations) and "success_zero_gradient"
    # (no solve happened) must never be read as evidence for loosening.
    assert not is_interior("Failure")
    assert not is_interior("success_zero_gradient")
    assert not is_interior(None)


# --- the controller -------------------------------------------------------

def test_empty_history_returns_current():
    # First solve at a site runs at whatever the call site would have used.
    assert adapt([], 1e-7, **KW) == pytest.approx(1e-7)
    assert adapt([], None, **KW) == pytest.approx(1e-7)


def test_all_interior_loosens():
    assert adapt([I] * 4, 1e-7, **KW) == pytest.approx(1e-6)


def test_loosening_compounds_toward_the_ceiling():
    tol = 1e-7
    for _ in range(3):
        tol = adapt([I] * 4, tol, **KW)
    assert tol == pytest.approx(1e-4)


def test_all_boundary_snaps_back():
    assert adapt([B] * 4, 1e-5, **KW) == pytest.approx(1e-6)


def test_middle_band_holds():
    # 2 of 4 interior = 0.50, between interior_lo (0.25) and interior_hi (0.75)
    assert adapt([I, B, I, B], 1e-5, **KW) == pytest.approx(1e-5)


def test_snap_back_threshold_is_inclusive_at_the_measured_edge():
    # 1 of 4 interior = 0.25 = interior_lo exactly.  The documented semantics
    # are "at or below LO -> snap back", and with a window of 4 this exact
    # value is common, so the float slack must not make the test exclusive.
    assert adapt([B, B, B, I], 1e-5, **KW) == pytest.approx(1e-6)


def test_recent_boundary_vetoes_loosening_despite_a_good_window():
    # 3 of 4 interior = 0.75 >= interior_hi, but the newest exit hit the
    # boundary -- hold instead of loosening into it.
    assert adapt([I, I, I, B], 1e-6, **KW) == pytest.approx(1e-6)


def test_veto_can_be_disabled():
    kw = dict(KW, recent=0)
    assert adapt([I, I, I, B], 1e-6, **kw) == pytest.approx(1e-5)


def test_window_is_respected():
    # Ancient boundary exits fall out of the window and stop mattering.
    hist = [B] * 10 + [I] * 4
    assert adapt(hist, 1e-7, **KW) == pytest.approx(1e-6)


# --- bounds ---------------------------------------------------------------

def test_ceiling_is_never_exceeded():
    assert adapt([I] * 4, 1e-4, **KW) == pytest.approx(1e-4)


def test_floor_defaults_to_base_so_it_can_never_tighten_below_it():
    # The whole "tighten the subproblem" family is closed (0 for 5 on MgH+
    # CAS(8,12)); the default design must not be able to re-enter it.
    assert adapt([B] * 4, 1e-7, **KW) == pytest.approx(1e-7)


def test_floor_can_be_lowered_explicitly():
    kw = dict(KW, lo=1e-9)
    assert adapt([B] * 4, 1e-7, **kw) == pytest.approx(1e-8)


def test_a_current_tol_outside_the_bounds_is_clamped():
    assert adapt([I, B, I, B], 1e-2, **KW) == pytest.approx(1e-4)
    assert adapt([I, B, I, B], 1e-12, **KW) == pytest.approx(1e-7)


# --- the two measured regimes it has to separate --------------------------
#
# These are the actual per-site exit mixes from the runs that motivated the
# knob, replayed through the controller.  They are the reason it exists: rho
# could not tell these two systems apart, the exit reason does.

def test_c10h8_main_site_mix_loosens():
    # c10h8 CAS(10,10), site "gltr": 3 interior / 1 boundary.  Loosening was
    # MEASURED to be free there (QED_GLTR_TOL_SCALE=1000 gave a bit-identical
    # trajectory and -15% matvecs).
    assert adapt([B, I, I, I], 1e-7, **KW) == pytest.approx(1e-6)


def test_mgh_main_site_mix_stays_at_base():
    # MgH+ CAS(8,12), site "gltr": ~45 boundary / ~10 interior.  The boundary
    # is doing the work; the controller must not loosen into it.
    hist = ([B] * 9 + [I]) * 5
    tol = 1e-7
    for i in range(len(hist)):
        tol = adapt(hist[: i + 1], tol, **KW)
    assert tol == pytest.approx(1e-7)


def test_mgh_qn_site_mix_loosens():
    # MgH+ CAS(8,12), sites "gltr_qn"/"gltr_qn_bfgs": 100% interior, ZERO
    # boundary exits in every seed inspected -- the trust region is inert
    # there, so the same argument as c10h8 applies.  Documented explicitly
    # because it is what separates QED_SUBSPACE_TOL=2 from =1 on MgH+.
    tol = 1e-7
    for i in range(6):
        tol = adapt([I] * min(i + 1, 4), tol, **KW)
    assert tol == pytest.approx(1e-4)
