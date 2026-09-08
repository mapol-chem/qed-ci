"""Unit tests for QED_EXIT_CONTROL -- the trust-radius action keyed on the
GLTR exit reason.

Same observable as QED_SUBSPACE_TOL, different action.  The tolerance
controller loosens where the boundary is NOT binding, which by its own logic
has nothing to offer where the boundary IS binding -- and on MgH+ CAS(8,12) the
main site is boundary-dominated, which is exactly where the tail pathology
lives.  This knob shrinks the region when the subproblem exits on the boundary
persistently, on the claim that such a step is placed by the constraint rather
than by the model.

Only the SHRINK direction is untried: growth was falsified by QED_TR_MAX
(item 184), more Krylov subspace by QED_EXACT_TR (item 179), and step
acceptance has nothing to key on (zero rejections on MgH+ base runs).
"""

import pytest

from helper_PFCI import _exit_boundary_dominated as boundary_dominated
from helper_PFCI import _exit_control_radius as damp
from helper_PFCI import _exit_last_is_interior as last_interior

KW = dict(window=4, interior_lo=0.25)
DKW = dict(damp=0.7, lo=0.05, window=4, interior_lo=0.25)

I = "Interior (Newton) iter 3"
B = "Boundary iter 7"


# --- the growth gate (mode 2) ---------------------------------------------

def test_last_interior_gates_growth():
    assert last_interior([B, B, I])


def test_last_boundary_does_not_gate():
    assert not last_interior([I, I, B])


def test_empty_history_does_not_gate():
    # Nothing observed => the original 1.2x growth must stand.
    assert not last_interior([])


# --- the damping predicate (mode 1) ---------------------------------------

def test_all_boundary_is_dominated():
    assert boundary_dominated([B] * 4, **KW)


def test_all_interior_is_not_dominated():
    assert not boundary_dominated([I] * 4, **KW)


def test_a_single_boundary_exit_is_not_enough():
    # Ordinary trust-region behaviour; damping on it would fight the algorithm
    # rather than the pathology.
    assert not boundary_dominated([I, I, I, B], **KW)


def test_threshold_is_inclusive_at_the_measured_edge():
    # 1 of 4 interior = 0.25 = the low edge of the measured c10h8/MgH+ gap.
    assert boundary_dominated([B, B, B, I], **KW)


def test_empty_history_is_not_dominated():
    assert not boundary_dominated([], **KW)


def test_window_is_respected():
    # Ancient boundary exits fall out of the window and stop mattering.
    assert not boundary_dominated([B] * 10 + [I] * 4, **KW)


# --- the damping action ---------------------------------------------------

def test_damping_fires_when_dominated():
    assert damp(0.75, [B] * 4, **DKW) == pytest.approx(0.525)


def test_damping_is_a_no_op_when_not_dominated():
    assert damp(0.75, [I] * 4, **DKW) == pytest.approx(0.75)


def test_damping_compounds():
    r = 0.75
    for _ in range(3):
        r = damp(r, [B] * 4, **DKW)
    assert r == pytest.approx(0.75 * 0.7 ** 3)


def test_damping_respects_the_floor():
    assert damp(0.06, [B] * 4, **DKW) == pytest.approx(0.05)
    assert damp(0.05, [B] * 4, **DKW) == pytest.approx(0.05)


def test_damping_self_terminates_once_exits_go_interior():
    # The rule must stop firing on its own -- otherwise it drives the radius to
    # the floor and stalls the outer loop.
    hist = [B] * 4
    r = damp(0.75, hist, **DKW)
    assert r < 0.75
    hist = hist + [I] * 4
    assert damp(r, hist, **DKW) == pytest.approx(r)


# --- the two measured regimes ---------------------------------------------

def test_mgh_main_site_mix_damps():
    # MgH+ CAS(8,12) site "gltr": ~45 boundary / ~10 interior.  This is the
    # regime the knob exists for.
    assert boundary_dominated([B, B, B, B], **KW)
    assert boundary_dominated([I, B, B, B], **KW)


def test_c10h8_main_site_mix_does_not_damp():
    # c10h8 site "gltr": exits B, I, I, I.  Must be left alone -- loosening was
    # measured free there and the radius is not the problem.
    assert not boundary_dominated([B, I, I, I], **KW)
    assert damp(0.5, [B, I, I, I], **DKW) == pytest.approx(0.5)
