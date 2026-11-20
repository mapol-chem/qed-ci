"""Regression tests for the v3 ground-state acceptance rule and basis choice."""
import numpy as np

from finalize_v3 import ground_rule, choose_basis, NUMBER, COHERENT
from ground_rule_v3 import to_frame


def rec(de, dg, scores=(0.99, 0.99), ci=(True, True)):
    return dict(ground_delta_e=de, ground_delta_rdm=dg,
                rungs={str(n): dict(ground_score=s, ci_converged=c) for n, s, c in zip((6, 10), scores, ci)})


def test_rule_thresholds_inclusive_and_both_required():
    assert ground_rule(rec(1e-6, 1e-6)) == (True, 'accepted')
    assert ground_rule(rec(1e-7, 2e-6)) == (False, 'photon_truncation_rdm_only')
    assert ground_rule(rec(2e-6, 1e-7)) == (False, 'photon_truncation_energy')
    assert ground_rule(rec(2e-6, 2e-6)) == (False, 'photon_truncation_energy_and_rdm')


def test_rule_requires_tracking_and_ci():
    assert ground_rule(rec(0, 0, scores=(0.99, 0.3)))[1] == 'ground_tracking_ambiguous'
    assert ground_rule(rec(0, 0, ci=(True, False)))[1] == 'ci_not_converged'


def test_basis_choice_prefers_number_never_mixes():
    assert choose_basis(True, True) == NUMBER
    assert choose_basis(True, False) == NUMBER
    assert choose_basis(False, True) == COHERENT
    assert choose_basis(False, False) == NUMBER


def test_frame_alignment_removes_degenerate_rotation():
    rng = np.random.default_rng(1)
    C = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    theta = 0.7
    R = np.eye(5); R[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    gamma = np.diag([2, 0.6, 0.6, 0.1, 0.0]) + 0.01*np.eye(5)[0][:, None]*np.eye(5)[1]
    gamma = (gamma + gamma.T)/2
    C2 = C @ R                    # same space, rotated degenerate pair
    gamma2 = R.T @ gamma @ R      # same physical density in the rotated frame
    assert np.linalg.norm(gamma2 - gamma) > 1e-3
    assert np.linalg.norm(to_frame(gamma2, C2, C) - gamma) < 1e-12
