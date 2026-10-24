"""Unit tests for the QED_DEGEN_CANON degenerate-orbital gauge fix.

Motivation, in one line: MgH+ CAS(8,12) run three times identically gives
33/33/33 macroiterations at OMP_NUM_THREADS=1 (bit-identical energies) but
6/28/26 at 4 threads, because threaded MKL returns a different -- equally
valid -- gauge inside the degenerate pi pair, and the optimizer's trajectory
(unlike the energy) depends on that gauge.

The property that actually matters is in
test_canonicalize_is_invariant_to_incoming_gauge: the transform must map ANY
gauge of a cluster onto the SAME orbitals, not merely onto some convention.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from helper_PFCI import (  # noqa: E402
    _canonicalize_degenerate,
    _degenerate_clusters,
    _gauge_tiebreaker,
    _pin_orbital_signs,
)


# ---------------------------------------------------------------- clustering

def test_clusters_group_only_near_degenerate_orbitals():
    eps = [0.0, 0.0, 1.0, 2.0, 2.0]
    assert _degenerate_clusters(eps, [], 1e-6) == [(0, 2), (3, 5)]


def test_singletons_are_omitted():
    assert _degenerate_clusters([0.0, 1.0, 2.0], [], 1e-6) == []


def test_cluster_never_crosses_a_block_boundary():
    # Four exactly degenerate orbitals straddling the inactive|active edge at 2.
    # Mixing across it would change the inactive SPACE and thus the energy.
    eps = [0.0, 0.0, 0.0, 0.0]
    assert _degenerate_clusters(eps, [2], 1e-6) == [(0, 2), (2, 4)]


def test_boundary_splits_even_when_it_isolates_an_orbital():
    eps = [0.0, 0.0, 0.0]
    # Edge at 1 leaves a singleton on the left, which is then dropped.
    assert _degenerate_clusters(eps, [1], 1e-6) == [(1, 3)]


def test_single_linkage_chains_pairwise_close_orbitals():
    tol = 1e-6
    eps = [0.0, 0.6e-6, 1.2e-6]  # ends differ by more than tol, chain links them
    assert _degenerate_clusters(eps, [], tol) == [(0, 3)]


def test_gap_exactly_at_tol_is_not_degenerate():
    # Strict '>' means a gap of exactly tol still clusters; guard the intent.
    assert _degenerate_clusters([0.0, 1e-6], [], 1e-6) == [(0, 2)]
    assert _degenerate_clusters([0.0, 1.000001e-6], [], 1e-6) == []


# ------------------------------------------------------- the core invariance

def _random_orthonormal(n, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q


def test_canonicalize_is_invariant_to_incoming_gauge():
    """THE property: any gauge of a degenerate cluster maps to the same result.

    This is what makes the fix work across thread counts -- it is not a
    convention applied to whatever came in, it erases what came in.
    """
    n = 8
    C = _random_orthonormal(n, seed=11)
    T = _gauge_tiebreaker(n)
    eps = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    ref = _canonicalize_degenerate(C, T, eps, [], 1e-6)

    for seed in (1, 2, 3):
        # An arbitrary rotation INSIDE the degenerate cluster only.
        Q = np.eye(n)
        Q[0:3, 0:3] = _random_orthonormal(3, seed)
        got = _canonicalize_degenerate(C @ Q, T, eps, [], 1e-6)
        # Columns are fixed only up to sign by eigh; pin before comparing.
        assert np.allclose(_pin_orbital_signs(got), _pin_orbital_signs(ref),
                           atol=1e-10)


def test_canonicalize_preserves_the_cluster_subspace():
    n = 6
    C = _random_orthonormal(n, seed=5)
    eps = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    out = _canonicalize_degenerate(C, _gauge_tiebreaker(n), eps, [], 1e-6)

    # Same span => the projector onto the cluster is unchanged.
    P_in = C[:, 0:3] @ C[:, 0:3].T
    P_out = out[:, 0:3] @ out[:, 0:3].T
    assert np.allclose(P_in, P_out, atol=1e-12)


def test_canonicalize_preserves_orthonormality():
    n = 7
    C = _random_orthonormal(n, seed=9)
    eps = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    out = _canonicalize_degenerate(C, _gauge_tiebreaker(n), eps, [], 1e-6)
    assert np.allclose(out.T @ out, np.eye(n), atol=1e-12)


def test_non_degenerate_orbitals_are_left_alone():
    n = 5
    C = _random_orthonormal(n, seed=7)
    eps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = _canonicalize_degenerate(C, _gauge_tiebreaker(n), eps, [], 1e-6)
    assert np.array_equal(out, C)


def test_orbitals_outside_a_cluster_are_untouched():
    n = 6
    C = _random_orthonormal(n, seed=13)
    eps = np.array([0.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    out = _canonicalize_degenerate(C, _gauge_tiebreaker(n), eps, [], 1e-6)
    assert np.array_equal(out[:, 2:], C[:, 2:])
    assert not np.allclose(out[:, 0:2], C[:, 0:2])


def test_block_boundary_blocks_the_rotation_entirely():
    """A cluster split by a boundary must not mix across it."""
    n = 4
    C = _random_orthonormal(n, seed=17)
    eps = np.array([0.0, 0.0, 0.0, 0.0])
    out = _canonicalize_degenerate(C, _gauge_tiebreaker(n), eps, [2], 1e-6)
    # Each half keeps its own span.
    for sl in (slice(0, 2), slice(2, 4)):
        P_in = C[:, sl] @ C[:, sl].T
        P_out = out[:, sl] @ out[:, sl].T
        assert np.allclose(P_in, P_out, atol=1e-12)


# ------------------------------------------------------------ the tiebreaker

def test_tiebreaker_splits_a_symmetry_degenerate_pair():
    """diag(1..n) must not inherit the symmetry that caused the degeneracy.

    Model a pi_x/pi_y pair as two orbitals on disjoint AO sets related by the
    permutation that swaps those AOs; the tiebreaker has to distinguish them.
    """
    n = 4
    C = np.zeros((n, 2))
    C[0, 0] = 1.0  # "pi_x" on AO 0
    C[1, 1] = 1.0  # "pi_y" on AO 1
    T = _gauge_tiebreaker(n)
    w = np.linalg.eigvalsh(C.T @ T @ C)
    assert abs(w[1] - w[0]) > 1e-8


def test_tiebreaker_is_symmetric_and_deterministic():
    T1, T2 = _gauge_tiebreaker(6), _gauge_tiebreaker(6)
    assert np.array_equal(T1, T2)
    assert np.array_equal(T1, T1.T)


# ------------------------------------------------------------------- signing

def test_sign_pinning_makes_largest_coefficient_positive():
    C = np.array([[0.3, -0.9], [-0.8, 0.1]])
    out = _pin_orbital_signs(C)
    for j in range(out.shape[1]):
        k = int(np.argmax(np.abs(out[:, j])))
        assert out[k, j] > 0.0


def test_sign_pinning_is_idempotent():
    C = _random_orthonormal(5, seed=21)
    once = _pin_orbital_signs(C)
    assert np.array_equal(once, _pin_orbital_signs(once))


def test_sign_pinning_resolves_an_exact_tie_deterministically():
    """Symmetry-equivalent atoms give exactly tied coefficients differing only
    by threading noise; rounding must collapse the tie to the lowest index."""
    a = np.array([[0.5], [0.5 + 1e-15]])
    b = np.array([[0.5 + 1e-15], [0.5]])
    # Both must pick the SAME column index to test the sign of.
    assert np.array_equal(np.sign(_pin_orbital_signs(a)),
                          np.sign(_pin_orbital_signs(b)))


def test_inputs_are_not_mutated():
    C = _random_orthonormal(4, seed=23)
    before = C.copy()
    _pin_orbital_signs(C)
    _canonicalize_degenerate(C, _gauge_tiebreaker(4),
                             np.array([0.0, 0.0, 1.0, 2.0]), [], 1e-6)
    assert np.array_equal(C, before)


# ---------------------------------------------------------------- knob state

def test_knob_defaults_to_off():
    """The whole mechanism must be dormant unless explicitly enabled."""
    import helper_PFCI

    if "QED_DEGEN_CANON" in os.environ:
        pytest.skip("QED_DEGEN_CANON set in the environment")
    assert helper_PFCI._QED_DEGEN_CANON == 0
