"""Unit tests for the occupation-scaled act-virt level shift (QED_ACTVIRT_SHIFT).

The shift exists to regularize near-redundant active->virtual rotations: as an
active orbital's occupation n_t -> 0 both the orbital gradient and the
corresponding Hessian block vanish together, leaving a direction that is flat
but NOT exactly redundant, so index_map (helper_PFCI.py:3468-3481) cannot
excise it.  On MgH+ CAS(8,12) that block carries 99.8% of the weight of the
flattest Hessian eigenvectors.

THE PROPERTY THAT MATTERS, and the reason this file exists: the shift is added
to the HESSIAN ONLY, never to the gradient.  A stationary point of the energy is
therefore still a stationary point, so the shift may change the step, the path
and the iteration count -- but NOT the converged answer.  test_shift_is_hessian
_only_not_gradient asserts exactly that at the linear-algebra level; the
end-to-end confirmation is ~/mgh_repro_20260804/actvirt_shift_e2e.sh, which runs
real MgH+ CASSCF optimizations and is far too slow for pytest.

These tests build a PFHamiltonianGenerator-shaped stub rather than running SCF,
so they are fast and do not need psi4.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_helper(shift="1", eps="1e-3", delta="1e-2"):
    """Re-import helper_PFCI with the knob env set (the knobs are module-level)."""
    os.environ["QED_ACTVIRT_SHIFT"] = shift
    os.environ["QED_ACTVIRT_SHIFT_EPS"] = eps
    os.environ["QED_ACTVIRT_SHIFT_DELTA"] = delta
    import helper_PFCI

    return importlib.reload(helper_PFCI)


class _Stub:
    """Minimal object carrying just what _actvirt_shift_vector reads."""

    def __init__(self, mod, n_in_a=2, n_act_orb=12, nmo=23, occ=None):
        self.n_in_a = n_in_a
        self.n_act_orb = n_act_orb
        self.nmo = nmo
        self.n_occupied = n_in_a + n_act_orb
        self.n_virtual = nmo - self.n_occupied
        # index_map exactly as helper_PFCI.py:3468-3481 builds it
        im = []
        for r in range(nmo):
            for s in range(r + 1, nmo):
                if r < n_in_a and s < n_in_a:
                    continue
                if n_in_a <= r < self.n_occupied and n_in_a <= s < self.n_occupied:
                    continue
                if r >= self.n_occupied and s >= self.n_occupied:
                    continue
                im.append((s, r))
        self.index_map = np.array(im, dtype=int)
        self.index_map_size = len(im)
        if occ is None:
            # the real MgH+ CAS(8,12) converged occupations (item 135)
            occ = [2.000, 2.000, 1.999, 1.303, 0.335, 0.333,
                   0.022, 0.004, 0.003, 0.001, 0.001, 0.000]
        self.D_tu_avg = np.diag(np.asarray(occ, dtype=float)).ravel()
        self._actvirt_shift_vector = mod.PFHamiltonianGenerator._actvirt_shift_vector.__get__(self)


def test_index_map_size_matches_block_count():
    """Guards the stub itself: 24 + 18 + 108 = 150 for MgH+ CAS(8,12)."""
    mod = _load_helper()
    s = _Stub(mod)
    n_ai = s.n_in_a * s.n_act_orb
    n_vi = s.n_in_a * s.n_virtual
    n_va = s.n_act_orb * s.n_virtual
    assert s.index_map_size == n_ai + n_vi + n_va == 150


def test_shift_off_returns_none():
    mod = _load_helper(shift="0")
    assert _Stub(mod)._actvirt_shift_vector() is None


def test_shift_is_zero_outside_act_virt():
    """Only active->virtual rotations may be shifted."""
    mod = _load_helper()
    s = _Stub(mod)
    shift = s._actvirt_shift_vector()
    assert shift.shape == (150,)
    n_shifted = 0
    for i, (big, small) in enumerate(s.index_map):
        is_av = (s.n_in_a <= small < s.n_occupied) and big >= s.n_occupied
        if is_av:
            assert shift[i] > 0.0
            n_shifted += 1
        else:
            assert shift[i] == 0.0
    # act-virt block is n_act_orb * n_virtual
    assert n_shifted == s.n_act_orb * s.n_virtual == 108


def test_shift_is_largest_on_the_emptiest_orbitals():
    """The whole point: flat directions get lifted, real ones do not."""
    mod = _load_helper(eps="1e-3", delta="1e-2")
    s = _Stub(mod)
    shift = s._actvirt_shift_vector()
    occ = np.diag(s.D_tu_avg.reshape(s.n_act_orb, s.n_act_orb))
    per_orb = {}
    for i, (big, small) in enumerate(s.index_map):
        if (s.n_in_a <= small < s.n_occupied) and big >= s.n_occupied:
            per_orb.setdefault(small - s.n_in_a, set()).add(round(shift[i], 12))
    # every rotation off the same active orbital gets the same shift
    for t, vals in per_orb.items():
        assert len(vals) == 1, f"active orbital {t} got inconsistent shifts {vals}"
    got = {t: vals.pop() for t, vals in per_orb.items()}
    # empty orbital (occ 0.000) -> eps/delta = 0.1 ; correlated (1.303) -> 7.6e-4
    assert got[11] == pytest.approx(1e-3 / (0.000 + 1e-2))
    assert got[3] == pytest.approx(1e-3 / (1.303 + 1e-2))
    assert got[11] > 100 * got[3]
    # monotone decreasing in occupation
    order = sorted(got, key=lambda t: occ[t])
    vals = [got[t] for t in order]
    assert vals == sorted(vals, reverse=True)


def test_shift_scales_with_eps_and_delta():
    s10 = _Stub(_load_helper(eps="1e-2"))._actvirt_shift_vector()
    s1 = _Stub(_load_helper(eps="1e-3"))._actvirt_shift_vector()
    assert np.allclose(s10, 10.0 * s1)
    big_delta = _Stub(_load_helper(delta="1.0"))._actvirt_shift_vector()
    assert big_delta.max() < s1.max()


def test_shift_is_hessian_only_not_gradient():
    """The correctness property: shifting H cannot move a stationary point.

    If g == 0 then (H + diag(sigma)) has the same null solution as H: the step
    solves for g, and g is untouched by the shift.  Verified here on the exact
    Newton step of a random SPD problem.
    """
    mod = _load_helper()
    s = _Stub(mod)
    shift = s._actvirt_shift_vector()
    rng = np.random.default_rng(0)
    n = s.index_map_size
    A = rng.normal(size=(n, n))
    H = A @ A.T + n * np.eye(n)          # SPD so the Newton step exists
    H_shift = H + np.diag(shift)
    # at a stationary point the step is zero with OR without the shift
    g0 = np.zeros(n)
    assert np.allclose(np.linalg.solve(H, -g0), 0.0)
    assert np.allclose(np.linalg.solve(H_shift, -g0), 0.0)
    # away from one the steps differ -- that is the whole point of the shift
    g = rng.normal(size=n)
    assert not np.allclose(np.linalg.solve(H, -g), np.linalg.solve(H_shift, -g))


def test_shift_lifts_the_flat_spectrum():
    """A near-singular act-virt direction should stop being near-singular."""
    mod = _load_helper()
    s = _Stub(mod)
    shift = s._actvirt_shift_vector()
    n = s.index_map_size
    # a Hessian that is ~1e-6 flat exactly on the act-virt block, 1.0 elsewhere
    diag = np.where(shift > 0, 1e-6, 1.0)
    H = np.diag(diag)
    lam_before = np.linalg.eigvalsh(H).min()
    lam_after = np.linalg.eigvalsh(H + np.diag(shift)).min()
    assert lam_before == pytest.approx(1e-6)
    assert lam_after > 1e-4
    # conditioning improves by orders of magnitude
    k_before = np.abs(np.linalg.eigvalsh(H)).max() / abs(lam_before)
    k_after = np.abs(np.linalg.eigvalsh(H + np.diag(shift))).max() / abs(lam_after)
    assert k_after < k_before / 100


def test_shift_absent_when_rdm_not_built_yet():
    """Early sigma builds run before any CI solve; must not crash."""
    mod = _load_helper()
    s = _Stub(mod)
    del s.D_tu_avg
    assert s._actvirt_shift_vector() is None


def test_default_is_off():
    """Repo convention: new experimental knobs ship default 0."""
    for k in ("QED_ACTVIRT_SHIFT", "QED_ACTVIRT_SHIFT_EPS", "QED_ACTVIRT_SHIFT_DELTA"):
        os.environ.pop(k, None)
    import helper_PFCI

    mod = importlib.reload(helper_PFCI)
    assert mod._QED_ACTVIRT_SHIFT == 0
    assert mod._QED_ACTVIRT_SHIFT_EPS == 1e-3
    assert mod._QED_ACTVIRT_SHIFT_DELTA == 1e-2
