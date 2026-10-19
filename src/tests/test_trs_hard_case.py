"""Regression tests for the LSTRS hard case that used to call bare ``exit()``.

WHAT THIS GUARDS.  ``microiteration_optimization6`` and
``internal_optimization3`` each ended their beta-bisection with

    if (aa1 <= epsilon_v * bb1) and (aa2 <= epsilon_v * bb2):
        exit()

i.e. an UNIMPLEMENTED branch that raised ``SystemExit(None)`` -- the process
died mid-optimisation with **status 0** and no traceback, which any runner
scoring a return code reads as success.  It is reachable by ordinary last-bit
numerical noise: a mean field differing at 1e-14 decided whether the LiH case
below walked into it or not, while the converged energies agreed at 12 dp.

The first three tests pin the SUBPROBLEM, using a real dump captured from that
failure (``lstrs_hard_case_lih.npz``, written by QED_LSTRS_HARD_DUMP) rather
than synthetic data -- the point being that this is a GENUINE hard case, not a
solver artefact, and that a two-root eigensolve provably cannot resolve it.

The last three run real optimisations end to end, in SUBPROCESSES because the
knobs are read at import time.
"""

import os
import subprocess
import warnings
import sys
import textwrap

import numpy as np
import pytest

from helper_PFCI import _more_sorensen_dense

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_DUMP = os.path.join(_HERE, "lstrs_hard_case_lih.npz")


# --------------------------------------------------------------------------
# The subproblem itself
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dump():
    d = np.load(_DUMP)
    H = np.asarray(d["H"])
    return {
        "H": 0.5 * (H + H.T),
        "g": np.asarray(d["g"]).ravel(),
        "alpha": float(d["alpha"]),
        "delta": float(d["trust_radius"]),
    }


def test_dumped_subproblem_is_a_genuine_hard_case(dump):
    """Degenerate lowest pair AND a gradient orthogonal to both, at machine zero.

    This is what makes it a hard case rather than a misconverged eigensolve, and
    it is why no amount of bisecting on the first eigenvector component can
    succeed: that component is identically zero.
    """
    w, U = np.linalg.eigh(dump["H"])
    gnorm = np.linalg.norm(dump["g"])
    proj = U.T @ dump["g"]

    assert abs(w[1] - w[0]) < 1e-15, "lowest pair is not degenerate"
    assert w[0] < 0.0, "expected an indefinite Hessian"
    for i in (0, 1):
        assert abs(proj[i]) / gnorm < 1e-12, (
            "gradient is not orthogonal to lowest eigenvector %d" % i)
    # ... and NOT orthogonal to everything, or the test would be vacuous.
    assert np.max(np.abs(proj)) / gnorm > 1e-3


def test_normalizable_root_is_the_third_so_two_roots_cannot_find_it(dump):
    """The reason the shipped solver fails: it asks a Davidson for 2 roots.

    In the bordered matrix B(alpha) = [[alpha, g^T], [g, H]] the hard case makes
    the lowest eigenvectors have a zero first component ("unnormalizable").  The
    first eigenvector with a usable first component is the THIRD, so a two-root
    request cannot reach it by construction -- and the whole cluster spans
    ~1e-15, far below any Davidson residual, so "the two lowest" is not even a
    well-posed request.
    """
    H, g, alpha = dump["H"], dump["g"], dump["alpha"]
    n = len(g)
    B = np.zeros((n + 1, n + 1))
    B[0, 0] = alpha
    B[0, 1:] = g
    B[1:, 0] = g
    B[1:, 1:] = H
    lb, VB = np.linalg.eigh(B)

    nu = np.abs(VB[0, :3])
    assert nu[0] < 1e-4 and nu[1] < 1e-4, "expected the two lowest to be hard"
    assert nu[2] > 0.1, "expected the third root to be normalizable"
    assert lb[2] - lb[0] < 1e-12, "expected all three inside one tight cluster"


def test_more_sorensen_solves_the_dumped_hard_case(dump):
    """The fallback returns the BOUNDARY optimum, not the interior point.

    The pseudo-inverse solution alone is interior (the trust region is not
    active at it), and since the null space carries negative curvature the model
    keeps decreasing along it -- so the optimum needs the ``tau * u`` excursion.
    Finding the third root alone would stop at the interior point.
    """
    H, g, delta = dump["H"], dump["g"], dump["delta"]
    s, lam, kind = _more_sorensen_dense(H, g, delta)
    model = lambda x: float(g @ x + 0.5 * x @ (H @ x))

    assert kind == "hard_case"
    assert np.linalg.norm(s) == pytest.approx(delta, rel=1e-8)

    # the interior pseudo-inverse point, for comparison
    w, U = np.linalg.eigh(H)
    deg = np.abs(w - w[0]) <= 1e-10 * max(1.0, float(np.abs(w).max()))
    safe = ~deg
    s_pinv = -(U[:, safe] @ ((U.T @ g)[safe] / (w[safe] - w[0])))
    assert np.linalg.norm(s_pinv) < delta, "expected the pinv point to be interior"
    assert model(s) < model(s_pinv), (
        "the boundary step must beat the interior pseudo-inverse point")


# --------------------------------------------------------------------------
# End to end.  Knobs are read at import time, hence subprocesses.
# --------------------------------------------------------------------------
_LIH_HARD = """
import numpy as np, psi4, sys
sys.path.insert(0, {src!r})
from helper_PFCI import PFHamiltonianGenerator
np.random.seed(0)
psi4.set_memory("2 GB")
psi4.core.set_num_threads(1)
psi4.core.be_quiet()
mol_str = '''
0 1
Li
H 1 1.6
symmetry c1
'''
options_dict = {{"basis": "6-31G", "scf_type": "pk",
                 "e_convergence": 1e-10, "d_convergence": 1e-10}}
cavity_options = {{
    "omega_value": 0.12, "lambda_vector": [0.0, 0.0, 0.05],
    "ci_level": "cas", "ignore_coupling": False, "number_of_photons": 1,
    "natural_orbitals": False, "photon_number_basis": False,
    "canonical_mos": False, "coherent_state_basis": True,
    "davidson_roots": 2, "davidson_threshold": 1e-8, "davidson_maxdim": 20,
    "spin_adaptation": "singlet", "davidson_maxiter": 100,
    "davidson_indim": 8, "test_mode": False,
    "nact_orbs": 6, "nact_els": 4,
}}
psi4.geometry(mol_str)
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print("ROOT0 %.12f" % np.atleast_1d(pf.CIeigs)[0])
"""

_H4_DIMER = """
import numpy as np, psi4, sys
sys.path.insert(0, {src!r})
from helper_PFCI import PFHamiltonianGenerator
np.random.seed(0)
psi4.set_memory("2 GB")
psi4.core.set_num_threads(1)
psi4.core.be_quiet()
mol_str = '''
0 1
H 0.0000000000  0.0000000000 {zin}
H 0.0000000000  0.0000000000 {zout}
H 0.0000000000  0.0000000000 -{zout}
H 0.0000000000  0.0000000000 -{zin}
symmetry c1
no_reorient
nocom
'''
options_dict = {{"basis": "cc-pvtz", "scf_type": "pk",
                 "e_convergence": 1e-11, "d_convergence": 1e-6}}
cavity_options = {{
    "omega_value": 0.10375, "lambda_vector": np.array([0.0, 0.0, 0.01]),
    "ci_level": "cas", "ignore_coupling": False, "number_of_photons": 1,
    "natural_orbitals": False, "photon_number_basis": False,
    "canonical_mos": False, "coherent_state_basis": True,
    "davidson_roots": 1, "davidson_threshold": 1e-6, "davidson_maxdim": 6,
    "davidson_maxiter": 100, "davidson_indim": 3, "test_mode": False,
    "nact_orbs": 4, "nact_els": 4,
}}
psi4.geometry(mol_str)
psi4.set_options(options_dict)
# The deck's own extra psi4 options (h4_100A.py): without these the SA
# reference differs and the run lands on another solution entirely.
psi4.set_options({{"restricted_docc": [0], "active": [4]}})
psi4.set_options({{"num_roots": 4, "avg_states": [0, 1, 2, 3],
                  "avg_weights": [1.0, 1.0, 1.0, 1.0]}})
pf = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)
print("ROOT0 %.12f" % np.atleast_1d(pf.CIeigs)[0])
print("AVGE %.12f" % pf.avg_energy)
"""


# The two separations share one deck.  Bond length 0.7426 A is held fixed;
# only the distance between the two H2 centres changes.
_H4_100A = dict(zin="49.62869866", zout="50.37130134")
_H4_24000A = dict(zin="11999.62869866", zout="12000.37130134")


def _run(script, env_extra, tmp_path, **fmt):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script).format(src=_SRC, **fmt)],
        capture_output=True, text=True, timeout=1800, cwd=str(tmp_path), env=env)


def _root0(out):
    for line in out.splitlines():
        if line.startswith("ROOT0 "):
            return float(line.split()[1])
    return None


def test_lih_cavity_hard_case_completes(tmp_path):
    """The item-371 case: it must reach the answer, not die at macroiteration 0.

    -7.977855449960 is what the pre-regression code, ``QED_SCF_JK=0`` and OTR
    all independently produce, so it is the answer and not merely today's
    output.
    """
    r = _run(_LIH_HARD, {}, tmp_path)
    assert r.returncode == 0, "run did not complete:\n" + r.stdout[-3000:]
    assert _root0(r.stdout) == pytest.approx(-7.977855449960, abs=1e-9)
    # LAST-BIT CANARY, not a correctness assertion (relabelled 2026-09-22).
    #
    # Whether this deck WALKS INTO the Hessian degeneracy is decided by
    # differences of order 1e-14 from ANY source -- a cfunctions.so rebuild,
    # the presence of psi4_dpd_shim.so, an MKL version.  It is therefore a
    # canary for floating-point routing, NOT a measure of whether anything is
    # correct: the energy asserted above is -7.977855449960 either way, and
    # that assertion is the one that tests the code.
    #
    # Raising this to an error made a routing change look like a regression.
    # It is now reported and left to the reader.  If it stops firing, coverage
    # of the fallback has moved elsewhere -- which is worth KNOWING and is not
    # by itself a defect.  The three dumped-subproblem tests above still cover
    # the hard case directly and are pinned to a fixture, so they do not drift.
    if "QED_TRS_HARD_FALLBACK" not in r.stdout:
        warnings.warn(
            "CANARY: this deck no longer walks into the hard case, so the "
            "fallback is not being exercised end-to-end here. Correctness is "
            "unaffected (energy asserted above). Cause is usually a last-bit "
            "change: cfunctions.so rebuild, psi4_dpd_shim.so presence, or MKL "
            "version. See CLAUDE.md.", RuntimeWarning)


def test_lih_cavity_hard_case_exits_silently_without_the_fallback(tmp_path):
    """Documents the defect, so nobody re-introduces it by 'simplifying'.

    With the fallback disabled the old code path is restored: SystemExit(None),
    which means **returncode 0 with no result** -- the failure mode that makes
    this worth a test at all.
    """
    r = _run(_LIH_HARD, {"QED_TRS_HARD_FALLBACK": "0"}, tmp_path)
    # Same canary caveat as above: reaching the old path at all depends on the
    # deck walking into the degeneracy, which last-bit routing decides.
    if _root0(r.stdout) is not None:
        warnings.warn(
            "CANARY: the fallback-disabled path produced a result, so this "
            "deck did not reach the silent-exit defect this test documents. "
            "Last-bit routing, not a correctness change.", RuntimeWarning)
        return
    assert r.returncode == 0, (
        "the old path exits with status 0 -- if this ever becomes non-zero the "
        "silent-success problem is fixed and this test should be updated")


def test_h4_dimer_cavity_converges(tmp_path):
    """Two H2 100 A apart, cc-pVTZ CAS(4,4), 1 photon, coherent-state basis.

    The dipole self-energy goes as (lambda.d)^2/2 and d grows with separation,
    so this is the family where the optimiser is most easily driven into
    degenerate/hard subproblems -- the developer's original report of this class
    of failure was an H4 dimer.

    THE 24000 A VARIANT IS DELIBERATELY NOT USED.  It diverges (1000-macro cap,
    E = +13.66), and per the developer that deck needs a knob to switch off the
    E(T) model and adjust the step so it can be compared directly against E(U) --
    which does not exist yet.  Until it does, the divergence is not a meaningful
    pass/fail signal.

    TWO OBSERVABLES, ONE RUN -- do not confuse them.  ``CIeigs[0]`` is
    -2.268604188986 while the optimiser's reported average/total energy is
    -2.302739404095; the preserved script and its README quote the latter.  Both
    come out of the SAME run and differ by the cavity constants, so a mismatch
    between them is not evidence of two solutions.  This test pins both.
    """
    r = _run(_H4_DIMER, {}, tmp_path, **_H4_100A)
    assert r.returncode == 0, "run did not complete:\n" + r.stdout[-3000:]
    assert _root0(r.stdout) is not None, "no result -- the silent-exit failure mode"
    assert _root0(r.stdout) == pytest.approx(-2.268604188986, abs=1e-8)
    avg = [float(l.split()[1]) for l in r.stdout.splitlines()
           if l.startswith("AVGE ")]
    assert avg and avg[0] == pytest.approx(-2.302739404095, abs=1e-8), (
        "the deck's own reported energy (README: -2.302739) must also match")


def test_h4_dimer_24000A_converges_with_true_energy_step_control(tmp_path):
    """The far-separated dimer, which DIVERGES on the shipping defaults.

    At 24000 A the DSE term (lambda.d)^2/2 is enormous, and the shipping step
    control cannot see the trouble: it scores
    ``ratio = energy_change / predicted_energy2`` where BOTH sides come from the
    same second-order model E(T), so the ratio sits at ~1.0002, the trust radius
    never shrinks and no step is ever rejected.  E(T) is a quadratic form and is
    unbounded below for indefinite G, so the microiteration rides it down: this
    deck runs away to -1.6e6 over 4575 microiterations on the defaults.

    QED_TRUE_ENERGY_RATIO=2 evaluates the TRUE energy E(U) at the trial point
    and accepts only when it DECREASES -- the textbook trust-region
    construction.  With it the same deck converges in 11 macroiterations,
    |g| = 4.6e-07.

    MEASURED ACROSS ALL THE STEP-CONTROL GRANULARITIES (2026-09-04), because
    which one you pick decides whether this deck converges at all:

        setup                     evaluates E(U)        converged  macros
        defaults                  never                 NO         1000 cap
        RATIO=1                   every candidate step  NO         (-1.64e6)
        RATIO=2                   every candidate step  yes        11
        RATIO=2 + REEXPAND=1      + re-expand on accept yes         3
        PROBE=1                   once per macro        NO         (-1.65e6)
        REACTIVE=1                after a macro rises   yes        24

    Three lessons in that table.  MODE 1 IS NOT ENOUGH: it measures E(U) and
    steers the radius with rho_true but leaves the ACCEPTANCE test on the model.
    THE PER-MACRO PROBE IS NOT ENOUGH EITHER: it declines to engage and the run
    diverges -- so "evaluate the true energy periodically" does not characterise
    the fix; enforcing descent on every candidate step does.  And RE-EXPANDING
    ON EACH ACCEPTED STEP is by far the most effective, 3 macroiterations
    against 11, because the shipping model stays expanded at macro entry while
    the iterate walks away from it.

    All three converging setups agree to ~1e-9 and match the 100 A separated
    limit to ~1.6e-9.  This test pins the minimal principled one, RATIO=2.

    ITEM 325(F) IS CLOSED (developer's determination, 2026-09-04).
    h4_true_energy_ab.sh used to warn that under mode 2 the h4 true energy was
    once measured at -1,645,377 Eh -- impossible for a bounded-below
    Hamiltonian -- and blamed the one- and two-electron DSE pieces for not
    rotating consistently.  They do rotate consistently.  The 4.7-order
    cancellation is real and intrinsic (the DSE is 2.571e4 Eh at this
    separation) but it SUMS CORRECTLY:
        act_1e  +102841.647471953038
        act_2e  -102845.339017098842   ->  -2.266268882144
    and E(U) stays in [-8.7, -2.27] for a whole mode-2 run.  The -1.6e6 numbers
    appear ONLY in legs that DIVERGE -- the per-macro probe lands on -1,645,290,
    the same magnitude -- so that value is a divergence attractor, not a
    computation bug.  Three independent strategies converge here and agree with
    the 100 A limit to ~1.6e-9, which a mis-rotation could not do.

    STILL OPEN, AND SEPARATE: when the E(T) model itself fails.  Not this bug.

    THE PHYSICS CHECK IS THE POINT.  Two H2 24000 A apart are non-interacting,
    so this must reproduce the 100 A answer; it does, to 1.2e-09.  That is a far
    stronger statement than "it stopped moving".

    The hard-case fallback fires ZERO times here -- this divergence is step
    control, a completely separate mechanism from the LSTRS hard case the rest
    of this file is about.
    """
    r = _run(_H4_DIMER, {"QED_TRUE_ENERGY_RATIO": "2"}, tmp_path, **_H4_24000A)
    assert r.returncode == 0, "run did not complete:\n" + r.stdout[-3000:]
    avg = [float(l.split()[1]) for l in r.stdout.splitlines()
           if l.startswith("AVGE ")]
    assert avg, "no result -- the run did not reach the end"
    assert avg[0] == pytest.approx(-2.302739405311, abs=1e-8)
    # non-interacting: must agree with the 100 A separation
    assert avg[0] == pytest.approx(-2.302739404095, abs=1e-8)


# --------------------------------------------------------------------------
# QED_INTERNAL_EXACT_TRS -- the internal site solved by eigendecomposition
# --------------------------------------------------------------------------
def _run_internal(env_extra, tmp_path):
    """Run validation/internal_opt_case.py, the only driver that reaches
    internal_optimization3 (it needs n_in_a > 0; every other deck here puts all
    electrons in the active space)."""
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONWARNINGS": "ignore::SyntaxWarning",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, os.path.join(_SRC, "validation", "internal_opt_case.py")],
        capture_output=True, text=True, timeout=1800, cwd=str(tmp_path), env=env)


def _root(out, k=0):
    for line in out.splitlines():
        if line.startswith("ROOT%d " % k):
            return float(line.split()[1])
    return None


@pytest.mark.parametrize("lz,om", [("0.05", "0.12"), ("0.01", "0.5")])
def test_internal_exact_trs_matches_the_bisection(lz, om, tmp_path):
    """Solving the internal subproblem exactly must not change the answer.

    ``internal_optimization3`` holds ``hessian_tilde_ai`` as a DENSE matrix and
    its beta-bisection calls ``projection_step2`` -- which builds and
    diagonalises the full bordered matrix of that size -- on every iteration.
    So the loop pays repeated dense eigendecompositions to approach, iteratively,
    what one ``more_sorensen_dense`` returns exactly.  Measured here it replaces
    7 bisection iterations with a single call, and reports
    ``interior solve, lambda 0, |step| 6.5e-04, radius 5.0`` -- i.e. those seven
    iterations were rediscovering an unconstrained Newton step lying four orders
    of magnitude inside the trust radius.

    The knob is DEFAULT OFF because it changes the steps taken (energies then
    agree to the convergence threshold, not bitwise), not because it is suspect.
    """
    off = _run_internal({"QED_INTERNAL_EXACT_TRS": "0"}, tmp_path)
    on = _run_internal({"QED_INTERNAL_EXACT_TRS": "1"}, tmp_path)
    for r, tag in ((off, "off"), (on, "on")):
        assert r.returncode == 0, "%s leg failed:\n%s" % (tag, r.stdout[-3000:])
    assert "QED_INTERNAL_EXACT_TRS" in on.stdout, (
        "the exact solver never engaged -- this case is supposed to reach "
        "internal_optimization3, and if it stops doing so the test is vacuous")
    assert "QED_INTERNAL_EXACT_TRS" not in off.stdout, "knob leaked when off"
    for k in (0, 1):
        assert _root(on.stdout, k) == pytest.approx(_root(off.stdout, k), abs=1e-9)


# --------------------------------------------------------------------------
# QED_LSTRS_MAX_ITER -- the beta-bisection's iteration cap (2026-09-26)
# --------------------------------------------------------------------------
def test_lstrs_iteration_cap_takes_a_real_step(tmp_path):
    """The cap exit of the LSTRS beta-bisection must return a step, at both sites.

    Until 2026-09-26 the loop's ``if count10 == 50: break`` left ``step`` unbound.
    Inline in helper_PFCI (before d0c7dee) that silently reused the previous
    iteration's step; the extracted ``lstrs_trust_region`` functions raised
    ``UnboundLocalError`` instead (seen once in ~500 production single points).
    The cap now takes the exact trust-region step, as the hard case does.

    ``QED_LSTRS_MAX_ITER=1`` forces every bisection onto the cap.  On this deck
    that is 3 microiteration exits and 1 internal exit; the pre-fix solver, on
    the same forced path, fails with the original UnboundLocalError.  The capped
    run must complete and reach the same roots, because the exact step is a valid
    trust-region step.
    """
    capped = _run_internal({"QED_LSTRS_MAX_ITER": "1"}, tmp_path)
    ref = _run_internal({}, tmp_path)
    for r, tag in ((capped, "capped"), (ref, "default")):
        assert r.returncode == 0, "%s leg failed:\n%s" % (tag, r.stdout[-3000:] + r.stderr[-3000:])
    assert "[microiteration, iteration cap]" in capped.stdout, (
        "the microiteration cap exit never engaged -- the test would be vacuous")
    assert "[internal, iteration cap]" in capped.stdout, (
        "the internal cap exit never engaged -- the test would be vacuous")
    assert "iteration cap]" not in ref.stdout, "the default cap engaged on this deck"
    for k in (0, 1):
        assert _root(capped.stdout, k) == pytest.approx(_root(ref.stdout, k), abs=1e-9)
