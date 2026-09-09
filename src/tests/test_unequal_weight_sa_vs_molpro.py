"""Unequal-weight SA-CASSCF gradients and derivative couplings, against Molpro.

WHAT THIS GUARDS.  ``project_out_all`` orthogonalises a CI vector against EVERY
state-averaged CI vector.  Used in ``solve3``'s z-vector solve that deletes a
real ``(W_n - W_m) t_mn`` contribution, so it is wrong exactly when the weights
are unequal -- by 1.4e-03..7.5e-03 on gradients and 9-25% on couplings.  The
minimal projection (``z^(n).c^(n) = 0`` only, ``QED_PROJ_ALL=0``) is correct in
both regimes.  See nuclear_grad.py's knob comment and
~/mgh_repro_20260804/unequal_weight_sa/PROJECTION_EXPLAINED.md.

WHY A C1 GEOMETRY.  On C2v water ``<c(m)|H~|c(n)>`` vanishes by symmetry, the
term the projection would discard is machine zero, and the defect is invisible.
Every unequal-weight validation before 2026-09-08 used symmetric geometries and
so proved nothing about the general case -- which is why the bug survived them.

WHY EQUAL WEIGHTS IS NOT OPTIONAL.  It is the control: it makes a disagreement
in an unequal leg attributable to the WEIGHTS rather than to the frame, the
basis, the active space or the state ordering.

The runs are SUBPROCESSES because the projection knobs are read at import time.

SENSITIVITY -- WHICH LEGS ACTUALLY CATCH THE DEFECT.  Measured by re-running the
whole module under QED_PROJ_ALL=1, which restores the old projector:

    gradients  1,1,1     passes  -- control, (W_n - W_m) = 0, MUST pass
    gradients  5,2,1     FAILS   -- 1.0e-01 off
    gradients  28,2,2    passes  -- 5.3e-08 off: genuinely INSENSITIVE
    couplings  1,1,1     passes  -- control
    couplings  5,2,1     FAILS
    couplings  28,2,2    FAILS

So three of the six catch it.  **The 28,2,2 GRADIENT leg does not, and its
passing proves nothing about the projector** -- at that weighting the average is
0.875/0.0625/0.0625, close enough to state-specific that <c(m)|H~|c(n)> is tiny
for the wide-spread pairs.  It is kept because it pins the VALUE against Molpro
at a third weighting, and because the 28,2,2 COUPLING leg is sensitive, but do
not read it as coverage of the defect.  If you add a weighting, re-run the null
test above and record what it moves; a test that cannot fail for the reason it
exists creates false confidence.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from molpro_unequal_weight_reference import MOLPRO

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)

WEIGHTINGS = ["1,1,1", "5,2,1", "28,2,2"]

# Gradients agree at ~1e-07 and couplings at ~7e-07; 5e-06 leaves room for
# thread-count and BLAS variation without admitting the 1e-03 defect, which is
# three orders clear of it.
GTOL = 5e-6

_DRIVER = r"""
import numpy as np, psi4, sys
sys.path.insert(0, "{src}")
from nuclear_grad import nuclear_grad

psi4.core.be_quiet()
geom = '''
0 1
   O            0.000000000000     0.000000000000    -0.068516219320
   H            0.000000000000    -0.790689573744     0.543701060715
   H            0.134217000000     0.831145000000     0.601233000000
symmetry c1
no_reorient
nocom
'''
options = dict(basis="6-31g", scf_type="pk", e_convergence=1e-12,
               d_convergence=1e-12)
cavity = dict(omega_value=0.0, lambda_vector=np.array([0.0, 0.0, 0.0]),
              ci_level="cas", ignore_coupling=False, number_of_photons=0,
              natural_orbitals=False, photon_number_basis=False,
              canonical_mos=False, coherent_state_basis=True,
              spin_adaptation="singlet", davidson_roots=3,
              davidson_threshold=1e-9, davidson_maxdim=30,
              davidson_maxiter=200, davidson_indim=10, test_mode=False,
              nact_orbs=5, nact_els=6,
              casscf_weight=np.array([{weights}]))
psi4.set_options(options)
psi4.core.set_output_file("psi4.out", False)
calc = nuclear_grad(geom, options, cavity)
for i in range(3):
    print("### GRADIENT OF STATE %d" % i, flush=True)
    calc.compute_grad(i, i)
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    print("### COUPLING STATES %d-%d" % (i, j), flush=True)
    calc.compute_grad(i, j, deriv_type="full")
"""


def _run(weights, proj_all, tmp_path):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    if proj_all is not None:
        env["QED_PROJ_ALL"] = proj_all
    script = _DRIVER.format(src=_SRC, weights=weights.replace(",", ", "))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=3600,
                          cwd=str(tmp_path), env=env)


def _blocks(stdout):
    """-> ({state: grad}, {(i,j): nac}); both (natom, 3), Hartree/Bohr."""
    import re
    num = re.compile(r"-?\d+\.\d+")
    lines = stdout.splitlines()
    grads, nacs, state, pair = {}, {}, None, None
    for i, l in enumerate(lines):
        m = re.match(r"### GRADIENT OF STATE (\d)", l)
        if m:
            state, pair = int(m.group(1)), None
        m = re.match(r"### COUPLING STATES (\d)-(\d)", l)
        if m:
            pair, state = (int(m.group(1)), int(m.group(2))), None
        t = l.strip()
        want = (state is not None and t == "Total gradient") or (
            pair is not None and t.startswith("NAC ") and not t.startswith("NAC * dE"))
        if not want:
            continue
        rows = []
        for j in range(i + 1, min(i + 12, len(lines))):
            if lines[j].startswith("Row "):
                rows.append([float(x) for x in num.findall(lines[j].split(":", 1)[1])[:3]])
                if len(rows) == 3:
                    break
        if len(rows) != 3:
            continue
        if state is not None:
            grads.setdefault(state, np.array(rows))
        else:
            nacs.setdefault(pair, np.array(rows))
    return grads, nacs


@pytest.fixture(scope="module")
def results(request, tmp_path_factory):
    """One subprocess per weighting; both gradients and couplings from each."""
    out = {}
    for w in WEIGHTINGS:
        d = tmp_path_factory.mktemp("w" + w.replace(",", ""))
        r = _run(w, None, d)
        assert r.returncode == 0, f"weights {w} did not run:\n{r.stdout[-3000:]}"
        out[w] = _blocks(r.stdout)
    return out


@pytest.mark.parametrize("weights", WEIGHTINGS)
def test_gradients_match_molpro(weights, results):
    grads, _ = results[weights]
    ref = MOLPRO[weights]["grad"]
    for state, want in ref.items():
        got = grads.get(state)
        assert got is not None, f"no gradient printed for state {state}"
        d = np.abs(got - np.array(want)).max()
        assert d < GTOL, (f"weights {weights} state {state}: max|diff| {d:.3e}\n"
                          f"ours\n{got}\nmolpro\n{np.array(want)}")


@pytest.mark.parametrize("weights", WEIGHTINGS)
def test_couplings_match_molpro(weights, results):
    """Molpro's SA-MC NACME is the NAC WITH the CSF term, NOT scaled by dE.

    A derivative coupling carries the arbitrary relative phase of the two CI
    vectors, so it matches only up to an overall sign per pair -- which can
    differ between codes and between runs.  A global flip is convention, not
    disagreement, so the comparison minimises over both signs.
    """
    _, nacs = results[weights]
    ref = MOLPRO[weights]["nac"]
    for pair, want in ref.items():
        got = nacs.get(pair)
        assert got is not None, f"no coupling printed for pair {pair}"
        want = np.array(want)
        d = min(np.abs(s * got - want).max() for s in (1, -1))
        assert d < GTOL, (f"weights {weights} pair {pair}: max|diff| {d:.3e}\n"
                          f"ours\n{got}\nmolpro\n{want}")


def test_project_out_all_reintroduces_the_defect(tmp_path):
    """Pin the bug itself, so a revert of the default cannot pass silently.

    With QED_PROJ_ALL=1 the unequal-weight gradient MUST disagree with Molpro by
    ~1e-03.  If this test starts failing, the all-states projection is no longer
    reachable and the guard above has become untestable -- update it rather than
    deleting it.
    """
    r = _run("5,2,1", "1", tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    grads, _ = _blocks(r.stdout)
    ref = MOLPRO["5,2,1"]["grad"]
    worst = max(np.abs(grads[s] - np.array(ref[s])).max() for s in ref)
    assert worst > 1e-4, (
        f"expected the all-states projection to be visibly wrong, got {worst:.3e}")


def test_equal_weight_leg_is_insensitive_to_the_projector(tmp_path):
    """The control: at equal weights (W_n - W_m) = 0, so the choice is a no-op.

    This is what makes the unequal-weight disagreement attributable to the
    weights rather than to the projector being wrong in general.
    """
    r = _run("1,1,1", "1", tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    grads, _ = _blocks(r.stdout)
    ref = MOLPRO["1,1,1"]["grad"]
    worst = max(np.abs(grads[s] - np.array(ref[s])).max() for s in ref)
    assert worst < GTOL, f"equal-weight leg should be projector-independent, got {worst:.3e}"
