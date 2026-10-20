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

SENSITIVITY -- WHICH LEGS ACTUALLY CATCH THE DEFECT.  Measured by re-running
the whole module under QED_PROJ_ALL=1, which restores the old projector.  NINE
of the eighteen parameterised legs fail, i.e. actually guard something, and all
four systems contribute one gradient and one coupling leg at unequal weights:

    gradients  h2o_c1  5,2,1  FAILS      couplings  h2o_c1  5,2,1   FAILS
    gradients  h2o_c1b 5,2,1  FAILS      couplings  h2o_c1  28,2,2  FAILS
    gradients  ch2o_c1 5,2,1  FAILS      couplings  h2o_c1b 5,2,1   FAILS
    gradients  c2h4_c1 3,1    FAILS      couplings  ch2o_c1 5,2,1   FAILS
                                         couplings  c2h4_c1 3,1     FAILS

    every equal-weight leg      passes -- controls, (W_n - W_m) = 0, MUST pass
    gradients h2o_c1 28,2,2     passes -- 5.3e-08 off: genuinely INSENSITIVE

**The 28,2,2 GRADIENT leg does not catch the defect and its passing proves
nothing about the projector.**  At 0.875/0.0625/0.0625 the average is close
enough to state-specific that <c(m)|H~|c(n)> is tiny for the wide-spread pairs.
It is kept because it pins the VALUE at a third weighting, and because the
28,2,2 COUPLING leg IS sensitive -- but do not read it as coverage.

If you add a system or a weighting, RE-RUN THE NULL TEST and record what it
moves.  A test that cannot fail for the reason it exists creates false
confidence, which is worse than no test.

c2h4_c1 ALSO PINS THE CONVERGENCE KNOBS.  It is the one system here that needs
non-default settings (see SYSTEMS["c2h4_c1"]["env"]): at the shipped defaults
two absolute backstops inside microiteration_optimization6 -- a gradient cutoff
at 1e-7 and an energy floor at 1e-10 -- stop the inner orbital optimisation
early, the macro gradient floors at ~9.8e-08 instead of reaching ~8e-10, the
individual state energies drift by 5.3e-07 (equal and opposite, average exact)
and the gradients move by 4e-06.  If someone re-hardcodes either backstop, this
system stops matching Molpro and these four legs fail.
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

# Geometries must match the Molpro decks EXACTLY (frame included: no_reorient
# /nocom here, orient,noorient there).  A rotated frame makes gradient and
# coupling VECTORS silently incomparable.
#
# `env` carries per-system convergence overrides.  c2h4_c1 needs them: at the
# SHIPPED defaults our SA-CASSCF cannot converge it, because two absolute
# backstops inside microiteration_optimization6 -- a gradient cutoff at 1e-7
# and an energy floor at 1e-10 -- stop the inner orbital optimisation before
# the outer loop's target is reached.  With those lowered it converges to
# |g| ~ 8e-10 and agrees with Molpro to 4.3e-08 on gradients and 1.2e-07 on
# couplings; with them at their defaults it floors at |g| ~ 9.8e-08 and the
# individual state energies drift by 5.3e-07, moving the gradients by 4e-06.
# The other systems converge at the defaults and set no env, so they keep
# testing the SHIPPED configuration.
SYSTEMS = {
    "h2o_c1": dict(basis="6-31g", nact_orbs=5, nact_els=6, roots=3, env={}, geom="""
0 1
   O            0.000000000000     0.000000000000    -0.068516219320
   H            0.000000000000    -0.790689573744     0.543701060715
   H            0.134217000000     0.831145000000     0.601233000000
symmetry c1
no_reorient
nocom
"""),
    "h2o_c1b": dict(basis="6-31g", nact_orbs=5, nact_els=6, roots=3, env={}, geom="""
0 1
   O            0.000000000000     0.000000000000    -0.068516219320
   H            0.000000000000    -0.850000000000     0.600000000000
   H            0.134217000000     0.831145000000     0.601233000000
symmetry c1
no_reorient
nocom
"""),
    "ch2o_c1": dict(basis="6-31g", nact_orbs=5, nact_els=6, roots=3, env={}, geom="""
0 1
   C            0.000000000000     0.000000000000    -0.522500000000
   O            0.000000000000     0.000000000000     0.677500000000
   H            0.150000000000     0.937000000000    -1.062500000000
   H            0.000000000000    -0.937000000000    -1.112500000000
symmetry c1
no_reorient
nocom
"""),
    "c2h4_c1": dict(basis="cc-pvdz", nact_orbs=7, nact_els=4, roots=2,
                    env={"QED_MACRO_ENERGY_CONV": "0",
                         "QED_GRAD_CONV": "1e-9",
                         "QED_MICRO_GRAD_FLOOR": "1e-12",
                         "QED_MICRO_ENERGY_FLOOR": "1e-14"},
                    geom="""
0 1
   C       -0.67874677       0.00000002      -0.00000002
   C        0.67874683      -0.00000004       0.00000001
   H       -1.19058330       0.92302210       0.13000000
   H       -1.24058332      -0.92302209       0.00000001
   H        1.24058327       0.92302208      -0.00000000
   H        1.24058328      -0.92302208      -0.00000001
symmetry c1
no_reorient
nocom
"""),
}

CASES = [(s, w) for s in MOLPRO for w in MOLPRO[s]]

# Gradients agree at ~1e-07 and couplings at ~7e-07; 5e-06 leaves room for
# thread-count and BLAS variation without admitting the 1e-03 defect, which is
# three orders clear of it.
GTOL = 5e-6

_DRIVER = r"""
import numpy as np, psi4, sys
sys.path.insert(0, "{src}")
from nuclear_grad import nuclear_grad

psi4.core.be_quiet()
geom = '''{geom}'''
options = dict(basis="{basis}", scf_type="pk", e_convergence=1e-12,
               d_convergence=1e-12)
cavity = dict(omega_value=0.0, lambda_vector=np.array([0.0, 0.0, 0.0]),
              ci_level="cas", ignore_coupling=False, number_of_photons=0,
              natural_orbitals=False, photon_number_basis=False,
              canonical_mos=False, coherent_state_basis=True,
              spin_adaptation="singlet", davidson_roots={roots},
              davidson_threshold={cithr}, davidson_maxdim=30,
              davidson_maxiter=200, davidson_indim=10, test_mode=False,
              nact_orbs={nact_orbs}, nact_els={nact_els},
              casscf_weight=np.array([{weights}]))
psi4.set_options(options)
psi4.core.set_output_file("psi4.out", False)
calc = nuclear_grad(geom, options, cavity)
for i in range({roots}):
    print("### GRADIENT OF STATE %d" % i, flush=True)
    calc.compute_grad(i, i)
for (i, j) in [(a, b) for a in range({roots}) for b in range(a + 1, {roots})]:
    print("### COUPLING STATES %d-%d" % (i, j), flush=True)
    calc.compute_grad(i, j, deriv_type="full")
import json
print("STRUCTURED_PROPERTIES " + json.dumps(calc.result.summary()["properties"]))
"""


def _run(system, weights, proj_all, tmp_path):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    sd = SYSTEMS[system]
    env.update(sd["env"])
    if proj_all is not None:
        env["QED_PROJ_ALL"] = proj_all
    script = _DRIVER.format(src=_SRC, geom=sd["geom"], basis=sd["basis"],
                            roots=sd["roots"], nact_orbs=sd["nact_orbs"],
                            nact_els=sd["nact_els"],
                            cithr="1e-12" if sd["env"] else "1e-9",
                            weights=weights.replace(",", ", "))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=3600,
                          cwd=str(tmp_path), env=env)


def _blocks(stdout):
    """-> ({state: grad}, {(i,j): nac}); both (natom, 3), Hartree/Bohr."""
    import json
    properties = json.loads(next(line.removeprefix("STRUCTURED_PROPERTIES ")
                                 for line in stdout.splitlines()
                                 if line.startswith("STRUCTURED_PROPERTIES ")))
    grads, nacs = {}, {}
    for prop in properties.values():
        if 'states' not in prop:
            continue
        i, j = prop['states']
        if i == j:
            grads[i] = np.asarray(prop['value'])
        else:
            nacs[i, j] = np.asarray(prop['value'])
    return grads, nacs


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """One subprocess per (system, weighting); gradients AND couplings from each."""
    out = {}
    for system, w in CASES:
        d = tmp_path_factory.mktemp(system + w.replace(",", ""))
        r = _run(system, w, None, d)
        assert r.returncode == 0, (
            f"{system} weights {w} did not run:\n{r.stdout[-3000:]}")
        out[(system, w)] = _blocks(r.stdout)
    return out


@pytest.mark.parametrize("system,weights", CASES)
def test_gradients_match_molpro(system, weights, results):
    grads, _ = results[(system, weights)]
    ref = MOLPRO[system][weights]["grad"]
    for state, want in ref.items():
        got = grads.get(state)
        assert got is not None, f"no gradient printed for state {state}"
        d = np.abs(got - np.array(want)).max()
        assert d < GTOL, (f"{system} weights {weights} state {state}: "
                          f"max|diff| {d:.3e}\nours\n{got}\nmolpro\n{np.array(want)}")


@pytest.mark.parametrize("system,weights", CASES)
def test_couplings_match_molpro(system, weights, results):
    """Molpro's SA-MC NACME is the NAC WITH the CSF term, NOT scaled by dE.

    A derivative coupling carries the arbitrary relative phase of the two CI
    vectors, so it matches only up to an overall sign per pair -- which can
    differ between codes and between runs.  A global flip is convention, not
    disagreement, so the comparison minimises over both signs.
    """
    _, nacs = results[(system, weights)]
    ref = MOLPRO[system][weights]["nac"]
    for pair, want in ref.items():
        got = nacs.get(pair)
        assert got is not None, f"no coupling printed for pair {pair}"
        want = np.array(want)
        d = min(np.abs(sg * got - want).max() for sg in (1, -1))
        assert d < GTOL, (f"{system} weights {weights} pair {pair}: "
                          f"max|diff| {d:.3e}\nours\n{got}\nmolpro\n{want}")


def test_project_out_all_reintroduces_the_defect(tmp_path):
    """Pin the bug itself, so a revert of the default cannot pass silently.

    With QED_PROJ_ALL=1 the unequal-weight gradient MUST disagree with Molpro by
    ~1e-03.  If this test starts failing, the all-states projection is no longer
    reachable and the guard above has become untestable -- update it rather than
    deleting it.
    """
    r = _run("h2o_c1", "5,2,1", "1", tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    grads, _ = _blocks(r.stdout)
    ref = MOLPRO["h2o_c1"]["5,2,1"]["grad"]
    worst = max(np.abs(grads[s] - np.array(ref[s])).max() for s in ref)
    assert worst > 1e-4, (
        f"expected the all-states projection to be visibly wrong, got {worst:.3e}")


def test_equal_weight_leg_is_insensitive_to_the_projector(tmp_path):
    """The control: at equal weights (W_n - W_m) = 0, so the choice is a no-op.

    This is what makes the unequal-weight disagreement attributable to the
    weights rather than to the projector being wrong in general.
    """
    r = _run("h2o_c1", "1,1,1", "1", tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    grads, _ = _blocks(r.stdout)
    ref = MOLPRO["h2o_c1"]["1,1,1"]["grad"]
    worst = max(np.abs(grads[s] - np.array(ref[s])).max() for s in ref)
    assert worst < GTOL, f"equal-weight leg should be projector-independent, got {worst:.3e}"
