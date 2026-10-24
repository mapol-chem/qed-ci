"""Analytic vs NUMERICAL gradients with the CAVITY ON and UNEQUAL weights.

WHY THIS EXISTS.  Every Molpro comparison in this suite is necessarily
CAVITY-OFF -- Molpro has no cavity -- so the QED terms in the gradient are not
covered by any of them.  Central differences on the converged state energy are
the only available reference for those terms, and they check the whole chain:
the QED contributions, the state-average weights, and the orbital and CI
response together.

WHAT IT VALIDATES THAT test_unequal_weight_sa_vs_molpro.py CANNOT
    * omega and lambda_vector nonzero, one photon
    * unequal weights ON TOP of that

TWO THINGS DECIDE WHETHER THE NUMERICAL SIDE IS USABLE, and both are why this
module looks the way it does.

1. CONVERGENCE.  A central difference divides by 2*delta, here 3.8e-04 bohr, so
   it AMPLIFIES energy noise by ~2600x.  At the shipped microiteration floors
   the per-state energies carry ~2e-09 of convergence noise and the per-state
   numerical gradient is only good to ~5e-06 (state 2: 3.6e-05).  With the
   floors lowered it is 1.1e-07 / 1.0e-07 / 7.4e-07 -- a 48x improvement from
   convergence alone, on identical physics.  So this module sets them, and in
   doing so also pins them: re-hardcode either backstop and these numbers get
   an order of magnitude worse.

2. STATE ORDERING.  A per-state numerical gradient assigns states by ENERGY
   ORDER at each displaced geometry.  Near a conical intersection a displacement
   can swap which physical state is "state 0", the difference then straddles two
   surfaces, and the result is meaningless -- a defect of the NUMERICAL
   reference, not of the analytic gradient.  E_SA = sum_n W_n E_n is invariant
   under relabelling WITHIN the averaged set, so the state-average leg stays
   valid through the intersection and is tested separately and more tightly.

The full tool, with the ethylene FC and conical-intersection geometries from
the developer's c2h4_fc.py / c2h4_ci.py, is
validation/numerical_gradient_check.py.  Those systems are aug-cc-pVDZ and far
too slow for a suite; the water case here is the same test at a size that runs.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")

# Components tested, out of 3N.  Each costs two converged CASSCF points, so
# this is the runtime dial.
NCOMP = 2

TIGHT = {
    "QED_MACRO_ENERGY_CONV": "0",
    "QED_GRAD_CONV": "1e-9",
    "QED_MICRO_GRAD_FLOOR": "1e-12",
    "QED_MICRO_ENERGY_FLOOR": "1e-14",
}


def _run(weights, ncomp, tmp_path, extra_env=None):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable,
           os.path.join(_ROOT, "validation", "numerical_gradient_check.py"),
           "--system", "h2o_cav", "--weights", weights,
           "--components", str(ncomp), "--tol", "1e-4"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(tmp_path), env=env)


def _worsts(stdout):
    """-> {section name: worst |analytic - numerical|}"""
    out, name = {}, None
    for line in stdout.splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            name = line.strip("= ").strip()
        elif "worst |analytic - numerical|" in line and name:
            out[name] = float(line.split("=")[-1].split()[0])
    return out


@pytest.fixture(scope="module")
def unequal(tmp_path_factory):
    r = _run("5,2,1", NCOMP, tmp_path_factory.mktemp("uw"))
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    w = _worsts(r.stdout)
    assert w, "no comparison table parsed:\n" + r.stdout[-3000:]
    return w


def test_state_average_gradient_matches_finite_difference(unequal):
    """The state-average leg -- the one that stays valid through a CI.

    Tighter tolerance than the per-state legs on purpose: E_SA is stationary at
    the CASSCF solution, so it is converged far better than the individual
    energies and its finite difference is correspondingly cleaner.
    """
    key = next(k for k in unequal if k.startswith("STATE-AVERAGE"))
    assert unequal[key] < 1e-6, f"state-average gradient off by {unequal[key]:.3e}"


@pytest.mark.parametrize("state", [0, 1, 2])
def test_per_state_gradient_matches_finite_difference(state, unequal):
    """Cavity ON, weights 5,2,1 -- the combination nothing else here covers."""
    key = f"STATE {state} GRADIENT"
    assert key in unequal, f"{key} missing from the report"
    assert unequal[key] < 5e-6, (
        f"state {state} gradient off by {unequal[key]:.3e}; if this regressed "
        f"check the microiteration floors first -- at the shipped defaults the "
        f"same case is only good to 5e-06, and to 3.6e-05 for state 2")


def test_convergence_floors_are_what_make_the_per_state_legs_work(tmp_path):
    """Pin the knobs: at the SHIPPED floors the same comparison is ~an order worse.

    This is the evidence for setting them above, and it fails if someone makes
    the floors unreachable again -- in which case the per-state numerical
    gradient stops being a usable reference at this displacement.
    """
    loose = {"QED_MACRO_ENERGY_CONV": "1e-10", "QED_GRAD_CONV": "1e-6",
             "QED_MICRO_GRAD_FLOOR": "1e-7", "QED_MICRO_ENERGY_FLOOR": "1e-10"}
    # NCOMP, not 1.  With one component this samples only O0-X, where the
    # degradation happens to be 2.2e-08 (state 0) and 1.6e-07 (state 2) -- both
    # UNDER the threshold below, so the test failed intermittently depending on
    # run-to-run noise.  The effect lives in the Y and Z components
    # (3.7e-06 .. 3.6e-05).  A test whose signal depends on which component it
    # happens to sample is not measuring what it claims to.
    r = _run("5,2,1", NCOMP, tmp_path, extra_env=loose)
    assert r.returncode in (0, 1), r.stdout[-3000:]
    w = _worsts(r.stdout)
    per_state = [v for k, v in w.items() if k.startswith("STATE ")]
    assert per_state, "no per-state sections parsed"
    assert max(per_state) > 1e-6, (
        "expected the shipped floors to visibly degrade the finite-difference "
        f"comparison, got {max(per_state):.3e} -- if the defaults were changed, "
        "update this test rather than deleting it")
