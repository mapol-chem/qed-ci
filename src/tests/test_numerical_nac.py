"""Derivative couplings against a NUMERICAL reference, by wavefunction overlap.

A NAC cannot be obtained from energies, so this is the one check in the suite
that validates a coupling without another quantum-chemistry package:

    d_ij ~ [ <Psi_i(x)|Psi_j(x+h)> - <Psi_i(x)|Psi_j(x-h)> ] / (2h)

evaluated with AO overlaps across two basis-set centres, MO overlaps from the
CASSCF orbitals at each geometry, and determinant overlaps factorised into
alpha and beta string overlaps.  See validation/numerical_nac_check.py.

WHY IT IS WORTH HAVING ALONGSIDE THE MOLPRO TESTS.  Those pin the coupling
against another code, which is the stronger statement where it applies -- but
it applies only where Molpro can follow us, so it stops at cavity-off
SA-CASSCF and it dies with the licence.  This one is self-contained: it needs
nothing but psi4, so it keeps working on systems and Hamiltonians no external
reference covers.

Because the AO basis moves with the nuclei, the finite difference returns the
FULL coupling -- WITH the CSF contribution, i.e. Molpro's "SA-MC NACME" and
pyscf's kernel(state=(i,j)), which is our
compute_grad(i, j, deriv_type="full") "NAC (with CSF)" block.

The tool fixes the two things that would otherwise make this meaningless: the
arbitrary per-geometry CI phase, and state ordering -- states are matched to
the reference by MAXIMUM OVERLAP rather than by energy, which is also what lets
the method work where energy ordering swaps under displacement.

CAVITY-OFF, deliberately.  In the coherent-state basis the photon states are
built on <d_HF> frozen at the reference geometry, so at a displaced geometry
the photon basis is a different basis and <p(x)|p'(x')> is not delta_pp'.  That
needs a photon overlap too and is not attempted.  NOTE this is a limit of the
NUMERICAL REFERENCE, not of the analytic implementation: the Lagrangian carries
the QED-HF orbital response explicitly through its kappa_ai f_ai term and the
HF orthonormality constraint, so <d_HF> being frozen as a VALUE within one
calculation does not mean its response to displacement is missing.

Method note: validation/NUMERICAL_NAC_METHOD.md
"""
import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)

# The comparison is limited by finite-difference truncation, not by the code
# under test: at delta = 1e-3 A the residual is ~2e-04 on a coupling of order 2
# and it falls as delta^2 (see the scaling test below).
DELTA = "1e-3"
TOL = 1e-3

# The floors matter here for the same reason as in the numerical GRADIENT test:
# a finite difference amplifies noise in the quantity being differenced.
TIGHT = {"QED_MACRO_ENERGY_CONV": "0", "QED_GRAD_CONV": "1e-9",
         "QED_MICRO_GRAD_FLOOR": "1e-12", "QED_MICRO_ENERGY_FLOOR": "1e-14"}


def _run(weights, ncomp, delta, tmp_path):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    cmd = [sys.executable,
           os.path.join(_SRC, "validation", "numerical_nac_check.py"),
           "--system", "h2o_c1", "--weights", weights,
           "--components", str(ncomp), "--delta", str(delta),
           "--tol", str(TOL)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(tmp_path), env=env)


def _worst(stdout):
    m = re.search(r"WORST OVERALL\s+([0-9.eE+-]+)", stdout)
    assert m, "no WORST OVERALL line:\n" + stdout[-3000:]
    return float(m.group(1))


def _per_pair(stdout):
    out, pair = {}, None
    for line in stdout.splitlines():
        m = re.match(r"=== NAC (\d)-(\d)", line)
        if m:
            pair = (int(m.group(1)), int(m.group(2)))
        elif line.strip().startswith("worst ") and pair:
            out[pair] = float(line.split()[1])
            pair = None
    return out


@pytest.mark.parametrize("weights", ["1,1,1", "5,2,1"])
def test_numerical_nac_matches_analytic(weights, tmp_path):
    """Equal AND unequal weights: the analytic coupling reproduces the overlap FD.

    The unequal leg is the one nothing else in this suite could check without
    Molpro, and it is the regime where the projection defect lived.
    """
    r = _run(weights, 2, DELTA, tmp_path)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    pairs = _per_pair(r.stdout)
    assert len(pairs) == 3, f"expected 3 state pairs, got {sorted(pairs)}"
    for p, d in pairs.items():
        assert d < TOL, f"weights {weights} pair {p}: {d:.3e}"


def test_finite_difference_error_scales_as_delta_squared(tmp_path):
    """The residual is TRUNCATION, not disagreement -- and this proves it.

    Central differences carry an O(delta^2) error, so halving delta must reduce
    the analytic-vs-numerical difference by ~4x.  Agreement at one step size
    could be luck or a compensating error; the SCALING is what shows the
    numerical estimate is converging to the analytic value.

    Measured 2.590e-04 / 6.479e-05 / 1.619e-05 at delta = 4e-3 / 2e-3 / 1e-3 A,
    i.e. ratios of 4.00 and 4.00.
    """
    errs = []
    for d in ("4e-3", "2e-3", "1e-3"):
        r = _run("1,1,1", 1, d, tmp_path)
        assert r.returncode in (0, 1), r.stdout[-3000:]
        errs.append(_worst(r.stdout))
    assert errs[0] > errs[1] > errs[2], f"error did not fall with delta: {errs}"
    for big, small in zip(errs, errs[1:]):
        ratio = big / small
        assert 3.0 < ratio < 5.0, (
            f"expected ~4x per halving of delta (O(delta^2)), got {ratio:.2f} "
            f"from {errs} -- if this drifts, the finite difference has stopped "
            f"being truncation-limited and something else is wrong")
