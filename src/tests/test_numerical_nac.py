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

CAVITY ON IS COVERED TOO, and the photon question has a measured answer.  The
coherent-state origin really does move with geometry (d_exp -0.007361 ->
-0.008198 over 1e-3 A, i.e. gamma/(2h) = -0.45, so a photon-basis derivative
term would be O(0.5) on a coupling of order 1).  Measured against the analytic
coupling: treating the photon states as geometry-independent agrees to 1.5e-05
and converges as h^2, while adding the displaced-basis overlap is off by ~0.9
with either sign.  The two implementations therefore share a convention, and
the likelier reason is a FRAME distinction: the code works in the
coherent-state-transformed frame, so both sides produce the WORKING-FRAME
coupling, while the displaced-basis factor moves toward the LAB-FRAME one.  A
double-counting reading against the QED-HF response does not fit the
magnitudes -- dropping kappa.f entirely is worth 1.6e-02, the photon factor
~8.8e-01.

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


def _run(weights, ncomp, delta, tmp_path, system="h2o_c1", extra_env=None):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable,
           os.path.join(_SRC, "validation", "numerical_nac_check.py"),
           "--system", system, "--weights", weights,
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


@pytest.mark.parametrize("weights", ["1,1,1", "5,2,1"])
def test_cavity_on_nac_matches_analytic(weights, tmp_path):
    """The cavity terms in the COUPLING -- covered by nothing else.

    Every Molpro comparison is cavity-off, so omega and lambda nonzero was
    checked only in the gradient (test_numerical_gradient_cavity.py) until
    this.  The photon states are treated as geometry-independent, which is what
    the measurement in the tool's docstring supports.
    """
    r = _run(weights, 1, DELTA, tmp_path, system="h2o_cav")
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    pairs = _per_pair(r.stdout)
    assert pairs, "no state pairs parsed:\n" + r.stdout[-3000:]
    for p, d in pairs.items():
        assert d < TOL, f"cavity on, weights {weights}, pair {p}: {d:.3e}"


def test_qed_hf_orbital_response_is_required(tmp_path):
    """The sum_ai kappa_ai f_ai term is NECESSARY, not merely present.

    A term that is present and a term that is right are different claims, and
    agreement alone only shows the analytic and numerical sides are consistent.
    This makes it falsifiable: QED_NO_HF_RESPONSE=1 discards the solved CP-HF
    response and leaves everything else intact, so the coupling must stop
    matching a finite-difference reference that contains the CQED-HF response
    BY CONSTRUCTION -- every displaced point re-converges CQED-HF from scratch.

    Measured, cavity on, weights 5,2,1, h = 1e-3 A:
        kappa.f included   1.476e-05
        kappa.f dropped    1.624e-02      -- 1100x worse

    The CP-HF solve only runs when lambda_vector is nonzero, so this is a
    cavity-only statement; with no cavity kappa is zero anyway.
    """
    ok = _run("5,2,1", 1, DELTA, tmp_path, system="h2o_cav")
    assert ok.returncode in (0, 1), ok.stdout[-3000:]
    without = _run("5,2,1", 1, DELTA, tmp_path, system="h2o_cav",
                   extra_env={"QED_NO_HF_RESPONSE": "1"})
    assert without.returncode in (0, 1), without.stdout[-3000:]
    w_ok, w_no = _worst(ok.stdout), _worst(without.stdout)
    assert w_ok < TOL, f"cavity-on coupling should agree, got {w_ok:.3e}"
    assert w_no > 100 * w_ok, (
        f"dropping the QED-HF orbital response barely changed the coupling "
        f"({w_ok:.3e} -> {w_no:.3e}); either the term stopped contributing or "
        f"the diagnostic switch stopped working -- both worth knowing")


def test_photon_basis_term_matches_the_frame_relation(tmp_path):
    """Validate the displaced-basis photon overlap against THEORY, not our code.

    The coherent-state transformation U = exp[z (b+ - b)], z = <d_HF>/sqrt(2w),
    depends on geometry, so the working-frame and lab-frame states have
    different derivative couplings:

        d_lab = d_CS + (dz/dx) <Psi_m|(b+ - b)|Psi_n>

    The right-hand side follows from the CI vectors and d(<d_HF>)/dx alone, so
    this checks the photon implementation without reference to the analytic
    gradient -- the two finite-difference legs must differ by exactly this.

    Measured (h2o_cav, 5,2,1, h = 1e-3 A): 7.4e-05 / 8.8e-07 / 8.3e-08 across
    the three pairs, against a d_lab - d_CS of -0.899 on pair 0-1.  Note the
    two legs differ by MORE than the coupling itself there, which is why the
    frame choice is not cosmetic.
    """
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    r = subprocess.run(
        [sys.executable,
         os.path.join(_SRC, "validation", "numerical_nac_check.py"),
         "--system", "h2o_cav", "--weights", "5,2,1",
         "--frame-decomposition", "--delta", DELTA, "--tol", str(TOL)],
        capture_output=True, text=True, timeout=3600, cwd=str(tmp_path), env=env)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    worst = _worst(r.stdout)
    assert worst < TOL, (
        f"d_lab - d_CS departs from (dz/dx)<m|b+ - b|n> by {worst:.3e}; the "
        f"photon overlap or its contraction is wrong\n{r.stdout[-2000:]}")
    # and the term must be LARGE here, or the test is not exercising anything
    assert "-0.89" in r.stdout or "0.89" in r.stdout, (
        "expected an O(0.9) frame difference on pair 0-1:\n" + r.stdout[-2000:])


def test_nac_sign_convention(tmp_path):
    """Pin the SIGN of the coupling -- which no other test here can see.

    A NAC carries the arbitrary phase of the CI vectors, so any comparison
    between two independent calculations must allow an overall sign, and every
    other check in this suite does.  That makes them all blind to a systematic
    sign error, which is how an inverted energy denominator survived: the
    theory has d^{mn}_CI = h^{mn}/(E^n - E^m) with m = state1, n = state2, and
    nuclear_grad computed h0/(E[state1] - E[state2]).

    --sign-check removes the freedom by using ONE nuclear_grad object both for
    the analytic coupling and as the bra of the finite-difference overlap, so
    both routes share a set of CI vectors and their signs are comparable.
    Measured after the fix: 1.58e-05 / 1.33e-06 / 1.58e-05 across the three
    pairs, agreeing in sign; before it, the analytic was the exact negative.
    """
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    r = subprocess.run(
        [sys.executable,
         os.path.join(_SRC, "validation", "numerical_nac_check.py"),
         "--system", "h2o_c1", "--weights", "5,2,1", "--sign-check",
         "--delta", DELTA, "--tol", str(TOL)],
        capture_output=True, text=True, timeout=3600, cwd=str(tmp_path), env=env)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    worst = _worst(r.stdout)
    assert worst < TOL, (
        f"analytic and finite-difference couplings disagree by {worst:.3e} with "
        f"NO sign freedom allowed.  A value near twice the coupling means the "
        f"energy denominator has been inverted again -- it must be "
        f"E[state2] - E[state1].\n{r.stdout[-2000:]}")


def test_photon_frame_term_gives_the_lab_frame_coupling(tmp_path):
    """QED_PHOTON_FRAME=1 must reproduce the LAB-frame finite difference.

    Off (the default) the analytic coupling is the working-frame one and must
    match the finite difference computed with the photon states treated as
    geometry-independent.  On, it must match the finite difference computed WITH
    the displaced-coherent-state photon overlap.  Both legs are checked with no
    sign freedom, so this pins the magnitude AND the direction of the term.

    Measured (h2o_cav, 5,2,1, h = 1e-3 A, pair 0-1): off gives +0.0108873714
    against +0.0108865795; on gives -0.8883020832 against -0.8882284376.  The
    term is larger than the working-frame coupling itself, so a leg that
    silently stopped contributing would be obvious here.
    """
    def run(env_extra, photon_fd):
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                    "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
        env.update(TIGHT)
        env.update(env_extra)
        cmd = [sys.executable,
               os.path.join(_SRC, "validation", "numerical_nac_check.py"),
               "--system", "h2o_cav", "--weights", "5,2,1", "--sign-check",
               "--delta", DELTA, "--tol", str(TOL)]
        if photon_fd:
            cmd.append("--photon-basis-derivative")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                              cwd=str(tmp_path), env=env)

    off = run({"QED_PHOTON_FRAME": "0"}, photon_fd=False)
    assert off.returncode in (0, 1), off.stdout[-3000:]
    assert _worst(off.stdout) < TOL, (
        "with the frame term OFF the analytic coupling should be the "
        f"working-frame one:\n{off.stdout[-2000:]}")

    on = run({"QED_PHOTON_FRAME": "1"}, photon_fd=True)
    assert on.returncode in (0, 1), on.stdout[-3000:]
    assert _worst(on.stdout) < TOL, (
        "with the frame term ON the analytic coupling should be the LAB-frame "
        f"one, matching the displaced-basis finite difference:\n{on.stdout[-2000:]}")


# Richardson extrapolation, (4 f(h/2) - f(h))/3, removes the O(h^2) truncation
# that otherwise dominates the finite-difference comparison.  With it the
# reference stops being the limiting factor and the tolerance can be ~500x
# tighter than TOL, at the cost of one extra pair of displaced points.
RICH_DELTA = "2e-3"
RICH_TOL = 2e-6


def _run_rich(tmp_path, extra_env=None, photon_fd=False):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "QED_SKIP_GKL": "1",
                "PYTHONPATH": _SRC + os.pathsep + env.get("PYTHONPATH", "")})
    env.update(TIGHT)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable,
           os.path.join(_SRC, "validation", "numerical_nac_check.py"),
           "--system", "h2o_cav", "--weights", "5,2,1", "--sign-check",
           "--richardson", "--delta", RICH_DELTA, "--tol", str(RICH_TOL)]
    if photon_fd:
        cmd.append("--photon-basis-derivative")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(tmp_path), env=env)


@pytest.mark.parametrize("frame", ["working", "lab"])
def test_richardson_extrapolated_coupling(frame, tmp_path):
    """The tightest check available on the cavity-on coupling.

    Plain central differences limit the comparison to ~7e-05 RELATIVE, which is
    truncation in the REFERENCE, not error in the analytic coupling -- the
    residual falls as h^2 (see the scaling test).  Richardson removes that term
    and the agreement drops to ~1e-07 relative, which is where the analytic
    couplings already sit against Molpro (2.5e-07).  So after extrapolation the
    finite difference is no longer the limiting factor.

    That matters most for the LAB-frame leg, which has NO external reference at
    all -- Molpro has no cavity, and no other code computes this quantity in
    this frame.  Richardson is the only thing that pins it at Molpro-grade
    precision.

    Measured, worst over the three pairs (h = 2e-3 A):
        working frame   5.90e-05 -> 9.51e-08
        lab frame       2.94e-04 -> 1.48e-07
    """
    on = frame == "lab"
    r = _run_rich(tmp_path, {"QED_PHOTON_FRAME": "1" if on else "0"}, photon_fd=on)
    assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-2000:]
    worst = _worst(r.stdout)
    assert worst < RICH_TOL, (
        f"{frame}-frame coupling vs Richardson-extrapolated finite difference: "
        f"{worst:.3e}, tol {RICH_TOL:.0e}.  Plain-h agreement is ~1e-04 here, so "
        f"a failure at this tolerance is a real discrepancy, not truncation."
        f"\n{r.stdout[-2000:]}")


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
