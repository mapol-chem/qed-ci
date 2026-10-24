"""CIS is solved by full diagonalization -- proof that this changed no number.

BACKGROUND.  Until 2026-09-04, `ci_level="cis"` with the default
`full_diagonalization=False` was solved by the Python `Davidson` in
helper_PFCI.py.  That looked like a direct/matrix-free solver but was not: for
CIS, `Davidson`'s sigma build fell through to the dense
`S = np.einsum("ip,pq->iq", Q, H)` against `self.H_PF` -- the very matrix that
had already been built and stored a few lines earlier.  So it was a subspace
iteration converging to what `np.linalg.eigh` returns exactly, on a matrix that
was resident either way.  (The `build_sigma` callback it was handed was never
invoked; the call is commented out at what was helper_PFCI.py:8964.)

CIS is now routed to `eigh` at helper_PFCI.py:4754 and the Davidson branch is
gone, with `Davidson` retired to legacy_python_ci.py.

WHAT THIS SCRIPT CHECKS.  The eigenvalues the suite's two CIS decks assert on,
against values captured from the DAVIDSON path while it still existed.  Both
decks are MgH+/cc-pVDZ, matching tests/test_helperPFCI.py.

    python validation/cis_full_diagonalization_check.py

Expected: every root inside 1e-11 of the reference, which is 100x looser than
the 2.3e-13 the two paths actually differed by when measured side by side, and
still far tighter than the 1e-8 Davidson threshold those numbers were converged
to.  A failure here means the CIS solve moved, not that the tolerance is tight.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import psi4                                          # noqa: E402
from helper_PFCI import PFHamiltonianGenerator       # noqa: E402

psi4.core.be_quiet()

MOL = """
Mg
H 1 2.2
symmetry c1
1 1
"""
OPTIONS = {"basis": "cc-pVDZ", "scf_type": "pk",
           "e_convergence": 1e-10, "d_convergence": 1e-10}

# Captured 2026-09-04 from the Davidson path, immediately before it was removed.
REFERENCE = {
    "no_cavity": (0.0, np.array([0, 0, 0]), [
        -199.86395910419120, -199.86395910419117, -199.83928938990221,
        -199.83928938990218, -199.69011028330070, -199.69011028330067,
        -199.68108668752052, -199.68108668752049]),
    "with_cavity": (4.75 / 27.211, np.array([0, 0, 0.0125]), [
        -199.86358254607811, -199.83887696384295, -199.69775974760245,
        -199.68071186854633, -199.68071186854627, -199.68066368599128,
        -199.66486557818814, -199.63031837082858]),
}

TOL = 1e-11
failures = 0

for label, (omega, lam, ref) in REFERENCE.items():
    psi4.geometry(MOL)
    psi4.set_options(OPTIONS)
    cav = {"omega_value": omega, "lambda_vector": lam, "ci_level": "cis",
           "davidson_roots": 8, "davidson_threshold": 1e-8}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pf = PFHamiltonianGenerator(MOL, OPTIONS, cav)

    eigs = np.asarray(pf.CIeigs)
    vecs = np.asarray(pf.CIvecs)
    print("\n=== %s ===" % label)
    print("  CIeigs %s, CIvecs %s  (full spectrum, as expected)"
          % (eigs.shape, vecs.shape))
    if eigs.shape[0] < len(ref):
        print("  *** FAIL: fewer roots than the reference has")
        failures += 1
        continue
    print("  root %24s %24s %12s" % ("now (eigh)", "was (Davidson)", "|diff|"))
    for i, r in enumerate(ref):
        d = abs(eigs[i] - r)
        bad = d > TOL
        failures += bad
        print("   %2d  %24.14f %24.14f %12.3e%s"
              % (i, eigs[i], r, d, "   *** FAIL" if bad else ""))

print("\n%s" % ("CIS FULL-DIAGONALIZATION CHECK PASSED"
                if not failures else "*** %d ROOT(S) MOVED ***" % failures))
sys.exit(1 if failures else 0)
