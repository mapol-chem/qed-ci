"""Validate QED_LIBTRANS against the mo_eri path it replaces.

Runs the SAME system twice in one process -- once with QED_LIBTRANS off (the
mo_eri route) and once with it on (mints.integrals + IntegralTransform +
psi4_dpd_shim.so) -- and compares self.twoeint element by element, plus the CI
energies that come out the far end.

    cd src && python validation/dpd_readback_check.py

Needs psi4_dpd_shim.so built next to helper_PFCI.py:

    g++ -O2 -fPIC -shared -std=c++17 -fopenmp \\
        -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/psi4 \\
        psi4_dpd_shim.cpp -o psi4_dpd_shim.so

The packed array is compared, not the dense nmo^4 one, because on the
QED_LIBTRANS path the dense array is never built -- that is half the point of
the change (8.40 GB at c10h8 dims).
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOL = """
Li 0.0 0.0 -0.70
H  0.0 0.0  0.70
nocom
noreorient
symmetry c1
"""
OPTIONS = {
    "basis": "6-31g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}
CAVITY = {
    "omega_value": 0.12086,
    "lambda_vector": np.array([0.0, 0.0, 0.05]),
    "ci_level": "fci",
    "davidson_roots": 4,
    "number_of_photons": 1,
    "photon_number_basis": False,
    "canonical_mos": True,
    "coherent_state_basis": True,
}


def run(libtrans):
    """Fresh import each time so the module-level knob is re-read."""
    os.environ["QED_LIBTRANS"] = "1" if libtrans else "0"
    for mod in ("helper_PFCI",):
        if mod in sys.modules:
            del sys.modules[mod]
    helper_PFCI = importlib.import_module("helper_PFCI")
    obj = helper_PFCI.PFHamiltonianGenerator(MOL, OPTIONS, dict(CAVITY))
    return np.copy(obj.twoeint), np.copy(obj.CIeigs[: CAVITY["davidson_roots"]])


print("=" * 70)
print("QED_LIBTRANS=0  (mo_eri, the current path)")
print("=" * 70, flush=True)
ref_twoeint, ref_eigs = run(False)

print("\n" + "=" * 70)
print("QED_LIBTRANS=1  (libtrans + psi4_dpd_shim.so)")
print("=" * 70, flush=True)
got_twoeint, got_eigs = run(True)

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"twoeint shape      {ref_twoeint.shape} vs {got_twoeint.shape}")
d = np.max(np.abs(ref_twoeint - got_twoeint))
print(f"max|d twoeint|     {d:.3e}")
de = np.max(np.abs(ref_eigs - got_eigs))
print(f"max|d CI energy|   {de:.3e}")
for i, (a, b) in enumerate(zip(ref_eigs, got_eigs)):
    print(f"  root {i}: {a:.12f}  {b:.12f}   {abs(a-b):.2e}")

ok = d < 1e-10 and de < 1e-10
print(f"\n{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
