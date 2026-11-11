"""Exercise the Psi4 orbital-space handling used by geometry restarts."""

import numpy as np
import psi4
import pytest

from ortho_script import ortho_orbs


def test_restart_orthogonalizes_zero_frozen_spaces_and_rejects_frozen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    psi4.core.clean_options()
    psi4.core.set_output_file(str(tmp_path / "psi4.out"), False)
    try:
        psi4.set_options({"basis": "sto-3g", "scf_type": "pk",
                          "restricted_docc": [1], "active": [2],
                          "frozen_docc": [0], "frozen_uocc": [0]})

        def reference(distance):
            molecule = psi4.geometry(f"0 1\nLi\nH 1 {distance}\nsymmetry c1\n"
                                     "no_reorient\nno_com")
            return psi4.energy("scf", molecule=molecule, return_wfn=True)[1]

        previous, current = reference(1.60), reference(1.65)
        original = np.asarray(previous.Ca()).copy()
        overlap = np.asarray(current.S())
        # The old coefficients do need orthogonalization at the new geometry.
        assert np.linalg.norm(original.T @ overlap @ original - np.eye(current.nmo())) > 1e-5
        for semi in (False, True):
            coefficients = np.asarray(ortho_orbs(previous, current, semi=semi))
            assert np.isfinite(coefficients).all()
            np.testing.assert_allclose(coefficients.T @ overlap @ coefficients,
                                       np.eye(current.nmo()), atol=1e-10)
        np.testing.assert_array_equal(np.asarray(previous.Ca()), original)

        for option in ("frozen_docc", "frozen_uocc"):
            psi4.set_options({option: [1]})
            with pytest.raises(ValueError, match="frozen orbitals"):
                ortho_orbs(previous, current)
            psi4.set_options({option: [0]})
    finally:
        psi4.core.clean_options()
        psi4.core.close_outfile()
