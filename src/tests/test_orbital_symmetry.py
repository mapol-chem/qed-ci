"""Tests for the optional axial-symmetry constraint (QED_ORBITAL_SYMMETRY).

These test orbital_symmetry.py directly -- it has no dependency on
helper_PFCI.py, which is the point of keeping it a separate module.
"""
import numpy as np
import psi4
import pytest

import orbital_symmetry as osym


@pytest.fixture(scope="module")
def mgh():
    psi4.core.be_quiet()
    psi4.set_output_file("test_orbital_symmetry_psi4.out")
    mol = psi4.geometry(
        """
1 1
MG 0.0 0.0 -0.040324507114
H  0.0 0.0  0.959675492886
symmetry c1
units angstrom
"""
    )
    psi4.set_options(
        {"basis": "cc-pVDZ", "scf_type": "pk", "puream": True,
         "e_convergence": 1e-12, "d_convergence": 1e-10}
    )
    _, wfn = psi4.energy("scf", return_wfn=True)
    bs = wfn.basisset()
    S = np.asarray(psi4.core.MintsHelper(bs).ao_overlap())
    return bs, S, np.asarray(wfn.Ca())


def test_ao_species_matches_the_basis_layout(mgh):
    bs, _, _ = mgh
    sp = osym.ao_species(bs)
    # Mg: 4 s | 3 p shells | 1 d shell ; H: 2 s | 1 p shell   (23 AOs)
    expected = ([0] * 4 + [0, 1, -1] * 3 + [0, 1, -1, 2, -2]
                + [0, 0] + [0, 1, -1])
    assert sp.tolist() == expected


def test_overlap_is_block_diagonal_in_m(mgh):
    """The assumption the whole constraint rests on."""
    bs, S, _ = mgh
    worst = osym.validate_axis(S, osym.ao_species(bs))
    assert worst < 1e-10


def test_bent_molecule_is_rejected_loudly(mgh):
    """A non-linear molecule must fail, not silently give a wrong constraint."""
    psi4.core.be_quiet()
    psi4.set_output_file("test_orbital_symmetry_psi4.out")
    psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
units angstrom
""")
    psi4.set_options({"basis": "cc-pVDZ", "puream": True})
    wfn = psi4.core.Wavefunction.build(
        psi4.core.get_active_molecule(), psi4.core.get_global_option("BASIS")
    )
    bs = wfn.basisset()
    S = np.asarray(psi4.core.MintsHelper(bs).ao_overlap())
    with pytest.raises(ValueError, match="NOT block diagonal"):
        osym.validate_axis(S, osym.ao_species(bs))


def test_symmetrize_preserves_orthonormality_and_the_block_split(mgh):
    bs, S, C = mgh
    sp = osym.ao_species(bs)
    labels, _, _, _ = osym.classify(C, S, sp)
    Cs = osym.symmetrize(C, S, sp, labels)
    assert np.abs(Cs.T @ S @ Cs - np.eye(C.shape[1])).max() < 1e-10
    # closed (0:2) and closed+active (0:14) must still span the same spaces,
    # otherwise orbitals have migrated between blocks
    for sl in (slice(0, 2), slice(0, 14)):
        sv = np.linalg.svd(C[:, sl].T @ S @ Cs[:, sl], compute_uv=False)
        assert sv.min() > 1.0 - 1e-8


def test_symmetrized_orbitals_are_pure(mgh):
    bs, S, C = mgh
    sp = osym.ao_species(bs)
    labels, _, _, _ = osym.classify(C, S, sp)
    _, purity, _, _ = osym.classify(osym.symmetrize(C, S, sp, labels), S, sp)
    assert purity.min() > 1.0 - 1e-10


def test_filtered_map_is_a_subset_of_the_unconstrained_one(mgh):
    """The constraint may only REMOVE rotations, never invent one."""
    bs, S, C = mgh
    sp = osym.ao_species(bs)
    labels, _, _, _ = osym.classify(C, S, sp)
    n_in_a, n_occupied, nmo = 2, 14, 23
    full = osym.filtered_index_map(np.zeros(nmo, dtype=int), n_in_a, n_occupied, nmo)
    sym = osym.filtered_index_map(labels, n_in_a, n_occupied, nmo)
    assert len(full) == 150                      # the documented MgH+ count
    assert len(sym) < len(full)
    assert {tuple(p) for p in sym} <= {tuple(p) for p in full}
    # and every surviving pair really does join two orbitals of one species
    for s, r in sym:
        assert labels[s] == labels[r]


def test_pack_reduced_matches_the_hand_written_loop(mgh):
    """pack_reduced must reproduce the enumeration helper_PFCI.py open-codes."""
    bs, S, C = mgh
    n_in_a, n_occupied, nmo = 2, 14, 23
    imap = osym.filtered_index_map(np.zeros(nmo, dtype=int), n_in_a, n_occupied, nmo)
    rng = np.random.default_rng(0)
    full = rng.standard_normal((nmo, nmo))
    ref, count = np.zeros(len(imap)), 0
    for k in range(n_occupied):
        for r in range(k + 1, nmo):
            if k < n_in_a and r < n_in_a:
                continue
            if n_in_a <= k < n_occupied and n_in_a <= r < n_occupied:
                continue
            ref[count] = full[r][k]
            count += 1
    out = np.zeros(len(imap))
    osym.pack_reduced(full, imap, out)
    assert count == len(imap)
    assert np.array_equal(out, ref)


def test_cartesian_basis_is_rejected():
    psi4.core.be_quiet()
    psi4.set_output_file("test_orbital_symmetry_psi4.out")
    psi4.geometry("""
1 1
MG 0.0 0.0 0.0
H  0.0 0.0 1.0
symmetry c1
units angstrom
""")
    psi4.set_options({"basis": "cc-pVDZ", "puream": False})
    wfn = psi4.core.Wavefunction.build(
        psi4.core.get_active_molecule(), psi4.core.get_global_option("BASIS")
    )
    with pytest.raises(ValueError, match="CARTESIAN"):
        osym.ao_species(wfn.basisset())
