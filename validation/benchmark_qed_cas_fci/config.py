"""Configuration for the QED-CAS/QED-FCI 1-RDM benchmark."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DIMENSION_CAP = 2_000_000
PHOTON_STATES_FOR_SCREEN = 10
BRIGHT_DIPOLE_THRESHOLD = 0.05  # e bohr
LAMBDA_MAGNITUDES = (0.01, 0.05, 0.1)
PHOTON_STATE_COUNTS = (2, 4, 6, 10)
BASIS_PRIORITY = ("6-311g", "6-31g", "sto-3g")

# Geometries are equilibrium starting points in Angstrom.  The exact source
# and citation for each geometry are recorded in geometry_sources.csv after
# the literature values are verified; optimized values may replace these.
SYSTEMS = {
    "H2": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nH 0 0 0\nH 0 0 0.7414\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 2, "valence_orbitals": 2, "core_orbitals": 0},
    "HeH+": {"charge": 1, "multiplicity": 1, "geometry": "1 1\nHe 0 0 0\nH 0 0 0.772\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 2, "valence_orbitals": 2, "core_orbitals": 0},
    "H3+": {"charge": 1, "multiplicity": 1, "geometry": "1 1\nH 0 0 0\nH 0.87275 0 0\nH 0.436375 0.755896 0\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 2, "valence_orbitals": 3, "core_orbitals": 0},
    "LiH": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nLi 0 0 0\nH 0 0 1.595\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 2, "valence_orbitals": 2, "core_orbitals": 1},
    "He2": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nHe 0 0 0\nHe 0 0 2.965\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 4, "valence_orbitals": 2, "core_orbitals": 0},
    "BH": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nB 0 0 0\nH 0 0 1.232\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 4, "valence_orbitals": 5, "core_orbitals": 1},
    "BeH2": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nBe 0 0 0\nH 0 0 1.326\nH 0 0 -1.326\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 4, "valence_orbitals": 6, "core_orbitals": 1},
    "Li2": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nLi 0 0 0\nLi 0 0 2.673\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 2, "valence_orbitals": 2, "core_orbitals": 2},
    "HF": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nH 0 0 0\nF 0 0 0.917\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 8, "valence_orbitals": 6, "core_orbitals": 1},
    "H2O": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nO 0 0 0\nH 0 0.7572 -0.5865\nH 0 -0.7572 -0.5865\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 8, "valence_orbitals": 6, "core_orbitals": 1},
    "NH3": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nN 0 0 0\nH 0 -0.9377 -0.3816\nH 0.8121 0.4689 -0.3816\nH -0.8121 0.4689 -0.3816\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 8, "valence_orbitals": 7, "core_orbitals": 1},
    "CH4": {"charge": 0, "multiplicity": 1, "geometry": "0 1\nC 0 0 0\nH 0.6273 0.6273 0.6273\nH -0.6273 -0.6273 0.6273\nH -0.6273 0.6273 -0.6273\nH 0.6273 -0.6273 -0.6273\nunits angstrom\nsymmetry c1\nno_reorient\nno_com", "valence_electrons": 8, "valence_orbitals": 8, "core_orbitals": 1},
}

STRETCHED = {
    "H2_2re": {"system": "H2", "scale": 2.0},
    "LiH_2re": {"system": "LiH", "scale": 2.0},
    "Li2_1p5re": {"system": "Li2", "scale": 1.5},
}
