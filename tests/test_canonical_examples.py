"""Lightweight contracts for canonical examples with persistent outputs."""

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "examples/canonical/lih_polariton_surface_scan.py"


def load_scan_module():
    spec = importlib.util.spec_from_file_location("lih_polariton_surface_scan", SCAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lih_scan_writes_archival_json_and_plot_ready_csv(tmp_path):
    scan = load_scan_module()
    scan.JSON_OUTPUT = tmp_path / "scan.json"
    scan.CSV_OUTPUT = tmp_path / "scan.csv"
    result = {
        "schema_version": 1,
        "energies": {"final": [-8.0, -7.9], "excitation": [0.0, 0.1]},
        "converged": True,
        "status": "converged",
        "warnings": [],
    }
    records = [{"bond_length_angstrom": 1.4, "result": result}]
    rows = [{
        "bond_length_angstrom": 1.4,
        "state": 0,
        "energy_Eh": -8.0,
        "excitation_energy_Eh": 0.0,
        "converged": True,
        "status": "converged",
        "warning_count": 0,
    }]

    scan.write_outputs(records, rows)

    archive = json.loads(scan.JSON_OUTPUT.read_text(encoding="utf-8"))
    assert archive["configuration"]["method"] == "fci"
    assert archive["configuration"]["psi4_options"]["basis"] == "6-311g"
    assert archive["points"] == records
    with scan.CSV_OUTPUT.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["bond_length_angstrom"] == "1.4"
    assert table[0]["state"] == "0"
    assert table[0]["energy_Eh"] == "-8.0"
