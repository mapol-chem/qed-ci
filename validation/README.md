# Scientific validation drivers

These programs preserve larger or more specialized scientific checks that do
not belong in the canonical user examples. They ship in the source distribution
for reproducibility but are not installed as runtime modules.

- `benchmarks/c10h8_casscf.py` is the naphthalene CASSCF benchmark used in
  solver-development studies.
- `drivers/ethylene_mecp_scan.py` is the ASE-based quasi-Newton ethylene MECP
  scan. It requires ASE in addition to the normal QED-CI environment.
- The scripts in this directory exercise numerical gradients, nonadiabatic
  couplings, integral readback, invariances, and solver trajectories. Several
  are intentionally much more expensive than ordinary unit tests.

Run these from an installed/editable checkout or with `PYTHONPATH=src`. They may
write calculation outputs and should generally be launched from a scratch
directory. Their scientific settings and reference geometries are preserved;
moving them here does not promote every driver to a stable public interface.
