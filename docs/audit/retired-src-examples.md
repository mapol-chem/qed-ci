# Retired source-tree examples

The October 2026 source-layout cleanup removed the historical `src/examples/`
tree and four standalone LiH scripts from `src/`. They predated the validated
configuration/result API, mixed generated output with source, or contained
machine-specific paths. Maintained examples now live under
`examples/canonical/` and `examples/quickstart/`.

The old LiH bond scan was replaced by
`examples/canonical/lih_polariton_surface_scan.py`, which demonstrates
QED-FCI/6-311G and writes both structured JSON and plot-ready CSV. The old RDM
driver was retired pending definition of a stable public RDM contract.

The scientifically important naphthalene CASSCF benchmark and ethylene MECP
scan were preserved under `validation/benchmarks/` and `validation/drivers/`.

Git history remains the recovery record. Before this cleanup is committed, the
files can be inspected from the current parent commit, for example:

```sh
git show HEAD:src/LiH_qedfci_r_scan.py
git show HEAD:src/examples/H2_dimers/README.md
```

After the cleanup is committed, replace `HEAD` with the commit immediately
before the cleanup (or use `git log --all -- <path>` to locate it).
