# Source layout

This directory contains the Python package sources and the C CI/CASSCF kernel.

- `helper_cqed_rhf.py` — CQED-RHF mean field (coherent-state Pauli-Fierz
  Hamiltonian), with DIIS.
- `helper_PFCI.py` — CI/CASSCF skeleton, ctypes bindings to the native kernel,
  and the established `PFHamiltonianGenerator` API.
- `qed_ci/` — validated `CQEDConfig`, scoped output, versioned results, and the
  public `calculate()` entry point.
- `ci_solver.c`, `orbital.c` (and headers) — native CI Davidson, sigma-vector,
  RDM, and integral-transformation kernels.

## Using the sources

Install the package as described in [the installation guide](../docs/installation.md):

```sh
python -m pip install --no-build-isolation --no-deps -e .
```

An editable install builds `src/qed_ci/_native/cfunctions.so` and adds this
directory to Python's import path. Public imports (`qed_ci`, `helper_PFCI`,
`helper_cqed_rhf`, `nuclear_grad`) then work from any working directory without
`PYTHONPATH`.

The historical source-checkout workflow is still supported for compatibility:

```sh
python tools/build_backend.py
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Do not mix the two modes while diagnosing imports or native loading; an active
`PYTHONPATH` can hide an installed package or its `_native` library.

Developer testing and native rebuild instructions are in
[development](../docs/development.md).
