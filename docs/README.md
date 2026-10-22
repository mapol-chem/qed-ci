# QED-CI user guide

This guide is organized by user task. The configuration reference is the single
authoritative option table; other pages link to it rather than repeating it.

| Task | Start here |
|---|---|
| Set up an environment, install, or fix a build/load problem | [Installation and troubleshooting](installation.md) |
| Understand what can be calculated and what is restricted | [Supported methods and limitations](methods.md) |
| Configure a calculation, route output, read results, or migrate legacy dictionaries | [Configuration, output, and results](configuration-and-output.md) |
| Run a standalone example for each method | [Canonical examples](../examples/canonical/README.md) |
| Run a small calculation | [Runnable examples](../examples/quickstart/README.md) |
| Develop, rebuild the native library, add options/results, or run tests | [Developer setup and testing](development.md) |
| Review the staged validation record | [Stage 5 audit](audit/stage5/README.md), [Stage 4 audit](audit/stage4/README.md), [Stage 1 audit](audit/stage1/README.md) |

## Current feature boundary

Implemented: CQED-RHF/RHF references, QED-CIS, QED-FCI, QED-CASCI,
state-averaged QED-CASSCF, unequal state weights, analytic SA-QED-CASSCF
gradients/derivative couplings, `CQEDConfig`, scoped output, and the version-1
`CalculationResult` schema.

Not part of this stage: a Calculator dispatcher, cqed-scf reference
wavefunctions, TAMM/distributed backends, support for additional reference
types, and changes to scientific algorithms/defaults.

## Recommended first path

1. Install into a Psi4 environment using [installation](installation.md).
2. Run the quiet CASCI example in [the examples guide](../examples/quickstart/README.md).
3. Read the small `CQEDConfig` example and option reference in
   [configuration and output](configuration-and-output.md).
4. Use `calculation.result.summary()` rather than parsing console reports.
5. Choose a method and check its restrictions in [methods](methods.md).
