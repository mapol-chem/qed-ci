"""Lightweight configuration and lazy entry points for QED-CI."""
__version__ = "0.1.0"

from .results import CalculationResult
from .config import CQEDConfig, CISolverOptions, CASSCFOptions

__all__ = ['CalculationResult', 'CQEDConfig', 'CISolverOptions', 'CASSCFOptions', 'calculate', 'PFHamiltonianGenerator']


def calculate(molecule, config):
    """Run an existing CI/CASSCF calculation with a validated configuration.

    Returns the existing PFHamiltonianGenerator object; result shapes and
    numerical methods are unchanged. Import of the engine is lazy.
    """
    if not isinstance(config, CQEDConfig):
        raise TypeError('config must be a CQEDConfig')
    from .output import calculation_output, engine_output
    with calculation_output(config):
        with engine_output(config):
            from helper_PFCI import PFHamiltonianGenerator
        return PFHamiltonianGenerator(molecule, config=config)


def __getattr__(name):
    if name == 'PFHamiltonianGenerator':
        from helper_PFCI import PFHamiltonianGenerator
        return PFHamiltonianGenerator
    raise AttributeError(name)
