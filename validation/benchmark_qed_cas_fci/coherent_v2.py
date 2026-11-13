"""Benchmark-local canonical-RHF/coherent-photon adapter.

CQEDConfig currently only accepts qed_rhf/coherent or canonical/number.
The orbital-guess path semicanonicalizes with the QED Fock matrix, so it is
not appropriate here. Supply the canonical reference through the existing
reference-builder seam, using exactly helper_cqed_rhf's coherent constants.
No package files are changed. The patch is scoped to a single calculation.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import helper_PFCI


@contextmanager
def canonical_coherent_reference():
    original = helper_PFCI.cqed_rhf
    def reference(lam, molecule, options, **kwargs):
        data = original(lam, molecule, options, canonical_basis=True)
        density = data['RHF DENSITY MATRIX']
        dipole = data['1-E DIPOLE MATRIX AO']
        mean = float(2 * np.einsum('pq,pq->', density, dipole))
        data.update({
            'CQED-RHF C': data['RHF C'].copy(),
            'CQED-RHF DENSITY MATRIX': density.copy(),
            'CQED-RHF FOCK MATRIX': np.asarray(data['PSI4 WFN'].Fa()).copy(),
            # Informational reference fields; not CI Hamiltonian constants.
            'CQED-RHF ENERGY': data['RHF ENERGY'],
            'CQED-RHF ONE-ENERGY': float(2*np.einsum('pq,pq->', density,
                data['1-E KINETIC MATRIX AO']+data['1-E POTENTIAL MATRIX AO'])),
            'COHERENT STATE EXPECTATION VALUE OF d': mean,
            'COHERENT STATE DIPOLE ENERGY': 0.5*mean**2,
            'COHERENT STATE 1-E SCALED DIPOLE MATRIX AO': -mean*dipole,
        })
        return data
    with patch.object(helper_PFCI, 'cqed_rhf', reference):
        yield


def displacement(owner):
    """Physical a = b + alpha for the displaced photon oscillator."""
    return float((owner.d_exp + np.dot(owner.lambda_vector,
        owner.nuclear_dipole_moment))/np.sqrt(2*owner.omega))
