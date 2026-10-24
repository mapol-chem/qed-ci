"""Analytic ground-state SA-QED-CASSCF gradient for LiH/STO-3G (Eh/bohr)."""
from qed_ci import CQEDConfig, CISolverOptions
from qed_ci.output import calculation_output


def main():
    config = CQEDConfig(
        method='casscf', active_electrons=2, active_orbitals=2,
        n_roots=2, state_weights=(0.5, 0.5),
        psi4_options={'basis': 'sto-3g', 'scf_type': 'pk',
                      'e_convergence': 1e-10, 'd_convergence': 1e-10},
        lambda_vector=(0., 0., 0.05), omega=0.12,
        ci_solver=CISolverOptions(threshold=1e-9), quiet=True)
    # The existing gradient subclass is a supported legacy import. Scope its
    # import as well as construction to silence lazy engine initialization.
    with calculation_output(config):
        from nuclear_grad import nuclear_grad
        calculation = nuclear_grad(
            'Li\nH 1 1.4\nunits angstrom\nno_reorient\nno_com\nsymmetry c1', config=config)
        calculation.compute_grad(0, 0)
    print('Convergence status:', calculation.result.converged)
    print('Gradient (Eh/bohr):', calculation.result.summary()['properties']['gradient_state_0']['value'])


if __name__ == '__main__':
    main()
