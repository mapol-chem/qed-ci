"""Small LiH/STO-3G examples: python methods.py [cis|fci|casci|casscf|sa]."""
import argparse
from qed_ci import CQEDConfig, CISolverOptions, calculate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['cis', 'fci', 'casci', 'casscf', 'sa'])
    parser.add_argument('--output', help='write the report here instead of the console')
    parser.add_argument('--quiet', action='store_true', help='suppress routine engine output')
    args = parser.parse_args()
    method = 'casscf' if args.method == 'sa' else args.method
    active = dict(active_electrons=2, active_orbitals=2) if method in {'casci', 'casscf'} else {}
    config = CQEDConfig(
        method=method, psi4_options={'basis': 'sto-3g', 'scf_type': 'pk',
                                    'e_convergence': 1e-10, 'd_convergence': 1e-10},
        lambda_vector=(0., 0., 0.05), omega=0.12, n_roots=2,
        state_weights=(0.5, 0.5) if args.method == 'sa' else (1., 0.),
        ci_solver=CISolverOptions(threshold=1e-9),
        quiet=args.quiet, output_file=args.output, **active)
    calculation = calculate('Li\nH 1 1.4\nunits angstrom\nsymmetry c1', config)
    result = calculation.result
    # Application output remains under the caller's control, even in quiet mode.
    print('Final energies (Eh):', result.energies['final'][:config.n_roots])
    print('Convergence status:', result.converged)
    if method == 'casscf':
        print('State-average energy (Eh):', result.energies['state_average'])


if __name__ == '__main__':
    main()
