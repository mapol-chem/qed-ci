"""Human presentation of the versioned scientific result interface."""
def heading(title):
    print(f'\n  ==> {title} <==\n')


def report(result):
    r = result.summary()
    print('  ==> QED-CI calculation <==\n')
    p, c, e = r['provenance'], r['requested'], r['effective']
    print(f"  Method: {c['method'].upper()}   Effective: {e['method'].upper()}   Reference: {c['reference'].upper()}")
    print(f"  Psi4 {p['psi4']}   NumPy {p['numpy']}   Python {p['python']}")
    print(f"  Backend: {p['backend']}\n  SHA256: {p['backend_sha256']}")
    print(f"  Basis: {c['psi4_options'].get('basis', c['psi4_options'].get('BASIS', 'Psi4 default'))}")
    heading('Geometry (bohr)')
    print('  Atom               X                 Y                 Z')
    for symbol, xyz in zip(r['geometry']['symbols'], r['geometry']['coordinates']):
        print(f"  {symbol:<5}" + ''.join(f'{x:18.10f}' for x in xyz))
    print(f"  Charge: {r['geometry']['charge']}   Multiplicity: {r['geometry']['multiplicity']}")
    heading('Cavity and CI space')
    eng = e['engine']
    print(f"  Frequency: {c['omega']} Eh   Coupling (a.u.): {c['lambda_vector']}")
    print(f"  Photon basis: {'coherent' if eng['coherent_state_basis'] else 'number'}   Maximum photon occupation: {eng['N_p']}")
    print(f"  Actual photon sectors: {eng['photon_sectors']}   Reference orbitals: {'canonical' if eng['canonical_mos'] else 'CQED-RHF'}")
    print(f"  Ignore coupling: {c['ignore_coupling']}   Omit DSE: {c['ignore_dse_terms']}")
    print(f"  Active electrons/orbitals: {eng['n_act_el']} / {eng['n_act_orb']}   CI dimension: {eng['H_dim']}")
    print(f"  Requested roots: {c['n_roots']}   Spin selection: {c['spin']}")
    print(f"  Normalized state weights: {e['state_weights']}")
    heading('Convergence')
    print('  CI residuals and energy changes: Eh; orbital gradient: atomic units')
    print('  Stage          Converged   Iterations   Stopping reason')
    for name, d in r['convergence'].items():
        status = 'n/a' if not d.get('applicable', True) else str(d['converged']) if d['converged'] is not None else 'unknown'
        print(f"  {name:<14} {status:<11} {str(d.get('iterations')):<12} {d['reason']}")
        for key in ('mean_residual', 'residuals', 'threshold', 'density_rms', 'energy_change', 'gradient_norm', 'thresholds'):
            if d.get(key) is not None:
                print(f'    {key}: {d[key]}')
    history = r['convergence']['casscf'].get('history', [])
    if history:
        print('  Macro       Average energy (Eh)       Delta E (Eh)     Lagged gradient')
        for row in history:
            print(f"  {row['iteration']:5d} {row['energy']:25.12f} {row['energy_change']:18.6e} {str(row['gradient_norm']):>19}")
    heading(f"Final state energies ({r['status']})")
    print(f"@ Reference energy                {r['energies']['reference']:20.12f} Eh")
    print('  State      Total energy (Eh)       Excitation (Eh)')
    for i, (energy, excitation) in enumerate(zip(r['energies']['final'][:c['n_roots']], r['energies']['excitation'][:c['n_roots']])):
        print(f'  {i:5d} {energy:24.12f} {excitation:21.12f}')
    if r['energies']['state_average'] is not None:
        print(f"@ State-average energy            {r['energies']['state_average']:20.12f} Eh")
    if 'dipole' in r['properties']:
        prop = r['properties']['dipole']
        heading(f"Dipole matrix ({prop['units']}; {prop['energy_source']} states)")
        print('  Bra   Ket               X                 Y                 Z')
        for i, row in enumerate(prop['value']):
            for j, xyz in enumerate(row):
                print(f'  {i:3d}   {j:3d}' + ''.join(f'{x:18.10f}' for x in xyz))
    heading('Warnings, timing and files')
    for message in r['warnings']:
        print(f'  WARNING: {message}')
    print(f"  Calculation time: {r['timings']['calculation']:.3f} s")
    for name, path in r['files'].items():
        print(f'  {name}: {path}')
    print('  ==> End QED-CI report <==')


def property_report(key, prop):
    heading(f"{key} ({prop['units']})")
    print('  Atom               X                 Y                 Z')
    for i, row in enumerate(prop['value']):
        print(f'  {i:5d}' + ''.join(f'{x:18.10f}' for x in row))
