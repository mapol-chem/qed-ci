from oo_cqed_rhf import CQEDRHFCalculator
import numpy as np
import psi4
import sys
import json
sys.path.append("/home/jfoley19/Code/qed-ci/src/")
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
from nuclear_grad import *
np.set_printoptions(threshold=sys.maxsize)
psi4.core.be_quiet()

def generate_lih_geometries(
    x_vals=[0.0], y_vals=[0.0], z_vals=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
):
    """Generate a list of molecule geometry strings for LiH at different coordinates."""
    molecules = []
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                mol_str = f"""
0 1
Li 0.0 0.0 0.0
H  {x:.3f} {y:.3f} {z:.3f}
no_reorient
nocom
symmetry c1
"""
                molecules.append(mol_str.strip())
    return molecules


def run_gradient_test(original_mol_string, lambda_vector):
    from oo_cqed_rhf import CQEDRHFCalculator
    import numpy as np
    import psi4
    import sys
    sys.path.append("/home/jfoley19/Code/qed-ci/src/")
    from helper_PFCI import PFHamiltonianGenerator
    from helper_cqed_rhf import cqed_rhf
    from nuclear_grad import nuclear_grad

    np.set_printoptions(threshold=sys.maxsize)
    psi4.core.be_quiet()

    def within_order_of_magnitude(a, b, orders=2):
        if a == 0 or b == 0:
            return a == b
        ratio = max(a, b) / min(a, b)
        return ratio <= 10**orders

    BOHR_TO_ANGSTROM = 0.52917721092
    delta_ang = 1e-4
    delta_au = delta_ang / BOHR_TO_ANGSTROM

    mol = psi4.geometry(original_mol_string)
    num_atoms = mol.natom()

    options_dict = {
        "basis": "6-31g",
        "save_jk": True,
        "scf_type": "pk",
        "e_convergence": 1e-7,
        "d_convergence": 1e-5,
    }

    ### STEP 1: CQED-RHF
    calc = CQEDRHFCalculator(lambda_vector, original_mol_string, options_dict)
    calc.molecule_string = original_mol_string

    en, cqed_rhf_analytical_grad, _ = calc.calc_force_and_energy(original_mol_string, use_psi4_scf_grad=False)
    calc.compute_numerical_gradient()
    cqed_rhf_numerical_grad = calc.numerical_energy_gradient

    ### STEP 2: CQED-CASSCF
    cavity_options = {
        'omega_value': 0.12086,
        'lambda_vector': lambda_vector,
        'ci_level': 'cas',
        'number_of_photons': 1,
        'photon_number_basis': False,
        'canonical_mos': False,
        'coherent_state_basis': True,
        'spin_adaptation': "singlet",
        'davidson_roots': 3,
        'davidson_threshold': 1e-7,
        'davidson_maxdim': 8,
        'davidson_maxiter': 100,
        'davidson_indim': 6,
        'nact_orbs': 4,
        'nact_els': 4,
    }

    CASG = nuclear_grad(original_mol_string, options_dict, cavity_options)
    n_states = cavity_options["davidson_roots"]

    cqed_cas_analytic_grad = np.zeros((n_states, num_atoms, 3))
    for i in range(n_states):
        CASG.compute_grad(i)
        cqed_cas_analytic_grad[i, :, :] = CASG.total_gradient.reshape(num_atoms, 3)

    cqed_cas_numeric_grad = np.zeros((n_states, num_atoms, 3))
    for i in range(num_atoms):
        for j in range(3):
            disp = np.zeros((num_atoms, 3))
            disp[i, j] = delta_ang
            mol_f = calc.modify_geometry_string(original_mol_string, disp)
            mol_b = calc.modify_geometry_string(original_mol_string, -disp)
            CAS_F = PFHamiltonianGenerator(mol_f, options_dict, cavity_options)
            CAS_B = PFHamiltonianGenerator(mol_b, options_dict, cavity_options)
            for k in range(n_states):
                cqed_cas_numeric_grad[k, i, j] = (CAS_F.CASSCFeigs[k] - CAS_B.CASSCFeigs[k]) / (2 * delta_ang / BOHR_TO_ANGSTROM)

    cqed_rhf_grad_norm = np.linalg.norm(cqed_rhf_analytical_grad - cqed_rhf_numerical_grad)
    cqed_cas_norms = [np.linalg.norm(cqed_cas_analytic_grad[i] - cqed_cas_numeric_grad[i]) for i in range(n_states)]
    cqed_cas_norm_pass = []

    print(f"\nResults for field {lambda_vector}:")
    print(f"CQED-RHF error norm: {cqed_rhf_grad_norm}")
    for i, norm in enumerate(cqed_cas_norms):
        print(f"  State {i} CASSCF error norm: {norm}")
        if within_order_of_magnitude(norm, 1e-6, orders=2):
            print("    ✅ Acceptable")
            cqed_cas_norm_pass.append(True)
        else:
            print("    ❌ Not acceptable")
            cqed_cas_norm_pass.append(False)

    return {
        "cqed_rhf_error": cqed_rhf_grad_norm,
        "cqed_cas_errors": cqed_cas_norms,
        "cqed_cas_norm_pass": cqed_cas_norm_pass
    }


molecules = generate_lih_geometries(x_vals=[0.0, 0.1], y_vals=[0.0, 0.1], z_vals=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] )
fields = [
    np.array([0, 0, 0.0]),
    np.array([0, 0, 0.01]),
    np.array([0, 0.01, 0]),
    np.array([0.01, 0, 0]),
    np.array([0.0, 0.01, 0.01]),
    np.array([0.01, 0.0, 0.01]),
    np.array([0.01, 0.01, 0.0]),
    np.array([0.01, 0.01, 0.01]),
    np.array([0, 0, 0.05]),
    np.array([0, 0.05, 0]),
    np.array([0.05, 0, 0])
]

results_all = []
for mol in molecules:
    for field in fields:
        res = run_gradient_test(mol, field)
        results_all.append({
            "molecule": mol.strip().split("\n")[1:3],  # atoms only
            "lambda_vector": field.tolist(),
            **res
        })

with open("gradient_test_results.json", "w") as f:
    json.dump(results_all, f, indent=2)