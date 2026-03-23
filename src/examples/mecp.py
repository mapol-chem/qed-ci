import numpy as np
import os
import sys
from ase import Atoms
from ase.io import write

# --- 1. SETUP ENVIRONMENT & CALCULATOR ---
# Path to your custom plugin
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf9/qed-ci/src/")
import psi4
from nuclear_grad import *

# (Use your existing options_dict and cavity_options from lih_surface5.py)
options_dict = {
    'basis': '6-311g',
    'scf_type': 'pk',
    'e_convergence': 1e-10,
    'd_convergence': 1e-10
}

cavity_options = {
    'omega_value' : 0.11342269942,
    'lambda_vector' : np.array([0.0, 0.0, 0.05]),
    'ci_level' : 'cas',
    'davidson_roots' : 3,
    'nact_orbs' : 9,
    'nact_els' : 4,
    'coherent_state_basis' : True,
    'spin_adaptation': "singlet",
    'davidson_threshold' : 1e-7,
    'davidson_maxdim': 31,
    'davidson_maxiter':100,
    'davidson_indim':21,

}

# --- 2. INITIAL GEOMETRY ---
r_start = 1.1  # Starting bond length as requested
m_li, m_h = 7.016004, 1.007825
d_li = (m_h / (m_li + m_h)) * r_start
d_h = (m_li / (m_li + m_h)) * r_start

# Molecule starts on the Z-axis
initial_pos = [
    (0, 0, -d_li),
    (0, 0, d_h)
]
mol = Atoms('LiH', positions=initial_pos)

# --- 3. MECP SEARCH FUNCTION ---
def optimize_to_mecp(atoms, state_p, state_q, max_steps=50, step_size=0.1):
    print(f"Starting MECP search between State {state_p} and {state_q}...")
    
    for step in range(max_steps):
        # Format geometry for Psi4/nuclear_grad
        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        mol_str = "\n".join([f"{s} {p[0]} {p[1]} {p[2]}" for s, p in zip(syms, pos)])
        mol_str += "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom"

        # A. Get Gradient of State P
        job_p = nuclear_grad(mol_str, options_dict, cavity_options)
        job_p.compute_grad(state_p, state_p)
        e_p = job_p.eigenvals[state_p]
        grad_p = np.array(job_p.total_gradient).reshape(-1, 3)

        # B. Get Gradient of State Q
        job_q = nuclear_grad(mol_str, options_dict, cavity_options)
        job_q.compute_grad(state_q, state_q)
        e_q = job_q.eigenvals[state_q]
        grad_q = np.array(job_q.total_gradient).reshape(-1, 3)

        # C. Get h-vector (Derivative Coupling) - i != j triggers NACV logic 
        job_h = nuclear_grad(mol_str, options_dict, cavity_options)
        job_h.compute_grad(state_p, state_q)
        h = np.array(job_h.h0).reshape(-1, 3)

        # D. MECP Math
        gap = (e_q - e_p) * 27.2114  # eV
        g = grad_q - grad_p          # Branching vector 1
        sigma = 0.5 * (grad_p + grad_q) # Mean gradient

        # Orthogonalize h to g to define the plane
        u_g = g / np.linalg.norm(g)
        h_perp = h - np.dot(h.flatten(), u_g.flatten()) * u_g
        u_h = h_perp / np.linalg.norm(h_perp)

        # Projected Step:
        # 1. Close the energy gap along g
        # 2. Minimize average energy along the seam (orthogonal to g and h)
        f_gap = gap * u_g
        f_seam = sigma - np.dot(sigma.flatten(), u_g.flatten()) * u_g \
                       - np.dot(sigma.flatten(), u_h.flatten()) * u_h
        
        total_force = -(f_gap + f_seam)
        
        # Update atoms
        new_pos = atoms.get_positions() + step_size * total_force
        atoms.set_positions(new_pos)

        print(f"Step {step}: Gap = {gap:.6f} eV | R = {atoms.get_distance(0,1):.4f} A")

        if abs(gap) < 1e-4 and np.linalg.norm(f_seam) < 1e-3:
            print("Successfully converged to MECP.")
            break

    return atoms

import numpy as np

def find_mecp_with_kick(atoms, state_p, state_q, max_steps=500, step_size=0.1):
    print(f"Starting MECP search...")

    for step in range(max_steps):
        # 1. Setup Mol String for Psi4
        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        mol_str = "\n".join([f"{s} {p[0]} {p[1]} {p[2]}" for s, p in zip(syms, pos)])
        mol_str += "\nunits angstrom\nsymmetry c1\nno_reorient\nnocom"

        # 2. Get Gradients and Coupling
        job_p = nuclear_grad(mol_str, options_dict, cavity_options)
        job_p.compute_grad(state_p, state_p)
        grad_p = np.array(job_p.total_gradient).reshape(-1, 3)
        e_p = job_p.eigenvals[state_p]

        job_q = nuclear_grad(mol_str, options_dict, cavity_options)
        job_q.compute_grad(state_q, state_q)
        grad_q = np.array(job_q.total_gradient).reshape(-1, 3)
        e_q = job_q.eigenvals[state_q]

        job_h = nuclear_grad(mol_str, options_dict, cavity_options)
        job_h.compute_grad(state_p, state_q)
        h = np.array(job_h.total_gradient).reshape(-1, 3)

        # 3. Math for the Branching Plane
        gap = (e_q - e_p) * 27.2114 # eV
        g = grad_q - grad_p
        sigma = 0.5 * (grad_p + grad_q)

        # Normalize
        u_g = g / np.linalg.norm(g)

        # --- Symmetry Breaking Check ---
        # Check colinearity: cos(phi) = (g . h) / (|g||h|)
        cos_phi = np.abs(np.dot(u_g.flatten(), h.flatten()) / np.linalg.norm(h))

        if cos_phi > 0.98:  # Vectors are nearly parallel (e.g., at theta=0)
            print(f"Step {step}: Symmetry detected. Applying rotational kick...")
            # Apply a displacement perpendicular to the current bond axis
            # If bond is on Z, we kick in X
            kick = np.zeros_like(pos)
            kick[0, 0] = 0.05  # Move Li in +X
            kick[1, 0] = -0.05 # Move H in -X
            atoms.set_positions(pos + kick)
            continue # Recalculate at the new geometry

        # 4. Standard Orthogonalization (Gram-Schmidt)
        h_perp = h - np.dot(h.flatten(), u_g.flatten()) * u_g
        u_h = h_perp / np.linalg.norm(h_perp)

        # 5. Calculate Step
        # Close gap along g, minimize energy along seam (orthogonal to g and h)
        f_gap = gap * u_g
        f_seam = sigma - np.dot(sigma.flatten(), u_g.flatten()) * u_g \
                       - np.dot(sigma.flatten(), u_h.flatten()) * u_h

        total_force = -(f_gap + f_seam)

        # 6. Update positions
        curr_r = atoms.get_distance(0, 1)
        atoms.set_positions(atoms.get_positions() + step_size * total_force)

        print(f"Step {step}: Gap={gap:.4f} eV | R={curr_r:.4f} A | cos(phi)={cos_phi:.3f}")

        if abs(gap) < 1e-4 and np.linalg.norm(f_seam) < 1e-3:
            print("MECP Converged!")
            break

    return atoms

# Run the optimization
mecp_geometry = find_mecp_with_kick(mol, state_p=1, state_q=2)
write('mecp_optimized.xyz', mecp_geometry)
