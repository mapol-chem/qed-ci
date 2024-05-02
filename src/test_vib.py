from morse import Morse
from vibronic_helper import Vibronic
import numpy as np


mol_str = """
Mg
H 1 2.0168266478525405
1 1
symmetry c1
"""

# options for excited state
options_e_state = {
"number_of_photons" : 0,
"number_of_electronic_states" : 5,
"omega" : 0,
"basis" : "cc-pVTZ",
"lambda_vector" : np.array([0, 0, 0]),
"target_root" : 2,
"mass_A" : 23.985041689,
"mass_B" : 1.00,
"gradient_tol" : 1e-4,
"r_step" : 1e-4,
"qed_type" : "qed-ci",
"ci_level" : "cas",
"nact_orbs" : 12,
"nact_els" : 8,
"molecule_template" :
"""
Mg
H 1 **R**
1 1
symmetry c1
""",
"guess_bondlength" : 2.0168266478525405,
}

X = Vibronic(options_e_state)

#X.optimize_geometry_full_nr()

X.compute_morse_parameters()

X_v = Morse(X.mA, X.mB, X.morse_omega_wn, X.morse_omega_wn * X.morse_xe, X.r_eq_SI, X.Te_wn)
X_v.make_rgrid()
X_v.V = X_v.Vmorse(X_v.r)

# calculate psi_
psi_0 = X_v.calc_psi_z(0)

# compute Morse eigenfunctions

# repeat for next electronic state of interest
mol_str = """
Mg
H 1 1.6919628184820623
1 1
symmetry c1
"""
options_g_state =  options_e_state.copy()
options_g_state["target_root"] = 0
options_g_state["guess_bondlength"] = 1.6919628184820623
                         
Y = Vibronic(options_g_state)

#X.optimize_geometry_full_nr()

Y.compute_morse_parameters()

Y_v = Morse(Y.mA, Y.mB, Y.morse_omega_wn, Y.morse_omega_wn * Y.morse_xe, Y.r_eq_SI, Y.Te_wn)
Y_v.make_rgrid()
Y_v.V = Y_v.Vmorse(Y_v.r)

# calculate psi_
psip_0 = Y_v.calc_psi_z(0)

norm_psi0 = np.trapz(psi_0 ** 2, X_v.r)
norm_psip0 = np.trapz(psip_0 ** 2, Y_v.r)
FC_00 = np.trapz(psi_0 * psip_0, X_v.r)
FC_00p = np.trapz(psi_0 * psip_0, Y_v.r)

print(F" Normalization of |psi_0> is {norm_psi0}")
print(F" Normalization of |psi'_0> is {norm_psip0}")
print(F" Franck-Condon Factor 00 is {FC_00}")
print(F" Franck-Condon Factor 00 is {FC_00p}")
