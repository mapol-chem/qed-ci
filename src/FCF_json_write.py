from morse import Morse
from vibronic_helper import Vibronic
import numpy as np

mol_str = """
H
F 1 0.9178507603174318
symmetry c1
"""

g_options = {
"only_singlets" : True,
"number_of_photons" : 0,
"number_of_electronic_states" : 15,
"omega" : 0,
"basis" : "6-311++G**",
"lambda_vector" : np.array([0, 0, 0]),
"target_root" : 0,
"mass_A" : 1.0,
"mass_B" : 19.0,
"qed_type" : "qed-ci",
"ci_level" : "cas",
"nact_orbs" : 16,
"nact_els" : 6,
"damping_factor" : 0.5,
"molecule_template" :
"""
H
F 1 **R**
symmetry c1
""",
"guess_bondlength" : 0.9178507603174318,
}


Xv = Vibronic(g_options)
Xv.compute_morse_parameters()

X = Morse(Xv.mA, Xv.mB, Xv.morse_omega_wn, Xv.morse_omega_wn * Xv.morse_xe, Xv.r_eq_SI, 0)

X.make_rgrid()
X.V = X.Vmorse(X.r)



d_options =  g_options.copy()
d_options["target_root"] = 6
d_options["guess_bondlength"] = 0.9750100896865223

Dv = Vibronic(d_options)
Dv.compute_morse_parameters()

D = Morse(Dv.mA, Dv.mB, Dv.morse_omega_wn, Dv.morse_omega_wn * Dv.morse_xe, Dv.r_eq_SI, 0)

D.make_rgrid()
D.V = D.Vmorse(D.r)



# calculate vibrational wavefunctions in atomic units for ground state
temp = X.calc_psi_z(0)
psi_g_0 = np.copy(X.psi_au)

temp = X.calc_psi_z(1)
psi_g_1 = np.copy(X.psi_au)

temp = X.calc_psi_z(2)
psi_g_2 = np.copy(X.psi_au)

temp = X.calc_psi_z(3)
psi_g_3 = np.copy(X.psi_au)


temp = D.calc_psi_z(0)
psi_d_0 = np.copy(D.psi_au)

temp = D.calc_psi_z(1)
psi_d_1 = np.copy(D.psi_au)

temp = D.calc_psi_z(2)
psi_d_2 = np.copy(D.psi_au)

temp = D.calc_psi_z(3)
psi_d_3 = np.copy(D.psi_au)


ng00 = np.trapz(psi_g_0**2, X.r_au)
ng11 = np.trapz(psi_g_1**2, X.r_au)
ng22 = np.trapz(psi_g_2**2, X.r_au)
ng33 = np.trapz(psi_g_3**2, X.r_au)

nd00 = np.trapz(psi_d_0**2, X.r_au)
nd11 = np.trapz(psi_d_1**2, X.r_au)
nd22 = np.trapz(psi_d_2**2, X.r_au)
nd33 = np.trapz(psi_d_3**2, X.r_au)

FCF00 = np.trapz(psi_g_0 * psi_d_0, X.r_au)
FCF01 = np.trapz(psi_g_0 * psi_d_1, X.r_au)
FCF02 = np.trapz(psi_g_0 * psi_d_2, X.r_au)
FCF03 = np.trapz(psi_g_0 * psi_d_3, X.r_au)

FCF10 = np.trapz(psi_g_1 * psi_d_0, X.r_au)
FCF11 = np.trapz(psi_g_1 * psi_d_1, X.r_au)
FCF12 = np.trapz(psi_g_1 * psi_d_2, X.r_au)
FCF13 = np.trapz(psi_g_1 * psi_d_3, X.r_au)

FCF20 = np.trapz(psi_g_2 * psi_d_0, X.r_au)
FCF21 = np.trapz(psi_g_2 * psi_d_1, X.r_au)
FCF22 = np.trapz(psi_g_2 * psi_d_2, X.r_au)
FCF23 = np.trapz(psi_g_2 * psi_d_3, X.r_au)

FCF30 = np.trapz(psi_g_3 * psi_d_0, X.r_au)
FCF31 = np.trapz(psi_g_3 * psi_d_1, X.r_au)
FCF32 = np.trapz(psi_g_3 * psi_d_2, X.r_au)
FCF33 = np.trapz(psi_g_3 * psi_d_3, X.r_au)

print(F' ng00: {ng00}')
print(F' ng11: {ng11}')
print(F' ng22: {ng22}')
print(F' ng33: {ng33}')

print(F' nd00: {nd00}')
print(F' nd11: {nd11}')
print(F' nd22: {nd22}')
print(F' nd33: {nd33}')


print(F' FCF00: {FCF00}')
print(F' FCF01: {FCF01}')
print(F' FCF02: {FCF02}')
print(F' FCF03: {FCF03}')

print(F' FCF10: {FCF10}')
print(F' FCF11: {FCF11}')
print(F' FCF12: {FCF12}')
print(F' FCF13: {FCF13}')

print(F' FCF20: {FCF20}')
print(F' FCF21: {FCF21}')
print(F' FCF22: {FCF22}')
print(F' FCF23: {FCF23}')
