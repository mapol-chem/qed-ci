from morse import Morse
import numpy as np
from matplotlib import pyplot as plt

mA = 1.0
mB = 19.0

# parameters for ground state
X_morse_omega_wn = 4165.1570588584555
X_morse_omega_xe_wn = 115.73758535305562
X_r_eq_SI = 9.178507603174319e-11
X_Te = 0.

# parameters for excited state
D_morse_omega_wn = 3370.510408803543
D_morse_omega_xe_wn = 176.0443972578461
D_r_eq_SI = 9.750100896865223e-11
D_Te = 0.

# create Morse object for ground state
X = Morse(mA, mB, X_morse_omega_wn, X_morse_omega_xe_wn, X_r_eq_SI, X_Te)
#X.make_rgrid(n = 500, rmin=4e-11, rmax=5e-10)
#X.V = X.Vmorse(X.r)

# create Morse object for excited state
D = Morse(mA, mB, D_morse_omega_wn, D_morse_omega_xe_wn, D_r_eq_SI, D_Te)
#D.make_rgrid(n = 500, rmin=4e-11, rmax=5e-10)
#D.V = D.Vmorse(D.r)

# calculate psi_
_tmp = X.calc_psi_z(0)
X_psi0 = np.copy(X.psi_au)

_tmp = D.calc_psi_z(0)
D_psi0 = np.copy(D.psi_au)

plt.plot(X.r_au, X_psi0)
plt.plot(D.r_au, D_psi0)
#plt.plot(X.r_au, X.V)
#plt.plot(D.r_au, D.V)
plt.show()

print("<x0|x0> on X.r_au")
print(np.trapz(X_psi0 ** 2, X.r_au))

print("<x0|x0> on D.r_au")
print(np.trapz(X_psi0 ** 2, D.r_au))

print("<d0|d0> on X.r_au")
print(np.trapz(D_psi0 ** 2, X.r_au))

print("<d0|d0> on D.r_au")
print(np.trapz(D_psi0 ** 2, D.r_au))

#norm_psip0 = np.trapz(psip_0 ** 2, Y_v.r)
#FC_00 = np.trapz(psi_0 * psip_0, X_v.r)
#FC_00p = np.trapz(psi_0 * psip_0, Y_v.r)

#print(F" Normalization of |psi_0> is {norm_psi0}")
#print(F" Normalization of |psi'_0> is {norm_psip0}")
#print(F" Franck-Condon Factor 00 is {FC_00}")
#print(F" Franck-Condon Factor 00 is {FC_00p}")
