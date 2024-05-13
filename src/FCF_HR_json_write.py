from morse import Morse
from vibronic_helper import Vibronic
import numpy as np
import json

mol_str = """
H
F 1 0.9178507603174318
symmetry c1
"""

g_options = {
"only_singlets" : True,
"number_of_photons" : 0,
"number_of_electronic_states" : 15,
"max_vibrational_state" : 10,
"omega" : 0,
"basis" : "6-311++G**",
"lambda_vector" : [0, 0, 0],
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
x0p = Xv.x0_offset_p_au
x0m = Xv.x0_offset_m_au
X = Morse(Xv.mA, Xv.mB, Xv.morse_omega_wn, Xv.morse_omega_wn * Xv.morse_xe, Xv.r_eq_SI, 0)

#X.make_rgrid(n = 500, rmin = 4e-11, rmax = 5e-10)
#X.V = X.Vmorse(X.r)



d_options =  g_options.copy()
d_options["target_root"] = 6
d_options["guess_bondlength"] = 0.9750100896865223

Dv = Vibronic(d_options)
Dv.compute_morse_parameters()

D = Morse(Dv.mA, Dv.mB, Dv.morse_omega_wn, Dv.morse_omega_wn * Dv.morse_xe, Dv.r_eq_SI, 0)

#D.make_rgrid(n = 500, rmin = 4e-11, rmax = 5e-10 )
#D.V = D.Vmorse(D.r)


# prepare to compute FCF and overlaps
g_max = g_options["max_vibrational_state"]
d_max = d_options["max_vibrational_state"]


# arrays for Franck-Condon and overlaps
fcf = []
s1_o = []
s2_o = []

for i in range(g_max):
    # bra states 
    temp_X = X.calc_psi_z(i)
    temp_D = D.calc_psi_z(i)

    bra_x = np.copy(X.psi_au)
    bra_d = np.copy(D.psi_au)
    for j in range(d_max):
        # ket states
        temp_X = X.calc_psi_z(j)
        temp_D = D.calc_psi_z(j)
        ket_x = np.copy(X.psi_au)
        ket_d = np.copy(D.psi_au)

        # overlaps
        xij = np.trapz(bra_x * ket_x, X.r_au)
        dij = np.trapz(bra_d * ket_d, X.r_au)

        # fcf
        fij = np.trapz(bra_x * ket_d, X.r_au)

        s1_o.append(xij)
        s2_o.append(dij)
        fcf.append(fij)


state_1_k = Xv.f_xx
state_1_mu = Xv.mu_au
DeltaX = Dv.r_eq_au - Xv.r_eq_au
Sp = 0.5 * (DeltaX / x0p) ** 2
Sm = 0.5 * (DeltaX / x0m) ** 2
x0m_ang = x0m  * 0.529177210903
x0p_ang = x0p * 0.529177210903

S2 = Xv.mu_au * Xv.harmonic_omega_au * DeltaX ** 2 / 2

# prepare dictionary for json writing
results_dict = {
    "molecule" : g_options,

    "return_results" : {
        "state_1_energy" : Xv.f,
        "state_1_gradient" : Xv.f_x,
        "state_1_equilibrium_geometry_SI" : Xv.r_eq_SI,
        "state_1_equilibrium_geometry" : g_options["guess_bondlength"],
        "state_1_equilibrium_geometry_au" : Xv.r_eq_au,
        "state_1_we_wn" : Xv.morse_omega_wn,
        "state_1_wexe_wn" : Xv.morse_omega_wn * Xv.morse_xe,
        "state_1_target_root" : g_options["target_root"],
        "state_1_force_constant_au" : state_1_k,
        "reduced_mass_au" : state_1_mu,
        "state_2_energy" : Dv.f,
        "state_2_gradient" : Dv.f_x,
        "state_2_equilibrium_geometry_SI" : Dv.r_eq_SI,
        "state_2_equilibrium_geometry" : d_options["guess_bondlength"],
        "state_2_equilibrium_geometry_au" : Dv.r_eq_au,
        "state_2_we_wn" : Dv.morse_omega_wn,
        "state_2_wexe_wn" : Dv.morse_omega_wn * Dv.morse_xe,
        "franck_condon_factors" : fcf,
        "state_1_overlap_matrix" : s1_o,
        "state_2_overlap_matrix" : s2_o,
        "maximum_franck_condon_factor" : (g_max, d_max),
        "state_1_turning_points (Bohr)" : [x0m, x0p],
        "huang_rhys_factors" : [Sp, Sm],
        "huang_rhys_factor_2" : S2,
        "state_1_turning_points (Angstroms)" : [x0m_ang, x0p_ang],
        "Delta X" : DeltaX,
        }
}

json_object = json.dumps(results_dict, indent=4)
with open("test_fcf_hr2.json", "w") as outfile:
    outfile.write(json_object)
