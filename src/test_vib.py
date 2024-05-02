from morse import Morse
from vibronic_helper import Vibronic
import numpy as np

mol_str = """
Li
H 1 0.74
symmetry c1
"""

options = {
"number_of_photons" : 0,
"number_of_electronic_states" : 10,
"omega" : 0,
"basis" : "6-31G",
"lambda_vector" : np.array([0, 0, 0]),
"target_root" : 0,
"mass_A" : 1,
"mass_B" : 1,
"qed_type" : "qed-ci",
"molecule_template" :
"""
Li
H 1 **R**
symmetry c1
""",
"guess_bondlength" : 0.74,
}
X = Vibronic(options)

X.optimize_geometry_full_nr()

X.compute_morse_parameters()

X_v = Morse(X.mA, X.mB, X.morse_omega_wn, X.morse_omega_wn * X.morse_x, X.r_eq_SI, X.Te_wn)
X_v.make_rgrid()
X_v.V = X_v.Vmorse(X_v.r)

# calculate psi_20
psi = X_v.calc_psi_z(0)

# create Morse grid - will make grids in SI (self.r) and atomic units (self.r_au)
X.make_rgrid()

# create Morse potential on SI grid, will make potential in SI (self.V) and atomic units (self.V_au)
X.V = X.Vmorse(X.r)
# get Morse parameters
X.compute_morse_parameters()
# instantiate Morse object
    def __init__(self, mA, mB, we, wexe, re, Te=0):
        """Initialize the Morse model for a diatomic molecule.

        mA, mB are the atom masses (atomic mass units).
        we, wexe are the Morse parameters (cm-1).
        re is the equilibrium bond length (m).
        Te is the electronic energy (minimum of the potential well; origin
            of the vibrational state energies).

        """

# compute Morse eigenfunctions

# repeat for next electronic state of interest
                         
