from helper_PFCI import PFHamiltonianGenerator
import numpy as np
from scipy import optimize
from scipy.constants import h, hbar, c, u
from scipy.special import factorial
from scipy.special import genlaguerre, gamma


class Vibronic:
    """A class for performing geometry optimization and vibrational analysis with QED methods"""

    def __init__(self, options):
        # make sure all values are lower case
        options = {k.lower(): v for k, v in options.items()}

        if "molecule_template" in options:
            self.molecule_template = options["molecule_template"]
            print(self.molecule_template)
        else:
            print(f"molecule_template!  Please restart with a proper molecule_template")
            exit()

        if "guess_bondlength" in options:
            self.r = options["guess_bondlength"]
        else:
            print(f"guess_bondlength not specified!  Please restart with a guess r")
            exit()
        self.zmatrix_string = self.molecule_template.replace("**R**", str(self.r))

        if "qed_type" in options:
            self.qed_type = options["qed_type"]
        else:
            self.qed_type = "qed-ci"
        print(self.qed_type)

        if "ci_level" in options:
            self.ci_level = options["ci_level"]
        else:
            self.ci_level = "fci"

        if self.ci_level == "cas":
            if "nact_orbs" in options:
                self.nact_orbs = options["nact_orbs"]
            else:
                print(
                    " Specification of the number of active orbitals 'nact_orbs' needed"
                )
                exit()
            if "nact_els" in options:
                self.nact_els = options["nact_els"]
            else:
                print(
                    " Specification of the number of active electrons 'nact_els' needed"
                )
                exit()

        if "basis" in options:
            self.orbital_basis = options["basis"]
        else:
            self.orbital_basis = "sto-3g"
        if "number_of_photons" in options:
            self.number_of_photons = options["number_of_photons"]
        else:
            self.number_of_photons = 0
        if "number_of_electronic_states" in options:
            self.number_of_electron_states = options["number_of_electronic_states"]
        else:
            self.number_of_electron_states = 10
        if "omega" in options:
            self.omega = options["omega"]
        else:
            self.omega = 0.0

        if "lambda_vector" in options:
            self.lambda_vector = np.array(options["lambda_vector"])
        else:
            self.lambda_vector = np.array([0, 0, 0])
        # which root do you want to follow for geometry opt / frequency analysis?
        if "target_root" in options:
            self.target_root = options["target_root"]
        else:
            # default is ground state
            self.target_root = 0

        if "r_step" in options:
            self.dr = options["r_step"]
        else:
            self.dr = 0.0005

        if "mass_A" in options:
            self.mA = options["mass_A"]
            print(f" Mass of atom A is {self.mA} AMUs")
        else:
            print(
                "mass_A not defined!  Please restart and specify both mass_A and mass_B in amu"
            )

        if "mass_B" in options:
            self.mB = options["mass_B"]
            print(f" Mass of atom B is {self.mB} AMUs")
        else:
            print(
                "mass_B not defined!  Please restart and specify both mass_A and mass_B in amu"
            )
        self.mu_AMU = self.mA * self.mB / (self.mA + self.mB)

        self.amu_to_au = 1822.89
        self.mu_au = self.mu_AMU * self.amu_to_au

    def compute_qed_gradient(self):
        # displaced geomeries in angstroms
        r_array = np.array(
            [
                self.r - 2 * self.dr,
                self.r - self.dr,
                self.r,
                self.r + self.dr,
                self.r + 2 * self.dr,
            ]
        )
        self.au_to_ang = 0.52917721067121

        h = self.dr / self.au_to_ang
        f = np.zeros(5)
        for i in range(5):
            self.r = r_array[i]
            f[i] = self.compute_qed_energy()

        self.f_x = (1 * f[0] - 8 * f[1] + 0 * f[2] + 8 * f[3] - 1 * f[4]) / (
            12 * 1.0 * h
        )
        self.f_xx = (-1 * f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - 1 * f[4]) / (
            12 * 1.0 * h**2
        )
        self.f_xxx = (-1 * f[0] + 2 * f[1] + 0 * f[2] - 2 * f[3] + 1 * f[4]) / (
            2 * 1.0 * h**3
        )

        # going to return energy and gradient
        return f[2], self.f_x

    def compute_qed_energy(self):
        # make sure geometry is up to date!
        self.zmatrix_string = self.molecule_template.replace("**R**", str(self.r))
        # if qed-ci is specified, prepare dictionaries for qed-ci
        if self.qed_type == "qed-ci":
            print("GOING to run QED-CI")
            options_dict = {
                "basis": self.orbital_basis,
                "scf_type": "pk",
                "e_convergence": 1e-10,
                "d_convergence": 1e-10,
            }
            cavity_dict = {
                "omega_value": self.omega,
                "lambda_vector": self.lambda_vector,
                "ci_level": self.ci_level,
                "davidson_roots": self.number_of_electron_states,
                "canonical_mos": True,
                "coherent_state_basis": False,
                "full_diagonalization": False,
                "number_of_photons": self.number_of_photons,
                "nact_orbs": 0,
                "nact_els": 0,
            }
            if self.ci_level == "cas":
                cavity_dict["nact_orbs"] = self.nact_orbs
                cavity_dict["nact_els"] = self.nact_els

            print(self.zmatrix_string, options_dict, cavity_dict)
            qed_ci_inst = PFHamiltonianGenerator(
                self.zmatrix_string, options_dict, cavity_dict
            )
            return qed_ci_inst.CIeigs[self.target_root]

        elif self.qed_type == "pcqed":
            options_dict = {
                "basis": self.orbital_basis,
                "scf_type": "pk",
                "e_convergence": 1e-10,
                "d_convergence": 1e-10,
            }
            cavity_dict = {
                "omega_value": 0,
                "lambda_vector": np.array([0, 0, 0]),
                "ci_level": self.ci_level,
                "davidson_roots": self.number_of_electron_states,
                "canonical_mos": True,
                "coherent_state_basis": False,
                "full_diagonalization": False,
                "number_of_photons": 0,
                "nact_orbs": 0,
                "nact_els": 0,
            }
            if self.ci_level == "cas":
                cavity_dict["nact_orbs"] = self.nact_orbs
                cavity_dict["nact_els"] = self.nact_els

            print(self.mol_str, options_dict, cavity_dict)

            qed_ci_inst = PFHamiltonianGenerator(
                self.mol_str, options_dict, cavity_dict
            )
            self.fast_build_pcqed_pf_hamiltonian(
                self.number_of_electron_states,
                self.number_of_photons + 1,
                self.omega,
                self.lambda_vector,
                qed_ci_inst.CIeigs,
                qed_ci_inst.dipole_array,
            )
            return qed_ci_inst.CIeigs[self.target_root]

    def optimize_geometry(self):
        print(f" Going to perform a geometry optimization using {self.qed_type}")
        energy_start, gradient_start = self.compute_qed_gradient()
        print(f" Initial Energy is     {energy_start}")
        print(f" Initial bondlength is {self.r}")
        print(f" Initial Gradient is     {gradient_start}")

        optimize_return = minimize(
            self.compute_qed_gradient(),
            self.r,
            method="BFGS",
            jac=True,
        )

        # make sure we capture the optimized bond length
        self.r = optimize_return.x
        print(f" Geometry Optimization Complete!")
        energy_end, gradient_end = self.compute_qed_gradient()
        print(f" Final Energy is     {energy_end}")
        print()
        print(f" Final bondlength is {self.r}")
        print(f" Final gradient is   {gradient_end} ")

    def compute_morse_parameters(self):
        # compute the second and third derivatives
        self.compute_qed_gradient()

        # compute beta and De parameters from 2nd and 3rd derivatives
        self.morse_beta_au = -2 * self.f_xxx / (6 * self.f_xx)
        self.morse_De_au = self.f_xx / (2 * self.morse_beta**2)

        # compute omega_e and xe in atomic units
        self.morse_omega_au = np.sqrt(
            2 * self.morse_De_au * self.morse_beta_au**2 / self.mu_au
        )
        self.morse_xe_au = self.morse_omega_au / (4 * self.morse_De_au)

        # convert to wavenumbers
        self.au_to_wn = 219474.63068

        self.morse_omega_wn = self.morse_omega_au * self.au_to_wn
        self.morse_xe_wn = self.morse_xe_au * self.au_to_wn

    def fast_build_pcqed_pf_hamiltonian(
        self,
        n_el,
        n_ph,
        omega,
        lambda_vector,
        E_array,
        mu_array,
        neglect_DSE=False,
        coherent_state_option=False,
    ):
        """
        Given an array of n_el E_R values and an n_ph states with fundamental energy omega
        build the PF Hamiltonian

        n_el : int
            the number of electronic states (n_el = 1 means only ground-state)

        n_ph : int
            the number of photon occupation states (n_ph = 1 means only the |0> state)

        omega : float
            the photon frequency

        lambda_vector : numpy array of floats
            the lambda vector

        E_array : n_el np.array of floats
            the electronic energies

        mu_array : (n_el x n_el x 3) np.array of floats
            mu[i, j, k] is the kth cartesian component of the dipole moment expectation value between
            state i and state j


        """

        self.PCQED_H_PF = np.zeros((n_el * n_ph, n_el * n_ph))
        self.PCQED_H_EL = np.zeros((n_el * n_ph, n_el * n_ph))
        self.PCQED_H_PH = np.zeros((n_el * n_ph, n_el * n_ph))
        self.PCQED_H_DSE = np.zeros((n_el * n_ph, n_el * n_ph))
        self.PCQED_H_BLC = np.zeros((n_el * n_ph, n_el * n_ph))
        self.PCQED_MU = np.zeros((n_el * n_ph, n_el * n_ph))

        # create identity array of dimensions n_el x n_el
        _I = np.eye(n_el)

        # create the array _A of electronic energies
        _A = E_array[:n_el] * _I

        # create the array _O of omega values
        _O = omega * _I

        # create _d array using einsum
        # add call to self.build_d_array(lambda_vector, mu_array, coherent_state=False)
        # and modify the line below so that _d = np.copy(self.d_array)
        # _d = np.einsum("k,ijk->ij", lambda_vector, mu_array[:n_el,:n_el,:])
        # _d_exp = _d[0,0]
        self.build_d_array(
            n_el, lambda_vector, mu_array, coherent_state=coherent_state_option
        )
        _d = np.copy(self.d_array)

        # create D array using matrix multiplication
        if neglect_DSE:
            _D = np.zeros((n_el, n_el))
        else:
            _D = 1 / 2 * _d @ _d

        for n in range(n_ph):
            # diagonal indices
            b_idx = n * n_el
            f_idx = (n + 1) * n_el
            # diagonal entries
            self.PCQED_H_PF[b_idx:f_idx, b_idx:f_idx] = _A + n * _O + _D
            self.PCQED_H_EL[b_idx:f_idx, b_idx:f_idx] = _A
            self.PCQED_H_DSE[b_idx:f_idx, b_idx:f_idx] = _D
            self.PCQED_H_PH[b_idx:f_idx, b_idx:f_idx] = n * _O
            self.PCQED_MU[b_idx:f_idx, b_idx:f_idx] = _d

            # off-diagonal entries
            if n == 0:
                m = n + 1
                bra_s = n * n_el
                bra_e = (n + 1) * n_el
                ket_s = m * n_el
                ket_e = (m + 1) * n_el

                self.PCQED_H_PF[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m)
                )
                self.PCQED_H_BLC[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m)
                )

            elif n == (n_ph - 1):
                m = n - 1
                bra_s = n * n_el
                bra_e = (n + 1) * n_el
                ket_s = m * n_el
                ket_e = (m + 1) * n_el

                self.PCQED_H_PF[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )
                self.PCQED_H_BLC[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )

            else:
                m = n + 1
                bra_s = n * n_el
                bra_e = (n + 1) * n_el
                ket_s = m * n_el
                ket_e = (m + 1) * n_el

                self.PCQED_H_PF[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m)
                )
                self.PCQED_H_BLC[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m)
                )

                m = n - 1
                bra_s = n * n_el
                bra_e = (n + 1) * n_el
                ket_s = m * n_el
                ket_e = (m + 1) * n_el

                self.PCQED_H_PF[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )
                self.PCQED_H_BLC[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )

        eigs, vecs = np.linalg.eigh(self.PCQED_H_PF)
        self.PCQED_pf_eigs = np.copy(eigs)
        self.PCQED_pf_vecs = np.copy(vecs)

    def build_d_array(
        self,
        n_el,
        lambda_vector,
        mu_array,
        coherent_state=False,
        upper_triangular_Mu_array=False,
    ):
        """
        method to compute the array d = \lambda \cdot \mu if coherent_state==False
        or d = \lambda \cdot (\mu - <\mu>) if coherent_state == True
        and store to attribute self.d_array
        """

        if coherent_state == False:
            self.d_array = np.einsum(
                "k,ijk->ij", lambda_vector, mu_array[:n_el, :n_el, :]
            )

        else:
            _I = np.eye(n_el)
            self.d_array = np.einsum(
                "k,ijk->ij", lambda_vector, mu_array[:n_el, :n_el, :]
            )
            _d_exp = self.d_array[0, 0]
            self.d_array = self.d_array - _I * _d_exp

        if upper_triangular_Mu_array:
            self.d_array = (
                self.d_array + self.d_array.T - np.diag(np.diag(self.d_array))
            )
