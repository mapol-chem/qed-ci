from helper_PFCI import PFHamiltonianGenerator
import numpy as np
from scipy.optimize import minimize
from scipy.constants import h, hbar, c, u
from scipy.special import factorial
from scipy.special import genlaguerre, gamma
import json

class Vibronic:
    """A class for performing geometry optimization and vibrational analysis with QED methods"""

    def __init__(self, opt_dict):
        # make sure all values are lower case
        opt_dict = {k.lower(): v for k, v in opt_dict.items()}
        print(opt_dict)
        print("Going to parse options!")
        self.wn_to_J = 100 * h * c

        self.parseOptions(opt_dict)

    def parseOptions(self, options_dictionary):
        # do we want to only keep singlets from qed-ci?
        # needs a bit more work to keep only singlets from pcqed
        if "only_singlets" in options_dictionary:
            self.only_singlets = options_dictionary["only_singlets"]
        else:
            self.only_singlets = False

        if "damping_factor" in options_dictionary:
            self.damping_factor = options_dictionary["damping_factor"]
        else:
            self.damping_factor = 1
        if "molecule_template" in options_dictionary:
            self.molecule_template = options_dictionary["molecule_template"]
            print(self.molecule_template)
        else:
            print(f"molecule_template!  Please restart with a proper molecule_template")
            exit()
        if "guess_bondlength" in options_dictionary:
            self.r = np.array([options_dictionary["guess_bondlength"]])
        else:
            print(f"guess_bondlength not specified!  Please restart with a guess r")
            exit()
        if "qed_type" in options_dictionary:
            self.qed_type = options_dictionary["qed_type"]
        else:
            self.qed_type = "qed-ci"
        if "ci_level" in options_dictionary:
            self.ci_level = options_dictionary["ci_level"]
        else:
            self.ci_level = "fci"
        if "basis" in options_dictionary:
            self.orbital_basis = options_dictionary["basis"]
        else:
            self.orbital_basis = "sto-3g"
        if "number_of_photons" in options_dictionary:
            self.number_of_photons = options_dictionary["number_of_photons"]
            print(f" SET NUMBER OF PHOTONS TO {self.number_of_photons} ")
        else:
            self.number_of_photons = 0
        if "number_of_electronic_states" in options_dictionary:
            self.number_of_electronic_states = options_dictionary[
                "number_of_electronic_states"
            ]
        else:
            self.number_of_electronic_states = 10
        if "omega" in options_dictionary:
            self.omega = options_dictionary["omega"]
        else:
            self.omega = 0.0

        if "lambda_vector" in options_dictionary:
            self.lambda_vector = np.array(options_dictionary["lambda_vector"])
        else:
            self.lambda_vector = np.array([0, 0, 0])
        # which root do you want to follow for geometry opt / frequency analysis?
        if "target_root" in options_dictionary:
            self.target_root = options_dictionary["target_root"]
        else:
            # default is ground state
            self.target_root = 0
        if "r_step" in options_dictionary:
            self.dr = options_dictionary["r_step"]
        else:
            self.dr = 0.0005
        if "mass_a" in options_dictionary:
            self.mA = options_dictionary["mass_a"]
            print(f" Mass of atom A is {self.mA} AMUs")
        else:
            print(
                "mass_A not defined!  Please restart and specify both mass_a and mass_a in amu"
            )
        if "mass_b" in options_dictionary:
            self.mB = options_dictionary["mass_b"]
            print(f" Mass of atom B is {self.mB} AMUs")
        else:
            print(
                "mass_B not defined!  Please restart and specify both mass_a and mass_a in amu"
            )
        if "gradient_tol" in options_dictionary:
            self.gradient_tol = options_dictionary["gradient_tol"]
        else:
            self.gradient_tol = 1e-4
        if "step_tol" in options_dictionary:
            self.step_tol = options_dictionary["step_tol"]
        else:
            self.step_tol = 0.5

        # self.mA = 1
        # self.mB = 1
        print(self.qed_type)
        self.zmatrix_string = self.molecule_template.replace("**R**", str(self.r[0]))
        self.mu_AMU = self.mA * self.mB / (self.mA + self.mB)

        self.amu_to_au = 1822.89
        self.mu_au = self.mu_AMU * self.amu_to_au
        self.mu_SI = self.mu_AMU * u

        if self.ci_level == "cas":
            if "nact_orbs" in options_dictionary:
                self.nact_orbs = options_dictionary["nact_orbs"]
            else:
                print(
                    " Specification of the number of active orbitals 'nact_orbs' needed"
                )
                exit()
            if "nact_els" in options_dictionary:
                self.nact_els = options_dictionary["nact_els"]
            else:
                print(
                    " Specification of the number of active electrons 'nact_els' needed"
                )
                exit()

    def compute_qed_gradient(self, r0):
        # copy r0 element to a value
        rv = r0[0]
        # displaced geomeries in angstroms
        r_array = np.array(
            [
                rv - 2 * self.dr,
                rv - self.dr,
                rv,
                rv + self.dr,
                rv + 2 * self.dr,
            ]
        )
        self.au_to_ang = 0.52917721067121

        h = self.dr / self.au_to_ang
        f = np.zeros(5)
        for i in range(5):
            self.r = np.copy(np.array([r_array[i]]))
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

        # energy at mindpoint
        self.f = f[2]
        # return self.r to midpoint
        self.r = np.copy(r0)
        # going to return energy and gradient
        return f[2], self.f_x

    def compute_qed_energy(self, properties=False):
        # added optional properties flag so that the QED-CI can compute dipole moments
        # make sure geometry is up to date!
        self.zmatrix_string = self.molecule_template.replace("**R**", str(self.r[0]))
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
                "davidson_roots": self.number_of_electronic_states,
                "canonical_mos": True,
                "coherent_state_basis": False,
                "full_diagonalization": False,
                "number_of_photons": self.number_of_photons,
                "nact_orbs": 0,
                "nact_els": 0,
                "compute_properties": properties,
                "check_rdms": False,
            }
            if self.ci_level == "cas":
                cavity_dict["nact_orbs"] = self.nact_orbs
                cavity_dict["nact_els"] = self.nact_els

            print(self.zmatrix_string, options_dict, cavity_dict)
            qed_ci_inst = PFHamiltonianGenerator(
                self.zmatrix_string, options_dict, cavity_dict
            )
            if self.only_singlets:
                self.qed_energies = np.copy(qed_ci_inst.CISingletEigs)
                if properties:
                    self.qed_dipole_moments = np.copy(qed_ci_inst.singlet_dipole_array)
                    self.qed_dipole_dim = qed_ci_inst.singlet_count
                return qed_ci_inst.CISingletEigs[self.target_root]
            else:
                self.qed_energies = np.copy(qed_ci_inst.CIeigs)
                if properties:
                    self.qed_dipole_moments = np.copy(qed_ci_inst.dipole_array)
                    self.qed_dipole_dim = len(qed_ci_inst.CIeigs)
                return qed_ci_inst.CIeigs[self.target_root]
            
        elif self.qed_type == "qed-pt" or self.qed_type == "pcqed":
            properties = True
            davidson_thresh = 1e-9

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
                "davidson_roots": self.number_of_electronic_states,
                "canonical_mos": True,
                "coherent_state_basis": False,
                "full_diagonalization": False,
                "number_of_photons": 0,
                "nact_orbs": 0,
                "nact_els": 0,
                "davidson_threshold" : davidson_thresh,
                "compute_properties": True,
                "check_rdms": False,
            }
            if self.ci_level == "cas":
                cavity_dict["nact_orbs"] = self.nact_orbs
                cavity_dict["nact_els"] = self.nact_els

            print(self.zmatrix_string, options_dict, cavity_dict)

            qed_ci_inst = PFHamiltonianGenerator(
                self.zmatrix_string, options_dict, cavity_dict
            )
            if self.only_singlets:
                _N_el = qed_ci_inst.singlet_count
                _energies = np.copy(qed_ci_inst.CISingletEigs)
                _dipoles = np.copy(qed_ci_inst.singlet_dipole_array)
                self.qed_dipole_moments = np.copy(qed_ci_inst.singlet_dipole_array)
                self.qed_dipole_dim = qed_ci_inst.singlet_count
            else:
                _N_el = self.number_of_electronic_states
                _energies = np.copy(qed_ci_inst.CIeigs)
                _dipoles = np.copy(qed_ci_inst.dipole_array)
                self.qed_dipole_moments = np.copy(qed_ci_inst.dipole_array)
                self.qed_dipole_dim = len(_energies)

            print(F' JUST RAN Instantiated PFHamiltonianGenerator')
            print(F' CHECKING DIPOLES')
            print(F' DIPOLES [:4,:4] IN .dipole_array')
            print(qed_ci_inst.dipole_array[:4,:4,:] )
            print(F' DIPOLES [:4,:4] IN .singlet_dipole_array')
            print(qed_ci_inst.singlet_dipole_array[:4,:4,:] )
            print(F' DIPOLES [:4,:4] IN _dipoles')
            print(_dipoles[:4,:4,:])
            print(F' DIPOLES [:4,:4] IN self.qed_dipole_moments')
            print(self.qed_dipole_moments[:4,:4,:])


            if self.qed_type == "pcqed":
                # this call should work for both singlet-only and singlet + triplet
                self.fast_build_pcqed_pf_hamiltonian(
                    _N_el,
                    self.number_of_photons + 1,
                    self.omega,
                    self.lambda_vector,
                    _energies,
                    _dipoles
                )
                self.qed_energies = np.copy(self.PCQED_pf_eigs)
                return self.PCQED_pf_eigs[self.target_root]
            
            elif self.qed_type == "qed-pt":
                print(" GOING TO COMPUTE PERTURBATIVE CORRECTIONS")
                E_0_so = self.compute_qed_pt_energy(_N_el, self.omega, self.lambda_vector, _energies, _dipoles, state_index = 0, order=2)
                print(F' ZEROTH ORDER ENERGY IS {_energies[0]}')
                print(F' FIRST ORDER CORRECTION IS {self.first_order_energy_correction}')
                print(F' SECOND ORDER CORRECTION IS {self.second_order_energy_correction}')
                #E_0_so = _energies[0] + self.first_order_energy_correction + self.second_order_energy_correction
                print(F' ENERGY TO SECOND ORDER IS {E_0_so}')


    

    def optimize_geometry_full_nr(self):
        print(
            f" Going to perform a full Newton-Raphson geometry optimization using {self.qed_type}"
        )
        energy_start, gradient_start = self.compute_qed_gradient(self.r)
        print(f" Initial Energy is     {energy_start}")
        print(f" Initial bondlength is {self.r}")
        print(f" Initial Gradient is   {gradient_start}")
        print(f" Initial Hessian is    {self.f_xx}")
        _delta_x = -self.f_x / self.f_xx
        print(f" Initial Update is     {_delta_x} ")

        iter_count = 1
        while np.abs(self.f_x) > self.gradient_tol and iter_count < 200:
            if np.abs(_delta_x) < self.step_tol:
                self.r[0] += self.damping_factor * _delta_x
            else:
                self.r[0] += _delta_x / np.abs(_delta_x) * self.step_tol
            energy, gradient = self.compute_qed_gradient(self.r)
            print(f" Geometry Update {iter_count}")
            print(f" Bondlength:     {self.r[0]}")
            print(f" Energy:         {energy}")
            print(f" Gradient:       {gradient}")
            print(f" Hessian:        {self.f_xx}")
            _delta_x = -self.f_x / self.f_xx
            print(f" Update:         {_delta_x}")
            iter_count += 1
        self.r_eq_SI = self.r[0] * 1e-10

    def compute_potential_scan(
        self, r_min=0.5, r_max=2.5, N_points=50, filename="test"
    ):
        r_array = np.linspace(r_min, r_max, N_points)
        # this will typically be larger than it needs to be if we are only selecting singlets
        pes_write_array = np.zeros((self.number_of_electronic_states+1, N_points))
        dipole_write_array = np.zeros((self.number_of_electronic_states, self.number_of_electronic_states, 3, N_points))

        json_file_name = filename + ".json"
        mu_npy_file_name = filename + "_dipoles.npy"
        en_npy_file_name = filename + "_pes.npy"

        json_dict =   {
            "molecule"  : {
                "geometry_template" : self.molecule_template,
                "bond_length" : []
            },
            "model" : {
                "method" : self.qed_type,
                "orbital_basis" : self.orbital_basis,
                "number_of_photon_states" : str(self.number_of_photons),
                "number_of_electronic_states" : str(self.number_of_electronic_states),
                "lambda" : list(self.lambda_vector),
                "omega" : self.omega,
            },
            "return_result" : {
                "bond_length" : [],
                "energy" : [],
                "dipole_moment_file" : filename,
            },
        }

        # perform the scan
        for i in range(N_points):
            self.r[0] = r_array[i]
            self.compute_qed_energy(properties=True)
            _mu_dim = self.qed_dipole_dim
            # store dipoles to numpy array for .npy dump
            dipole_write_array[:_mu_dim,:_mu_dim,:,i] = np.copy(self.qed_dipole_moments)
            pes_write_array[0,i] = r_array[i]
            pes_write_array[1:, i] = np.copy(self.qed_energies)

            # store to json_dict for json write
            json_dict["molecule"]["bond_length"].append(r_array[i])
            json_dict["return_result"]["bond_length"].append(r_array[i])
            json_dict["return_result"]["energy"].append(list(self.qed_energies))

        np.save(mu_npy_file_name, dipole_write_array)
        np.save(en_npy_file_name, pes_write_array)

        return json_dict
        ### Uncomment to write json!
        #json_object = json.dumps(json_dict, indent=4)
        #with open(json_file_name, "w") as outfile:
        #    outfile.write(json_object)



    def optimize_geometry(self):
        print(f" Going to perform a geometry optimization using {self.qed_type}")
        energy_start, gradient_start = self.compute_qed_gradient(self.r)
        print(f" Initial Energy is     {energy_start}")
        print(f" Initial bondlength is {self.r}")
        print(f" Initial Gradient is     {gradient_start}")

        optimize_return = minimize(
            self.compute_qed_gradient,
            self.r,
            method="BFGS",
            jac=True,
        )

        # make sure we capture the optimized bond length
        self.r = np.copy(optimize_return.x)
        print(f" Geometry Optimization Complete!")
        energy_end, gradient_end = self.compute_qed_gradient(self.r)
        print(f" Final Energy is     {energy_end}")
        print()
        print(f" Final bondlength is {self.r}")
        print(f" Final gradient is   {gradient_end} ")
        self.r_eq_SI = self.r[0] * 1e-10

    def compute_morse_parameters(self):
        # assume current bondlength is the equilibrium!
        self.r_eq_SI = self.r[0] * 1e-10
        # compute the second and third derivatives
        self.compute_qed_gradient(self.r)
        print(" Going to compute vibrational frequencies")
        print(f" Current bondlength is  {self.r[0]} ")
        print(f" Current Energy is      {self.f}")
        print(f" Current Gradient is    {self.f_x} ")
        print(f" Current Hessian is     {self.f_xx} ")
        print(f" Current 3rd Deriv is   {self.f_xxx} ")

        # compute beta and De parameters from 2nd and 3rd derivatives
        self.morse_beta_au = -2 * self.f_xxx / (6 * self.f_xx)
        self.morse_De_au = self.f_xx / (2 * self.morse_beta_au**2)

        # compute omega_e and xe in atomic units
        self.morse_omega_au = np.sqrt(
            2 * self.morse_De_au * self.morse_beta_au**2 / self.mu_au
        )
        # this is unitless!
        self.morse_xe = self.morse_omega_au / (4 * self.morse_De_au)

        # convert to wavenumbers
        self.au_to_wn = 219474.63068
        self.morse_De_wn = self.morse_De_au * self.au_to_wn
        self.morse_De_J = self.morse_De_wn * self.wn_to_J

        # conversion from au to meters
        self.au_to_m = 5.29177210903e-11
        self.r_eq_au = self.r_eq_SI / self.au_to_m

        self.morse_omega_wn = self.morse_omega_au * self.au_to_wn
        print(f" Morse we:           {self.morse_omega_wn} cm^-1")
        print(f" Morse wexe:         {self.morse_omega_wn * self.morse_xe} cm^-1")

        self.harmonic_omega_au = np.sqrt(self.f_xx / self.mu_au)
        self.harmonic_omega_wn = self.harmonic_omega_au * self.au_to_wn

        # electronic energy in cm^-1
        self.Te_wn = self.f * self.au_to_wn

        E_1_morse = (
            self.morse_omega_au * 3 / 2
            - self.morse_omega_au * self.morse_xe * 3 / 2**2
        )
        E_0_morse = (
            self.morse_omega_au * 1 / 2
            - self.morse_omega_au * self.morse_xe * 1 / 2**2
        )
        morse_fundamental_au = E_1_morse - E_0_morse
        morse_fundamental_wn = morse_fundamental_au * self.au_to_wn
        print(f" Harmonic Fundamental Frequency: {self.harmonic_omega_wn} cm^-1")
        print(f" Morse Fundamental Frequency:    {morse_fundamental_wn} cm^-1")
        print(f" Electronic Energy:              {self.Te_wn} cm^-1")

        # compute classical turning point solving
        # k x_0^2 - 2 k x_eq x_0 + (x_eq^2 - sqrt{k / mu})
        # 
        a = self.f_xx 
        b = -2 * self.f_xx * self.r_eq_au
        c = self.f_xx * self.r_eq_au ** 2 - self.harmonic_omega_au

        self.x0_p_au = (-b + np.sqrt(b ** 2 - 4 * a * c) ) / (2 * a)
        self.x0_m_au = (-b - np.sqrt(b ** 2 - 4 * a * c) ) / (2 * a)

        # let's also define x_0 as offsets from equilibrium, e.g. solve for when r_eq = 0:
        self.x0_offset_p_au = ( 1 / (self.f_xx * self.mu_au) ) ** (1/4)
        self.x0_offset_m_au = -( 1 / (self.f_xx * self.mu_au) ) ** (1/4)

        assert np.isclose(self.x0_p_au - self.r_eq_au, self.x0_offset_p_au)
        assert np.isclose(self.x0_m_au - self.r_eq_au, self.x0_offset_m_au)
        

        # check to make sure this checks out using 1/2 k (x0 - x_e) ** 2 = 1/2 sqrt(k/mu) -> k (x0 - x_e) ** 2 = sqrt(k/mu)
        V_x0_p = self.f_xx * (self.x0_p_au - self.r_eq_au) ** 2
        V_x0_m = self.f_xx * (self.x0_m_au - self.r_eq_au) ** 2

        assert np.isclose(V_x0_m, self.harmonic_omega_au)
        assert np.isclose(V_x0_p, self.harmonic_omega_au)



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
        _dim = n_el * n_ph
        print(f" DIMENSIONS OF THE PCQED HAMILTONIAN: {_dim}")

        self.PCQED_H_PF = np.zeros((_dim, _dim))
        self.PCQED_H_EL = np.zeros((_dim, _dim))
        self.PCQED_H_PH = np.zeros((_dim, _dim))
        self.PCQED_H_DSE = np.zeros((_dim, _dim))
        self.PCQED_H_BLC = np.zeros((_dim, _dim))
        self.PCQED_MU = np.zeros((_dim, _dim))

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
        print(f" SHAPE OF _d ARRAY: {np.shape(_d)}")

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

                print(
                    f" n : {n}, bra_s : {bra_s}, bra_e : {bra_e}, ket_s : {ket_s}, ket_e : {ket_e}"
                )
                print("Printing shape")
                print(np.shape(np.sqrt(omega / 2) * _d * np.sqrt(m)))

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

                print(
                    f" n : {n}, bra_s : {bra_s}, bra_e : {bra_e}, ket_s : {ket_s}, ket_e : {ket_e}"
                )
                print("Printing shape")
                print(np.shape(np.sqrt(omega / 2) * _d * np.sqrt(m)))
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

                print(
                    f" n : {n}, bra_s : {bra_s}, bra_e : {bra_e}, ket_s : {ket_s}, ket_e : {ket_e}"
                )
                print("Printing shape")
                print(np.shape(np.sqrt(omega / 2) * _d * np.sqrt(m)))

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

                print(
                    f" n : {n}, bra_s : {bra_s}, bra_e : {bra_e}, ket_s : {ket_s}, ket_e : {ket_e}"
                )
                print("Printing shape")
                print(np.shape(np.sqrt(omega / 2) * _d * np.sqrt(m)))
                self.PCQED_H_PF[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )
                self.PCQED_H_BLC[bra_s:bra_e, ket_s:ket_e] = (
                    -np.sqrt(omega / 2) * _d * np.sqrt(m + 1)
                )

        eigs, vecs = np.linalg.eigh(self.PCQED_H_PF)
        self.PCQED_pf_eigs = np.copy(eigs)
        self.PCQED_pf_vecs = np.copy(vecs)

    def compute_qed_pt_energy(self, n_el, omega, lambda_vector, E_array, mu_array, state_index = 0, order=2):
        # defaults to second order for the ground state
        self.build_d_array(
            n_el, lambda_vector, mu_array, coherent_state=False
        )
        print("Printing Mu Array")
        print(mu_array[:4,:4,:])

        print("Printing D array")
        print(self.d_array[:4,:4])

        if order==2:
            # compute first order energy correction
            self.compute_first_order_energy_correction(n_el, state_index = 0)
            
            # compute second order energy correction
            self.compute_second_order_energy_correction(n_el, omega, E_array, state_index = 0)

            return E_array[0] + self.first_order_energy_correction + self.second_order_energy_correction
    

    def compute_first_order_energy_correction(self,n_el, state_index = 0):

        # defaults to the ground state
        _N = state_index
        
        E_n_1 = 0
        for g in range(n_el):
            E_n_1 += self.d_array[_N, g] * self.d_array[g, _N]

        # einsum
        E_n_1_es = np.einsum("i,i->", self.d_array[_N,:], self.d_array[:,_N], optimize=True)
        
        self.first_order_energy_correction = 0.5 * E_n_1
        assert np.isclose(E_n_1_es, E_n_1)

    
    def compute_second_order_energy_correction(self, n_el, omega, E_array, state_index = 0):

        if state_index != 0:
            print(" We can't handle PT2 for excited states yet!")
            exit()

        # defaults to ground electronic state
        mu_n = state_index

        # prepare inverse E_mu_n - E_mu_m array
        E_mn = np.zeros_like(E_array)
        E_mn_min_omega = np.zeros_like(E_array)

        # again assumes ground-state
        E_mn[1:] = 1 / (E_array[mu_n] - E_array[1:])
        E_mn_min_omega = 1 / (E_array[mu_n] - E_array - omega)


        # defaults to zero photon s
        m_n = 0
        blc_term_1 = 0 
        blc_term_1_num = 0
        blc_term_1_den = 0
        dse_term = 0

        # blc first term - note there is no restriction on the electronic index in this sum
        for mu_m in range(n_el):
            # ml = mn+1
            blc_term_1 += omega / 2 * (self.d_array[mu_m, mu_n] * np.sqrt(m_n + 1)) ** 2 / (E_array[mu_n] - E_array[mu_m] - omega)
            blc_term_1_num += (self.d_array[mu_m, mu_n] * np.sqrt(m_n + 1)) # ** 2 
            blc_term_1_den += 1 / (E_array[mu_n] - E_array[mu_m] - omega)

        # sum numerator and denominator of blc term 1 separately
        blc_t1_num_es = np.einsum("i->", self.d_array[:,mu_n] * np.sqrt(m_n + 1), optimize=True)
        blc_t1_den_es = np.einsum("i->", E_mn_min_omega, optimize=True)

        print(F'Loop-based BLC Unsquared Numerator: {blc_term_1_num}')
        print(F'EinS-based BLC Unsquared Numerator: {blc_t1_num_es}')
        print(F'Loop-based BLC Unsquared Denominat: {blc_term_1_den}')
        print(F'EinS-based BLC Unsquared Denominat: {blc_t1_den_es}')

        blc_t1_es = omega / 2 * blc_t1_num_es ** 2 * blc_t1_den_es

        dse_num_es = np.einsum("ij,i->", self.d_array, self.d_array[:,mu_n], optimize=True) - np.einsum("i,i->", self.d_array[mu_n,:], self.d_array[:,mu_n], optimize=True)
        dse_den_es = np.einsum("i->", E_mn)

        dse_es = 1 / 4 * dse_num_es ** 2 * dse_den_es


        # dse term
        for mu_m in range(n_el):
            if mu_m != mu_n:
                dse_inner = 0
                for g in range(n_el):
                    dse_inner += self.d_array[mu_m, g] * self.d_array[g, mu_n]
                    
                dse_term += 1/4 * dse_inner ** 2 / (E_array[mu_n] - E_array[mu_m])

        self.second_order_energy_correction = blc_term_1 + dse_term
        second_order_correction_es = dse_es + blc_t1_es
        print(F'Loop based blc   {blc_term_1}')
        print(F'Einsum based blc {blc_t1_es}')
        print(F'Loop based dse   {dse_term}')
        print(F'Einsum based blc {dse_es}')
        #assert np.isclose(second_order_correction_es, self.second_order_energy_correction)

        



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
