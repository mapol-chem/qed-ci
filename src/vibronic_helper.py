from helper_PFCI import PFHamiltonianGenerator
import numpy as np
from scipy import optimize

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
