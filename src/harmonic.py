import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from numpy import trapz
from scipy.special import hermite
from math import factorial


class Harmonic:
    """A class representing the Harmonic oscillator model of a diatomic."""

    def __init__(self, mu, k, re, Te=0):
        # store reduced mass in atomic units from input
        self.mu_au = mu

        # store force constant in atomic units from input
        self.k_au = k

        # compute omega in atomic units
        self.omega_au = np.sqrt(self.k_au / self.mu_au)

        # compute alpha in atomic units
        self.alpha_au = self.mu_au * self.omega_au

        # store r_eq from input
        self.r_eq_au = re

        # store Te from input
        self.Te_au = Te

    def make_r_grid(self, n_points=500, fac=0.9):
        r_min = self.r_eq_au - fac * self.r_eq_au
        r_max = self.r_eq_au + fac * self.r_eq_au
        self.r = np.linspace(r_min, r_max, n_points)

    def N(self, n):
        return np.sqrt(1 / (2**n * factorial(n))) * (self.alpha_au / np.pi) ** (1 / 4)

    def psi(self, n):
        Hr = hermite(n)
        psi_n = (
            N(n)
            * Hr(np.sqrt(self.alpha_au) * (self.r - self.r_eq_au))
            * np.exp(-0.5 * self.alpha_au * (self.r - self.r_eq_au) ** 2)
        )
        return psi_n
