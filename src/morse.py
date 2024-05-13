import numpy as np
from scipy.constants import h, hbar, c, u
from scipy.special import factorial
from scipy.special import genlaguerre, gamma

# Factor for conversion from cm-1 to J
FAC = 100 * h * c

class Morse:
    """A class representing the Morse oscillator model of a diatomic."""

    def __init__(self, mA, mB, we, wexe, re, Te=0):
        """Initialize the Morse model for a diatomic molecule.

        mA, mB are the atom masses (atomic mass units).
        we, wexe are the Morse parameters (cm-1).
        re is the equilibrium bond length (m).
        Te is the electronic energy (minimum of the potential well; origin
            of the vibrational state energies).

        """
        # atomic mass units to kg
        self.amu_to_kg = 1.66054e-27
        
        # angstroms to meters
        self.ang_to_m = 1e-10
        
        # electron volts to Jouls
        self.eV_to_J = 1.60218e-19
        
        # electron volts to atomic units of energy (Hartrees)
        self.eV_to_au = 1 / 27.211 #0.0367493
        
        # angstroms to atomic units of length (Bohr radii)
        self.au_to_ang = 0.52917721067121

        # meters to atomic units
        self.m_to_au = 1 / self.ang_to_m * 1 / self.au_to_ang

        
        # atomic mass units to atomic units of mass
        self.amu_to_au = 1822.89

        # masses originally in atomic mass units
        self.mA, self.mB = mA, mB

        # reduced mass in SI units
        self.mu = mA*mB/(mA+mB) * u

        #  Morse parameters in wavenumbers
        self.we, self.wexe = we, wexe

        # equilibrium bondlength in SI units
        self.re = re

        # energy offset in atomic units
        self.Te = Te

        # dissociation energy in Joules
        self.De = we**2 / 4 / wexe * FAC

        # force constant 
        self.ke = (2 * np.pi * c * 100 * we)**2 * self.mu

        #  Morse parameters, a and lambda.
        self.a = self.calc_a()
        self.lam = np.sqrt(2 * self.mu * self.De) / self.a / hbar

        # Maximum vibrational quantum number.
        self.vmax = int(np.floor(self.lam - 0.5))

        # grid in SI
        self.make_rgrid(n = 1000, rmin = 3e-11, rmax = 6e-10)

        # potential in SI
        self.V = self.Vmorse(self.r)

        # potential in atomic units
        self.V_au = self.V / self.eV_to_J * self.eV_to_au

        # grid in atomic units
        self.r_au = self.r * self.m_to_au

        self.r_eq_au = re * self.m_to_au

        # parameter z in SI
        self.z = 2 * self.lam * np.exp(-self.a * (self.r - self.re))
    

    def make_rgrid(self, n=500, rmin=None, rmax=None, retstep=False):
        """Make a suitable grid of internuclear separations."""

        print(" Did I get the values of rmin and rmax right?")
        print(rmin, rmax)
        self.rmin, self.rmax = rmin, rmax
        if rmin is None:
            # minimum r where V(r)=De on repulsive edge
            self.rmin = self.re - 1.75 * np.log(2) / self.a
        if rmax is None:
            # maximum r where V(r)=f.De
            f = 0.999
            self.rmax = self.re - 1.25 *  np.log(1-f)/self.a
        self.r, self.dr = np.linspace(self.rmin, self.rmax, n,
                                      retstep=True)
        print(" just formed self.r")
        print(" First point is ", self.r[0])
        print(" Last point is ", self.r[n-1])
        if retstep:
            return self.r, self.dr
        return self.r

    def calc_a(self):
        """Calculate the Morse parameter, a.

        Returns the Morse parameter, a, from the equilibrium
        vibrational wavenumber, we in cm-1, and the dissociation
        energy, De in J.

        """

        return (self.we * np.sqrt(2 * self.mu/self.De) * np.pi *
                c * 100)

    def Vmorse(self, r):
        """Calculate the Morse potential, V(r).

        Returns the Morse potential at r (in m) for parameters De
        (in J), a (in m-1) and re (in m).

        """

        return self.De * (1 - np.exp(-self.a*(r - self.re)))**2

    def Emorse(self, v):
        """Calculate the energy of a Morse oscillator in state v.

        Returns the energy of a Morse oscillator parameterized by
        equilibrium vibrational frequency we and anharmonicity
        constant, wexe (both in cm-1).

        """
        vphalf = v + 0.5
        return (self.we * vphalf - self.wexe * vphalf**2) * FAC

    def calc_turning_pts(self, E):
        """Calculate the classical turning points at energy E.

        Returns rm and rp, the classical turning points of the Morse
        oscillator at energy E (provided in J). rm < rp.

        """

        b = np.sqrt(E / self.De)
        return (self.re - np.log(1+b) / self.a,
                self.re - np.log(1-b) / self.a)

    def calc_psi(self, v, r=None, normed=True, psi_max=1):
        """Calculates the Morse oscillator wavefunction, psi_v.

        Returns the Morse oscillator wavefunction at vibrational
        quantum number v. The returned function is "normalized" to
        give peak value psi_max.

        """

        if r is None:
            r = self.r
        z = 2 * self.lam * np.exp(-self.a*(r - self.re))
        alpha = 2*(self.lam - v) - 1
        psi = (z**(self.lam-v-0.5) * np.exp(-z/2) *
               genlaguerre(v, alpha)(z))
        rho = psi * np.conj(psi)
	#psi *= np.conj(psi)
        psi *= psi_max / np.sqrt(np.max(rho))
        return psi

    def calc_psi_z(self, v):
        z = self.z 
        alpha = 2*(self.lam - v) - 1
        psi = (z**(self.lam-v-0.5) * np.exp(-z/2) *
               genlaguerre(v, alpha)(z))
        Nv = np.sqrt(factorial(v) * (2*self.lam - 2*v - 1) /
                     gamma(2*self.lam - v))
        
        self.psi_si = psi * Nv
        self.norm_au = np.trapz(self.psi_si ** 2, self.r_au)
        self.psi_au = self.psi_si / np.sqrt(self.norm_au)
        self.norm_si = np.trapz(self.psi_si ** 2, self.r)
        self.psi_si /= np.sqrt(self.norm_si)
        return Nv * psi

    def plot_V(self, ax, **kwargs):
        """Plot the Morse potential on Axes ax."""

        ax.plot(self.r*1.e10, self.V / FAC + self.Te, **kwargs)

    def get_vmax(self):
        """Return the maximum vibrational quantum number."""

        return int(self.we / 2 / self.wexe - 0.5)

    def draw_Elines(self, vlist, ax, **kwargs):
        """Draw lines on Axes ax representing the energy level(s) in vlist."""

        if isinstance(vlist, int):
            vlist = [vlist]
        for v in vlist:
            E = self.Emorse(v)
            rm, rp = self.calc_turning_pts(E)
            ax.hlines(E / FAC + self.Te, rm*1.e10, rp*1e10, **kwargs)

    def label_levels(self, vlist, ax):
        if isinstance(vlist, int):
            vlist = [vlist]

        for v in vlist:
            E = self.Emorse(v)
            rm, rp = self.calc_turning_pts(E)
            ax.text(s=r'$v={}$'.format(v), x=rp*1e10 + 0.6,
                    y=E / FAC + self.Te, va='center')

    def plot_psi(self, vlist, ax, r_plot=None, scaling=1, **kwargs):
        """Plot the Morse wavefunction(s) in vlist on Axes ax."""
        if isinstance(vlist, int):
            vlist = [vlist]
        for v in vlist:
            E = self.Emorse(v)
            if r_plot is None:
                rm, rp = self.calc_turning_pts(E)
                x = self.r[self.r<rp*1.2]
            else:
                x = r_plot
            psi = self.calc_psi(v, r=x, psi_max=self.we/2)
            psi_plot = psi*scaling + self.Emorse(v)/FAC + self.Te
            ax.plot(x*1.e10, psi_plot, **kwargs)
