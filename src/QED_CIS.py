"""
A Psi4 input script to compute Full Configuration Interaction from a SCF reference

Requirements:
SciPy 0.13.0+, NumPy 1.7.2+

References:
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_cqed_rhf import cqed_rhf

# Check energy against psi4?
compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol_str = """
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
"""

options_dict = {'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8
                  }

# define the lambda vector
lambda_vector = np.array([0, 0, 0.01])

mol = psi4.geometry(mol_str)


psi4.set_options(options_dict)

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# now compute cqed-rhf to get transformation vectors with cavity
cqed_rhf_dict = cqed_rhf(lambda_vector, mol_str, options_dict)

# grab necessary quantities from cqed_rhf_dict
scf_e = cqed_rhf_dict["RHF ENERGY"]
cqed_scf_e = cqed_rhf_dict["CQED-RHF ENERGY"]
wfn = cqed_rhf_dict["PSI4 WFN"]
C = cqed_rhf_dict["CQED-RHF C"]
D = cqed_rhf_dict["CQED-RHF DENSITY MATRIX"]
eps = cqed_rhf_dict["CQED-RHF EPS"]
dc = cqed_rhf_dict["DIPOLE ENERGY"]

# collect rhf wfn object as dictionary
wfn_dict = psi4.core.Wavefunction.to_file(wfn)

# update wfn_dict with orbitals from CQED-RHF
wfn_dict["matrix"]["Ca"] = C
wfn_dict["matrix"]["Cb"] = C
# update wfn object
wfn = psi4.core.Wavefunction.from_file(wfn_dict)

# Grab data from wavfunction class
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()

# Compute size of Hamiltonian in GB
from scipy.special import comb
nDet = comb(nmo, ndocc)**2
H_Size = nDet**2 * 8e-9
print('\nSize of the Hamiltonian Matrix will be %4.2f GB.' % H_Size)
if H_Size > numpy_memory:
    #clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (H_Size, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())

# JJF To-do here:
# - get quadrupole integrals
# - dot them into lambda vector 
# - add them to H below, transform to CQED-RHF basis
# quadrupole arrays
Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

# Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
Q_PF = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
Q_PF -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
Q_PF -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

# accounting for the fact that Q_ij = Q_ji
# by weighting Q_ij x 2 which cancels factor of 1/2
Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

H = 0 * np.asarray(mints.ao_kinetic()) + 0 * np.asarray(mints.ao_potential()) # + Q_PF
#print(H)

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))
print(MO[0,0,ndocc:,ndocc:])

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)
#print(H)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

from helper_CI import Determinant, HamiltonianGenerator, compute_excitation_level
from itertools import combinations

print('Generating %d Full CI Determinants...' % (nDet))
t = time.time()
detList = []
for alpha in combinations(range(nmo), ndocc):
    alpha_ex_level = compute_excitation_level(alpha, ndocc)
    for beta in combinations(range(nmo), ndocc):
        beta_ex_level = compute_excitation_level(beta, ndocc)
        if alpha_ex_level + beta_ex_level == 1:
            #print(F' adding alpha: {alpha} and beta: {beta}\n') 
            detList.append(Determinant(alphaObtList=alpha, betaObtList=beta))

print('..finished generating determinants in %.3f seconds.\n' % (time.time() - t))
print(F'..there are {len(detList)} determinants \n')

print('Generating Hamiltonian Matrix...')

t = time.time()
Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)
print("printing H")
#print(Hamiltonian_matrix)

print('..finished generating Matrix in %.3f seconds.\n' % (time.time() - t))

print('Diagonalizing Hamiltonian Matrix...')

t = time.time()

e_fci, wavefunctions = np.linalg.eigh(Hamiltonian_matrix)
print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))
for i in range(0,10):
    en = e_fci[i] #+mol.nuclear_repulsion_energy()+dc
    print(F'{en}\n')
fci_mol_e = e_fci[0] + mol.nuclear_repulsion_energy()

print('# Determinants:     % 16d' % (len(detList)))

print('SCF energy:         % 16.10f' % (scf_e))
print('FCI correlation:    % 16.10f' % (fci_mol_e - scf_e))
print('Total FCI energy:   % 16.10f' % (fci_mol_e))

#if compare_psi4:
#    psi4.compare_values(psi4.energy('FCI'), fci_mol_e, 6, 'FCI Energy')
