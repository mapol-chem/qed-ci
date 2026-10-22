import numpy as np
from pyscf import gto, scf, mp, mcscf

mol = gto.M(
    atom = '''
    H 0.0000000000   0.0000000000  11999.62869866
    H 0.0000000000   0.0000000000  12000.37130134
    H 0.0000000000   0.0000000000 -12000.37130134
    H 0.0000000000   0.0000000000 -11999.62869866
    ''',
    basis = 'cc-pvtz',
    spin = 0,
    charge =  0,
    verbose = 4)
# Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
#mymp = mp.UMP2(myhf).run()
mf = scf.RHF(mol)
ehf = mf.kernel()
#noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
#mycas = mcscf.CASCI(myhf, ncas, nelecas)
n_states = 6
weights = np.array([1, 1, 1, 0, 0, 1]) / 4
#np.ones(n_states)/4
mc = mcscf.CASSCF(mf, 8, 4).state_average_(weights)
mc.fix_spin_(ss=0)
mc.conv_tol = 1e-12
mc.kernel()
