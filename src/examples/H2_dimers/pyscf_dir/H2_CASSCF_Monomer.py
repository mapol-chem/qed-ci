import numpy as np
from pyscf import gto, scf, mp, mcscf

mol = gto.M(
    atom = '''
         H            0.000000000000     0.000000000000    -0.371301335400
         H            0.000000000000     0.000000000000     0.371301335400    
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
n_states = 2
weights = np.ones(n_states)/n_states
mc = mcscf.CASSCF(mf, 2, 2).state_average_(weights)
mc.fix_spin_(ss=0)
mc.conv_tol = 1e-12
mc.kernel()
