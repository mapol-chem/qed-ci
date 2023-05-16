import numpy as np
import sys
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_7fb63ad/qed-ci/src")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
np.set_printoptions(threshold=sys.maxsize)

mol_str = """
    0 1
    o
    H 1 1.0
    H 1 1.0 2 104.5
    symmetry c1
"""

options_dict = {'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

cavity_options = {
    'omega_value' :0.0,
    'lambda_vector' : np.array([0, 0, 0.0]),
    'ci_level' : 'cas',
    'ignore_coupling' : False,
    'number_of_photons' : 1,
    'natural_orbitals' : True,
    'davidson_roots' : 10,
    'davidson_maxiter' : 1000,
    #'davidson_guess' : 'random guess',
    'rdm_weights': np.array([1,1,1,1,1,1,1,1]),
    'davidson_maxdim':6,
    'nact_orbs' : 4,
    'nact_els' : 4 
}

psi4.set_options(options_dict)
psi4.core.set_output_file('h2o_1.0_0.out', False)

H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)
###fci_e,wavefunction =np.linalg.eigh(H2_PF.H_PF)
###for i in range(H2_PF.H_PF.shape[1]):
###    print(fci_e[i])
###singletcount =0
###for i in range(wavefunction.shape[1]):
###            #print("state",i, "energy =",theta[i])
###            print("        amplitude","      position", "         most important determinants","             number of photon")
###            index=np.argsort(np.abs(wavefunction[:,i]))
###            c0 = index[wavefunction.shape[0]-1]%(H2_PF.H_PF.shape[1]//2)
###            d0 = (index[wavefunction.shape[0]-1]-c0)//(H2_PF.H_PF.shape[1]//2)
###            a0,b0 = H2_PF.detmap[c0]
###            alphalist = Determinant.obtBits2ObtIndexList(a0)
###            betalist = Determinant.obtBits2ObtIndexList(b0)
###            singlet = 1
###            for j in range(min(H2_PF.H_PF.shape[1],10)):
###                c = index[wavefunction.shape[0]-j-1]%(H2_PF.H_PF.shape[1]//2)
###                d = (index[wavefunction.shape[0]-j-1]-c)//(H2_PF.H_PF.shape[1]//2)
###                a,b = H2_PF.detmap[c]
###                if a == b0 and b == a0 and np.abs(wavefunction[index[wavefunction.shape[0]-j-1]][i]-(-1)*wavefunction[index[wavefunction.shape[0]-1]][i]) < 1e-5:
###                    singlet = singlet * 0
###                else:
###                    singlet = singlet * 1
###                alphalist = Determinant.obtBits2ObtIndexList(a)
###                betalist = Determinant.obtBits2ObtIndexList(b)
###
###                print("%20.12lf"%(wavefunction[index[wavefunction.shape[0]-j-1]][i]),"%9.3d"%(index[wavefunction.shape[0]-j-1]),"      alpha",alphalist,"   beta",betalist,"%4.1d"%(d), "photon")
###            #print("state",i, "energy =",theta[i], singlet)
###            if singlet == 1:
###                print("state",i, "energy =",fci_e[i], '  singlet',singletcount//2,"%2.1d"%(d0), "photon")
###                singletcount +=1
###            else:
###                print("state",i, "energy =",fci_e[i], '  triplet',"%2.1d"%(d0), "photon")

# now compute cqed-rhf to get transformation vectors with cavity
##cqed_rhf_dict = cqed_rhf(lambda_vector, mol_str, options_dict)
##rhf_reference_energy = cqed_rhf_dict["RHF ENERGY"]
##cqed_reference_energy = cqed_rhf_dict["CQED-RHF ENERGY"]
##C = cqed_rhf_dict["CQED-RHF C"]



# collect rhf wfn object as dictionary
##wfn_dict = psi4.core.Wavefunction.to_file(wfn)

# update wfn_dict with orbitals from CQED-RHF
##wfn_dict["matrix"]["Ca"] = C
##wfn_dict["matrix"]["Cb"] = C
# update wfn object
##wfn = psi4.core.Wavefunction.from_file(wfn_dict)
psi4.set_options({'restricted_docc': [3],'active': [4],'num_roots':2})
fci_energy = psi4.energy('fci',ref_wfn=wfn)
#print(np.allclose(H2_PF.cis_e[0],fci_energy,1e-8,1e-8))

