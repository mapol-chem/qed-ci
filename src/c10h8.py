import numpy as np
import sys
sys.path.append("/home/nvu12/software/qed_ci_main/qed_ci_casscf4/qed-ci/src/")
#np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
from helper_PFCI import PFHamiltonianGenerator
from helper_PFCI import Determinant
from helper_cqed_rhf import cqed_rhf
np.set_printoptions(threshold=sys.maxsize)

mol_str = """
0 1
C            0.000004551444    -0.711108766560     0.000010757138
C           -0.000004765464     0.711109074728     0.000001509641
C            1.231111212344    -1.392932324772    -0.000008872321
C            1.231092818353     1.392949010021    -0.000005886632
C           -1.231092617348    -1.392949084219     0.000002684292
C           -1.231111293835     1.392932340511     0.000007560162
C            2.416252154882    -0.702767074233    -0.000017781801
C            2.416242565430     0.702799552187    -0.000000723902
C           -2.416242703804    -0.702799153654     0.000010510747
C           -2.416251802225     0.702766484279     0.000002457080
H            1.229116150588    -2.471858679894    -0.000015215757
H            1.229083140542     2.471874840369    -0.000007074480
H           -1.229084430358    -2.471875468868    -0.000008700875
H           -1.229118355038     2.471858460776     0.000007352885
H            3.350158261997    -1.238508894745    -0.000013806252
H            3.350141246250     1.238554060765     0.000000769827
H           -3.350140729721    -1.238555163085     0.000018452996
H           -3.350156710411     1.238510150658    -0.000008144872
symmetry c1
no_reorient
nocom
"""

options_dict = {'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

mol = psi4.geometry(mol_str)

cavity_options = {
    'omega_value' : 0.,
    'lambda_vector' : np.array([0.0, 0.0, 0.0]),
    'ci_level' : 'cas',
    'ignore_coupling' : False,
    'number_of_photons' : 0,
    'natural_orbitals' : False,
    'photon_number_basis' : False,
    'canonical_mos' : False,
    'coherent_state_basis' : True,
    'davidson_roots' : 3,
    'davidson_threshold' : 1e-5,
    'davidson_maxdim':40,
    'spin_adaptation': "singlet",
    #'casscf_weight':np.array([1,1,1]),
    'davidson_maxiter':100,
    'davidson_indim':8,
    'test_mode': False,
    'nact_orbs' : 10,
    'nact_els' : 10 
}

psi4.set_options(options_dict)
psi4.core.set_output_file('c10h8.out', False)

H2_PF = PFHamiltonianGenerator(mol_str, options_dict, cavity_options)


# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

