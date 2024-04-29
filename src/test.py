from vibronic_helper import Vibronic


mol_str = """
H
H 1 0.74
symmetry c1
"""

options = {
"qed_type" : "qed-ci",
"molecule_template" : 
"""
H
H 1 **R**
symmetry c1
""",
"guess_bondlength" : 0.74,
}
X = Vibronic(options)

X.compute_qed_energy()
