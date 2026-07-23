#!/usr/bin/env bash
#
# casscf_python_vs_cpp.sh
# =======================
# A minimal, readable demo of how to run SA-QED-CASSCF **two ways** and compare
# them -- the Python reference and the standalone C++ port.
#
#   (A) Python reference : helper_PFCI.py's PFHamiltonianGenerator -- the real
#       production CASSCF. It loads the plain-C backend (cfunctions.so) through
#       ctypes and does the whole macroiteration loop in Python.
#
#   (B) C++ port         : tools/run_macroiteration_driver -- the standalone
#       C++/Eigen CASSCF (cpp_casscf/). It links the *same* C backend, compiled
#       from source, and does the macroiteration loop in C++.
#
# They are SEPARATE programs; they do NOT call each other. The bridge is a
# "bootstrap dump" on disk:
#
#     Python run  --writes-->  <dump dir>  --read by-->  C++ driver
#
# The Python run (A) writes its exact starting state (integrals, initial CI
# vector, weights, dims, ...) AND its own converged energy into the dump dir.
# The C++ driver (B) loads that dir, runs CASSCF from the identical starting
# point, and prints its energy next to Python's plus the difference.
#
# Run:      bash casscf_python_vs_cpp.sh
#
# Needs:    - the p4dev conda env active (for psi4):   conda activate p4dev
#           - the C++ driver built (one-time):
#               cd .. && cmake -S . -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
#                     && cmake --build build -j
# --------------------------------------------------------------------------

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ======================= things you can change ============================ #

# The chemistry. Defaults: LiH, minimal active space, 1 root -- runs in seconds.
# For a harder, state-averaged case try (uncomment):
#     MOLECULE=h2o ; NACT_ELS=4 ; NACT_ORBS=4 ; ROOTS=2
MOLECULE=lih          # lih | h2o
NACT_ELS=2            # active electrons
NACT_ORBS=2           # active orbitals
ROOTS=1               # number of CI roots to state-average over

# C++ driver verbosity: silent | normal | debug | trace
#   normal = per-macroiteration energy;  debug = + orbital-optimization steps;
#   trace  = + every CI Davidson iteration.  (Same levels the Python C backend
#   uses via the QED_PRINT_LEVEL env var.)
PRINT_LEVEL=normal

DUMP="$HERE/dump_casscf_demo"     # where the bootstrap dump is written

# ========================================================================== #

# Pin threads. The CI solver is multithreaded; with >1 thread the macroiteration
# COUNT varies run-to-run (tiny floating-point noise flips a trust-region
# accept/reject), though the final ENERGY is unaffected. 1 thread == reproducible.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DRIVER="$HERE/../build/run_macroiteration_driver"
if [[ ! -x "$DRIVER" ]]; then
  echo "ERROR: C++ driver not found/built at:"
  echo "    $DRIVER"
  echo "Build it once with:"
  echo "    cd $HERE/.. && cmake -S . -B build -DCMAKE_PREFIX_PATH=\$CONDA_PREFIX && cmake --build build -j"
  exit 1
fi

echo "############################################################"
echo "# STEP A -- Python reference CASSCF  (writes the dump)"
echo "############################################################"
echo "# command:"
echo "#   OMP_NUM_THREADS=1 python dump_lih_case.py \\"
echo "#       --molecule $MOLECULE --nact-els $NACT_ELS --nact-orbs $NACT_ORBS \\"
echo "#       --davidson-roots $ROOTS --dump-dir $DUMP"
echo "#"
echo "# (dump_lih_case.py just wraps PFHamiltonianGenerator with dump hooks on;"
echo "#  it is the same solver as your own PFHamiltonianGenerator(...) script.)"
echo
python "$HERE/dump_lih_case.py" \
    --molecule "$MOLECULE" --nact-els "$NACT_ELS" --nact-orbs "$NACT_ORBS" \
    --davidson-roots "$ROOTS" --dump-dir "$DUMP" > "$DUMP.python.log" 2>&1
py_status=$?
if [[ $py_status -ne 0 ]]; then
  echo "Python run FAILED (exit $py_status). Last lines of $DUMP.python.log:"
  tail -20 "$DUMP.python.log"
  exit 1
fi
echo "Python finished. Its converged result:"
grep -E "avg energy final|OPTIMIZATION CONVERGED" "$DUMP.python.log" | tail -2 | sed 's/^/    /'
echo "    (full Python log: $DUMP.python.log)"

echo
echo "############################################################"
echo "# STEP B -- C++ port CASSCF  (replays the same dump)"
echo "############################################################"
echo "# command:"
echo "#   ./run_macroiteration_driver $DUMP --print-level=$PRINT_LEVEL"
echo
"$DRIVER" "$DUMP" --print-level="$PRINT_LEVEL"

echo
echo "############################################################"
echo "# HOW TO READ THIS"
echo "############################################################"
echo "# The '--- Result ---' block above is the comparison:"
echo "#   C++  avg_energy   = the C++ port's converged CASSCF energy"
echo "#   Python avg_energy = the Python reference's converged energy (from the dump)"
echo "#   |diff|            = |C++ - Python|; expect ~1e-11 or smaller."
echo "#"
echo "# macroiterations_run may differ from Python's by a step or two -- that is"
echo "# an expected convergence-path effect (both reach the SAME minimum). Judge"
echo "# correctness by |diff| on the final energy, not by the iteration count."
