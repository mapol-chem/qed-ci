#!/usr/bin/env bash
# Runs the CASSCF step of examples/h2o_grad.py's system through the C++ port,
# at print level "normal".
#
# WHAT THIS IS / IS NOT
# ---------------------
# examples/h2o_grad.py does SCF -> SA-QED-CASSCF -> nuclear gradient, all in
# Python. This script replaces ONLY the middle step with the C++ port:
#
#   Python (psi4)          C++ port                    (not done here)
#   SCF, integrals,   ->   MacroiterationDriver::run -> nuclear gradient
#   CI setup, dumps        (all 4 real collaborators)
#
# The C++ port has no psi4 and no integral generation of its own, so Python
# must run first to produce the dump; the port then replays the CASSCF
# macroiteration loop from it. The nuclear gradient is NOT ported at all --
# nothing in nuclear_grad.py has a C++ counterpart.
#
# SYSTEM -- matches examples/h2o_grad.py exactly:
#   H2O, 6-31G, O-H = 1.0 Ang, H-O-H = 104.5 deg   (verified from that file's
#     explicit Cartesians: O-H = 1.000000 Ang, angle = 104.5000 deg)
#   CAS(6,5), 4 roots, singlet
#   omega = 0.1, lambda = (0, 0, 0.05), 1 photon, coherent-state basis
#
# One deviation worth knowing: h2o_grad.py sets davidson_maxdim 8 / indim 6;
# dump_lih_case.py's defaults are used unless overridden below, and they are
# passed explicitly here so the two agree.
set -eu
cd "$(dirname "$0")"

DUMP_DIR=${DUMP_DIR:-dumps_h2o_grad_casscf}
LEVEL=${LEVEL:-normal}

if [ ! -d "$DUMP_DIR/macroiteration_bootstrap_000" ]; then
    echo "=== 1/2  Python: SCF + CASSCF with dump hooks (needs the p4dev env) ==="
    python dump_lih_case.py \
        --molecule h2o \
        --basis 6-31g \
        --h2o-oh-bond-length 1.0 \
        --h2o-hoh-angle 104.5 \
        --nact-orbs 5 \
        --nact-els 6 \
        --davidson-roots 4 \
        --davidson-maxdim 8 \
        --davidson-indim 6 \
        --omega 0.1 \
        --lambda-vector 0.0 0.0 0.05 \
        --n-photons 1 \
        --random-seed 0 \
        --dump-dir "$DUMP_DIR"
else
    echo "=== 1/2  reusing existing $DUMP_DIR (delete it to regenerate) ==="
fi

echo
echo "=== 2/2  C++: MacroiterationDriver::run at --print-level=$LEVEL ==="
../build/run_macroiteration_driver "$DUMP_DIR" --print-level="$LEVEL"
