#!/usr/bin/env bash
# Runs a single CASSCF config through both Python (helper_PFCI.py, real QN
# default) and this C++ port (both QN=on and QN=off), then prints the
# per-macroiteration energy from all three side by side.
#
# Usage (run from this directory, cpp_casscf/validation/):
#   ./compare_macroiterations.sh <case-name> [dump_lih_case.py args...]
#
# <case-name> names the working directory (dumps_compare_<case-name>/,
# logs_compare_<case-name>/) -- pick anything descriptive, it's not read by
# dump_lih_case.py itself. Everything after it is forwarded verbatim to
# dump_lih_case.py -- see that script's own --help for the full list
# (--molecule lih|h2o, --basis, --bond-length, --nact-orbs, --nact-els,
# --davidson-roots/--davidson-maxdim/--davidson-indim, --omega,
# --random-seed).
#
# Examples:
#   ./compare_macroiterations.sh lih_631g_4_4 \
#       --molecule lih --basis 6-31g --bond-length 1.6 --nact-orbs 4 --nact-els 4
#   ./compare_macroiterations.sh h2o_ccpvdz_4_4 \
#       --molecule h2o --basis cc-pVDZ --nact-orbs 4 --nact-els 4
#
# Needs: the `p4dev` conda env active (has psi4, needed by dump_lih_case.py)
# and run_macroiteration_driver already built (cmake --build ../build -j).
#
# Prereqs are NOT checked/activated automatically -- if `python
# dump_lih_case.py` fails with `ModuleNotFoundError: No module named
# 'psi4'`, run `conda activate p4dev` first.
set -eu
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <case-name> [dump_lih_case.py args...]" >&2
    exit 1
fi

CASE_NAME="$1"
shift

BUILD_DIR=../build
BIN="$BUILD_DIR/run_macroiteration_driver"
if [ ! -x "$BIN" ]; then
    echo "run_macroiteration_driver not built -- run: cmake --build $BUILD_DIR -j" >&2
    exit 1
fi

DUMP_DIR="dumps_compare_${CASE_NAME}"
LOG_DIR="logs_compare_${CASE_NAME}"
mkdir -p "$LOG_DIR"
rm -rf "$DUMP_DIR"

echo "=== Running Python (helper_PFCI.py, real QN default) ==="
python dump_lih_case.py "$@" --dump-dir "$DUMP_DIR" > "$LOG_DIR/python.log" 2>&1
if ! grep -q "^Done\. CASSCF converged: True" "$LOG_DIR/python.log"; then
    echo "Python run did not report convergence -- see $LOG_DIR/python.log" >&2
    tail -20 "$LOG_DIR/python.log" >&2
    exit 1
fi
echo "  ok -- $LOG_DIR/python.log"

echo "=== Running C++ (QN enabled, the default) ==="
"$BIN" "$DUMP_DIR" > "$LOG_DIR/cpp_qn_on.log" 2>&1
echo "  ok -- $LOG_DIR/cpp_qn_on.log"

echo "=== Running C++ (QN disabled, --disable-qn) ==="
"$BIN" "$DUMP_DIR" --disable-qn > "$LOG_DIR/cpp_qn_off.log" 2>&1
echo "  ok -- $LOG_DIR/cpp_qn_off.log"

echo
python3 compare_macroiterations.py \
    --python "$LOG_DIR/python.log" \
    --cpp-qn-on "$LOG_DIR/cpp_qn_on.log" \
    --cpp-qn-off "$LOG_DIR/cpp_qn_off.log"
