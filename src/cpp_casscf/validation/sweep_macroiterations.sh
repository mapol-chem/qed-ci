#!/usr/bin/env bash
# Sweeps the full MacroiterationDriver::run pipeline (all 4 real
# collaborators wired together, see cpp_casscf/README.md's "End-to-end
# integration" section) against real Python CASSCF runs across several
# geometries/active-spaces/molecules -- the macroiteration-level analogue
# of sweep.sh, which validates individual trust-region solver calls in
# isolation rather than the whole driver loop.
#
# Unlike sweep.sh, this script does NOT regenerate dumps itself. It reads
# from the dumps_macro_sweep_<name>/ fixtures already checked into this
# directory. Those were generated with Python's QN/BFGS activation trigger
# temporarily disabled (`if step_norm < 0.05:` -> `if False and
# step_norm < 0.05:` in helper_PFCI.py's microiteration_optimization6,
# reverted immediately after via `git checkout` -- confirmed clean via
# `git diff`) -- CasscfMicroiterationOptimizationStep deliberately does not
# implement the QN/BFGS path (a real, documented, separate gap; see
# README.md), and QN's activation is known to shift the exact
# macroiteration count and intermediate trajectory without being a bug
# (confirmed this session on two separate systems by disabling QN in
# Python directly and re-comparing). Comparing against Python's literal
# default (QN-enabled) behavior would fail here for a reason that has
# nothing to do with this driver's own correctness.
#
# To regenerate these fixtures (e.g. after changing the config list),
# reuse this exact pattern: apply the one-line QN-disable edit above,
# generate all dumps_macro_sweep_<name>/ directories with
# `python dump_lih_case.py <config args> --random-seed 0 --dump-dir
# dumps_macro_sweep_<name>`, then revert the edit via `git checkout --
# helper_PFCI.py` and confirm `git diff` is empty before committing.
# This script only ever reads macroiteration_bootstrap_000/ and
# macroiteration_convergence_000/ out of each dump directory -- unlike
# sweep.sh, it has no use for the hundreds of individual per-solver-call
# dump directories dump_lih_case.py also produces (dominated by the H2O
# configs, ~80MB each full vs ~1MB pruned), so prune everything else
# before committing, e.g.: `find dumps_macro_sweep_<name> -mindepth 1
# -maxdepth 1 -type d ! -name 'macroiteration_*' -exec rm -rf {} +`.
#
# Run from this directory: ./sweep_macroiterations.sh
set -u
cd "$(dirname "$0")"

BUILD_DIR=../build
BIN="$BUILD_DIR/run_macroiteration_driver"

if [ ! -x "$BIN" ]; then
    echo "run_macroiteration_driver not built -- run cmake --build $BUILD_DIR -j first"
    exit 1
fi

# Same config list as sweep.sh (kept in sync intentionally -- see that
# script for the individual-solver-level comparison on the same systems).
names=(
    lih_sto3g_compressed
    lih_sto3g_equilibrium
    lih_sto3g_stretched
    lih_631g_2_2
    lih_631g_4_4
    lih_sto3g_2roots
    h2o_631g_2_2
    h2o_631g_2roots
)

# Energy tolerance for the final avg_energy comparison, and how many
# macroiterations of slack to allow before treating a count mismatch as a
# failure. A count differing by more than 1, or an energy differing by
# more than the tolerance, is treated as a real discrepancy worth
# investigating -- not silently accepted. A count off by exactly 1 with
# energy still within tolerance is the expected signature of a
# near-convergence-boundary sensitivity (both trajectories converging
# monotonically toward the same fixed point, crossing the tight 1e-10
# per-macroiteration energy_convergence threshold one iteration apart due
# to ordinary floating-point-level trajectory differences -- confirmed by
# direct per-macroiteration tracing on lih_631g_4_4, see README.md).
ENERGY_TOL=1e-8
MAX_MACRO_SLACK=1

summary=()
fail_count=0

for name in "${names[@]}"; do
    dump_dir="dumps_macro_sweep_${name}"
    if [ ! -d "$dump_dir" ]; then
        echo "=== $name: SKIPPED (no $dump_dir -- see this script's header for how to regenerate) ==="
        summary+=("$name: SKIPPED (missing fixture)")
        continue
    fi

    out=$("$BIN" "$dump_dir" 2>&1)
    py_line=$(echo "$out" | grep "^Python:")
    result_line=$(echo "$out" | grep "^converged=")
    cpp_energy=$(echo "$out" | grep "^C++  avg_energy=" | sed -E 's/.*=//')
    py_energy=$(echo "$out" | grep "^Python avg_energy=" | sed -E 's/.*=//')
    diff_val=$(echo "$out" | grep "^|diff|=" | sed -E 's/.*=//')

    py_macro=$(echo "$py_line" | sed -E 's/.*after ([0-9]+) macroiterations.*/\1/')
    cpp_macro=$(echo "$result_line" | sed -E 's/.*macroiterations_run=([0-9]+).*/\1/')
    converged=$(echo "$result_line" | sed -E 's/converged=([01]).*/\1/')

    macro_diff=$(( cpp_macro > py_macro ? cpp_macro - py_macro : py_macro - cpp_macro ))
    energy_ok=$(python3 -c "print(1 if abs($cpp_energy - ($py_energy)) < $ENERGY_TOL else 0)")

    status="PASS"
    if [ "$converged" != "1" ]; then
        status="FAIL (did not converge)"
    elif [ "$energy_ok" != "1" ]; then
        status="FAIL (energy diff $diff_val exceeds tolerance $ENERGY_TOL)"
    elif [ "$macro_diff" -gt "$MAX_MACRO_SLACK" ]; then
        status="FAIL (macroiteration count off by $macro_diff: C++=$cpp_macro Python=$py_macro)"
    elif [ "$macro_diff" -eq 1 ]; then
        status="PASS (macroiteration count off by 1: C++=$cpp_macro Python=$py_macro -- see header comment)"
    fi

    echo "=== $name ==="
    echo "  $py_line"
    echo "  $result_line, |diff|=$diff_val"
    echo "  $status"

    if [[ "$status" == FAIL* ]]; then
        fail_count=$((fail_count + 1))
    fi
    summary+=("$name: $status")
done

echo
echo "=== Summary ==="
for line in "${summary[@]}"; do
    echo "$line"
done

if [ "$fail_count" -gt 0 ]; then
    echo
    echo "$fail_count config(s) FAILED."
    exit 1
fi
