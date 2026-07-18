#!/usr/bin/env bash
# Sweeps a handful of geometries/active-spaces/molecules through
# dump_lih_case.py, then replays each dump directory through
# validate_against_python. Prints a per-config summary at the end.
#
# Run from this directory: ./sweep.sh
set -u
cd "$(dirname "$0")"

BUILD_DIR=../build
BIN="$BUILD_DIR/validate_against_python"

if [ ! -x "$BIN" ]; then
    echo "validate_against_python not built -- run cmake --build $BUILD_DIR -j first"
    exit 1
fi

# name : extra dump_lih_case.py args
configs=(
    "lih_sto3g_compressed:--molecule lih --basis sto-3g --bond-length 1.2 --nact-orbs 2 --nact-els 2"
    "lih_sto3g_equilibrium:--molecule lih --basis sto-3g --bond-length 1.6 --nact-orbs 2 --nact-els 2"
    "lih_sto3g_stretched:--molecule lih --basis sto-3g --bond-length 2.2 --nact-orbs 2 --nact-els 2"
    "lih_631g_2_2:--molecule lih --basis 6-31g --bond-length 1.6 --nact-orbs 2 --nact-els 2"
    "lih_631g_4_4:--molecule lih --basis 6-31g --bond-length 1.6 --nact-orbs 4 --nact-els 4"
    "lih_sto3g_2roots:--molecule lih --basis sto-3g --bond-length 1.6 --nact-orbs 2 --nact-els 2 --davidson-roots 2 --davidson-maxdim 3 --davidson-indim 2"
    "h2o_631g_2_2:--molecule h2o --basis 6-31g --nact-orbs 2 --nact-els 2"
    "h2o_631g_2roots:--molecule h2o --basis 6-31g --nact-orbs 2 --nact-els 2 --davidson-roots 2 --davidson-maxdim 3 --davidson-indim 2"
)

summary=()

for entry in "${configs[@]}"; do
    name="${entry%%:*}"
    cargs="${entry#*:}"
    dump_dir="dumps_sweep_${name}"
    log="sweep_${name}.log"

    echo "=== $name ($cargs) ==="
    rm -rf "$dump_dir"
    python dump_lih_case.py $cargs --dump-dir "$dump_dir" > "$log" 2>&1
    py_status=$?

    if [ $py_status -ne 0 ]; then
        echo "  Python run FAILED (exit $py_status) -- see $log"
        summary+=("$name: PYTHON FAILED")
        continue
    fi

    converged=$(grep -o "CASSCF converged: [A-Za-z]*" "$log" | tail -1)
    n_cases=$(find "$dump_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  $converged, $n_cases dump cases"

    if [ "$n_cases" -eq 0 ]; then
        summary+=("$name: $converged, 0 cases (nothing to validate)")
        continue
    fi

    cpp_out=$("$BIN" "$dump_dir" 2>&1)
    result_line=$(echo "$cpp_out" | tail -1)
    fail_count=$(echo "$cpp_out" | grep -c "^FAIL")
    echo "  $result_line"
    if [ "$fail_count" -gt 0 ]; then
        echo "$cpp_out" | grep "^FAIL"
    fi
    summary+=("$name: $converged, $result_line")
done

echo
echo "=== Summary ==="
for line in "${summary[@]}"; do
    echo "$line"
done
