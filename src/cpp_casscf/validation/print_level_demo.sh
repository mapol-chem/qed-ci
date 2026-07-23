#!/usr/bin/env bash
#
# print_level_demo.sh -- see how the CI print levels look.
#
# Two independent print-level knobs exist:
#
#   1. QED_PRINT_LEVEL   (Python path, read by the C backend ci_solver.c)
#        0=Silent 1=Normal 2=Debug 3=Trace  -- default Trace.
#        Governs ONLY davidson_spin's output. Python-side prints
#        (macroiteration energy, the always-on root analysis, STATE lines)
#        are NOT gated by it.
#
#   2. --print-level=    (C++ port, tools/run_macroiteration_driver)
#        silent|normal|debug|trace -- governs the driver's CASSCF_LOG records.
#        The C++ root-analysis block is printed at EVERY level.
#
# Run from cpp_casscf/validation/ with the p4dev conda env active (psi4).
# Usage:  bash print_level_demo.sh
#
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${TMPDIR:-/tmp}/print_level_demo"
mkdir -p "$OUT"

# One thread => the tiny LiH run is reproducible across levels, so any output
# difference is the print level, not numerical noise.
export OMP_NUM_THREADS=1

echo "############################################################"
echo "# Part 1: Python path -- QED_PRINT_LEVEL (C backend CI output)"
echo "############################################################"
echo "# Same LiH sto-3g CAS(2,2) run at each level. Counting the"
echo "# C-backend (davidson_spin) markers each level emits:"
echo

names=(Silent Normal Debug Trace)
printf "%-8s %-7s | %-9s %-9s %-9s %-11s %-9s | %-8s %s\n" \
  LEVEL VALUE "ITERATION" "ROOTtable" "buildsigma" "spintable" "spinWARN" "rootanl" "final avg energy"
printf -- "---------------------------------------------------------------------------------------------------------\n"
for lvl in 0 1 2 3; do
  log="$OUT/qed_level_${lvl}.log"
  QED_PRINT_LEVEL=$lvl python "$HERE/dump_lih_case.py" --dump-dir "$OUT/dump_$lvl" >"$log" 2>&1
  it=$(grep -cE 'ITERATION[[:space:]]+[0-9]+ subspace' "$log")
  rt=$(grep -cE 'ROOT[[:space:]]+RESIDUAL NORM' "$log")
  bs=$(grep -cE 'build sigma took' "$log")
  st=$(grep -cE 'WFN TOTAL SPIN <S\^2>' "$log")
  sw=$(grep -cE 'SPIN CONTAMINATION' "$log")
  ra=$(grep -qE 'most important determinants' "$log" && echo yes || echo no)
  en=$(grep -E 'avg energy final' "$log" | tail -1 | awk '{print $NF}')
  printf "%-8s %-7s | %-9s %-9s %-9s %-11s %-9s | %-8s %s\n" \
    "${names[$lvl]}" "$lvl" "$it" "$rt" "$bs" "$st" "$sw" "$ra" "$en"
done

echo
echo "# Full logs (open to read the actual text at each level):"
for lvl in 0 1 2 3; do echo "#   ${names[$lvl]}: $OUT/qed_level_${lvl}.log"; done

echo
echo "# How ONE CI solve renders at each level: context around the first"
echo "# 'converged'. Watch the spin checkpoint appear at Debug, and the"
echo "# per-iteration ITERATION/ROOT tables appear at Trace:"
for lvl in 0 1 2 3; do
  log="$OUT/qed_level_${lvl}.log"
  echo
  echo "=== QED_PRINT_LEVEL=$lvl (${names[$lvl]}) ==="
  # context around the first converged, cut before the (always-on) root
  # analysis so this stays focused on the CI-solve rendering.
  grep -m1 -B6 -A8 '^converged$' "$log" | sed '/ACTIVE PART OF/,$d' | sed 's/^/    /'
done

echo
echo "# Root analysis is ALWAYS ON (Python-side, not gated by QED_PRINT_LEVEL)."
echo "# Identical at every level -- the final block from the Normal run:"
lastln=$(grep -n "ACTIVE PART OF" "$OUT/qed_level_1.log" | tail -1 | cut -d: -f1)
if [[ -n "${lastln:-}" ]]; then
  sed -n "${lastln},$((lastln+14))p" "$OUT/qed_level_1.log" | sed 's/^/    /'
fi

echo
echo "############################################################"
echo "# Part 2: C++ port -- run_macroiteration_driver --print-level"
echo "############################################################"
DRV="$HERE/../build/run_macroiteration_driver"
DUMP="$HERE/dumps_macro_lih"
if [[ -x "$DRV" && -d "$DUMP" ]]; then
  printf "%-8s | %-9s %-9s %-9s | %s\n" LEVEL "[macro]" "[ci]" "[orb]" "rootanalysis"
  printf -- "-----------------------------------------------------------------\n"
  for lvl in silent normal debug trace; do
    log="$OUT/cpp_${lvl}.log"
    "$DRV" "$DUMP" --print-level=$lvl >"$log" 2>&1
    m=$(grep -cE '^\[macro\]' "$log")
    c=$(grep -cE '^\[ci\]'    "$log")
    o=$(grep -cE '^\[orb\]'   "$log")
    ra=$(grep -qE 'most important determinants' "$log" && echo yes || echo no)
    printf "%-8s | %-9s %-9s %-9s | %s\n" "$lvl" "$m" "$c" "$o" "$ra"
  done
  echo "# Full C++ logs: $OUT/cpp_<level>.log"
else
  echo "# skipped: build the driver first (cmake --build build -j) and generate"
  echo "#   the bootstrap dump: python dump_lih_case.py --dump-dir dumps_macro_lih"
fi
