#!/bin/bash
# ============================================================================
# BIT-EXACT GATE FOR THE LSTRS EXTRACTION.
#
# The LSTRS beta-bisection lives inline in FOUR places (internal_optimization3,
# microiteration_optimization5 (dead), microiteration_optimization6,
# ah_orbital_optimization) and is being moved into lstrs_trust_region.py behind
# the same interface otr_trust_region.py already uses.  A move like that is only
# safe if it is PROVABLY a no-op, so this captures a reference set BEFORE and
# compares AFTER.
#
# WHY TRAJECTORY AND NOT JUST THE ENERGY.  Item 371: two mean fields differing
# at 1e-14 gave IDENTICAL converged energies at 12 dp and yet one walked into an
# unhandled solver branch and the other did not.  An energy check would have
# passed both.  The trajectory is the only thing that catches a refactor which
# perturbs the arithmetic.
#
# The cases are chosen to exercise both outcomes of the hard-case branch:
#   lz=0.05 om=0.12  -- FIRES the hard-case fallback (the item 371 case)
#   lz=0.02 om=0.12  -- fires it only with the patience cap on
#   lz=0.01 om=0.12  -- never fires it
#   lz=0.1  om=0.5   -- never fires it, different omega
# and each is run with the patience cap OFF as well, because with it off the
# whole change set must be bit-identical to the pre-refactor code.
#
# usage:  ./validation/lstrs_refactor_gate.sh capture <dir>   # before
#         ./validation/lstrs_refactor_gate.sh check   <dir>   # after
# ============================================================================
set -u
MODE=${1:?usage: $0 capture|check <dir>}
DIR=${2:?usage: $0 capture|check <dir>}
SRC=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DIR"

# Timing, RSS, and anything carrying a SOURCE LINE NUMBER are noise, not
# trajectory: moving a function shifts every line number after it, and a
# gate that fails on that is a gate that cannot pass.
# Stage 4 adds a presentation block after the unchanged trace. Exclude that
# bounded block from the trajectory comparison; retain standalone driver records.
strip () {
    awk '/^  ==> QED-CI calculation <==$/ {report=1}
         /^  ==> End QED-CI report <==$/ {report=0; next}
         !report {print}' "$1" | grep -vaE "seconds|took [0-9]|resident|wall |Percent of CPU|[0-9]\.[0-9]+ s(,|$| )|QED_LIBTRANS:|SyntaxWarning|helper_PFCI\.py:[0-9]|^[0-9]+\.[0-9]+$"
}

run () {  # $1 tag, $2 lambda_z, $3 omega, $4 patience, $5 driver (default h2e_disk_case)
    # PYTHONWARNINGS: helper_PFCI has two invalid-escape SyntaxWarnings, and
    # python echoes the OFFENDING SOURCE LINE under each -- but only when the
    # .pyc cache is cold, so the same code produced traces that differed by two
    # docstring lines depending on cache state.  Silence it rather than filter it.
    LAMBDA_Z=$2 OMEGA=$3 NPHOTON=1 OMP_NUM_THREADS=1 QED_SKIP_GKL=1 \
    PYTHONWARNINGS=ignore::SyntaxWarning \
    QED_LSTRS_HARD_PATIENCE=$4 \
        taskset -c ${GATE_CPU:-4} timeout 1800 \
        python "$SRC/validation/${5:-h2e_disk_case}.py" > "$DIR/$1.raw" 2>&1
    echo "  $1 rc=$? macros=$(grep -ac -i macroiteration "$DIR/$1.raw") \
energy=$(grep -aoE 'root 0 E= -?[0-9.]+|ROOT0 -?[0-9.]+' "$DIR/$1.raw" | head -1) \
fallback=$(grep -ac QED_TRS_HARD_FALLBACK "$DIR/$1.raw")"
    strip "$DIR/$1.raw" > "$DIR/$1.$OUTEXT"
}

# tag:lambda_z:omega:driver.  The `i*` cases use internal_opt_case.py, which is
# the ONLY driver here that reaches internal_optimization3 -- the others put
# every electron in the active space (n_in_a == 0) and never call it, so without
# them a change to the internal site passes this gate while being untested.
CASES="a:0.05:0.12:h2e_disk_case b:0.02:0.12:h2e_disk_case \
c:0.01:0.12:h2e_disk_case d:0.1:0.5:h2e_disk_case \
i1:0.05:0.12:internal_opt_case i2:0.01:0.5:internal_opt_case"

if [ "$MODE" = capture ]; then
    OUTEXT=ref
    echo "=== capturing reference into $DIR (helper $(md5sum $SRC/helper_PFCI.py | cut -c1-8)) ==="
    for c in $CASES; do IFS=: read -r t lz om drv <<< "$c"
        run "${t}_pat25" "$lz" "$om" 25 "$drv"
        run "${t}_pat0"  "$lz" "$om" 0 "$drv"
    done
    echo "reference captured"
elif [ "$MODE" = check ]; then
    OUTEXT=trace
    echo "=== checking against $DIR (helper $(md5sum $SRC/helper_PFCI.py | cut -c1-8)) ==="
    fail=0
    for c in $CASES; do IFS=: read -r t lz om drv <<< "$c"
        for p in pat25 pat0; do
            tag="${t}_${p}"
            [ -f "$DIR/$tag.ref" ] || { echo "  $tag: NO REFERENCE"; fail=1; continue; }
            run "$tag" "$lz" "$om" "$([ $p = pat25 ] && echo 25 || echo 0)" "$drv"
            if diff -q "$DIR/$tag.ref" "$DIR/$tag.trace" >/dev/null; then
                echo "    $tag: BIT-IDENTICAL"
            else
                echo "    $tag: *** DIFFERS ***"
                diff "$DIR/$tag.ref" "$DIR/$tag.trace" | head -6
                fail=1
            fi
        done
    done
    [ $fail = 0 ] && echo "GATE PASS: the extraction is a proven no-op" \
                  || { echo "GATE FAIL"; exit 1; }
else
    echo "unknown mode $MODE" >&2; exit 1
fi
