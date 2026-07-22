#!/usr/bin/env bash
# MgH+ CAS(8,12) QN variant comparison -- PART 3.
#
# Two jobs, both into the same mgh_det_logs/ as parts 1 and 2:
#
#  (a) v6 for ALL six seeds, under the REWRITTEN QED_RESET_ON_BAD_STEP, which
#      now REPLACES the predicted-energy reset trigger instead of OR-ing onto
#      it. The old OR form was dead code -- 372 of 523 MgH+ QN steps raised the
#      actual energy and none had predicted_energy <= 0, so the default trigger
#      had already fired every time. Those three old v6 runs are archived in
#      superseded_v6_or_semantics/ and are NOT comparable to these.
#      Verified live on lih/h2o/beh2 x 3 seeds: the trigger now differs from v0
#      on 25-30% of reset checks and halves the number of exact-Hessian
#      rebuilds (230 -> 115 over 400 checks) at identical macroiteration counts.
#
#  (b) the v0/v7 cells that part 1 never reached -- its loop was killed to stop
#      it launching more old-semantics v6 runs, which also cancelled the queued
#      v0/v7 work. s1/s2/s3 v0, s1 v7 are already done; s2_v7 was left running
#      and is excluded here. Re-check before relaunching if that has changed:
#        ls mgh_det_logs/ | sort
#
# OMP_NUM_THREADS=1 is mandatory -- see mgh_det.sh's header and
# orbguess_findings.md. Concurrency 6; mgh_det2.sh may still be running.
set -u
cd /home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src
PY=~/miniforge3/envs/p4dev/bin/python
OUT="$1"; mkdir -p "$OUT"

run() {  # seed variant
    local sd="$1" v="$2" E=""
    case "$v" in
        v6) E="QED_RESET_ON_BAD_STEP=1" ;;
        v7) E="QED_QN_REJECT=1" ;;
        v0) E="" ;;
    esac
    env OMP_NUM_THREADS=1 QED_RANDOM_ORBITAL_SEED="$sd" $E \
        $PY cpp_casscf/validation/run_mgh_case.py --seed "$sd" \
        > "$OUT/s${sd}_${v}.log" 2>&1
}

CELLS="1:v6 2:v6 3:v6 4:v6 5:v6 6:v6 3:v7 4:v0 4:v7 5:v0 5:v7 6:v0 6:v7"

N=0
for cell in $CELLS; do
    run "${cell%%:*}" "${cell##*:}" &
    N=$((N+1))
    if [ $((N % 6)) -eq 0 ]; then wait; fi
done
wait
echo MGH_DET3_DONE
