#!/usr/bin/env bash
# MgH+ CAS(8,12) QN variant comparison, DETERMINISTIC re-run -- PART 2.
#
# Companion to mgh_det.sh, which covers v0/v6/v7. This adds the three variants
# that the small-system sweep found indistinguishable (v2/v3/v4): those systems
# converged in ~8 macroiterations and never exercised the pathology, so "no
# effect there" is not evidence of "no effect here". MgH+ is the hard case, so
# every variant gets tested on it rather than only the ones that survived an
# easy screen.
#
#   v2 = QED_ENERGY_TRIGGER_ACTUAL=1     (actual dE, not predicted)
#   v3 = QED_HESSIAN_RESET_NEG_CURV=3    (exact-H reset after 3 damp events)
#   v4 = v2 + v3
#
# Writes into the SAME log directory as mgh_det.sh (s<seed>_<variant>.log, no
# filename collision) so tabulate_mgh_det.py produces one 6-variant table.
# v0 is not repeated here -- mgh_det.sh's v0 column is the shared baseline, and
# it is a valid baseline for these runs because OMP=1 makes a run's result
# independent of what else is on the machine.
#
# OMP_NUM_THREADS=1 is mandatory, not a performance choice -- see mgh_det.sh's
# header for the measurement, and orbguess_findings.md for the follow-up
# showing that pinning the initial orbitals does NOT remove the need for it
# (threaded noise from a bitwise-identical guess still amplified to 4.5e-6
# mid-trajectory).
#
# Concurrency is 6 rather than mgh_det.sh's 8 because that sweep may still be
# running; correctness does not depend on this, only wall time.
set -u
cd /home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src
PY=~/miniforge3/envs/p4dev/bin/python
OUT="$1"; mkdir -p "$OUT"

run() {  # seed variant
    local sd="$1" v="$2" E=""
    case "$v" in
        v2) E="QED_ENERGY_TRIGGER_ACTUAL=1" ;;
        v3) E="QED_HESSIAN_RESET_NEG_CURV=3" ;;
        v4) E="QED_ENERGY_TRIGGER_ACTUAL=1 QED_HESSIAN_RESET_NEG_CURV=3" ;;
    esac
    env OMP_NUM_THREADS=1 QED_RANDOM_ORBITAL_SEED="$sd" $E \
        $PY cpp_casscf/validation/run_mgh_case.py --seed "$sd" \
        > "$OUT/s${sd}_${v}.log" 2>&1
}

N=0
for sd in 1 2 3 4 5 6; do
    for v in v2 v3 v4; do
        run "$sd" "$v" &
        N=$((N+1))
        if [ $((N % 6)) -eq 0 ]; then wait; fi
    done
done
wait
echo MGH_DET2_DONE
