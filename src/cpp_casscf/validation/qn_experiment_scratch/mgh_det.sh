#!/usr/bin/env bash
# MgH+ CAS(8,12) QN variant comparison, DETERMINISTIC re-run.
#
# OMP_NUM_THREADS=1 is mandatory, not a performance choice: ci_solver.c's
# threaded reductions are not bitwise-reproducible, and at >1 thread repeated
# runs of an identical seeded config diverge (measured on h2o/6-31G: OMP=1
# reproduces exactly; OMP=3 gave 8 vs 11 macroiterations on back-to-back runs
# and converged to a different solution 7.5e-3 away). Every earlier MgH+ run
# in this experiment used OMP=3 and is therefore NOT comparable.
#
# Single-threaded runs are slower individually, so parallelism moves to the
# process level (8 concurrent, 32 cores available).
set -u
cd /home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src
PY=~/miniforge3/envs/p4dev/bin/python
OUT="$1"; mkdir -p "$OUT"

run() {  # seed variant
    local sd="$1" v="$2" E=""
    case "$v" in
        v6) E="QED_RESET_ON_BAD_STEP=1" ;;
        v7) E="QED_QN_REJECT=1" ;;
        v1) E="QED_DISABLE_QN=1" ;;
    esac
    env OMP_NUM_THREADS=1 QED_RANDOM_ORBITAL_SEED="$sd" $E \
        $PY cpp_casscf/validation/run_mgh_case.py --seed "$sd" \
        > "$OUT/s${sd}_${v}.log" 2>&1
}

N=0
for sd in 1 2 3 4 5 6; do
    for v in v0 v6 v7; do
        run "$sd" "$v" &
        N=$((N+1))
        if [ $((N % 8)) -eq 0 ]; then wait; fi
    done
done
wait
echo MGH_DET_DONE
