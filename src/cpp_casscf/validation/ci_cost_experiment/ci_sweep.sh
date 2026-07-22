#!/usr/bin/env bash
# CI-cost sweep: MgH+ CAS(8,12)/cc-pVDZ, seeds 1-6 x 4 variants, paired.
#
# WHY THESE TWO KNOBS -- from profiling the real v0 MgH+ runs
# (qn_experiment_scratch/mgh_det_logs, see ../README.md in this directory):
#   CI step is 71.1% of total runtime. Within all Davidson time,
#     coupled QN solves    833 calls  13469 s  79.0%   16.2 s/call, 25.9 iters
#     uncoupled (Kreplin)  521 calls   1344 s   7.9%    2.6 s/call
#     macro hitting cap      9 calls   2247 s  13.2%  249.7 s/call
#   The ENTIRE orbital side is 3.03 s = 0.053%.
# So the only lever with enough mass is the coupled CI solve count.
#
#   QED_CI_COUPLED_FACTOR    the 0.1 in threshold = 0.1*||g_orb||  (default 0.1)
#   QED_CI_UNCOUPLED_MAXITER the Kreplin uncoupled cap             (default 5)
#
# OMP_NUM_THREADS=1 IS MANDATORY, not a performance choice: from a
# bitwise-identical pinned guess, threaded CI-solver noise amplifies to 4.5e-6
# by macroiteration 13 -- enough to flip a trust-radius accept/reject or the
# step_norm<0.05 QN trigger. See qn_experiment_scratch/orbguess_findings.md.
#
# Runs are PAIRED: same QED_RANDOM_ORBITAL_SEED and --seed per (seed, variant)
# cell, so every variant sees an identical starting guess and noise stream.
#
# NOTE: wall time is the PRIMARY metric here (unlike the QN experiment, where
# it was macroiteration count) -- but never read it alone. A variant that
# converges to a worse solution or takes more macroiterations can look fast.
# tabulate_ci_sweep.py reports all three together.
set -u
cd "$(dirname "$0")"
SRC=$(cd ../../.. && pwd)
OUT=logs
mkdir -p "$OUT"
MAXJOBS=${MAXJOBS:-12}   # ~4 GB peak per run; 12 x 4 = 48 GB of ~106 GB free

run_cell () {
    local sd=$1 var=$2 env=""
    case "$var" in
        base)  env="" ;;
        f0.3)  env="QED_CI_COUPLED_FACTOR=0.3" ;;
        f0.5)  env="QED_CI_COUPLED_FACTOR=0.5" ;;
        unc10) env="QED_CI_UNCOUPLED_MAXITER=10" ;;
        unc20) env="QED_CI_UNCOUPLED_MAXITER=20" ;;
        *) echo "unknown variant $var" >&2; return 1 ;;
    esac
    local log="$OUT/s${sd}_${var}.log"
    [ -s "$log" ] && grep -q "CASSCF converged" "$log" && { echo "skip $log (done)"; return 0; }
    env OMP_NUM_THREADS=1 QED_RANDOM_ORBITAL_SEED="$sd" $env \
        python "$SRC/cpp_casscf/validation/run_mgh_case.py" --seed "$sd" > "$log" 2>&1
    echo "done s${sd}_${var}"
}

for sd in 1 2 3 4 5 6; do
    for var in base f0.3 f0.5 unc10 unc20; do
        while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
        run_cell "$sd" "$var" &
    done
done
wait
echo "ALL CELLS COMPLETE"
