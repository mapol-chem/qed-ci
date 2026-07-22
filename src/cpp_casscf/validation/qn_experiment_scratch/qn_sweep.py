#!/usr/bin/env python3
"""QN/BFGS convergence experiment sweep.

Runs run_qn_case.py over (system x seed x variant), collects macroiteration
counts / final energy / convergence flag, and tabulates.

Variants (env toggles inside helper_PFCI.py, all default no-op):
  v0_base    : production QN as-is
  v1_qnoff   : QED_DISABLE_QN=1                 (uncoupled only)
  v2_actual  : QED_ENERGY_TRIGGER_ACTUAL=1      (actual dE, not predicted)
  v3_hreset  : QED_HESSIAN_RESET_NEG_CURV=3     (exact-H reset on 3 damp events)
  v4_both    : v2 + v3
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

SRC = "/home/nvu12/software/qed_ci_main/qed_ci_casscf14/qed-ci/src"
PY = os.path.expanduser("~/miniforge3/envs/p4dev/bin/python")
RUNNER = os.path.join(SRC, "cpp_casscf/validation/run_qn_case.py")

VARIANTS = {
    "v0_base":   {},
    "v1_qnoff":  {"QED_DISABLE_QN": "1"},
    "v2_actual": {"QED_ENERGY_TRIGGER_ACTUAL": "1"},
    "v3_hreset": {"QED_HESSIAN_RESET_NEG_CURV": "3"},
    "v4_both":   {"QED_ENERGY_TRIGGER_ACTUAL": "1", "QED_HESSIAN_RESET_NEG_CURV": "3"},
    "v6_badstep":{"QED_RESET_ON_BAD_STEP": "1"},
    "v7_reject": {"QED_QN_REJECT": "1"},
}

# label -> run_qn_case.py args
SYSTEMS = {
    "lih_631g_4_4":   ["--mol", "lih",  "--basis", "6-31g", "--r", "1.6",
                       "--nact-orbs", "4", "--nact-els", "4", "--roots", "1",
                       "--maxdim", "8", "--indim", "6"],
    "hf_631g_4_4":    ["--mol", "hf",   "--basis", "6-31g", "--r", "0.95",
                       "--nact-orbs", "4", "--nact-els", "4", "--roots", "1",
                       "--maxdim", "8", "--indim", "6"],
    "beh2_631g_4_4":  ["--mol", "beh2", "--basis", "6-31g", "--r", "1.33",
                       "--nact-orbs", "4", "--nact-els", "4", "--roots", "1",
                       "--maxdim", "8", "--indim", "6"],
    "h2o_631g_4_4":   ["--mol", "h2o",  "--basis", "6-31g", "--r", "0.96",
                       "--nact-orbs", "4", "--nact-els", "4", "--roots", "1",
                       "--maxdim", "8", "--indim", "6"],
    "lih_631g_6_6":   ["--mol", "lih",  "--basis", "6-31g", "--r", "1.6",
                       "--nact-orbs", "6", "--nact-els", "4", "--roots", "1",
                       "--maxdim", "8", "--indim", "6"],
    "lih_631g_2root": ["--mol", "lih",  "--basis", "6-31g", "--r", "1.6",
                       "--nact-orbs", "4", "--nact-els", "4", "--roots", "2",
                       "--maxdim", "3", "--indim", "2"],
}

RE_MACRO = re.compile(r"^Macroiteration", re.M)
RE_ENERGY = re.compile(r"new CI energy\s+(-?[0-9.]+)")
RE_CONV = re.compile(r"CASSCF converged:\s*(\w+)")
RE_WALL = re.compile(r"wall=([0-9.]+)s")


def run_one(system, seed, variant, outdir, timeout):
    env = dict(os.environ)
    env.update(VARIANTS[variant])
    env["OMP_NUM_THREADS"] = "1"
    if seed is not None:
        env["QED_RANDOM_ORBITAL_SEED"] = str(seed)
    # --seed seeds np.random, which GLTR's noise draw consumes.  Without it
    # an identical config varies by +-3 macroiterations run to run (measured),
    # swamping any variant effect.  Passing it makes each cell deterministic
    # and makes the variant comparison paired on the same noise stream.
    cmd = [PY, RUNNER] + SYSTEMS[system]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    log = os.path.join(outdir, f"{system}__seed{seed}__{variant}.log")
    try:
        r = subprocess.run(cmd, cwd=SRC, env=env, capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout + r.stderr
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "") + "\n*** TIMEOUT ***\n"
        rc = -99
    with open(log, "w") as f:
        f.write(out)
    energies = RE_ENERGY.findall(out)
    conv = RE_CONV.search(out)
    wall = RE_WALL.search(out)
    return {
        "system": system, "seed": seed, "variant": variant, "rc": rc,
        "macro": len(RE_MACRO.findall(out)),
        "energy": float(energies[-1]) if energies else None,
        "converged": conv.group(1) if conv else None,
        "wall": float(wall.group(1)) if wall else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tasks = [(s, seed, v) for s in args.systems for seed in args.seeds
             for v in args.variants]
    print(f"{len(tasks)} runs, {args.jobs} parallel", flush=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, s, seed, v, args.outdir, args.timeout): (s, seed, v)
                for s, seed, v in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(f"[{i}/{len(tasks)}] {res['system']} seed={res['seed']} "
                  f"{res['variant']}: macro={res['macro']} conv={res['converged']} "
                  f"E={res['energy']} rc={res['rc']}", flush=True)

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.outdir}/results.json")


if __name__ == "__main__":
    sys.exit(main())
