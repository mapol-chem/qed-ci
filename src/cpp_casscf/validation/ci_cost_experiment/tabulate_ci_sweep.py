"""Tabulate the CI-cost sweep (ci_sweep.sh).

Reports wall time, macroiteration count AND final energy together, always.
Reading wall time alone is the trap this experiment is most exposed to: a
variant with a looser CI threshold can finish faster simply by converging to a
worse solution or by stopping at a looser point, which looks like a win.

Runs still in progress are excluded from every summary -- a partially written
log otherwise reads as a fast-converging variant, which is exactly backwards.

Usage:  python tabulate_ci_sweep.py [logdir]
"""
import glob
import os
import re
import statistics as st
import sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else "logs"
VARIANTS = ["base", "f0.3", "f0.5", "unc10", "unc20"]
SEEDS = [1, 2, 3, 4, 5, 6]

re_conv = re.compile(r"CASSCF converged:\s*(\w+)")
re_wall = re.compile(r"wall=([0-9.]+)s")
re_macro = re.compile(r"^Macroiteration", re.M)
re_energy = re.compile(r"avg energy final\s+\d+\s+(-?[0-9.]+)")
re_dav = re.compile(r"Complete Davidson in ([0-9.]+) seconds")
re_nci = re.compile(r"number of CI iteration (\d+)")
re_iter = re.compile(r"^ITERATION\s+(\d+)\s+subspace size", re.M)


def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    conv = re_conv.search(txt)
    if not conv:
        return {"running": True, "macro": len(re_macro.findall(txt))}
    wall = re_wall.search(txt)
    en = re_energy.findall(txt)
    dav = [float(x) for x in re_dav.findall(txt)]
    return {
        "running": False,
        "converged": conv.group(1) == "True",
        "wall": float(wall.group(1)) if wall else None,
        "macro": len(re_macro.findall(txt)),
        "energy": float(en[-1]) if en else None,
        "dav_total": sum(dav),
        "dav_calls": len(dav),
        "coupled": txt.count("number of CI iteration 10000"),
    }


def main():
    data = {(s, v): parse(os.path.join(LOGDIR, f"s{s}_{v}.log")) for s in SEEDS for v in VARIANTS}
    running = [k for k, d in data.items() if d and d.get("running")]
    missing = [k for k, d in data.items() if d is None]
    print(f"MgH+ CAS(8,12)/cc-pVDZ, OMP_NUM_THREADS=1, CI-cost sweep")
    print(f"cells: {len(data)}  finished: {sum(1 for d in data.values() if d and not d['running'])}"
          f"  running: {len(running)}  not started: {len(missing)}\n")
    if running:
        print(f"  IN PROGRESS (excluded from all summaries): "
              f"{', '.join(f's{s}_{v}' for s, v in sorted(running))}\n")

    for title, key, fmt in [("wall time (s)", "wall", "{:>9.0f}"),
                            ("macroiterations", "macro", "{:>9d}"),
                            ("Davidson total (s)", "dav_total", "{:>9.0f}")]:
        print(f"--- {title} ---")
        print("seed  " + "".join(f"{v:>10}" for v in VARIANTS))
        for s in SEEDS:
            row = f"{s:<6}"
            for v in VARIANTS:
                d = data[(s, v)]
                row += "{:>10}".format("..." if (not d or d["running"]) else
                                       (fmt.format(d[key]) if d[key] is not None else "?"))
            print(row)
        print()

    # Complete seeds only, so the comparison is paired.
    full = [s for s in SEEDS if all(data[(s, v)] and not data[(s, v)]["running"] for v in VARIANTS)]
    if not full:
        print("No seed has all variants finished yet -- no paired summary.")
        return
    print(f"=== paired summary over seeds with ALL variants finished: {full} ===")
    print(f"{'variant':<8}{'mean wall':>11}{'vs base':>10}{'mean macro':>12}{'mean Dav':>10}{'coupled':>9}{'worst dE vs best':>18}")
    base_wall = st.mean([data[(s, 'base')]["wall"] for s in full])
    for v in VARIANTS:
        w = st.mean([data[(s, v)]["wall"] for s in full])
        m = st.mean([data[(s, v)]["macro"] for s in full])
        dv = st.mean([data[(s, v)]["dav_total"] for s in full])
        cp = st.mean([data[(s, v)]["coupled"] for s in full])
        worst = 0.0
        for s in full:
            best = min(data[(s, x)]["energy"] for x in VARIANTS if data[(s, x)]["energy"] is not None)
            e = data[(s, v)]["energy"]
            if e is not None:
                worst = max(worst, e - best)
        print(f"{v:<8}{w:>11.0f}{100*(w-base_wall)/base_wall:>9.1f}%{m:>12.1f}{dv:>10.0f}{cp:>9.0f}{worst:>18.2e}")
    print("\n'worst dE vs best' is how much higher this variant's final energy is than the best")
    print("variant's on the same seed, worst case. Anything above ~1e-6 means the variant found a")
    print("DIFFERENT (worse) solution -- a speedup there is not a speedup, and the cell is not")
    print("comparable. Check it before believing any wall-time win.")
    nonconv = [(s, v) for s in full for v in VARIANTS if not data[(s, v)]["converged"]]
    if nonconv:
        print(f"\n*** NON-CONVERGED cells: {nonconv} ***")


if __name__ == "__main__":
    main()
