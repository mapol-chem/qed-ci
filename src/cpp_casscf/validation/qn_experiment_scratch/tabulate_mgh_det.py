#!/usr/bin/env python3
"""Tabulate the deterministic MgH+ CAS(8,12) QN-variant sweep (mgh_det.sh).

mgh_det.sh writes one log per (seed, variant) as s<seed>_<variant>.log.  This
reads them with the same regexes qn_sweep.py uses and prints a seed x variant
macroiteration table plus the final energies, so an energy-raising QN step that
costs macroiterations (or lands on a different solution) is visible per seed
rather than only in an average.

Every cell of a row shares one seed, i.e. one identical starting orbital set and
one identical GLTR noise stream, so differences down a row are the variant's
doing -- this is the paired comparison the OMP=1 requirement exists to protect
(see mgh_det.sh's header for why OMP>1 invalidates it).

    python tabulate_mgh_det.py [logdir]
"""
import os
import re
import sys
from collections import defaultdict

RE_MACRO = re.compile(r"^Macroiteration", re.M)
RE_ENERGY = re.compile(r"new CI energy\s+(-?[0-9.]+)")
RE_CONV = re.compile(r"CASSCF converged:\s*(\w+)")
RE_LOG = re.compile(r"^s(\d+)_(v\d+)\.log$")
# run_mgh_case.py prints this only after the driver returns, so its ABSENCE is
# how a still-running run is told apart from a finished-but-unconverged one.
# Without this distinction a partially-written log reads as a fast-converging
# variant, which is exactly backwards -- a run 2 macroiterations in looks like
# the best cell in the table.
RE_DONE = re.compile(r"^Done\.", re.M)

logdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mgh_det_logs")

cells = {}
seeds, variants = set(), set()
for name in sorted(os.listdir(logdir)):
    m = RE_LOG.match(name)
    if not m:
        continue
    seed, variant = int(m.group(1)), m.group(2)
    with open(os.path.join(logdir, name), errors="replace") as f:
        out = f.read()
    energies = RE_ENERGY.findall(out)
    conv = RE_CONV.search(out)
    cells[(seed, variant)] = {
        "macro": len(RE_MACRO.findall(out)),
        "energy": float(energies[-1]) if energies else None,
        "converged": conv.group(1) if conv else None,
        "done": bool(RE_DONE.search(out)),
    }
    seeds.add(seed)
    variants.add(variant)

seeds = sorted(seeds)
variants = sorted(variants)
if not cells:
    sys.exit(f"no s<seed>_<variant>.log files in {logdir}")

print(f"MgH+ CAS(8,12)/cc-pVDZ, OMP_NUM_THREADS=1, {len(cells)} runs\n")
print("macroiterations   '...' = STILL RUNNING (the count is only progress so far,")
print("                   NOT a result)    '!' = finished without converging")
print("seed  " + "".join(f"{v:>10s}" for v in variants))
print("----  " + "".join("  --------" for _ in variants))
for seed in seeds:
    row = f"{seed:<4d}  "
    for v in variants:
        c = cells.get((seed, v))
        if c is None:
            row += f"{'--':>10s}"
        else:
            if not c["done"]:
                flag = "..."
            elif c["converged"] != "True":
                flag = "!"
            else:
                flag = ""
            row += f"{str(c['macro']) + flag:>10s}"
    print(row)

running = sum(1 for c in cells.values() if not c["done"])
if running:
    print(f"\n*** {running} run(s) STILL IN PROGRESS -- excluded from the summaries below. ***")

print("\nmean macroiterations over seeds FINISHED for every variant")
common = [s for s in seeds
          if all((s, v) in cells and cells[(s, v)]["done"] for v in variants)]
if not common:
    print("  (no seed has finished every variant yet)")
for v in variants:
    mac = [cells[(s, v)]["macro"] for s in common]
    if mac:
        print(f"  {v:8s} {sum(mac) / len(mac):6.2f}   (n={len(mac)})")

print("\nfinal energies (Eh), FINISHED seeds only; '*' = differs from this seed's")
print("v0 by >1e-6, i.e. a different solution rather than a different path to it")
for seed in common:
    ref = cells.get((seed, "v0"), {}).get("energy")
    parts = []
    for v in variants:
        e = cells.get((seed, v), {}).get("energy")
        if e is None:
            parts.append(f"{v}=none")
        else:
            mark = "*" if ref is not None and abs(e - ref) > 1e-6 else " "
            parts.append(f"{v}={e:.9f}{mark}")
    print(f"  seed {seed}: " + "  ".join(parts))
