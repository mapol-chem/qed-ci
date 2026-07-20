#!/usr/bin/env python3
"""Tabulates per-macroiteration energies from a Python log and one or two
run_macroiteration_driver logs (QN-enabled and/or QN-disabled), side by
side, by macroiteration index.

Both sides print an identically-shaped line per macroiteration:
    Macroiteration <N> old CI energy <old> new CI energy <new>
(Python: helper_PFCI.py:2597-2600; C++: run_macroiteration_driver's
on_macroiteration_end hook, macroiteration_driver.hpp) -- this script just
regex-extracts "new CI energy" from each log and joins by <N>.

Usage:
    python compare_macroiterations.py --python python.log \
        --cpp-qn-on cpp_qn_on.log --cpp-qn-off cpp_qn_off.log

Any of --cpp-qn-on / --cpp-qn-off may be omitted.
"""
import argparse
import re
import sys

LINE_RE = re.compile(
    r"Macroiteration\s+(\d+)\s+old CI energy\s+([-\d.eE]+)\s+new CI energy\s+([-\d.eE]+)"
)


def parse_log(path):
    """Returns {macroiteration: new_energy}."""
    energies = {}
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                idx = int(m.group(1))
                energies[idx] = float(m.group(3))
    return energies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="Path to a log of a real helper_PFCI.py run's stdout")
    ap.add_argument("--cpp-qn-on", help="Path to a run_macroiteration_driver (QN enabled, the default) log")
    ap.add_argument("--cpp-qn-off", help="Path to a run_macroiteration_driver --disable-qn log")
    args = ap.parse_args()

    py = parse_log(args.python)
    cpp_on = parse_log(args.cpp_qn_on) if args.cpp_qn_on else {}
    cpp_off = parse_log(args.cpp_qn_off) if args.cpp_qn_off else {}

    if not py:
        print(f"No 'Macroiteration N ... new CI energy ...' lines found in {args.python}", file=sys.stderr)
        sys.exit(1)

    all_idx = sorted(set(py) | set(cpp_on) | set(cpp_off))

    headers = ["macro", "Python"]
    if cpp_on:
        headers += ["C++ QN=on", "|diff|"]
    if cpp_off:
        headers += ["C++ QN=off", "|diff|"]

    rows = []
    for i in all_idx:
        row = [str(i), f"{py[i]:.10f}" if i in py else "-"]
        if cpp_on:
            if i in cpp_on and i in py:
                row += [f"{cpp_on[i]:.10f}", f"{abs(cpp_on[i] - py[i]):.3e}"]
            elif i in cpp_on:
                row += [f"{cpp_on[i]:.10f}", "-"]
            else:
                row += ["-", "-"]
        if cpp_off:
            if i in cpp_off and i in py:
                row += [f"{cpp_off[i]:.10f}", f"{abs(cpp_off[i] - py[i]):.3e}"]
            elif i in cpp_off:
                row += [f"{cpp_off[i]:.10f}", "-"]
            else:
                row += ["-", "-"]
        rows.append(row)

    widths = [max(len(h), *(len(r[c]) for r in rows)) if rows else len(h) for c, h in enumerate(headers)]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))

    print()
    print(f"Python final ({max(py)} macroiterations): {py[max(py)]:.12f}")
    if cpp_on:
        print(f"C++ QN=on  final ({max(cpp_on)} macroiterations): {cpp_on[max(cpp_on)]:.12f}")
    if cpp_off:
        print(f"C++ QN=off final ({max(cpp_off)} macroiterations): {cpp_off[max(cpp_off)]:.12f}")


if __name__ == "__main__":
    main()
