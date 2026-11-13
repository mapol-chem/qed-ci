"""Plot cached benchmark metrics and write summary tables/captions."""
import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import FIGURES, LAMBDA_MAGNITUDES, RESULTS


# Three ordered samples from viridis, a color-vision-accessible sequential map.
COLORS = {0.01: "#440154", 0.05: "#21918c", 0.1: "#90d743"}
MARKERS = {"ground": "o", "LP": "s", "UP": "^"}
STATES = ("ground", "LP", "UP")
NATURAL_BASELINE_FLOOR = 1.0e-8


def read_metrics(prefix):
    with (RESULTS / f"{prefix}_metrics.csv").open() as f:
        return list(csv.DictReader(f))


def finite_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def select_rows(rows, key, ratio_key, natural=False):
    selected = []
    for row in rows:
        if row.get("state") not in {"ground", "bright_dressed"}:
            continue
        if row.get("status") != "converged":
            continue
        if str(row.get("assignment_ambiguous", "False")).lower() == "true":
            continue
        lam = finite_float(row["lambda_value"])
        x = finite_float(row[key])
        y = finite_float(row[ratio_key])
        if lam is None or not any(np.isclose(lam, l, rtol=0, atol=1e-8) for l in LAMBDA_MAGNITUDES) or x is None or y is None:
            continue
        if natural and x < NATURAL_BASELINE_FLOOR:
            continue
        if x <= 0 or y <= 0:
            continue
        row = dict(row)
        row["lambda_num"] = lam
        row["x_num"] = x
        row["ratio_num"] = y
        row["y_num"] = x * y
        selected.append(row)
    return selected


def summary(rows, metric_name, ratio_key, natural=False):
    output = []
    for lam in LAMBDA_MAGNITUDES:
        vals = np.array([r["ratio_num"] for r in rows if r["state"] in {"ground", "bright_dressed"} and np.isclose(r["lambda_num"], lam, rtol=0, atol=1e-8)], dtype=float)
        output.append(dict(metric=metric_name, lambda_value=lam, n=int(len(vals)),
                           median_ratio=float(np.median(vals)) if len(vals) else "",
                           min_ratio=float(np.min(vals)) if len(vals) else "",
                           max_ratio=float(np.max(vals)) if len(vals) else "",
                           natural_baseline_floor=NATURAL_BASELINE_FLOOR if natural else ""))
    return output


def plot_metric(rows, x_key, ratio_key, y_label, stem, natural=False):
    # Share the default selection with the combined figures: no LP/UP ratios.
    from plot_combined import figure
    return figure(rows, x_key, ratio_key, y_label, stem, natural)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("pilot", "full"), default="pilot")
    prefix = parser.parse_args().prefix
    if prefix == "full":
        from finalize_v2 import main as finalize
        return finalize()
    rows = read_metrics(prefix)
    with (RESULTS / f"{prefix}_photon_convergence.csv").open() as f:
        convergence = list(csv.DictReader(f))
    failed_groups = [r for r in convergence
                     if r["photon_states"] == "10" and r["converged_by_max_basis"] == "False"]
    prefix = prefix + "_v2"
    metrics = [
        ("1-RDM Frobenius", "cavity_free_frobenius", "ratio",
         "Frobenius norm", "rdm_invariance" , False),
        ("Natural occupations", "cavity_free_natural_occupation_error", "natural_occupation_ratio",
         "max occupation error", "natural_occupation_invariance", True),
    ]
    summary_rows = []
    caption_lines = []
    for name, xkey, rkey, ylabel, stem, is_natural in metrics:
        data = plot_metric(rows, xkey, rkey, ylabel, f"{prefix}_{stem}", natural=is_natural)
        stats = summary(data, name, rkey, natural=is_natural)
        summary_rows.extend(stats)
        parts = []
        for s in stats:
            if s["n"]:
                parts.append(f"|λ|={s['lambda_value']:.2f}: median {s['median_ratio']:.3g} (range {s['min_ratio']:.3g}–{s['max_ratio']:.3g}, n={s['n']})")
            else:
                parts.append(f"|λ|={s['lambda_value']:.2f}: no usable ratios")
        caption_lines.append((name, parts, is_natural))
    with (RESULTS / f"{prefix}_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (RESULTS / f"{prefix}_summary.md").open("w") as f:
        f.write(f"# {'Full-set' if prefix == 'full' else 'Pilot'} summary\n\n")
        f.write("| Metric | λ (a.u.) | n | Median ratio | Range |\n|---|---:|---:|---:|---:|\n")
        for r in summary_rows:
            span = f"{r['min_ratio']:.4g}–{r['max_ratio']:.4g}" if r["n"] else "—"
            med = f"{r['median_ratio']:.4g}" if r["n"] else "—"
            f.write(f"| {r['metric']} | {r['lambda_value']:.2f} | {r['n']} | {med} | {span} |\n")
        f.write(f"\nNatural-occupation ratios omit cases with a cavity-free error below {NATURAL_BASELINE_FLOOR:g}.\n")
    for (name, parts, is_natural) in caption_lines:
        note = (f" Natural-occupation ratios exclude reference errors below {NATURAL_BASELINE_FLOOR:g}." if is_natural else "")
        scope = ("The full configured molecule set and stretched geometries" if prefix == "full"
                 else "H₂ and LiH at equilibrium and stretched geometries")
        ambiguous = sum(str(r.get("assignment_ambiguous", "False")).lower() == "true" for r in rows)
        text = (f"{scope} compare the QED-CAS electronic 1-RDM error with its cavity-free counterpart; only converged resonant ground and blue-detuned bright-connected rows are plotted. LP/UP and blue-detuned grounds are excluded. Panel (a) shows parity and panel (b) the QED/cavity-free error ratio: "
                + "; ".join(parts) + f". {len(failed_groups)} geometry/coupling groups did not converge by 10 photon states; {ambiguous} rows have ambiguous or unavailable polariton assignments. The full run did not persist QED-CAS assignment scores; its saved LP/UP roots were chosen by the same projection criterion, so their assignment confidence cannot be independently audited from this cache." + note)
        suffix = "natural_occupations" if is_natural else "rdm"
        (RESULTS / f"{prefix}_draft_caption_{suffix}.txt").write_text(text + "\n")
    print(f"Figures: {FIGURES}")
    print(f"Summary and captions: {RESULTS}")


if __name__ == "__main__":
    main()
