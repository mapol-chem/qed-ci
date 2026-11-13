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
        if row.get("status") != "converged":
            continue
        if str(row.get("assignment_ambiguous", "False")).lower() == "true":
            continue
        lam = finite_float(row["lambda_value"])
        x = finite_float(row[key])
        y = finite_float(row[ratio_key])
        if lam not in LAMBDA_MAGNITUDES or x is None or y is None:
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
        vals = np.array([r["ratio_num"] for r in rows if r["lambda_num"] == lam], dtype=float)
        output.append(dict(metric=metric_name, lambda_value=lam, n=int(len(vals)),
                           median_ratio=float(np.median(vals)) if len(vals) else "",
                           min_ratio=float(np.min(vals)) if len(vals) else "",
                           max_ratio=float(np.max(vals)) if len(vals) else "",
                           natural_baseline_floor=NATURAL_BASELINE_FLOOR if natural else ""))
    return output


def plot_metric(rows, x_key, ratio_key, y_label, stem, natural=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = select_rows(rows, x_key, ratio_key, natural=natural)
    if not data:
        raise RuntimeError(f"No usable rows for {stem}")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.12, 3.00))
    # Parity panel. Filled symbols are equilibrium; hollow symbols are stretched.
    xvals = np.array([r["x_num"] for r in data])
    yvals = np.array([r["y_num"] for r in data])
    log_axes = (xvals.max() / xvals.min() > 100 or yvals.max() / yvals.min() > 100)
    for lam in LAMBDA_MAGNITUDES:
        for state in STATES:
            subset = [r for r in data if r["lambda_num"] == lam and r["state"] == state]
            for stretched in (False, True):
                group = [r for r in subset if r["geometry"] != "equilibrium"] if stretched else [r for r in subset if r["geometry"] == "equilibrium"]
                if not group:
                    continue
                ax0.scatter([r["x_num"] for r in group], [r["y_num"] for r in group],
                    marker=MARKERS[state], s=23, linewidths=0.9,
                    facecolors="none" if stretched else COLORS[lam],
                    edgecolors=COLORS[lam], alpha=0.9)
    if log_axes:
        ax0.set_xscale("log")
        ax0.set_yscale("log")
    lo = min(float(xvals.min()), float(yvals.min()))
    hi = max(float(xvals.max()), float(yvals.max()))
    if log_axes:
        lo = max(lo, np.finfo(float).tiny)
        ax0.plot([lo, hi], [lo, hi], color="0.25", lw=0.9, ls="--")
    else:
        pad = 0.04 * (hi - lo if hi > lo else hi)
        lo, hi = max(0, lo - pad), hi + pad
        ax0.plot([lo, hi], [lo, hi], color="0.25", lw=0.9, ls="--")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("Cavity-free CAS error")
    ax0.set_ylabel(f"QED-CAS error ({y_label})")
    ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, va="top", ha="left", fontweight="bold")
    ax0.grid(False)

    # Ratio distributions: light box plots with deterministic, small jitter.
    box_data = [[r["ratio_num"] for r in data if r["lambda_num"] == lam] for lam in LAMBDA_MAGNITUDES]
    bp = ax1.boxplot(box_data, positions=np.arange(1, 4), widths=0.46,
                     patch_artist=True, showfliers=False, medianprops={"color": "0.15", "lw": 1.0},
                     whiskerprops={"color": "0.45", "lw": 0.8}, capprops={"color": "0.45", "lw": 0.8})
    for box in bp["boxes"]:
        box.set_facecolor("#E6E9EF")
        box.set_edgecolor("#9AA1AA")
        box.set_linewidth(0.8)
    rng = np.random.default_rng(731)
    for idx, lam in enumerate(LAMBDA_MAGNITUDES, start=1):
        vals = np.array([r["ratio_num"] for r in data if r["lambda_num"] == lam])
        if len(vals):
            ax1.scatter(idx + rng.uniform(-0.16, 0.16, len(vals)), vals,
                        s=12, color=COLORS[lam], edgecolors="white", linewidths=0.35, alpha=0.85)
    ax1.axhline(1.0, color="0.25", lw=0.9, ls="--")
    if natural:
        ax1.set_yscale("log")
    ax1.set_xticks((1, 2, 3), ("0.01", "0.05", "0.10"))
    ax1.set_xlabel(r"$|\lambda|$ (a.u.)")
    ax1.set_ylabel("QED / cavity-free error")
    ax1.text(0.02, 0.98, "(b)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
    ax1.grid(axis="y", color="0.90", lw=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    state_handles = [Line2D([], [], marker=MARKERS[state], linestyle="", color="0.15",
                            markerfacecolor="none", label=state) for state in STATES]
    lambda_handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[lam],
                             label=fr"{lam:g}") for lam in LAMBDA_MAGNITUDES]
    stretched_handle = Line2D([], [], marker="o", linestyle="", color="0.2",
                              markerfacecolor="none", label="stretched")
    fig.legend(handles=state_handles + lambda_handles + [stretched_handle],
               labels=["ground", "LP", "UP", r"$|\lambda|=0.01$", r"$|\lambda|=0.05$", r"$|\lambda|=0.10$", "stretched"],
               fontsize=6.4, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.3, columnspacing=0.8,
               borderpad=0.1)
    for ax in (ax0, ax1):
        ax.tick_params(labelsize=7, width=0.6, length=2.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.37, wspace=0.38)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("pilot", "full"), default="pilot")
    prefix = parser.parse_args().prefix
    rows = read_metrics(prefix)
    with (RESULTS / f"{prefix}_photon_convergence.csv").open() as f:
        convergence = list(csv.DictReader(f))
    failed_groups = [r for r in convergence
                     if r["photon_states"] == "10" and r["converged_by_max_basis"] == "False"]
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
        text = (f"{scope} compare the QED-CAS electronic 1-RDM error with its cavity-free counterpart; only photon-basis-converged rows without an ambiguous QED-FCI LP/UP assignment are plotted. Panel (a) shows parity and panel (b) the QED/cavity-free error ratio: "
                + "; ".join(parts) + f". {len(failed_groups)} geometry/coupling groups did not converge by 10 photon states; {ambiguous} rows have ambiguous or unavailable polariton assignments. The full run did not persist QED-CAS assignment scores; its saved LP/UP roots were chosen by the same projection criterion, so their assignment confidence cannot be independently audited from this cache." + note)
        suffix = "natural_occupations" if is_natural else "rdm"
        (RESULTS / f"{prefix}_draft_caption_{suffix}.txt").write_text(text + "\n")
    print(f"Figures: {FIGURES}")
    print(f"Summary and captions: {RESULTS}")


if __name__ == "__main__":
    main()
