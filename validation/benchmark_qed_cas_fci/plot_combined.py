"""Plot resonant polaritons with spectrum-screened bright-connected states."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import FIGURES, LAMBDA_MAGNITUDES, RESULTS

COLORS = {0.01: "#440154", 0.05: "#21918c", 0.1: "#90d743"}
MARKERS = {"ground": "o", "ground_offres": "P", "LP": "s", "UP": "^",
           "bright_dressed": "D"}
PARITY_STATES = tuple(MARKERS)
RATIO_STATES = {"ground", "ground_offres", "bright_dressed"}
NATURAL_BASELINE_FLOOR = 1.0e-8


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def combined_rows():
    resonant = read(RESULTS / "full_metrics.csv")
    detuned = read(RESULTS / "offres_metrics.csv")
    rows = []
    for r in resonant:
        if r.get("status") != "converged" or r.get("state") not in {"ground", "LP", "UP"}:
            continue
        if str(r.get("assignment_ambiguous", "False")).lower() == "true":
            continue
        x = dict(r)
        x["condition"] = "resonant"
        rows.append(x)
    for r in detuned:
        if r.get("status") != "converged" or r.get("state") not in {"ground", "bright_dressed"}:
            continue
        if r["state"] == "bright_dressed" and str(r.get("bright_dominated", "False")).lower() != "true":
            continue
        x = dict(r)
        x["state"] = "ground_offres" if r["state"] == "ground" else "bright_dressed"
        x["condition"] = "blue_detuned"
        rows.append(x)
    return rows


def selected(rows, xkey, rkey, natural=False, ratio=False):
    out = []
    for row in rows:
        if ratio:
            if row["state"] not in RATIO_STATES:
                continue
        lam, x, y = num(row.get("lambda_value")), num(row.get(xkey)), num(row.get(rkey))
        if lam not in LAMBDA_MAGNITUDES or x is None or y is None or x <= 0 or y <= 0:
            continue
        if natural and x < NATURAL_BASELINE_FLOOR:
            continue
        new = dict(row, lambda_num=lam, x_num=x, ratio_num=y, y_num=x * y)
        out.append(new)
    return out


def stats(rows, metric):
    result = []
    for lam in LAMBDA_MAGNITUDES:
        values = [r["ratio_num"] for r in rows if r["lambda_num"] == lam]
        result.append(dict(metric=metric, lambda_value=lam, n=len(values),
            median_ratio=float(np.median(values)) if values else "",
            min_ratio=min(values) if values else "", max_ratio=max(values) if values else ""))
    return result


def figure(rows, xkey, rkey, label, stem, natural=False):
    parity = selected(rows, xkey, rkey, natural=natural)
    ratios = selected(rows, xkey, rkey, natural=natural, ratio=True)
    if not parity or not ratios:
        raise RuntimeError(f"Insufficient rows for {stem}")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.12, 3.00))
    xs = np.asarray([r["x_num"] for r in parity]); ys = np.asarray([r["y_num"] for r in parity])
    log_axes = (xs.max() / xs.min() > 100 or ys.max() / ys.min() > 100)
    for lam in LAMBDA_MAGNITUDES:
        for state in PARITY_STATES:
            subset = [r for r in parity if r["lambda_num"] == lam and r["state"] == state]
            for stretched in (False, True):
                group = [r for r in subset if (r["geometry"] != "equilibrium") == stretched]
                if group:
                    ax0.scatter([r["x_num"] for r in group], [r["y_num"] for r in group],
                        marker=MARKERS[state], s=23, linewidths=0.9,
                        facecolors="none" if stretched else COLORS[lam],
                        edgecolors=COLORS[lam], alpha=0.9)
    if log_axes:
        ax0.set_xscale("log"); ax0.set_yscale("log")
    lo, hi = min(float(xs.min()), float(ys.min())), max(float(xs.max()), float(ys.max()))
    if log_axes:
        lo = max(lo, np.finfo(float).tiny)
    else:
        pad = 0.04 * (hi - lo if hi > lo else hi); lo, hi = max(0, lo-pad), hi+pad
    ax0.plot([lo, hi], [lo, hi], color="0.25", lw=0.9, ls="--")
    ax0.set_xlim(lo, hi); ax0.set_ylim(lo, hi)
    ax0.set_xlabel("Cavity-free CAS error"); ax0.set_ylabel(f"QED-CAS error ({label})")
    ax0.text(.02,.98,"(a)",transform=ax0.transAxes,va="top",ha="left",fontweight="bold")

    box_data = [[r["ratio_num"] for r in ratios if r["lambda_num"] == lam] for lam in LAMBDA_MAGNITUDES]
    bp = ax1.boxplot(box_data, positions=(1,2,3), widths=.46, patch_artist=True,
        showfliers=False, medianprops={"color":"0.15","lw":1.0},
        whiskerprops={"color":"0.45","lw":.8},capprops={"color":"0.45","lw":.8})
    for box in bp["boxes"]: box.set_facecolor("#E6E9EF"); box.set_edgecolor("#9AA1AA")
    rng=np.random.default_rng(731)
    for ix,lam in enumerate(LAMBDA_MAGNITUDES,start=1):
        vals=np.asarray([r["ratio_num"] for r in ratios if r["lambda_num"]==lam])
        if len(vals): ax1.scatter(ix+rng.uniform(-.16,.16,len(vals)),vals,s=12,color=COLORS[lam],edgecolors="white",linewidths=.35,alpha=.85)
    ax1.axhline(1,color="0.25",lw=.9,ls="--")
    if natural: ax1.set_yscale("log")
    ax1.set_xticks((1,2,3),("0.01","0.05","0.10"));ax1.set_xlabel(r"$|\lambda|$ (a.u.)")
    ax1.set_ylabel("QED / cavity-free error");ax1.text(.02,.98,"(b)",transform=ax1.transAxes,va="top",ha="left",fontweight="bold")
    ax1.grid(axis="y",color="0.90",lw=.5);ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False);ax1.spines["right"].set_visible(False)
    state_labels={"ground":"ground (resonant)","ground_offres":"ground (blue-detuned)",
                  "LP":"LP","UP":"UP","bright_dressed":"bright-connected (blue-detuned)"}
    state_handles=[Line2D([],[],marker=MARKERS[s],linestyle="",color="0.15",markerfacecolor="none",label=state_labels[s]) for s in PARITY_STATES]
    lambda_handles=[Line2D([],[],marker="o",linestyle="",color=COLORS[l],label=fr"$|\lambda|={l:g}$") for l in LAMBDA_MAGNITUDES]
    stretched=Line2D([],[],marker="o",linestyle="",color=".2",markerfacecolor="none",label="stretched")
    fig.legend(handles=state_handles+lambda_handles+[stretched],fontsize=5.7,frameon=False,ncol=4,
        loc="lower center",bbox_to_anchor=(.5,.005),handletextpad=.25,columnspacing=.65,borderpad=.1)
    for ax in (ax0,ax1): ax.tick_params(labelsize=7,width=.6,length=2.5);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    fig.subplots_adjust(left=.12,right=.99,top=.98,bottom=.40,wspace=.38)
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURES/f"{stem}.pdf",bbox_inches="tight");fig.savefig(FIGURES/f"{stem}.png",dpi=300,bbox_inches="tight");plt.close(fig)
    return parity, ratios


def main():
    rows=combined_rows()
    metrics=[("1-RDM Frobenius","cavity_free_frobenius","ratio","Frobenius norm","combined_rdm_invariance",False),
        ("Natural occupations","cavity_free_natural_occupation_error","natural_occupation_ratio","max occupation error","combined_natural_occupation_invariance",True)]
    summary_rows=[]
    for name,xkey,rkey,ylabel,stem,natural in metrics:
        parity,ratios=figure(rows,xkey,rkey,ylabel,f"full_{stem}",natural)
        summary_rows.extend(stats(ratios,name))
    write=RESULTS/"full_combined_summary.csv"
    with write.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    with (RESULTS/"full_combined_summary.md").open("w") as f:
        f.write("# Combined resonant and blue-detuned summary\n\n")
        f.write("Parity points include resonant ground/LP/UP and blue-detuned ground/bright-connected states. Ratios include ground states and only bright-connected off-resonant states.\n\n")
        f.write("| Metric | λ | n | Median ratio | Range |\n|---|---:|---:|---:|---:|\n")
        for r in summary_rows:
            span=f"{r['min_ratio']:.4g}–{r['max_ratio']:.4g}" if r['n'] else "—"
            med=f"{r['median_ratio']:.4g}" if r['n'] else "—"
            f.write(f"| {r['metric']} | {r['lambda_value']:.2f} | {r['n']} | {med} | {span} |\n")
        f.write(f"\nNatural-occupation ratios omit cavity-free reference errors below {NATURAL_BASELINE_FLOOR:g}.\n")
    print(f"Wrote combined figures and summary under {FIGURES} and {RESULTS}")


if __name__ == "__main__":
    main()
