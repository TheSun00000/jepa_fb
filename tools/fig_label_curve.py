"""Label-budget figure (return vs number of labelled states), Meta-style blue template.

Reads the tracker document `paper/label_curve` (saved with ArtifactData list ... out_dir=hpc/db_read)
and writes paper/figures/label_curve.pdf (+ .png for a quick look).

    python tools/fig_label_curve.py [--src hpc/db_read/paper/label_curve.json] [--out paper/figures/label_curve]
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BLUE, BLUE_L, NAVY, GRAY, GRAY_L, INK = "#0064E0", "#7DB3F5", "#1C2B33", "#8A98A5", "#C4CCD3", "#1C2B33"
# key, legend label, colour, marker, linestyle, linewidth, zorder
SERIES = [("fbwm", "FBWM (MPPI)", BLUE, "o", "-", 2.4, 5),
          ("fbwm_frozen", "FB on frozen WM", BLUE_L, "o", (0, (1.5, 1.5)), 1.8, 4),
          ("fb", "FB", NAVY, "s", "-", 1.6, 3),
          ("adapt_labels", "Adapted WM (labels only)", GRAY, "D", "-", 1.6, 2)]
ENVS = ["walker", "cheetah", "quadruped"]


def pick_font():
    have = {f.name for f in fm.fontManager.ttflist}
    for n in ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]:
        if n in have:
            return n
    return "sans-serif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hpc/db_read/paper/label_curve.json")
    ap.add_argument("--out", default="paper/figures/label_curve")
    a = ap.parse_args()
    c = json.load(open(a.src)); c = c.get("data", c)
    S = {s["key"]: s for s in c["series"]}
    R = {r["key"]: r for r in c["refs"]}
    plt.rcParams.update({
        "font.family": pick_font(), "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": GRAY_L, "xtick.color": GRAY, "ytick.color": GRAY, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.5), sharex=True)
    for ax, e in zip(axs, ENVS):
        ref = R["tdmpc"]["per_env"][e]
        ax.axhline(ref, color=GRAY_L, lw=1.1, ls=(0, (4, 3)), zorder=1)
        for k, lab, col, mk, ls, lw, z in SERIES:
            pts = S[k]["points"]
            ax.plot([p["N"] for p in pts], [p["per_env"][e] for p in pts], color=col, ls=ls, lw=lw, zorder=z,
                    marker=mk, ms=4.2 if k == "fbwm" else 3.4, mfc=col, mec="white", mew=0.8, label=lab,
                    solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_ylim(0, 900); ax.set_yticks([0, 300, 600, 900])
        ax.grid(axis="y", color="#EEF1F4", lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=4)
        ax.set_title(e.capitalize(), fontsize=9.5, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel("Labelled states $N$", fontsize=8.5, labelpad=5)
        ax.text(1.0, ref + 12, "TD-MPC", color=GRAY, fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    axs[0].set_ylabel("Return", fontsize=8.5)
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.36),
               handlelength=2.2, columnspacing=1.8)
    fig.subplots_adjust(wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
