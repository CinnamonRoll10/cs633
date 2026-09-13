"""
plot_boxplot.py  –  Boxplot of execution times for Assignment 2
Usage:  python3 plot_boxplot.py
Output: boxplot.pdf  and  boxplot.png 

Data: 5 SLURM runs per configuration on PARAM Rudra
      d=7, T=5, F=2, seed=1000, isovalue=500
      P in {32, 48, 64, 96}  |  nx=ny=nz in {120, 240}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Timing data (seconds) extracted from SLURM output files ──────────
# Key: (P, nx)  →  list of 5 run times
times = {
    (32, 120): [0.931039, 0.838224, 0.837680, 0.833514, 0.831772],
    (48, 120): [0.923572, 0.928945, 0.927285, 0.926439, 0.927624],
    (64, 120): [0.928037, 0.918290, 0.928014, 0.918450, 0.928776],
    (96, 120): [0.930433, 0.929890, 0.924112, 0.938078, 0.942659],

    (32, 240): [8.200271, 7.013630, 6.941709, 6.976569, 6.986681],
    (48, 240): [7.945173, 7.975379, 8.023718, 7.984967, 7.972415],
    (64, 240): [7.975691, 7.850545, 7.917984, 7.872502, 7.982502],
    (96, 240): [7.993240, 7.985098, 7.870532, 8.000168, 7.940612],
}

P_vals  = [32, 48, 64, 96]
NX_vals = [120, 240]

# ── Colour palette──────────────────────────────────
COL = {
    120: "#1A5291",   # deep blue
    240: "#D2691E",   # chocolate orange
}
MEDIAN_COL  = "white"
FLIER_COL   = "dimgray"
GRID_COL    = "#dddddd"

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.patch.set_facecolor("white")

for ax, nx in zip(axes, NX_vals):
    data    = [times[(P, nx)] for P in P_vals]
    labels  = [str(P) for P in P_vals]
    color   = COL[nx]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.45,
        positions=range(1, len(P_vals) + 1),
        medianprops=dict(color=MEDIAN_COL, linewidth=2.5),
        whiskerprops=dict(color=color, linewidth=1.4, linestyle="--"),
        capprops=dict(color=color, linewidth=1.8),
        boxprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markerfacecolor=FLIER_COL,
                        markeredgecolor=FLIER_COL, markersize=5, alpha=0.7),
    )

    # Fill boxes and add individual run dots
    for patch, d in zip(bp["boxes"], data):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    for i, (d, pos) in enumerate(zip(data, range(1, len(P_vals) + 1))):
        ax.scatter(
            [pos] * len(d), d,
            color="white", edgecolors=color, s=30, zorder=5,
            linewidths=1.2, alpha=0.9,
        )

    # Annotate medians
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + (0.003 if nx == 120 else 0.025),
                f"{med:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold", zorder=6)

    # Styling
    ax.set_xticks(range(1, len(P_vals) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Number of Processes  P", fontsize=12, labelpad=6)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12, labelpad=6)
    ax.set_title(
        f"$n_x = n_y = n_z = {nx}$  (per-process sub-domain)",
        fontsize=12, fontweight="bold", color="#1A5291", pad=8
    )
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aaaaaa")
    ax.tick_params(axis="both", labelsize=11, colors="#333333")

    # Weak-scaling efficiency annotations
    ref_mean = np.mean(times[(32, nx)])
    for i, P in enumerate(P_vals):
        mean_t = np.mean(times[(P, nx)])
        eff    = ref_mean / mean_t
        ax.text(i + 1, ax.get_ylim()[0] if i == 0 else ax.get_ylim()[0],
                f"E={eff:.2f}", ha="center", va="top",
                fontsize=7, color="#555555")

fig.suptitle(
    "MPI 3D Stencil — Execution Time vs Process Count\n"
    "PARAM Rudra  |  d=7, T=5, F=2, seed=1000, isovalue=500",
    fontsize=13, fontweight="bold", color="#1A5291", y=1.03,
)

# Legend
patches = [
    mpatches.Patch(facecolor=COL[nx], alpha=0.72, label=f"$n_x={nx}$")
    for nx in NX_vals
]
fig.legend(handles=patches, loc="upper right", fontsize=10,
           frameon=True, framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("boxplot.pdf", bbox_inches="tight", dpi=200)
plt.savefig("boxplot.png", bbox_inches="tight", dpi=200)
print("Saved boxplot.pdf and boxplot.png")
