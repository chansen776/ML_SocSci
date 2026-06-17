"""
Generates Slides/figures/Dube_sensitivity.png.

Static forest plot of the 12 candidate learners in the JEL paper's reanalysis
of Dube et al. (2020) monopsony, plus the three meta-learning summaries.
Numbers taken from Ahrens et al. (JEL) Tables 4 and 5 (xclust_K3 results).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Slides" / "figures" / "Dube_sensitivity.png"

LEARNERS = [
    ("OLS",      "Linear", -0.005, 0.058),
    ("CV-Lasso", "Linear", -0.008, 0.066),
    ("CV-Ridge", "Linear", -0.016, 0.076),
    ("RF 1",     "RF",     -0.191, 0.131),
    ("RF 2",     "RF",     -0.193, 0.137),
    ("RF 3",     "RF",     -0.184, 0.140),
    ("XGB 1",    "XGB",    -0.061, 0.024),
    ("XGB 2",    "XGB",    -0.056, 0.020),
    ("XGB 3",    "XGB",    -0.061, 0.029),
    ("NN 1",     "NN",     -0.071, 0.033),
    ("NN 2",     "NN",     -0.055, 0.042),
    ("NN 3",     "NN",     -0.060, 0.035),
]

META = [
    ("Stacking (NNLS)", -0.064, 0.019),
    ("Single-best",     -0.061, 0.016),
    ("CVC",             -0.063, 0.017),
]

FAMILY_COLOR = {
    "Linear": "#1f77b4",
    "RF":     "#d62728",
    "XGB":    "#2ca02c",
    "NN":     "#9467bd",
    "Meta":   "black",
}

DUBE_LO, DUBE_HI = -0.198, -0.030

rows = META[::-1] + [("", 0.0, 0.0)] + [(n, b, s) for (n, _, b, s) in LEARNERS][::-1]
n = len(rows)
ys = np.arange(n)

fig, ax = plt.subplots(figsize=(9.0, 5.4))

ax.axvspan(DUBE_LO, DUBE_HI, color="lightgray", alpha=0.45,
           label=r"Dube et al. (2020) range")
ax.axvline(0, color="black", linewidth=0.7)

for y, row in zip(ys, rows):
    name, beta, se = row
    if name == "":
        continue
    if name in [m[0] for m in META]:
        color = FAMILY_COLOR["Meta"]
        marker = "D"
        ms = 9
        lw = 2.2
    else:
        fam = next(f for n_, f, _, _ in LEARNERS if n_ == name)
        color = FAMILY_COLOR[fam]
        marker = "o"
        ms = 7
        lw = 1.6
    ax.errorbar(beta, y, xerr=1.96 * se, fmt=marker, color=color,
                ecolor=color, elinewidth=lw, capsize=3, markersize=ms)

ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlabel(r"$\hat\theta$  (log labor supply elasticity, neg.)", fontsize=11)
ax.set_xlim(-0.36, 0.10)
ax.grid(axis="x", color="0.85", linewidth=0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLOR["Linear"], markersize=8, label="Linear"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLOR["RF"], markersize=8, label="Random Forest"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLOR["XGB"], markersize=8, label="XGBoost"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLOR["NN"], markersize=8, label="Neural Net"),
    plt.Line2D([0], [0], marker="D", color="w",
               markerfacecolor="black", markersize=9, label="Meta-learner"),
    plt.Rectangle((0, 0), 1, 1, color="lightgray", alpha=0.45,
                  label="Dube et al. (2020) range"),
]
ax.legend(handles=handles, loc="upper center", frameon=False,
          fontsize=9, ncol=6,
          bbox_to_anchor=(0.5, -0.14))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}")
