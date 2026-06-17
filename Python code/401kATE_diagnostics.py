"""
Reconstruct the per-spec ATE estimates and learner-quality diagnostics from
saved cross-fitted predictions in Data/401kATE.dta. Produces three figures
for the new "diagnostics" frames in §7 of ML and Causal Inference.tex:

    Slides/figures/ATE_variation_hist.png      - F6 variation across learners
    Slides/figures/ATE_psr2_leaderboard.png    - F7 pseudo-R^2 by learner
    Slides/figures/ATE_overlap_panels.png      - F8 propensity overlap

Numbers prototyped from the 13 Jun 2025 .dta. When the in-flight S22/S23-style
rerun finishes, point DATA at the fresh .dta and rerun this script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "401kATE.dta"
FIGDIR = ROOT / "Slides" / "figures"
NOTES = ROOT / "notes"
FIGDIR.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

N_REPS = 10
Y_LEARNERS = ["Y1_reg", "Y2_reg", "Y3_pystacked", "Y4_pystacked",
              "Y5_pystacked", "Y6_pystacked"]
D_LEARNERS = ["D1_logit", "D2_pystacked", "D3_pystacked", "D4_pystacked",
              "D5_pystacked", "D6_pystacked"]

# Display labels matching the .do file's learner specification.
Y_LABELS = [r"reg($X$)",
            r"reg($X^{poly}$)",
            r"Lasso",
            r"Ridge",
            r"RF",
            r"NN"]
D_LABELS = [r"logit($X$)",
            r"logit($X^{poly}$)",
            r"Lasso (Logit)",
            r"Ridge (Logit)",
            r"RF",
            r"NN"]

# Maroon to match the slide theme; gray + light-blue for histogram split.
MAROON = "#7B1F2A"
GOOD_BLUE = "#3C6FB6"
GRAY = "#A0A0A0"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print(f"reading {DATA}")
df = pd.read_stata(DATA)
print(f"shape: {df.shape}")

Y = df["net_tfa"].values.astype(float)
D = df["e401"].values.astype(float)
N = len(df)

p_bar = D.mean()
var_D = p_bar * (1 - p_bar)
var_Y0 = Y[D == 0].var(ddof=0)
var_Y1 = Y[D == 1].var(ddof=0)
print(f"N={N}, p_bar={p_bar:.4f}, Var(D)={var_D:.4f}")
print(f"Var(Y|D=0)={var_Y0:.3e}, Var(Y|D=1)={var_Y1:.3e}")


# ---------------------------------------------------------------------------
# Pseudo-R^2 per learner per arm
# ---------------------------------------------------------------------------

def y_psr2(arm):
    """Mean over reps of pseudo-R^2 for each Y learner on D==arm subset."""
    mask = D == arm
    var = Y[mask].var(ddof=0)
    rows = []
    for L in Y_LEARNERS + ["Y_net_tfa_ss"]:
        per_rep = []
        per_rep_mse = []
        for r in range(1, N_REPS + 1):
            yhat = df[f"{L}{arm}_{r}"].values.astype(float)
            mse = np.mean((Y[mask] - yhat[mask]) ** 2)
            per_rep.append(1 - mse / var)
            per_rep_mse.append(mse)
        rows.append({"learner": L,
                     "psr2_mean": np.mean(per_rep),
                     "psr2_min": np.min(per_rep),
                     "psr2_max": np.max(per_rep),
                     "mse_mean": np.mean(per_rep_mse)})
    return pd.DataFrame(rows)


def d_psr2():
    rows = []
    for L in D_LEARNERS + ["D_e401_ss"]:
        per_rep = []
        per_rep_mse = []
        for r in range(1, N_REPS + 1):
            phat = df[f"{L}_{r}"].values.astype(float)
            mse = np.mean((D - phat) ** 2)
            per_rep.append(1 - mse / var_D)
            per_rep_mse.append(mse)
        rows.append({"learner": L,
                     "psr2_mean": np.mean(per_rep),
                     "psr2_min": np.min(per_rep),
                     "psr2_max": np.max(per_rep),
                     "mse_mean": np.mean(per_rep_mse)})
    return pd.DataFrame(rows)


psr2_y0 = y_psr2(0)
psr2_y1 = y_psr2(1)
psr2_d = d_psr2()
print("\npseudo-R^2 E[Y|X,D=0]:")
print(psr2_y0.to_string(index=False))
print("\npseudo-R^2 E[Y|X,D=1]:")
print(psr2_y1.to_string(index=False))
print("\npseudo-R^2 E[D|X]:")
print(psr2_d.to_string(index=False))


# ---------------------------------------------------------------------------
# Overlap diagnostics for D learners
# ---------------------------------------------------------------------------

def d_overlap():
    rows = []
    for L in D_LEARNERS + ["D_e401_ss"]:
        phats = np.concatenate([df[f"{L}_{r}"].values.astype(float)
                                for r in range(1, N_REPS + 1)])
        # tiny clip to avoid divide-by-zero in leverage; doesn't affect bins
        ph = np.clip(phats, 1e-8, 1 - 1e-8)
        frac_extreme = float(np.mean((ph < 0.05) | (ph > 0.95)))
        max_lev = float(np.max(1.0 / (ph * (1 - ph))))
        min_marg = float(np.min(np.minimum(ph, 1 - ph)))
        rows.append({"learner": L,
                     "frac_extreme": frac_extreme,
                     "max_leverage": max_lev,
                     "min_marg": min_marg,
                     "phats": ph})
    return rows


overlap = d_overlap()
print("\noverlap diagnostics for D:")
for row in overlap:
    print(f"  {row['learner']:15s}  frac_extreme={row['frac_extreme']:.3%}  "
          f"max_leverage={row['max_leverage']:.0f}  "
          f"min(p,1-p)={row['min_marg']:.4g}")


# ---------------------------------------------------------------------------
# Allcombos b/SE per (rep, Y(0)_learner, Y(1)_learner, D_learner)
# ---------------------------------------------------------------------------

def aipw_score(Y, D, m0, m1, p):
    """Per-obs AIPW influence function. theta=mean; SE=std/sqrt(n).

    Clip p to (.01, .99) so a learner with p_hat = 0 or 1 (e.g. RF leaves
    pure D=0 or D=1 sets, NN saturating activations) does not explode the
    score; ddml's internal allcombos uses similarly bounded leverage and
    gives a comparable b range.
    """
    p_clip = np.clip(p, 0.01, 0.99)
    return m1 - m0 + D * (Y - m1) / p_clip - (1 - D) * (Y - m0) / (1 - p_clip)


print("\nbuilding allcombos table (216 specs x 10 reps = 2160 rows)...")
allcombos_rows = []
for r in range(1, N_REPS + 1):
    for L0 in Y_LEARNERS:
        m0 = df[f"{L0}0_{r}"].values.astype(float)
        for L1 in Y_LEARNERS:
            m1 = df[f"{L1}1_{r}"].values.astype(float)
            for LD in D_LEARNERS:
                p = df[f"{LD}_{r}"].values.astype(float)
                psi = aipw_score(Y, D, m0, m1, p)
                allcombos_rows.append({"rep": r, "Y0": L0, "Y1": L1, "D": LD,
                                       "b": psi.mean(),
                                       "se": psi.std(ddof=1) / np.sqrt(N)})
allcombos = pd.DataFrame(allcombos_rows)
print(f"allcombos shape: {allcombos.shape}")
print(f"  b range: [{allcombos.b.min():.0f}, {allcombos.b.max():.0f}]")
print(f"  median b across all specs: {allcombos.b.median():.0f}")
allcombos.to_csv(NOTES / "401kATE_allcombos_recon.csv", index=False)


# Per-rep best (min-MSE Y0, Y1, D individually) and shortstack picks.
def best_picks_per_rep():
    out = []
    for r in range(1, N_REPS + 1):
        best_y0 = min(Y_LEARNERS, key=lambda L: np.mean(
            (Y[D == 0] - df[f"{L}0_{r}"].values[D == 0]) ** 2))
        best_y1 = min(Y_LEARNERS, key=lambda L: np.mean(
            (Y[D == 1] - df[f"{L}1_{r}"].values[D == 1]) ** 2))
        best_d = min(D_LEARNERS, key=lambda L: np.mean(
            (D - df[f"{L}_{r}"].values) ** 2))
        m0 = df[f"{best_y0}0_{r}"].values.astype(float)
        m1 = df[f"{best_y1}1_{r}"].values.astype(float)
        p = df[f"{best_d}_{r}"].values.astype(float)
        psi = aipw_score(Y, D, m0, m1, p)
        out.append({"rep": r, "Y0": best_y0, "Y1": best_y1, "D": best_d,
                    "b": psi.mean(), "se": psi.std(ddof=1) / np.sqrt(N)})
    return pd.DataFrame(out)


def stack_picks_per_rep():
    out = []
    for r in range(1, N_REPS + 1):
        m0 = df[f"Y_net_tfa_ss0_{r}"].values.astype(float)
        m1 = df[f"Y_net_tfa_ss1_{r}"].values.astype(float)
        p = df[f"D_e401_ss_{r}"].values.astype(float)
        psi = aipw_score(Y, D, m0, m1, p)
        out.append({"rep": r, "b": psi.mean(),
                    "se": psi.std(ddof=1) / np.sqrt(N)})
    return pd.DataFrame(out)


best = best_picks_per_rep()
stack = stack_picks_per_rep()
print(f"\nbest picks median b: {best.b.median():.0f}; mean b: {best.b.mean():.0f}")
print(f"stack picks median b: {stack.b.median():.0f}; mean b: {stack.b.mean():.0f}")


# ---------------------------------------------------------------------------
# Frame 6 - variation histogram (color-coded by good/bad)
# ---------------------------------------------------------------------------

# "Good" = passes BOTH diagnostics on conservative thresholds:
#   * pseudo-R^2 non-negative => excludes Y2_reg.
#   * frac extreme < 1% AND max leverage < 1000 => only D3, D4 qualify.
#     (D1 logit fails pseudo-R^2 marginally; D2 fails max-leverage on
#     a rogue obs; D5, D6 fail overlap visibly.)
GOOD_Y = {"Y1_reg", "Y3_pystacked", "Y4_pystacked",
          "Y5_pystacked", "Y6_pystacked"}
GOOD_D = {"D3_pystacked", "D4_pystacked"}

allcombos["is_good"] = (allcombos.Y0.isin(GOOD_Y) & allcombos.Y1.isin(GOOD_Y)
                        & allcombos.D.isin(GOOD_D))
print(f"\ngood specs: {allcombos.is_good.sum()} of {len(allcombos)} "
      f"({allcombos.is_good.mean():.1%})")
print(f"good b range: [{allcombos[allcombos.is_good].b.min():.0f}, "
      f"{allcombos[allcombos.is_good].b.max():.0f}]")
print(f"any-bad b range: [{allcombos[~allcombos.is_good].b.min():.0f}, "
      f"{allcombos[~allcombos.is_good].b.max():.0f}]")


def fig_variation():
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    XLO, XHI = -6000, 22000
    bins = np.linspace(XLO, XHI, 57)  # ~$500 bin width
    bad = allcombos[~allcombos.is_good].b.values
    good = allcombos[allcombos.is_good].b.values
    bad_in = bad[(bad >= XLO) & (bad <= XHI)]
    good_in = good[(good >= XLO) & (good <= XHI)]
    bad_lo = int(np.sum(bad < XLO))
    bad_hi = int(np.sum(bad > XHI))

    ax.hist(bad_in, bins=bins, color=GRAY, alpha=0.80,
            label=(r"$\geq 1$ learner fails a diagnostic"
                   f"  (n = {len(bad)})"),
            edgecolor="white", linewidth=0.3)
    ax.hist(good_in, bins=bins, color=GOOD_BLUE, alpha=0.95,
            label=("all learners pass both diagnostics"
                   f"  (n = {len(good)})"),
            edgecolor="white", linewidth=0.3)

    ymax = ax.get_ylim()[1]
    tick_y_best = -ymax * 0.08
    tick_y_stack = -ymax * 0.16
    ax.scatter(best.b.values, [tick_y_best] * len(best),
               marker=(5, 1, 0), s=90, color=MAROON, zorder=5,
               clip_on=False, label="best (per-rep, n=10)")
    ax.scatter(stack.b.values, [tick_y_stack] * len(stack),
               marker="^", s=70, color="black", zorder=5,
               clip_on=False, label="shortstack (per-rep, n=10)")

    # Off-axis annotations -- keep below legend.
    if bad_hi > 0:
        ax.annotate(f"{bad_hi} estimates above \\${XHI//1000}k\n"
                    f"(involve $Y_2$ reg($X^{{poly}}$))",
                    xy=(XHI, ymax * 0.30),
                    xytext=(XHI * 0.78, ymax * 0.50),
                    ha="center", fontsize=8.5, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.8))
    if bad_lo > 0:
        ax.annotate(f"{bad_lo} estimates below \\${XLO//1000}k\n"
                    f"(extreme propensities)",
                    xy=(XLO, ymax * 0.55),
                    xytext=(XLO * 0.55, ymax * 0.78),
                    ha="center", fontsize=8.5, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.8))

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(tick_y_stack * 1.4, ymax * 1.05)
    ax.set_xlabel(r"ATE estimate $\hat\theta$ (\$)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of ATE estimates across all 216 specifications "
                 "$\\times$ 10 resamples = 2,160 estimates",
                 fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(FIGDIR / "ATE_variation_hist.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR / 'ATE_variation_hist.png'}")


# ---------------------------------------------------------------------------
# Frame 7 - pseudo-R^2 leaderboard (3 panels)
# ---------------------------------------------------------------------------

def fig_psr2():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)

    panels = [(axes[0], psr2_y0, Y_LABELS + ["shortstack"],
               r"$E[Y\mid X, D=0]$"),
              (axes[1], psr2_y1, Y_LABELS + ["shortstack"],
               r"$E[Y\mid X, D=1]$"),
              (axes[2], psr2_d,  D_LABELS + ["shortstack"],
               r"$E[D\mid X]$")]

    for ax, dfp, labels, title in panels:
        vals = dfp["psr2_mean"].values
        ypos = np.arange(len(vals))[::-1]
        colors = []
        for L, v in zip(dfp["learner"].values, vals):
            if L.startswith("Y_net_tfa_ss") or L.startswith("D_e401_ss"):
                colors.append(MAROON)
            elif v < 0:
                colors.append("#C2433A")
            else:
                colors.append(GOOD_BLUE)

        # Plot bars: clip negative values to a uniform short red stub so the
        # in-bar area stays free for an "off chart" label inside.
        plotted = np.where(vals < 0, -0.015, vals)
        ax.barh(ypos, plotted, color=colors, edgecolor="white", linewidth=0.5)

        for i, v in enumerate(vals):
            ypos_i = ypos[i]
            if v < 0:
                # Place "off chart" label to the right of the red stub,
                # in red, with the actual value.
                if abs(v) >= 1:
                    txt = f"$R^2 = {v:.1f}$  ⚠ off chart"
                else:
                    txt = f"$R^2 = {v:+.2f}$  ⚠"
                ax.annotate(txt, xy=(0.005, ypos_i),
                            xytext=(0, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=9,
                            color="#C2433A", fontweight="bold")
            else:
                ax.annotate(f"{v:.3f}", xy=(v, ypos_i),
                            xytext=(3, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=8.5)

        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("pseudo-$R^2$")
        ax.set_title(title, fontsize=10.5)
        xmax = max(0.05, np.max(vals) * 1.40)
        ax.set_xlim(-0.025, xmax)
        ax.axvline(0, color="black", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle(r"Cross-fit pseudo-$R^2 = 1 - \mathrm{MSE}/\mathrm{Var}"
                 r"(\mathrm{target})$  (mean over 10 resamples)",
                 fontsize=11)
    fig.savefig(FIGDIR / "ATE_psr2_leaderboard.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR / 'ATE_psr2_leaderboard.png'}")


# ---------------------------------------------------------------------------
# Frame 8 - propensity overlap (7 small histograms + table)
# ---------------------------------------------------------------------------

def fig_overlap():
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 7, wspace=0.30, hspace=0.45,
                          height_ratios=[3.0, 1.1],
                          left=0.04, right=0.99, top=0.78, bottom=0.05)
    bins = np.linspace(0, 1, 41)
    labels = D_LABELS + ["shortstack"]

    threshold_extreme = 0.01  # 1% — anything beyond this is the "bad" group
    threshold_lev = 1000

    for i, (row, label) in enumerate(zip(overlap, labels)):
        ax = fig.add_subplot(gs[0, i])
        phats = row["phats"]
        ax.hist(phats, bins=bins, color=GOOD_BLUE, alpha=0.95,
                edgecolor="white", linewidth=0.2)
        extreme = phats[(phats < 0.05) | (phats > 0.95)]
        if len(extreme) > 0:
            ax.hist(extreme, bins=bins, color="#C2433A", alpha=0.95,
                    edgecolor="white", linewidth=0.2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#404040")
        ax.set_title(label, fontsize=9.5)

    # Summary table row
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")
    cell_text = [["frac. of $\\hat p$ in $[.05,.95]^c$"],
                 ["max  $1/[\\hat p (1 - \\hat p)]$"]]
    for i, row in enumerate(overlap):
        bad_extr = row["frac_extreme"] > threshold_extreme
        bad_lev = row["max_leverage"] > threshold_lev
        # Note: append extreme col then leverage col per learner
        # We'll build columns inline.
        cell_text[0].append(f"{row['frac_extreme']*100:.2f}%")
        # Format leverage: scientific if huge.
        lv = row["max_leverage"]
        if lv >= 1e4:
            cell_text[1].append(f"{lv:.1e}")
        else:
            cell_text[1].append(f"{lv:.0f}")
    col_labels = [""] + labels
    tbl = table_ax.table(cellText=cell_text, colLabels=col_labels,
                         loc="upper center", cellLoc="center",
                         colWidths=[0.18] + [0.115] * 7)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)
    # Color the "bad" cells red
    for j, row in enumerate(overlap, start=1):
        if row["frac_extreme"] > threshold_extreme:
            tbl[(1, j)].set_text_props(color="#C2433A", fontweight="bold")
        if row["max_leverage"] > threshold_lev:
            tbl[(2, j)].set_text_props(color="#C2433A", fontweight="bold")
    # Hide the leftmost cell borders (label column)
    for r in range(3):
        cell = tbl[(r, 0)]
        cell.set_text_props(ha="right", fontsize=8.5)
        cell.visible_edges = ""
    # Hide the column-header cell at (0, 0)
    tbl[(0, 0)].set_text_props(text="")

    fig.suptitle(r"Distribution of $\hat p(X) = \widehat{P}(D=1\mid X)$  "
                 "(pooled over 10 resamples)", fontsize=11.5, y=0.96)

    legend_handles = [
        Patch(facecolor=GOOD_BLUE, label=r"$\hat p \in [.05, .95]$"),
        Patch(facecolor="#C2433A",
              label=r"$\hat p \notin [.05, .95]$  (extreme; high leverage)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.90))

    fig.savefig(FIGDIR / "ATE_overlap_panels.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR / 'ATE_overlap_panels.png'}")


fig_variation()
fig_psr2()
fig_overlap()
print("\ndone.")
