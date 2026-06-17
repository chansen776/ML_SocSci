"""
Diagnostic visualizations for the Abortion/Crime PLM-FE example.

Reads the tidy CSV produced by "Stata Code/DLfePLM_viz.do" and writes PNG
figures to Slides/figures/.  These are *optional* teaching/intuition figures
illustrating what the partially-linear regression with flexible state trends is
"loading on": as we go raw -> two-way-FE residual -> double-selection residual,
the variation that identifies the abortion coefficient collapses -- and it
collapses much harder for the (very smooth) treatment series than for the
outcomes.  The post-lasso fit panels show that a handful of baseline x cubic
trend terms track the effective-abortion series almost exactly.

Numbers are Stata-canonical (the double-selection sets are those chosen by
pdslasso in DLfePLM.do).  Run from anywhere:

    python "Python code/DLfePLM_viz.py"
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Switch: also emit the 31 auxiliary figures (raw / residual / small-multiples /
# full-spaghetti / variance-collapse)?  OFF by default -- only the six
# two-column 10-state overlays are produced.  Flip to True here, or pass --all
# on the command line.
GENERATE_EXTRA = False

# ---- paths (resolve repo root whether run from root or a subfolder) ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "Data")) else os.path.dirname(HERE)
CSV = os.path.join(ROOT, "Data", "DLfePLM_viz.csv")
FIGDIR = os.path.join(ROOT, "Slides", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- variable metadata ----
VARS = ["lpc_viol", "lpc_prop", "lpc_murd", "efaviol", "efaprop", "efamurd"]
LABEL = {
    "lpc_viol": "log violent-crime rate",
    "lpc_prop": "log property-crime rate",
    "lpc_murd": "log murder rate",
    "efaviol":  "effective abortion (violent)",
    "efaprop":  "effective abortion (property)",
    "efamurd":  "effective abortion (murder)",
}
# title-case labels for figure titles (used by the curated 10-state overlay)
TITLE_LABEL = {
    "lpc_viol": "Log Violent-Crime Rate",
    "lpc_prop": "Log Property-Crime Rate",
    "lpc_murd": "Log Murder Rate",
    "efaviol":  "Effective Abortion Rate (violent)",
    "efaprop":  "Effective Abortion Rate (property)",
    "efamurd":  "Effective Abortion Rate (murder)",
}
IS_TREAT = {v: v.startswith("efa") for v in VARS}
# crime-type partner (for FWL sanity check)
PARTNER = {"efaviol": "lpc_viol", "efaprop": "lpc_prop", "efamurd": "lpc_murd"}

# statenum -> USPS abbreviation. The data carries only statenum; the numbering
# is alphabetical-incl-DC (1=AL ... 9=DC ... 12=HI ... 51=WY).  Verified two
# ways: DLfePLM.do drops statenum 2/9/12 = AK/DC/HI, and the largest-population
# statenums (5/33/44/10/39/14) decode to CA/NY/TX/FL/PA/IL.
STATE_ABBR = {
    1: "AL", 2: "AK", 3: "AZ", 4: "AR", 5: "CA", 6: "CO", 7: "CT", 8: "DE",
    9: "DC", 10: "FL", 11: "GA", 12: "HI", 13: "ID", 14: "IL", 15: "IN",
    16: "IA", 17: "KS", 18: "KY", 19: "LA", 20: "ME", 21: "MD", 22: "MA",
    23: "MI", 24: "MN", 25: "MS", 26: "MO", 27: "MT", 28: "NE", 29: "NV",
    30: "NH", 31: "NJ", 32: "NM", 33: "NY", 34: "NC", 35: "ND", 36: "OH",
    37: "OK", 38: "OR", 39: "PA", 40: "RI", 41: "SC", 42: "SD", 43: "TN",
    44: "TX", 45: "UT", 46: "VT", 47: "VA", 48: "WA", 49: "WV", 50: "WI",
    51: "WY",
}
ABBR_TO_NUM = {v: k for k, v in STATE_ABBR.items()}

# Curated 10-state overlay: 5 "conservative" + 5 "liberal".  NY and CA legalized
# abortion in 1970 (pre-Roe), so the liberal series ramp earlier/higher.
OVERLAY_CONSERVATIVE = ["TX", "MS", "AL", "UT", "OK"]
OVERLAY_LIBERAL = ["CA", "NY", "MA", "OR", "WA"]

MAROON = "#7a1f2b"
BLUE = "#2c5d8a"
GREEN = "#1b7837"
DPI = 200

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})


def to_matrix(df, col):
    """year x state matrix for a given column."""
    m = df.pivot(index="year", columns="statenum", values=col).sort_index()
    return m


def spaghetti(mat, title, ylabel, ylim, outpath, color=BLUE):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    years = mat.index.values
    ax.plot(years, mat.values, color=color, lw=0.7, alpha=0.35)
    ax.plot(years, mat.mean(axis=1).values, color="black", lw=2.2,
            label="cross-state mean")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("year")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.margins(x=0.01)
    ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fit_smallmultiples(actual, fit, twfe, var, ylim, outpath):
    states = list(actual.columns)
    years = actual.index.values
    nrow, ncol = 6, 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 11),
                             sharex=True, sharey=True)
    for k, st in enumerate(states):
        ax = axes.flat[k]
        ax.plot(years, twfe[st].values, color=BLUE, lw=1.2, ls=":")
        ax.plot(years, fit[st].values, color=MAROON, lw=1.2, ls="--")
        ax.plot(years, actual[st].values, color="black", lw=1.3)
        ax.set_title(STATE_ABBR.get(int(st), f"state {int(st)}"), fontsize=8)
        ax.tick_params(labelsize=7)
        if ylim is not None:
            ax.set_ylim(ylim)
    for k in range(len(states), nrow * ncol):
        axes.flat[k].axis("off")
    handles = [plt.Line2D([], [], color="black", lw=1.6, label="actual"),
               plt.Line2D([], [], color=MAROON, lw=1.6, ls="--",
                          label="post-lasso fit"),
               plt.Line2D([], [], color=BLUE, lw=1.6, ls=":",
                          label="baseline TWFE fit")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{LABEL[var]}: actual vs. post-lasso fit, by state",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fit_spaghetti(actual, fit, twfe, var, ylim, outpath, states=None,
                  scope_label="all states", alpha=0.30, lw=0.7, note=None,
                  title=None, dl=None, figsize=(7.2, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    years = actual.index.values
    a = actual if states is None else actual[states]
    f = fit if states is None else fit[states]
    t = twfe if states is None else twfe[states]
    ax.plot(years, t.values, color=BLUE, lw=lw, alpha=alpha * 0.85, ls=":")
    if dl is not None:
        d = dl if states is None else dl[states]
        ax.plot(years, d.values, color=GREEN, lw=lw, alpha=alpha, ls="-.")
    ax.plot(years, f.values, color=MAROON, lw=lw, alpha=alpha, ls="-")
    ax.plot(years, a.values, color="black", lw=lw, alpha=alpha)
    handles = [plt.Line2D([], [], color="black", lw=1.6, label="actual"),
               plt.Line2D([], [], color=MAROON, lw=1.6, label="post-lasso fit"),
               plt.Line2D([], [], color=BLUE, lw=1.6, ls=":",
                          label="baseline TWFE fit")]
    if dl is not None:
        handles.append(plt.Line2D([], [], color=GREEN, lw=1.6, ls="-.",
                                  label="DL baseline fit"))
        # keep the plot-to-legend gap ~constant in inches as the figure grows
        yoff = -0.13 * (4.6 / figsize[1])
        ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                  fontsize=9, bbox_to_anchor=(0.5, yoff))
    else:
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("year")
    ax.set_ylabel(LABEL[var])
    ttl = title if title is not None else \
        f"{LABEL[var]}: actual vs. post-lasso fit ({scope_label})"
    ax.set_title(ttl, fontsize=11)
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8,
                color="gray", ha="left", va="top")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI,
                bbox_inches="tight" if dl is not None else None)
    plt.close(fig)


def main():
    df = pd.read_csv(CSV)

    # pre-compute matrices
    raw = {v: to_matrix(df, v) for v in VARS}
    r2 = {v: to_matrix(df, f"r2_{v}") for v in VARS}
    r3 = {v: to_matrix(df, f"r3_{v}") for v in VARS}
    fit = {v: to_matrix(df, f"fit_{v}") for v in VARS}
    dl_fit = {v: to_matrix(df, f"base_{v}") for v in VARS}
    # baseline two-way-FE fit (original scale) = actual - TWFE residual
    twfe_fit = {v: raw[v] - r2[v] for v in VARS}

    # curated 10-state overlay: 5 conservative + 5 liberal (listed
    # alphabetically on the figure, with no political grouping shown)
    SUBSET = [ABBR_TO_NUM[a] for a in OVERLAY_CONSERVATIVE + OVERLAY_LIBERAL]
    SUBSET_NOTE = "states: " + ", ".join(
        sorted(OVERLAY_CONSERVATIVE + OVERLAY_LIBERAL))

    # ---- shared y-limits ----
    # group A: raw + post-lasso fit (original scale)
    # group B: the two residual stages (centered at 0, symmetric)
    ylim_orig, ylim_resid = {}, {}
    for v in VARS:
        lo = min(raw[v].values.min(), fit[v].values.min(),
                 twfe_fit[v].values.min(), dl_fit[v].values.min())
        hi = max(raw[v].values.max(), fit[v].values.max(),
                 twfe_fit[v].values.max(), dl_fit[v].values.max())
        pad = 0.04 * (hi - lo)
        ylim_orig[v] = (lo - pad, hi + pad)
        m = max(np.abs(r2[v].values).max(), np.abs(r3[v].values).max())
        ylim_resid[v] = (-1.04 * m, 1.04 * m)

    gen_extra = GENERATE_EXTRA or ("--all" in sys.argv)

    # per-figure lower-y overrides for the 10-state overlays (trim white space)
    YLO10 = {"lpc_viol": 0.5}

    # ---- per-variable figures ----
    for v in VARS:
        # the curated two-column 10-state overlay (always generated)
        ylim10 = (YLO10.get(v, ylim_orig[v][0]), ylim_orig[v][1])
        fit_spaghetti(raw[v], fit[v], twfe_fit[v], v, ylim10,
                      os.path.join(FIGDIR, f"FEPLM_fitspag10_{v}.png"),
                      states=SUBSET, alpha=0.75, lw=1.1, note=SUBSET_NOTE,
                      title=f"{TITLE_LABEL[v]}: Actual vs Fits",
                      dl=dl_fit[v], figsize=(7.2, 10.35))

        if not gen_extra:
            continue
        # ---- auxiliary figures (opt-in via GENERATE_EXTRA / --all) ----
        spaghetti(raw[v], f"{LABEL[v]}: raw series", LABEL[v],
                  ylim_orig[v], os.path.join(FIGDIR, f"FEPLM_raw_{v}.png"))
        spaghetti(r2[v], f"{LABEL[v]}: residual on state + year FE",
                  "residual", ylim_resid[v],
                  os.path.join(FIGDIR, f"FEPLM_residTWFE_{v}.png"))
        spaghetti(r3[v], f"{LABEL[v]}: residual on FE + DS controls",
                  "residual", ylim_resid[v],
                  os.path.join(FIGDIR, f"FEPLM_residDS_{v}.png"))
        fit_smallmultiples(raw[v], fit[v], twfe_fit[v], v, ylim_orig[v],
                           os.path.join(FIGDIR, f"FEPLM_fit_{v}.png"))
        fit_spaghetti(raw[v], fit[v], twfe_fit[v], v, ylim_orig[v],
                      os.path.join(FIGDIR, f"FEPLM_fitspag_{v}.png"))

    # ---- variance-collapse summary ----
    rows = []
    for v in VARS:
        var2 = np.nanvar(r2[v].values)
        var3 = np.nanvar(r3[v].values)
        rows.append((v, var2, var3, var3 / var2))
    summ = pd.DataFrame(rows, columns=["var", "v_twfe", "v_ds", "frac"])

    if gen_extra:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        x = np.arange(len(VARS))
        colors = [MAROON if IS_TREAT[v] else BLUE for v in VARS]
        ax.bar(x, summ["frac"].values, color=colors, alpha=0.85)
        for xi, f in zip(x, summ["frac"].values):
            ax.text(xi, f + 0.01, f"{f:.2f}", ha="center", va="bottom",
                    fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[v] for v in VARS], rotation=30, ha="right",
                           fontsize=9)
        ax.set_ylabel("Var(DS residual) / Var(two-way-FE residual)")
        ax.set_title("Within-variation surviving the flexible-trend selection")
        handles = [plt.Line2D([], [], color=BLUE, lw=8, label="outcome"),
                   plt.Line2D([], [], color=MAROON, lw=8, label="treatment")]
        ax.legend(handles=handles, frameon=False, fontsize=10)
        ax.set_ylim(0, max(1.0, summ["frac"].max() * 1.15))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "FEPLM_variance_collapse.png"),
                    dpi=DPI)
        plt.close(fig)

    # ---- console report: FWL sanity check + variance table ----
    print("Variance of residuals (across 624 obs):")
    print(summ.to_string(index=False,
                         formatters={"v_twfe": "{:.4e}".format,
                                     "v_ds": "{:.4e}".format,
                                     "frac": "{:.3f}".format}))
    print("\nFWL check  cov(r3_y, r3_d)/var(r3_d)  vs pdslasso PDS-selected:")
    for d, y in PARTNER.items():
        rd = r3[d].values.ravel()
        ry = r3[y].values.ravel()
        beta = np.cov(ry, rd, ddof=0)[0, 1] / np.var(rd)
        print(f"  {y} ~ {d}: FWL beta = {beta:.5f}")
    print(f"\n10-state overlay subset: "
          f"{[STATE_ABBR.get(int(s)) for s in SUBSET]}")
    print(f"\nWrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
