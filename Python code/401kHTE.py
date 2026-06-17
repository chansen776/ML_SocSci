"""
401(k) Heterogeneous Treatment Effects — Day-2 Section 11 (canonical Python).

Worked example for the "Heterogeneous Effects" section, on the standard
9,915-observation 401(k) eligibility sample (Data/restatw.dat — the same sample
as the Section-7 ATE/LATE example, with the book's fuller covariate set).
Eligibility e401 is the (conditionally exogenous) treatment D; net total
financial assets net_tfa is the outcome Y.

Design (slide -> data fold):
  * Nuisances (m, g) are cross-fit ONCE over the full sample, giving a
    doubly-robust signal Y(eta) for every observation (DR-ATE ~ $8.3k, matching
    Section 7). Everything downstream treats Y(eta) as the outcome.
  * "Pre-specified scientific summaries" -- GATES by income, cubic-in-log-income
    BLP -- use the FULL sample with standard sqrt(n) semiparametric 95% bands.
  * Flexible CATE estimation + validation + policy learning use ONE 60/20/20
    train / validation / test split:
        - slide 73 illustration  : RF DR-learner fit on TRAIN, distilled to a tree
        - slide 74 model selection: 4 candidates fit on TRAIN, DR loss on VALIDATION
        - slide 75 calibration/het: VALIDATION, all four learners
        - slide 76 TOC / RATE     : VALIDATION, winner highlighted
        - slide 77 policy tree    : fit on TRAIN
        - slide 78 policy value   : evaluated out-of-sample on TEST
    Because the goal here is learning an implementable policy (decision theory),
    not testing a pre-specified scientific object, these frames report 68%
    (+/- 1 s.e.) bands -- 95% is an arbitrary benchmark with shaky
    decision-theoretic justification (Manski 2019).

Run from anywhere:  python "Python code/401kHTE.py"
Figures -> Slides/figures/HTE_401k_*.png ; tables -> notes/.
R/Stata are teaching parallels, refreshed separately.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression, RidgeCV
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              StackingRegressor, StackingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree

# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "Data" / "restatw.dat"
FIGDIR = ROOT / "Slides" / "figures"
NOTEDIR = ROOT / "notes"
FIGDIR.mkdir(parents=True, exist_ok=True)
NOTEDIR.mkdir(parents=True, exist_ok=True)

SEED = 731
N_FOLDS = 5
EPS = 0.02  # propensity trimming for the H-transform

# Critical values. The pre-specified summaries (GATES, BLP) use 95%; the
# flexible-modeling / policy frames use 68% (+/- 1 s.e.) -- see module docstring.
Z95 = 1.96
Z68 = 1.0                  # pointwise two-sided 68%
Z68_1S = float(norm.ppf(0.68))   # one-sided lower 68% (~0.468), for RATE bands

# Heterogeneity / control covariates (book's 401(k) set). X = Z (let the
# methods find the relevant dimensions); income drives the headline story.
COVARS = ["age", "inc", "educ", "fsize", "marr", "male", "twoearn",
          "db", "pira", "nohs", "hs", "smcol", "col", "hown"]

# colour per learner, reused across slides 74/75/76
LEARNER_COLORS = {"Boosting (d4, early-stop)": "C0", "Linear (age, income)": "C1",
                  "Random forest (mtry=p/3)": "C2", "Regularized linear (Ridge)": "C3"}
AI_COLS = [COVARS.index("age"), COVARS.index("inc")]   # age + income only

plt.rcParams.update({"figure.dpi": 200, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------
# Nuisance learners: a stacked ensemble (penalized-linear-on-poly + RF + HGB,
# combined by cross-validated stacking), mirroring Section-7's shortstack. A
# sensitivity check (notes/HTE_nuisance_sensitivity.py) confirmed the DR-ATE
# (~$8.3k) and the income gradient are stable across learner choice.
# --------------------------------------------------------------------------
def _reg():
    return StackingRegressor(
        estimators=[
            ("lasso", make_pipeline(StandardScaler(),
                                    PolynomialFeatures(2, include_bias=False),
                                    LassoCV(cv=5, max_iter=5000, n_jobs=1))),
            ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=20,
                                         random_state=SEED, n_jobs=1)),
            ("hgb", HistGradientBoostingRegressor(random_state=SEED)),
        ], cv=5, n_jobs=4)


def _clf():
    return StackingClassifier(
        estimators=[
            ("logit", make_pipeline(StandardScaler(),
                                    PolynomialFeatures(2, include_bias=False),
                                    LogisticRegressionCV(cv=5, max_iter=5000, n_jobs=1))),
            ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                          random_state=SEED, n_jobs=1)),
            ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
        ], cv=5, stack_method="predict_proba", n_jobs=4)


def crossfit_nuisances(Y, D, Z):
    """5-fold cross-fitted mu(Z)=P(D=1|Z), g0(Z)=E[Y|D=0,Z], g1(Z)=E[Y|D=1,Z]."""
    n = len(Y)
    mu = np.zeros(n)
    g0 = np.zeros(n)
    g1 = np.zeros(n)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(Z, D):
        Ztr, Dtr, Ytr = Z[tr], D[tr], Y[tr]
        m = _clf().fit(Ztr, Dtr)
        mu[te] = m.predict_proba(Z[te])[:, 1]
        c = _reg().fit(Ztr[Dtr == 0], Ytr[Dtr == 0])
        t = _reg().fit(Ztr[Dtr == 1], Ytr[Dtr == 1])
        g0[te] = c.predict(Z[te])
        g1[te] = t.predict(Z[te])
    mu = np.clip(mu, EPS, 1 - EPS)
    return mu, g0, g1


def dr_signal(Y, D, mu, g0, g1):
    """Doubly-robust pseudo-outcome Y(eta) = H (Y - g_D) + g1 - g0  (E[.|X]=CATE)."""
    H = D / mu - (1 - D) / (1 - mu)
    gD = D * g1 + (1 - D) * g0
    return H * (Y - gD) + g1 - g0


# --------------------------------------------------------------------------
# Inference helpers: OLS of the DR signal on a low-dim basis p(X), with
# pointwise and uniform (sup-t) confidence bands (Semenova-Chernozhukov style).
# --------------------------------------------------------------------------
def ols_blp(P, Ydr):
    """OLS beta of Ydr on basis P (n x d); HC-robust sandwich covariance."""
    Q = P.T @ P / len(P)
    Qinv = np.linalg.inv(Q)
    beta = Qinv @ (P.T @ Ydr / len(P))
    resid = Ydr - P @ beta
    meat = (P * resid[:, None]).T @ (P * resid[:, None]) / len(P)
    cov = Qinv @ meat @ Qinv / len(P)
    return beta, cov


def supt_crit(cov, alpha=0.05, one_sided=False, ndraw=20000, rng=None):
    """sup-t critical value for simultaneous bands from N(0,cov).
    one_sided=True gives the lower/upper one-sided uniform critical value.
    Uses an eigendecomposition (clipping tiny negative eigenvalues) so it is
    robust to a rank-deficient covariance -- e.g. tied CATE predictions make
    adjacent TOC grid points share a quantile threshold, giving duplicate
    influence columns."""
    rng = rng or np.random.default_rng(SEED)
    d = cov.shape[0]
    se = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    w, vec = np.linalg.eigh((cov + cov.T) / 2)
    L = vec @ np.diag(np.sqrt(np.clip(w, 0, None)))
    draws = (L @ rng.standard_normal((d, ndraw))) / se[:, None]
    stat = np.max(draws, axis=0) if one_sided else np.max(np.abs(draws), axis=0)
    return np.quantile(stat, 1 - alpha)


# --------------------------------------------------------------------------
# 60/20/20 split (governs the flexible-model / policy second stage only).
# --------------------------------------------------------------------------
def make_splits(n, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


# Flexible CATE candidates: name -> (estimator factory, feature set), where the
# feature set is "all" (the 14 regressors) or a list of covariate columns. Each
# regresses Y(eta) on its features and is scored by the doubly-robust loss out of
# sample. The set was chosen after a loss scan (notes/HTE_model_scan.py, since
# deleted): a well-regularized boosting and random forest (mtry = p/3, the
# regression rule of thumb), a structural model with only linear age+income
# heterogeneity (an economist's prior), and a regularized linear model over all
# regressors. Sensible regularization matters here: an un-tuned boosting badly
# overfits the heavy-tailed DR signal, while these all sit near or above the
# constant.
def candidate_specs():
    return {
        "Boosting (d4, early-stop)": (lambda: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=2000, early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.15, random_state=SEED), "all"),
        "Linear (age, income)": (LinearRegression, AI_COLS),
        "Random forest (mtry=p/3)": (lambda: RandomForestRegressor(
            n_estimators=2000, min_samples_leaf=120, max_features=1.0 / 3,
            random_state=SEED, n_jobs=4), "all"),
        "Regularized linear (Ridge)": (lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30))), "all"),
    }


def take(Z, feat):
    return Z if feat == "all" else Z[:, feat]


def cate_predict(fit, Zrows):
    """Predict a fitted candidate (est, feat) on the right feature subset."""
    est, feat = fit
    return est.predict(take(Zrows, feat))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_FILE, sep=r"\s+")
    Y = df["net_tfa"].to_numpy(float)
    D = df["e401"].to_numpy(int)
    Z = df[COVARS].to_numpy(float)
    inc = df["inc"].to_numpy(float)
    print(f"loaded {DATA_FILE.name}: n={len(Y)}, P(D=1)={D.mean():.3f}, "
          f"raw diff in means = {Y[D==1].mean()-Y[D==0].mean():,.0f}")

    mu, g0, g1 = crossfit_nuisances(Y, D, Z)
    Ydr = dr_signal(Y, D, mu, g0, g1)
    print(f"cross-fitted: mu in [{mu.min():.3f},{mu.max():.3f}]; "
          f"DR-ATE = mean Y(eta) = {Ydr.mean():,.1f} "
          f"(se {Ydr.std(ddof=1)/np.sqrt(len(Ydr)):,.1f})")

    # ===== Pre-specified scientific summaries (FULL sample, 95% bands) ======
    gates_by_income(inc, Ydr)
    blp_cubic_income(inc, Ydr)

    # ===== Flexible modeling / policy: 60/20/20 split, 68% bands ============
    tr, va, te = make_splits(len(Y))
    print(f"\nsplit: train={len(tr)}, val={len(va)}, test={len(te)}")

    # fit the four candidate CATE models once on TRAIN; reuse everywhere
    fitted = {}
    for name, (spec, feat) in candidate_specs().items():
        fitted[name] = (spec().fit(take(Z[tr], feat), Ydr[tr]), feat)

    # slide 73: illustration -- distil the random-forest DR-learner (train) to a tree
    illustration_tree(fitted["Random forest (mtry=p/3)"], Z, tr)

    # slide 74: model selection by DR loss on VALIDATION (delta-method s.e.)
    best = model_selection(fitted, Z, Ydr, tr, va)

    # slide 75: calibration + heterogeneity test on VALIDATION (all 4 learners)
    validation_calibration(fitted, Z, Ydr, va, best)

    # slide 76: TOC / RATE on VALIDATION (all 4; winner highlighted)
    validation_targeting(fitted, Z, Ydr, va, best)

    # slide 77: policy tree fit on TRAIN
    ptree = policy_tree(Z, Ydr, tr)

    # slide 78: policy value -- evaluate learned policies out-of-sample on TEST
    policy_value(fitted[best], ptree, Z, Ydr, tr, te)

    print(f"\nsaved figures -> {FIGDIR}/HTE_401k_*.png")


# --------------------------------------------------------------------------
# Slides 71-72: pre-specified summaries (full sample, 95%)
# --------------------------------------------------------------------------
def gates_by_income(inc, Ydr):
    q = pd.qcut(inc, 5, labels=False)
    G = np.column_stack([(q == k).astype(float) for k in range(5)])
    beta, cov = ols_blp(G, Ydr)
    se = np.sqrt(np.diag(cov))
    cunif = supt_crit(cov, alpha=0.05)
    qlab = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xpos = np.arange(5)
    ax.errorbar(xpos, beta, yerr=cunif * se, fmt="none", ecolor="0.55",
                elinewidth=7, alpha=0.5, capsize=0, label="uniform 95% band")
    ax.errorbar(xpos, beta, yerr=Z95 * se, fmt="o", color="C3", ms=6,
                elinewidth=2, capsize=4, label="pointwise 95% CI")
    ax.axhline(Ydr.mean(), ls="--", color="0.3", lw=1, label="ATE")
    ax.set_xticks(xpos); ax.set_xticklabels(qlab)
    ax.set_ylabel("GATE: effect on net financial assets ($)")
    ax.set_xlabel("Income quintile")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_gates.png"); plt.close(fig)
    tbl = pd.DataFrame({"income_quintile": [f"Q{k+1}" for k in range(5)],
                        "GATE": beta, "se": se})
    tbl.to_csv(NOTEDIR / "HTE_401k_gates.csv", index=False)
    print("\nGATES by income quintile (95%):\n", tbl.round(0).to_string(index=False))


def blp_cubic_income(inc, Ydr):
    loginc = np.log(np.clip(inc, 1, None))
    lc = (loginc - loginc.mean()) / loginc.std()
    P = np.column_stack([np.ones_like(lc), lc, lc**2, lc**3])
    beta3, cov3 = ols_blp(P, Ydr)

    grid_inc = np.linspace(np.quantile(inc, .02), np.quantile(inc, .98), 200)
    gl = (np.log(grid_inc) - loginc.mean()) / loginc.std()
    Pg = np.column_stack([np.ones_like(gl), gl, gl**2, gl**3])
    fit = Pg @ beta3
    seg = np.sqrt(np.einsum("gi,ij,gj->g", Pg, cov3, Pg))
    cunif3 = supt_crit(cov3, alpha=0.05)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(grid_inc / 1000, fit, color="C0", lw=2, label="CATE (cubic in log-income)")
    ax.fill_between(grid_inc / 1000, fit - cunif3 * seg, fit + cunif3 * seg,
                    color="C0", alpha=0.15, label="uniform 95% band")
    ax.plot(grid_inc / 1000, fit - Z95 * seg, color="C0", ls="--", lw=1)
    ax.plot(grid_inc / 1000, fit + Z95 * seg, color="C0", ls="--", lw=1,
            label="pointwise 95% band")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Income ($000)")
    ax.set_ylabel("CATE: effect on net financial assets ($)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_cate_income.png"); plt.close(fig)
    print(f"CATE(log-income) cubic BLP; uniform sup-t crit = {cunif3:.2f}")


# --------------------------------------------------------------------------
# Slide 73: illustration -- RF DR-learner distilled to a shallow tree (TRAIN)
# --------------------------------------------------------------------------
def illustration_tree(rf_fit, Z, tr):
    tau_tr = cate_predict(rf_fit, Z[tr])
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300,
                               random_state=SEED).fit(Z[tr], tau_tr)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    plot_tree(dt, feature_names=COVARS, filled=True, rounded=True, precision=0,
              impurity=False, fontsize=9, ax=ax)
    ax.set_title("Distillation tree for the random-forest DR-learner CATE "
                 "(leaf = GATE, $; fit on training data)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_distill_tree.png"); plt.close(fig)
    leaves = dt.apply(Z[tr])
    lo = tau_tr[leaves == leaves[np.argmin(tau_tr)]].mean()
    hi = tau_tr[leaves == leaves[np.argmax(tau_tr)]].mean()
    print(f"\nslide 73 distillation tree: leaf CATEs span ~{lo:,.0f} to ~{hi:,.0f}")


# --------------------------------------------------------------------------
# Slide 74: model selection by out-of-sample DR loss on VALIDATION.
#   normalized improvement over a constant model, R = mean(a)/mean(loss_c),
#   with a, loss_c per-obs; s.e. by the RATIO delta method (accounts for
#   denominator variability + numerator-denominator covariance).
# --------------------------------------------------------------------------
def model_selection(fitted, Z, Ydr, tr, va):
    const = Ydr[tr].mean()
    loss_c = (Ydr[va] - const) ** 2
    denom = loss_c.mean()
    rows = []
    for name in candidate_specs():
        tau = cate_predict(fitted[name], Z[va])
        loss_k = (Ydr[va] - tau) ** 2
        a = loss_c - loss_k                       # per-obs improvement over constant
        R = a.mean() / denom
        psi = (a - R * loss_c) / denom            # ratio delta-method influence
        se = psi.std(ddof=1) / np.sqrt(len(va))
        rows.append((name, R, se))
    tbl = pd.DataFrame(rows, columns=["model", "norm_score", "se"])
    best = tbl.loc[tbl["norm_score"].idxmax(), "model"]   # best (non-constant) model

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    yp = np.arange(len(tbl))[::-1]
    for y, (_, r) in zip(yp, tbl.iterrows()):
        ax.errorbar(r["norm_score"], y, xerr=Z68 * r["se"], fmt="o",
                    color=LEARNER_COLORS[r["model"]], ms=8, capsize=4, elinewidth=2)
    ax.axvline(0, color="0.5", ls="--", lw=1.2)
    ax.text(0, len(tbl) - 0.35, "constant effect ", color="0.4", fontsize=10,
            ha="right", va="top")
    ax.set_yticks(yp); ax.set_yticklabels(tbl["model"])
    ax.set_ylim(-0.6, len(tbl) - 0.15)
    ax.set_xlabel("Validation DR-loss improvement over constant "
                  "($>0$ is better; $\\pm$1 s.e.)")
    fig.set_size_inches(8.2, 4.2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "HTE_401k_model_comparison.png"); plt.close(fig)
    tbl["beats_constant_68"] = tbl["norm_score"] - Z68 * tbl["se"] > 0
    tbl.to_csv(NOTEDIR / "HTE_401k_model_comparison.csv", index=False)
    print("\nModel selection on validation (normalized DR-loss improvement, 68%):\n",
          tbl.round(3).to_string(index=False))
    print(f"  selected (best non-constant) model: {best}")
    return best


# --------------------------------------------------------------------------
# Slide 75: calibration + heterogeneity test on VALIDATION (all 4 learners).
#   het-test : OLS of Y(eta) on (1, tau_k - mean); slope>0 => heterogeneity,
#              slope=1 under perfect calibration (Chernozhukov et al. generic ML).
#   calibration: bin val by predicted-CATE quartile; the group's average Y(eta)
#                should line up with its mean predicted CATE (45-degree line).
# --------------------------------------------------------------------------
def validation_calibration(fitted, Z, Ydr, va, best):
    Yv = Ydr[va]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    het_rows = []
    allx, ally = [], []
    for name in candidate_specs():
        tau = cate_predict(fitted[name], Z[va])
        # (a) heterogeneity test
        tc = tau - tau.mean()
        if np.std(tc) < 1e-8:                 # degenerate (constant) predictor
            het_rows.append((name, Yv.mean(), 0.0, np.nan))
            continue                          # nothing to bin / plot
        beta, cov = ols_blp(np.column_stack([np.ones_like(tc), tc]), Yv)
        b1, se1 = beta[1], np.sqrt(cov[1, 1])
        het_rows.append((name, beta[0], b1, se1))
        # (b) calibration by predicted-CATE quartile
        qb = pd.qcut(tau, 4, labels=False, duplicates="drop")
        pred = np.array([tau[qb == k].mean() for k in np.unique(qb)])
        gate = np.array([Yv[qb == k].mean() for k in np.unique(qb)])
        gse = np.array([Yv[qb == k].std(ddof=1) / np.sqrt((qb == k).sum())
                        for k in np.unique(qb)])
        c = LEARNER_COLORS[name]
        full = (name == best)
        ax.errorbar(pred, gate, yerr=Z68 * gse, fmt="o-" if full else "o", color=c,
                    ms=7 if full else 5, lw=2.2 if full else 1.0,
                    alpha=1.0 if full else 0.5, capsize=3,
                    label=name + (" (selected)" if full else ""))
        allx += list(pred); ally += list(gate - Z68 * gse) + list(gate + Z68 * gse)

    # x (mean predicted CATE) and y (realized Y(eta)) live on different scales --
    # the learners predict modest CATEs but realized bin means run much larger --
    # so scale each axis to its own data instead of a single shared range (which
    # stretched x out to the y-extent and left the right half of the panel empty).
    xlo, xhi = min(allx), max(allx)
    ylo, yhi = min(ally), max(ally)
    xpad, ypad = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    lo = min(xlo - xpad, ylo - ypad)
    hi = max(xhi + xpad, yhi + ypad)
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1, label="perfect calibration")
    ax.set_xlim(xlo - xpad, xhi + xpad)   # set after plotting so the 45-degree
    ax.set_ylim(ylo - ypad, yhi + ypad)   # line is clipped to the data, not autoscaled to
    ax.set_xlabel("Mean predicted CATE in group ($)")
    ax.set_ylabel(r"Average $Y(\hat\eta)$ in CATE group (with 68% CI)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_calibration.png"); plt.close(fig)

    het = pd.DataFrame(het_rows, columns=["model", "ATE(intercept)", "slope", "se"])
    het["t"] = het["slope"] / het["se"]
    het.to_csv(NOTEDIR / "HTE_401k_het_test.csv", index=False)
    print("\nHeterogeneity test on validation (slope of Y(eta) on tau_k; 68%=+/-1se):\n",
          het.round(2).to_string(index=False))


# --------------------------------------------------------------------------
# Slide 76: TOC / RATE on VALIDATION. Prioritize val units by tau_k; how much
#   better does the top-q group do than average?  Winner full opacity + its
#   one-sided 68% uniform band; others faded. AUTOC (+/-1 s.e.) for each.
# --------------------------------------------------------------------------
def validation_targeting(fitted, Z, Ydr, va, best):
    Yv = Ydr[va]; n = len(va)
    theta0 = Yv.mean()
    qgrid = np.linspace(0.05, 0.95, 50); G = len(qgrid); dq = qgrid[1] - qgrid[0]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    autoc_rows = []
    for name in candidate_specs():
        tau = cate_predict(fitted[name], Z[va])
        toc = np.zeros(G); P = np.zeros((n, G))
        for j, q in enumerate(qgrid):
            thr = np.quantile(tau, 1 - q)
            I = (tau >= thr).astype(float); pi = I.mean()
            toc[j] = (Yv * I).sum() / I.sum() - theta0
            P[:, j] = (Yv - theta0) * (I / pi - 1)         # TOC influence
        V = P.T @ P / n
        autoc = (toc * dq).sum()
        se_autoc = (P @ np.full(G, dq)).std(ddof=1) / np.sqrt(n)
        autoc_rows.append((name, autoc, se_autoc))
        c = LEARNER_COLORS[name]; full = (name == best)
        ax.plot(qgrid, toc, color=c, lw=2.6 if full else 1.4,
                alpha=1.0 if full else 0.4,
                label=f"{name}: AUTOC {autoc:,.0f} ($\\pm${se_autoc:,.0f})"
                      + (" [selected]" if full else ""))
        if full:
            cc = supt_crit(V, alpha=0.32, one_sided=True)   # one-sided 68% uniform
            se_t = np.sqrt(np.diag(V) / n)
            ax.plot(qgrid, toc - cc * se_t, color=c, lw=1, ls="--",
                    label="one-sided 68% uniform band (selected)")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xlabel("Fraction treated (prioritized by $\\hat\\tau$)")
    ax.set_ylabel("TOC: GATE(top q) $-$ ATE ($)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_toc.png"); plt.close(fig)

    at = pd.DataFrame(autoc_rows, columns=["model", "AUTOC", "se"])
    at["lo_68_1sided"] = at["AUTOC"] - Z68_1S * at["se"]
    at.to_csv(NOTEDIR / "HTE_401k_autoc.csv", index=False)
    print("\nRATE / AUTOC on validation (one-sided 68% lower bound):\n",
          at.round(0).to_string(index=False))


# --------------------------------------------------------------------------
# Slide 77: policy tree fit on TRAIN. Optimal policy treats where CATE>0, so
#   learning it is cost-sensitive classification: label sign(Y(eta)),
#   weight |Y(eta)|. A shallow tree gives an interpretable rule.
# --------------------------------------------------------------------------
def policy_tree(Z, Ydr, tr):
    ylab = (Ydr[tr] > 0).astype(int)
    w = np.abs(Ydr[tr])
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=300,
                                 random_state=SEED).fit(Z[tr], ylab, sample_weight=w)
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(clf, feature_names=COVARS, filled=True, rounded=True,
              impurity=False, precision=0, fontsize=9, ax=ax,
              class_names=["don't treat", "treat"])
    ax.set_title("Policy tree (fit on training data): maximize "
                 "V(pi)=E[pi(X) Y(eta)]   (node value=[neg,pos] |Y(eta)|)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(FIGDIR / "HTE_401k_policy_tree.png"); plt.close(fig)
    pol = clf.predict(Z[tr])
    print(f"\nslide 77 policy tree: treats {pol.mean():.0%} of training units")
    return clf


def budget_policy_from_tree(clf, Z, Ydr, tr, te, budget=0.5, seed=SEED):
    """Impose a budget on the policy TREE: rank its leaves by training GATE and
    treat the highest-GATE leaves on TEST until the budget is filled,
    randomizing within the marginal (budget-crossing) leaf."""
    leaves_tr = clf.apply(Z[tr])
    gate = {lf: Ydr[tr][leaves_tr == lf].mean() for lf in np.unique(leaves_tr)}
    leaves_te = clf.apply(Z[te])
    n = len(te); pi = np.zeros(n); filled = 0.0
    rng = np.random.default_rng(seed + 5)
    for lf in sorted(gate, key=gate.get, reverse=True):
        if gate[lf] <= 0:
            continue
        mask = leaves_te == lf; frac = mask.mean()
        if filled + frac <= budget:
            pi[mask] = 1.0; filled += frac
        else:
            rem = budget - filled
            if rem <= 0:
                break
            idx = np.where(mask)[0]
            k = int(round((rem / frac) * len(idx)))
            pi[rng.choice(idx, k, replace=False)] = 1.0
            break
    return pi


# --------------------------------------------------------------------------
# Slide 78: policy VALUE + inference, evaluated out-of-sample on TEST.
#   V(pi) = E[pi(X) Y(eta)] (per-capita gain over treating no one); s.e. is the
#   s.e. of an average -- treating the policy as FIXED (learned on train), which
#   is the relevant uncertainty for rolling a chosen policy out. Compares the
#   selected CATE model and the policy tree, each with and without a 50% budget.
# --------------------------------------------------------------------------
def policy_value(best_fit, ptree, Z, Ydr, tr, te):
    Yt = Ydr[te]; n = len(te)
    tau_te = cate_predict(best_fit, Z[te])
    thr50 = np.median(tau_te)                    # 50% budget cutoff on the test pop
    rng = np.random.default_rng(SEED + 2)

    pol = {
        "Treat all": np.ones(n),
        "Boosting: CATE>0": (tau_te > 0).astype(float),
        "Boosting: top 50%": (tau_te >= thr50).astype(float),
        "Policy tree": ptree.predict(Z[te]).astype(float),
        "Policy tree: 50% budget": budget_policy_from_tree(ptree, Z, Ydr, tr, te),
        "Random 50%": (rng.random(n) < 0.5).astype(float),
    }

    def value(pi):
        v = pi * Yt
        return v.mean(), v.std(ddof=1) / np.sqrt(n)

    rows = [("Treat none", 0.0, 0.0, 0.0)]
    for name, pi in pol.items():
        m, se = value(pi)
        rows.append((name, pi.mean(), m, se))
    tbl = pd.DataFrame(rows, columns=["policy", "treated_frac", "value_gain", "se"])

    def contrast(pa, pb):
        d = (pol[pa] - pol[pb]) * Yt
        return d.mean(), d.std(ddof=1) / np.sqrt(n)

    show = tbl[tbl.policy != "Treat none"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    yp = np.arange(len(show))[::-1]
    ax.errorbar(show["value_gain"], yp, xerr=Z68 * show["se"], fmt="o",
                color="C0", ms=8, capsize=4, elinewidth=2)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(yp)
    ax.set_yticklabels([f"{p}  ({f:.0%})" for p, f in
                        zip(show["policy"], show["treated_frac"])])
    ax.set_xlabel("Policy value on test: gain over treating no one "
                  "(\\$, $\\pm$1 s.e.)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "HTE_401k_policy_value.png"); plt.close(fig)

    tbl.to_csv(NOTEDIR / "HTE_401k_policy_value.csv", index=False)
    disp = tbl.copy()
    disp["treated_frac"] = (disp["treated_frac"] * 100).round(0).astype(int).astype(str) + "%"
    disp["value_gain"] = disp["value_gain"].round(0).astype(int)
    disp["se"] = disp["se"].round(0).astype(int)
    print("\nPolicy value on TEST (gain over treating no one, 68%=+/-1se):\n",
          disp.to_string(index=False))
    cs, scs = contrast("Boosting: top 50%", "Random 50%")
    cp, scp = contrast("Policy tree: 50% budget", "Random 50%")
    cu, scu = contrast("Boosting: CATE>0", "Treat all")
    print(f"  Boosting top-50% vs random-50%: {cs:,.0f} (se {scs:,.0f})")
    print(f"  policy-tree 50% budget vs random-50%: {cp:,.0f} (se {scp:,.0f})")
    print(f"  Boostings CATE>0 vs treat-all: {cu:,.0f} (se {scu:,.0f})")


if __name__ == "__main__":
    main()
