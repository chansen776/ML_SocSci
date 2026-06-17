"""401(k) eligibility ATE -- Python port of Stata Code/401kATE.do.

Cross-fit DML estimation of E[Y(1) - Y(0)] where Y = net_tfa, D = e401.
Mirrors the Stata pipeline: 6 Y-learners x 2 D-arms, 6 D-learners,
5-fold cross-validation, 10 resamples, propensity trimming at [.01, .99].

Outputs:
    - Per-rep best-learner (min-MSE) and shortstack point estimates + SEs.
    - Cross-fitted predictions saved to Data/401kATE_python.parquet
      (column convention parallel to Data/401kATE.dta so the diagnostics
       script can be re-pointed at this file).

Reps run in parallel via joblib.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    LassoCV, RidgeCV, LogisticRegressionCV,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "sipp1991.dta"
OUT_FILE = ROOT / "Data" / "401kATE_python.parquet"

N_FOLDS = 5
N_REPS = 10
SEED_BASE = 71423   # matches Stata `set seed 71423`
TRIM = (0.01, 0.99)
N_PARALLEL_REPS = 10  # joblib n_jobs over reps

Y_LEARNER_NAMES = ["Y1_reg", "Y2_reg",
                   "Y3_pystacked", "Y4_pystacked",
                   "Y5_pystacked", "Y6_pystacked"]
D_LEARNER_NAMES = ["D1_logit", "D2_pystacked",
                   "D3_pystacked", "D4_pystacked",
                   "D5_pystacked", "D6_pystacked"]
# Each learner uses one of three feature sets:
#   raw  = 9 raw covariates (matches Stata $X)
#   poly = polynomial expansion (matches Stata $Xpoly, 25 features)
#   inter = full pairwise interactions of raw X (matches `c.($X)##c.($X)`,
#           54 features: 9 main + 9 squares + 36 pairwise)
Y_X_KIND = ["raw", "poly", "inter", "inter", "raw", "raw"]
D_X_KIND = ["raw", "poly", "inter", "inter", "raw", "raw"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data():
    """Load sipp1991.dta, replicate the Stata data preparation."""
    df = pd.read_stata(DATA_FILE)
    df = df.copy()
    df["age"]   = df["age"]   / 64.0
    df["inc"]   = df["inc"]   / 250000.0
    df["educ"] = df["educ"] / 18.0
    df["fsize"] = df["fsize"] / 13.0
    for i in range(2, 7):
        df[f"age{i}"] = df["age"] ** i
    for i in range(2, 9):
        df[f"inc{i}"] = df["inc"] ** i
    for i in range(2, 5):
        df[f"educ{i}"] = df["educ"] ** i
    df["fsize2"] = df["fsize"] ** 2

    raw_vars = ["age", "inc", "educ", "fsize",
                "marr", "twoearn", "db", "pira", "hown"]
    poly_vars = (["age"] + [f"age{i}" for i in range(2, 7)]
                 + ["inc"] + [f"inc{i}" for i in range(2, 9)]
                 + ["educ"] + [f"educ{i}" for i in range(2, 5)]
                 + ["fsize", "fsize2",
                    "marr", "twoearn", "db", "pira", "hown"])
    X_raw = df[raw_vars].to_numpy(dtype=np.float64)
    X_poly = df[poly_vars].to_numpy(dtype=np.float64)
    pf = PolynomialFeatures(degree=2, interaction_only=False,
                            include_bias=False)
    X_inter = pf.fit_transform(X_raw)
    Y = df["net_tfa"].to_numpy(dtype=np.float64)
    D = df["e401"].to_numpy(dtype=np.int64)
    return df, X_raw, X_poly, X_inter, Y, D


# ---------------------------------------------------------------------------
# Learner factories (built fresh per fold; no shared state across workers)
# ---------------------------------------------------------------------------

def build_y_learner(idx):
    """Return a fresh sklearn Y-learner of index idx (0..5)."""
    if idx == 0:
        return LinearRegression()
    if idx == 1:
        return LinearRegression()
    if idx == 2:
        return LassoCV(cv=5, n_alphas=100, max_iter=10000, n_jobs=1)
    if idx == 3:
        return RidgeCV(cv=5, alphas=np.logspace(-3, 3, 100))
    if idx == 4:
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=20,
            random_state=720, n_jobs=1)
    if idx == 5:
        # Scale both inputs and target so the MLP optimizer doesn't have to
        # chase Y values up to $1.5M; matches what pystacked does internally.
        nn = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(50, 50, 50, 50), alpha=0,
                max_iter=200, random_state=720)),
        ])
        return TransformedTargetRegressor(regressor=nn, transformer=StandardScaler())
    raise ValueError(idx)


def build_d_learner(idx):
    """Return a fresh sklearn D-learner of index idx (0..5)."""
    if idx == 0:
        return LogisticRegression(max_iter=1000)
    if idx == 1:
        return LogisticRegression(max_iter=1000)
    if idx == 2:
        # L1 binary CV via liblinear (much faster than saga on this size).
        return LogisticRegressionCV(
            penalty="l1", solver="liblinear", cv=5, Cs=10,
            max_iter=500, scoring="neg_log_loss", n_jobs=1)
    if idx == 3:
        return LogisticRegressionCV(
            penalty="l2", solver="lbfgs", cv=5, Cs=10,
            max_iter=500, scoring="neg_log_loss", n_jobs=1)
    if idx == 4:
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20,
            random_state=720, n_jobs=1)
    if idx == 5:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(50, 50, 50, 50), alpha=0,
                max_iter=200, random_state=720)),
        ])
    raise ValueError(idx)


# ---------------------------------------------------------------------------
# Cross-fit one rep
# ---------------------------------------------------------------------------

def cross_fit_one_rep(rep, X_raw, X_poly, X_inter, Y, D):
    """Cross-fit all 6 Y learners x 2 arms and 6 D learners for one rep.

    Returns dict with:
        pred_y[(i, arm)] : array (N,) of cross-fit Y predictions
        pred_d[i]        : array (N,) of cross-fit D propensities
        fold_id          : array (N,) of fold IDs (0..N_FOLDS-1)
    """
    n = len(Y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True,
               random_state=SEED_BASE + rep)
    fold_id = np.zeros(n, dtype=np.int64)
    for k, (_, test_idx) in enumerate(kf.split(np.arange(n))):
        fold_id[test_idx] = k

    Xs = {"raw": X_raw, "poly": X_poly, "inter": X_inter}
    pred_y = {(i, arm): np.full(n, np.nan)
              for i in range(6) for arm in (0, 1)}
    pred_d = {i: np.full(n, np.nan) for i in range(6)}

    t0 = time.time()
    for k, (tr, te) in enumerate(kf.split(np.arange(n))):
        Y_tr, D_tr = Y[tr], D[tr]
        # Y learners
        for i in range(6):
            Xkind = Y_X_KIND[i]
            X_tr_y = Xs[Xkind][tr]
            X_te_y = Xs[Xkind][te]
            for arm in (0, 1):
                mask = D_tr == arm
                m = build_y_learner(i)
                m.fit(X_tr_y[mask], Y_tr[mask])
                pred_y[(i, arm)][te] = m.predict(X_te_y)
        # D learners
        for i in range(6):
            Xkind = D_X_KIND[i]
            X_tr_d = Xs[Xkind][tr]
            X_te_d = Xs[Xkind][te]
            m = build_d_learner(i)
            m.fit(X_tr_d, D_tr)
            pred_d[i][te] = m.predict_proba(X_te_d)[:, 1]
    elapsed = time.time() - t0
    print(f"  rep {rep+1:2d}: {elapsed:6.1f}s", flush=True)

    return rep, fold_id, pred_y, pred_d


# ---------------------------------------------------------------------------
# Estimation: AIPW score, shortstack, best, allcombos
# ---------------------------------------------------------------------------

def aipw_score(Y, D, m0, m1, p):
    p = np.clip(p, TRIM[0], TRIM[1])
    return (m1 - m0
            + D * (Y - m1) / p
            - (1 - D) * (Y - m0) / (1 - p))


def aipw_estimate(Y, D, m0, m1, p):
    psi = aipw_score(Y, D, m0, m1, p)
    n = len(psi)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(n)


def shortstack_weights_y(Y, D, arm, fold_id, pred_y_for_rep):
    """NNLS combination on D==arm subset, normalized to sum 1."""
    mask = D == arm
    P = np.column_stack([pred_y_for_rep[(i, arm)][mask] for i in range(6)])
    yobs = Y[mask]
    w, _ = nnls(P, yobs)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.full(6, 1.0 / 6.0)
    return w


def shortstack_weights_d(D, pred_d_for_rep):
    """NNLS combination of D-learner predictions, normalized to sum 1."""
    P = np.column_stack([pred_d_for_rep[i] for i in range(6)])
    w, _ = nnls(P, D.astype(float))
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.full(6, 1.0 / 6.0)
    return w


def shortstack_predictions(D, pred_y_for_rep, pred_d_for_rep, fold_id, Y):
    """Combine the 6 per-learner predictions into a single shortstacked
    prediction per moment via NNLS weights fit on out-of-fold predictions."""
    w_y0 = shortstack_weights_y(Y, D, 0, fold_id, pred_y_for_rep)
    w_y1 = shortstack_weights_y(Y, D, 1, fold_id, pred_y_for_rep)
    w_d  = shortstack_weights_d(D, pred_d_for_rep)
    P_y0 = np.column_stack([pred_y_for_rep[(i, 0)] for i in range(6)])
    P_y1 = np.column_stack([pred_y_for_rep[(i, 1)] for i in range(6)])
    P_d  = np.column_stack([pred_d_for_rep[i]      for i in range(6)])
    m0_ss = P_y0 @ w_y0
    m1_ss = P_y1 @ w_y1
    p_ss  = P_d  @ w_d
    return m0_ss, m1_ss, p_ss, w_y0, w_y1, w_d


def best_learner_picks_for_rep(D, pred_y_for_rep, pred_d_for_rep, Y):
    """Per-rep min-MSE pick for each of the three moments."""
    mse_y0 = [np.mean((Y[D == 0] - pred_y_for_rep[(i, 0)][D == 0]) ** 2)
              for i in range(6)]
    mse_y1 = [np.mean((Y[D == 1] - pred_y_for_rep[(i, 1)][D == 1]) ** 2)
              for i in range(6)]
    mse_d = [np.mean((D - pred_d_for_rep[i]) ** 2)
             for i in range(6)]
    return int(np.argmin(mse_y0)), int(np.argmin(mse_y1)), int(np.argmin(mse_d))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("401(k) ATE -- Python port of Stata Code/401kATE.do")
    print("=" * 70)
    df, X_raw, X_poly, X_inter, Y, D = prepare_data()
    print(f"loaded sipp1991.dta: n = {len(Y)}, D mean = {D.mean():.4f}")

    print(f"\ncross-fitting {N_REPS} reps x {N_FOLDS} folds x 6+6 learners "
          f"in parallel (n_jobs={N_PARALLEL_REPS})...")
    t0 = time.time()
    results = Parallel(n_jobs=N_PARALLEL_REPS, verbose=0)(
        delayed(cross_fit_one_rep)(r, X_raw, X_poly, X_inter, Y, D)
        for r in range(N_REPS)
    )
    print(f"cross-fit total wall: {time.time() - t0:.1f}s")

    # ---- Summary tables ----
    rows = []
    print("\nPer-rep estimates (parallels Stata `ddml estimate, robust`):")
    print(f"  {'rep':>3}  "
          f"{'best b':>10} {'best SE':>9}  "
          f"{'stack b':>10} {'stack SE':>9}  "
          f"{'best learners (Y0, Y1, D)':>40}")
    for rep, fold_id, pred_y, pred_d in sorted(results):
        # best
        bi_y0, bi_y1, bi_d = best_learner_picks_for_rep(D, pred_y, pred_d, Y)
        b_best, se_best = aipw_estimate(
            Y, D, pred_y[(bi_y0, 0)], pred_y[(bi_y1, 1)], pred_d[bi_d])
        # shortstack
        m0_ss, m1_ss, p_ss, _, _, _ = shortstack_predictions(
            D, pred_y, pred_d, fold_id, Y)
        b_ss, se_ss = aipw_estimate(Y, D, m0_ss, m1_ss, p_ss)
        rows.append({
            "rep": rep + 1,
            "best_b": b_best, "best_se": se_best,
            "stack_b": b_ss, "stack_se": se_ss,
            "best_y0": Y_LEARNER_NAMES[bi_y0],
            "best_y1": Y_LEARNER_NAMES[bi_y1],
            "best_d":  D_LEARNER_NAMES[bi_d],
        })
        print(f"  {rep+1:3d}  "
              f"{b_best:10.1f} {se_best:9.1f}  "
              f"{b_ss:10.1f} {se_ss:9.1f}  "
              f"{Y_LEARNER_NAMES[bi_y0]:14s} "
              f"{Y_LEARNER_NAMES[bi_y1]:14s} "
              f"{D_LEARNER_NAMES[bi_d]:14s}")

    summary = pd.DataFrame(rows)
    print(f"\n  mean   {summary.best_b.mean():10.1f} {summary.best_se.mean():9.1f}  "
          f"{summary.stack_b.mean():10.1f} {summary.stack_se.mean():9.1f}")
    print(f"  median {summary.best_b.median():10.1f} {summary.best_se.median():9.1f}  "
          f"{summary.stack_b.median():10.1f} {summary.stack_se.median():9.1f}")

    # ---- Save predictions in .dta-compatible column layout ----
    out = pd.DataFrame()
    # raw passthrough
    for c in ["nifa", "net_tfa", "tw", "age", "inc", "fsize", "educ",
              "db", "marr", "twoearn", "e401", "p401", "pira", "hown"]:
        out[c] = df[c].values
    for c in ["age2", "age3", "age4", "age5", "age6",
              "inc2", "inc3", "inc4", "inc5", "inc6", "inc7", "inc8",
              "educ2", "educ3", "educ4", "fsize2"]:
        out[c] = df[c].values
    for rep, fold_id, pred_y, pred_d in sorted(results):
        out[f"m0_fid_{rep+1}"] = fold_id + 1  # match Stata's 1..5 convention
        # Y predictions: <name><arm>_<rep+1>
        for i, name in enumerate(Y_LEARNER_NAMES):
            for arm in (0, 1):
                out[f"{name}{arm}_{rep+1}"] = pred_y[(i, arm)]
        # Y shortstack
        m0_ss, m1_ss, p_ss, *_ = shortstack_predictions(
            D, pred_y, pred_d, fold_id, Y)
        out[f"Y_net_tfa_ss0_{rep+1}"] = m0_ss
        out[f"Y_net_tfa_ss1_{rep+1}"] = m1_ss
        # D predictions: <name>_<rep+1>
        for i, name in enumerate(D_LEARNER_NAMES):
            out[f"{name}_{rep+1}"] = pred_d[i]
        out[f"D_e401_ss_{rep+1}"] = p_ss
    out.to_parquet(OUT_FILE, index=False)
    print(f"\nsaved cross-fit predictions to {OUT_FILE}")
    print(f"  ({out.shape[0]} obs x {out.shape[1]} vars)")

    summary.to_csv(ROOT / "notes" / "401kATE_python_summary.csv", index=False)
    print(f"saved summary table to notes/401kATE_python_summary.csv")


if __name__ == "__main__":
    main()
