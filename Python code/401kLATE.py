"""401(k) participation LATE -- Python port of Stata Code/401kLATE.do.

Cross-fit DML estimation of LATE = E[Y(1)-Y(0) | complier] using e401 as
an instrument for p401. Mirrors the Stata interactive-IV pipeline:
  - Y = net_tfa, D = p401, Z = e401
  - 5 nuisance moments: E[Y|X,Z=0], E[Y|X,Z=1], E[D|X,Z=0], E[D|X,Z=1], E[Z|X]
  - 6 learners per moment; 5-fold CV; 10 resamples
  - Propensity trimming at [.01, .99]

Outputs:
    - Per-rep best (min-MSE) and shortstack point estimates + SEs.
    - Cross-fitted predictions saved to Data/401kLATE_python.parquet
      (column convention parallel to Data/401kLATE.dta).

LATE estimator (interactive IV / DR-IV):
    psi_Y_i = m_Y1(X_i) - m_Y0(X_i)
              + Z_i (Y_i - m_Y1(X_i)) / p_Z(X_i)
              - (1 - Z_i) (Y_i - m_Y0(X_i)) / (1 - p_Z(X_i))
    psi_D_i = analogous with D in place of Y
    beta = mean(psi_Y) / mean(psi_D)
    SE   = sd(IF) / sqrt(n) where IF_i = (psi_Y_i - beta psi_D_i) / mean(psi_D)
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
OUT_FILE = ROOT / "Data" / "401kLATE_python.parquet"

N_FOLDS = 5
N_REPS = 10
SEED_BASE = 71423
TRIM = (0.01, 0.99)
N_PARALLEL_REPS = 10

Y_LEARNER_NAMES = ["Y1_reg", "Y2_reg",
                   "Y3_pystacked", "Y4_pystacked",
                   "Y5_pystacked", "Y6_pystacked"]
D_LEARNER_NAMES = ["D1_logit", "D2_pystacked",
                   "D3_pystacked", "D4_pystacked",
                   "D5_pystacked", "D6_pystacked"]
Z_LEARNER_NAMES = ["Z1_logit", "Z2_pystacked",
                   "Z3_pystacked", "Z4_pystacked",
                   "Z5_pystacked", "Z6_pystacked"]

# Feature-set assignment (matches Stata pipeline; same logic for Y, D, Z).
LEARNER_X_KIND = ["raw", "poly", "inter", "inter", "raw", "raw"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data():
    df = pd.read_stata(DATA_FILE).copy()
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
    Dval = df["p401"].to_numpy(dtype=np.int64)
    Zval = df["e401"].to_numpy(dtype=np.int64)
    return df, X_raw, X_poly, X_inter, Y, Dval, Zval


# ---------------------------------------------------------------------------
# Learner factories (top-level for joblib pickling)
# ---------------------------------------------------------------------------

def build_reg_learner(idx):
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
        nn = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(50, 50, 50, 50), alpha=0,
                max_iter=200, random_state=720)),
        ])
        return TransformedTargetRegressor(regressor=nn, transformer=StandardScaler())
    raise ValueError(idx)


def build_clf_learner(idx):
    if idx == 0:
        return LogisticRegression(max_iter=1000)
    if idx == 1:
        return LogisticRegression(max_iter=1000)
    if idx == 2:
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

def cross_fit_one_rep(rep, X_raw, X_poly, X_inter, Y, Dval, Zval):
    """Cross-fit all 5 nuisance moments x 6 learners for one rep."""
    n = len(Y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True,
               random_state=SEED_BASE + rep)
    fold_id = np.zeros(n, dtype=np.int64)
    for k, (_, te) in enumerate(kf.split(np.arange(n))):
        fold_id[te] = k

    Xs = {"raw": X_raw, "poly": X_poly, "inter": X_inter}
    pred_y = {(i, arm): np.full(n, np.nan)
              for i in range(6) for arm in (0, 1)}
    pred_d = {(i, arm): np.full(n, np.nan)
              for i in range(6) for arm in (0, 1)}
    pred_z = {i: np.full(n, np.nan) for i in range(6)}

    t0 = time.time()
    for k, (tr, te) in enumerate(kf.split(np.arange(n))):
        Y_tr, D_tr, Z_tr = Y[tr], Dval[tr], Zval[tr]
        for i in range(6):
            Xkind = LEARNER_X_KIND[i]
            X_tr_full = Xs[Xkind][tr]
            X_te_full = Xs[Xkind][te]
            # Y conditional on X, Z=arm
            for arm in (0, 1):
                mask = Z_tr == arm
                m = build_reg_learner(i)
                m.fit(X_tr_full[mask], Y_tr[mask])
                pred_y[(i, arm)][te] = m.predict(X_te_full)
            # D conditional on X, Z=arm
            #   For Z=0 in 401(k), D == 0 always (no defiers); fitting a
            #   classifier on a degenerate target trips sklearn. Predict 0.
            for arm in (0, 1):
                mask = Z_tr == arm
                D_arm = D_tr[mask]
                if len(np.unique(D_arm)) < 2:
                    pred_d[(i, arm)][te] = float(D_arm.mean())
                else:
                    m = build_clf_learner(i)
                    m.fit(X_tr_full[mask], D_arm)
                    pred_d[(i, arm)][te] = m.predict_proba(X_te_full)[:, 1]
            # Z propensity (full sample)
            m = build_clf_learner(i)
            m.fit(X_tr_full, Z_tr)
            pred_z[i][te] = m.predict_proba(X_te_full)[:, 1]
    elapsed = time.time() - t0
    print(f"  rep {rep+1:2d}: {elapsed:6.1f}s", flush=True)

    return rep, fold_id, pred_y, pred_d, pred_z


# ---------------------------------------------------------------------------
# Estimation: LATE score, shortstack, best
# ---------------------------------------------------------------------------

def late_score(Y, Dval, Zval, mY0, mY1, mD0, mD1, pZ):
    pZ = np.clip(pZ, TRIM[0], TRIM[1])
    psi_Y = (mY1 - mY0
             + Zval * (Y - mY1) / pZ
             - (1 - Zval) * (Y - mY0) / (1 - pZ))
    psi_D = (mD1 - mD0
             + Zval * (Dval - mD1) / pZ
             - (1 - Zval) * (Dval - mD0) / (1 - pZ))
    return psi_Y, psi_D


def late_estimate(psi_Y, psi_D):
    EY, ED = psi_Y.mean(), psi_D.mean()
    beta = EY / ED
    n = len(psi_Y)
    IF = (psi_Y - beta * psi_D) / ED
    return beta, IF.std(ddof=1) / np.sqrt(n)


def shortstack_weights_from_obs(target_obs, P):
    w, _ = nnls(P, target_obs)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.full(P.shape[1], 1.0 / P.shape[1])
    return w


def shortstack_predictions(Y, Dval, Zval, pred_y, pred_d, pred_z):
    """Shortstack on each of the 5 moments via NNLS on the cross-fit
    predictions. For E[D|X,Z=0] D is identically 0 -- predictions are 0
    and weights default to a flat blend."""
    n = len(Y)
    # Y arms
    m0_y = np.column_stack([pred_y[(i, 0)][Zval == 0] for i in range(6)])
    m1_y = np.column_stack([pred_y[(i, 1)][Zval == 1] for i in range(6)])
    w_y0 = shortstack_weights_from_obs(Y[Zval == 0], m0_y)
    w_y1 = shortstack_weights_from_obs(Y[Zval == 1], m1_y)
    P_y0 = np.column_stack([pred_y[(i, 0)] for i in range(6)])
    P_y1 = np.column_stack([pred_y[(i, 1)] for i in range(6)])
    mY0_ss = P_y0 @ w_y0
    mY1_ss = P_y1 @ w_y1

    # D arms (D|Z=0 trivial; D|Z=1 fit on Z=1 obs)
    P_d0 = np.column_stack([pred_d[(i, 0)] for i in range(6)])
    P_d1 = np.column_stack([pred_d[(i, 1)] for i in range(6)])
    mD0_ss = P_d0.mean(axis=1)  # all 0 anyway
    m1_d = np.column_stack([pred_d[(i, 1)][Zval == 1] for i in range(6)])
    w_d1 = shortstack_weights_from_obs(Dval[Zval == 1].astype(float), m1_d)
    mD1_ss = P_d1 @ w_d1

    # Z propensity
    P_z = np.column_stack([pred_z[i] for i in range(6)])
    w_z = shortstack_weights_from_obs(Zval.astype(float), P_z)
    pZ_ss = P_z @ w_z

    return mY0_ss, mY1_ss, mD0_ss, mD1_ss, pZ_ss


def best_picks(Y, Dval, Zval, pred_y, pred_d, pred_z):
    mse_y0 = [np.mean((Y[Zval == 0] - pred_y[(i, 0)][Zval == 0]) ** 2)
              for i in range(6)]
    mse_y1 = [np.mean((Y[Zval == 1] - pred_y[(i, 1)][Zval == 1]) ** 2)
              for i in range(6)]
    # D|Z=0 degenerate; pick arbitrarily (won't affect score since D=0 fixed)
    mse_d1 = [np.mean((Dval[Zval == 1] - pred_d[(i, 1)][Zval == 1]) ** 2)
              for i in range(6)]
    mse_z = [np.mean((Zval - pred_z[i]) ** 2) for i in range(6)]
    return (int(np.argmin(mse_y0)),
            int(np.argmin(mse_y1)),
            0,  # D|Z=0
            int(np.argmin(mse_d1)),
            int(np.argmin(mse_z)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("401(k) LATE -- Python port of Stata Code/401kLATE.do")
    print("=" * 70)
    df, X_raw, X_poly, X_inter, Y, Dval, Zval = prepare_data()
    print(f"loaded sipp1991.dta: n = {len(Y)}, "
          f"D mean = {Dval.mean():.4f}, Z mean = {Zval.mean():.4f}")

    print(f"\ncross-fitting {N_REPS} reps x {N_FOLDS} folds x 5 moments x 6 "
          f"learners in parallel (n_jobs={N_PARALLEL_REPS})...")
    t0 = time.time()
    results = Parallel(n_jobs=N_PARALLEL_REPS, verbose=0)(
        delayed(cross_fit_one_rep)(r, X_raw, X_poly, X_inter, Y, Dval, Zval)
        for r in range(N_REPS)
    )
    print(f"cross-fit total wall: {time.time() - t0:.1f}s")

    # ---- Summary tables ----
    rows = []
    print("\nPer-rep estimates (parallels Stata `ddml estimate, robust`):")
    print(f"  {'rep':>3}  "
          f"{'best b':>10} {'best SE':>9}  "
          f"{'stack b':>10} {'stack SE':>9}  "
          f"{'best learners (Y0, Y1, D1, Z)':>40}")
    for rep, fold_id, pred_y, pred_d, pred_z in sorted(results):
        bi_y0, bi_y1, _, bi_d1, bi_z = best_picks(
            Y, Dval, Zval, pred_y, pred_d, pred_z)
        psi_Y, psi_D = late_score(
            Y, Dval, Zval,
            pred_y[(bi_y0, 0)], pred_y[(bi_y1, 1)],
            pred_d[(0, 0)],  # any (D|Z=0 ≡ 0)
            pred_d[(bi_d1, 1)],
            pred_z[bi_z])
        b_best, se_best = late_estimate(psi_Y, psi_D)
        mY0, mY1, mD0, mD1, pZ = shortstack_predictions(
            Y, Dval, Zval, pred_y, pred_d, pred_z)
        psi_Y, psi_D = late_score(Y, Dval, Zval, mY0, mY1, mD0, mD1, pZ)
        b_ss, se_ss = late_estimate(psi_Y, psi_D)
        rows.append({
            "rep": rep + 1,
            "best_b": b_best, "best_se": se_best,
            "stack_b": b_ss, "stack_se": se_ss,
            "best_y0": Y_LEARNER_NAMES[bi_y0],
            "best_y1": Y_LEARNER_NAMES[bi_y1],
            "best_d1": D_LEARNER_NAMES[bi_d1],
            "best_z":  Z_LEARNER_NAMES[bi_z],
        })
        print(f"  {rep+1:3d}  "
              f"{b_best:10.1f} {se_best:9.1f}  "
              f"{b_ss:10.1f} {se_ss:9.1f}  "
              f"{Y_LEARNER_NAMES[bi_y0]:14s} "
              f"{Y_LEARNER_NAMES[bi_y1]:14s} "
              f"{D_LEARNER_NAMES[bi_d1]:14s} "
              f"{Z_LEARNER_NAMES[bi_z]:14s}")

    summary = pd.DataFrame(rows)
    print(f"\n  mean   {summary.best_b.mean():10.1f} {summary.best_se.mean():9.1f}  "
          f"{summary.stack_b.mean():10.1f} {summary.stack_se.mean():9.1f}")
    print(f"  median {summary.best_b.median():10.1f} {summary.best_se.median():9.1f}  "
          f"{summary.stack_b.median():10.1f} {summary.stack_se.median():9.1f}")

    # ---- Save predictions ----
    out = pd.DataFrame()
    for c in ["nifa", "net_tfa", "tw", "age", "inc", "fsize", "educ",
              "db", "marr", "twoearn", "e401", "p401", "pira", "hown"]:
        out[c] = df[c].values
    for c in ["age2", "age3", "age4", "age5", "age6",
              "inc2", "inc3", "inc4", "inc5", "inc6", "inc7", "inc8",
              "educ2", "educ3", "educ4", "fsize2"]:
        out[c] = df[c].values
    for rep, fold_id, pred_y, pred_d, pred_z in sorted(results):
        out[f"m0_fid_{rep+1}"] = fold_id + 1
        for i, name in enumerate(Y_LEARNER_NAMES):
            for arm in (0, 1):
                out[f"{name}{arm}_{rep+1}"] = pred_y[(i, arm)]
        for i, name in enumerate(D_LEARNER_NAMES):
            for arm in (0, 1):
                out[f"{name}{arm}_{rep+1}"] = pred_d[(i, arm)]
        for i, name in enumerate(Z_LEARNER_NAMES):
            out[f"{name}_{rep+1}"] = pred_z[i]
        mY0, mY1, mD0, mD1, pZ = shortstack_predictions(
            Y, Dval, Zval, pred_y, pred_d, pred_z)
        out[f"Y_net_tfa_ss0_{rep+1}"] = mY0
        out[f"Y_net_tfa_ss1_{rep+1}"] = mY1
        out[f"D_p401_ss0_{rep+1}"] = mD0
        out[f"D_p401_ss1_{rep+1}"] = mD1
        out[f"Z_e401_ss_{rep+1}"] = pZ
    out.to_parquet(OUT_FILE, index=False)
    print(f"\nsaved cross-fit predictions to {OUT_FILE}")
    print(f"  ({out.shape[0]} obs x {out.shape[1]} vars)")

    summary.to_csv(ROOT / "notes" / "401kLATE_python_summary.csv", index=False)
    print(f"saved summary table to notes/401kLATE_python_summary.csv")


if __name__ == "__main__":
    main()
