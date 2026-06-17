"""
Extended overfitting / cross-fitting simulation matching the original Stata
DML simulation (Ahrens ddml_simulations sim_Overfit, dgp 5 + dgp 3) as
closely as PyTorch allows.

DGPs (from `mysim` in the Stata script):
  X ~ N(0, S_X) with [S_X]_{i,j} = 0.5^|i-j|, n=1000, p=50, alpha_0 = 0.5
  v, e ~ N(0, 1) iid; v independent of (e, X); e independent of (v, X)

  Heteroskedastic noise scaling (per Stata's `gen sigd=(1+cd*Xall)^2;
  replace sigd = sqrt(sigd/mean(sigd))`):
    sigd = sqrt((1 + cd*Xall)^2 / mean((1 + cd*Xall)^2))
    sigy = sqrt((1 + beta*D + cy*Xall)^2 / mean((1 + beta*D + cy*Xall)^2))

  D = cd * Xall + v * sigd
  Y = beta * D + cy * Xall + e * sigy

  DGP 5 -- linear (approximately sparse):
    Xall = sum_{j=1..p} (0.9)^j * X_j        (Stata: starts at 0.9 for j=1)
    cy = 0.189, cd = 0.298

  DGP 3 -- nonlinear (indicator product):
    Xall = 1(X_1 > 0.3) * 1(X_2 > 0) * 1(X_3 > -1)
    cy = 1.42, cd = 2.29

Oracle conditional means (used for the oracle estimator):
  E[D|X] = cd * Xall                  (since v independent of X)
  E[Y|X] = (beta*cd + cy) * Xall      (substituting D and integrating v out)

DNN learner: pystacked NN -> sklearn MLPRegressor with hidden_layer_sizes=
(20, 20) and ALL OTHER PARAMETERS at sklearn defaults. Ported to PyTorch:
  - Architecture: 2 hidden ReLU layers of 20 units + linear output
  - Glorot-uniform weight init, zero bias init
  - Adam optimizer (lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
  - L2 weight decay alpha=1e-4 via torch's `weight_decay`
  - max_iter=200 epochs, batch_size = min(200, n_samples)
  - Training-loss tol-based early stopping: tol=1e-4, n_iter_no_change=10
  - shuffle=True each epoch

Estimators per rep:
  oracle    : OLS on r_Y ~ r_D using TRUE E[Y|X], E[D|X]
  dml_las   : 5-fold cross-fit DML with sklearn LassoCV nuisances
  dml_dnn   : 5-fold cross-fit DML with sklearn-MLP-equivalent PyTorch
              nuisances
  full_dnn  : (DGP 3 only) full-sample DNN (no cross-fitting), the overfit
              baseline

Writes the per-rep estimates to Data/InferenceOverviewSimExtended.csv;
slide figures are produced separately by InferenceOverviewSimExtended_plots.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, toeplitz
from sklearn.linear_model import LassoCV
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

HERE    = Path(__file__).resolve().parent
DATADIR = (HERE.parent / "Data").resolve()


# ---------- shared parameters ----------

N        = 1000
P        = 50
ALPHA0   = 0.5
K_FOLDS  = 5

# X covariance: S_X[i,j] = 0.5^|i-j|; Cholesky used to draw X = Z @ SX_chol.
_SX_CHOL = cholesky(toeplitz(0.5 ** np.arange(P)))

# Linear DGP coefficient vector: beta_j = (0.9)^j for j = 1..p (matches
# Stata `forvalues j = 1(1)`p' { replace Xall = Xall + (.9)^(`j')*x`j' }`)
_BETA = 0.9 ** np.arange(1, P + 1)


def _draw_X(rng):
    return rng.standard_normal((N, P)) @ _SX_CHOL


def _hetero_sigma(z):
    """Stata's `(1+z)^2` rescaled so its sample mean is 1, then sqrt."""
    s2 = (1.0 + z) ** 2
    return np.sqrt(s2 / s2.mean())


# ---------- DGPs ----------

def linear_dgp(rng):
    """DGP 5 (linear / approximately sparse)."""
    cy, cd = 0.189, 0.298
    X = _draw_X(rng)
    Xall = X @ _BETA
    v = rng.standard_normal(N)
    e = rng.standard_normal(N)

    sigd = _hetero_sigma(cd * Xall)
    D = cd * Xall + v * sigd

    sigy = _hetero_sigma(ALPHA0 * D + cy * Xall)
    Y = ALPHA0 * D + cy * Xall + e * sigy

    EDX = cd * Xall
    EYX = (ALPHA0 * cd + cy) * Xall
    return X, D, Y, EYX, EDX


def nonlinear_dgp(rng):
    """DGP 3 (nonlinear / indicator product)."""
    cy, cd = 1.42, 2.29
    X = _draw_X(rng)
    Xall = ((X[:, 0] > 0.3) & (X[:, 1] > 0.0) & (X[:, 2] > -1.0)).astype(np.float64)
    v = rng.standard_normal(N)
    e = rng.standard_normal(N)

    sigd = _hetero_sigma(cd * Xall)
    D = cd * Xall + v * sigd

    sigy = _hetero_sigma(ALPHA0 * D + cy * Xall)
    Y = ALPHA0 * D + cy * Xall + e * sigy

    EDX = cd * Xall
    EYX = (ALPHA0 * cd + cy) * Xall
    return X, D, Y, EYX, EDX


# ---------- DNN: PyTorch port of sklearn MLPRegressor defaults ----------

def fit_sklearn_like_mlp(X, y,
                          hidden=(20, 20),
                          alpha=1e-4,           # sklearn: L2 weight-decay coef
                          lr=1e-3,              # sklearn: learning_rate_init
                          max_iter=200,         # sklearn: max_iter
                          batch_size_cap=200,   # sklearn: batch_size = min(200, n_samples)
                          tol=1e-4,
                          n_iter_no_change=10):
    """Single-call MLP fit mirroring sklearn MLPRegressor behaviour with
    default settings. Returns the trained nn.Module."""
    Xa = np.asarray(X, dtype=np.float32)
    ya = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    n, p = Xa.shape
    bs = min(batch_size_cap, n)

    layers = []
    prev = p
    for h in hidden:
        lin = nn.Linear(prev, h)
        nn.init.xavier_uniform_(lin.weight)   # sklearn uses Glorot-uniform
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
        prev = h
    out = nn.Linear(prev, 1)
    nn.init.xavier_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    model = nn.Sequential(*layers)

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                      weight_decay=alpha)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(Xa)
    yt = torch.from_numpy(ya)

    best_loss = float("inf")
    no_imp = 0
    model.train()
    for _ in range(max_iter):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            l = loss_fn(model(Xt[idx]), yt[idx])
            l.backward()
            opt.step()
            running += float(l.item()) * idx.numel()
        epoch_loss = running / n

        # sklearn early-stop logic (training-loss based; > strict on count)
        if epoch_loss > best_loss - tol:
            no_imp += 1
        else:
            no_imp = 0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        if no_imp > n_iter_no_change:
            break

    return model


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return model(Xt).cpu().numpy().flatten()


# ---------- Estimators ----------

def oracle_alpha(Y, D, EYX, EDX):
    rY = Y - EYX
    rD = D - EDX
    return float(rD @ rY / (rD @ rD))


def _kfold_indices(n, K, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, K)


def dml_alpha_lasso(X, Y, D, rng):
    n = X.shape[0]
    folds = _kfold_indices(n, K_FOLDS, rng)
    rY = np.empty(n); rD = np.empty(n)
    for k in range(K_FOLDS):
        test  = folds[k]
        train = np.concatenate([folds[j] for j in range(K_FOLDS) if j != k])
        my = LassoCV(cv=5, alphas=50, max_iter=5000).fit(X[train], Y[train])
        md = LassoCV(cv=5, alphas=50, max_iter=5000).fit(X[train], D[train])
        rY[test] = Y[test] - my.predict(X[test])
        rD[test] = D[test] - md.predict(X[test])
    return float(rD @ rY / (rD @ rD))


def dml_alpha_dnn(X, Y, D, rng):
    n = X.shape[0]
    folds = _kfold_indices(n, K_FOLDS, rng)
    rY = np.empty(n); rD = np.empty(n)
    for k in range(K_FOLDS):
        test  = folds[k]
        train = np.concatenate([folds[j] for j in range(K_FOLDS) if j != k])
        my = fit_sklearn_like_mlp(X[train], Y[train])
        md = fit_sklearn_like_mlp(X[train], D[train])
        rY[test] = Y[test] - predict_torch(my, X[test])
        rD[test] = D[test] - predict_torch(md, X[test])
    return float(rD @ rY / (rD @ rD))


def full_dnn_alpha(X, Y, D):
    my = fit_sklearn_like_mlp(X, Y)
    md = fit_sklearn_like_mlp(X, D)
    rY = Y - predict_torch(my, X)
    rD = D - predict_torch(md, X)
    return float(rD @ rY / (rD @ rD))


# ---------- Per-rep ----------

def run_one_rep(ii):
    seed = 1234 + ii
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    Xa, Da, Ya, EYa, EDa = linear_dgp(rng)
    Xb, Db, Yb, EYb, EDb = nonlinear_dgp(rng)

    return {
        "lin_oracle":  oracle_alpha(Ya, Da, EYa, EDa),
        "lin_dml_las": dml_alpha_lasso(Xa, Ya, Da, rng),
        "lin_dml_dnn": dml_alpha_dnn(Xa, Ya, Da, rng),
        "nl_oracle":   oracle_alpha(Yb, Db, EYb, EDb),
        "nl_dml_las":  dml_alpha_lasso(Xb, Yb, Db, rng),
        "nl_dml_dnn":  dml_alpha_dnn(Xb, Yb, Db, rng),
        "nl_full_dnn": full_dnn_alpha(Xb, Yb, Db),
    }


# ---------- Driver ----------

def run_simulation(n_rep, n_jobs=1, verbose=True):
    if verbose:
        print(f"Running extended sim (Stata DGP 5 + 3, sklearn-MLP-equivalent): "
              f"n_rep={n_rep}, n_jobs={n_jobs}", flush=True)
    t0 = time.perf_counter()
    if n_jobs == 1:
        results = []
        for ii in range(n_rep):
            results.append(run_one_rep(ii))
            if verbose and ((ii + 1) % max(1, n_rep // 10) == 0 or ii < 3):
                elapsed = time.perf_counter() - t0
                per_rep = elapsed / (ii + 1)
                eta = per_rep * (n_rep - ii - 1)
                print(f"  rep {ii+1}/{n_rep}  elapsed={elapsed:.1f}s  per-rep={per_rep:.2f}s  ETA={eta:.0f}s",
                      flush=True)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(run_one_rep)(ii) for ii in range(n_rep)
        )
    elapsed = time.perf_counter() - t0
    return pd.DataFrame(results), elapsed


# ---------- main ----------

def summarize(df):
    print()
    print(f"alpha_0 = {ALPHA0}")
    print(f"  {'estimator':<14} {'mean':>8} {'sd':>8} {'bias':>9}")
    for c in ("lin_oracle", "lin_dml_las", "lin_dml_dnn",
              "nl_oracle",  "nl_dml_las",  "nl_dml_dnn",  "nl_full_dnn"):
        v = df[c]
        print(f"  {c:<14} {v.mean():>8.3f} {v.std():>8.3f} {v.mean() - ALPHA0:>+9.3f}")


if __name__ == "__main__":
    n_rep  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    df, elapsed = run_simulation(n_rep, n_jobs=n_jobs)
    print(f"\nTotal wall: {elapsed:.1f}s ({elapsed/60:.2f} min) for {n_rep} reps "
          f"(eff per-rep {elapsed/n_rep:.2f}s; projected 1000 = {1000*elapsed/n_rep/60:.1f} min)")
    summarize(df)

    DATADIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATADIR / "InferenceOverviewSimExtended.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw -> {csv_path}")
    print("Run InferenceOverviewSimExtended_plots.py to regenerate slide figures.")
