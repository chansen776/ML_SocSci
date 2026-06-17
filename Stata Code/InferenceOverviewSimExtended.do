*** Extended overfitting / cross-fitting simulation: Stata parallel of
*** `Python code/InferenceOverviewSimExtended.py`.
***
*** Mirrors the Stata DML simulation (Ahrens ddml_simulations sim_Overfit, dgp 5
*** + dgp 3) using Stata's embedded Python (mlssshort env, where torch lives).
*** DGP and DML loops are in Python end-to-end so the DNN learner matches the
*** canonical .py byte-for-byte (sklearn-MLPRegressor-equivalent PyTorch
*** trainer).
***
*** Sequential by design: joblib loky backend hangs inside Stata's embedded
*** Python on Windows. For canonical parallel runs use the
*** Python script: `python "Python code/InferenceOverviewSimExtended.py" 1000 8`.
***
*** USAGE (Windows, from project root):
***   "StataMP-64" /e do "Stata Code/InferenceOverviewSimExtended.do" [n_rep] where StataMP-64 calls Stata 
***   E.g., on my work PC:
***   & "C:\Program Files\Stata18\StataMP-64.exe" /e do "Stata Code/InferenceOverviewSimExtended.do" [1000]
*** From git-bash, prefix with `MSYS_NO_PATHCONV=1`.
***
*** Output: smoke runs (n_rep < 50) -> notes/ (gitignored).
***         Canonical runs (n_rep >= 50) -> Data/InferenceOverviewSimExtended_stata.csv.
***         Python's canonical CSV at Data/InferenceOverviewSimExtended.csv is
***         never touched by this script (different filename).

clear all
set more off

* n_rep from the optional positional arg; default 3 (smoke) when called with
* none (e.g. from RunAll.do). NB: cond("`1'"!="", `1', 3) fails on an empty
* `1' because the middle term expands to nothing -> "invalid syntax".
if "`1'" != "" {
    local n_rep = `1'
}
else {
    local n_rep = 3
}
display as text "Stata / InferenceOverviewSimExtended:  n_rep=`n_rep'  (sequential)"

python:
import os
import time
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, toeplitz
from sklearn.linear_model import LassoCV
import torch
import torch.nn as nn
import torch.optim as optim
from sfi import Macro

torch.set_num_threads(1)

# ---- shared parameters ----
N        = 1000
P        = 50
ALPHA0   = 0.5
K_FOLDS  = 5

_SX_CHOL = cholesky(toeplitz(0.5 ** np.arange(P)))
_BETA    = 0.9 ** np.arange(1, P + 1)


def _draw_X(rng):
    return rng.standard_normal((N, P)) @ _SX_CHOL


def _hetero_sigma(z):
    s2 = (1.0 + z) ** 2
    return np.sqrt(s2 / s2.mean())


# ---- DGPs ----
def linear_dgp(rng):
    cy, cd = 0.189, 0.298
    X = _draw_X(rng)
    Xall = X @ _BETA
    v = rng.standard_normal(N)
    e = rng.standard_normal(N)
    sigd = _hetero_sigma(cd * Xall)
    D = cd * Xall + v * sigd
    sigy = _hetero_sigma(ALPHA0 * D + cy * Xall)
    Y = ALPHA0 * D + cy * Xall + e * sigy
    EYX = (ALPHA0 * cd + cy) * Xall
    EDX = cd * Xall
    return X, D, Y, EYX, EDX


def nonlinear_dgp(rng):
    cy, cd = 1.42, 2.29
    X = _draw_X(rng)
    Xall = ((X[:, 0] > 0.3) & (X[:, 1] > 0.0) & (X[:, 2] > -1.0)).astype(np.float64)
    v = rng.standard_normal(N)
    e = rng.standard_normal(N)
    sigd = _hetero_sigma(cd * Xall)
    D = cd * Xall + v * sigd
    sigy = _hetero_sigma(ALPHA0 * D + cy * Xall)
    Y = ALPHA0 * D + cy * Xall + e * sigy
    EYX = (ALPHA0 * cd + cy) * Xall
    EDX = cd * Xall
    return X, D, Y, EYX, EDX


# ---- DNN: PyTorch port of sklearn MLPRegressor defaults ----
def fit_sklearn_like_mlp(X, y,
                          hidden=(20, 20),
                          alpha=1e-4,
                          lr=1e-3,
                          max_iter=200,
                          batch_size_cap=200,
                          tol=1e-4,
                          n_iter_no_change=10):
    Xa = np.asarray(X, dtype=np.float32)
    ya = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    n, p = Xa.shape
    bs = min(batch_size_cap, n)

    layers = []
    prev = p
    for h in hidden:
        lin = nn.Linear(prev, h)
        nn.init.xavier_uniform_(lin.weight)
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

    best_loss = float('inf')
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


# ---- Estimators ----
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


def run_one_rep(ii):
    seed = 1234 + ii
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
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


# ---- Driver ----
n_rep = int(Macro.getLocal("n_rep"))
print(f"  Python: running {n_rep} reps sequentially "
      f"(joblib parallel disabled in Stata embedded Python)")

t0 = time.perf_counter()
records = []
for ii in range(n_rep):
    records.append(run_one_rep(ii))
    if (ii + 1) <= 3 or (ii + 1) % max(1, n_rep // 10) == 0:
        elapsed = time.perf_counter() - t0
        per = elapsed / (ii + 1)
        eta = per * (n_rep - ii - 1)
        print(f"  rep {ii+1}/{n_rep}  elapsed={elapsed:.1f}s  per-rep={per:.2f}s  ETA={eta:.0f}s",
              flush=True)
elapsed = time.perf_counter() - t0
df = pd.DataFrame(records)

print(f"\n  Total wall: {elapsed:.1f}s ({elapsed/60:.2f} min) for {n_rep} reps")
print(f"  alpha_0 = {ALPHA0}\n")
print(f"  {'estimator':<14} {'mean':>8} {'sd':>8} {'bias':>9}")
for c in ("lin_oracle", "lin_dml_las", "lin_dml_dnn",
          "nl_oracle",  "nl_dml_las",  "nl_dml_dnn",  "nl_full_dnn"):
    v = df[c]
    print(f"  {c:<14} {v.mean():>8.3f} {v.std():>8.3f} {v.mean() - ALPHA0:>+9.3f}")

# Output: smoke vs canonical paths.
project_root = os.getcwd()
if n_rep >= 50:
    out_dir  = os.path.join(project_root, "Data")
    out_name = "InferenceOverviewSimExtended_stata.csv"
else:
    out_dir  = os.path.join(project_root, "notes")
    out_name = "InferenceOverviewSimExtended_stata_smoke.csv"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, out_name)
df.to_csv(out_csv, index=False)
print(f"\n  Saved raw -> {out_csv}")
end
