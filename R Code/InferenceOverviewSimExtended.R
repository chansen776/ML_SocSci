# Extended overfitting / cross-fitting simulation: R parallel of
# `Python code/InferenceOverviewSimExtended.py`.
#
# Mirrors the Stata DML simulation (Ahrens ddml_simulations sim_Overfit, dgp 5
# + dgp 3) as closely as the R + reticulate combo allows. DGP and DML loops are
# in R; the DNN learner is a sklearn-MLPRegressor-equivalent PyTorch trainer
# defined in embedded Python (mlssshort env, where torch lives) so it matches
# the canonical Python file's nuisance learner exactly.
#
# DGPs (from Stata `mysim`):
#   X ~ N(0, S_X) with S_X[i,j] = 0.5^|i-j|, n=1000, p=50, alpha_0 = 0.5
#   v, e ~ N(0,1) iid; v independent of (e,X); e independent of (v,X)
#
#   Heteroskedastic noise scaling:
#     sigd = sqrt((1 + cd*Xall)^2 / mean((1 + cd*Xall)^2))
#     sigy = sqrt((1 + alpha0*D + cy*Xall)^2 / mean(.))
#   D = cd*Xall + v*sigd
#   Y = alpha0*D + cy*Xall + e*sigy
#
#   DGP 5 -- linear (approximately sparse):
#     Xall = sum_{j=1..p} (0.9)^j * X_j        (Stata: starts at 0.9 for j=1)
#     cy = 0.189, cd = 0.298
#
#   DGP 3 -- nonlinear (indicator product):
#     Xall = 1(X_1 > 0.3) * 1(X_2 > 0) * 1(X_3 > -1)
#     cy = 1.42, cd = 2.29
#
# Estimators per rep (matches the .py):
#   oracle    : OLS on r_Y ~ r_D using TRUE E[Y|X], E[D|X]
#   dml_las   : 5-fold cross-fit DML with cv.glmnet (lambda.min) nuisances
#   dml_dnn   : 5-fold cross-fit DML with sklearn-MLP-equivalent PyTorch
#               nuisances (via reticulate -> mlssshort env)
#   full_dnn  : (DGP 3 only) full-sample DNN, the overfit baseline
#
# Output:
#   For n_rep >= 50 (canonical):
#     Data/InferenceOverviewSimExtended_R.csv
#   For n_rep <  50 (smoke test, gitignored):
#     notes/InferenceOverviewSimExtended_R_smoke.csv
# These are distinct from the Python canonical figure source
# (Data/InferenceOverviewSimExtended.csv); R and Python output do NOT collide.
#
# Run as:
#   "Rscript" --vanilla "R Code/InferenceOverviewSimExtended.R" n_rep where "Rscript" is the path to your Rscript executable. 
# E.g., on my desktop with R 4.5.2:
#   & "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" --vanilla "R Code/InferenceOverviewSimExtended.R" 1000
# Defaults to n_rep = 3 (smoke test) when no arg.

suppressPackageStartupMessages({
  here_dir <- if (dir.exists("R Code")) "R Code" else "."
  source(file.path(here_dir, "setup_reticulate.R"))
  library(reticulate)
  library(glmnet)
})

if (!interactive()) pdf(NULL)

# ---- args ----
args <- commandArgs(trailingOnly = TRUE)
n_rep <- if (length(args) >= 1L) as.integer(args[[1]]) else 3L
cat(sprintf("InferenceOverviewSimExtended (R parallel): n_rep=%d\n", n_rep))

# ---- shared parameters ----
N        <- 1000L
P        <- 50L
ALPHA0   <- 0.5
K_FOLDS  <- 5L

# X covariance: S_X[i,j] = 0.5^|i-j|; Cholesky used to draw X = Z %*% SX_chol.
SX_CHOL <- chol(0.5 ^ abs(outer(seq_len(P), seq_len(P), "-")))

# Linear DGP coefficient vector: beta_j = (0.9)^j for j = 1..p (matches Stata).
BETA <- 0.9 ^ seq_len(P)

# ---- DNN: PyTorch port of sklearn MLPRegressor defaults (defined in Python) ----
py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
        lin = nn.Linear(prev, int(h))
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
        prev = int(h)
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
    for _ in range(int(max_iter)):
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


def torch_seed(seed):
    torch.manual_seed(int(seed))


def torch_set_num_threads(k):
    torch.set_num_threads(int(k))
")
py$torch_set_num_threads(1L)

fit_dnn <- function(X, y) {
  py$fit_sklearn_like_mlp(X, y)
}
predict_dnn <- function(model, X) {
  py$predict_torch(model, X)
}

# ---- DGPs ----
draw_X <- function() {
  matrix(rnorm(N * P), N, P) %*% SX_CHOL
}

hetero_sigma <- function(z) {
  s2 <- (1 + z) ^ 2
  sqrt(s2 / mean(s2))
}

linear_dgp <- function() {
  cy <- 0.189; cd <- 0.298
  X <- draw_X()
  Xall <- as.numeric(X %*% BETA)
  v <- rnorm(N); e <- rnorm(N)
  sigd <- hetero_sigma(cd * Xall)
  D <- cd * Xall + v * sigd
  sigy <- hetero_sigma(ALPHA0 * D + cy * Xall)
  Y <- ALPHA0 * D + cy * Xall + e * sigy
  list(X = X, D = D, Y = Y,
       EYX = (ALPHA0 * cd + cy) * Xall,
       EDX = cd * Xall)
}

nonlinear_dgp <- function() {
  cy <- 1.42; cd <- 2.29
  X <- draw_X()
  Xall <- as.numeric((X[, 1] > 0.3) & (X[, 2] > 0.0) & (X[, 3] > -1.0))
  v <- rnorm(N); e <- rnorm(N)
  sigd <- hetero_sigma(cd * Xall)
  D <- cd * Xall + v * sigd
  sigy <- hetero_sigma(ALPHA0 * D + cy * Xall)
  Y <- ALPHA0 * D + cy * Xall + e * sigy
  list(X = X, D = D, Y = Y,
       EYX = (ALPHA0 * cd + cy) * Xall,
       EDX = cd * Xall)
}

# ---- Estimators ----
oracle_alpha <- function(Y, D, EYX, EDX) {
  rY <- Y - EYX
  rD <- D - EDX
  as.numeric(sum(rD * rY) / sum(rD * rD))
}

kfold_indices <- function(n, K) {
  idx <- sample.int(n)
  split(idx, cut(seq_along(idx), breaks = K, labels = FALSE))
}

dml_alpha_lasso <- function(X, Y, D) {
  n <- nrow(X)
  folds <- kfold_indices(n, K_FOLDS)
  rY <- numeric(n); rD <- numeric(n)
  for (k in seq_len(K_FOLDS)) {
    test  <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    fy <- cv.glmnet(X[train, ], Y[train], nfolds = 5, standardize = TRUE)
    fd <- cv.glmnet(X[train, ], D[train], nfolds = 5, standardize = TRUE)
    rY[test] <- Y[test] - as.numeric(predict(fy, newx = X[test, ], s = "lambda.min"))
    rD[test] <- D[test] - as.numeric(predict(fd, newx = X[test, ], s = "lambda.min"))
  }
  sum(rD * rY) / sum(rD * rD)
}

dml_alpha_dnn <- function(X, Y, D) {
  n <- nrow(X)
  folds <- kfold_indices(n, K_FOLDS)
  rY <- numeric(n); rD <- numeric(n)
  for (k in seq_len(K_FOLDS)) {
    test  <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    my <- fit_dnn(X[train, ], Y[train])
    md <- fit_dnn(X[train, ], D[train])
    rY[test] <- Y[test] - predict_dnn(my, X[test, ])
    rD[test] <- D[test] - predict_dnn(md, X[test, ])
  }
  sum(rD * rY) / sum(rD * rD)
}

full_dnn_alpha <- function(X, Y, D) {
  my <- fit_dnn(X, Y)
  md <- fit_dnn(X, D)
  rY <- Y - predict_dnn(my, X)
  rD <- D - predict_dnn(md, X)
  sum(rD * rY) / sum(rD * rD)
}

# ---- Per-rep ----
run_one_rep <- function(ii) {
  seed <- 1234L + ii
  set.seed(seed)
  py$torch_seed(seed)

  a <- linear_dgp()
  b <- nonlinear_dgp()

  list(
    lin_oracle  = oracle_alpha(a$Y, a$D, a$EYX, a$EDX),
    lin_dml_las = dml_alpha_lasso(a$X, a$Y, a$D),
    lin_dml_dnn = dml_alpha_dnn(a$X, a$Y, a$D),
    nl_oracle   = oracle_alpha(b$Y, b$D, b$EYX, b$EDX),
    nl_dml_las  = dml_alpha_lasso(b$X, b$Y, b$D),
    nl_dml_dnn  = dml_alpha_dnn(b$X, b$Y, b$D),
    nl_full_dnn = full_dnn_alpha(b$X, b$Y, b$D)
  )
}

# ---- Driver ----
t0 <- Sys.time()
results <- vector("list", n_rep)
for (ii in seq_len(n_rep)) {
  results[[ii]] <- run_one_rep(ii - 1L)  # match Python: ii in 0..n_rep-1
  if (ii <= 3L || ii %% max(1L, n_rep %/% 10L) == 0L) {
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    per <- elapsed / ii
    eta <- per * (n_rep - ii)
    cat(sprintf("  rep %d/%d  elapsed=%.1fs  per-rep=%.2fs  ETA=%.0fs\n",
                ii, n_rep, elapsed, per, eta))
  }
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
df <- do.call(rbind.data.frame, results)

cat(sprintf("\nTotal wall: %.1fs (%.2f min) for %d reps\n",
            elapsed, elapsed / 60, n_rep))
cat(sprintf("alpha_0 = %g\n", ALPHA0))
cat(sprintf("  %-14s %8s %8s %9s\n", "estimator", "mean", "sd", "bias"))
for (c in c("lin_oracle", "lin_dml_las", "lin_dml_dnn",
            "nl_oracle",  "nl_dml_las",  "nl_dml_dnn",  "nl_full_dnn")) {
  v <- df[[c]]
  cat(sprintf("  %-14s %8.3f %8.3f %+9.3f\n",
              c, mean(v), sd(v), mean(v) - ALPHA0))
}

# Output: smoke runs go to notes/ (gitignored); larger runs go to Data/.
# Python's canonical CSV at Data/InferenceOverviewSimExtended.csv is never
# touched by this script (different filename).
project_root <- if (basename(getwd()) == "R Code") dirname(getwd()) else getwd()
if (n_rep >= 50L) {
  out_dir  <- file.path(project_root, "Data")
  out_name <- "InferenceOverviewSimExtended_R.csv"
} else {
  out_dir  <- file.path(project_root, "notes")
  out_name <- "InferenceOverviewSimExtended_R_smoke.csv"
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, out_name)
write.csv(df, out_path, row.names = FALSE)

cat(sprintf("\nSaved raw -> %s\n", out_path))
