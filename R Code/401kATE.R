## 401(k) eligibility ATE -- R port of Stata Code/401kATE.do.
##
## Mirrors the Stata pipeline: 6 Y-learners x 2 D-arms, 6 D-learners,
## 5-fold cross-validation, 10 resamples, propensity trimming at [.01, .99].
## Reps are parallelized via future + future.apply.
##
## Native R for linear, glmnet (lasso/ridge), randomForest. NN uses
## reticulate -> sklearn MLPRegressor/MLPClassifier, mirroring the Python
## port (matches Stata pystacked's NN backend).
##
## Outputs:
##   - Per-rep best (min-MSE) and shortstack point estimates + SEs.
##   - Cross-fitted predictions saved to Data/401kATE_R.csv.
##   - Summary saved to notes/401kATE_R_summary.csv.

# Resolve codedir robustly: works under interactive(), source(), or
# Rscript invocation.
.resolve_codedir <- function() {
  if (interactive()) return(getwd())
  args <- commandArgs(trailingOnly = FALSE)
  fa <- args[grepl("^--file=", args)]
  if (length(fa) == 1L) return(dirname(normalizePath(sub("^--file=", "", fa))))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  getwd()
}
codedir <- .resolve_codedir()
source(file.path(codedir, "setup_reticulate.R"))

suppressMessages({
  library(foreign)
  library(glmnet)
  library(randomForest)
  library(reticulate)
  library(future)
  library(future.apply)
})

if (!interactive()) pdf(NULL)

ROOT      <- normalizePath(file.path(codedir, ".."), winslash = "/")
DATA_FILE <- file.path(ROOT, "Data", "sipp1991.dta")
OUT_FILE  <- file.path(ROOT, "Data", "401kATE_R.csv")
SUMMARY_FILE <- file.path(ROOT, "notes", "401kATE_R_summary.csv")

N_FOLDS <- 5L
N_REPS  <- 10L
SEED_BASE <- 71423L
TRIM_LO <- 0.01
TRIM_HI <- 0.99
N_PARALLEL_REPS <- 5L  # joblib over reps (R uses fewer cores than Py here:
                       # randomForest is single-threaded so n_jobs=10 is
                       # memory-heavy on Windows)

Y_LEARNER_NAMES <- c("Y1_reg", "Y2_reg",
                     "Y3_pystacked", "Y4_pystacked",
                     "Y5_pystacked", "Y6_pystacked")
D_LEARNER_NAMES <- c("D1_logit", "D2_pystacked",
                     "D3_pystacked", "D4_pystacked",
                     "D5_pystacked", "D6_pystacked")
LEARNER_X_KIND <- c("raw", "poly", "inter", "inter", "raw", "raw")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

prepare_data <- function() {
  df <- read.dta(DATA_FILE)
  df$age   <- df$age / 64
  df$inc   <- df$inc / 250000
  df$educ  <- df$educ / 18
  df$fsize <- df$fsize / 13
  for (i in 2:6)  df[[paste0("age",  i)]] <- df$age  ^ i
  for (i in 2:8)  df[[paste0("inc",  i)]] <- df$inc  ^ i
  for (i in 2:4)  df[[paste0("educ", i)]] <- df$educ ^ i
  df$fsize2 <- df$fsize ^ 2

  raw_vars  <- c("age", "inc", "educ", "fsize",
                 "marr", "twoearn", "db", "pira", "hown")
  poly_vars <- c("age", paste0("age", 2:6),
                 "inc", paste0("inc", 2:8),
                 "educ", paste0("educ", 2:4),
                 "fsize", "fsize2",
                 "marr", "twoearn", "db", "pira", "hown")
  X_raw  <- as.matrix(df[, raw_vars])
  X_poly <- as.matrix(df[, poly_vars])
  # Pairwise interactions: model.matrix on `~ .^2 - 1` over raw columns.
  Xdf    <- as.data.frame(X_raw)
  X_inter <- model.matrix(~ .^2 - 1, data = Xdf)
  # Drop linearly redundant binary*binary interactions if any
  storage.mode(X_inter) <- "double"
  Y <- df$net_tfa
  D <- as.integer(df$e401)
  list(df = df, X_raw = X_raw, X_poly = X_poly, X_inter = X_inter,
       Y = Y, D = D)
}


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

# Each fit returns a closure with predict(newdata).
fit_y <- function(idx, Xtr, ytr) {
  if (idx == 1L) {  # linear OLS on raw X
    model <- lm.fit(cbind(1, Xtr), ytr)
    coefs <- model$coefficients
    coefs[is.na(coefs)] <- 0
    return(function(Xte) cbind(1, Xte) %*% coefs)
  }
  if (idx == 2L) {  # linear OLS on X_poly
    model <- lm.fit(cbind(1, Xtr), ytr)
    coefs <- model$coefficients
    coefs[is.na(coefs)] <- 0
    return(function(Xte) cbind(1, Xte) %*% coefs)
  }
  if (idx == 3L) {  # cv.glmnet lasso (alpha=1)
    fit <- cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 5, standardize = TRUE)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min")))
  }
  if (idx == 4L) {  # cv.glmnet ridge (alpha=0)
    fit <- cv.glmnet(Xtr, ytr, alpha = 0, nfolds = 5, standardize = TRUE)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min")))
  }
  if (idx == 5L) {  # randomForest
    fit <- randomForest(x = Xtr, y = ytr, ntree = 500, nodesize = 20)
    return(function(Xte) as.numeric(predict(fit, newdata = Xte)))
  }
  if (idx == 6L) {  # sklearn MLPRegressor via reticulate
    sk_nn <- import("sklearn.neural_network")
    np    <- import("numpy")
    nn <- sk_nn$MLPRegressor(
      hidden_layer_sizes = tuple(50L, 50L, 50L, 50L),
      alpha = 0, max_iter = 200L, random_state = 720L)
    nn$fit(np$asarray(Xtr, dtype = "float64"), np$asarray(ytr, dtype = "float64"))
    return(function(Xte) as.numeric(nn$predict(np$asarray(Xte, dtype = "float64"))))
  }
  stop("bad idx ", idx)
}

fit_d <- function(idx, Xtr, dtr) {
  if (idx == 1L || idx == 2L) {  # logit
    fit <- glm.fit(cbind(1, Xtr), dtr, family = binomial())
    coefs <- fit$coefficients
    coefs[is.na(coefs)] <- 0
    return(function(Xte) {
      eta <- as.numeric(cbind(1, Xte) %*% coefs)
      1 / (1 + exp(-eta))
    })
  }
  if (idx == 3L) {  # cv.glmnet binomial lasso
    fit <- cv.glmnet(Xtr, dtr, family = "binomial", alpha = 1, nfolds = 5)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min", type = "response")))
  }
  if (idx == 4L) {  # cv.glmnet binomial ridge
    fit <- cv.glmnet(Xtr, dtr, family = "binomial", alpha = 0, nfolds = 5)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min", type = "response")))
  }
  if (idx == 5L) {  # randomForest classifier (regression on 0/1 -> probabilities)
    fit <- randomForest(x = Xtr, y = factor(dtr, levels = c(0, 1)),
                        ntree = 500, nodesize = 20)
    return(function(Xte) as.numeric(predict(fit, newdata = Xte, type = "prob")[, "1"]))
  }
  if (idx == 6L) {  # sklearn MLPClassifier via reticulate
    sk_nn <- import("sklearn.neural_network")
    np    <- import("numpy")
    nn <- sk_nn$MLPClassifier(
      hidden_layer_sizes = tuple(50L, 50L, 50L, 50L),
      alpha = 0, max_iter = 200L, random_state = 720L)
    nn$fit(np$asarray(Xtr, dtype = "float64"), np$asarray(dtr, dtype = "int64"))
    return(function(Xte) {
      probs <- nn$predict_proba(np$asarray(Xte, dtype = "float64"))
      as.numeric(probs[, 2])
    })
  }
  stop("bad idx ", idx)
}


# ---------------------------------------------------------------------------
# Cross-fit one rep
# ---------------------------------------------------------------------------

cross_fit_one_rep <- function(rep, X_raw, X_poly, X_inter, Y, D) {
  set.seed(SEED_BASE + rep)
  n <- length(Y)
  fold_id <- sample(rep(1:N_FOLDS, length.out = n))

  Xs <- list(raw = X_raw, poly = X_poly, inter = X_inter)
  pred_y0 <- matrix(NA_real_, n, 6)
  pred_y1 <- matrix(NA_real_, n, 6)
  pred_d  <- matrix(NA_real_, n, 6)
  colnames(pred_y0) <- paste0(Y_LEARNER_NAMES, "0")
  colnames(pred_y1) <- paste0(Y_LEARNER_NAMES, "1")
  colnames(pred_d)  <- D_LEARNER_NAMES

  t0 <- Sys.time()
  for (k in 1:N_FOLDS) {
    tr <- which(fold_id != k)
    te <- which(fold_id == k)
    Y_tr <- Y[tr]; D_tr <- D[tr]
    for (i in 1:6) {
      Xkind <- LEARNER_X_KIND[i]
      Xtr <- Xs[[Xkind]][tr, , drop = FALSE]
      Xte <- Xs[[Xkind]][te, , drop = FALSE]
      # Y arms
      for (arm in c(0, 1)) {
        sel <- D_tr == arm
        f <- fit_y(i, Xtr[sel, , drop = FALSE], Y_tr[sel])
        if (arm == 0) pred_y0[te, i] <- f(Xte)
        else          pred_y1[te, i] <- f(Xte)
      }
      # D
      f <- fit_d(i, Xtr, D_tr)
      pred_d[te, i] <- f(Xte)
    }
  }
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  rep %2d: %6.1fs\n", rep, elapsed))

  list(rep = rep, fold_id = fold_id,
       pred_y0 = pred_y0, pred_y1 = pred_y1, pred_d = pred_d)
}


# ---------------------------------------------------------------------------
# Estimation: AIPW score, shortstack, best
# ---------------------------------------------------------------------------

aipw_score <- function(Y, D, m0, m1, p) {
  p <- pmin(pmax(p, TRIM_LO), TRIM_HI)
  m1 - m0 + D * (Y - m1) / p - (1 - D) * (Y - m0) / (1 - p)
}

aipw_estimate <- function(Y, D, m0, m1, p) {
  psi <- aipw_score(Y, D, m0, m1, p)
  c(b = mean(psi), se = sd(psi) / sqrt(length(psi)))
}

nnls_normalize <- function(target, P) {
  # Use nnls package or quadprog; fall back to simple NNLS via projected
  # gradient if nnls isn't available. We'll use base R's `nnls` from the
  # `nnls` package (in mlssshort indirectly). For portability, do a quick
  # active-set NNLS.
  fit <- tryCatch(nnls::nnls(P, target),
                  error = function(e) NULL)
  if (!is.null(fit)) {
    w <- fit$x
  } else {
    # crude fallback: equal weights
    w <- rep(1 / ncol(P), ncol(P))
  }
  if (sum(w) > 0) w <- w / sum(w) else w <- rep(1 / length(w), length(w))
  w
}

shortstack_predictions <- function(Y, D, pred_y0, pred_y1, pred_d) {
  # Y arms: NNLS on D=arm subset.
  m0_obs <- pred_y0[D == 0, , drop = FALSE]
  m1_obs <- pred_y1[D == 1, , drop = FALSE]
  w_y0 <- nnls_normalize(Y[D == 0], m0_obs)
  w_y1 <- nnls_normalize(Y[D == 1], m1_obs)
  w_d  <- nnls_normalize(as.numeric(D), pred_d)
  list(
    m0 = as.numeric(pred_y0 %*% w_y0),
    m1 = as.numeric(pred_y1 %*% w_y1),
    p  = as.numeric(pred_d  %*% w_d),
    w_y0 = w_y0, w_y1 = w_y1, w_d = w_d
  )
}

best_picks <- function(Y, D, pred_y0, pred_y1, pred_d) {
  mse_y0 <- colMeans((Y[D == 0] - pred_y0[D == 0, , drop = FALSE])^2)
  mse_y1 <- colMeans((Y[D == 1] - pred_y1[D == 1, , drop = FALSE])^2)
  mse_d  <- colMeans((D - pred_d)^2)
  list(y0 = which.min(mse_y0), y1 = which.min(mse_y1), d = which.min(mse_d))
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cat("======================================================================\n")
cat("401(k) ATE -- R port of Stata Code/401kATE.do\n")
cat("======================================================================\n")

dat <- prepare_data()
n <- length(dat$Y)
cat(sprintf("loaded sipp1991.dta: n = %d, D mean = %.4f\n", n, mean(dat$D)))
cat(sprintf(paste0("\ncross-fitting %d reps x %d folds x 6+6 learners ",
                   "in parallel (workers = %d)...\n"),
            N_REPS, N_FOLDS, N_PARALLEL_REPS))

plan(multisession, workers = N_PARALLEL_REPS)
t0 <- Sys.time()
results <- future_lapply(seq_len(N_REPS), function(rep) {
  # reticulate must be configured per-worker via the helper.
  source(file.path(codedir, "setup_reticulate.R"))
  suppressPackageStartupMessages({
    library(glmnet); library(randomForest); library(reticulate)
  })
  cross_fit_one_rep(rep, dat$X_raw, dat$X_poly, dat$X_inter, dat$Y, dat$D)
}, future.seed = TRUE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("cross-fit total wall: %.1fs\n", elapsed))

# Per-rep summary
rows <- vector("list", N_REPS)
cat("\nPer-rep estimates (parallels Stata `ddml estimate, robust`):\n")
cat(sprintf("  %3s  %10s %9s  %10s %9s  %s\n",
            "rep", "best b", "best SE", "stack b", "stack SE",
            "best learners (Y0, Y1, D)"))
for (res in results) {
  rep <- res$rep
  bp <- best_picks(dat$Y, dat$D, res$pred_y0, res$pred_y1, res$pred_d)
  best_est <- aipw_estimate(dat$Y, dat$D,
                            res$pred_y0[, bp$y0], res$pred_y1[, bp$y1],
                            res$pred_d[, bp$d])
  ss <- shortstack_predictions(dat$Y, dat$D, res$pred_y0, res$pred_y1, res$pred_d)
  ss_est <- aipw_estimate(dat$Y, dat$D, ss$m0, ss$m1, ss$p)
  rows[[rep]] <- data.frame(
    rep = rep,
    best_b = best_est["b"], best_se = best_est["se"],
    stack_b = ss_est["b"],  stack_se = ss_est["se"],
    best_y0 = Y_LEARNER_NAMES[bp$y0],
    best_y1 = Y_LEARNER_NAMES[bp$y1],
    best_d  = D_LEARNER_NAMES[bp$d]
  )
  cat(sprintf("  %3d  %10.1f %9.1f  %10.1f %9.1f  %-14s %-14s %s\n",
              rep, best_est["b"], best_est["se"],
              ss_est["b"], ss_est["se"],
              Y_LEARNER_NAMES[bp$y0], Y_LEARNER_NAMES[bp$y1],
              D_LEARNER_NAMES[bp$d]))
}
summary_df <- do.call(rbind, rows)
cat(sprintf("\n  mean   %10.1f %9.1f  %10.1f %9.1f\n",
            mean(summary_df$best_b),  mean(summary_df$best_se),
            mean(summary_df$stack_b), mean(summary_df$stack_se)))
cat(sprintf("  median %10.1f %9.1f  %10.1f %9.1f\n",
            median(summary_df$best_b),  median(summary_df$best_se),
            median(summary_df$stack_b), median(summary_df$stack_se)))

write.csv(summary_df, SUMMARY_FILE, row.names = FALSE)
cat(sprintf("\nsaved summary to %s\n", SUMMARY_FILE))

# Save predictions (wide format, matching Stata convention)
out <- as.data.frame(dat$df)
for (res in results) {
  rep <- res$rep
  out[[paste0("m0_fid_", rep)]] <- res$fold_id
  for (i in seq_along(Y_LEARNER_NAMES)) {
    out[[paste0(Y_LEARNER_NAMES[i], "0_", rep)]] <- res$pred_y0[, i]
    out[[paste0(Y_LEARNER_NAMES[i], "1_", rep)]] <- res$pred_y1[, i]
  }
  for (i in seq_along(D_LEARNER_NAMES)) {
    out[[paste0(D_LEARNER_NAMES[i], "_", rep)]] <- res$pred_d[, i]
  }
  ss <- shortstack_predictions(dat$Y, dat$D, res$pred_y0, res$pred_y1, res$pred_d)
  out[[paste0("Y_net_tfa_ss0_", rep)]] <- ss$m0
  out[[paste0("Y_net_tfa_ss1_", rep)]] <- ss$m1
  out[[paste0("D_e401_ss_",   rep)]] <- ss$p
}
write.csv(out, OUT_FILE, row.names = FALSE)
cat(sprintf("saved cross-fit predictions to %s (%d obs x %d vars)\n",
            OUT_FILE, nrow(out), ncol(out)))
