## 401(k) participation LATE -- R port of Stata Code/401kLATE.do.
##
## Cross-fit DML LATE estimation with e401 as the instrument for p401.
## Mirrors the Stata interactive-IV pipeline: 5 nuisance moments
## (E[Y|X,Z=0], E[Y|X,Z=1], E[D|X,Z=0], E[D|X,Z=1], E[Z|X]) x 6 learners,
## 5-fold CV, 10 reps, propensity trimming at [.01, .99].
##
## Reps run in parallel via future + future.apply.
##
## Outputs:
##   - Per-rep best (min-MSE) and shortstack point estimates + SEs.
##   - Cross-fitted predictions saved to Data/401kLATE_R.csv.
##   - Summary saved to notes/401kLATE_R_summary.csv.

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
  library(nnls)
})

if (!interactive()) pdf(NULL)

ROOT      <- normalizePath(file.path(codedir, ".."), winslash = "/")
DATA_FILE <- file.path(ROOT, "Data", "sipp1991.dta")
OUT_FILE  <- file.path(ROOT, "Data", "401kLATE_R.csv")
SUMMARY_FILE <- file.path(ROOT, "notes", "401kLATE_R_summary.csv")

N_FOLDS <- 5L
N_REPS  <- 10L
SEED_BASE <- 71423L
TRIM_LO <- 0.01
TRIM_HI <- 0.99
N_PARALLEL_REPS <- 5L

Y_LEARNER_NAMES <- c("Y1_reg", "Y2_reg",
                     "Y3_pystacked", "Y4_pystacked",
                     "Y5_pystacked", "Y6_pystacked")
D_LEARNER_NAMES <- c("D1_logit", "D2_pystacked",
                     "D3_pystacked", "D4_pystacked",
                     "D5_pystacked", "D6_pystacked")
Z_LEARNER_NAMES <- c("Z1_logit", "Z2_pystacked",
                     "Z3_pystacked", "Z4_pystacked",
                     "Z5_pystacked", "Z6_pystacked")
LEARNER_X_KIND <- c("raw", "poly", "inter", "inter", "raw", "raw")


# ---------------------------------------------------------------------------
# Data prep + learners (reuse the ATE-side definitions)
# ---------------------------------------------------------------------------

# Self-contained: redefine prepare_data + learner factories below.

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
  Xdf    <- as.data.frame(X_raw)
  X_inter <- model.matrix(~ .^2 - 1, data = Xdf)
  storage.mode(X_inter) <- "double"
  Y    <- df$net_tfa
  Dval <- as.integer(df$p401)
  Zval <- as.integer(df$e401)
  list(df = df, X_raw = X_raw, X_poly = X_poly, X_inter = X_inter,
       Y = Y, D = Dval, Z = Zval)
}

fit_reg <- function(idx, Xtr, ytr) {
  if (idx == 1L || idx == 2L) {
    model <- lm.fit(cbind(1, Xtr), ytr)
    coefs <- model$coefficients; coefs[is.na(coefs)] <- 0
    return(function(Xte) cbind(1, Xte) %*% coefs)
  }
  if (idx == 3L) {
    fit <- cv.glmnet(Xtr, ytr, alpha = 1, nfolds = 5, standardize = TRUE)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min")))
  }
  if (idx == 4L) {
    fit <- cv.glmnet(Xtr, ytr, alpha = 0, nfolds = 5, standardize = TRUE)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min")))
  }
  if (idx == 5L) {
    fit <- randomForest(x = Xtr, y = ytr, ntree = 500, nodesize = 20)
    return(function(Xte) as.numeric(predict(fit, newdata = Xte)))
  }
  if (idx == 6L) {
    sk_nn <- import("sklearn.neural_network")
    np    <- import("numpy")
    nn <- sk_nn$MLPRegressor(
      hidden_layer_sizes = tuple(50L, 50L, 50L, 50L),
      alpha = 0, max_iter = 200L, random_state = 720L)
    nn$fit(np$asarray(Xtr, dtype = "float64"),
           np$asarray(ytr, dtype = "float64"))
    return(function(Xte) as.numeric(nn$predict(np$asarray(Xte, dtype = "float64"))))
  }
  stop("bad idx ", idx)
}

fit_clf <- function(idx, Xtr, dtr) {
  if (idx == 1L || idx == 2L) {
    fit <- glm.fit(cbind(1, Xtr), dtr, family = binomial())
    coefs <- fit$coefficients; coefs[is.na(coefs)] <- 0
    return(function(Xte) {
      eta <- as.numeric(cbind(1, Xte) %*% coefs)
      1 / (1 + exp(-eta))
    })
  }
  if (idx == 3L) {
    fit <- cv.glmnet(Xtr, dtr, family = "binomial", alpha = 1, nfolds = 5)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min", type = "response")))
  }
  if (idx == 4L) {
    fit <- cv.glmnet(Xtr, dtr, family = "binomial", alpha = 0, nfolds = 5)
    return(function(Xte) as.numeric(predict(fit, newx = Xte, s = "lambda.min", type = "response")))
  }
  if (idx == 5L) {
    fit <- randomForest(x = Xtr, y = factor(dtr, levels = c(0, 1)),
                        ntree = 500, nodesize = 20)
    return(function(Xte) as.numeric(predict(fit, newdata = Xte, type = "prob")[, "1"]))
  }
  if (idx == 6L) {
    sk_nn <- import("sklearn.neural_network")
    np    <- import("numpy")
    nn <- sk_nn$MLPClassifier(
      hidden_layer_sizes = tuple(50L, 50L, 50L, 50L),
      alpha = 0, max_iter = 200L, random_state = 720L)
    nn$fit(np$asarray(Xtr, dtype = "float64"),
           np$asarray(dtr, dtype = "int64"))
    return(function(Xte) {
      probs <- nn$predict_proba(np$asarray(Xte, dtype = "float64"))
      as.numeric(probs[, 2])
    })
  }
  stop("bad idx ", idx)
}


# ---------------------------------------------------------------------------
# Cross-fit one rep (LATE: 5 nuisance moments)
# ---------------------------------------------------------------------------

cross_fit_one_rep <- function(rep, X_raw, X_poly, X_inter, Y, Dval, Zval) {
  set.seed(SEED_BASE + rep)
  n <- length(Y)
  fold_id <- sample(rep(1:N_FOLDS, length.out = n))

  Xs <- list(raw = X_raw, poly = X_poly, inter = X_inter)
  pred_y0 <- matrix(NA_real_, n, 6)
  pred_y1 <- matrix(NA_real_, n, 6)
  pred_d0 <- matrix(NA_real_, n, 6)
  pred_d1 <- matrix(NA_real_, n, 6)
  pred_z  <- matrix(NA_real_, n, 6)

  t0 <- Sys.time()
  for (k in 1:N_FOLDS) {
    tr <- which(fold_id != k); te <- which(fold_id == k)
    Y_tr <- Y[tr]; D_tr <- Dval[tr]; Z_tr <- Zval[tr]
    for (i in 1:6) {
      Xkind <- LEARNER_X_KIND[i]
      Xtr <- Xs[[Xkind]][tr, , drop = FALSE]
      Xte <- Xs[[Xkind]][te, , drop = FALSE]
      # Y conditional on Z arm
      for (arm in c(0, 1)) {
        sel <- Z_tr == arm
        f <- fit_reg(i, Xtr[sel, , drop = FALSE], Y_tr[sel])
        if (arm == 0) pred_y0[te, i] <- f(Xte) else pred_y1[te, i] <- f(Xte)
      }
      # D conditional on Z arm.
      # Z=0 obs have D=0 always (no defiers in 401(k)); skip the fit and
      # set predictions to the empirical mean (= 0).
      for (arm in c(0, 1)) {
        sel <- Z_tr == arm
        D_arm <- D_tr[sel]
        if (length(unique(D_arm)) < 2) {
          pred_d_val <- mean(D_arm)
          if (arm == 0) pred_d0[te, i] <- pred_d_val
          else          pred_d1[te, i] <- pred_d_val
        } else {
          f <- fit_clf(i, Xtr[sel, , drop = FALSE], D_arm)
          if (arm == 0) pred_d0[te, i] <- f(Xte)
          else          pred_d1[te, i] <- f(Xte)
        }
      }
      # Z propensity
      f <- fit_clf(i, Xtr, Z_tr)
      pred_z[te, i] <- f(Xte)
    }
  }
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  rep %2d: %6.1fs\n", rep, elapsed))

  list(rep = rep, fold_id = fold_id,
       pred_y0 = pred_y0, pred_y1 = pred_y1,
       pred_d0 = pred_d0, pred_d1 = pred_d1,
       pred_z = pred_z)
}


# ---------------------------------------------------------------------------
# LATE estimation
# ---------------------------------------------------------------------------

late_score <- function(Y, Dval, Zval, mY0, mY1, mD0, mD1, pZ) {
  pZ <- pmin(pmax(pZ, TRIM_LO), TRIM_HI)
  psi_Y <- mY1 - mY0 + Zval * (Y - mY1) / pZ - (1 - Zval) * (Y - mY0) / (1 - pZ)
  psi_D <- mD1 - mD0 + Zval * (Dval - mD1) / pZ - (1 - Zval) * (Dval - mD0) / (1 - pZ)
  list(psi_Y = psi_Y, psi_D = psi_D)
}

late_estimate <- function(psi_Y, psi_D) {
  EY <- mean(psi_Y); ED <- mean(psi_D)
  beta <- EY / ED
  IF <- (psi_Y - beta * psi_D) / ED
  c(b = beta, se = sd(IF) / sqrt(length(IF)))
}

nnls_normalize <- function(target, P) {
  fit <- tryCatch(nnls::nnls(P, target), error = function(e) NULL)
  w <- if (!is.null(fit)) fit$x else rep(1 / ncol(P), ncol(P))
  if (sum(w) > 0) w <- w / sum(w) else w <- rep(1 / length(w), length(w))
  w
}

shortstack_predictions <- function(Y, Dval, Zval, pred_y0, pred_y1,
                                   pred_d0, pred_d1, pred_z) {
  m0_obs <- pred_y0[Zval == 0, , drop = FALSE]
  m1_obs <- pred_y1[Zval == 1, , drop = FALSE]
  w_y0 <- nnls_normalize(Y[Zval == 0], m0_obs)
  w_y1 <- nnls_normalize(Y[Zval == 1], m1_obs)
  # D|Z=0 trivial; just use mean of learner predictions.
  w_d0 <- rep(1 / 6, 6)
  m_d1_obs <- pred_d1[Zval == 1, , drop = FALSE]
  w_d1 <- nnls_normalize(as.numeric(Dval[Zval == 1]), m_d1_obs)
  w_z  <- nnls_normalize(as.numeric(Zval), pred_z)
  list(
    mY0 = as.numeric(pred_y0 %*% w_y0),
    mY1 = as.numeric(pred_y1 %*% w_y1),
    mD0 = as.numeric(pred_d0 %*% w_d0),
    mD1 = as.numeric(pred_d1 %*% w_d1),
    pZ  = as.numeric(pred_z %*% w_z)
  )
}

best_picks <- function(Y, Dval, Zval, pred_y0, pred_y1, pred_d1, pred_z) {
  mse_y0 <- colMeans((Y[Zval == 0] - pred_y0[Zval == 0, , drop = FALSE])^2)
  mse_y1 <- colMeans((Y[Zval == 1] - pred_y1[Zval == 1, , drop = FALSE])^2)
  mse_d1 <- colMeans((Dval[Zval == 1] - pred_d1[Zval == 1, , drop = FALSE])^2)
  mse_z  <- colMeans((Zval - pred_z)^2)
  list(y0 = which.min(mse_y0), y1 = which.min(mse_y1),
       d1 = which.min(mse_d1), z = which.min(mse_z))
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cat("======================================================================\n")
cat("401(k) LATE -- R port of Stata Code/401kLATE.do\n")
cat("======================================================================\n")

dat <- prepare_data()
n <- length(dat$Y)
cat(sprintf("loaded sipp1991.dta: n = %d, D mean = %.4f, Z mean = %.4f\n",
            n, mean(dat$D), mean(dat$Z)))
cat(sprintf(paste0("\ncross-fitting %d reps x %d folds x 5 moments x 6 ",
                   "learners (workers = %d)...\n"),
            N_REPS, N_FOLDS, N_PARALLEL_REPS))

plan(multisession, workers = N_PARALLEL_REPS)
t0 <- Sys.time()
results <- future_lapply(seq_len(N_REPS), function(rep) {
  source(file.path(codedir, "setup_reticulate.R"))
  suppressPackageStartupMessages({
    library(glmnet); library(randomForest); library(reticulate); library(nnls)
  })
  cross_fit_one_rep(rep, dat$X_raw, dat$X_poly, dat$X_inter,
                    dat$Y, dat$D, dat$Z)
}, future.seed = TRUE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("cross-fit total wall: %.1fs\n", elapsed))

rows <- vector("list", N_REPS)
cat("\nPer-rep estimates (parallels Stata `ddml estimate, robust`):\n")
cat(sprintf("  %3s  %10s %9s  %10s %9s\n",
            "rep", "best b", "best SE", "stack b", "stack SE"))
for (res in results) {
  rep <- res$rep
  bp <- best_picks(dat$Y, dat$D, dat$Z, res$pred_y0, res$pred_y1,
                   res$pred_d1, res$pred_z)
  scs <- late_score(dat$Y, dat$D, dat$Z,
                    res$pred_y0[, bp$y0], res$pred_y1[, bp$y1],
                    res$pred_d0[, 1],  # any (D|Z=0 trivial)
                    res$pred_d1[, bp$d1],
                    res$pred_z[, bp$z])
  best_est <- late_estimate(scs$psi_Y, scs$psi_D)
  ss <- shortstack_predictions(dat$Y, dat$D, dat$Z,
                                res$pred_y0, res$pred_y1,
                                res$pred_d0, res$pred_d1, res$pred_z)
  scs_ss <- late_score(dat$Y, dat$D, dat$Z,
                       ss$mY0, ss$mY1, ss$mD0, ss$mD1, ss$pZ)
  ss_est <- late_estimate(scs_ss$psi_Y, scs_ss$psi_D)
  rows[[rep]] <- data.frame(
    rep = rep,
    best_b = best_est["b"], best_se = best_est["se"],
    stack_b = ss_est["b"],  stack_se = ss_est["se"]
  )
  cat(sprintf("  %3d  %10.1f %9.1f  %10.1f %9.1f\n",
              rep, best_est["b"], best_est["se"],
              ss_est["b"], ss_est["se"]))
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

# Save predictions
out <- as.data.frame(dat$df)
for (res in results) {
  rep <- res$rep
  out[[paste0("m0_fid_", rep)]] <- res$fold_id
  for (i in seq_along(Y_LEARNER_NAMES)) {
    out[[paste0(Y_LEARNER_NAMES[i], "0_", rep)]] <- res$pred_y0[, i]
    out[[paste0(Y_LEARNER_NAMES[i], "1_", rep)]] <- res$pred_y1[, i]
  }
  for (i in seq_along(D_LEARNER_NAMES)) {
    out[[paste0(D_LEARNER_NAMES[i], "0_", rep)]] <- res$pred_d0[, i]
    out[[paste0(D_LEARNER_NAMES[i], "1_", rep)]] <- res$pred_d1[, i]
  }
  for (i in seq_along(Z_LEARNER_NAMES)) {
    out[[paste0(Z_LEARNER_NAMES[i], "_", rep)]] <- res$pred_z[, i]
  }
  ss <- shortstack_predictions(dat$Y, dat$D, dat$Z,
                                res$pred_y0, res$pred_y1,
                                res$pred_d0, res$pred_d1, res$pred_z)
  out[[paste0("Y_net_tfa_ss0_", rep)]] <- ss$mY0
  out[[paste0("Y_net_tfa_ss1_", rep)]] <- ss$mY1
  out[[paste0("D_p401_ss0_", rep)]] <- ss$mD0
  out[[paste0("D_p401_ss1_", rep)]] <- ss$mD1
  out[[paste0("Z_e401_ss_", rep)]] <- ss$pZ
}
write.csv(out, OUT_FILE, row.names = FALSE)
cat(sprintf("saved cross-fit predictions to %s (%d obs x %d vars)\n",
            OUT_FILE, nrow(out), ncol(out)))
