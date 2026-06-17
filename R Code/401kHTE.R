# =====================================================================
# 401(k) Heterogeneous Treatment Effects -- Day-2 Section 11 (R parallel)
#
# R companion to "Python code/401kHTE.py" (the canonical source). Mirrors the
# same pipeline as a cross-language teaching parallel:
#   * cross-fit nuisances (m, g0, g1) -> doubly-robust signal Gamma = Y(eta)
#   * pre-specified summaries (GATES by income, cubic-in-log-income BLP) on the
#     FULL sample with 95% bands
#   * flexible CATE estimation / validation / policy on one 60/20/20
#     train / validation / test split, reporting 68% (+/- 1 s.e.) bands:
#       - model selection by out-of-sample DR loss over four learners
#         (boosting d4 early-stop, linear age+income, random forest mtry=p/3,
#          regularized linear ridge)
#       - calibration + heterogeneity test (validation, all learners)
#       - TOC / RATE (validation, selected learner highlighted)
#       - policy tree (train) + out-of-sample policy value (test)
# No causal forest here (the canonical pipeline dropped it). Nuisances use
# ranger regression/probability forests, so numbers are close to -- not
# bit-identical to -- the Python stacked-ensemble version. Figures carry a
# _R suffix; the deck uses the unsuffixed Python figures.
#
# Run:  Rscript --vanilla "R Code/401kHTE.R"
# =====================================================================

suppressMessages({
  library(ranger)
  library(xgboost)
  library(glmnet)
  library(rpart)
})

SEED <- 731
Z95 <- 1.96; Z68 <- 1.0; Z68_1S <- qnorm(0.68)   # critical values

# ---- portable paths -------------------------------------------------
cand_paths <- c(file.path("..", "Data", "restatw.dat"),
                file.path("Data", "restatw.dat"), "restatw.dat")
dpath <- cand_paths[file.exists(cand_paths)][1]
if (is.na(dpath)) stop("restatw.dat not found")
root <- normalizePath(file.path(dirname(dpath), ".."))
figdir <- file.path(root, "Slides", "figures")
dir.create(figdir, showWarnings = FALSE, recursive = TRUE)
fig <- function(nm) file.path(figdir, nm)

df <- read.table(dpath, header = TRUE)
covars <- c("age", "inc", "educ", "fsize", "marr", "male", "twoearn",
            "db", "pira", "nohs", "hs", "smcol", "col", "hown")
Y <- df$net_tfa
W <- df$e401
X <- as.matrix(df[, covars])
inc <- df$inc
P <- length(covars)
LC <- c("Boosting (d4, early-stop)" = "#1f77b4", "Linear (age, income)" = "#ff7f0e",
        "Random forest (mtry=p/3)" = "#2ca02c", "Regularized linear (Ridge)" = "#d62728")
cat(sprintf("loaded restatw.dat: n=%d, P(W=1)=%.3f\n", nrow(df), mean(W)))

# ---- cross-fit nuisances -> doubly-robust signal --------------------
crossfit <- function(X, Y, W, K = 5, seed = SEED) {
  n <- length(Y); set.seed(seed); fold <- sample(rep(1:K, length.out = n))
  m <- g0 <- g1 <- numeric(n)
  for (k in 1:K) {
    tr <- fold != k; te <- fold == k
    pf <- ranger(x = X[tr, ], y = factor(W[tr]), probability = TRUE,
                 num.trees = 500, seed = seed)
    m[te] <- predict(pf, X[te, ])$predictions[, "1"]
    r0 <- ranger(x = X[tr, ][W[tr] == 0, ], y = Y[tr][W[tr] == 0],
                 num.trees = 500, seed = seed)
    r1 <- ranger(x = X[tr, ][W[tr] == 1, ], y = Y[tr][W[tr] == 1],
                 num.trees = 500, seed = seed)
    g0[te] <- predict(r0, X[te, ])$predictions
    g1[te] <- predict(r1, X[te, ])$predictions
  }
  m <- pmin(pmax(m, 0.02), 0.98)
  H <- W / m - (1 - W) / (1 - m)
  gW <- W * g1 + (1 - W) * g0
  H * (Y - gW) + g1 - g0
}

Gamma <- crossfit(X, Y, W)
theta_full <- mean(Gamma)
cat(sprintf("cross-fitted DR-ATE = mean Gamma = %.0f (se %.0f)\n",
            theta_full, sd(Gamma) / sqrt(length(Gamma))))

# ---- robust sup-t critical value (eigendecomposition) ---------------
supt_crit <- function(cov, alpha = 0.05, one_sided = FALSE, ndraw = 20000) {
  d <- nrow(cov); se <- sqrt(pmax(diag(cov), 1e-24))
  e <- eigen((cov + t(cov)) / 2, symmetric = TRUE)
  L <- e$vectors %*% diag(sqrt(pmax(e$values, 0)))
  draws <- (L %*% matrix(rnorm(d * ndraw), d, ndraw)) / se
  stat <- if (one_sided) apply(draws, 2, max) else apply(draws, 2, function(x) max(abs(x)))
  as.numeric(quantile(stat, 1 - alpha))
}

# =====================================================================
# Pre-specified summaries (FULL sample, 95%): GATES + BLP cubic
# =====================================================================
qn <- cut(inc, quantile(inc, 0:5 / 5), labels = FALSE, include.lowest = TRUE)
gates <- sapply(1:5, function(k) {
  g <- Gamma[qn == k]; c(mean(g), sd(g) / sqrt(length(g)))
})
cat("\nGATES by income quintile (95%):\n")
for (k in 1:5) cat(sprintf("  Q%d: %8.0f (se %5.0f)\n", k, gates[1, k], gates[2, k]))
png(fig("HTE_401k_gates_R.png"), 7.2, 4.6, units = "in", res = 200)
par(mar = c(4.2, 4.8, 1, 1), cex = 1.05)
plot(1:5, gates[1, ], ylim = range(gates[1, ] + 2 * gates[2, ], gates[1, ] - 2 * gates[2, ]),
     pch = 19, col = "#b3402f", xaxt = "n", xlab = "Income quintile",
     ylab = "GATE: effect on net financial assets ($)", bty = "l")
axis(1, 1:5, paste0("Q", 1:5)); abline(h = theta_full, lty = 2, col = "grey40")
arrows(1:5, gates[1, ] - Z95 * gates[2, ], 1:5, gates[1, ] + Z95 * gates[2, ],
       angle = 90, code = 3, length = .05, col = "#b3402f", lwd = 2)
dev.off()

lc <- as.numeric(scale(log(pmax(inc, 1))))
lc_ctr <- mean(log(pmax(inc, 1))); lc_scl <- sd(log(pmax(inc, 1)))
fitp <- lm(G ~ poly(lc, 3), data = data.frame(G = Gamma, lc = lc))
gi <- seq(quantile(inc, .02), quantile(inc, .98), length.out = 200)
gl <- (log(gi) - lc_ctr) / lc_scl
newp <- predict(fitp, newdata = data.frame(lc = gl), se.fit = TRUE)
png(fig("HTE_401k_cate_income_R.png"), 7.2, 4.6, units = "in", res = 200)
par(mar = c(4.2, 4.8, 1, 1), cex = 1.05)
plot(gi / 1000, newp$fit, type = "l", col = "#1f5fae", lwd = 2.5,
     ylim = range(newp$fit + 2.5 * newp$se.fit, newp$fit - 2.5 * newp$se.fit, 0),
     xlab = "Income ($000)", ylab = "CATE: effect on net financial assets ($)", bty = "l")
polygon(c(gi, rev(gi)) / 1000,
        c(newp$fit - Z95 * newp$se.fit, rev(newp$fit + Z95 * newp$se.fit)),
        col = adjustcolor("#1f5fae", .15), border = NA)
abline(h = 0, col = "grey60"); dev.off()

# =====================================================================
# Flexible modeling / policy: 60/20/20 split, 68% bands
# =====================================================================
set.seed(SEED)
idx <- sample(length(Y)); n <- length(Y)
tr <- idx[1:floor(.6 * n)]
va <- idx[(floor(.6 * n) + 1):floor(.8 * n)]
te <- idx[(floor(.8 * n) + 1):n]
cat(sprintf("\nsplit: train=%d, val=%d, test=%d\n", length(tr), length(va), length(te)))

# four candidate CATE learners: each returns a predict closure -------------
fit_candidates <- function(Xtr, Gtr) {
  list(
    "Boosting (d4, early-stop)" = local({
      set.seed(SEED); iv <- sample(nrow(Xtr), floor(.15 * nrow(Xtr)))
      bst <- xgb.train(list(max_depth = 4, eta = 0.03, objective = "reg:squarederror"),
                       xgb.DMatrix(Xtr[-iv, ], label = Gtr[-iv]), nrounds = 2000,
                       watchlist = list(val = xgb.DMatrix(Xtr[iv, ], label = Gtr[iv])),
                       early_stopping_rounds = 25, verbose = 0)
      function(Xn) predict(bst, xgb.DMatrix(Xn))
    }),
    "Linear (age, income)" = local({
      m <- lm(G ~ age + inc, data.frame(G = Gtr, age = Xtr[, "age"], inc = Xtr[, "inc"]))
      function(Xn) as.numeric(predict(m, data.frame(age = Xn[, "age"], inc = Xn[, "inc"])))
    }),
    "Random forest (mtry=p/3)" = local({
      m <- ranger(x = Xtr, y = Gtr, num.trees = 1000, min.node.size = 120,
                  mtry = floor(P / 3), seed = SEED)
      function(Xn) predict(m, Xn)$predictions
    }),
    "Regularized linear (Ridge)" = local({
      set.seed(SEED); m <- cv.glmnet(Xtr, Gtr, alpha = 0)
      function(Xn) as.numeric(predict(m, Xn, s = "lambda.min"))
    })
  )
}
preds <- fit_candidates(X[tr, ], Gamma[tr])
nms <- names(preds)

# ---- slide 73: illustration -- distil the RF candidate (train) ------
tau_rf_tr <- preds[["Random forest (mtry=p/3)"]](X[tr, ])
distill <- rpart(tau ~ ., data = data.frame(X[tr, ], tau = tau_rf_tr),
                 control = rpart.control(maxdepth = 3, cp = 0, minbucket = 300))
png(fig("HTE_401k_distill_tree_R.png"), 11, 5.5, units = "in", res = 200)
par(mar = c(1, 1, 2, 1)); plot(distill, uniform = TRUE, margin = .08)
text(distill, use.n = FALSE, cex = .8, digits = 0)
title("Distillation tree for the random-forest DR-learner CATE ($)"); dev.off()

# ---- slide 74: model selection by DR loss on VALIDATION -------------
Gva <- Gamma[va]; const <- mean(Gamma[tr]); loss_c <- (Gva - const)^2; denom <- mean(loss_c)
sel <- data.frame(model = nms, score = NA, se = NA)
for (i in seq_along(nms)) {
  tau <- preds[[nms[i]]](X[va, ])
  a <- loss_c - (Gva - tau)^2                # per-obs improvement over constant
  R <- mean(a) / denom
  psi <- (a - R * loss_c) / denom            # ratio delta-method influence
  sel$score[i] <- R; sel$se[i] <- sd(psi) / sqrt(length(va))
}
best <- sel$model[which.max(sel$score)]
cat("\nModel selection on validation (normalized DR-loss improvement, 68%):\n")
print(within(sel, { score <- round(score, 3); se <- round(se, 3) }), row.names = FALSE)
cat(sprintf("  selected (best non-constant) model: %s\n", best))
png(fig("HTE_401k_model_comparison_R.png"), 8.2, 4.2, units = "in", res = 200)
par(mar = c(4.4, 11, 1.5, 1), cex = 1.05)
yp <- nrow(sel):1
plot(sel$score, yp, xlim = range(0, sel$score + Z68 * sel$se, sel$score - Z68 * sel$se),
     pch = 19, col = LC[sel$model], yaxt = "n", bty = "l",
     xlab = "Validation DR-loss improvement over constant (>0 better; +/-1 s.e.)", ylab = "")
axis(2, yp, sel$model, las = 1); abline(v = 0, lty = 2, col = "grey50")
arrows(sel$score - Z68 * sel$se, yp, sel$score + Z68 * sel$se, yp,
       angle = 90, code = 3, length = .05, col = LC[sel$model], lwd = 2); dev.off()

# ---- slide 75: calibration + heterogeneity test on VALIDATION -------
ols_slope <- function(x, y) {            # slope + se of y ~ x (HC-robust)
  X1 <- cbind(1, x); b <- solve(crossprod(X1), crossprod(X1, y))
  r <- y - X1 %*% b; V <- solve(crossprod(X1)) %*% crossprod(X1 * as.numeric(r)) %*% solve(crossprod(X1))
  c(slope = b[2], se = sqrt(V[2, 2]))
}
cat("\nHeterogeneity test on validation (slope of Gamma on tau_k; 68%=+/-1se):\n")
png(fig("HTE_401k_calibration_R.png"), 6.8, 5.6, units = "in", res = 200)
par(mar = c(4.6, 4.8, 1, 1), cex = 1.05); first <- TRUE
allx <- ally <- c()
calib <- list()
for (nm in nms) {
  tau <- preds[[nm]](X[va, ])
  s <- ols_slope(tau - mean(tau), Gva)
  cat(sprintf("  %-26s slope %6.2f (se %5.2f)\n", nm, s[1], s[2]))
  qb <- cut(tau, quantile(tau, 0:4 / 4), labels = FALSE, include.lowest = TRUE)
  cg <- sapply(sort(unique(qb)), function(k)
    c(mean(tau[qb == k]), mean(Gva[qb == k]), sd(Gva[qb == k]) / sqrt(sum(qb == k))))
  calib[[nm]] <- cg
  allx <- c(allx, cg[1, ]); ally <- c(ally, cg[2, ] + Z68 * cg[3, ], cg[2, ] - Z68 * cg[3, ])
}
# scale each axis to its own data (predicted CATE vs. realized Y(eta) live on
# different ranges) so x is not stretched to the y-extent, leaving the panel empty.
xr <- range(allx) + c(-.05, .05) * diff(range(allx))
yr <- range(ally) + c(-.05, .05) * diff(range(ally))
plot(NA, xlim = xr, ylim = yr, bty = "l",
     xlab = "Mean predicted CATE in group ($)",
     ylab = "Average Gamma in CATE group (with 68% CI)")
abline(0, 1, lty = 2, col = "grey55")  # clipped to the plot box
for (nm in nms) {
  cg <- calib[[nm]]; col <- LC[nm]; full <- nm == best
  arrows(cg[1, ], cg[2, ] - Z68 * cg[3, ], cg[1, ], cg[2, ] + Z68 * cg[3, ],
         angle = 90, code = 3, length = .04, col = adjustcolor(col, ifelse(full, 1, .5)), lwd = ifelse(full, 2.2, 1))
  points(cg[1, ], cg[2, ], pch = 19, col = adjustcolor(col, ifelse(full, 1, .5)), cex = ifelse(full, 1.2, .9))
  if (full) lines(cg[1, ], cg[2, ], col = col, lwd = 2.2)
}
legend("topleft", bty = "n", cex = .85, legend = c(paste0(nms, ifelse(nms == best, " (selected)", "")), "perfect calibration"),
       col = c(LC[nms], "grey55"), pch = c(rep(19, length(nms)), NA), lty = c(rep(NA, length(nms)), 2))
dev.off()

# ---- slide 76: TOC / RATE on VALIDATION (selected highlighted) ------
theta0 <- mean(Gva); qgrid <- seq(.05, .95, length.out = 50); G <- length(qgrid); dq <- qgrid[2] - qgrid[1]
cat("\nRATE / AUTOC on validation (one-sided 68% lower bound):\n")
toc_store <- list(); autoc_lab <- c()
for (nm in nms) {
  tau <- preds[[nm]](X[va, ])
  toc <- numeric(G); Pi <- matrix(0, length(va), G)
  for (j in 1:G) {
    thr <- quantile(tau, 1 - qgrid[j]); I <- as.numeric(tau >= thr); pic <- mean(I)
    toc[j] <- sum(Gva * I) / sum(I) - theta0
    Pi[, j] <- (Gva - theta0) * (I / pic - 1)
  }
  V <- crossprod(Pi) / length(va)
  autoc <- sum(toc * dq); se_autoc <- sd(Pi %*% rep(dq, G)) / sqrt(length(va))
  toc_store[[nm]] <- list(toc = toc, V = V)
  autoc_lab[nm] <- sprintf("%s: AUTOC %.0f (+/-%.0f)%s", nm, autoc, se_autoc,
                           ifelse(nm == best, " [sel]", ""))
  cat(sprintf("  %-26s AUTOC %6.0f (se %5.0f)  lo68 %6.0f\n",
              nm, autoc, se_autoc, autoc - Z68_1S * se_autoc))
}
png(fig("HTE_401k_toc_R.png"), 7.6, 4.8, units = "in", res = 200)
par(mar = c(4.4, 4.8, 1, 1), cex = 1.05)
yl <- range(sapply(toc_store, function(s) s$toc))
plot(NA, xlim = c(0, 1), ylim = yl, bty = "l",
     xlab = "Fraction treated (prioritized by tau-hat)", ylab = "TOC: GATE(top q) - ATE ($)")
abline(h = 0, col = "grey55")
for (nm in nms) {
  s <- toc_store[[nm]]; full <- nm == best
  lines(qgrid, s$toc, col = adjustcolor(LC[nm], ifelse(full, 1, .4)), lwd = ifelse(full, 2.6, 1.4))
  if (full) {
    cc <- supt_crit(s$V, alpha = 0.32, one_sided = TRUE)
    lines(qgrid, s$toc - cc * sqrt(diag(s$V) / length(va)), col = LC[nm], lwd = 1, lty = 2)
  }
}
legend("topright", bty = "n", cex = .8, legend = autoc_lab, col = LC[nms], lwd = 2)
dev.off()

# ---- slide 77: policy tree on TRAIN (weighted classification) -------
dft <- data.frame(X[tr, ]); dft$y <- factor(Gamma[tr] > 0)
ptree <- rpart(y ~ ., data = dft, weights = abs(Gamma[tr]), method = "class",
               control = rpart.control(maxdepth = 3, minbucket = 300, cp = 0))
png(fig("HTE_401k_policy_tree_R.png"), 11, 5.5, units = "in", res = 200)
par(mar = c(1, 1, 2, 1)); plot(ptree, uniform = TRUE, margin = .08)
text(ptree, use.n = TRUE, cex = .8)
title("Policy tree (train): treat where sign(Gamma) positive, weight |Gamma|"); dev.off()
treat_tr <- predict(ptree, dft, type = "class") == "TRUE"
cat(sprintf("\nslide 77 policy tree: treats %.0f%% of training units\n", 100 * mean(treat_tr)))

# ---- slide 78: policy value on TEST ---------------------------------
Gte <- Gamma[te]; nt <- length(te)
tau_te <- preds[[best]](X[te, ]); thr50 <- median(tau_te)
set.seed(SEED + 2); rnd <- runif(nt) < .5
# budget policy tree: rank test units by the tree's treat-probability, fill 50%
pp <- predict(ptree, data.frame(X[te, ]), type = "prob")[, "TRUE"]
ord <- order(pp + runif(nt) * 1e-9, decreasing = TRUE)
pt_budget <- numeric(nt); pt_budget[ord[1:floor(.5 * nt)]] <- 1
pols <- list(
  "Treat all" = rep(1, nt),
  "Selected model: CATE>0" = as.numeric(tau_te > 0),
  "Selected model: top 50%" = as.numeric(tau_te >= thr50),
  "Policy tree" = as.numeric(predict(ptree, data.frame(X[te, ]), type = "class") == "TRUE"),
  "Policy tree: 50% budget" = pt_budget,
  "Random 50%" = as.numeric(rnd))
val <- t(sapply(pols, function(p) c(mean(p), mean(p * Gte), sd(p * Gte) / sqrt(nt))))
colnames(val) <- c("treated", "value", "se")
cat("\nPolicy value on TEST (gain over treating no one, 68%=+/-1se):\n")
for (nm in rownames(val))
  cat(sprintf("  %-26s %3.0f%% treated  value %7.0f (se %5.0f)\n",
              nm, 100 * val[nm, "treated"], val[nm, "value"], val[nm, "se"]))
ctr <- function(a, b) { d <- (pols[[a]] - pols[[b]]) * Gte
  sprintf("%.0f (se %.0f)", mean(d), sd(d) / sqrt(nt)) }
cat("  selected top-50% vs random-50%:", ctr("Selected model: top 50%", "Random 50%"), "\n")
cat("  policy-tree 50% budget vs random-50%:", ctr("Policy tree: 50% budget", "Random 50%"), "\n")
png(fig("HTE_401k_policy_value_R.png"), 8.2, 4.6, units = "in", res = 200)
par(mar = c(4.4, 12, 1, 1), cex = 1.05)
yp <- nrow(val):1
plot(val[, "value"], yp, xlim = range(0, val[, "value"] + 2 * val[, "se"]),
     pch = 19, col = "#1f5fae", yaxt = "n", bty = "l",
     xlab = "Policy value on test: gain over treating no one ($, +/-1 s.e.)", ylab = "")
axis(2, yp, sprintf("%s (%.0f%%)", rownames(val), 100 * val[, "treated"]), las = 1)
abline(v = 0, col = "grey60")
arrows(val[, "value"] - Z68 * val[, "se"], yp, val[, "value"] + Z68 * val[, "se"], yp,
       angle = 90, code = 3, length = .05, col = "#1f5fae", lwd = 2); dev.off()

cat("\nAll R parallel figures saved (HTE_401k_*_R.png).\n")
