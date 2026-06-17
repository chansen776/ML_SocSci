# High-Dimensional Linear Model: simulation illustration of the three PLM
# moment conditions under post-selection lasso with plug-in penalty.
#
# Companion to Day-2 deck section "Setup -> HDLM: Simulation Illustration".
#
# Design:
#   Y = alpha * D + 0.5*X1 + 1.0*X2 + N(0,1)
#   D = 1.0*X1 + 0.5*X2 + N(0,1)
#   X ~ N(0, I_p),  n = 200, p = 200, alpha_0 = 0.25
#
# Three moment conditions:
#   (1) Lasso of Y on (D, X) with D unpenalized -> post-OLS of Y on (D, X_sel)
#   (2) Lasso of D on X -> partial out -> alpha = <r_D, Y> / <r_D, D>
#   (3) Double selection: post-OLS of Y on (D, X_{sel(D~X) U sel(Y~X)})
#
# Run as: Rscript --vanilla "R Code/HDLMSim.R"

suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(xtable))

if (!interactive()) pdf(NULL)
# plotdir resolves whether script is launched from repo root or "R Code/".
plotdir <- if (dir.exists("Slides/figures")) {
  normalizePath("Slides/figures", mustWork = TRUE)
} else if (dir.exists("../Slides/figures")) {
  normalizePath("../Slides/figures", mustWork = TRUE)
} else {
  stop("Slides/figures not found from getwd()=", getwd())
}

set.seed(5192022)
nRep <- 1000
n <- 200
p <- 200
betam <- c(1, 0.5, rep(0, p - 2))
betay <- c(0.5, 1, rep(0, p - 2))
alpha0 <- 0.25

# Plug-in lambda from BCH (Belloni-Chernozhukov-Hansen).  Plug-in penalty
# tuned for sparse recovery; targets the population sup-score.
lambda.r <- 2.2 / sqrt(n) * qnorm(1 - (0.1 / log(n)) / (2 * p))

alphahat <- matrix(0, nRep, 3)
t0 <- Sys.time()
for (ii in seq_len(nRep)) {
  X <- matrix(rnorm(n * p), n)
  D <- X %*% betam + rnorm(n)
  Y <- D * alpha0 + X %*% betay + rnorm(n)

  # moment (1): lasso of Y on (D, X) with D unpenalized.
  res1 <- glmnet(cbind(D, X), Y, lambda = lambda.r,
                 penalty.factor = c(0, rep(1, p)))
  sel1 <- which(res1$beta[2:(p + 1)] != 0)
  alphahat[ii, 1] <- if (length(sel1) > 0) {
    coef(lm(Y ~ D + X[, sel1]))[2]
  } else {
    coef(lm(Y ~ D))[2]
  }

  # moment (2): lasso of D on X -> post-OLS -> partial out -> IV-style.
  res2 <- glmnet(X, D, lambda = lambda.r)
  sel2 <- which(res2$beta != 0)
  rD <- if (length(sel2) > 0) {
    D - predict(lm(D ~ X[, sel2]))
  } else {
    D - mean(D)
  }
  alphahat[ii, 2] <- as.numeric(crossprod(rD, Y) / crossprod(rD, D))

  # moment (3): double selection -> post-OLS on union.
  res3 <- glmnet(X, Y, lambda = lambda.r)
  use3 <- union(which(res2$beta != 0), which(res3$beta != 0))
  alphahat[ii, 3] <- if (length(use3) > 0) {
    coef(lm(Y ~ D + X[, use3]))[2]
  } else {
    as.numeric(crossprod(rD, Y - mean(Y)) / crossprod(rD, rD))
  }
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("nRep=%d: elapsed = %.1fs\n", nRep, elapsed))

tab <- matrix(0, 2, 3)
tab[1, ] <- colMeans(alphahat)
tab[2, ] <- apply(alphahat, 2, sd)
colnames(tab) <- c("Moment(1)", "Moment(2)", "Moment(3)")
rownames(tab) <- c("Mean", "Std. Dev.")
cat("Summary (true alpha = 0.25):\n")
print(round(tab, 3))
cat("\nLaTeX:\n")
print(xtable(tab, digits = 3), comment = FALSE)

# Save histogram figure (R is the canonical generator of N6HDLMSim.png).
# Use a common xlim that contains the truth (so abline at alpha0 is always
# inside the panel) and the bulk of all three histograms.
xlim_common <- range(c(alpha0, alphahat))
filenm <- file.path(plotdir, "N6HDLMSim.png")
png(file = filenm, height = 240, width = 720)
par(mfcol = c(1, 3))
for (j in 1:3) {
  hist(alphahat[, j],
       main = sprintf("Moment (%d)", j),
       xlab = "Coefficient", col = "cadetblue1", breaks = 25,
       xlim = xlim_common)
  abline(v = alpha0, lwd = 2, col = "red")
}
dev.off()
cat(sprintf("Saved figure: %s\n", filenm))

# Single-panel version of the Neyman-orthogonal moment (3) for the
# "Is NO enough?" comparison frame in section 4.
filenm_no <- file.path(plotdir, "N6HDLMSim_NO.png")
png(file = filenm_no, height = 360, width = 540)
par(mar = c(4.2, 4.0, 1.0, 1.0))
hist(alphahat[, 3],
     main = "",
     xlab = "Coefficient",
     col = "cadetblue1", breaks = 25,
     xlim = xlim_common)
abline(v = alpha0, lwd = 2, col = "red")
dev.off()
cat(sprintf("Saved figure: %s\n", filenm_no))
