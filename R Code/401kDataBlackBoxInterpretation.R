# 401(k) data: black-box interpretation -- Random Forest
# Permutation VI, partial dependence, and SHAP for the same RF.

library(randomForest)
library(xtable)
library(treeshap)
library(shapviz)

# Switch to save plots; relative to this script's directory.
dpl     = FALSE
plotdir = "../Slides/figures"

# Suppress stray Rplots.pdf when run noninteractively (e.g. Rscript --vanilla).
if (!interactive()) pdf(NULL)

# Read data
data401k = read.table("../Data/restatw.dat", header = TRUE)

# Variables we care about: net total financial assets (net_tfa),
# 401(k) eligibility (e401), age, income (inc), years schooling (educ),
# family size (fsize), marital status (marr), two-earner household
# (twoearn), defined-benefit pension (db), private retirement account
# ownership (pira), home ownership (hown).
data401k = subset(data401k, select = c(net_tfa, e401, age, inc, educ,
                                       fsize, marr, twoearn, db, pira, hown))

# Set up training and testing samples
set.seed(8261977)
ntrain = 8000
tr     = sample(1:nrow(data401k), ntrain)
train  = data401k[tr, ]
test   = data401k[-tr, ]

# Random forest: nodesize = 60
rForest.60 = randomForest(net_tfa ~ ., data = train, nodesize = 60)


#########################################
## Permutation variable importance

var.imp <- function(pred.model, newY, newX, B = 100) {
  e0 = sum((newY - predict(pred.model, newX))^2)
  n  = nrow(newX); p = ncol(newX)
  eB = matrix(0, B, p)
  for (b in 1:B) {
    for (j in 1:p) {
      newXb = newX
      newXb[, j] = newX[sample(1:n), j]
      eB[b, j]  = sum((newY - predict(pred.model, newXb))^2)
    }
  }
  list(e0 = e0, eB = eB / e0, VI = colMeans(eB / e0))
}

VI.rf = var.imp(rForest.60, newY = test[, 1], newX = test[, -1], B = 100)
colnames(VI.rf$eB) = colnames(test[, -1])

VI.table = matrix(VI.rf$VI[order(-VI.rf$VI)], ncol = 1)
rownames(VI.table) = colnames(test[, -1])[order(-VI.rf$VI)]
colnames(VI.table) = "VI"
print(xtable(VI.table, digits = c(0, 3)))

if (dpl) {
  png(file = file.path(plotdir, "VIBox.png"), height = 640, width = 800)
  boxplot(VI.rf$eB[, order(-VI.rf$VI)])
  dev.off()
}


#########################################
## Partial dependence

partial.dep <- function(pred.model, eval.points, var.ind, dataX) {
  n = nrow(dataX); s = length(eval.points)
  h = numeric(s)
  for (b in 1:s) {
    newXb = dataX
    newXb[, var.ind] = rep(eval.points[b], n)
    h[b] = mean(predict(pred.model, newXb))
  }
  h
}

# Income (continuous): $0 to $250k on a $10k grid.
inc.vals = seq(0, 250000, 10000)
h.inc    = partial.dep(rForest.60, eval.points = inc.vals,
                       var.ind = which(colnames(data401k) == "inc"),
                       dataX = data401k)

if (dpl) {
  png(file = file.path(plotdir, "PDInc.png"), height = 640, width = 800)
  plot(inc.vals, h.inc, xlab = "Income",
       ylab = "Average Predicted Net Assets",
       type = "l", col = "blue")
  dev.off()
}

# Binary: pira and 401(k) eligibility.
h.pira = partial.dep(rForest.60, eval.points = c(0, 1),
                     var.ind = which(colnames(data401k) == "pira"),
                     dataX = data401k)
h.elig = partial.dep(rForest.60, eval.points = c(0, 1),
                     var.ind = which(colnames(data401k) == "e401"),
                     dataX = data401k)

cat(sprintf("PD pira = 0:  %8.0f    pira = 1:  %8.0f\n", h.pira[1], h.pira[2]))
cat(sprintf("PD e401 = 0:  %8.0f    e401 = 1:  %8.0f\n", h.elig[1], h.elig[2]))


#########################################
## SHAP via treeshap (exact TreeSHAP for tree ensembles)

# Convert the randomForest object into the treeshap unified format,
# then compute SHAP values on the held-out test set. Baseline E[g(X)]
# isn't returned by treeshap for randomForest -- compute it from train.
rf.unified = randomForest.unify(rForest.60, train[, -1])
shap.rf    = treeshap(rf.unified, test[, -1], verbose = FALSE)
baseline   = mean(predict(rForest.60, train[, -1]))
sv         = shapviz(shap.rf, baseline = baseline)

# Sanity check: g(x_i) = E[g(X)] + sum_j phi_j(x_i)
recon = baseline + sum(shap.rf$shaps[1, ])
cat(sprintf("\nbaseline E[g(X)]: %.4f\n", baseline))
cat(sprintf("SHAP check (obs 1):  baseline + sum(phi) = %.4f   model pred = %.4f\n",
            recon, predict(rForest.60, test[1, -1])))

# Mean |SHAP| importance ranking (parallels Python's mean(|phi_j|)).
mean.abs.shap = sort(colMeans(abs(shap.rf$shaps)), decreasing = TRUE)
SHAP.table    = matrix(mean.abs.shap, ncol = 1)
rownames(SHAP.table) = names(mean.abs.shap)
colnames(SHAP.table) = "mean(|SHAP|)"
print(xtable(SHAP.table, digits = c(0, 1)))

# Plots: waterfall (one obs), beeswarm (full test), and bar (mean |SHAP|).
if (dpl) {
  ggplot2::ggsave(file.path(plotdir, "401k_shap_waterfall_R.pdf"),
                  sv_waterfall(sv, row_id = 1, max_display = 10),
                  width = 7, height = 4.5)
  ggplot2::ggsave(file.path(plotdir, "401k_shap_beeswarm_R.pdf"),
                  sv_importance(sv, kind = "beeswarm", max_display = 10),
                  width = 7, height = 4.5)
  ggplot2::ggsave(file.path(plotdir, "401k_shap_importance_R.pdf"),
                  sv_importance(sv, kind = "bar", max_display = 10),
                  width = 7, height = 4.5)
}
