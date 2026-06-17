# install_R_packages.R
# ---------------------------------------------------------------------------
# One-shot installer for the R packages used by the short course's R scripts.
# Run once from a fresh R install:
#
#     Rscript "R Code/install_R_packages.R"
#   or, inside an R session:
#     source("R Code/install_R_packages.R")
#
# Tested with R version 4.5.0 (2025-04-11 ucrt). The exact package versions the
# current decks' R code was run against are listed in the table below; this
# script installs the latest CRAN versions of anything missing (it does NOT
# force-downgrade already-installed packages). For strict version pinning use
# renv (see the note at the bottom).
#
# R also calls Python via reticulate, pointed at the `mlssshort` conda env — see
# R Code/setup_reticulate.R and "Note on Software for Lectures.txt".
#
# Tested versions (R 4.5.0, captured 2026-06-15):
#   glmnet        4.1.9       randomForest  4.7.1.2     ranger        0.17.0
#   xgboost       1.7.11.1    rpart         4.1.24      rpart.plot    3.1.2
#   plotmo        3.6.4       kknn          1.4.1       LowRankQP     1.0.6
#   nnls          1.6         hdi           0.1.10      glmnet/...    (above)
#   treeshap      0.4.0       shapviz       0.10.3      ggplot2       3.5.2
#   xtable        1.8.4       fastDummies   1.7.5       sandwich      3.1.1
#   BMisc         1.4.8       foreign       0.8.90      reticulate    1.42.0
#   future        1.58.0      future.apply  1.20.0
# ---------------------------------------------------------------------------

required <- c(
  # penalized regression / high-dimensional inference
  "glmnet", "hdi",
  # trees, forests, boosting
  "rpart", "rpart.plot", "plotmo", "randomForest", "ranger", "xgboost", "kknn",
  # SHAP / interpretation
  "treeshap", "shapviz",
  # short-stacking / DML plumbing
  "nnls", "LowRankQP", "future", "future.apply", "reticulate",
  # DiD / panel helpers
  "BMisc", "fastDummies", "sandwich",
  # data IO + tables + plotting
  "foreign", "xtable", "ggplot2"
)

installed <- rownames(installed.packages())
missing   <- setdiff(required, installed)

if (length(missing) == 0L) {
  cat("All required R packages are already installed.\n")
} else {
  cat("Installing missing packages:\n  ", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Report what is now present, with versions, for the record.
cat("\nInstalled versions:\n")
for (p in sort(required)) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING")
  cat(sprintf("  %-14s %s\n", p, v))
}

cat("\n", R.version.string, "\n", sep = "")

# ---------------------------------------------------------------------------
# Optional strict reproducibility with renv:
#   install.packages("renv"); renv::init()        # then commit renv.lock
#   renv::restore()                                # on another machine
# ---------------------------------------------------------------------------
