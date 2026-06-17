# run_all.R -- end-to-end runner for the CURRENT-deck R code.
#
# Runs every current R script used by the two decks, each in a FRESH Rscript
# process (so libraries / variables do not bleed across scripts), continuing
# past any single failure and printing a summary at the end.
#
# USAGE (from the "R Code" directory):
#     Rscript run_all.R                 # everything
#     Rscript run_all.R --skip-long     # skip the multi-minute simulations
#     Rscript run_all.R 401kHTE.R       # run only the named script(s)
#
# PREREQUISITES
#   * Packages: run install_R_packages.R once (R 4.5.0 tested).
#   * reticulate-using scripts (401kDataNN.R, 401kDataStacking.R, 401kATE.R,
#     401kLATE.R, InferenceOverviewSimExtended.R) need R pointed at the
#     mlssshort Python. Set RETICULATE_PYTHON to that python.exe, or rely on
#     setup_reticulate.R. This runner forwards RETICULATE_PYTHON to children.
#   * Several scripts are PARALLELS to the Stata canonical numbers; agreement
#     is close-not-identical by design (see Map of Code to Material.md).

args  <- commandArgs(trailingOnly = TRUE)
skip_long <- "--skip-long" %in% args
only      <- args[!startsWith(args, "--")]

# (file, is_long).  Order: Day-1 examples, then Day-2 / DML.
steps <- list(
  c("401kDataValidationIllustration.R", "0"),
  c("401kDataHDLMs.R",                  "0"),
  c("401kDataTrees.R",                  "0"),
  c("401kDataStacking.R",              "0"),
  c("401kDataNN.R",                    "0"),
  c("401kDataBlackBoxInterpretation.R","0"),
  c("RiboflavinExample.R",             "0"),
  c("HDLMSim.R",                       "0"),  # canonical N6HDLMSim figures
  c("InferenceOverviewSimExtended.R",  "1"),  # LONG
  c("401kATE.R",                       "1"),  # LONG
  c("401kLATE.R",                      "1"),  # LONG
  c("ExampleMWDiD.R",                  "1"),
  c("ExampleFEPLM.R",                  "0"),
  c("401kHTE.R",                       "0")
)

if (length(only) > 0) {
  steps <- Filter(function(s) s[[1]] %in% only, steps)
}

rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
failed <- character(0)

for (s in steps) {
  f <- s[[1]]; is_long <- s[[2]] == "1"
  if (skip_long && is_long) { cat(sprintf("--- SKIP (long): %s\n", f)); next }
  if (!file.exists(f))      { cat(sprintf("  !! missing: %s\n", f)); failed <- c(failed, f); next }
  cat(sprintf("\n%s\n>>> RUNNING: %s\n%s\n", strrep("=", 72), f, strrep("=", 72)))
  t0 <- Sys.time()
  rc <- system2(rscript, args = shQuote(f), stdout = "", stderr = "")
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (rc == 0) cat(sprintf(">>> OK: %s  (%.0fs)\n", f, dt))
  else { cat(sprintf(">>> FAILED: %s  rc=%d  (%.0fs)\n", f, rc, dt)); failed <- c(failed, sprintf("%s(rc=%d)", f, rc)) }
}

cat("\n", strrep("=", 72), "\n", sep = "")
if (length(failed) == 0) cat(">>> ALL OK.\n") else cat(">>> FAILURES:", paste(failed, collapse = ", "), "\n")
cat(strrep("=", 72), "\n", sep = "")
if (length(failed) > 0) quit(status = 1)
