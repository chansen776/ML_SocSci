suppressPackageStartupMessages({
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Install reticulate first: install.packages('reticulate')")
  }
  library(reticulate)
})

# Course Python env (Anaconda): mlssshort. Python 3.12.x.
# Search a few standard install locations so this works on multiple machines
# without editing. Override by setting MLSSSHORT_PY env var.
conda_env <- "mlssshort"
candidates <- c(
  Sys.getenv("MLSSSHORT_PY", unset = NA_character_),
  file.path(Sys.getenv("USERPROFILE", unset = Sys.getenv("HOME")),
            "AppData", "Local", "anaconda3", "envs", conda_env, "python.exe"),
  file.path("C:/ProgramData/anaconda3/envs", conda_env, "python.exe"),
  file.path(Sys.getenv("USERPROFILE", unset = Sys.getenv("HOME")),
            ".conda", "envs", conda_env, "python.exe"),
  file.path(Sys.getenv("USERPROFILE", unset = Sys.getenv("HOME")),
            "anaconda3", "envs", conda_env, "python.exe"),
  file.path(Sys.getenv("USERPROFILE", unset = Sys.getenv("HOME")),
            "miniconda3", "envs", conda_env, "python.exe")
)
candidates <- candidates[!is.na(candidates)]
hit <- candidates[file.exists(candidates)]
if (length(hit) == 0L) {
  stop("Could not find mlssshort Python executable. Tried:\n  ",
       paste(candidates, collapse = "\n  "),
       "\nSet MLSSSHORT_PY env var to override.")
}
py_exe <- normalizePath(hit[[1L]], winslash = "/", mustWork = TRUE)

Sys.setenv(RETICULATE_PYTHON = py_exe)
use_python(py_exe, required = TRUE)

cfg <- py_config()
message("reticulate pointing at: ", cfg$python)
message("Python version: ", format(cfg$version))
invisible(cfg)
