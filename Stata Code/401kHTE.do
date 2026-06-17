*** 401(k) Heterogeneous Treatment Effects -- Day-2 Section 11 (Stata) *********
*
* The modern HTE toolkit (doubly-robust learner, out-of-sample model selection,
* RATE/uplift, policy trees) is not native to Stata. As with the Section-4
* simulations, the idiomatic way to reach these tools from Stata is its Python
* integration (the `mlssshort' environment; see PLAN.md Section 8). To keep a
* single source of truth, this do-file:
*   (1) loads the data and reports the naive (no-controls) ATE in native Stata;
*   (2) runs the canonical pipeline "Python code/401kHTE.py" in-process through
*       Stata's Python, reproducing every frame -- cross-fit DR signal, the
*       pre-specified GATES/BLP summaries (full sample, 95%), and the flexible
*       train/validation/test analysis (DR-loss model selection over four
*       learners, calibration + heterogeneity test, TOC/RATE, policy tree, and
*       the out-of-sample DR policy-value inference) -- writing the same
*       Slides/figures/HTE_401k_*.png figures (bit-identical to a direct Python run).
*
* Three-language parity: Python is canonical; this do-file is the Stata route to
* the identical analysis, and "R Code/401kHTE.R" is a native-R teaching parallel
* (close, not bit-identical, since it uses its own nuisance learners). Run from
* the "Stata Code" folder:  do 401kHTE.do
*******************************************************************************

capture log close
log using "401kHTE.txt", text replace

clear all
set more off

* ---- (1) Stata-native: load data + naive eligibility ATE -------------------
import delimited using "../Data/restatw.dat", delimiter(whitespace) ///
    varnames(1) clear
quietly count
display as text "loaded restatw.dat: N = " as result r(N)
* Naive ATE of 401(k) eligibility on net financial assets (no controls):
regress net_tfa e401, robust

* ---- (2) Drive the HTE pipeline through Stata's Python integration ---------
* python query   // (uncomment to confirm Stata points at the mlssshort exe)
python:
import importlib.util
from pathlib import Path

# Locate the canonical pipeline whether Stata's CWD is the repo root or the
# "Stata Code" subfolder.
cands = [Path.cwd() / "Python code" / "401kHTE.py",
         Path.cwd().parent / "Python code" / "401kHTE.py"]
src = next((c for c in cands if c.exists()), None)
if src is None:
    raise FileNotFoundError("Could not find 'Python code/401kHTE.py' from "
                            f"Stata CWD = {Path.cwd()}")

spec = importlib.util.spec_from_file_location("hte401k", str(src.resolve()))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # runs imports; main() guarded by __main__
mod.main()                     # reproduces every frame + writes the figures
end

log close
