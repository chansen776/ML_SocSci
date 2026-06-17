#!/usr/bin/env python
"""
run_all.py -- end-to-end runner for the CURRENT-deck Python code.

Runs every current Python script (.py) and notebook (.ipynb) used by the two
decks, in dependency order, continuing past any single failure and printing a
summary at the end.

USAGE (from the "Python code" directory, with the mlssshort env active):
    python run_all.py                 # everything except externally-gated steps
    python run_all.py --skip-long     # skip the multi-minute simulations
    python run_all.py --skip-notebooks
    python run_all.py --only 401kHTE.py InferenceOverviewSimExtended_plots.py
    python run_all.py --list

PREREQUISITES / CROSS-LANGUAGE NOTES
  * mlssshort env (Python 3.12.7). See ../environment.yml.
  * 401kATE_diagnostics.py reads ../Data/401kATE.dta, produced by the STATA
    401kATE.do run -- run Stata first, or that step will fail.
  * InferenceOverviewSimExtended_plots.py reads
    ../Data/InferenceOverviewSimExtended.csv (the canonical figure source);
    InferenceOverviewSimExtended.py (long) regenerates it.
  * DLfePLM_viz.py reads ../Data/DLfePLM_viz.csv, produced by the STATA
    DLfePLM_viz.do run.
  * xray_resnet_training.ipynb needs the external xray_data.pt (Dropbox) +
    torchvision weights; it is EXCLUDED by default (--include-xray to add it).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (filename, is_long, note).  Order = Day-1 examples, then Day-2 / DML.
STEPS = [
    # --- Day 1 (notebooks) ---
    ("401kDataValidationIllustration.ipynb", False, "Day1 validation scatter"),
    ("401kDataHDLMs.ipynb",                  False, "Day1 penalized regression"),
    ("401kDataTrees.ipynb",                  False, "Day1 trees -> tree_cv/fourLeaves"),
    ("401kDataStacking.ipynb",               False, "Day1 stacking"),
    ("401kDataNN.ipynb",                     False, "Day1 DNN (PyTorch)"),
    ("401kDataBlackBoxInterpretation.ipynb", False, "Day1 SHAP -> 401k_shap_*"),
    ("RiboflavinExample.ipynb",              False, "Day2 §1 riboflavin"),
    ("BaggingSmoothingDemo.ipynb",           False, "illustration"),
    ("BoostingIllustration.ipynb",           False, "illustration"),
    # --- Day 2 / DML (scripts) ---
    ("HDLMSim.py",                            True,  "§3 HDLM sim (R is canonical figure)"),
    ("InferenceOverviewSimExtended.py",       True,  "§4 sim -> Data/...csv (LONG, ~1hr)"),
    ("InferenceOverviewSimExtended_plots.py", False, "§4 Slide_26/30/34 (needs the csv)"),
    ("401kATE.py",                            True,  "§7 ATE"),
    ("401kATE_diagnostics.py",                False, "§7 ATE_* figs (needs Stata 401kATE.dta)"),
    ("401kLATE.py",                           True,  "§7 LATE"),
    ("ExampleMWDiD.ipynb",                    True,  "§8 DiD"),
    ("ExampleFEPLM.ipynb",                    False, "§10 FE-PLM"),
    ("DLfePLM_viz.py",                        False, "§10 FEPLM_fitspag10 (needs Stata DLfePLM_viz.csv)"),
    ("dube_sensitivity_plot.py",              False, "§9 Dube_sensitivity"),
    ("401kHTE.py",                            True,  "§11 heterogeneous effects (Python-canonical)"),
]

XRAY = ("xray_resnet_training.ipynb", True, "Day1 transfer learning (external data)")

# Scripts that take positional CLI args. The §4 sim defaults to 3 reps when run
# with no args AND overwrites the canonical Data/InferenceOverviewSimExtended.csv
# (which the slide figures are built from) -- so a no-arg run would silently
# clobber the canonical 1000-rep file with a 3-rep smoke. Pass the full rep count
# here. It is flagged is_long, so --skip-long still skips it and falls back to the
# committed CSV. n_jobs from CPU count (joblib). The sim is per-rep seeded, so a
# full run reproduces the committed CSV.
_NJOBS = str(min(8, (os.cpu_count() or 4)))
SCRIPT_ARGS = {
    "InferenceOverviewSimExtended.py": ["1000", _NJOBS],
}


def run_step(name: str) -> int:
    path = HERE / name
    if not path.exists():
        print(f"  !! missing: {name}")
        return 127
    if name.endswith(".ipynb"):
        # Force the course kernel. The notebooks embed kernelspec name "python3",
        # which on this machine resolves to a DIFFERENT conda env (mbaml), so
        # without this every notebook would execute under the wrong package set.
        # Override via MLSSSHORT_KERNEL if the kernel is registered under another
        # name (`jupyter kernelspec list` to check).
        kernel = os.environ.get("MLSSSHORT_KERNEL", "mlssshort")
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
               "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
               f"--ExecutePreprocessor.kernel_name={kernel}", name]
    else:
        cmd = [sys.executable, name, *SCRIPT_ARGS.get(name, [])]
    return subprocess.call(cmd, cwd=str(HERE)).__int__()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--skip-notebooks", action="store_true")
    ap.add_argument("--include-xray", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    steps = list(STEPS) + ([XRAY] if args.include_xray else [])
    if args.only:
        steps = [s for s in steps if s[0] in set(args.only)]
    if args.list:
        for n, lng, note in steps:
            print(f"  {'[long] ' if lng else '       '}{n:42s} {note}")
        return 0

    failed = []
    for name, is_long, note in steps:
        if args.skip_long and is_long:
            print(f"--- SKIP (long): {name}")
            continue
        if args.skip_notebooks and name.endswith(".ipynb"):
            print(f"--- SKIP (notebook): {name}")
            continue
        print("\n" + "=" * 72 + f"\n>>> RUNNING: {name}  ({note})\n" + "=" * 72)
        t0 = time.perf_counter()
        rc = run_step(name)
        dt = time.perf_counter() - t0
        if rc == 0:
            print(f">>> OK: {name}  ({dt:.0f}s)")
        else:
            print(f">>> FAILED: {name}  rc={rc}  ({dt:.0f}s)")
            failed.append(f"{name}(rc={rc})")

    print("\n" + "=" * 72)
    print(">>> ALL OK." if not failed else ">>> FAILURES: " + ", ".join(failed))
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
