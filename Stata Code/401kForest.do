*** Code for 401(k) example illustrating random forests
*** Uses Stata's Python integration + scikit-learn (replaces the rforest-based
*** version, which required >20 min to run the OOB-iteration loop).
*** The pre-rewrite version is archived at Old Code/Stata Code/401kForest.do
capture log close
log using 401kForest.txt , text replace

clear all
timer clear 1
timer on 1

use "..\Data\sipp1991.dta", clear  /* Load data. Assumes I am in my code subfolder */

*** Set seed for reproducibility (governs the train/test split only)
set seed 71423

*** Split data into a training (~80%) and test (~20%) set
gen trdata = runiform()
replace trdata = trdata < .8

* Sample sizes
count if trdata
count if !trdata

*** Hand off to Python for all RF fits
python:

import numpy as np
import pandas as pd
from sfi import Data
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Pull data from Stata into numpy
# -----------------------------------------------------------------------------
Xcols = ["e401", "age", "inc", "educ", "fsize", "marr", "twoearn",
         "db", "pira", "hown"]
X = np.asarray(Data.get(" ".join(Xcols)))
y = np.asarray(Data.get("net_tfa"))
T = np.asarray(Data.get("trdata"))

X_train = X[T == 1]
y_train = y[T == 1]
X_test  = X[T == 0]
y_test  = y[T == 0]

y_train_mean = float(y_train.mean())

def val_r2(model):
    sse = np.sum((y_test - model.predict(X_test)) ** 2)
    sst = np.sum((y_test - y_train_mean) ** 2)
    return 1.0 - sse / sst

def oob_rmse_of(model):
    pred = model.oob_prediction_
    return float(np.sqrt(np.mean((y_train - pred) ** 2)))

# -----------------------------------------------------------------------------
# (a) OOB RMSE vs. number of trees
#     warm_start=True grows the forest incrementally -- only the newly added
#     trees are fit at each step, so the full sweep is fast.
# -----------------------------------------------------------------------------
rf_curve = RandomForestRegressor(
    n_estimators=1,
    warm_start=True,
    bootstrap=True,
    oob_score=True,
    max_features="sqrt",
    random_state=720,
    n_jobs=1,
)

iters = list(range(1, 2002, 20))
oob_rmse_vals = []
for i in iters:
    rf_curve.set_params(n_estimators=i)
    rf_curve.fit(X_train, y_train)
    pred = rf_curve.oob_prediction_
    oob_rmse_vals.append(float(np.sqrt(np.mean((y_train - pred) ** 2))))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, oob_rmse_vals, color="blue")
ax.set_xlabel("Number of Trees (B)")
ax.set_ylabel("OOB RMSE")
fig.tight_layout()
fig.savefig("..\\Slides\\figures\\Stata_OOB.png", dpi=150)
plt.close(fig)

print("")
print("OOB RMSE at selected B:")
sample_pts = list(range(0, len(iters), 10)) + [len(iters) - 1]
for idx in sample_pts:
    print("  B = {:5d}   OOB RMSE = {:.2f}".format(iters[idx], oob_rmse_vals[idx]))

# -----------------------------------------------------------------------------
# (b) Tuning-choice table: 6 RFs at n_estimators = 1000
# -----------------------------------------------------------------------------
specs = [
    ("Default",            dict(n_estimators=1000, max_features="sqrt")),
    ("No X Randomization", dict(n_estimators=1000, max_features=1.0)),
    ("Min Size(20)",       dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=20)),
    ("Min Size(40)",       dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=40)),
    ("Min Size(80)",       dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=80)),
    ("Min Size(160)",      dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=160)),
]

rows = []
for name, params in specs:
    rf = RandomForestRegressor(
        bootstrap=True,
        oob_score=True,
        random_state=720,
        n_jobs=1,
        **params,
    )
    rf.fit(X_train, y_train)
    rows.append([name, oob_rmse_of(rf), val_r2(rf)])

out = pd.DataFrame(rows, columns=["Model", "OOB RMSE", "Validation R2"])
print("")
print("Random-forest tuning table (401(k) example):")
print(out.to_string(
    index=False,
    formatters={
        "OOB RMSE":      lambda x: "{:.0f}".format(x),
        "Validation R2": lambda x: "{:.3f}".format(x),
    },
))

end

timer off 1
timer list 1

log close
