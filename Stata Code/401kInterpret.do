*** 401(k) example: black-box interpretation -- Random Forest
*** Permutation VI, partial dependence, and SHAP for the same RF.
*** All numerical work runs in Stata's embedded Python (mlssshort env).
capture log close
log using 401kInterpret.txt , text replace

clear all

timer on 1

use "..\Data\sipp1991.dta", clear   /* assumes cwd is Stata Code\ */

* Normalize inc, age, fsize, educ to roughly [0,1].
replace age   = age / 64
replace inc   = inc / 250000
replace fsize = fsize / 13
replace educ  = educ / 18

* Set seed and split ~80/20.
set seed 71423
gen trdata = runiform()
replace trdata = trdata < .8

count if trdata
count if !trdata


****************************************************************************
* All RF interpretation work happens in embedded Python.

python:

from sfi import Data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.inspection import permutation_importance as permimp
from sklearn.inspection import partial_dependence as partdep
import shap

# Pull data from Stata.
X = np.array(Data.get("e401 age inc educ fsize marr twoearn db pira hown"))
y = np.array(Data.get("net_tfa"))
T = np.array(Data.get("trdata"))

Xcols = np.array(["e401", "age", "inc", "educ", "fsize", "marr",
                  "twoearn", "db", "pira", "hown"])
p = X.shape[1]

Xtrain, ytrain = X[T == 1], y[T == 1]
Xtest,  ytest  = X[T == 0], y[T == 0]

print(f"ntrain={len(Xtrain)}  ntest={len(Xtest)}  features={p}")

# Random forest. n_jobs=1 -- joblib-loky backend hangs inside Stata's
# embedded Python on Windows (see PLAN section 5, session 6).
reg = rf(n_estimators=500, min_samples_leaf=20, random_state=720, n_jobs=1)
reg.fit(Xtrain, ytrain)

print(f"RF train R^2: {reg.score(Xtrain, ytrain):.3f}")
print(f"RF test  R^2: {reg.score(Xtest,  ytest):.3f}")


############################################################
# Permutation variable importance (sklearn).

rfimp = permimp(reg, Xtest, ytest, n_repeats=100, random_state=720, n_jobs=1)

print("\nPermutation VI (decrease in R^2, sorted):")
for i in rfimp.importances_mean.argsort()[::-1]:
    print(f"  {Xcols[i]:<8}  {rfimp.importances_mean[i]:7.4f}"
          f"  +/- {rfimp.importances_std[i]:7.4f}")


############################################################
# Partial dependence on income (continuous) and pira (binary).

pdrf_inc = partdep(reg, features=[2], X=X, categorical_features=(0,4,5,6,7,8,9))
pdrf_inc_x = np.asarray(pdrf_inc.grid_values[0]) * 250000   # un-normalize income
pdrf_inc_y = np.asarray(pdrf_inc.average[0])

print("\nPartial dependence (RF) on income, 10k grid:")
for xv, yv in zip(pdrf_inc_x[::3], pdrf_inc_y[::3]):
    print(f"  inc={int(xv):>9,}    avg pred net assets={float(yv):>9,.0f}")

pdrf_pira = partdep(reg, features=[8], X=X, categorical_features=(0,4,5,6,7,8,9))
print(f"\nPD pira = 0:  {pdrf_pira.average[0][0]:8.0f}"
      f"    pira = 1:  {pdrf_pira.average[0][1]:8.0f}")

pdrf_e401 = partdep(reg, features=[0], X=X, categorical_features=(0,4,5,6,7,8,9))
print(f"PD e401 = 0:  {pdrf_e401.average[0][0]:8.0f}"
      f"    e401 = 1:  {pdrf_e401.average[0][1]:8.0f}")


############################################################
# SHAP via TreeExplainer (exact, fast for tree ensembles).

explainer   = shap.TreeExplainer(reg)
shap_values = explainer(Xtest)

expected = float(np.asarray(explainer.expected_value).item())
print(f"\nSHAP expected value E[g(X)]: {expected:.4f}")

# Sanity check: g(x_i) = E[g(X)] + sum_j phi_j(x_i)
recon = expected + shap_values.values[0].sum()
print(f"SHAP check (obs 0):  baseline + sum(phi) = {recon:.4f}"
      f"    model pred = {reg.predict(Xtest[[0]])[0]:.4f}")

mean_abs = np.abs(shap_values.values).mean(axis=0)
print("\nMean |SHAP| importance ranking:")
for j in np.argsort(-mean_abs):
    print(f"  {Xcols[j]:<8}  {mean_abs[j]:8.1f}")

end


timer off 1
timer list
timer clear

log close
