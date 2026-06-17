"""High-Dimensional Linear Model: simulation illustration of three PLM moment
conditions under post-selection lasso with plug-in penalty.

Companion to Day-2 deck section 'Setup -> HDLM: Simulation Illustration'.

Design:
    Y = alpha * D + 0.5*X1 + 1.0*X2 + N(0,1)
    D = 1.0*X1 + 0.5*X2 + N(0,1)
    X ~ N(0, I_p),  n = 200, p = 200, alpha_0 = 0.25

Three moment conditions:
    (1) Lasso of Y on (D, X) with D unpenalized -> post-OLS of Y on (D, X_sel)
    (2) Lasso of D on X -> partial out -> alpha = <r_D, Y> / <r_D, D>
    (3) Double selection: post-OLS of Y on (D, X_{sel(D~X) U sel(Y~X)})

Run as: python "Python code/HDLMSim.py"
"""
import os
import time
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso

np.random.seed(5192022)
nRep = 1000
n = 200
p = 200
betam = np.concatenate([[1.0, 0.5], np.zeros(p - 2)])
betay = np.concatenate([[0.5, 1.0], np.zeros(p - 2)])
alpha0 = 0.25

# Plug-in lambda from BCH.
lambda_r = 2.2 / np.sqrt(n) * stats.norm.ppf(1 - (0.1 / np.log(n)) / (2 * p))


def lasso_with_unpenalized_d(D, X, Y, lam):
    """Lasso of Y on (D, X) with D unpenalized -> selected X columns.

    sklearn's Lasso has no penalty_factor, but we can implement the
    D-unpenalized variant exactly via Frisch-Waugh-Lovell on D alone
    (which is what 'unpenalized' means: D enters with no shrinkage).
    Equivalently: partial D out of Y and out of each X_j by *unpenalized*
    OLS, then lasso the partialled outcomes.  The resulting selected set
    matches glmnet(., penalty.factor = c(0, 1, ..., 1)).
    """
    ols_y_d = LinearRegression().fit(D.reshape(-1, 1), Y)
    r_Y = Y - ols_y_d.predict(D.reshape(-1, 1))
    ols_x_d = LinearRegression().fit(D.reshape(-1, 1), X)
    r_X = X - ols_x_d.predict(D.reshape(-1, 1))
    fit = Lasso(alpha=lam, fit_intercept=False).fit(r_X, r_Y)
    return np.where(fit.coef_ != 0)[0]


alphahat = np.zeros((nRep, 3))
t0 = time.time()
for ii in range(nRep):
    X = np.random.normal(0, 1, (n, p))
    D = X @ betam + np.random.normal(0, 1, n)
    Y = D * alpha0 + X @ betay + np.random.normal(0, 1, n)

    # moment (1): D-unpenalized lasso -> post-OLS of Y on (D, X_sel).
    sel1 = lasso_with_unpenalized_d(D, X, Y, lambda_r)
    if len(sel1) > 0:
        Z1 = np.column_stack([D, X[:, sel1]])
        alphahat[ii, 0] = LinearRegression().fit(Z1, Y).coef_[0]
    else:
        alphahat[ii, 0] = LinearRegression().fit(D.reshape(-1, 1), Y).coef_[0]

    # moment (2): lasso of D on X -> post-OLS -> partial out.
    res2 = Lasso(alpha=lambda_r).fit(X, D)
    sel2 = np.where(res2.coef_ != 0)[0]
    if len(sel2) > 0:
        ols_d_x = LinearRegression().fit(X[:, sel2], D)
        r_D = D - ols_d_x.predict(X[:, sel2])
    else:
        r_D = D - D.mean()
    alphahat[ii, 1] = (r_D @ Y) / (r_D @ D)

    # moment (3): double selection -> post-OLS on union.
    res3 = Lasso(alpha=lambda_r).fit(X, Y)
    sel3 = np.where(res3.coef_ != 0)[0]
    use3 = np.union1d(sel2, sel3)
    if len(use3) > 0:
        Z3 = np.column_stack([D, X[:, use3]])
        alphahat[ii, 2] = LinearRegression().fit(Z3, Y).coef_[0]
    else:
        r_Y = Y - Y.mean()
        alphahat[ii, 2] = (r_D @ r_Y) / (r_D @ r_D)

elapsed = time.time() - t0
print(f"nRep={nRep}: elapsed = {elapsed:.1f}s")

print("\nSummary (true alpha = 0.25):")
print(f"  Mean  : {alphahat.mean(axis=0).round(3)}")
print(f"  StdDev: {alphahat.std(axis=0).round(3)}")

# Optional: write a snapshot CSV for later comparison across stacks.
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notes"))
os.makedirs(out_dir, exist_ok=True)
np.savetxt(os.path.join(out_dir, "HDLMSim_python.csv"), alphahat,
           delimiter=",", header="mom1,mom2,mom3", comments="")
