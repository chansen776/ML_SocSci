*** Code for 401(k) example illustrating deep nets (PyTorch)
*** Four architectures on the same base network
***   Input -> 50 -> 50 -> 50 -> 50 -> Output, ReLU activations
*** Varying regularization:
***   Model A: no explicit regularization
***   Model B: .5 dropout on hidden layers
***   Model C: L2 weight decay (0.01) on all layers
***   Model D: early stopping (patience 200, max 2000 epochs)

capture log close
log using 401kDNN.txt , text replace

clear all

use "..\Data\sipp1991.dta", clear /* Load data. Assumes I am in my code subfolder */

* Normalize inc, age, fsize, educ (helps NN training)
replace age = age/64
replace inc = inc/250000
replace fsize = fsize/13
replace educ = educ/18

*** Set seed for reproducibility
set seed 71423

*** Split data into a training (~80%) and test (~20%) set
gen trdata = runiform()
replace trdata = trdata < .8

* Sample sizes
count if trdata
count if !trdata

****************************************************************************
*** Fit deep neural network models. Drop into Python / PyTorch.

python:

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sfi import Data

# Pull variables from Stata
X = np.array(Data.get("e401 age inc educ fsize marr twoearn db pira hown"),
             dtype=np.float32)
y = np.array(Data.get("net_tfa"), dtype=np.float32)
T = np.array(Data.get("trdata"))

Xtrain = X[T == 1]
ytrain = y[T == 1]
Xtest  = X[T == 0]
ytest  = y[T == 0]

# Constant-model reference for OOS R^2
ymean = np.mean(ytrain)
oostss = np.mean(np.square(ytest - ymean))

# Standardize features using training-set moments
mu = Xtrain.mean(axis=0)
sd = Xtrain.std(axis=0, ddof=0)
sd[sd == 0] = 1.0
Xtrain = (Xtrain - mu) / sd
Xtest  = (Xtest  - mu) / sd
p = Xtrain.shape[1]

# Sub-train / validation split (last 20% of training rows)
nval = int(round(0.2 * len(Xtrain)))
nsub = len(Xtrain) - nval
sub_X, val_X = Xtrain[:nsub], Xtrain[nsub:]
sub_y, val_y = ytrain[:nsub], ytrain[nsub:]

def _t(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return torch.from_numpy(a)

sub_Xt, sub_yt = _t(sub_X), _t(sub_y)
val_Xt, val_yt = _t(val_X), _t(val_y)
test_Xt, test_yt = _t(Xtest), _t(ytest)

def make_mlp(p, hidden=50, n_hidden=4, dropout=0.0):
    layers = [nn.Linear(p, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)

def fit_model(dropout=0.0, weight_decay=0.0, early_stop_patience=None,
              epochs=200, batch_size=200, lr=1e-3, seed=724):
    torch.manual_seed(int(seed))
    model = make_mlp(p, dropout=dropout)
    opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(sub_Xt, sub_yt), batch_size=int(batch_size),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(int(seed)))
    train_hist, val_hist = [], []
    best_val, best_state, best_epoch, patience = float("inf"), None, 0, 0
    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_hist.append(loss_fn(model(sub_Xt), sub_yt).item())
            v = loss_fn(model(val_Xt), val_yt).item()
            val_hist.append(v)
        if early_stop_patience is not None:
            if v < best_val:
                best_val, best_state, best_epoch, patience = (
                    v, copy.deepcopy(model.state_dict()), epoch, 0
                )
            else:
                patience += 1
                if patience >= int(early_stop_patience):
                    break
    if early_stop_patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yhat = model(test_Xt).numpy().flatten()
    return np.asarray(train_hist), np.asarray(val_hist), yhat, int(best_epoch)

def save_plot(train_hist, val_hist, title, fname, best_epoch=None):
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(train_hist))
    ax.plot(np.sqrt(val_hist))
    if best_epoch is not None:
        ax.axvline(best_epoch, color="k", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.legend(["train", "val"], loc="upper left")
    fig.savefig(fname)
    plt.close(fig)

def report(tag, yhat):
    mse = float(np.mean(np.square(ytest - yhat)))
    rmse = float(np.sqrt(mse))
    r2 = 1.0 - mse / oostss
    print(f"{tag}: Validation RMSE = {rmse:.0f}")
    print(f"{tag}: Validation Rsq  = {r2:.3f}")

# Model A: no explicit regularization
thA, vhA, yhatA, _ = fit_model()
report("A", yhatA)
save_plot(thA, vhA, "DNN Input/50/50/50/50/Output",
          "..\\Slides\\figures\\DNNA_iterations.pdf")

# Model B: .5 dropout on hidden layers
thB, vhB, yhatB, _ = fit_model(dropout=0.5)
report("B", yhatB)
save_plot(thB, vhB, "DNN Input/50/50/50/50/Output, .5 Dropout",
          "..\\Slides\\figures\\DNNB_iterations.pdf")

# Model C: L2 weight decay (0.01) on all layers
thC, vhC, yhatC, _ = fit_model(weight_decay=0.01)
report("C", yhatC)
save_plot(thC, vhC, "DNN Input/50/50/50/50/Output, L2 penalty",
          "..\\Slides\\figures\\DNNC_iterations.pdf")

# Model D: early stopping (patience 200, max 2000 epochs)
thD, vhD, yhatD, best_iter = fit_model(epochs=2000, early_stop_patience=200)
report("D", yhatD)
print(f"D: Early stopping best epoch = {best_iter} (ran {len(vhD)} epochs)")
save_plot(thD, vhD, "DNN Input/50/50/50/50/Output, Early Stopping",
          "..\\Slides\\figures\\DNND_iterations.pdf", best_epoch=best_iter)

end


***************************************************************************
*** Use pystacked to look at performance under several additional choices
*** pystacked has many less options than PyTorch, but is easy to play with here

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20 20) alpha(0) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 20"
matrix list r(m)
qui predict yhat1 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20 20) alpha(.1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 20 a(.1)"
matrix list r(m)
qui predict yhat2 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20 20) alpha(1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 20 a(1)"
matrix list r(m)
qui predict yhat3 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20 20) alpha(0) random_state(720) ///
		validation_fraction(.2) n_iter_no_change(10) max_iter(1000)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 20 es"
matrix list r(m)
qui predict yhat4 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(50 10 50) alpha(0) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "50 10 50"
matrix list r(m)
qui predict yhat5 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(50 10 50) alpha(.1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "50 10 50 a(.1)"
matrix list r(m)
qui predict yhat6 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(50 10 50) alpha(1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "50 10 50 a(1)"
matrix list r(m)
qui predict yhat7 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(50 10 50) alpha(0) random_state(720) ///
		validation_fraction(.2) n_iter_no_change(10) max_iter(1000)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "50 10 50 es"
matrix list r(m)
qui predict yhat8 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20) alpha(0) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20"
matrix list r(m)
qui predict yhat9 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20) alpha(.1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 a(.1)"
matrix list r(m)
qui predict yhat10 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20) alpha(1) random_state(720)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 a(1)"
matrix list r(m)
qui predict yhat11 , basexb

qui pystacked net_tfa e401 age inc educ fsize marr twoearn db pira hown || ///
	m(nnet) opt(hidden_layer_sizes(20) alpha(0) random_state(720) ///
		validation_fraction(.2) n_iter_no_change(10) max_iter(1000)) || ///
	if trdata , type(reg)

qui pystacked , table holdout
display "20 es"
matrix list r(m)
qui predict yhat12 , basexb

*** Fit constant model for reference later
qui reg net_tfa if trdata == 1
predict yhat_c
gen r_mean2 = (net_tfa - yhat_c)^2
qui sum r_mean2 if trdata == 0
local tssOut = r(mean)

* Tried 12 learners
forvalues i = 1/12 {
	qui gen r`i' = (net_tfa - yhat`i'1)^2
	qui sum r`i' if trdata == 0
	local essOut`i' = r(mean)
	local r2Out`i' = 1-(`essOut`i''/`tssOut')
	display "R^2 (in/CV/out) " `i' " = " `r2Out`i''
}



log close

