# Code for 401(k) example illustrating NNs (PyTorch via reticulate)
# Four architectures on the same base network
#   Input -> 50 -> 50 -> 50 -> 50 -> Output, ReLU activations
# Varying regularization:
#   Model A: no explicit regularization
#   Model B: .5 dropout on hidden layers
#   Model C: L2 weight decay (0.01) on all layers
#   Model D: early stopping (patience 200, max 2000 epochs)

######################### Libraries + env
# Point reticulate at the course Python env (mlssshort, where torch lives)
source("setup_reticulate.R")
library(reticulate)

# Define Python helpers (run once; objects live in reticulate's main module)
py_run_string("
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def _t(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return torch.from_numpy(a)

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

def fit_and_eval(sub_X, sub_y, val_X, val_y, test_X, test_y,
                 dropout=0.0, weight_decay=0.0, early_stop_patience=None,
                 epochs=200, batch_size=200, lr=1e-3, seed=724):
    torch.manual_seed(int(seed))
    p = int(sub_X.shape[1])
    model = make_mlp(p, dropout=float(dropout))
    opt = optim.RMSprop(model.parameters(), lr=float(lr),
                        weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    sx, sy = _t(sub_X), _t(sub_y)
    vx, vy = _t(val_X), _t(val_y)
    tx, ty = _t(test_X), _t(test_y)
    dl = DataLoader(TensorDataset(sx, sy), batch_size=int(batch_size),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(int(seed)))
    train_hist, val_hist = [], []
    best_val, best_state, best_epoch, patience = float('inf'), None, 0, 0
    for epoch in range(int(epochs)):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_hist.append(loss_fn(model(sx), sy).item())
            v = loss_fn(model(vx), vy).item()
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
        yhat = model(tx).numpy().flatten()
    return {
        'train_hist': np.asarray(train_hist),
        'val_hist': np.asarray(val_hist),
        'yhat': yhat,
        'best_epoch': int(best_epoch),
        'n_epochs': int(len(val_hist)),
    }
")

######################### Options
dpl <- FALSE    # TRUE to save plots
plotdir <- "../Slides/figures"
if (!interactive()) pdf(NULL)   # suppress Rplots.pdf under Rscript

######################### Data
data401k <- read.table("../Data/restatw.dat", header = TRUE)
data401k <- subset(data401k,
  select = c(net_tfa, e401, age, inc, educ, fsize, marr, twoearn, db, pira, hown))

# Normalize inc, age, fsize, educ (helps NN training)
data401k$age   <- data401k$age/64
data401k$inc   <- data401k$inc/250000
data401k$fsize <- data401k$fsize/13
data401k$educ  <- data401k$educ/18

######################### Train/test split
set.seed(8261977)
ntrain <- 8000
tr <- sample(1:nrow(data401k), ntrain)
train <- data401k[tr, ]
test  <- data401k[-tr, ]

train.X <- as.matrix(train[, -1])
train.y <- train[, "net_tfa"]
test.X  <- as.matrix(test[, -1])
test.y  <- test[, "net_tfa"]

# Standardize features using training-set moments
mu    <- colMeans(train.X)
sigma <- apply(train.X, 2, sd)
train.X <- sweep(sweep(train.X, 2, mu, "-"), 2, sigma, "/")
test.X  <- sweep(sweep(test.X,  2, mu, "-"), 2, sigma, "/")

# Sub-train / validation split within training data
nval <- round(0.2 * nrow(train.X))
nsub <- nrow(train.X) - nval
sub.X <- train.X[1:nsub, ]
sub.y <- train.y[1:nsub]
val.X <- train.X[(nsub + 1):nrow(train.X), ]
val.y <- train.y[(nsub + 1):nrow(train.X)]

cat(sprintf("Train %d  Val %d  Test %d\n",
            nrow(sub.X), nrow(val.X), nrow(test.X)))

######################### Helpers
ytrain_mean <- mean(train.y)
r2_oos <- function(y, yhat) {
  1 - sum((y - yhat)^2) / sum((y - ytrain_mean)^2)
}

plot_hist <- function(th, vh, title, best_epoch = NULL) {
  rmse_t <- sqrt(th); rmse_v <- sqrt(vh)
  plot(rmse_t, type = "l", col = "blue",
       ylim = range(rmse_t, rmse_v),
       xlab = "epoch", ylab = "RMSE", main = title)
  lines(rmse_v, col = "red")
  if (!is.null(best_epoch)) abline(v = best_epoch, lty = 2)
  legend("topright", c("train", "val"), col = c("blue", "red"), lty = 1)
}

######################### Model A: plain
cat("\n--- Model A: plain ---\n")
outA <- py$fit_and_eval(sub.X, sub.y, val.X, val.y, test.X, test.y)
R2A  <- r2_oos(test.y, outA$yhat)
cat(sprintf("  Validation R^2 = %.4f\n", R2A))
if (dpl) png(file.path(plotdir, "NNMonitorA.png"), height = 640, width = 1280)
plot_hist(outA$train_hist, outA$val_hist, "DNN Input/50/50/50/50/Output (Model A)")
if (dpl) dev.off()

######################### Model B: .5 dropout
cat("\n--- Model B: .5 dropout ---\n")
outB <- py$fit_and_eval(sub.X, sub.y, val.X, val.y, test.X, test.y,
                        dropout = 0.5)
R2B  <- r2_oos(test.y, outB$yhat)
cat(sprintf("  Validation R^2 = %.4f\n", R2B))
if (dpl) png(file.path(plotdir, "NNMonitorB.png"), height = 640, width = 1280)
plot_hist(outB$train_hist, outB$val_hist, "DNN w/ .5 Dropout (Model B)")
if (dpl) dev.off()

######################### Model C: L2 weight decay
cat("\n--- Model C: L2 weight decay 0.01 ---\n")
outC <- py$fit_and_eval(sub.X, sub.y, val.X, val.y, test.X, test.y,
                        weight_decay = 0.01)
R2C  <- r2_oos(test.y, outC$yhat)
cat(sprintf("  Validation R^2 = %.4f\n", R2C))
if (dpl) png(file.path(plotdir, "NNMonitorC.png"), height = 640, width = 1280)
plot_hist(outC$train_hist, outC$val_hist, "DNN w/ L2 penalty (Model C)")
if (dpl) dev.off()

######################### Model D: early stopping
cat("\n--- Model D: early stopping ---\n")
outD <- py$fit_and_eval(sub.X, sub.y, val.X, val.y, test.X, test.y,
                        epochs = 2000L,
                        early_stop_patience = 200L)
R2D  <- r2_oos(test.y, outD$yhat)
cat(sprintf("  Validation R^2 = %.4f  (best epoch %d, ran %d)\n",
            R2D, outD$best_epoch, outD$n_epochs))
if (dpl) png(file.path(plotdir, "NNMonitorD.png"), height = 640, width = 1280)
plot_hist(outD$train_hist, outD$val_hist, "DNN w/ Early Stopping (Model D)",
          best_epoch = outD$best_epoch)
if (dpl) dev.off()

######################### Summary
cat("\nValidation R^2 summary:\n")
cat(sprintf("  A (plain):        %.4f\n", R2A))
cat(sprintf("  B (dropout 0.5):  %.4f\n", R2B))
cat(sprintf("  C (L2 0.01):      %.4f\n", R2C))
cat(sprintf("  D (early stop):   %.4f\n", R2D))
