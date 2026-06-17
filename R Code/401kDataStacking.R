# Code for 401(k) example illustrating Stacking (PyTorch via reticulate)
# Ten candidate learners aggregated via 5-fold CV with stacking weights
# (unconstrained least squares + non-negative-sum-to-one constrained QP).
# Candidate-learner architectures aligned to the Stata pystacked specification
# (the canonical source for the slide's stacking results table).

######################### Libraries + env
source("setup_reticulate.R")   # points reticulate at mlssshort
library(reticulate)
library(glmnet)
library(randomForest)
library(xgboost)
library(xtable)
library(LowRankQP)

# PyTorch helper: plain MLP fit + predict (mirrors sklearn MLPRegressor)
py_run_string("
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

def make_mlp(p, hidden):
    layers, prev = [], int(p)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.ReLU()]
        prev = int(h)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def fit_dnn_predict(X_train, y_train, X_oof, X_test, hidden,
                    epochs=200, batch_size=200, lr=1e-3, seed=720):
    torch.manual_seed(int(seed))
    model = make_mlp(X_train.shape[1], list(hidden))
    opt = optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    sx, sy = _t(X_train), _t(y_train)
    dl = DataLoader(TensorDataset(sx, sy), batch_size=int(batch_size),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(int(seed)))
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return {
            'oof':  model(_t(X_oof)).numpy().flatten(),
            'test': model(_t(X_test)).numpy().flatten(),
        }
")

######################### Options
dpl <- FALSE
plotdir <- "../Slides/figures"
if (!interactive()) pdf(NULL)

######################### Data
data401k <- read.table("../Data/restatw.dat", header = TRUE)
data401k <- subset(data401k,
  select = c(net_tfa, e401, age, inc, educ, fsize, marr, twoearn, db, pira, hown))

# Normalize select columns
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
nTr <- nrow(train); nTe <- nrow(test)

# HDLM design matrices (poly + pairwise interactions; same as eq.int below)
eq.base <- net_tfa ~ e401 + age + inc + educ + fsize + marr + twoearn + db + pira + hown
eq.int  <- net_tfa ~ (e401 + poly(age, 6, raw=TRUE) + poly(inc, 8, raw=TRUE) +
                        poly(educ, 4, raw=TRUE) + poly(fsize, 2, raw=TRUE) +
                        marr + twoearn + db + pira + hown)^2
xtr <- model.matrix(eq.int, train)[, -1]
xte <- model.matrix(eq.int, test)[, -1]
ytr <- train$net_tfa
yte <- test$net_tfa

# Raw 10-feature matrices for RF / xgboost / DNN
trainXG <- as.matrix(train); testXG <- as.matrix(test)
xgb_train <- xgb.DMatrix(data = trainXG[, -1], label = trainXG[, 1])
xgb_test  <- xgb.DMatrix(data = testXG[, -1],  label = testXG[, 1])
trainNN.X <- trainXG[, -1]; testNN.X <- testXG[, -1]

######################### 5-fold CV
ntr <- length(ytr); Kf <- 5
sampleframe <- rep(1:Kf, ceiling(ntr/Kf))
cvgroup <- sample(sampleframe, size = ntr, replace = FALSE)

yhat.tr <- matrix(0, ntr, 10)
yhat.te <- matrix(0, length(yte), 10)

dnn_specs <- list(c(50, 10, 50), c(50, 50, 50, 50), c(100, 100, 100, 100, 100))

t0 <- Sys.time()
for (k in 1:Kf) {
  cat(sprintf("Fold %d/%d ...\n", k, Kf))
  indk <- cvgroup == k

  # 1. OLS - basic
  lsk <- lm(eq.base, data = train[!indk, ])
  yhat.tr[indk, 1]  <- predict(lsk, train[indk, ])
  yhat.te[,   1]    <- yhat.te[, 1] + predict(lsk, test) / Kf

  # 2. OLS - flexible (HDLM)
  lsk <- lm(eq.int, data = train[!indk, ])
  yhat.tr[indk, 2]  <- predict(lsk, train[indk, ])
  yhat.te[,   2]    <- yhat.te[, 2] + predict(lsk, test) / Kf

  # 3. Lasso (CV)
  lassok <- cv.glmnet(xtr[!indk, ], ytr[!indk])
  yhat.tr[indk, 3]  <- predict(lassok, newx = xtr[indk, ], s = "lambda.min")
  yhat.te[,   3]    <- yhat.te[, 3] + predict(lassok, newx = xte, s = "lambda.min") / Kf

  # 4. Ridge (CV)
  ridgek <- cv.glmnet(xtr[!indk, ], ytr[!indk], alpha = 0)
  yhat.tr[indk, 4]  <- predict(ridgek, newx = xtr[indk, ], s = "lambda.min")
  yhat.te[,   4]    <- yhat.te[, 4] + predict(ridgek, newx = xte, s = "lambda.min") / Kf

  # 5. Random Forest
  rfk <- randomForest(net_tfa ~ ., data = train[!indk, ])
  yhat.tr[indk, 5]  <- predict(rfk, train[indk, ])
  yhat.te[,   5]    <- yhat.te[, 5] + predict(rfk, test) / Kf

  # 6 & 7. Gradient-boosted trees (depth 3 and 5) with inner CV for n_rounds
  k.xgb_train <- xgb.DMatrix(data = trainXG[!indk, -1], label = trainXG[!indk, 1])
  k.xgb_oof   <- xgb.DMatrix(data = trainXG[ indk, -1], label = trainXG[ indk, 1])
  for (j in seq_len(2)) {
    depth <- c(3, 5)[j]
    btk_cv <- xgb.cv(data = k.xgb_train, nrounds = 500, verbose = 0,
                     eta = 0.1, max_depth = depth, nfold = 5)
    best.iter <- which.min(as.matrix(btk_cv$evaluation_log[, 4]))
    btk <- xgboost(data = k.xgb_train, nrounds = best.iter, verbose = 0,
                   eta = 0.1, max_depth = depth)
    yhat.tr[indk, 5 + j]  <- predict(btk, newdata = k.xgb_oof)
    yhat.te[,   5 + j]    <- yhat.te[, 5 + j] + predict(btk, newdata = xgb_test) / Kf
  }

  # 8, 9, 10. DNNs at three Stata-aligned architectures
  for (j in seq_along(dnn_specs)) {
    out <- py$fit_dnn_predict(trainNN.X[!indk, ], ytr[!indk],
                              trainNN.X[ indk, ], testNN.X,
                              hidden = dnn_specs[[j]], seed = 720L)
    yhat.tr[indk, 7 + j]  <- out$oof
    yhat.te[,   7 + j]    <- yhat.te[, 7 + j] + out$test / Kf
  }
}
cat(sprintf("CV loop done in %.1fs\n", as.numeric(Sys.time() - t0, units = "secs")))

######################### Stacking weights
# Unconstrained: OLS without intercept
w_unc <- coef(lm(ytr ~ yhat.tr - 1))

# Constrained: weights >= 0, sum = 1
Vmat <- crossprod(yhat.tr)
dvec <- -t(yhat.tr) %*% ytr
A    <- matrix(rep(1, 10), nrow = 1)
qp   <- LowRankQP(Vmat, dvec, A, 1, rep(1, 10), method = "LU")
w_con <- as.numeric(qp$alpha)

cat(sprintf("Constrained weights: sum = %.4f, min = %.4f, max = %.4f\n",
            sum(w_con), min(w_con), max(w_con)))

######################### Performance summary
ymean_tr <- mean(ytr)
sst_tr <- mean((ytr - ymean_tr)^2)
sst_te <- mean((yte - ymean_tr)^2)

cv_r2 <- c(colMeans((ytr - yhat.tr)^2),
           mean((ytr - yhat.tr %*% w_unc)^2),
           mean((ytr - yhat.tr %*% w_con)^2)) / sst_tr
cv_r2 <- 1 - cv_r2

te_r2 <- c(colMeans((yte - yhat.te)^2),
           mean((yte - yhat.te %*% w_unc)^2),
           mean((yte - yhat.te %*% w_con)^2)) / sst_te
te_r2 <- 1 - te_r2

learner_names <- c("OLS - basic", "OLS - flexible", "Lasso (CV)", "Ridge (CV)",
                   "Random forest", "Boosted trees - depth 3", "Boosted trees - depth 5",
                   "DNN - 50/10/50", "DNN - 50/50/50/50", "DNN - 100/100/100/100/100",
                   "Stacking (unconstrained)", "Stacking (constrained)")
results <- data.frame(`CV R2`   = round(cv_r2, 3),
                      `Test R2` = round(te_r2, 3),
                      Weight    = round(c(w_con, NA, NA), 3),
                      check.names = FALSE,
                      row.names = learner_names)
print(results)
print(xtable(results, digits = c(0, 3, 3, 3)))
