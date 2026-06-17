## Riboflavin example to illustrate hypothesis generation
rm(list = ls())

library(hdi)
library(glmnet)

if (!interactive()) pdf(NULL)

######################### Options
dpl = FALSE      # switch to save plots. Set to TRUE to write PDFs to plotdir
plotdir = file.path("..", "Slides", "figures") # relative to R Code/

####################################################
# Riboflavin example
set.seed(6072015)

data(riboflavin)
# riboflavin production by Bacillus subtilis 
# 71 observations, 4088 predictors
# outcome - log riboflavin production rate
# input - log expression level of 4088 genes

# LASSO
lasso.out = cv.glmnet(riboflavin$x,riboflavin$y, nfolds = 10)

if(dpl) pdf(paste(plotdir,'/RiboflavinCV','.pdf',sep=''),height=8,width=10)
plot(lasso.out)
if(dpl) dev.off()

s1 = sum(abs(coef(lasso.out, s = "lambda.1se")) > 0)
sm = sum(abs(coef(lasso.out, s = "lambda.min")) > 0)

cat(sprintf("non-zero coefs at lambda.min: %d\n", sm))
cat(sprintf("non-zero coefs at lambda.1se: %d\n", s1))
cat(sprintf("lambda.min = %.5f, lambda.1se = %.5f\n",
            lasso.out$lambda.min, lasso.out$lambda.1se))

if(dpl) pdf(paste(plotdir,'/RiboflavinCoef','.pdf',sep=''),height=8,width=10)
plot(abs(coef(lasso.out, s = "lambda.1se")), main = "Estimated Coefficients - 10 Fold CV",
     xlab = "j", ylab = "|b_j|", col = "red", pch = 19, cex = 1.5, bg = "red")
points(abs(coef(lasso.out, s = "lambda.min")), pch = 19, col = "blue", cex = 1.5, bg = "blue")
if(dpl) dev.off()

