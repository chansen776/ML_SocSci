*** Code for 401(k) example illustrating HDLMs
capture log close
log using 401kHDLMs.txt , text replace 

clear all
 
use "..\Data\sipp1991.dta", clear /* Load data. Assumes I am in my code subfolder */

*** Set seed for reproducibility
set seed 71423

*** Split data into a training (~80%) and test (~20%) set
gen trdata = runiform()
replace trdata = trdata < .8

* Sample sizes
count if trdata
count if !trdata

* Normalize inc,age,fsize,educ 
replace age = age/64
replace inc = inc/250000
replace fsize = fsize/13
replace educ = educ/18

* Create polynomials
forvalues i = 2/6 {
	gen age`i' = age^`i' 
}

forvalues i = 2/8 {
	gen inc`i' = inc^`i'
}

forvalues i = 2/4 {
	gen educ`i' = educ^`i'
}

gen fsize2 = fsize^2

**********************************************************************
*** Linear models in training data
* Baseline
qui reg net_tfa e401 age inc educ fsize marr twoearn db pira hown if trdata
display "Base number of variables: " e(df_m)
predict yhat_ols_base	
	
* Polynomial
qui reg net_tfa e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 ///
	inc6 inc7 inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown ///
	if trdata
display "Base number of variables: " e(df_m)
predict yhat_ols_poly	

* Interacted
qui reg net_tfa /// 
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown)## ///
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown) ///
	if trdata
display "Base number of variables: " e(df_m)
predict yhat_ols_int	
	
************************************************************************
*** Spaghetti plot for lasso in polynomial model
lasso2 net_tfa e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 ///
	inc6 inc7 inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown ///
	if trdata, plotpath(lambda) plotopt(legend(off) ylabel(none)) ///
	lambda(30010(250)10) lglmnet  /* lglmnet option makes lambda on same scale as glmnet */
graph export ..\Slides\figures\Stata_lassoSpaghetti.png , replace
graph close

****************************************************************************
*** Cross-validation for Lasso
* Use 5-fold CV here

*** Baseline model
* Lasso
* Seed option controls the randomized split into folds (replicable).
* Last option saves the fold variable as fvar for use all the way along
cvlasso net_tfa e401 age inc educ fsize marr twoearn db pira hown if trdata, ///
	lglmnet nfolds(5) seed(718) savefoldvar(fvar)	
* When you run this - you get that the optimal lambda is at the end of the range.
* Try again with a longer lambda sequence
cvlasso net_tfa e401 age inc educ fsize marr twoearn db pira hown if trdata, ///
	lglmnet foldvar(fvar) lambda(24010(50)10) 
	/* Reusing the folds from the first run */

local lbaseRMSE = sqrt(e(mspemin))
display "Baseline - Lasso CV " `lbaseRMSE'

cvlasso, plotcv /* Cross validation plot */
gr_edit .legend.draw_view.setstyle, style(no)
graph export ..\Slides\figures\Stata_LassoBasicCV.png , replace
graph close
	
* Store the model.
est store cv_lasso_base
* get the predictions (in-sample and out-of-sample)
predict yhat_lasso_base, lopt

* OLS 5-fold CV on the same folds as the Lasso (fvar)
gen ols_sqres_base = .
forvalues k = 1/5 {
	qui reg net_tfa e401 age inc educ fsize marr twoearn db pira hown ///
		if trdata & fvar != `k'
	tempvar yhat_k
	qui predict `yhat_k'
	qui replace ols_sqres_base = (net_tfa - `yhat_k')^2 if trdata & fvar == `k'
}
qui sum ols_sqres_base if trdata
local mspebase = sqrt(r(mean))
display "Baseline - OLS CV " `mspebase'
drop ols_sqres_base

*** Polynomial model
* Lasso
cvlasso net_tfa e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 ///
	inc6 inc7 inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown ///
	if trdata, lglmnet foldvar(fvar) plotcv
gr_edit .legend.draw_view.setstyle, style(no)	
graph export ..\Slides\figures\Stata_LassoPolyCV.png , replace
graph close

local lpolyRMSE = sqrt(e(mspemin))
display "Polynomial - Lasso CV " `lpolyRMSE'

* Store the model.
est store cv_lasso_poly
* get the predictions (in-sample and out-of-sample)
predict yhat_lasso_poly, lopt

* OLS 5-fold CV on the same folds as the Lasso (fvar)
gen ols_sqres_poly = .
forvalues k = 1/5 {
	qui reg net_tfa e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 ///
		inc6 inc7 inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown ///
		if trdata & fvar != `k'
	tempvar yhat_k
	qui predict `yhat_k'
	qui replace ols_sqres_poly = (net_tfa - `yhat_k')^2 if trdata & fvar == `k'
}
qui sum ols_sqres_poly if trdata
local mspepoly = sqrt(r(mean))
display "Polynomial - OLS CV " `mspepoly'
drop ols_sqres_poly

*** Interaction model
* Lasso
*** Takes forever to run
* cvlasso net_tfa /// 
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown)## ///
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown) ///
	if trdata, lglmnet foldvar(fvar) plotcv 
*** penalty parameter range chosen based on commented out run to get similar 
* results with a much coarser grid
cvlasso net_tfa ///
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown)## ///
	c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
	inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown) ///
	if trdata, lglmnet foldvar(fvar) plotcv lambda(32500(2500)5000)
gr_edit .legend.draw_view.setstyle, style(no)	
graph export ..\Slides\figures\Stata_LassoIntCV.png , replace
graph close

local lintRMSE = sqrt(e(mspemin))
display "Interactions - Lasso CV " `lintRMSE'

* Store the model.
est store cv_lasso_int
* get the predictions (in-sample and out-of-sample)
predict yhat_lasso_int, lopt

* OLS 5-fold CV on the same folds as the Lasso (fvar)
* With p=257 and ~6300 obs per training fold, OLS is identified but
* overfits badly --- expect the holdout RMSE to be enormous.
gen ols_sqres_int = .
forvalues k = 1/5 {
	qui reg net_tfa ///
		c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
		inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown)## ///
		c.(e401 age age2 age3 age4 age5 age6 inc inc2 inc3 inc4 inc5 inc6 inc7 ///
		inc8 educ educ2 educ3 educ4 fsize fsize2 marr twoearn db pira hown) ///
		if trdata & fvar != `k'
	tempvar yhat_k
	qui predict `yhat_k'
	qui replace ols_sqres_int = (net_tfa - `yhat_k')^2 if trdata & fvar == `k'
}
qui sum ols_sqres_int if trdata
local mspeint = sqrt(r(mean))
display "Interactions - OLS CV " `mspeint'
drop ols_sqres_int
	

****************************************************************************
* Out of sample R^2
gen r2_ols_base = (net_tfa - yhat_ols_base)^2
gen r2_ols_poly = (net_tfa - yhat_ols_poly)^2
gen r2_ols_int = (net_tfa - yhat_ols_int)^2

gen r2_lasso_base = (net_tfa - yhat_lasso_base)^2
gen r2_lasso_poly = (net_tfa - yhat_lasso_poly)^2
gen r2_lasso_int = (net_tfa - yhat_lasso_int)^2

* Constant model
qui reg net_tfa if trdata
predict yhat_constant	
gen r2_constant = (net_tfa - yhat_constant)^2

* Get R^2s
qui sum r2_constant if !trdata
local tssOut = r(mean)

qui sum r2_ols_base if !trdata
local essOut_ols_base = r(mean)
local r2Out_ols_base = 1-(`essOut_ols_base'/`tssOut')
qui sum r2_ols_poly if !trdata
local essOut_ols_poly = r(mean)
local r2Out_ols_poly = 1-(`essOut_ols_poly'/`tssOut')
qui sum r2_ols_int if !trdata
local essOut_ols_int = r(mean)
local r2Out_ols_int = 1-(`essOut_ols_int'/`tssOut')

display "OLS & " `r2Out_ols_base' " & " `r2Out_ols_poly' " & " `r2Out_ols_int' " \\"

qui sum r2_lasso_base if !trdata
local essOut_lasso_base = r(mean)
local r2Out_lasso_base = 1-(`essOut_lasso_base'/`tssOut')
qui sum r2_lasso_poly if !trdata
local essOut_lasso_poly = r(mean)
local r2Out_lasso_poly = 1-(`essOut_lasso_poly'/`tssOut')
qui sum r2_lasso_int if !trdata
local essOut_lasso_int = r(mean)
local r2Out_lasso_int = 1-(`essOut_lasso_int'/`tssOut')

display "Lasso & " `r2Out_lasso_base' " & " `r2Out_lasso_poly' " & " `r2Out_lasso_int' " \\"

*****************************************************************************
log close

