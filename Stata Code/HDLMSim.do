*** High-Dimensional Linear Model: simulation illustration of three PLM
*** moment conditions under post-selection lasso with plug-in penalty.
***
*** Companion to Day-2 deck section "Setup -> HDLM: Simulation Illustration".
***
*** Design:
***   Y = alpha * D + 0.5*X1 + 1.0*X2 + N(0,1)
***   D = 1.0*X1 + 0.5*X2 + N(0,1)
***   X ~ N(0, I_p),  n = 200, p = 200, alpha_0 = 0.25
***
*** Three moment conditions:
***   (1) rlasso of Y on (D, X) partialling out D -> post-OLS
***   (2) rlasso of D on X -> partial out -> alpha = <r_D, Y> / <r_D, D>
***   (3) Double selection: post-OLS of Y on (D, X_{sel2 U sel3})
***
*** rlasso (lassopack) uses BCH heteroskedastic plug-in penalty by default.
***
*** Run from "Stata Code/" via:
***     MSYS_NO_PATHCONV=1 StataMP-64 /e do HDLMSim.do
*** (the MSYS_NO_PATHCONV prefix is needed only from git-bash; PowerShell
*** does not need it)

clear all
set more off
set linesize 120
set seed 5192022

local nRep   = 1000
local n      = 200
local p      = 200
local alpha0 = 0.25

matrix alphahat = J(`nRep', 3, .)

local t0 = clock("$S_TIME","hms")
quietly {
    forvalues ii = 1/`nRep' {
        clear
        set obs `n'

        * Generate p covariates X1..Xp ~ N(0,1)
        forvalues j = 1/`p' {
            gen X`j' = rnormal()
        }
        * D = X1 + 0.5*X2 + N(0,1)
        gen D = X1 + 0.5*X2 + rnormal()
        * Y = alpha0*D + 0.5*X1 + 1*X2 + N(0,1)
        gen Y = `alpha0'*D + 0.5*X1 + 1*X2 + rnormal()

        local xvars
        forvalues j = 1/`p' {
            local xvars `xvars' X`j'
        }

        * moment (1): rlasso of Y on (D, X) with D partialled (=unpenalized)
        rlasso Y D `xvars', partial(D)
        local sel1 `e(selected)'
        if "`sel1'" == "" {
            reg Y D
        }
        else {
            reg Y D `sel1'
        }
        matrix alphahat[`ii', 1] = _b[D]

        * moment (2): rlasso of D on X -> post-OLS residuals.
        rlasso D `xvars'
        local sel2 `e(selected)'
        capture drop rD2
        if "`sel2'" == "" {
            sum D, meanonly
            gen double rD2 = D - r(mean)
        }
        else {
            reg D `sel2'
            predict double rD2, residuals
        }
        capture drop rDY rDD
        gen double rDY = rD2 * Y
        gen double rDD = rD2 * D
        sum rDY, meanonly
        scalar SrDY = r(sum)
        sum rDD, meanonly
        scalar SrDD = r(sum)
        matrix alphahat[`ii', 2] = SrDY / SrDD

        * moment (3): double selection -> OLS on union of selected.
        rlasso Y `xvars'
        local sel3 `e(selected)'
        local sel23 : list sel2 | sel3
        if "`sel23'" == "" {
            sum Y, meanonly
            capture drop rY2 rDrY
            gen double rY2 = Y - r(mean)
            gen double rDrY = rD2 * rY2
            sum rDrY, meanonly
            matrix alphahat[`ii', 3] = r(sum) / SrDD
        }
        else {
            reg Y D `sel23'
            matrix alphahat[`ii', 3] = _b[D]
        }

        if mod(`ii', 50) == 0 {
            noisily display "  rep `ii' done"
        }
    }
}
local t1 = clock("$S_TIME","hms")
local elapsed = (`t1' - `t0') / 1000
display "nRep=`nRep': elapsed = `elapsed's"

* Summarize.
clear
svmat alphahat, names(col)
rename c1 mom1
rename c2 mom2
rename c3 mom3
display "Summary (true alpha = 0.25):"
sum mom1 mom2 mom3

* Save snapshot for cross-stack comparison.
export delimited using "../notes/HDLMSim_stata.csv", replace
