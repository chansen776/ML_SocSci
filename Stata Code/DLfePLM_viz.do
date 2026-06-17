*** Diagnostic visualization data for the Abortion/Crime PLM-FE example
*** (companion to DLfePLM.do -- DOES NOT modify the slide-canonical pipeline).
***
*** Produces a tidy CSV of, for each of the 3 outcomes and 3 treatments:
***   - raw series                              (original scale)
***   - residual on state + year FE             (two-way within transform)
***   - residual on state+year FE + DS union    (FWL view of the PLR coef)
***   - post-lasso fitted value                 (state+year FE + per-eq
***                                               double-selection set; original
***                                               scale, = var - residual)
*** Plotting is done downstream in "Python code/DLfePLM_viz.py".
***
*** The double-selection sets below are copied verbatim from the canonical
*** pdslasso runs logged in DLfePLM.txt (PDS/CHS, cluster-lasso, xdep loadings,
*** FE + partial(i.year)). If DLfePLM.do's selection ever changes, re-sync the
*** *_ysel / *_dsel locals from the "Selecting HD controls for ..." lines.

capture log close
log using DLfePLM_viz.txt , text replace

clear all

* ---- Load + prep: identical to DLfePLM.do (order matters) ----
import delimited using "..\Data\levitt_ex.dat", clear
drop if statenum == 2 | statenum == 9 | statenum == 12
keep if year > 84 & year < 98
replace xxincome  = xxincome/100
replace xxpover   = xxpover/100
replace xxafdc15  = xxafdc15/10000
replace xxbeer    = xxbeer/100

* ---- Build the flexible-trend expansion: identical to DLfePLM.do ----
xtset statenum year
local V "lpc_viol lpc_prop lpc_murd efaviol efaprop efamurd xxprison xxpolice xxunemp xxincome xxpover xxafdc15 xxgunlaw xxbeer"
foreach v of varlist `V' {
	bysort statenum (year): gen `v'0 = `v'[1]
	by statenum: egen `v'bar = mean(`v')
}
gen trend1 = (year-85)/12
gen trend2 = trend1^2
gen trend3 = trend1^3
foreach v of varlist `V' {
	gen tr1X`v'0   = trend1*`v'0
	gen tr2X`v'0   = trend2*`v'0
	gen tr3X`v'0   = trend3*`v'0
	gen tr1X`v'bar = trend1*`v'bar
	gen tr2X`v'bar = trend2*`v'bar
	gen tr3X`v'bar = trend3*`v'bar
}

* ---- Double-selection sets from DLfePLM.txt (per crime type) ----
* viol
local viol_ysel  "tr3Xefaviolbar xxpover xxbeer tr1Xxxpover0"
local viol_dsel  "tr1Xefaviolbar tr2Xefaviolbar tr3Xefaviol0 xxincome xxbeer tr1Xxxprisonbar tr1Xxxafdc15bar tr2Xxxincomebar tr3Xxxincomebar"
local viol_union "tr1Xefaviolbar tr2Xefaviolbar tr3Xefaviol0 tr3Xefaviolbar xxincome xxpover xxbeer tr1Xxxprisonbar tr1Xxxpover0 tr1Xxxafdc15bar tr2Xxxincomebar tr3Xxxincomebar"
* prop
local prop_ysel  "tr1Xefaprop0 tr1Xxxafdc150 tr2Xxxafdc150 tr3Xxxincome0"
local prop_dsel  "tr1Xefapropbar xxbeer tr1Xxxprisonbar tr1Xxxincomebar"
local prop_union "tr1Xefaprop0 tr1Xefapropbar xxbeer tr1Xxxprisonbar tr1Xxxincomebar tr1Xxxafdc150 tr2Xxxafdc150 tr3Xxxincome0"
* murd
local murd_ysel  "tr3Xefamurd0"
local murd_dsel  "tr2Xefamurdbar tr3Xefamurd0 tr2Xxxincomebar"
local murd_union "tr2Xefamurdbar tr3Xefamurd0 tr2Xxxincomebar"

* The 8 baseline Donohue-Levitt controls (the slide's baseline reghdfe spec,
* before the flexible-trend expansion is formed).
local DL "xxprison xxpolice xxunemp xxincome xxpover xxafdc15 xxgunlaw xxbeer"

* ---- Map each modeled variable to its crime + role ----
*   outcome variables use the outcome-equation selection for the fit
*   treatment variables use the treatment-equation selection for the fit
*   both use the union for the double-selection residual
local vars  "lpc_viol lpc_prop lpc_murd efaviol efaprop efamurd"
local crime_lpc_viol viol
local crime_lpc_prop prop
local crime_lpc_murd murd
local crime_efaviol  viol
local crime_efaprop  prop
local crime_efamurd  murd
local role_lpc_viol  ysel
local role_lpc_prop  ysel
local role_lpc_murd  ysel
local role_efaviol   dsel
local role_efaprop   dsel
local role_efamurd   dsel

* ---- Compute residuals + post-lasso fits for each variable ----
foreach v of local vars {
	local c  "`crime_`v''"
	local r  "`role_`v''"
	local sel   "``c'_`r''"
	local union "``c'_union'"

	* (2) two-way FE residual
	quietly reghdfe `v', absorb(statenum year) resid
	quietly predict double r2_`v', resid

	* (3) double-selection residual (FE + year + union controls)
	quietly reghdfe `v' `union', absorb(statenum year) resid
	quietly predict double r3_`v', resid

	* (4) post-lasso fit on original scale = var - residual,
	*     where residual is on FE + year + per-equation selected controls
	quietly reghdfe `v' `sel', absorb(statenum year) resid
	quietly predict double e_`v', resid
	gen double fit_`v' = `v' - e_`v'
	drop e_`v'

	* (5) DL baseline fit = var on the 8 baseline controls + FE (orig scale)
	quietly reghdfe `v' `DL', absorb(statenum year) resid
	quietly predict double eb_`v', resid
	gen double base_`v' = `v' - eb_`v'
	drop eb_`v'

	display "Done: `v'  (crime=`c', fit-on=`r' [`sel'])"
}

* ---- Export tidy wide CSV for the Python plotter ----
keep statenum year `vars' r2_* r3_* fit_* base_*
order statenum year `vars' r2_* r3_* fit_* base_*
export delimited using "..\Data\DLfePLM_viz.csv", replace

log close
