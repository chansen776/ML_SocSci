*** Code for abortion example illustrating PLM with fixed effects

capture log close
log using DLfePLM.txt , text replace 

clear all
 
* Load data. Assumes I am in my code subfolder *
import delimited using "..\Data\levitt_ex.dat", clear 

* Drop DC, AK, HI
drop if statenum == 2 | statenum == 9 | statenum == 12

* Restrict to years from Donohue, Levitt paper
keep if year > 84 & year < 98

* Scale some variables
replace xxincome = xxincome/100
replace xxpover = xxpover/100
replace xxafdc15 = xxafdc15/10000
replace xxbeer = xxbeer/100

*** Get baseline results
reghdfe lpc_viol efaviol xx* , absorb(statenum year) cluster(statenum)
reghdfe lpc_prop efaprop xx* , absorb(statenum year) cluster(statenum)
reghdfe lpc_murd efamurd xx* , absorb(statenum year) cluster(statenum)

*** Create expansion
xtset statenum year

* Generate initial conditions and state means
local V "lpc_viol lpc_prop lpc_murd efaviol efaprop efamurd xxprison xxpolice xxunemp xxincome xxpover xxafdc15 xxgunlaw xxbeer"
foreach v of varlist `V' {
	bysort statenum (year): gen `v'0 = `v'[1]
	by statenum: egen `v'bar = mean(`v')
}

gen trend1 = (year-85)/12
gen trend2 = trend1^2
gen trend3 = trend1^3

foreach v of varlist `V' {
	gen tr1X`v'0 = trend1*`v'0
	gen tr2X`v'0 = trend2*`v'0
	gen tr3X`v'0 = trend3*`v'0
	gen tr1X`v'bar = trend1*`v'bar
	gen tr2X`v'bar = trend2*`v'bar
	gen tr3X`v'bar = trend3*`v'bar
}

*** Regression with everything
local X "xx* tr1Xxx* tr2Xxx* tr3Xxx*"
local Xviol "efaviol0 efaviolbar tr1Xefaviol* tr2Xefaviol* tr3Xefaviol*"
local Xprop "efaprop0 efapropbar tr1Xefaprop* tr2Xefaprop* tr3Xefaprop*"
local Xmurd "efamurd0 efamurdbar tr1Xefamurd* tr2Xefamurd* tr3Xefamurd*"

reghdfe lpc_viol efaviol `Xviol' `X' , absorb(statenum year) cluster(statenum)
display "Number of variables: " e(df_m)
reghdfe lpc_prop efaprop `Xprop' `X' , absorb(statenum year) cluster(statenum)
display "Number of variables: " e(df_m)
reghdfe lpc_murd efamurd `Xmurd' `X' , absorb(statenum year) cluster(statenum)
display "Number of variables: " e(df_m)

*** Lasso results using plug-in penalization and clustered loadings.
*** NOTE: loptions(xdep) uses an X-dependent (simulation-based) penalty that
*** draws from Stata's RNG, so without a seed the selected controls -- and hence
*** the post-double-selection estimates -- vary run to run. Set the seed before
*** each call so each pdslasso is reproducible regardless of run context.
set seed 72723
pdslasso lpc_viol efaviol (`Xviol' `X' i.year) , fe partial(i.year) cluster(statenum) loptions(xdep)
set seed 72723
pdslasso lpc_prop efaprop (`Xprop' `X' i.year) , fe partial(i.year) cluster(statenum) loptions(xdep)
set seed 72723
pdslasso lpc_murd efamurd (`Xmurd' `X' i.year) , fe partial(i.year) cluster(statenum) loptions(xdep)



log close



