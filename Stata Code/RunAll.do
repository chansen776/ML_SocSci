*** ===================================================================
*** RunAll.do  --  run the CURRENT-deck Stata do-files in order
*** ===================================================================
*** The Stata code is meant to be run ONE FILE AT A TIME (each do-file is
*** self-contained: it loads its data, runs, and writes its own log/figures).
*** This script just runs them all in sequence and should work; it continues
*** past any single failure and prints a summary at the end.
***
*** CAVEAT (not bullet-proof): everything runs in ONE Stata session. A few
*** do-files drive Stata's embedded Python (pystacked / torch / sklearn), and
*** on Windows that can occasionally destabilize the session so a later file
*** errors even though it is fine on its own. If that happens, just run the
*** affected file(s) individually.
***
*** PREREQUISITES: run from the "Stata Code" directory; Stata's Python =
*** mlssshort; packages per "Note on Software for Lectures".
*** RUNTIME: 401kATE ~75 min, 401kLATE ~2 hr, the sim tens of minutes; the
*** whole run is multi-hour. Comment out lines below to run a subset.
*** ===================================================================

clear all
set more off

local files ///
    401kValidationIllustration.do ///
    401kHDLMs.do ///
    401kTrees.do ///
    401kForest.do ///
    401kBoost.do ///
    401kDNN.do ///
    401kStacking.do ///
    401kInterpret.do ///
    riboflavinExample.do ///
    HDLMSim.do ///
    401kATE.do ///
    401kLATE.do ///
    mwDiD.do ///
    DLfePLM.do ///
    DLfePLM_viz.do ///
    401kHTE.do ///
    InferenceOverviewSimExtended.do
	
local failed ""
foreach f of local files {
    display _newline(2) "{hline 70}"
    display ">>> RUNNING: `f'   $S_TIME  $S_DATE"
    display "{hline 70}"
    capture noisily do "`f'"
    local rc = _rc
    if `rc' {
        display as error ">>> FAILED (`f'): rc = `rc'"
        local failed "`failed' `f'(rc=`rc')"
    }
    else {
        display as result ">>> OK: `f'"
    }
    clear all
}

display _newline(2) "{hline 70}"
if "`failed'" == "" {
    display as result ">>> ALL DO-FILES COMPLETED WITHOUT ERROR."
}
else {
    display as error ">>> FAILURES:`failed'"
}
display "{hline 70}"
