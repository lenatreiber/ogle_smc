# ogle_smc
investigation of OGLE V and I LCs of XRBs in the SMC

The latest notebook with automatic analysis is Spline_PulsatingSourcesBasic. For all but one example source, spline detrending (using rspline from the wotan package) 
is way more effective than Savitzky-Golay. The individual source investigations should also be updates with this method, and the analysis should go quicker. summtab.csv
is the summary table that is automatically updated, but also has categorical variables. Most recently, V and I mag skew and kurtosis were added as columns. The next update
will be to add the best period from the auto analysis as a column, and discrepancy from the established period can inform the order and type of further analysis.
