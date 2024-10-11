# Code and Data for "A Structural Meta-analysis of Welfare Reform Experiments and their Impacts on Children"

In this repo you will find code to replicate the analysis in A Structural Meta-analysis of Welfare Reform Experiments and their Impacts on Children by Mullins (2024).

## Data and Cleaning

The experimental data used in this paper are proprietary and cannot be published due to a data sharing agreement with their owner, [MDRC](https://www.mdrc.org). Researchers wishing to replicate this paper can access this data by contacting MDRC and making a data request.

The SIPP data are taken from CEPR's [uniform extracts](https://ceprdata.org/sipp-uniform-data-extracts/sipp-data/) and are included here in the [data directory](/data).

Below is a summary of the various scripts in [the R directory](/R) used to clean the data.

- [`cleanCTJF.R`](/R/cleanCTJF.R) loads and cleans the raw CTJF data.
- [`cleanMFIP.R`](/R/cleanMFIP.R) and [`cleanFTP.R`](/R/cleanFTP.R) do the same for those respective datasets.
- [`prep_SIPP.R`](/R/prep_SIPP.R) loads and prepares the SIPP data for estimation
- [`final_prep.R`](/R/final_prep.R) combines all datasets in to a single panel for estimation
- [`final_prep_table.R`](/R/final_prep_table.R) computes the summary statistics in Table 1

## Estimation and Analysis

Estimation of first stage parameters is performed in `julia` using the script [`estimate_model_childsample_K5.jl`](/scripts/estimate_model_childsample_K5.jl`) which links to all the necessary source code. This script was run on a high performance cluster using 50 cores.

The estimation exercise that excludes experimental data is performed in [`estimate_model_childsample_K5_noexp.jl`](/scripts/estimate_model_childsample_K5_noexp.jl)

Production parameters are estimated in the script [`estimate_production.jl`](/scripts/estimate_production.jl).

The two counterfactuals from the paper are performed in [`decomposition_counterfactual.jl`](/scripts/decomposition_counterfactual.jl) and [`nonselected_counterfactual.jl`](/scripts/nonselected_counterfactual.jl)


## Figures

The series of `R` scripts below generate all figures that appear in the paper.

- Figure 2 is created by [`initial_dist_figure.R`](/R/initial_dist_figure.R)
- Figures 3 and 4 are created by [`model_fit.R`](/R/model_fit.R)
- Figure 5 is created by [`figure_no_experiment.R`](/R/figure_no_experiment.R)
- Figures 6-10 by [`production_figures.R`](/R/production_figures.R)
- Figures 11-12 by [`counterfactual_figures.R`](/R/counterfactual_figures.R)