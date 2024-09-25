# List of figures and tables (and scripts that generate them)

- tab:summary | direct input | R/final_prep_table.R
- tab: prefs | tables/preference_ests | scripts/calc_standard_errors.jl<- estimate_model_childsample_K5.jl |
- tab:transition | tables/transition_ests  | ""
- tab:prices | tables/price_ests |  "" |
- fig:selection | Figures/InitialDistributions-Paper | R/initial_dist_figure.R <- initial_dists.csv <- calc_standard_errors | 
- fig:fit_in_sample | Figures/ModelFitPaper | model_fit or model_fit_simpler ?? <- model_stats_K5.csv/modelfit_exante_K5 <- estimate_model_childsample_K5  | 
- fig:fit_out_sample | Figures/OutModelFitPaper | "" | 
- fig:no_exp | Figures/NoExpData | R/figure_no_experiment.R <- modelfit_exante_K5_noexp.csv <- estimate_model_childsample_K5_noexp | 
- tab:no_exp | tables/no_exp_ests | scripts/no_exp_analysis.jl | 
- tab:measurement | tables/factor_analysis | scripts/run_factor_analysis.jl |
- fig:dI | Figures/dI_ests_paper | R/production_figures.R <- output/production_ests.csv <- estimate_production.jl | 
- fig:g1 | Figures/g1_ests_paper | "" |
- fig:g2 | "" | "" | 
- fig:dth | "" | "" | 
- fig:g1_hetero | Figures/g_ests_hetero_paper | "" | "" |
- fig:decomp | Figures/DecompCounterfactualPaper | 
- tab:decomp | tables/decomp_counterfactual | scripts/decomposition_counterfactual.jl | 
- fig:nonselected | Figures/non_selected_counterfactual | scripts/nonselected_counterfactual.jl | 
- tab:nonselected | tables/non_selected_counterfactual | 
- tab:types | tables/type_ests | src/calc_standard_errors.jl |


# TODO: formatting functions form, formse, tex_delimit can live somewhere.
# TODO: check wage_log_like
# ----- Script Summary

estimate_model_childsample_K5
    estimation_setup
    expectation_maximization
    basic_model_fit
    exante_model_fit
    savepars_vec

calc_standard_errors
    loadpars_vec
    pars_inv_full
    estimation_setup
    forward_back_threaded!
    log_likelihood_n
    pars_full
    tex_delimit #<- move this one!

decomposition_counterfactual
    loadpars_vec
    estimation_setup
    full_treatment
    incentives_only
    work_requirements_only
    time_limits_only
    control
    counterfactual
    pars_full
    prod_pars
    boot_cf ** this one written on the spot. adjust??

nonselected_counterfactual
    same as a above

# ------ A mindmap of functions

estimation_setup
    model_data
    likelihood_data
    get_EM_data

basic_model_fit
    basic_model_fit_chunk
        get_model
        solve!
        model_stats
            em_mean

exante_model_fit
    exante_model_fit_chunk
        initialize_exante!
            log_type_prob!
            initial_prob
        get_choice_state_distribution!
            choice_set
        model_stats_exante
            em_mean

savepars_vec
    pars_inv_full

expectation_maximization
    forward_back_threaded!
        forward_back_chunk!
            solve!
            update! #<- updates EM object
                log_type_prob
                wage_log_like
                initial_prob
                chcare_log_like
            forward_back!

    mstep_blocks
        mstep_major_block #<- maximizes over shared parameters
            log_likelihood_threaded(x,p,fnames,ft,EM,MD,data,n_idx)
                ** duplicated in lowmem
                log_likelihood_threaded(x,p,EM,MD,data,n_idx)
                    log_likelihood_chunk(x,p,chunk,EM,data,n_idx) #<- p passed because we need the type
                    ** updated in lowmem to keep at low memory
        mstep_k_block #<- maximizes over type specific parameters
            log_likelihood_threaded (with extra argument)
    mstep_types
    mstep_\pi\eta
    mstep_\sigma
    log_likelihood
    savepars_vec
    basic_model_fit

# ** also look to be many duplicate functions comparoing lowmem to lowmem_k
# ** log_likelihood_threaded_full! #<- this uses the update routine that uses all parameters by default. Do we use this at all?