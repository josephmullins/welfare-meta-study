include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5 #?
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

x0 = pars_inv(p)

# load the data
scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")

# pull out the cases with data on children
panel = make_child_sample(panel, scores)

MD,EM,data,n_idx = estimation_setup(panel);

# randomize model types to evenly distribute work over threads
Random.seed!(2020)
shuffle!(MD)


p = expectation_maximization(p,EM,MD,n_idx;max_iter = 150,mstep_iter = 5,save = true)

# calculate standard errors and save the variance-covariance matrix
x_est = pars_inv_full(p)
V, se = get_standard_errors(x_est,p,EM,MD,data,n_idx)
writedlm("output/var_est_K5",V_full)

# write estimates and standard errors to tables
p2 = pars_full(se_full,p)
write_estimates_table!(p,p2,Kτ)

# calculate data the initial distribution over types and states
#  saves data to output/initial_dists.csv
get_data_initial_distribution!(p,EM,MD,n_idx)

# calculate model fit and save data
basic_model_fit(p,EM,MD,data,n_idx,"model_stats_K5.csv")
d = exante_model_fit(p,EM,MD,data,n_idx,"modelfit_exante_K5.csv")
savepars_vec(p,"est_childsample_K5")
