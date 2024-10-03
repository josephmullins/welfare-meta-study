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


p = expectation_maximization(p,EM,MD,n_idx;max_iter = 1,mstep_iter = 1,save = false)

# calculate standard errors and save the variance-covariance matrix
x_est = pars_inv_full(p)

# try again here. it should be working?

V, se = get_standard_errors(x_est,p,EM,MD,data,n_idx)
