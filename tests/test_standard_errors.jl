include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")

x_est = pars_inv_full(p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp) #<- add this back in eventually
end

MD,EM,data,n_idx = estimation_setup(panel);

Random.seed!(2020)
shuffle!(MD)

forward_back_threaded!(p,EM,MD,data,n_idx)
LL = log_likelihood_n(x_est,p,EM,MD,data,n_idx)


V, se = get_standard_errors(x_est,p,EM,MD,data,n_idx)