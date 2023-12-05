include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 6 #
Kη = 5 #?
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

x0 = pars_inv(p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp)
end

MD,EM,data,n_idx = estimation_setup(panel);

Random.seed!(2020)
shuffle!(MD)

p = expectation_maximization(p,EM,MD,n_idx;max_iter = 100,mstep_iter = 5,save = true)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_K6.csv")
d = exante_model_fit(p,EM,MD,data,n_idx,"modelfit_exante_K6.csv")
savepars_vec(p,"est_childsample_K6")