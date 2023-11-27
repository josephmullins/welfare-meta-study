include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/EM_naive.jl")

Kτ = 4 #
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

p = naive_expectation_maximization(p,EM,MD,n_idx; max_iter = 20, mstep_iter = 10,save = true)
p = naive_expectation_maximization(p,EM,MD,n_idx; max_iter = 40, mstep_iter = 10,save = true)


#basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample_K4.csv")
#savepars_vec(p,"est_childsample_K4")