include("../src/model.jl")
include("../src/estimation.jl")

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

MD = MD[1:100]
forward_back_threaded!(p,EM,MD,data,n_idx)


block = [:αH,:βw,:ση]
ft = [1,1,2]
x = pars_inv(p,block,ft)
break
# ok, so we know it's coming from the choice probabilities.

include("../src/estimation/likelihood.jl")

log_likelihood_threaded(x,p,block,ft,EM,MD,data,n_idx)
@time log_likelihood_threaded(x,p,block,ft,EM,MD,data,n_idx)

include("../src/estimation/likelihood_alt.jl")

log_likelihood_threaded(x,p,block,ft,EM,MD,data,n_idx)
@time log_likelihood_threaded(x,p,block,ft,EM,MD,data,n_idx)

p = expectation_maximization(p,EM,MD,n_idx; max_iter = 4, mstep_iter = 20,save = true)
#p = expectation_maximization(p,EM,MD,n_idx;max_iter = 150,mstep_iter = 120,save = true)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample_K4.csv")
savepars_vec(p,"est_childsample_K4")
d = exante_model_fit(p,EM,MD,data,n_idx,"modelfit_exante.csv")
