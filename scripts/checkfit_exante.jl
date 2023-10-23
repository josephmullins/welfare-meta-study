# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/statistics_exante.jl") #<- overwrites some of the functions in statistics.jl

Kτ = 5 #
Kη = 4
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"est_childsample")
p = loadpars_vec(p,"current_est")

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

# get the initial conditions this way
forward_back_threaded!(p,EM,MD,data,n_idx)

d = basic_model_fit(p,EM,MD,data,n_idx,"modelfit_exante.csv")