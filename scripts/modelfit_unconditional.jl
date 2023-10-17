include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/statistics_exante.jl")

Kτ = 3 #
Kη = 6
p = pars(Kτ,Kη)

loadpars_vec!(p,"est_childsample")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again :-)
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp)
end

M,∂M,MD,EM,data,n_idx = estimation_setup(panel);
update!(p,M,(1,8,9))

x0 = update_inv(p)
G = zeros(length(x0),nthreads())
LL = zeros(nthreads())

Random.seed!(2020)
shuffle!(MD)

# remove ∂M, we don't need it.
∂M = 0.

basic_model_fit(p,EM,MD,data,n_idx,"modelfit_exante.csv")

# NEXT: 
# (2) get new functionsd for model stats
# (3) write function to put it all together. yes.
