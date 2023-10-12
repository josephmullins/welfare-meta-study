include("../src/model.jl")
include("../src/estimation.jl")

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

x0 = update_inv(p)
G = zeros(length(x0),nthreads())
LL = zeros(nthreads())

Random.seed!(2020)
shuffle!(MD)

expectation_maximization!(p,M,∂M,EM,MD,n_idx,100,true,major = 50,minor = 10)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample.csv")
savepars_vec(p,"est_childsample")
