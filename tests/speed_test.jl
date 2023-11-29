include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 4 #?
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

forward_back_threaded!(p,EM,MD,data,n_idx)

block = [:αH,:αA,:βw]
ft = [1,1,1]
x0 = pars_inv(p,block,ft)

using ForwardDiff
function get_G(x,p,block,ft,EM,MD,data,n_idx)
    ForwardDiff.gradient(x->log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx),x)
end
log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
get_G(x0,p,block,ft,EM,MD,data,n_idx)

@time log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
@time get_G(x0,p,block,ft,EM,MD,data,n_idx)

