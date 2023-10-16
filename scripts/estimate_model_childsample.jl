include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 3 #
Kη = 6
p = pars(Kτ,Kη)
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

MD_ = MD[1:400];

forward_back_threaded!(p,EM,MD,data,n_idx)

@time log_likelihood_threaded(x0,p,EM,MD,data,n_idx)
@time log_likelihood_threaded(x0,p,EM,MD,data,n_idx)

#optimize(x->-log_likelihood_chunk(,MD_[1:10],EM,data,n_idx))

break
#c_subset = DataFrame(case_idx = [md.case_idx for md in MD_subset])
#CSV.write("../output/case_subset.csv",c_subset)

# round cases to get faster? don't think it will amount to much.

expectation_maximization!(p,M,∂M,EM,MD,n_idx,100,true)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample.csv")
savepars_vec(p,"est_childsample")

# now do all the data
# forward_back_threaded!(p,EM,M,MD,data,n_idx)
# basic_model_fit(p,EM,MD,data,n_idx,"../output/model_stats_full.csv")

# practice getting bootstrap sample:
# easy peasy, nice :-).

# N = length(data)
# b_idx = [[] for md in MD] #<- initialize
# bt = rand(1:N,N)
# for b in bt
#     c_idx = data[b].case_idx
#     push!(b_idx[c_idx],b)
# end
