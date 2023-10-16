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

MD_ = MD[1:200];

forward_back_threaded!(p,EM,MD,data,n_idx)

@time log_likelihood_threaded(x0,p,EM,MD,data,n_idx)
@time log_likelihood_threaded(x0,p,EM,MD,data,n_idx)

p = mstep_major(p,EM,MD_,n_idx,2)
# (1.1): for robustness, a few steps of just preferences:
block = [1:(5p.Kτ+6);(7p.Kτ+19):(9p.Kτ+23)]
p = mstep_major_block(p,block,EM,MD_,n_idx,2)
# block2 = 
# mstep_major_block!(p,Gstore,LL,block2,M,∂M,EM,MD_,n_idx)

sources = ("SIPP","FTP","CTJF","MFIP")
blocks = (1:5,6:11,12:18,19:27)
pos = 1
s = 1

Kx = (5,6,7,9) #
source = sources[s]
block = blocks[s]
kx = Kx[s]
nβ = kx * (p.Kτ-1)
x0 = reshape(p.βτ[block,:],nβ)
type_obj(x) = -log_likelihood_type(x,kx,source,EM,MD,p,data,n_idx,J)
res = Optim.optimize(type_obj,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace=false,iterations=100))
@views p.βτ[block,:][:] .= res.minimizer[:]


# (2) type selection
mstep_types!(p,EM,MD_,data,n_idx,J)
# (3) η draw:
mstep_πη!(p,EM,MD_,data,n_idx,J)
# (4) measurement error
p = mstep_σ(p,EM,MD_,data,n_idx,J)

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
