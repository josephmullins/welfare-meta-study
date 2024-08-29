include("../src/model.jl")
include("../src/estimation.jl")
include("../src/counterfactuals.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")

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

p2 = expectation_maximization(p,EM,MD,n_idx;max_iter = 1,mstep_iter = 5,save = false)

x0 = pars_inv(p)
x1 = pars_inv(p2)


forward_back_threaded!(p,EM,MD,data,n_idx)

block = [:βw,:ση]
ft = [1,2]
N_ = sum(length(n_idx[md.case_idx]) for md in MD)
x0 = pars_inv(p,block,ft)
objective(x) = -log_likelihood_threaded(x,p,block,ft,EM,MD,data,n_idx) / N_

g = ForwardDiff.gradient(objective,x0)

using Plots
ng = 20
E = LinRange(-1e-1,1e-1,ng)
Q = zeros(ng)
P = []
for i in eachindex(x0)
    for ie in eachindex(E)
        x1 = copy(x0)
        x1[i] += E[ie]
        Q[ie] = objective(x1)
    end
    plot_i = plot(E,Q)
    push!(P,plot_i)
end
        


        



res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=4))

res2 = Optim.optimize(objective,x0,NelderMead(),Optim.Options(show_trace = true,iterations=100))

# NOTE: here the issue is that kinkds in the budget set make wage parameters no continuously differentiable

#p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)
