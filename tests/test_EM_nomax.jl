include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K4")

x_est = pars_inv_full(p)

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

function expectation_maximization_nomax(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx;max_iter = 1000)
    J = 9
    err = Inf
    iter = 0
    ll0 = -Inf

    while err>1e-8 && iter<max_iter
        println(" ===== EM-algorithm: iteration $iter =====")
        x0 = copy(p.πη) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,MD,data,n_idx)
        # M-step in 4 parts:
        # (1) most parameters here:
        # do the M-step for 9 separate blocks:
        #println(" -- doing the blocked step --- ")
        #p = mstep_blocks(p,EM,MD,n_idx,20)
        #println(" -- now doing everything simulataneously --- ")
        #p = mstep_major(p,EM,MD,n_idx,mstep_iter)
        # (1.1): for robustness, a few steps of just preferences:
        #block = [:αθ,:αH,:αA,:αS,:αF,:αP,:αR,:wq,:βΓ]
        #ft = [2,1,1,1,1,1,1,2,1]
        #p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

        # (2) type selection
        mstep_types!(p,EM,MD,data,n_idx,J)
        # (3) η draw:
        mstep_πη!(p,EM,MD,data,n_idx,J)
        # (4) measurement error
        p = mstep_σ(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx)
        x1 = p.πη
        err = min(ll - ll0,norm(x1 .- x0,Inf)) # convergence if likelihood starts to decrease
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $err")
    end
    return p
end

expectation_maximization_nomax(p,EM,MD,n_idx)

# this converges. The issue is with the m-step, maybe it goes down very marginally or something?