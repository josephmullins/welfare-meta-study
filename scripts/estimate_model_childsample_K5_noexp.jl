include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5 #?
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)
p = (;p...,λR = 0.)

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

# remove the treatment samples:
MD_est = MD[[md.arm==0 for md in MD]]


# for this exercise, custom write a max step that drops λR and αR
function mstep_blocks(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx,mstep_iter = 40)
    block = [:βw,:ση]
    ft = [1,2]
    p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

    block = [:wq,:αA,:αH,:αS,:αF,:αθ,:λ₀,:λ₁,:δ]
    ft = [2,1,1,1,1,2,3,3,3]
    for kτ in 1:p.Kτ
        p = mstep_k_block(p,kτ,block,ft,EM,MD,n_idx,mstep_iter)
        println("finished type $kτ....")
    end

    block = [:βf]
    ft = [1]
    p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

    block = [:μₒ,:σₒ]
    ft = [1,2]
    p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

    block = [:αP,:σ,:β,:βΓ]
    ft = [1,2,3,1]
    p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

    return p
end

p = expectation_maximization(p,EM,MD_est,n_idx;max_iter = 150,mstep_iter = 5,save = true)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_K5_noexp.csv")
d = exante_model_fit(p,EM,MD,data,n_idx,"modelfit_exante_K5_noexp.csv")
savepars_vec(p,"est_childsample_K5_noexp")