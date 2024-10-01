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

# pull out the cases with data on children
panel = make_child_sample(panel, scores)

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

# --- Calculate Standard Errors and Make a Comparison Table ---- #
x_est = pars_inv_full(p)
forward_back_threaded!(p,EM,MD,data,n_idx)
LL = log_likelihood_n(x_est,p,EM,MD,data,n_idx)

scores = get_score(x_est,p,EM,MD,data,n_idx)

# we want to look only at σ and β:
sc = scores[:,(11p.Kτ+21):(11p.Kτ+24)]
xe = x_est[(11p.Kτ+21):(11p.Kτ+24)]

N = sum(length(n_idx[md.case_idx]) for md in MD) 

V = inv(cov(sc)) / N
se = sqrt.(diag(V))

p2 = pars(se,p,[:σ;:β],[2,3])

# load the baseline estimates and standard errors

pb = loadpars_vec(p,"est_childsample_K5")
V = readdlm("output/var_est_K5")
se = sqrt.(diag(V))
pb2 = pars_full(se,pb)

# write a table that compares estimates to those from the baseline
write_comparison_table!(p,p2,pb,pb2)
