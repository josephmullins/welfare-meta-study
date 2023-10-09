
function get_bootsample(data::Vector{likelihood_data},MD::Vector{model_data})
    N = length(data)
    b_idx = [[] for md in MD] #<- initialize
    bt = rand(1:N,N)
    for b in bt
        c_idx = data[b].case_idx
        push!(b_idx[c_idx],b)
    end
    return b_idx
end

function bootstrap_trial(p::pars,Gstore,LL,M,∂M,EM,MD,n_idx)
    b_idx = get_bootsample(data,b_idx)
    J = M[1,1].J
    # Run the M-step one time.
    mstep_major!(p,Gstore,LL,M,∂M,EM,MD,b_idx)
    # (1.1): for robustness, a few steps of just preferences:
    block = [1:(5p.Kτ+6);(7p.Kτ+19):(9p.Kτ+23)]
    mstep_major_block!(p,Gstore,LL,block,M,∂M,EM,MD,b_idx)
    # block2 = 
    # mstep_major_block!(p,Gstore,LL,block2,M,∂M,EM,MD,b_idx)
    # (2) type selection
    mstep_types!(p,EM,MD,data,b_idx,J)
    # (3) η draw:
    mstep_πη!(p,EM,MD,data,b_idx,J)
    # (4) measurement error
    mstep_σ!(p,EM,MD,data,b_idx,J)

    # return the estimation results as a vector:
    x = update_inv_full(p)

    # calculate the model statistics:
    d = basic_model_fit(p,EM,MD,data,n_idx,"none",false)
    return x,d
end