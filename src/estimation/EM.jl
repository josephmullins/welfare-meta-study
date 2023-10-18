function expectation_maximization(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx;max_iter = 1000,save = false,mstep_iter = 20)
    J = 9
    err = Inf
    iter = 0
    ll0 = -Inf

    while err>1e-8 && iter<max_iter
        println(" ===== EM-algorithm: iteration $iter =====")
        x0 = pars_inv(p) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,MD,data,n_idx)
        # M-step in 4 parts:
        # (1) most parameters here:
        p = mstep_major(p,EM,MD,n_idx,mstep_iter)
        # (1.1): for robustness, a few steps of just preferences:
        block = [:αθ,:αH,:αA,:αS,:αF,:αP,:αR,:wq,:βΓ]
        ft = [2,1,1,1,1,1,1,2,1]
        p = mstep_major_block(p,block,ft,EM,MD,n_idx,mstep_iter)

        # (2) type selection
        mstep_types!(p,EM,MD,data,n_idx,J)
        # (3) η draw:
        mstep_πη!(p,EM,MD,data,n_idx,J)
        # (4) measurement error
        p = mstep_σ(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx)
        x1 = pars_inv(p)
        err = min(ll - ll0,norm(x1 .- x0,Inf)) # convergence if likelihood starts to decrease
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $err")
        if save & mod(iter,1)==0
            d = basic_model_fit(p,EM,MD,data,n_idx,"model_stats_progress.csv")
            savepars_vec(p,"current_est")
        end
    end
    return p
end


# this function calculates the full log-likelihood for the subset of cases in MD
function log_likelihood(EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    ll = 0.
    for md in MD
        for n in n_idx[md.case_idx]
            if data[n].use
                ll += log(sum(EM[n].α[:,1] .* EM[n].β[:,1]))
            end
        end
    end
    return ll
end
