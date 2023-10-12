function expectation_maximization!(p::pars,M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},n_idx,max_iter = 1000,save = false)
    LL = zeros(nthreads())
    x0 = update_inv(p)
    Gstore = zeros(length(x0),nthreads())
    J = M[1,1].J
    err = Inf
    iter = 0
    ll0 = -Inf

    while err>1e-8 && iter<max_iter
        println(" ===== EM-algorithm: iteration $iter =====")
        x0 = update_inv(p) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,M,MD,data,n_idx)
        # M-step in 4 parts:
        # (1) most parameters here:
        mstep_major!(p,Gstore,LL,M,∂M,EM,MD,n_idx,15)
        # (1.1): for robustness, a few steps of just preferences:
        block = [1:(5p.Kτ+6);(7p.Kτ+19):(9p.Kτ+23)]
        mstep_major_block!(p,Gstore,LL,block,M,∂M,EM,MD,n_idx,10)
        # block2 = 
        # mstep_major_block!(p,Gstore,LL,block2,M,∂M,EM,MD,n_idx)
        # (2) type selection
        mstep_types!(p,EM,MD,data,n_idx,J)
        # (3) η draw:
        mstep_πη!(p,EM,MD,data,n_idx,J)
        # (4) measurement error
        mstep_σ!(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx)
        x1 = update_inv(p)
        err = min(ll - ll0,norm(x1 .- x0,Inf)) # convergence if likelihood starts to decrease
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $err")
        if save & mod(iter,1)==0
            d = basic_model_fit(p,EM,MD,data,n_idx,"model_stats_progress.csv")
            savepars(p,"current_est")
        end
    end
end

function naive_expectation_maximization(p::pars,M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},n_idx::Vector{Vector{Int64}},max_iter = 1000,save = false)
    LL = zeros(nthreads())
    Gstore = zeros(length(x0),nthreads())
    J = M[1,1].J
    err = Inf
    iter = 0
    ll = -Inf

    while err>1e-8 && iter<max_iter
        println(" ===== Naive EM-algorithm: iteration $iter =====")
        x0 = update_inv(p) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,M,MD,data,n_idx)
        # M-step in 5 parts:
        # (0)
        mstep_prices!(p,EM,MD,data,n_idx,J)
        block = [1:(5p.Kτ+6);(7p.Kτ+19):(9p.Kτ+23)]
        # (1) preferences
        mstep_major_block!(p,Gstore,LL,block,M,∂M,EM,MD,n_idx,20)
        # (2) type selection
        mstep_types!(p,EM,MD,data,n_idx,J)
        # (3) η draw:
        mstep_πη!(p,EM,MD,data,n_idx,J)
        # (4) measurement error
        mstep_σ!(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx)
        x1 = update_inv(p)
        err = min(ll - ll0,norm(x1 .- x0,Inf))
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $err")
        if save & mod(iter,5)==0
            d = basic_model_fit(p,EM,MD,data,n_idx,"model_stats_progress.csv")
        end
    end
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
