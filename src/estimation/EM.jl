function expectation_maximization(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx;max_iter = 1000,save = false,mstep_iter = 20)
    J = 9
    err = Inf
    iter = 0
    ll0 = -Inf
    N = sum(length(n_idx[md.case_idx]) for md in MD) 
    (;Kτ) = p
    like_err = Inf
    while err>1e-3 && iter<max_iter && like_err>0 #<- this requires that the likelihood always be increasing
        println(" ===== EM-algorithm: iteration $iter =====")
        x0 = pars_inv(p) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,MD,data,n_idx)

        # M-step in 4 parts:
        # (1) most parameters here:
        #   do the M-step for 9 separate blocks:
        p = mstep_blocks(p,EM,MD,n_idx,mstep_iter) #<- 20 steps.

        # (2) type selection
        mstep_types!(p,EM,MD,data,n_idx,J)
        # (3) η draw:
        mstep_πη!(p,EM,MD,data,n_idx,J)
        # (4) measurement error
        p = mstep_σ(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx) / N
        x1 = pars_inv(p)
        like_err = ll - ll0
        @views err = norm(x1[1:6Kτ+1] .- x0[1:6Kτ+1],Inf)
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $like_err, $err")
        if save && mod(iter,1)==0
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

using ForwardDiff: jacobian, Chunk, JacobianConfig

# this function calculates the score for each observation in the data
function get_score(x_est, p, EM, MD, data, n_idx)
    ll(x) = log_likelihood_n(x,p,EM,MD,data,n_idx)
    cfg = JacobianConfig(ll, x_est, Chunk{4}());
    scores = jacobian(ll,x_est,cfg)
    return scores
end

# this function calculates standard errors using the covariance of the score
function get_standard_errors(x_est, p, EM, MD, data, n_idx)
    scores = get_score(x_est,p,EM,MD,data,n_idx)
    ss = sum(scores,dims=1)[:]
    i_drop = abs.(ss).<1e-10
    i_keep = .~i_drop
    
    N = sum(length(n_idx[md.case_idx]) for md in MD) #<- this ~slightly~ overstates the sample size?
    
    V = inv(cov(scores[:,i_keep])) / N
    se = sqrt.(diag(V))
    
    se_full = zeros(length(x_est))
    se_full[i_keep] .= se
    V_full = diagm(fill(1e-8,length(x_est))) 
    V_full[i_keep,i_keep] .= V
    
    return V_full, se_full
end