# this script contains routines to implement a "naive" E-M algorithm that:
# (1) chooses preferences to maximize the likelihood of choices
# (2) chooses price parameters to maximize the likelihood of prices
# (3) chooses transition parameters to maximize the likelihood of transitions
# (4) updates initial conditions identifically to before.

function naive_expectation_maximization(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx;max_iter = 1000,save = false,mstep_iter = 20)
    J = 9
    err = Inf
    iter = 0
    ll0 = -Inf

    while err>1e-8 && iter<max_iter
        println(" ===== EM-algorithm: iteration $iter =====")
        x0 = pars_inv(p) #<- use these parameters to measure convergence
        # E-step:
        forward_back_threaded!(p,EM,MD,data,n_idx)

        p = mstep_prices(p,EM,MD,n_idx,mstep_iter)

        p = mstep_transitions(p,EM,MD,n_idx,mstep_iter)

        p = mstep_prefs(p,EM,MD,n_idx,mstep_iter)

        mstep_types!(p,EM,MD,data,n_idx,J)

        mstep_πη!(p,EM,MD,data,n_idx,J)

        p = mstep_σ(p,EM,MD,data,n_idx,J)

        ll = log_likelihood(EM,MD,data,n_idx)
        x1 = pars_inv(p)
        err = min(ll - ll0,norm(x1 .- x0,Inf)) # convergence if likelihood starts to decrease
        ll0 = ll #<- update likelihood of current ests
        iter += 1 
        println("current likelihood: $ll")
        println("current error: $err")
        if save & mod(iter,1)==0
            #d = basic_model_fit(p,EM,MD,data,n_idx,"model_stats_progress.csv")
            savepars_vec(p,"current_est")
        end
    end
    return p
end


function mstep_prefs(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    block = [:αθ,:αA,:αH,:αS,:αF,:αR,:αP,:wq,:βΓ,:β,:σ]

    ft = [2,1,1,1,1,1,1,2,1,3,2]

    N_ = sum(length(n_idx[md.case_idx]) for md in MD)

    x0 = pars_inv(p,block,ft)

    objective(x) = -log_likelihood_choices_threaded(pars(x,p,block,ft),EM,MD,data,n_idx) / N_

    res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=iterations))

    return pars(res.minimizer,p,block,ft)
end

function mstep_prices(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    block = [:βw,:βf,:ση]

    ft = [1,1,2]

    N_ = sum(length(n_idx[md.case_idx]) for md in MD)

    x0 = pars_inv(p,block,ft)

    objective(x) = -log_likelihood_prices(pars(x,p,block,ft),EM,MD,data,n_idx) / N_

    res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=iterations))

    return pars(res.minimizer,p,block,ft)
end

function mstep_transitions(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    block = [:λ₀,:λ₁,:δ,:μₒ,:σₒ]

    ft = [3,3,3,1,2]

    N_ = sum(length(n_idx[md.case_idx]) for md in MD)

    x0 = pars_inv(p,block,ft)

    objective(x) = -log_likelihood_transitions(pars(x,p,block,ft),EM,MD,data,n_idx) / N_

    res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=iterations))

    return pars(res.minimizer,p,block,ft)
end


function log_likelihood_chunk_choices(p,MD,EM::Vector{EM_data},data::Vector{likelihood_data},n_idx)
    # setup data:
    T = 18*4
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    R = eltype(p.αH)
    logP = zeros(R,9,K,T)
    V = zeros(R,K,2)
    vj = zeros(R,J)
    ll = 0.
    for md in MD
        solve!(logP,V,vj,p,md)
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        K = prod(size(k_inv))
        s_inv = CartesianIndices((9,K))    
        ll = 0.
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += log_likelihood_choices(EM[n],logP,s_inv)
            end
        end
    end
    return ll
end
function log_likelihood_choices_threaded(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_chunk_choices(p,chunk,EM,data,n_idx)
    end
    ll = fetch.(tasks)
    return sum(ll)
end
function log_likelihood_prices(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    J = 9
    ll = 0.
    for md in MD
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        K = prod(size(k_inv))
        s_inv = CartesianIndices((9,K))    
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += prices_log_like(EM[n],p,md,data[n],J,s_inv,k_inv)
            end
        end
    end
    return ll
end
function log_likelihood_transitions(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    ll = 0.
    for md in MD
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        K = prod(size(k_inv))
        s_inv = CartesianIndices((9,K))    
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += log_likelihood_transitions(EM[n],p,k_inv,s_inv)
                if md.source=="SIPP"
                    ll += log_likelihood_η0(p,EM[n],s_inv,k_inv)
                end             
            end
        end
    end
    return ll
end