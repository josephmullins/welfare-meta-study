include("BaumWelch.jl")

function log_likelihood(x,g,∂θ,L,G,data,pars,model,∂m)
    update!(x,∂θ,pars,model,∂m) #<- updating step
    # want: to fill dθ/dx in the update step?
    solve!(model,∂m)
    N = length(data)
    for n in 1:N
        @views L[n] = log_likelihood(G[:,n],n,data,pars,model,∂m) #<- this function must be custom written
    end
    ll = sum(L)
    Gsum = sum(G,dims=2)[:]
    #@views sum!(G[:,1],G) #<- sum everything into the first column
    @views mul!(g,∂θ,Gsum) #<- results in a small number of allocations
    return ll
end

function log_likelihood(x,L,data,pars,model)
    update!(x,pars,model) #<- updating step
    # want: to fill dθ/dx in the update step?
    solve!(model)
    N = length(data)
    Threads.@threads for n in 1:N
        L[n] = log_likelihood(n,data,pars,model) #<- this function must be custom written
    end
    ll = sum(L)
    return ll
end

function log_likelihood(x,g,∂θ,L,G,EM::Vector{EM_data},data,pars,model,∂m)
    update!(x,∂θ,pars,model,∂m) #<- updating step
    # want: to fill dθ/dx in the update step?
    solve!(model,∂m)
    N = length(data)
    for n in 1:N
        @views L[n] = log_likelihood(G[:,n],n,EM[n],data,pars,model,∂m) #<- this function must be custom written
    end
    ll = sum(L)
    Gsum = sum(G,dims=2)[:]
    #@views sum!(G[:,1],G) #<- sum everything into the first column
    @views mul!(g,∂θ,Gsum) #<- results in a small number of allocations
    return ll
end

function log_likelihood(x,L,EM::Vector{EM_data},data,pars,model)
    update!(x,pars,model) #<- updating step
    # want: to fill dθ/dx in the update step?
    solve!(model)
    N = length(data)
    for n in 1:N
        @views L[n] = log_likelihood(n,EM[n],data,pars,model) #<- this function must be custom written
    end
    ll = sum(L)
    return ll
end


# function to evaluate log-likelihood of choices
function log_likelihood_choices(g,data,m::ddc_model,∂m::ddc_derivative)
    ll = 0.
    for t in axes(data.choice,2)
        s = data.S[t]
        @views ll += log_likelihood(data.choice[:,t],g,m.logP[:,s,t],∂m.logP[:,:,s,t],m.G)
    end
    return ll
end
# -- version without derivative
function log_likelihood_choices(data,m::ddc_model)
    ll = 0.
    for t in axes(data.choice,2)
        s = data.S[t]
        @views ll += log_likelihood(data.choice[:,t],m.logP[:,s,t],m.G)
    end
    return ll
end

# now do as above but for data coming from an EM algorithm:
# - here the choices and the states are partially unobserved
function log_likelihood_choices(g,em::EM_data,m::ddc_model,∂m::ddc_derivative)
    ll = 0.
    J,K,T = size(m.u)
    for t in axes(em.q_s,2)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = s_inv(s_idx,J)
            wght = em.q_s.nzval[s]
            @views ll += log_likelihood(j,g,m.logP[:,k,t],∂m.logP[:,:,k,t],m.G,wght)
        end
    end
    return ll
end
function log_likelihood_choices(em::EM_data,m::ddc_model)
    ll = 0.
    J,K,T = size(m.u)
    for t in axes(em.q_s,2)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = s_inv(s_idx,J)
            wght = em.q_s.nzval[s]
            @views ll += log_likelihood(j,m.logP[:,k,t],m.G,wght)
        end
    end
    return ll
end


# function to evaluate log-likelihood of transitions
function log_likelihood_transitions(g,data,m::ddc_model,∂m::ddc_derivative)
    ll = 0.
    L,T = size(data.choice)
    for t in 1:T-1
        s = data.S[t]
        s2 = data.S[t+1]
        j = data.choice[1,t]
        f_ss = m.F[j,t][s2,s]
        ll += log(f_ss)
        _p_ = 1
        for p in ∂m.F_idx
            g[p] += ∂m.F[_p_,j,t][s2,s] / f_ss
            _p_ += 1
        end
    end
    return ll
end

function log_likelihood_transitions(data,m::ddc_model)
    ll = 0.
    L,T = size(data.choice)
    for t in 1:T-1
        s = data.S[t]
        s2 = data.S[t+1]
        j = data.choice[1,t]
        f_ss = m.F[j,t][s2,s]
        ll += log(f_ss)
    end
    return ll
end

# same as above: two versions of the likelihood that use EM_data instead (states unobservable)
# function to evaluate log-likelihood of transitions
function log_likelihood_transitions(g,em::EM_data,m::ddc_model,∂m::ddc_derivative)
    ll = 0.
    J,K,T = size(m.u)
    for t in eachindex(em.q_ss)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = s_inv(s_idx,J)
            for sn in nzrange(em.q_ss[t],s_idx)
                sn_idx = em.q_ss[t].rowval[sn]
                jn,kn = s_inv(sn_idx,J)
                f_ss = m.F[j,t][kn,k]
                wght = em.q_ss[t].nzval[sn]
                ll += wght * log(f_ss)
                _p_ = 1
                for p in ∂m.F_idx
                    g[p] += wght * ∂m.F[_p_,j,t][kn,k] / f_ss
                    _p_ += 1
                end
            end
        end
    end
    return ll
end

function log_likelihood_transitions(em::EM_data,m::ddc_model)
    ll = 0.
    J,K,T = size(m.u)
    for t in eachindex(em.q_ss)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = s_inv(s_idx,J)
            for sn in nzrange(em.q_ss[t],s_idx)
                sn_idx = em.q_ss[t].rowval[sn]
                jn,kn = s_inv(sn_idx,J)
                f_ss = m.F[j,t][kn,k]
                wght = em.q_ss[t].nzval[sn]
                ll += wght * log(f_ss)
            end
        end
    end
    return ll
end


function expectation_maximization(x0,EM::Vector{EM_data},data,pars,model,∂m,max_iter=1000)
    N = length(data)
    np = length(x0)
    np_ = size(∂m.logP,1)
    g = zeros(np)
    ∂θ = zeros(np,np_)
    G = zeros(np,N)
    L = zeros(N)
    err = Inf
    iter = 0
    # define the function for optimization:

    function fg!(F,G,x)
        if G !== nothing
            g = zeros(9)
            ll   = -log_likelihood(x,G,∂θ,L,Gstore,EM,data,pars,model,∂m)
            G[:] .*= -1
            return ll
        else
            return -log_likelihood(x,L,data,EM,pars,model)
        end
    end
    

    while err>1e-8 && iter<max_iter
        println(" ===== EM-algorithm: iteration $iter =====")
        # E-step:
        update!(x0,pars,model) #<- updating step
        solve!(model) #<- solve the model and choice probabilities
        for n in eachindex(EM)
            update!(data[n],EM[n],model,pars)
            forward_back!(EM[n])
        end
        ll = log_likelihood(EM)
        # M-step:
        res = Optim.optimize(Optim.only_fg!(fg!),x0,BFGS(),Optim.Options(show_trace = false))
        err = norm(res.minimizer .- x0,Inf)
        iter += 1 
        x0 = res.minimizer
        println("current likelihood: $ll")
        println("current error: $err")
    end
    # eventually we want to calculate the standard errors here
    return x0
end

# # expectation-maximization routine also?
# function log_likelihood(x,g,data::Vector{ddc_data},b::basemod,model::ddc_model,∂m::ddc_derivative)
#     update!(x,b)
#     model.σ[2] = b.σ_H
#     utility!(b,model,∂m) #<- this is custom, too.
#     F_j!(b,model,∂m)
#     backward_induction!(model,∂m)

#     g[:] .= 0. #<- reset the gradient
#     ll = 0.
#     gσ = 0.
#     ll = log_likelihood(g,data,model,∂m,wage_log_like,b)
#     #g[6] = b.σ_W * gσ
#     g[7] *= b.σ_H
#     g[8] *= b.πW*(1-b.πW)
#     g[9] *= b.σ_W
#     return ll
# end