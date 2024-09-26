"""
    log_likelihood_threaded(x, p[, kτ], EM, MD, data, n_idx) 

Split the different versions of the model (given by the vector `MD`) 
into chunks and call [`log_likelihood_chunk`](@ref) over each chunk

If kτ is provided, a different method is called on each chunk that
evalutes the likelihood only for type kτ
"""
function log_likelihood_threaded(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_chunk(x,p,chunk,EM,data,n_idx)
    end
    ll = fetch.(tasks)
    return sum(ll)
end
function log_likelihood_threaded(x,p,kτ::Int64,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_chunk(x,p,kτ,chunk,EM,data,n_idx)
    end
    ll = fetch.(tasks)
    return sum(ll)
end

"""
    log_likelihood_threaded_full(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)

Call [`log_likelihood_threaded`](@ref) after first updating all parameters 
using either [`pars(x,p)`](@ref) if `fname` and `ft` are not provided as 
arguments and [`pars(x,p,fname,ft)`](@ref) if so.
"""
function log_likelihood_threaded_full(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    p = pars(x,p)
    return log_likelihood_threaded(x,p,EM,MD,data,n_idx)
end


function log_likelihood_threaded(x,p,fname::Vector{Symbol},ft::Vector{Int64},EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    p = pars(x,p,fname,ft)
    return log_likelihood_threaded(x,p,EM,MD,data,n_idx)
end

function log_likelihood_threaded(x,p,kτ::Int64,fname::Vector{Symbol},ft::Vector{Int64},EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    p = pars(x,p,kτ,fname,ft)
    return log_likelihood_threaded(x,p,kτ,EM,MD,data,n_idx)
end


"""
    log_likelihood_chunk(args...)

Iterate over the models in the vector `MD`, calculating the likelihood of each data 
object in `data` that corresponds to that model instance given the parameters `p`, 
as indicated by `n_idx`

If the argument `kτ` is additionally provided, a second is method 
is called that evaluates the likelihood for type kτ only.


"""
function log_likelihood_chunk(x,p,MD,
    EM::Vector{EM_data},data::Vector{likelihood_data},n_idx,kτ = 0)
    # create storage
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    logP,V,vj = get_model_buffer(x,K)
    ll = 0.
    for md in MD
        # reset the buffer
        fill!(V,0.)
        # get the time horizon
        T = max((17-md.ageyng)*4,md.T)
        # index the state space for this model instance
        state_idx = state_indexing_rules(p,md,kτ)
        tnow = 1
        # time varying component of likelihood
        for t in reverse(1:T)
            iterate!(logP,V,vj,p,md,t,tnow,state_idx)
            for n ∈ n_idx[md.case_idx]
                # recall that t+t0 in model time is t in data time
                if data[n].use
                    ll += log_likelihood(EM[n],md,p,logP,data[n],state_idx,t)
                end
            end
            tnow = 3 - tnow
        end
        # initial conditions
        if md.source=="SIPP"
            (;s_inv,k_inv) = state_idx
            for n ∈ n_idx[md.case_idx]
                if data[n].use
                    ll += log_likelihood_η0(p,EM[n],s_inv,k_inv)
                end
            end
        end 
    end
    return ll
end

# same as above but iterate holding type (kτ) fixed
function log_likelihood_chunk(x,p,kτ::Int64,MD,EM::Vector{EM_data},data::Vector{likelihood_data},n_idx)
    # create storage
    K = 2 * p.Kη * 9 #<- maximal state size
    logP,V,vj = get_model_buffer(x,K)
    ll = 0.
    for md in MD
        fill!(V,0.)
        T = max((17-md.ageyng)*4,md.T)
        state_idx = state_indexing_rules(p,md,kτ)
        tnow = 1
        # time varying component of likelihood
        for t in reverse(1:T)
            iterate_k!(logP,V,vj,p,md,t,tnow,state_idx)
            for n ∈ n_idx[md.case_idx]
                # recall that t+t0 in model time is t in data time
                if data[n].use
                    # add the likelihood of choices and transitions in period t
                    ll += log_likelihood(EM[n],md,p,logP,data[n],state_idx,t,kτ)
                end
            end
            tnow = 3 - tnow
        end
        # initial conditions
        if md.source=="SIPP"
            (; s_inv, k_inv) = state_idx
            for n ∈ n_idx[md.case_idx]
                if data[n].use
                    # add the likelihood of each inital η given SIPP is assumed to be steady state
                    #  need this because the transition parameters effect the steady state, which
                    #  we only assume for SIPP
                    ll += log_likelihood_η0(p,EM[n],s_inv,k_inv)
                end
            end
        end 
    end
    return ll
end

# returns storage objects for solving the model and getting the likelihood
#  x is the original argument passed to the likelihood
#  K is the maximum possible dimension of the state space
function get_model_buffer(x,K)
    R = eltype(x)
    logP = zeros(R,9,K)
    V = zeros(R,K,2)
    vj = zeros(R,9)
    return logP,V,vj
end

# returns the state indexing rules given parameters (p) and model data (md)
function state_indexing_rules(p,md,kτ = 0)
    K = 2 * p.Kη * p.Kτ * md.Kω
    # Indexing rules when iterating over states and types
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    k_idx = LinearIndices((2,p.Kη,md.Kω,p.Kτ))
    # Indexing rules when iterating over states holding type fixed
    kτ_inv = CartesianIndices((2,p.Kη,md.Kω))
    kτ_idx = LinearIndices((2,p.Kη,md.Kω))
    # Inverse indexing rule converting state*choice to state and choice
    s_inv = CartesianIndices((9,K))
    return (;K, k_inv, k_idx, s_inv, kτ_idx, kτ_inv, kτ)
end

"""
    log_likelihood(em, md, p, logP, data, state_idx, t[, kτ])

Evaluates the log-likelihood of `data` in model period `t` 
given the posterior weights stored in `em`, model primitives in `md`,
and parameters in `p`. 

# Arguments
- `state_idx` holds implied state indexing rules
- if `kτ` is additionally given, likelihood is evaluated only
  for type kτ
"""
function log_likelihood(em::EM_data,md::model_data,p,logP,data::likelihood_data,state_idx,t)
    ll = 0.
    t_data = t - data.t0
    if t_data>=1
        if t_data<=size(em.q_s,2)
            (;k_inv,s_inv) = state_idx
            ll += log_likelihood_choices(t_data,em,logP,s_inv)
            ll += prices_log_like(t_data,em,p,md,data,s_inv,k_inv)
        end
        if t_data<=length(em.q_ss)
            ll += log_likelihood_transitions(t_data,em,p,md,data,k_inv,s_inv)
        end
    end
    return ll
end

function log_likelihood(em::EM_data,md::model_data,p,logP,data::likelihood_data,state_idx,t,kτ)
    ll = 0.
    t_data = t - data.t0
    if t_data>=1
        if t_data<=size(em.q_s,2)
            (;k_inv,s_inv,kτ_idx) = state_idx
            ll += log_likelihood_choices(t_data,kτ,em,logP,s_inv,k_inv,kτ_idx)
            ll += prices_log_like(t_data,em,p,md,data,s_inv,k_inv)
        end
        if t_data<=length(em.q_ss)
            ll += log_likelihood_transitions(t_data,kτ,em,p,md,data,k_inv,s_inv)
        end
    end
    return ll
end

# - wage and childcare log likelihood routines
function wage_log_like(logW,p,md::model_data,kτ::Int64,kη::Int64,t::Int64)
    return -0.5 * ((logW - logwage(p,md,kτ,kη,t)) / p.σ_W)^2 - log(p.σ_W)
end

# - childcare price log-likelihood
# one (potential issue): no correlation between observed measurement error and whether>0. Surely correlation?
function chcare_log_like(chcare,p,md::model_data,kτ::Int64,t::Int64)
    logPF = logpriceF(p,md,kτ,t)
    resid = chcare - logPF
    ll = -0.5 * (resid / p.σ_PF)^2 - log(p.σ_PF)
    return ll
end

# evaluate the likelihood of choices
#  `em` contains weights iver choices and states
#   if choice is observed, all other choices will have zero entry in
#   the sparse weighting matrix
function log_likelihood_choices(t,em::EM_data,logP,s_inv)
    ll = 0.
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        wght = em.q_s.nzval[s]
        ll += wght * logP[j,k]
    end
    return ll
end
# same as above but do so only for type kτ
function log_likelihood_choices(t,kτ,em::EM_data,logP,s_inv,k_inv,kτ_idx)
    ll = 0.
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        kA,kη,kω,kτ2 = Tuple(k_inv[k])
        if kτ2==kτ
            k = kτ_idx[kA,kη,kω]        
            wght = em.q_s.nzval[s]
            ll += wght * logP[j,k]
        end
    end
    return ll
end

# evaluate the join likelihood of all potential transitions that 
#   are assigned positive weight by `em`
function log_likelihood_transitions(t,em::EM_data,p,md::model_data,data::likelihood_data,k_inv,s_inv)
    ll = 0.
    t0 = data.t0
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        _,A,_,_,_ = j_inv(j)
        jF = 1 + md.R*A
        _,kη,_,kτ = Tuple(k_inv[k])
        for sn in nzrange(em.q_ss[t],s_idx)
            sn_idx = em.q_ss[t].rowval[sn]
            _,kn = Tuple(s_inv[sn_idx])
            _,kη_next,_,_ = Tuple(k_inv[kn])
            f_ss = p.Fη[kη_next,kη,jF,kτ]
            wght = em.q_ss[t].nzval[sn]
            #@show t, k, kη, kη_next, wght, f_ss
            ll += wght * log(f_ss)
        end
    end
    return ll
end
# same as above but only for type kτ
function log_likelihood_transitions(t,kτ,em::EM_data,p,md::model_data,data::likelihood_data,k_inv,s_inv)
    ll = 0.
    t0 = data.t0
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        _,A,_,_,_ = j_inv(j)
        jF = 1 + md.R*A
        _,kη,_,kτ2 = Tuple(k_inv[k])
        if kτ2==kτ
            for sn in nzrange(em.q_ss[t],s_idx)
                sn_idx = em.q_ss[t].rowval[sn]
                _,kn = Tuple(s_inv[sn_idx])
                _,kη_next,_,_ = Tuple(k_inv[kn])
                f_ss = p.Fη[kη_next,kη,jF,kτ]
                wght = em.q_ss[t].nzval[sn]
                #@show t, k, kη, kη_next, wght, f_ss
                ll += wght * log(f_ss)
            end
        end
    end
    return ll
end

# weighted log-likelihood of each initial condition kη given type 
#   assuming stationary distribution
#   this is needed for SIPP observations
function log_likelihood_η0(p,em::EM_data,s_inv,k_inv)
    ll = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        _,kη,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        llk = log(p.πₛ[kη,kτ])
        ll += llk*wght
    end
    return ll
end

# log-likelihood of initial conidition kη given type and weights in em
#   this is needed for the MDRC observations
function log_likelihood_η0_full(p,md::model_data,em::EM_data,s_inv,k_inv)
    ll = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        kA,kη,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        if wght>eps(Float64)
            if md.source=="SIPP"
                llk = log(p.πₛ[kη,kτ])
            else
                loc = 1(md.source=="FTP") + 2(md.source=="CTJF") + 3(md.source=="MFIP")
                llk = log(p.πη[kA,kη,kτ,loc])
            end
            ll += llk*wght
        end
    end
    return ll
end



# writes each answer to a vector LL, for calculating the score
function log_likelihood_n(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    N = sum(length(n_idx[md.case_idx]) for md in MD) #<-?
    p = pars_full(x,p)
    LL = zeros(eltype(x),N)
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_n_chunk!(LL,p,EM,chunk,data,n_idx)
    end
    f = fetch.(tasks)
    return LL
end

function log_likelihood_n_chunk!(LL,p,EM,MD,data,n_idx)
    # create storage
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    logP,V,vj = get_model_buffer(LL,K)

    # get additional buffer for type probabilities
    R = eltype(LL)
    logπτ = zeros(R,p.Kτ)

    for md in MD
        fill!(V,0.)
        T = max((17-md.ageyng)*4,md.T)
        state_idx = state_indexing_rules(p,md)
        tnow = 1
        # time varying component of likelihood
        for t in reverse(1:T)
            iterate!(logP,V,vj,p,md,t,tnow,state_idx)
            for n ∈ n_idx[md.case_idx]
                # recall that t+t0 in model time is t in data time
                if data[n].use
                    LL[n] += log_likelihood(EM[n],md,p,logP,data[n],state_idx,t)
                end
            end
            tnow = 3 - tnow
        end
        # initial conditions
        if md.source=="SIPP"
            (; s_inv, k_inv) = state_idx
            for n ∈ n_idx[md.case_idx]
                if data[n].use
                    @views LL[n] += log_likelihood_type(logπτ,
                        p.βτ[md.type_block,:],EM[n],data[n],s_inv,k_inv)
                    LL[n] += log_likelihood_η0_full(p,md,EM[n],s_inv,k_inv)
                end
            end
        end 
    end
    return nothing
end

