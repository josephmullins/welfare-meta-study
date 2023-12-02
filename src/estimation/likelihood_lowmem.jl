# define the data that we're going to use:
using .Threads, Random

function log_likelihood_threaded(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_chunk(x,p,chunk,EM,data,n_idx)
    end
    ll = fetch.(tasks)
    return sum(ll)
end
function log_likelihood_threaded_full(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    p = pars(x,p)
    return log_likelihood_threaded(x,p,EM,MD,data,n_idx)
end
function log_likelihood_threaded(x,p,fname::Vector{Symbol},ft::Vector{Int64},EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    p = pars(x,p,fname,ft)
    return log_likelihood_threaded(x,p,EM,MD,data,n_idx)
end


function log_likelihood_chunk(x,p,MD,EM::Vector{EM_data},data::Vector{likelihood_data},n_idx)
    # setup data:
    #p = pars(x,p,fname,ft)
    T = 18*4
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    R = eltype(x)
    logP = zeros(R,9,K)
    V = zeros(R,K,2)
    vj = zeros(R,9)
    ll = 0.
    for md in MD
        T = max((17-md.ageyng)*4,md.T)
        K = 2 * p.Kη * p.Kτ * md.Kω
        fill!(V,0.)
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        k_idx = LinearIndices((2,p.Kη,md.Kω,p.Kτ))
        s_inv = CartesianIndices((9,K))
        state_idx = (;K,k_inv,k_idx,s_inv)
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
            for n ∈ n_idx[md.case_idx]
                if data[n].use
                    ll += log_likelihood_η0(p,EM[n],s_inv,k_inv)
                end
            end
        end 
    end
    return ll
end


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

function log_likelihood_transitions(t,em::EM_data,p,md::model_data,data::likelihood_data,k_inv,s_inv)
    ll = 0.
    t0 = data.t0
    loc = loc_ind(md)
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        _,A,_,_,_ = j_inv(j)
        WR = md.R*A
        unemp = md.unemp[t+t0]
        _,kη,_,kτ = Tuple(k_inv[k])
        for sn in nzrange(em.q_ss[t],s_idx)
            sn_idx = em.q_ss[t].rowval[sn]
            _,kn = Tuple(s_inv[sn_idx])
            _,kη_next,_,_ = Tuple(k_inv[kn])
            f_ss = p.Fη[kη_next,kη,kτ,loc]
            wght = em.q_ss[t].nzval[sn]
            #@show t, k, kη, kη_next, wght, f_ss
            ll += wght * log(f_ss)
        end
    end
    return ll
end

function prices_log_like(t,em::EM_data,p,md::model_data,data::likelihood_data,s_inv,k_inv)
    ll = 0.
    t0 = data.t0
    if data.wage_valid[t]
        logW = data.logW[t]
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            wght = em.q_s.nzval[s]
            _,k = Tuple(s_inv[s_idx])
            _,kη,_,kτ = Tuple(k_inv[k]) 
            llW = wage_log_like(logW,p,md,kτ,kη,t+t0)
            ll += wght * llW
        end
    end
    if md.source=="SIPP" && data.chcare_valid[t] && data.pay_care[t]
        #logP = data.log_chcare[t]
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            wght = em.q_s.nzval[s]
            _,k = Tuple(s_inv[s_idx])
            _,_,_,kτ = Tuple(k_inv[k]) 
            llW = chcare_log_like(data.log_chcare[t],p,md,kτ,t+t0)
            ll += wght * llW
        end
    end
    return ll
end
