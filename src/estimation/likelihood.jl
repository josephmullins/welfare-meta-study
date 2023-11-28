# define the data that we're going to use:
using .Threads, Random


# writes each answer to a vector LL, for calculating the score
function log_likelihood_n(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    N = sum(length(n_idx[md.case_idx]) for md in MD) #<-?
    p = pars_full(x,p)
    LL = zeros(eltype(x),N)
    @threads for md in MD
        (;vj,V,logP) = get_model(p)
        solve!(logP,V,vj,p,md)
        logπτ = zeros(eltype(x),p.Kτ)
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        K = prod(size(k_inv))
        s_inv = CartesianIndices((9,K))
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                LL[n] = log_likelihood(EM[n],md,p,logP,data[n])
                @views LL[n] += log_likelihood_type(logπτ,p.βτ[md.type_block,:],EM[n],data[n],s_inv,k_inv)
                LL[n] += log_likelihood_η0_full(p,md,EM[n],s_inv,k_inv) #<- probably still a rank issue here
            end
        end
    end
    return LL
end

# for some reason passing the data type leads to lots of additional allocations
# in test_allocations.jl, we see it comes from assigning to the arrays using other arguments
# one option is to pass through to here and use eltype(x) and update inside this chunk.
function log_likelihood_chunk(x,p,MD,EM::Vector{EM_data},data::Vector{likelihood_data},n_idx)
    # setup data:
    T = 18*4
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    R = eltype(x)
    logP = zeros(R,9,K,T)
    V = zeros(R,K,2)
    vj = zeros(R,9)
    ll = 0.
    for md in MD
        solve!(logP,V,vj,p,md)
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += log_likelihood(EM[n],md,p,logP,data[n])
            end
        end
    end
    return ll
end

function log_likelihood(em::EM_data,md::model_data,p,logP,data::likelihood_data)
    J = 9 #<- pass as parameter instead?
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((J,K))

    ll = 0.
    ll += log_likelihood_choices(em,data.t0,logP,s_inv)
    ll += log_likelihood_transitions(em,p,md,data,k_inv,s_inv)
    ll += prices_log_like(em,p,md,data,J,s_inv,k_inv)
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

function log_likelihood_η0_full(p,md::model_data,em::EM_data,s_inv,k_inv)
    ll = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        kA,kη,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        loc = (md.source=="FTP") + 2(md.source=="CTJF") + 3(md.source=="MFIP") + 4(md.source=="SIPP")
        llk = log(p.πη[kA,kη,kτ,loc])
        ll += llk*wght
    end
    return ll
end


function log_likelihood_choices(em::EM_data,t0,logP,s_inv)
    ll = 0.
    for t in axes(em.q_s,2)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            wght = em.q_s.nzval[s]
            ll += wght * logP[j,k,t+t0]
        end
    end
    return ll
end

function log_likelihood_transitions(em::EM_data,p,md::model_data,data::likelihood_data,k_inv,s_inv)
    ll = 0.
    Kη = size(k_inv,2)
    t0 = data.t0
    for t in eachindex(em.q_ss)
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
                f_ss = fη(kη_next,kη,kτ,WR,unemp,p)
                wght = em.q_ss[t].nzval[sn]
                #@show t, k, kη, kη_next, wght, f_ss
                ll += wght * log(f_ss)
            end
        end
    end
    return ll
end

function prices_log_like(em::EM_data,p,md::model_data,data::likelihood_data,J::Int64,s_inv,k_inv)
    ll = 0.
    t0 = data.t0
    for t in eachindex(data.logW)
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
    end
    return ll
end
