# define the data that we're going to use:
using .Threads, Random

function log_likelihood_threaded(x,LL::Vector{Float64},M::Matrix{ddc_model},EM::Vector{EM_data},MD::Vector{model_data},p::pars,data::Vector{likelihood_data},n_idx)
    update!(x,p)
    update!(p,M,(1,8,9))
    fill!(LL,0.)
    @threads :static for i in eachindex(MD)
        md = MD[i]
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))
        # update the number of time periods, the choice set, the utilities, and solve.
        solve!(M[m_idx,threadid()],md,p)
        ll = 0.
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += log_likelihood(EM[n],md,p,M[m_idx,threadid()],data[n])
            end
        end
        LL[threadid()] += ll
    end
    return sum(LL)
end

function log_likelihood_threaded(x,G::Matrix{Float64},LL::Vector{Float64},M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},p::pars,data::Vector{likelihood_data},n_idx)
    fill!(G,0.)
    fill!(LL,0.)
    update!(x,p)
    update!(p,M,∂M,(1,8,9))
    @threads :static for i in eachindex(MD) # in MD
        md = MD[i]
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))

        # ------- Step 1: get the log-likelihood of choices --------
        # to save on memory allocations we to do this for each t over all observations that fit this case, then we update and discard data from t.

        @views ll = log_likelihood_choices(p,G[:,threadid()],M[m_idx,threadid()],∂M[m_idx,threadid()],EM,data,md,n_idx)

        # ------- Step 2: get the log-likelihood of prices and transitions ----- #
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                @views ll += log_likelihood(G[:,threadid()],EM[n],md,p,M[m_idx,threadid()],∂M[m_idx,threadid()],data[n])
            end
        end
        LL[threadid()] += ll
    end
    for it in axes(G,2)
        @views apply_chain_rule!(G[:,it],p,∂M[1,1].σ_idx,∂M[1,1].β_idx)
    end
    return sum(LL)
end

# this function combines model solution with evaluation of choice probabilities
# it reduces memeory requirements in ∂m.logP
function log_likelihood_choices(p::pars,g,m::ddc_model,∂m::ddc_derivative,EM::Vector{EM_data},data::Vector{likelihood_data},md::model_data,n_idx)
    T = max((17-md.ageyng)*4,md.T)
    fill!(m.V,0.)
    fill!(∂m.v,0.)
    fill!(∂m.V,0.)
    @views fill!(∂m.logP,0.)
    tnow = 1
    ll = 0.
    for t in reverse(1:T)
        tnext = 3-tnow #<- tnow = 1 => tnext =2, tnow=2 => tnext =1
        solve!(m,∂m,md,p,t,tnow,tnext)
        # swap locations of next period and current period for V 
        # if tnow = 1, tnext
        tnow = tnext #<- if 1, moves to 2, if 2, moves to 1.

        # evaluate the likelihood of choices:
        for n ∈ n_idx[md.case_idx]
            if data[n].use && (t<=data[n].T)
                @views ll += log_likelihood_choices(g,EM[n],m,∂m,t)
            end
        end

    end
    return ll
end

function prices_log_likelihood(x,EM::Vector{EM_data},MD,p::pars,data::Vector{likelihood_data},n_idx,J::Int64)
    update!(x,p)
    ll = 0.
    for md ∈ MD #
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))
        s_inv = CartesianIndices((J,M[m_idx].K))
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += prices_log_like(EM[n],p,md,data[n],J,s_inv,k_inv)
            end
        end
    end
    return ll
end
function prices_log_likelihood(x,g,EM::Vector{EM_data},MD,p::pars,data::Vector{likelihood_data},n_idx,J::Int64)
    update!(x,p)
    ll = 0.
    for md ∈ MD #
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))
        # update the number of time periods, the choice set, the utilities, and solve.
        s_inv = CartesianIndices((J,M[m_idx].K))
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                ll += prices_log_like(g,EM[n],p,md,data[n],J,s_inv,k_inv)
            end
        end
    end
    return ll
end

function log_likelihood(em::EM_data,md::model_data,p::pars,model::ddc_model,data::likelihood_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))

    ll = 0.
    ll += log_likelihood_choices(em,model)
    ll += log_likelihood_transitions(em,model)
    ll += prices_log_like(em,p,md,data,model.J,s_inv,k_inv)
    if md.source=="SIPP"
        ll += log_likelihood_η0(p,em,s_inv,k_inv)
    end 
    return ll
end

function log_likelihood(g,em::EM_data,md::model_data,p::pars,model::ddc_model,∂m::ddc_derivative,data::likelihood_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))

    ll = 0.
    ll += log_likelihood_transitions(g,em,model,∂m)
    ll += prices_log_like(g,em,p,md,data,model.J,s_inv,k_inv) #<
    if md.source=="SIPP"
        ll += log_likelihood_η0(g,p,em,∂m.F_idx,s_inv,k_inv)
    end
    return ll
end


# - wage and childcare log likelihood routines
function wage_log_like(logW,p::pars,md::model_data,kτ::Int64,kη::Int64,t::Int64)
    return -0.5 * ((logW - logwage(p,md,kτ,kη,t)) / p.σ_W)^2 - log(p.σ_W)
end

function wage_log_like(g,logW,p::pars,md::model_data,kτ::Int64,kη::Int64,t::Int64,weight = 1.)
    resid = logW - logwage(p,md,kτ,kη,t)
    ll = -0.5 * (resid / p.σ_W)^2 - log(p.σ_W)
    pos = 5p.Kτ + 7
    dl = weight * resid / p.σ_W^2
    g[pos+kτ-1] += dl
    pos += p.Kτ
    g[pos] += dl * md.unemp[min(end,t)]
    g[pos+1] += dl * ((md.age0-18)*4 + t)
    pos += 12+p.Kτ
    g[pos] += dl * p.ηgrid[kη-1] 
    return ll
end

# - childcare price log-likelihood
function chcare_log_like(chcare,p::pars,md::model_data,kτ::Int64,t::Int64)
    resid = chcare - logpriceF(p,md,kτ,t)
    ll = -0.5 * (resid / p.σ_PF)^2 - log(p.σ_PF)
    return ll
end
function chcare_log_like(g,chcare,p::pars,md::model_data,kτ::Int64,t::Int64,weight = 1.)
    resid = chcare - logpriceF(p,md,kτ,t)
    ll = -0.5 * (resid / p.σ_PF)^2 - log(p.σ_PF)
    pos = 6p.Kτ+9
    dl = weight * resid / p.σ_PF^2
    g[pos+kτ-1] += dl
    pos += p.Kτ
    g[pos] += dl * md.unemp[min(t,end)]
    g[pos+1] += dl * md.numkids
    g[pos+2] += dl * (md.ageyng+fld(t-1,4)<=5)
    g[pos+3+md.loc_ind-1] += dl * (md.loc_ind>0)
    return ll
end

function prices_log_like(em::EM_data,p::pars,md::model_data,data::likelihood_data,J::Int64,s_inv,k_inv)
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
        if data.chcare_valid[t] && data.chcare[t]>0
            logP = data.log_chcare[t]
            for s in nzrange(em.q_s,t)
                s_idx = em.q_s.rowval[s]
                wght = em.q_s.nzval[s]
                _,k = Tuple(s_inv[s_idx])
                _,_,_,kτ = Tuple(k_inv[k]) 
                llW = chcare_log_like(logP,p,md,kτ,t+t0)
                ll += wght * llW
            end
        end
    end
    return ll
end

# same version as above but with derivative
function prices_log_like(g,em::EM_data,p::pars,md::model_data,data::likelihood_data,J::Int64,s_inv,k_inv)
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
                llW = wage_log_like(g,logW,p,md,kτ,kη,t+t0,wght)
                ll += wght * llW
            end
        end
        if data.chcare_valid[t] && data.chcare[t]>0
            logP = data.log_chcare[t]
            for s in nzrange(em.q_s,t)
                s_idx = em.q_s.rowval[s]
                wght = em.q_s.nzval[s]
                _,k = Tuple(s_inv[s_idx])
                _,_,_,kτ = Tuple(k_inv[k]) 
                llW = chcare_log_like(g,logP,p,md,kτ,t+t0,wght)
                ll += wght * llW
            end
        end
    end
    return ll
end
# - childcare expenditure log-likelihood routines

# doesn't look like this anymore!!! is this why?
function log_likelihood_η0(kτ,kη,p::pars)
    h = p.λ[kτ] / (p.δ[kτ] + p.λ[kτ])
    if kη==1
        return log(1-h)
    else
        return log(h)
    end
end

function log_likelihood_η0(g,kτ,kη,p::pars,F_idx,weight = 1.)
    δ = p.δ[kτ]
    λ = p.λ[kτ]
    denom = (δ + λ)
    h = λ / denom
    dλ = λ * (1-λ) #<- apply this because λ is a logit transform
    dδ = δ * (1-δ) #<- same as above
    if kη==1
        g[F_idx[kτ]] += weight * - 1/denom * dλ
        g[F_idx[p.Kτ+kτ]] += weight * ( 1/δ - 1/denom ) * dδ
        return log(1-h)
    else
        g[F_idx[kτ]] += weight * ( 1/λ - 1/denom ) * dλ
        g[F_idx[p.Kτ+kτ]] += weight * -1/denom * dδ
        return log(h)
    end
end
function log_likelihood_η0(p::pars,em::EM_data,s_inv,k_inv)
    ll = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        _,kη,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        llk = log_likelihood_η0(kτ,kη,p)
        ll += llk*wght
    end
    return ll
end
function log_likelihood_η0(g,p::pars,em::EM_data,F_idx::UnitRange{Int64},s_inv,k_inv)
    ll = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        _,kη,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        llk = log_likelihood_η0(g,kτ,kη,p,F_idx,wght)
        ll += llk*wght
    end
    return ll
end

# now do as above but for data coming from an EM algorithm:
# - here the choices and the states are partially unobserved
function log_likelihood_choices(g,em::EM_data,m::ddc_model,∂m::ddc_derivative,t::Int64)
    s_inv = CartesianIndices((m.J,m.K))
    ll = 0.
    for s in nzrange(em.q_s,t)
        s_idx = em.q_s.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        wght = em.q_s.nzval[s]
        @views ll += log_likelihood(j,g,m.logP[:,k,t],∂m.logP[:,:,k],m.G,wght)
    end
    return ll
end
function log_likelihood_choices(em::EM_data,m::ddc_model)
    s_inv = CartesianIndices((m.J,m.K))
    ll = 0.
    for t in axes(em.q_s,2)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            wght = em.q_s.nzval[s]
            @views ll += log_likelihood(j,m.logP[:,k,t],m.G,wght)
        end
    end
    return ll
end


# same as above: two versions of the likelihood that use EM_data instead (states unobservable)
# function to evaluate log-likelihood of transitions
function log_likelihood_transitions(g,em::EM_data,m::ddc_model,∂m::ddc_derivative)
    s_inv = CartesianIndices((m.J,m.K))
    ll = 0.
    for t in eachindex(em.q_ss)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            for sn in nzrange(em.q_ss[t],s_idx)
                sn_idx = em.q_ss[t].rowval[sn]
                jn,kn = Tuple(s_inv[sn_idx])
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
    s_inv = CartesianIndices((m.J,m.K))
    ll = 0.
    for t in eachindex(em.q_ss)
        for s in nzrange(em.q_s,t)
            s_idx = em.q_s.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            for sn in nzrange(em.q_ss[t],s_idx)
                sn_idx = em.q_ss[t].rowval[sn]
                jn,kn = Tuple(s_inv[sn_idx])
                f_ss = m.F[j,t][kn,k]
                wght = em.q_ss[t].nzval[sn]
                ll += wght * log(f_ss)
            end
        end
    end
    return ll
end