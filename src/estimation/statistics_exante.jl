#include("../tools/ModelStatistics.jl")
using DataFrames, Statistics, CSV

function get_choice_state_distribution!(Π::SparseMatrixCSC{Float64,Int64},logP,model,p) #<- need choice probs? T?
    fill!(Π,0.)
    J = 9
    (;K,π0,s_inv,k_inv,Kω,k_idx) = model
    T = size(Π,2)
    # initialize the first period:
    for k in 1:K
        if π0[k]>0
            _,kη,_,_ = Tuple(k_inv[k])
            for j in choice_set(kη>1)
                s = (k-1)*J + j
                Π[s,1] = π0[k] * exp(logP[j,k,1])
            end
        end
    end
    # iterate forward until the last period
    for t in 1:T-1 #
        for s in nzrange(Π,t)
            s_idx = Π.rowval[s]
            qs = Π.nzval[s]
            j,k = Tuple(s_inv[s_idx])
            _,kη,kω,kτ = Tuple(k_inv[k])
            _,A,_,_,_ = j_inv(j)
            kω_next = min(kω+A,Kω)
            kA_next = 1 + A
            for kη_next in 1:p.Kη
                fkk = p.Fη[kη_next,kη,kτ]
                kn = k_idx[kA_next,kη_next,kω_next,kτ]
                for jn in choice_set(kη_next>1)
                    sn = (kn - 1)*J + jn
                    Π[sn,t+1] += fkk * exp(logP[jn,kn,t+1]) * qs
                end
            end
        end
    end
end

function f_work(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    _,_,_,H,_ = j_inv(j)
    return H
end
function f_afdc(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    _,A,_,_,_ = j_inv(j)
    return A
end
function f_earn(s_idx,t,s_inv,k_inv,pars,md::model_data)
    EW = 0.
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    if kη==1
        return 0.
    else
        W = exp(logwage(pars,md::model_data,kτ,kη,t))
        _,_,_,H,_ = j_inv(j)
        return H * W
    end
end
function log_full(s_idx,t,s_inv,k_inv,p,md::model_data)
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)

    if kη>1
        W = exp(logwage(p,md,kτ,kη,t))
    else
        W = 0.
    end
    kid_developing = (md.ageyng+fld(t,4))<17
    prF = kid_developing * exp(logpriceF(p,md,kτ,t))
    year = min(2010,md.y0 + fld(md.q0 + t-1,4)) #<- assume expected policy environment is fixed beyond 2010
    Y,_ = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + A)
    full_income = p.wq*(112-30H) + max(Y - prF*F,0.)
    return log(full_income)
end

function basic_model_fit(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,fname = "model_stats.csv",write = true)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn basic_model_fit_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, :LOGFULL => Statistics.mean => :LOGFULL)
    if write
        CSV.write(string("output/",fname),d)
    end
    return d
end

function basic_model_fit_chunk(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    D = DataFrame(source = [],arm = [],year=[],Q=[],EMP=[],AFDC=[],case_idx = [],n_idx = [],EARN = [], est_sample = [],app_status = [], LOGFULL = [])
    logπτ = zeros(p.Kτ)
    (;V,vj,logP) = get_model(p)

    for md in MD
        solve!(logP,V,vj,p,md)
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        k_idx = LinearIndices((2,p.Kη,md.Kω,p.Kτ))
        K = prod(size(k_inv))
        Kω = md.Kω
        s_inv = CartesianIndices((9,K))
        π0 = zeros(K)
        #println(md.case_idx)
        for n in n_idx[md.case_idx]
            #initialize!(logπτ,EM[n],π0,p,md,data[n],(;k_inv,s_inv)) #<- get initial dist from priors
            initialize!(π0,EM[n],s_inv) #<- get initial dist from posterior
            get_choice_state_distribution!(EM[n].q_s,logP,(;K,π0,s_inv,k_inv,Kω,k_idx),p) #<- nice.
            d = model_stats(p,EM[n],md,data[n])
            d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    return D
end

function model_stats(p,em::EM_data,md::model_data,data::likelihood_data)
    T = size(em.q_s,2)
    Q = 0:T-1
    H = zeros(T)
    A = zeros(T)
    E = zeros(T)
    l_inc = zeros(T)
    l_full = zeros(T)
    
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((9,K))

    # year = md.y0 .+ fld.(md.q0-1 .+ (0:T-1),4)
    # Q = mod.(md.q0-1 .+ (0:T-1),4) .+ 1
    year = md.y0 .+ fld.(md.q0 .+ (0:T-1),4) #<- start with one quarter delay in this model.
    Q = mod.(md.q0 .+ (0:T-1),4) #<- as above
    for t in 1:T
        H[t] = em_mean(em.q_s,t,s->f_work(s,s_inv))
        A[t] = em_mean(em.q_s,t,s->f_afdc(s,s_inv))
        E[t] = em_mean(em.q_s,t,s->f_earn(s,t,s_inv,k_inv,p,md)) #,x->job_offer(x,s_inv,k_inv))
        l_full[t] = em_mean(em.q_s,t,s->log_full(s,t,s_inv,k_inv,p,md))
    end
    keep = .!data.choice_missing
    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end
    return DataFrame(source = md.source, arm = md.arm, year = year[keep], Q = Q[keep], EMP = H[keep], AFDC = A[keep],case_idx = md.case_idx,
        EARN = E[keep],est_sample = data.use,app_status = app_status,LOGFULL = l_full[keep])
end

function initialize!(logπτ,EM::EM_data,π0,p,md::model_data,data::likelihood_data,idx)
    J = 9
    (;k_inv,s_inv) = idx
    fill!(π0,0.)
    log_type_prob!(logπτ,p,md,data)

    # start with the initial conditions:
    for s in nzrange(EM.α,1)
        s_idx = EM.α.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        kA,kη,_,kτ = Tuple(k_inv[k])
        if π0[k]==0
            π0[k] = initial_prob(kA,kη,kτ,logπτ,p,md)
        end
    end
    π0 ./= sum(π0)
    return nothing
end
# fill π0 states with the posterior distribution instead
function initialize!(π0,EM::EM_data,s_inv)
    fill!(π0,0.)
    # start with the initial conditions:
    for s in nzrange(EM.q_s,1)
        s_idx = EM.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        π0[k] += EM.q_s[s_idx,1]
    end
    #π0 ./= sum(π0)
    return nothing
end

