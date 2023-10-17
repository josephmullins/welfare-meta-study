#include("../tools/ModelStatistics.jl")
using DataFrames, Statistics, CSV

function em_mean(Π,t::Int64,f::Function,condition::Function = (x->true))
    m = 0.
    d = 0.
    if issparse(Π)
        for s in nzrange(Π,t)
            s_idx = Π.rowval[s]
            if condition(s_idx)
                wght = Π[s_idx,t]
                m += wght * f(s_idx)
                d += wght
            end
        end
    else
        for s in axes(Π,1)
            if condition(s)
                wght = Π[s,t]
                m += wght * f(s)
                d += wght
            end
        end
    end

    return m / d
end

function f_work(s_idx,t,logP,s_inv,k_inv)
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    EH = 0
    for j in choice_set(kη>1)
        _,_,_,H,_ = j_inv(j)
        EH += H * exp(logP[j,k,t])
    end
    return EH
    #return H
end
function f_afdc(s_idx,t,logP,s_inv)
    j,k = Tuple(s_inv[s_idx])
    #_,A,_,_,_ = j_inv(j)
    EA = 0
    for j in 1:9
        _,A,_,_,_ = j_inv(j)
        EA += A * exp(logP[j,k,t])
    end
    return EA
end
function f_earn(s_idx,t,logP,s_inv,k_inv,pars,md::model_data)
    EW = 0.
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    if kη==1
        return 0.
    else
        W = exp(logwage(pars,md,kτ,kη,t))
        for j in 1:9
            _,_,_,H,_ = j_inv(j)
            EW += H * exp(logP[j,k,t]) * W
        end    
    end
    return EW  #<- earnings
end
function job_offer(s_idx,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    return kη>1
end

function log_full(s_idx,t,logP,s_inv,k_inv,pars,md::model_data)
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)
    if kη>1
        W = exp(logwage(pars,md,kτ,kη,t))
    else
        W = 0.
    end
    kid_developing = (md.ageyng+fld(t,4))<17
    prF = kid_developing * exp(logpriceF(pars,md,kτ,t))
    year = min(2010,md.y0 + fld(md.q0 + t-1,4)) #<- assume expected policy environment is fixed beyond 2010
    Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + A)
    full_income = p.wq*(112-30H) + max(Y - prF*F,0.)
    #full_income = 0.0001*(112-30H) + max(Y - prF*F,0.)
    return log(full_income)
end
function unpaid_care(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    S,A,P,H,F = j_inv(j)
    return H*(1-F)
end
function paid_care(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    S,A,P,H,F = j_inv(j)
    return F
end

function basic_model_fit(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,fname = "model_stats.csv",write = true)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn basic_model_fit_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, 
        #:LOGINC => (x->Statistics.mean(x[.!isnan.(x)])) => :LOGINC, 
        :LOGFULL => Statistics.mean => :LOGFULL)
    if write
        CSV.write(string("output/",fname),d)
    end
    return d
end

function basic_model_fit_chunk(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    (;V,vj,logP) = get_model(p)
    D = DataFrame(source = [],arm = [],year=[],Q=[],EMP=[],AFDC=[],case_idx = [],n_idx = [],EARN = [], est_sample = [],app_status = [], LOGFULL = [])
    for md in MD
        solve!(logP,V,vj,p,md)
        #println(md.case_idx)
        for n in n_idx[md.case_idx]
            d = model_stats(p,logP,EM[n],md,data[n])
            d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    return D
end

function model_stats(p,logP,em::EM_data,md::model_data,data::likelihood_data)
    T = size(em.q_s,2)
    Q = 0:T-1
    H = zeros(T)
    A = zeros(T)
    E = zeros(T)
    l_full = zeros(T)
    
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((9,K))

    # year = md.y0 .+ fld.(md.q0-1 .+ (0:T-1),4)
    # Q = mod.(md.q0-1 .+ (0:T-1),4) .+ 1
    year = md.y0 .+ fld.(md.q0 .+ (0:T-1),4) #<- start with one quarter delay in this model.
    Q = mod.(md.q0 .+ (0:T-1),4) #<- as above
    for t in 1:T
        H[t] = em_mean(em.q_s,t,s->f_work(s,t,logP,s_inv,k_inv))
        A[t] = em_mean(em.q_s,t,s->f_afdc(s,t,logP,s_inv))
        E[t] = em_mean(em.q_s,t,s->f_earn(s,t,logP,s_inv,k_inv,p,md)) #,x->job_offer(x,s_inv,k_inv))
        l_full[t] = em_mean(em.q_s,t,s->log_full(s,t,logP,s_inv,k_inv,p,md))
    end
    keep = .!data.choice_missing
    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[6] - data.X_type[7]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[6]
    else
        app_status = 0
    end
    return DataFrame(source = md.source, arm = md.arm, year = year[keep], Q = Q[keep], EMP = H[keep], AFDC = A[keep],case_idx = md.case_idx,
        EARN = E[keep],est_sample = data.use,app_status = app_status,LOGFULL = l_full[keep])
end