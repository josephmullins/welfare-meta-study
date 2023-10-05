include("../tools/ModelStatistics.jl")
using DataFrames, Statistics, CSV

function f_work(s_idx,t,model,pars,data)
    s_inv = CartesianIndices((model.J,model.K))
    j,k = Tuple(s_inv[s_idx])
    #_,_,_,H,_ = j_inv(j)
    EH = 0
    for j in 1:model.J
        if model.choice_set[j,k,t]
            _,_,_,H,_ = j_inv(j)
            EH += H * exp(model.logP[j,k,t])
        end
    end
    return EH
    #return H
end
function f_afdc(s_idx,t,model,pars,data)
    s_inv = CartesianIndices((model.J,model.K))
    j,k = Tuple(s_inv[s_idx])
    #_,A,_,_,_ = j_inv(j)
    EA = 0
    for j in 1:model.J
        if model.choice_set[j,k,t]
            _,A,_,_,_ = j_inv(j)
            EA += A * exp(model.logP[j,k,t])
        end
    end
    return EA
    #return A
end
function f_earn(s_idx,t,model,pars,data)
    EW = 0.
    s_inv = CartesianIndices((model.J,model.K))
    j,k = Tuple(s_inv[s_idx])
    k_inv = CartesianIndices((2,pars.Kη,data.Kω,pars.Kτ))
    _,kη,_,kτ = Tuple(k_inv[k])
    if kη==1
        return 0.
    else
        W = exp(logwage(pars,data,kτ,kη,t))
        for j in 1:model.J
            if model.choice_set[j,k,t]
                _,_,_,H,_ = j_inv(j)
                EW += H * exp(model.logP[j,k,t]) * W
            end
        end    
    end
    return EW  #<- earnings
end
function job_offer(s_idx,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    return kη>1
end
function log_inc(s_idx,t,model,p::pars,md::model_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
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
    #Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + A)
    Y,_ = Transfers.budget(W*H,0.,md.SOI,year,md.numkids,md.cpi[min(end,t)],P)
    full_income = 0.0001 + Y # max(Y - prF*F,0.)
    return log(full_income)
end
function log_earn(s_idx,t,model,p,md)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)
    if kη>1
        W = logwage(p,md,kτ,kη,t)
    else
        W = 0.
    end
    return W
end
function work(s_idx,s_inv)
    j,k = Tuple(s_inv[s_idx])
    _,_,_,H,_ = j_inv(j)
    return H==1
end
function benefits(s_idx,t,model,p,md)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)
    if kη>1
        W = exp(logwage(p,md,kτ,kη,t))
    else
        W = 0.
    end
    kid_developing = (md.ageyng+fld(t,4))<17
    #prF = kid_developing * exp(logpriceF(p,md,kτ,t))
    year = min(2010,md.y0 + fld(md.q0 + t-1,4)) #<- assume expected policy environment is fixed beyond 2010
    cpi = md.cpi[min(end,t)]
    if P>0
        if P>1 && md.numkids>0
            tanf,_ = Transfers.TANF(cpi*H*W,0.,md.SOI,year,md.numkids)
        else 
            tanf = 0.
        end
        snap,_,_ = Transfers.SNAP(cpi*H*W,0.,year,md.numkids,tanf)
    else
        tanf = 0.
        snap = 0.
    end
    return W + (tanf+snap)/cpi
end

function log_full(s_idx,t,model,p::pars,md::model_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
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
    Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + A)
    full_income = p.wq*(112-30H) + max(Y - prF*F,0.)
    #full_income = 0.0001*(112-30H) + max(Y - prF*F,0.)
    return log(full_income)
end
function unpaid_care(s_idx,t,model,p::pars,md::model_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)
    return H*(1-F)
end
function paid_care(s_idx,t,model,p::pars,md::model_data)
    s_inv = CartesianIndices((model.J,model.K))
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,kτ = Tuple(k_inv[k])
    S,A,P,H,F = j_inv(j)
    return F
end

function basic_model_fit(p::pars,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,fname = "model_stats.csv")
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn basic_model_fit_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, 
        :LOGINC => (x->Statistics.mean(x[.!isnan.(x)])) => :LOGINC, :LOGFULL => Statistics.mean => :LOGFULL)
    CSV.write(string("output/",fname),d)
    return d
end

function basic_model_fit_chunk(p::pars,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    M = update(p,(1,8,9))
    D = DataFrame(source = [],arm = [],year=[],Q=[],EMP=[],AFDC=[],case_idx = [],n_idx = [],EARN = [], est_sample = [],app_status = [],LOGINC = [], LOGFULL = [])
    for md in MD
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))
        solve!(M[m_idx],md,p)
        reduce_choice_probabilities!(M[m_idx])
        #println(md.case_idx)
        for n in n_idx[md.case_idx]
            d = model_stats(EM[n],md,M[m_idx],data[n])
            d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    return D
end

function model_stats(em::EM_data,md::model_data,m::ddc_model,data::likelihood_data)
    T = size(em.q_s,2)
    Q = 0:T-1
    H = zeros(T)
    A = zeros(T)
    E = zeros(T)
    l_inc = zeros(T)
    l_full = zeros(T)
    
    s_inv = CartesianIndices((m.J,m.K))
    #k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))

    # year = md.y0 .+ fld.(md.q0-1 .+ (0:T-1),4)
    # Q = mod.(md.q0-1 .+ (0:T-1),4) .+ 1
    year = md.y0 .+ fld.(md.q0 .+ (0:T-1),4) #<- start with one quarter delay in this model.
    Q = mod.(md.q0 .+ (0:T-1),4) #<- as above
    for t in 1:T
        H[t] = em_mean(em.q_s,t,m,nothing,nothing,f_work)
        A[t] = em_mean(em.q_s,t,m,nothing,nothing,f_afdc)
        E[t] = em_mean(em.q_s,t,m,p,md,f_earn) #,x->job_offer(x,s_inv,k_inv))
        #l_inc[t] = em_mean(em.q_s,t,m,p,md,log_inc)
        l_inc[t] = em_mean(em.q_s,t,m,p,md,benefits)
        #l_inc[t] = 0.0001 + em_mean(em.q_s,t,m,p,md,benefits)
        #l_inc[t] = em_mean(em.q_s,t,m,p,md,log_earn,x->work(x,s_inv))
        l_full[t] = em_mean(em.q_s,t,m,p,md,log_full)
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
        EARN = E[keep],est_sample = data.use,app_status = app_status,
        LOGINC=l_inc[keep],LOGFULL = l_full[keep])
end