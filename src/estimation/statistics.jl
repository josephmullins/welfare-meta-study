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

# calculated E[work | k] where k is extracted from s
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
# invert out the actual work choice given s
function f_work(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    _,_,_,H,_ = j_inv(j)
    return H
end

# calculate E[afdc | k] where k is extracted from s
function f_afdc(s_idx,t,logP,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    #_,A,_,_,_ = j_inv(j)
    EA = 0
    for j in choice_set(kη>1)
        _,A,_,_,_ = j_inv(j)
        EA += A * exp(logP[j,k,t])
    end
    return EA
end
# invert out the actual welfare choice given s
function f_afdc(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    _,A,_,_,_ = j_inv(j)
    return A
end

# calculate E[earn | k] where k is extracted from s
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
# invert out actual earnings given s
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

# invert out whether the individual *can* work given s
function job_offer(s_idx,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    return kη>1
end

# invert out full income given s
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
    full_income = p.wq[kτ] + max(Y - prF*F,0.)
    return log(full_income)
end

# invert out whether unpaid or paid care is used given s
function unpaid_care(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    S,A,P,H,F = j_inv(j)
    return H #*(1-F) <- a small edit.
end
function paid_care(s_idx,s_inv)
    j,_ = Tuple(s_inv[s_idx])
    S,A,P,H,F = j_inv(j)
    return F
end


#  ------------ ex-post model fit -------------- #
# a function that splits the work up and calculates predicted averages for each data source
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
# this function calculates moments at the individual level for all observations owned by the vector of model data MD. Used in the call above.
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
# a function that calculates individual level moments. Used in the function above.
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
        A[t] = em_mean(em.q_s,t,s->f_afdc(s,t,logP,s_inv,k_inv))
        E[t] = em_mean(em.q_s,t,s->f_earn(s,t,logP,s_inv,k_inv,p,md)) #,x->job_offer(x,s_inv,k_inv))
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


#  ------------ ex-ante model fit -------------- #
# some functions for looking at the "ex-ante" fit of the model,
#   - i.e. predicted moments from the model when the sequence of observed data ~ isn't ~ used to form priors over unobserved states
#   - the version below *does* however observed data to form a posterior over initial conditions (using initialize_expost!), and runs a model prediction from there.
#   - to switch to fully exante predictions, replace this function call with initialize_exante!

# iterate from some initial condition, given by π0 and calculate the distribution at each t
function get_choice_state_distribution!(Π::SparseMatrixCSC{Float64,Int64},logP,model,p,R::Int64) #<- need choice probs? T?
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
            jF = 1 + R*A
            for kη_next in 1:p.Kη
                fkk = p.Fη[kη_next,kη,jF,kτ]
                kn = k_idx[kA_next,kη_next,kω_next,kτ]
                for jn in choice_set(kη_next>1)
                    sn = (kn - 1)*J + jn
                    Π[sn,t+1] += fkk * exp(logP[jn,kn,t+1]) * qs
                end
            end
        end
    end
end

# just like the function basic_model_fit, this splits MD into chunks and passes the work to different threads
#   - exante_model_fit_chunk returns a data frame with individual level moments
#   - this function takes averages of these over important characteristics.
function exante_model_fit(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,fname = "model_stats.csv",write = true)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn exante_model_fit_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    #d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, :LOGFULL => Statistics.mean => :LOGFULL)
    d = combine(groupby(D,[:source,:arm,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, :LOGFULL => Statistics.mean => :LOGFULL)
    if write
        CSV.write(string("output/",fname),d)
    end
    return d
end
# performs the work described above.
# note the call to initialize_expost! which could be switched to initialize_exante!
function exante_model_fit_chunk(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
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
            #initialize_expost!(π0,EM[n],s_inv) #<- get initial dist from posterior
            initialize_exante!(logπτ,π0,p,md,data[n],k_idx) #<- get initial dist from posterior
            get_choice_state_distribution!(EM[n].q_s,logP,(;K,π0,s_inv,k_inv,Kω,k_idx),p,md.R) #<- nice.
            d = model_stats_exante(p,EM[n],md,data[n])
            d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    return D
end
# calculates individual level moments using the function em_mean to take averages
function model_stats_exante(p,em::EM_data,md::model_data,data::likelihood_data)
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

function initialize_exante!(logπτ,π0,p,md::model_data,data::likelihood_data,k_idx)
    J = 9
    #(;k_inv,s_inv) = idx
    fill!(π0,0.)
    log_type_prob!(logπτ,p,md,data)

    # start with the initial conditions:
    kA = 1 + data.kA
    kω = 1
    for kη in 1:p.Kη, kτ in 1:p.Kτ
        k = k_idx[kA,kη,kω,kτ]
        π0[k] = initial_prob(kA,kη,kτ,logπτ,p,md)
    end
    return nothing
end
# fill π0 states with the posterior distribution instead
function initialize_expost!(π0,EM::EM_data,s_inv)
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

function get_data_initial_distribution!(p,EM,MD,n_idx)
    (;Kη, Kτ) = p
    dist_η = zeros(Kη,2)
    denom_η = zeros(2)
    dist_τ = zeros(Kτ,2)
    denom_τ = zeros(2)
    for md in MD
        k_inv = CartesianIndices((2,Kη,md.Kω,Kτ))
        K = prod(size(k_inv))
        s_inv = CartesianIndices((9,K))
        l = 1 + 1*(md.source=="SIPP")
        for n in n_idx[md.case_idx]
            for s in nzrange(EM[n].q_s,1)
                s_idx = EM[n].q_s.rowval[s]
                wght = EM[n].q_s.nzval[s]
                _,k = Tuple(s_inv[s_idx])
                _,kη,kω,kτ = Tuple(k_inv[k])
                dist_η[kη,l] += wght
                denom_η[l] += wght
                dist_τ[kτ,l] += wght
                denom_τ[l] += wght
            end
        end
    end
    dist_η ./= denom_η'
    dist_τ ./= denom_τ'
    d = [DataFrame(var = "Wage Shock",value = 1:Kη,dist = dist_η[:,1],source="Experiments");
    DataFrame(var = "Wage Shock",value = 1:Kη,dist = dist_η[:,2],source="SIPP");
    DataFrame(var = "Type",value = 1:Kτ,dist = dist_τ[:,1],source="Experiments");
    DataFrame(var = "Type",value = 1:Kτ,dist = dist_τ[:,2],source="SIPP")]
    CSV.write("output/initial_dists.csv",d)
end

