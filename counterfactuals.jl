using Setfield

function counterfactual(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,fname = "model_stats.csv",write = true)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn exante_model_fit_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, :LOGFULL => Statistics.mean => :LOGFULL)
    if write
        CSV.write(string("output/",fname),d)
    end
    return d
end
# performs the work described above.
# note the call to initialize_expost! which could be switched to initialize_exante!
function counterfactual_chunk(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
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
            initialize_expost!(π0,EM[n],s_inv) #<- get initial dist from posterior
            get_choice_state_distribution!(EM[n].q_s,logP,(;K,π0,s_inv,k_inv,Kω,k_idx),p) #<- nice.
            d = counterfactual_stats(p,EM[n],md,data[n])
            d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    return D
end
# calculates individual level moments using the function em_mean to take averages
function counterfactual_stats(p,em::EM_data,md::model_data,data::likelihood_data)
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

