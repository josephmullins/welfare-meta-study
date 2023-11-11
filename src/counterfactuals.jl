using Setfield

function counterfactual(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)#,fname = "model_stats.csv",write = true)
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn counterfactual_chunk(p,EM,chunk,data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    #d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:EMP => Statistics.mean => :EMP,:AFDC => Statistics.mean => :AFDC, :EARN => Statistics.mean => :EARN, :LOGFULL => Statistics.mean => :LOGFULL)
    d = combine(groupby(D,[:source,:arm,:variable,:year,:Q]),:value => Statistics.mean => :value)
    # if write
    #     CSV.write(string("output/",fname),d)
    # end
    return d
end

# performs the work described above.
# note the call to initialize_expost! which could be switched to initialize_exante!
function counterfactual_chunk(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    D = DataFrame(source = [],arm = [],year=[],Q=[],variable = [],value = [],case_idx = [],n_idx = [],app_status = [])
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
        for n in n_idx[md.case_idx]
            #initialize!(logπτ,EM[n],π0,p,md,data[n],(;k_inv,s_inv)) #<- get initial dist from priors
            initialize_exante!(logπτ,EM[n],π0,p,md,data[n],k_idx) 
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
    end
    d = DataFrame(variable = "Emp",value = H,year = year,Q = Q)
    d = [d;DataFrame(variable = "AFDC",value = A,year = year,Q = Q)]
    d = [d;DataFrame(variable = "Earn",value = E,year = year,Q = Q)]

    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end
    d[!,:arm] .= md.arm
    d[!,:app_status] .= app_status
    d[!,:case_idx] .= md.case_idx
    d[!,:source] .= md.source
    return d
end

function cev(em::EM_data,V1,V0,s_inv,β)
    cev = 0.
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowvals[s]
        _,k = Tuple(s_inv[s_idx])
        cev += em.q_s.nzval[s] * (exp((1-β) * (V1[k] - V0[k])) - 1)
    end
    return cev
end

function calculate_treatment_effects(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    d = DataFrame(source = [],arm = [], variable = [], value = [], year = [],Q = [])
    for s in ("FTP","CTJF","MFIP")
        MDₛ = MD[[md.source==s for md in MD]]
        MDₛ = [@set md.arm=0 for md in MDₛ]
        d0 = counterfactual(p,EM,MDₛ,data,n_idx)
        MDₛ = [@set md.arm=1 for md in MDₛ]
        d1 = counterfactual(p,EM,MDₛ,data,n_idx)
        d1.value = d1.value .- d0.value
        d = [d;d1]
        if s=="MFIP"
            MDₛ = [@set md.arm=2 for md in MDₛ]
            d2 = counterfactual(p,EM,MDₛ,data,n_idx)
            d2.value = d2.value .- d0.value
            d = [d;d2]
        end
    end
    return d
end