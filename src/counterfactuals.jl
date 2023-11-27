using Setfield

function counterfactual(p,pB,pC,MD0,MD1,data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(eachindex(MD0),length(MD0) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn counterfactual_chunk(p,pB,pC,MD0[chunk],MD1[chunk],data,n_idx)
    end
    D = vcat(fetch.(tasks)...)
    d = combine(groupby(D,[:source,:variable,:year,:Q]),:value => Statistics.mean => :value) #here is the reduction
    return d
end

function model_size(p,md::model_data)
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    k_idx = LinearIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    Kω = md.Kω
    s_inv = CartesianIndices((9,K))
    return (;k_inv,k_idx,K,Kω,s_inv)
end

function counterfactual(p,pB,pC,logπτ,md0::model_data,md1::model_data,data::Vector{likelihood_data},n_idx)
    m0 = get_model(p)
    idx0 = model_size(p,md0)
    solve!(m0.logP,m0.V,m0.vj,p,md0)
    m1 = get_model(p)
    idx1 = model_size(p,md1)
    solve!(m1.logP,m1.V,m1.vj,p,md1)
    π0 = zeros(idx0.K) 
    π1 = zeros(idx1.K)

    D = DataFrame(source = [],year=[],Q=[],variable = [],value = [],case_idx = [],n_idx = [],app_status = [])
    for n in n_idx[md0.case_idx]
        #initialize!(logπτ,EM[n],π0,p,md,data[n],(;k_inv,s_inv)) #<- get initial dist from priors
        initialize_exante!(logπτ,π0,p,md0,data[n],idx0.k_idx)
        T = data[n].T
        Π = spzeros(9 * idx0.K,T)
        get_choice_state_distribution!(Π,m0.logP,(;idx0...,π0),p,md0.R)
        d0 = counterfactual_stats(p,pB,pC,Π,md0,data[n])

        initialize_exante!(logπτ,π1,p,md1,data[n],idx1.k_idx)
        Π = spzeros(9 * idx1.K,T)
        get_choice_state_distribution!(Π,m1.logP,(;idx1...,π0 = π1),p,md1.R)
        d1 = counterfactual_stats(p,pB,pC,Π,md1,data[n])
        
        d1.value .-= d0.value
        @views cev = calc_cev(m1.V[:,1],m0.V[:,1],π0,idx0.k_idx,idx1.k_idx,p.β)
        d1 = [d1 ; DataFrame(source = md0.source,Q = 0,year = md0.y0,app_status = d1.app_status[1],case_idx = md0.case_idx,variable = "CEV",value = cev)]
        d1[!,:n_idx] .= n
        D = [D;d1]
    end
    return D
end

function counterfactual_chunk(p,pB,pC,MD0,MD1,data::Vector{likelihood_data},n_idx)
    D = DataFrame(source = [],year=[],Q=[],variable = [],value = [],case_idx = [],n_idx = [],app_status = [])
    logπτ = zeros(p.Kτ)
    #T = 18*4
    #K = 2*p.Kη*9*p.Kτ
    #Π = spzeros(9 * K,T)

    for i in eachindex(MD0)
        d = counterfactual(p,pB,pC,logπτ,MD0[i],MD1[i],data,n_idx)
        D = [D;d]
    end
    return D
end


# a function to calcualte the expected value of formal and informal care inputs:
function f_gk(s,s_inv,k_inv,p)
    j,k = Tuple(s_inv[s])
    _,_,_,kτ = Tuple(k_inv[k])
    _,_,_,H,F = j_inv(j)
    return H * p.g₁[kτ] + F * p.g₂[kτ]
end

# calculates individual level moments using the function em_mean to take averages
function counterfactual_stats(p,pB,pC,Π,md::model_data,data::likelihood_data)
    T = data.T
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
    thC = 0.
    thB = 0.
    for t in 1:T
        H[t] = em_mean(Π,t,s->f_work(s,s_inv))
        A[t] = em_mean(Π,t,s->f_afdc(s,s_inv))
        E[t] = em_mean(Π,t,s->f_earn(s,t,s_inv,k_inv,p,md)) #,x->job_offer(x,s_inv,k_inv))
        log_full_t = em_mean(Π,t,s->log_full(s,t,s_inv,k_inv,p,md))
        gC = em_mean(Π,t,s->f_gk(s,s_inv,k_inv,pC)) #<- write functions to take the EV.
        gB = em_mean(Π,t,s->f_gk(s,s_inv,k_inv,pB))

        thB += pB.δI * log_full_t + gB + pB.δθ * thB
        thC += pC.δI * log_full_t + gC + pC.δθ * thC
    end
    d = DataFrame(variable = "Emp",value = H,year = year,Q = Q)
    d = [d;DataFrame(variable = "AFDC",value = A,year = year,Q = Q)]
    d = [d;DataFrame(variable = "Earn",value = E,year = year,Q = Q)]
    d = [d;DataFrame(variable = "Behavioral Skill",value = 100 *thB,year = md.y0,Q = 0)] #<- report in % of sd
    d = [d;DataFrame(variable = "Cognitive Skill",value = 100 * thC,year = md.y0,Q = 0)]

    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end
    d[!,:app_status] .= app_status
    d[!,:case_idx] .= md.case_idx
    d[!,:source] .= md.source
    return d
end

# problem: indexation is different depending on what we're doing.
function calc_cev(V1,V0,π0,k_idx0,k_idx1,β)
    cev = 0.
    kω = 1
    for kA in axes(k_idx0,1), kη in axes(k_idx0,2), kτ in axes(k_idx0,4)
        k0 = k_idx0[kA,kη,kω,kτ]
        k1 = k_idx1[kA,kη,kω,kτ]
        cev += π0[k0] * (exp((1-β) * (V1[k1] - V0[k0])) - 1)
    end
    return 100 * cev
end

function full_treatment(md::model_data)
    md = @set md.arm = 1
    if md.source=="FTP"
        md = @set md.R = 1
        md = @set md.Kω = 8
        md = @set md.TL = true
        md = @set md.arm = 1
    elseif md.source=="CTJF"
        md = @set md.R = 1
        md  = @set md.Kω = 9
        md = @set md.TL = true
        md = @set md.arm = 1
    elseif md.source=="MFIP"
        md = @set md.R = 1
    end
    return md
end

function incentives_only(md::model_data)
    md = @set md.arm = 1
    md = @set md.R = 0
    md = @set md.TL = false
    md = @set md.Kω = 1
    return md
end

function work_requirements_only(md::model_data)
    md = @set md.arm = 0
    md = @set md.R = 1
    md = @set md.TL = false
    md = @set md.Kω = 1
    return md
end

function time_limits_only(md::model_data)
    md = @set md.arm = 0
    md = @set md.R = 0
    if md.source=="FTP"
        md = @set md.Kω = 8
        md = @set md.TL = true
    elseif md.source=="CTJF"
        md  = @set md.Kω = 9
        md = @set md.TL = true
    end
    return md
end

function control(md::model_data)
    md = @set md.R = 0
    md = @set md.Kω = 1
    md = @set md.TL = false
    md = @set md.arm = 0
    return md
end

# OK: note: we have to convert the state regardless!!

function convert_sipp(md::model_data,site::String)
    md = @set md.R = 1
    md = @set md.arm = 1
    md = @set md.budget = site
    if site=="CTJF"
        md = @set md.TL = true
        md = @set md.Kω = 9
        md = @set md.SOI = 7
        md = @set md.y0 = 1996
    elseif site=="FTP"
        md = @set md.TL = true
        md = @set md.Kω = 8
        md = @set md.SOI = 10
        md = @set md.y0 = 1994
    else
        md = @set md.SOI = 24
        md = @set md.y0 = 1994
    end
    return md
end


# these gets rid of all the other things.
# BUT: how do I switch the budget off??
# for im in eachindex(MD)
#     MD[im] = @set MD[im].TL = false
#     MD[im] = @set MD[im].Kω = 1
#     MD[im] = @set MD[im].R = 0 #<- get rid of work requirement.
# end

# goal of decomposition:
# (1) just financial incentives
# (2) just work requirement
# (3) just time limits (but only for FTP and CTJF case)

#note: calculating welfare:

# (4) if you ignored SIPP, you could probably estimate choice probabilities better, no?
# p_{lk}(A_{-1},EMP_{-1},ω,η,k,j,nk,age) FUCK though?
# store as: 2 x 2 x Kη x Kω x Kτ x 4 x mother_age_bins x child_age_bins x num_arms array
# the question is then: how well does this model fit?
# p_{F} can be used again.
# put a fixed effect in wage equation also.
# can also use finite dependence to estimate using just instrumental variables?
# it would make estimation so much easier in the end?