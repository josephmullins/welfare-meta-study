# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")

function f_eta(s_idx,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    return kη
end

function f_ω(s_idx,s_inv,k_inv)
    _,k = Tuple(s_inv[s_idx])
    _,_,kω,_ = Tuple(k_inv[k])
    return kω
end


function state_stats(p,em::EM_data,md::model_data,data::likelihood_data)
    T = size(em.q_s,2)
    Q = 0:T-1
    E = zeros(T)
    J = zeros(T)
    Kω = zeros(T)
    
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((9,K))

    # year = md.y0 .+ fld.(md.q0-1 .+ (0:T-1),4)
    # Q = mod.(md.q0-1 .+ (0:T-1),4) .+ 1
    year = md.y0 .+ fld.(md.q0 .+ (0:T-1),4) #<- start with one quarter delay in this model.
    Q = mod.(md.q0 .+ (0:T-1),4) #<- as above
    for t in 1:T
        E[t] = em_mean(em.q_s,t,s->f_earn(s,t,s_inv,k_inv,p,md)) #,x->job_offer(x,s_inv,k_inv))
        J[t] = em_mean(em.q_s,t,s->f_eta(s,s_inv,k_inv),s->job_offer(s,s_inv,k_inv))
        Kω[t] = em_mean(em.q_s,t,s->f_ω(s,s_inv,k_inv))
    end
    keep = .!data.choice_missing
    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end
    return DataFrame(source = md.source, arm = md.arm, year = year[keep], Q = Q[keep],case_idx = md.case_idx,
        E = E[keep],J = J[keep],W = Kω[keep], est_sample = data.use,app_status = app_status,LOGFULL = l_full[keep])
end

function state_fit(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    D0 = DataFrame(source = [],arm = [],year=[],Q=[],E = [], J = [], W = [],case_idx = [], est_sample = [],app_status = [])
    D1 = DataFrame(source = [],arm = [],year=[],Q=[],E = [], J = [], W = [],case_idx = [], est_sample = [],app_status = [])
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
            d0 = state_stats(p,EM[n],md,data[n])
            D0 = [D0;d0]
            initialize_exante!(logπτ,π0,p,md,data[n],k_idx) #<- get initial dist from posterior
            get_choice_state_distribution!(EM[n].q_s,logP,(;K,π0,s_inv,k_inv,Kω,k_idx),p,md.R) #<- nice.
            d1 = model_stats_exante(p,EM[n],md,data[n])
            D1 = [D1;d1]
        end
    end
    d0 = combine(groupby(D0,[:source,:arm,:est_sample,:year,:Q]),:J => Statistics.mean => :J,:E => Statistics.mean => :E, :W => Statistics.mean => :W)
    d1 = combine(groupby(D1,[:source,:arm,:est_sample,:year,:Q]),:J => Statistics.mean => :J,:E => Statistics.mean => :E, :W => Statistics.mean => :W)
    d0[!,:case] .= "ex-post"
    d1[!,:case] .= "ex-ante"
    return [d0;d1]
end

Kτ = 5 #
Kη = 4
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"current_est")
p = loadpars_vec(p,"est_noSIPP_K5")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp)
end

MD,EM,data,n_idx = estimation_setup(panel);

Random.seed!(2020)
shuffle!(MD)

# get the initial conditions this way
forward_back_threaded!(p,EM,MD,data,n_idx)

d = state_fit(p,EM,MD,data,n_idx)

#write everything to file here.
