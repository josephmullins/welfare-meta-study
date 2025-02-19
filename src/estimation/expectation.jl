using .Threads

function forward_back_threaded!(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx::Vector{Vector{Int64}})
    chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn forward_back_chunk!(p,EM,chunk,data,n_idx)
    end
    fetch.(tasks)
    return nothing
end

function forward_back_chunk!(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    T = 18*4
    K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
    logP = zeros(9,K,T)
    V = zeros(K,2)
    vj = zeros(J)
    logπτ = zeros(p.Kτ)
    for md in MD
        solve!(logP,V,vj,p,md)
        for n ∈ n_idx[md.case_idx]
            update!(logπτ,EM[n],logP,p,md,data[n])
            forward_back!(EM[n])
        end
    end
    return nothing
end

# a function for updating EM_data with initial probabilities (α0) and transition probabilities P[s'|s] where s is a combination of states and choices. This allows for choices to be partially observed (which they are in my setting)
# Let yt be any data observed at t, then:
# P[t][s_{t+1},s_{t}] holds P[y_{t+1}|s_{t+1}]P[s_{t+1}|s_{t}] where recall that s_{t} is the choice *and* the state

function update!(logπτ,EM::EM_data,logP,p,md::model_data,data::likelihood_data)
    J = size(logP,1)
    log_type_prob!(logπτ,p,md,data)
    t0 = data.t0
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((J,K))

    # start with the initial conditions:
    for s in nzrange(EM.α,1)
        t = 1
        s_idx = EM.α.rowval[s]
        j,k = Tuple(s_inv[s_idx])
        ll = logP[j,k,t+t0]
        kA,kη,_,kτ = Tuple(k_inv[k])
        if data.wage_valid[t]
            ll += wage_log_like(data.logW[t],p,md,kτ,kη,t+t0)
        end
        EM.α.nzval[s] = initial_prob(kA,kη,kτ,logπτ,p,md) * exp(ll)
    end
    for t in eachindex(EM.P)
        for s in nzrange(EM.α,t)
            s_idx = EM.α.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            _,A,_,_,_ = j_inv(j)
            jF = 1 + md.R*A
            _,kη,_,kτ = Tuple(k_inv[k])
            for sn in nzrange(EM.P[t],s_idx)                    
                sn_idx = EM.P[t].rowval[sn]
                jn,kn = Tuple(s_inv[sn_idx])
                _,kη_next,_,_ = Tuple(k_inv[kn])

                ll = logP[jn,kn,t+t0+1]
                if data.wage_valid[t+1]
                    if kη_next==1
                        @show md.case_idx, t, sn_idx
                    end
                    ll += wage_log_like(data.logW[t+1],p,md,kτ,kη_next,t+t0+1)
                end
                if md.source=="SIPP" && data.chcare_valid[t+1] && data.pay_care[t+1]
                    ll += chcare_log_like(data.log_chcare[t+1],p,md,kτ,t+t0+1)
                end
                fkk = p.Fη[kη_next, kη, jF, kτ]
                #@show t, sn_idx, s_idx, kη_next, kη, fkk
                EM.P[t][sn_idx,s_idx] = fkk*exp(ll) #?
            end
        end
    end
end

# initial probability
function initial_prob(kA,kη,kτ,logπτ,p,md::model_data)
    if md.source=="SIPP"
        πη = p.πₛ[kη,kτ]
    else
        loc = (md.source=="FTP") + 2(md.source=="CTJF") + 3(md.source=="MFIP")
        πη = p.πη[kA,kη,kτ,loc]
    end
    return exp(logπτ[kτ]) * πη
end
