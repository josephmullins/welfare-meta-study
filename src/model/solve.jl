#I'm thinking: a named tuple for all the indexing,
#and a named tuple for all of the parameters?
function solve!(logP,V,vj,p,md::model_data)
    (;B,C) = p.nests
    T = max((17-md.ageyng)*4,md.T)
    K = 2 * p.Kη * p.Kτ * md.Kω
    fill!(V,0.)
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    k_idx = LinearIndices((2,p.Kη,md.Kω,p.Kτ))
    tnow = 1
    loc = loc_ind(md)
    for t in reverse(1:T)
        tnext = 3-tnow #<- tnow = 1 => tnext =2, tnow=2 => tnext =1
        for k in 1:K
            kA,kη,kω,kτ = Tuple(k_inv[k])
            job_offer = kη>1
            age_yng = md.ageyng + fld(t,4)
            eligible = (kω<md.Kω || !md.TL) && (age_yng<18) #<- change what we mean by eligible
            state = (;kA,kη,kω,kτ,k_idx)
            if job_offer
                for j in 1:9
                    @views vj[j] = calc_vj(j,V[:,tnext],md,state,p,t,eligible,loc)
                end
                # choice probs:
                @views V[k,tnow] = nested_logit(logP[:,k,t],vj;B,C,σ = p.σ) #<- or something like it.
            else
                for j in (1,4,7)
                    @views vj[j] = calc_vj(j,V[:,tnext],md,state,p,t,eligible,loc)
                end
                @views V[k,tnow] = plain_logit(logP[:,k,t],vj;B = (1,4,7),σ = p.σ[3])
            end
        end
        tnow = tnext
    end
    return nothing
end

function calc_vj(j,V,md::model_data,state,pars,t,eligible,loc)
    (;kA,kη,kω,kτ,k_idx) = state
    #(;β,λ,δ,πW,Kη) = pars
    S,A,_,H,F = j_inv(j)
    v = utility(S,A,H,F,pars,md,kA,kη,kτ,t,eligible)
    kA_next,kω_next = next(A,kA,kω;md.Kω)
    for kη_next in 1:pars.Kη
        k_next = k_idx[kA_next,kη_next,kω_next,kτ]
        j = 1 + md.R * A #<- indicates if the work requirement gives a bump to λ₀
        v += pars.β * pars.Fη[kη_next,kη,j,kτ,loc] * V[k_next]
    end
    return v
end




# this algorithm works as long as the partitions are written with nodes in ascending order. For example:
# the partition B[l] = [[1,2],[3,4,5]] is ok because the value for nest [1,2] will be written to vj[1] and vj[2] for nest [3,4,5]
# the partition B[l] = [[2,3],[1,4,5]] will be incorrect because the nest [2,3] will write to vj[1], which still needs to be used for the nest [1,4,5]
function nested_logit(logP,vj;B,C,σ)
    fill!(logP,0.)
    for l ∈ eachindex(B)
        Cₗ = C[l] #<- each Cₗ is a K(l)-vector of vectors containing the choices that belong to that node.
        for k ∈ eachindex(B[l])
            bₖ = B[l][k] #<- indicates which nodes are in the kth partition of layer l
            vmax = -Inf
            # find the maximum
            for j ∈ bₖ
                vj[j]>vmax ? vmax=vj[j] : nothing
            end
            norm = 0.
            for j ∈ bₖ
                norm += exp((vj[j] - vmax) / σ[l])
            end
            norm = log(norm)
            # then: 
            for jₗ ∈ bₖ
                for j ∈ Cₗ[jₗ]
                    logP[j] += (vj[jₗ] - vmax) / σ[l] - norm
                end
            end
            vj[k] = vmax + σ[l] * norm
        end
    end
    return vj[1]
end

function plain_logit(logP,vj;B,σ)
    norm = 0.
    vmax = -Inf
    for j ∈ B
        vj[j]>vmax ? vmax=vj[j] : nothing
    end
    for j ∈ B
        norm += exp((vj[j] - vmax) / σ)
    end
    norm = log(norm)
    for j ∈ B
        logP[j] = (vj[j] - vmax) / σ - norm
    end
    vj[1] = vmax + σ * norm
end