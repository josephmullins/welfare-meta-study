# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare
using Distributions
Φ(x,μ,σ) = cdf(Normal(μ,σ),x)

function fη(kη_next,kη,kτ,WR,unemp,p)
    if kη==1
        λ₀ = logit(p.λ₀[kτ] + p.λᵤ*unemp + p.λR*WR)
        if kη_next==1
            return 1-λ₀
        else
            return λ₀ * p.πₒ[kη_next-1,1]
        end
    else
        if kη_next==1
            return p.δ[kτ]
        elseif kη_next==kη
            λ₁ = logit(p.λ₁[kτ] + p.λᵤ*unemp)
            return (1-p.δ[kτ]) * ( (1-λ₁) + λ₁ * p.πₒ[kη_next-1,kη] )
        else
            λ₁ = logit(p.λ₁[kτ] + p.λᵤ*unemp)
            return (1-p.δ[kτ]) * λ₁ * p.πₒ[kη_next-1,kη]
        end
    end
end

# double check whether this leads to allocations
function p_offer(kη_next,kη,p)
    norm = Φ(p.ηgrid[end],p.μₒ,p.σₒ)
    if kη==1
        if kη_next==2
            return Φ(p.ηgrid[1],p.μₒ,p.σₒ) / norm
        else
            return (Φ(p.ηgrid[kη_next-1],p.μₒ,p.σₒ) - Φ(p.ηgrid[kη_next-2],p.μₒ,p.σₒ)) / norm
        end
    else
        if kη_next==kη
            return Φ(p.ηgrid[kη_next-1],p.μₒ,p.σₒ) / norm
        elseif kη_next>kη
            return (Φ(p.ηgrid[kη_next-1],p.μₒ,p.σₒ) - Φ(p.ηgrid[kη_next-2],p.μₒ,p.σₒ)) / norm
        else
            return 0.
        end
    end
end

function update_transitions(p)
    πₒ = zeros(eltype(p.μₒ),p.Kη-1,p.Kη)
    for k in axes(πₒ,2), k2 in axes(πₒ,1)
        πₒ[k2,k] = p_offer(k2+1,k,p)
    end
    return (;p...,πₒ)
end

function stat_dist(πₒ,λ₀,λ₁,δ)
    R = eltype(πₒ)
    K = length(πₒ)
    πₛ = zeros(R,K+1)
    u = δ / (δ + λ₀)
    πₛ[1] = u
    inₖ = u * λ₀
    keepₖ = (1-λ₁)
    for k in 1:K
        keepₖ += λ₁ * πₒ[k]
        πₛ[k+1] = πₒ[k] * inₖ / ((1 - (1-δ)*keepₖ))
        inₖ += (1-δ) * λ₁ * πₛ[k+1]
    end
    return πₛ
end

function next(A,kA,kω;Kω)
    kA_next = 1+A
    kω_next = min(kω + A,Kω)
    return kA_next,kω_next
end