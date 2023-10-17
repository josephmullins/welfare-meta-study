# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare
using Distributions
Φ(x,μ,σ) = cdf(Normal(μ,σ),x)

function update_transitions(p)
    R = eltype(p.λ₀)
    πₛ = zeros(R,p.Kη,p.Kτ)
    Fη = zeros(R,p.Kη,p.Kη,p.Kτ)
    for kτ in 1:p.Kτ
        πₒ = get_offer_dist(p.ηgrid,p.μₒ,p.σₒ,R)
        λ₁ = p.λ₁ * p.λ₀[kτ]
        Fη[:,:,kτ] .= Fη_mat(πₒ, p.λ₀[kτ], λ₁, p.δ[kτ], p.Kη-1)
        πₛ[:,kτ] .= stat_dist(πₒ, p.λ₀[kτ], λ₁, p.δ[kτ])
    end
    return (;p...,Fη,πₛ)
end

function get_offer_dist(ηgrid,μ,σ,R)
    K = length(ηgrid)
    π0 = zeros(R,length(ηgrid))
    norm = Φ(ηgrid[end],μ,σ)
    π0[1] = Φ(ηgrid[1], μ, σ) / norm
    for k in 2:K
        π0[k] = (Φ(ηgrid[k], μ, σ) - Φ(ηgrid[k-1],μ,σ)) / norm
    end
    return π0
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

function Fη_mat(πₒ,λ₀,λ₁,δ,K)
    F = zeros(eltype(πₒ),K+1,K+1)
    F[1,1] = (1 - λ₀)
    for k in 1:K
        F[k+1,1] = λ₀ * πₒ[k]
    end
    for k in 1:K
        F[1,k+1] = δ
        F[k+1,k+1] = (1-δ) * (1-λ₁)
        for kn in 1:K
            F[max(k+1,kn+1),k+1] += (1-δ) * λ₁ * πₒ[kn]
        end
    end
    return F     
end

function next(A,kA,kω;Kω)
    kA_next = 1+A
    kω_next = min(kω + A,Kω)
    return kA_next,kω_next
end