# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare
using Distributions

function update_transitions(p)
    Fηk = Fη_mat(p.μη,p.Kη)
    R = eltype(p.μη)
    πₛ = zeros(R,p.Kη,p.Kτ)
    Fη = zeros(R,p.Kη,p.Kη,p.Kτ)
    for kτ in 1:p.Kτ
        Fη[:,:,kτ] .= Fηk
        πₛ[:,kτ] .= stat_dist(πₒ, p.λ₀[kτ], p.λ₁[kτ], p.δ[kτ])
    end
    return (;p...,Fη,πₛ)
end

πK = (1-∑ₖπₖ)


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

function Fη_mat(μη ,K)
    F = zeros(eltype(πₒ),K+1,K+1)
    for k in 1:K
        @views norm = 1 + sum(exp.(μη[:,k]))
        F[1,k] = 1 / norm
        for kn in 2:K
            F[kn,k] = exp(μη[kn]) / norm
        end
    end
    return F     
end

function next(A,kA,kω;Kω)
    kA_next = 1+A
    kω_next = min(kω + A,Kω)
    return kA_next,kω_next
end