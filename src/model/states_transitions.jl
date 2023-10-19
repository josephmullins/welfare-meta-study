# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare
using LinearAlgebra

function update_transitions(p)
    Fηk = Fη_mat(p.μη,p.Kη)
    πₛₖ = stat_dist(Fηk)
    R = eltype(p.μη)
    πₛ = zeros(R,p.Kη,p.Kτ)
    Fη = zeros(R,p.Kη,p.Kη,p.Kτ)
    for kτ in 1:p.Kτ
        Fη[:,:,kτ] .= Fηk
        πₛ[:,kτ] .= πₛₖ
    end
    return (;p...,Fη,πₛ)
end

function stat_dist(F)
    K = size(F,1)
    πₛ = @views inv(I(K-1) .+ diagm(F[1:K-1,K]) * ones(K-1,K-1) .- F[1:K-1,1:K-1]) * F[1:K-1,K]
    return [πₛ;1-sum(πₛ)]
end 

function Fη_mat(μη ,K)
    F = zeros(eltype(μη),K,K)
    for k in 1:K
        @views norm = 1 + sum(exp.(μη[:,k]))
        F[1,k] = 1 / norm
        for kn in 2:K
            F[kn,k] = exp(μη[kn-1,k]) / norm
        end
    end
    return F     
end

function next(A,kA,kω;Kω)
    kA_next = 1+A
    kω_next = min(kω + A,Kω)
    return kA_next,kω_next
end