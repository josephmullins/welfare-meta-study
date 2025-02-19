using SparseArrays

using LinearAlgebra
logit(x) = exp(x) / (1 + exp(x))
logit_inv(x) = log(x / (1 - x))

function get_model(p)
    T = 18*4
    K = 2 * p.Kη * 9 * p.Kτ
    R = eltype(p.αA)
    logP = zeros(R,9,K,T)
    V = zeros(R,K,T)
    vj = zeros(R,9)
    return (;logP,vj,V)
end

function pars(Kτ::Int64,Kη::Int64)
    β = 0.98
    wq = fill(50.,Kτ)
    αC = 1.
    αθ = 0.1 * ones(Kτ)
    αH = zeros(Kτ)
    αA = zeros(Kτ)
    αS = zeros(Kτ)
    αF = zeros(Kτ)
    αR = 0.
    λR = 0.05
    αP = 0.
    σ = fill(2.,3)

    σ_W = 2.
    σ_PF = 2.
    #μ_PF = 1.
    #σ_PF2 = 2.

    δ = 0.1*ones(Kτ)
    λ₀ = 0.5*ones(Kτ)
    λ₁ = 0.5*ones(Kτ)
    μₒ = 0.
    σₒ = 1.

    ση = 2.

    βΓ = zeros(2)
    βw = [LinRange(6,7.5,Kτ);-0.2;0.003]
    βf = [fill(3.,Kτ);zeros(10)]
    βτ = zeros(23,Kτ-1)
    πη = ones(2,Kη,Kτ,3) / Kη
    
    ηgrid = LinRange(-1,1,Kη-1)
    return (;Kτ,Kη,β,wq,αC,αθ,αH,αA,αS,αF,αR,λR,αP,σ,σ_W,σ_PF,δ,λ₀,λ₁,μₒ,σₒ,ση,βΓ,βw,βf,βτ,πη,ηgrid)
end

# NEXT: write out which are needed to solve model and which are needed to evaluate the likelihood/create EM data
struct model_data #<- should this contain everything!?!?
    case_idx::Int64
    y0::Int64 #<- year to begin problem
    q0::Int64 #<- quarter to begin problem
    T::Int64 #<- length of problem
    age0::Int64 # <- mother's age at start of sample
    ageyng::Int64 #<- age of youngest kid
    source::String #<- what's the location
    budget::String #<- what budget to use.
    arm::Int64 #<- treatment arm in the location
    loc_ind::Int64 #<- indexes for the "location" (source x arm combo)
    SOI::Int64 #<- state SOI
    numkids::Int64 #<- number of kids in household

    unemp::Vector{Float64} #<- unemployment rate
    cpi::Vector{Float64} #<- cpi
    R::Int64 #<- indicates if work requirement (can change to a vector if need be)
    Kω::Int64 #<- indicates length of time limit
    TL::Bool #<- indicating that time limit is effective
    type_block::UnitRange{Int64} #<- βτ[type_block,:] holds type selection parameters for this 
end

include("model/solve.jl")

include("model/update.jl")

include("model/choices.jl")

include("model/prices.jl")

include("budget.jl")
include("model/utility.jl")
include("model/states_transitions.jl")
include("tools/input_output.jl")