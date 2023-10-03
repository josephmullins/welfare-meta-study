using SparseArrays
include("tools/NestedLogit.jl")


using LinearAlgebra
logit(x) = exp(x) / (1 + exp(x))
logit_inv(x) = log(x / (1 - x))


# these are the underlying parameters that apply to every instance of the model
mutable struct pars
    Kτ::Int64 #<- number of parent types
    Kη::Int64 #<- number of wage shocks (η)
    β::Float64 #<- discount factor
    wq::Float64 #<- productivity of home production
    αC::Float64 #<- normalize to 1.
    αθ::Vector{Float64} #<- weight on child skills
    αH::Vector{Float64} #<- disutility from work
    αA::Vector{Float64} #<- disutility from AFDC participation
    αS::Vector{Float64} #<- disutility from SNAP participation
    αF::Vector{Float64} #<- (dis)utility from formal/paid care use
    αR::Vector{Float64} #<- work requirement parameters
    αP::Float64 #<- additional application cost

    σ::Vector{Float64} #<- dispersion of shocks

    σ_W::Float64 #<- measurement error in wage eq
    σ_PF::Float64 #<- measurement error in care prices

    δ::Vector{Float64} #<- "separation rate"
    λ::Vector{Float64} #<- job arrival rate
    πW::Float64 #<- probabability of staying on grid point (vs moving up and down)
    ση::Float64 #<- scale parameter on ηgrid

    #
    βΓ::Vector{Float64} #<- approximation for Γt
    βw::Vector{Float64} #<- wage coefficients
    βf::Vector{Float64} #<- care price coefficients
    βτ::Matrix{Float64} #<- type selection coefficients
    πη::Array{Float64,4} #<- initial probabilities of η for the three experimental sites
    ηgrid::Vector{Float64}
end

function pars(Kτ,Kη)
    β = 0.98
    wq = 1.
    αC = 1.
    αθ = 0.1 * ones(Kτ)
    αH = zeros(Kτ)
    αA = zeros(Kτ)
    αS = zeros(Kτ)
    αF = zeros(Kτ)
    αR = zeros(2)
    αP = 0.
    σ = ones(3)

    σ_W = 1.
    σ_PF = 1.

    δ = 0.1*ones(Kτ)
    λ = 0.5*ones(Kτ)
    πW = 0.9
    ση = 1.

    βΓ = zeros(2)
    βw = zeros(Kτ+2)
    βf = zeros(Kτ+3+7)
    βτ = zeros(27,Kτ-1)
    πη = ones(2,Kη,Kτ,3) / Kη
    
    ηgrid = LinRange(-1,1,Kη-1)
    return pars(Kτ,Kη,β,wq,αC,αθ,αH,αA,αS,αF,αR,αP,σ,σ_W,σ_PF,δ,λ,πW,ση,βΓ,βw,βf,βτ,πη,ηgrid)
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

struct ddc_model
    T::Int64 #<- number of periods
    K::Int64 #<- number of states
    J::Int64 #<- number of choices
    β::Float64 #<- discount factor
    V::Matrix{Float64} #<- storage for V[x,t] (Emax function)
    v::Vector{Float64} #<- storage for v[j,x,t] (choice-specific values over all nodes)
    logP::Array{Float64,3} #<- storage for log(P[j|x,t]), choice probabilities over all nodes
    G::NestedLogitTree #<- structure for nested logit shocks
    σ::Vector{Float64} #<- scale parameter for shocks at each layer of the nested logit
    F::Array{SparseMatrixCSC{Float64,Int64},2} #<- storage for F_{t}[x'|x,j] in sparse structure
    π0::Vector{Float64} #<- distribution over states in the first period
    choice_set::Array{Bool,3} #<- indicates if j is in choice set for state k,t
end

function ddc_model(K,T,G::NestedLogitTree)
    J = G.nchoices
    V = zeros(K,2)
    v = zeros(G.nnodes)
    logP = zeros(G.nnodes,K,T)
    σ = ones(G.layers[end])
    F = [spzeros(K,K) for j in 1:J, t in 1:T]
    π0 = zeros(K)
    choice_set = fill(true,G.nnodes,K,T)
    return ddc_model(T,K,J,0.95,V,v,logP,G,σ,F,π0,choice_set)
end

function ddc_model(p::pars,G::NestedLogitTree,Kω)
    Tmax = 18*4
    TL = Kω>1
    K = 2 * p.Kη * p.Kτ * Kω
    m = ddc_model(K,Tmax,G)
    for t in 1:Tmax, j in 1:G.nchoices
        m.F[j,t] = F_j(j,p.Kη,Kω,p.Kτ,TL)
    end
    # update the choice set
    choice_set!(m,p.Kη,Kω,p.Kτ)
    return m
end


# below is an object for storing the derivatives of the model objects with respect to a set of parameters
struct ddc_derivative
    v::Matrix{Float64} #<- dv / dθ and dV / dθ all stacked together (?)
    V::Array{Float64,3}
    logP::Array{Float64,3} #<- don't store aross T as it uses too much memory (we think)
    F::Array{SparseMatrixCSC{Float64,Int64},3}
    u_idx::UnitRange{Int64} #<- location of parameters that affect utility
    F_idx::UnitRange{Int64} #<- location of parameters that effect F
    σ_idx::UnitRange{Int64} #<- location of parameters that determine σ
    β_idx::UnitRange{Int64} #<- location of β
end

function ddc_derivative(p::pars,m::ddc_model,Kω)
    TL = Kω>1
    nu = 5 * p.Kτ + 3 + 1 + 2 + 2*p.Kτ + 13
    nF = 2 * p.Kτ + 1
    nσ = 3
    u_idx = 1:nu
    F_idx = (nu+1):(nu+nF)
    σ_idx = (nu+nF+1):(nu+nF+3)
    β_idx = (nu+nF+4):(nu+nF+4)
    np = (nu+nF+4)
    v = zeros(np,size(m.v,1))
    V = zeros(np,size(m.V,1),2)
    logP = zeros(np,size(m.logP,1),size(m.logP,2))
    J = m.J
    K = m.K
    T = m.T
    dF = [F_j(j,p.Kη,Kω,p.Kτ,TL) for f in 1:nF, j in 1:J, t in 1:T]
    return ddc_derivative(v,V,logP,dF,u_idx,F_idx,σ_idx,β_idx)
end
# a function for solving the model given model data (in particular whether ∃ work requirements)

include("model/solve.jl")
include("model/update.jl")

include("model/choices.jl")

include("model/prices.jl")

include("budget.jl")
include("model/utility.jl")
include("model/states_transitions.jl")

using DelimitedFiles
# function to save the parameters
function savepars(p::pars,f::String)
    x = update_inv(p)
    writedlm(string("output/",f,"_main.csv"),x)
    writedlm(string("output/",f,"_types.csv"),p.βτ)
    writedlm(string("output/",f,"_eta.csv"),p.πη)
    writedlm(string("output/",f,"_sigma.csv"),[p.σ_W,p.σ_PF])
end
function loadpars!(p::pars,f::String)
    x = readdlm(string("output/",f,"_main.csv"))[:]
    update!(x,p)
    p.βτ[:] = readdlm(string("output/",f,"_types.csv"))[:]
    p.πη[:] = readdlm(string("output/",f,"_eta.csv"))[:]
    sigma = readdlm(string("output/",f,"_sigma.csv"))[:]
    p.σ_W = sigma[1]
    p.σ_PF = sigma[2]
end