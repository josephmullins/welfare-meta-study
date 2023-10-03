include("NestedLogit.jl")
using SparseArrays
using LoopVectorization

mutable struct ddc_model{R}
    T::Int64 #<- number of periods
    K::Int64 #<- number of states
    J::Int64 #<- number of choices
    β::R #<- discount factor
    V::Array{R,2} #<- storage for V[x,t] (Emax function)
    v::Array{R,3} #<- storage for v[j,x,t] (choice-specific values over all nodes)
    u::Array{R,3} #<- storage for u[j,x,t] (period utility)
    logP::Array{R,3} #<- storage for log(P[j|x,t]), choice probabilities over all nodes
    G::NestedLogitTree #<- structure for nested logit shocks
    σ::Vector{R} #<- scale parameter for shocks at each layer of the nested logit
    F::Array{SparseMatrixCSC{R,Int64},2} #<- storage for F_{t}[x'|x,j] in sparse structure
    π0::Vector{R} #<- distribution over states in the first period
    choice_set::Array{Bool,3} #<- indicates if j is in choice set for state k,t
end

function ddc_model(K,T,G::NestedLogitTree)
    J = G.nchoices
    V = zeros(K,T+1)
    v = zeros(G.nnodes,K,T+1)
    u = zeros(J,K,T)
    logP = zeros(G.nnodes,K,T)
    σ = ones(G.layers[end])
    F = [spzeros(K,K) for j in 1:J, t in 1:T]
    π0 = zeros(K)
    choice_set = fill(true,G.nnodes,K,T)
    return ddc_model(T,K,J,0.95,V,v,u,logP,G,σ,F,π0,choice_set)
end


# below is an object for storing the derivatives of the model objects with respect to a set of parameters
mutable struct ddc_derivative
    v::Array{Float64,4} #<- dv / dθ and dV / dθ all stacked together (?)
    u::Array{Float64,4} #<- du / dθ
    logP::Array{Float64,4} #<- dlogP / dθ (?)
    F::Array{SparseMatrixCSC{Float64,Int64},3}
    u_idx::UnitRange{Int64} #<- location of parameters that affect utility
    F_idx::UnitRange{Int64} #<- location of parameters that effect F
    σ_idx::UnitRange{Int64} #<- location of parameters that determine σ
    β_idx::UnitRange{Int64} #<- location of β
end

function solve_static!(dp::ddc_model,∂dp::ddc_derivative)
    #@views fill!(∂dp.v,0.)
    @views fill!(∂dp.logP,0.)
    for t in reverse(1:dp.T)
        for k in 1:dp.K
            @views choice_probs!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,dp.choice_set[:,k,t],dp.G)
            # calculate derivatives with respect to utility parameters 
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[∂dp.u_idx,:,k,t],∂dp.v[∂dp.u_idx,:,k,t],dp.choice_set[:,k,t],dp.G)
            # calculate derivatives with respect to scale parameters
            @views get_derivatives_dσ!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,∂dp.logP[∂dp.σ_idx,:,k,t],∂dp.v[∂dp.σ_idx,:,k,t],dp.choice_set[:,k,t],dp.G,false)
        end
    end
end
function solve_static!(dp::ddc_model)
    for t in reverse(1:dp.T)
        for k in 1:dp.K
            @views choice_probs!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,dp.choice_set[:,k,t],dp.G)
        end
    end
end


function backward_induction!(dp)
    @views fill!(dp.V[:,dp.T+1],0.)
    for t in reverse(1:dp.T)
        for k in 1:dp.K
            for j in 1:dp.J
                if dp.choice_set[j,k,t]
                    dp.v[j,k,t] = dp.u[j,k,t]
                    cv = 0.
                    # calculate the continuation value for the choice
                    for k_ind in nzrange(dp.F[j,t],k)#  
                        k2 = dp.F[j,t].rowval[k_ind]
                        cv += dp.β*dp.F[j,t].nzval[k_ind]*dp.V[k2,t+1]
                    end
                    dp.v[j,k,t] += cv
                end
            end
            # now normalize by the highest value:
            @views choice_probs!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,dp.choice_set[:,k,t],dp.G)
            dp.V[k,t] = dp.v[end,k,t] #<- is V really necessary?
        end
    end
end

function backward_induction!(dp::ddc_model,∂dp::ddc_derivative)
    Kθ,J,K,T = size(∂dp.u)
    @views fill!(dp.V[:,dp.T+1],0.)
    @views fill!(dp.v[:,:,dp.T+1],0.)
    @views fill!(∂dp.v,0.)
    @views fill!(∂dp.logP,0.)

    for t in reverse(1:dp.T)
        for k in 1:K
            for j in 1:J
                if dp.choice_set[j,k,t]
                    dp.v[j,k,t] = dp.u[j,k,t]
                    for p in 1:Kθ
                        ∂dp.v[p,j,k,t] = ∂dp.u[p,j,k,t]
                    end
                    # calculate the continuation value for the choice
                    for k_ind in nzrange(dp.F[j,t],k)#  
                        k2 = dp.F[j,t].rowval[k_ind]
                        F = dp.F[j,t].nzval[k_ind]
                        βF = dp.β*F
                        dp.v[j,k,t] += βF*dp.V[k2,t+1]
                        for p in ∂dp.u_idx
                            ∂dp.v[p,j,k,t] += βF*∂dp.v[p,end,k2,t+1]
                        end
                        for p in ∂dp.β_idx
                            ∂dp.v[p,j,k,t] += F * dp.v[end,k2,t+1]
                            ∂dp.v[p,j,k,t] += βF * ∂dp.v[p,end,k2,t+1]
                        end
                        for p in ∂dp.σ_idx
                            ∂dp.v[p,j,k,t] += βF*∂dp.v[p,end,k2,t+1]
                        end
                        l = 1
                        for p in ∂dp.F_idx
                            ∂dp.v[p,j,k,t] += βF*∂dp.v[p,end,k2,t+1]
                            ∂dp.v[p,j,k,t] += dp.β*∂dp.F[l,j,t].nzval[k_ind]*dp.v[end,k2,t+1]
                            l += 1
                        end
                    end
                end
            end
            # now normalize by the highest value:
            @views choice_probs!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,dp.choice_set[:,k,t],dp.G)
            dp.V[k,t] = dp.v[end,k,t]
            
            # now finish the recursion for parameters that effect u, F, and σ
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[∂dp.u_idx,:,k,t],∂dp.v[∂dp.u_idx,:,k,t],dp.choice_set[:,k,t],dp.G)
            # F enters only through v
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[∂dp.F_idx,:,k,t],∂dp.v[∂dp.F_idx,:,k,t],dp.choice_set[:,k,t],dp.G)
            # β
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[∂dp.β_idx,:,k,t],∂dp.v[∂dp.β_idx,:,k,t],dp.choice_set[:,k,t],dp.G)
            # σ is enters into V both through v and through Emax
            @views get_derivatives_dσ!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,∂dp.logP[∂dp.σ_idx,:,k,t],∂dp.v[∂dp.σ_idx,:,k,t],dp.choice_set[:,k,t],dp.G,false)

        end 
    end
end


# this function gets derivatives wrt θ given an initial du/dθ
# NOTE: this function only works if:
# (1) the model dp has been solved; and
# (2) the option to reduce the conditional node probabilities logP to unconditional choice probabilities has *not* been invoked.
function calc_derivatives_du!(dp::ddc_model,∂dp::ddc_derivative,idx::UnitRange)
    Kθ,J,K,T = size(∂dp.u)
    for t in reverse(1:dp.T-1)
        for k in 1:K
            for j in 1:J
                if dp.choice_set[j,k,t]
                    for p in 1:Kθ
                        ∂dp.v[p,j,k,t] = ∂dp.u[p,j,k,t]
                        cv = 0.
                        # calculate the continuation value for the choice
                        for k_ind in nzrange(dp.F[j,t],k)#  
                            k2 = dp.F[j,t].rowval[k_ind]
                            cv += dp.β*dp.F[j,t].nzval[k_ind]*∂dp.v[p,end,k2,t+1]
                        end
                        ∂dp.v[p,j,k,t] += cv
                    end
                end
            end
            # w/ ∂v/∂θ. this function calculates ∂V'/∂θ and ∂logP' / ∂θ
            #logP,σ,dlogP,dV,NL
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[idx,:,k,t],∂dp.v[idx,:,k,t],dp.choice_set[:,k,t],dp.G)
        end
    end
end

function calc_derivatives_dσ!(dp::ddc_model,∂dp::ddc_derivative,idx::UnitRange)
    Kθ,J,K,T = size(∂dp.u)
    @views fill!(∂dp.v,0.)
    @views fill!(∂dp.logP,0.)
    for t in reverse(1:dp.T-1)
        for k in 1:K
            for j in 1:J
                if dp.choice_set[j,k,t]
                    for p in 1:Kθ
                        ∂dp.v[p,j,k,t] = 0.
                        cv = 0.
                        # calculate the continuation value for the choice
                        for k_ind in nzrange(dp.F[j,t],k)#  
                            k2 = dp.F[j,t].rowval[k_ind]
                            cv += dp.β*dp.F[j,t].nzval[k_ind]*∂dp.v[p,end,k2,t+1]
                        end
                        ∂dp.v[p,j,k,t] += cv
                    end
                end
            end
            @views get_derivatives_dσ!(dp.logP[:,k,t],dp.v[:,k,t],dp.σ,∂dp.logP[idx,:,k,t],∂dp.v[idx,:,k,t],dp.choice_set[:,k,t],dp.G,false)
        end
    end
end

function calc_derivatives_dF!(dp::ddc_model,∂dp::ddc_derivative,dF,idx::UnitRange)
    Kθ,J,K,T = size(∂dp.u)
    for t in reverse(1:dp.T-1)
        for k in 1:K
            for j in 1:J
                if dp.choice_set[j,k,t]
                    for p in 1:Kθ
                        ∂dp.v[p,j,k,t] = 0.
                        cv = 0.
                        # calculate the derivative with respect to continuation value
                        for k_ind in nzrange(dp.F[j,t],k)#  
                            k2 = dp.F[j,t].rowval[k_ind]
                            # this line captures contribution of each dV[k',t+1] to dv[j,k,t]/dθ
                            cv += dp.β*dp.F[j,t].nzval[k_ind]*∂dp.v[p,end,k2,t+1]
                            # this line captures the direct contribution of dF[k'|k,j,t]/dθ
                            cv += dp.β*dF[p,j,t].nzval[k_ind]*dp.v[end,k2,t+1]
                        end
                        ∂dp.v[p,j,k,t] += cv
                    end
                end
            end
            # w/ ∂v/∂θ. this function calculates ∂V'/∂θ and ∂logP' / ∂θ
            #logP,σ,dlogP,dV,NL
            @views get_derivatives_du!(dp.logP[:,k,t],dp.σ,∂dp.logP[idx,:,k,t],∂dp.v[idx,:,k,t],dp.G)
        end
    end
end

function reduce_choice_probabilities!(P,model)
    for t in axes(P,3), k in axes(P,2)
        @views reduce_choice_probs!(P[:,k,t],model.logP[:,k,t],model.choice_set[:,k,t],model.G)
    end
end

function reduce_choice_probabilities!(model)
    for t in axes(model.logP,3), k in axes(model.logP,2)
        @views reduce_choice_probs!(model.logP[:,k,t],model.choice_set[:,k,t],model.G)
    end
end
