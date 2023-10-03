# this "package" implements the Baum & Welch "forward-back" or "α-β" algorithm for constructing a posterior over states given observed outcomes in a Hidden Markov Model.

## 1. the structure here is general enough to handle missing data and sparse transition matrices
## 2. to achieve this generality, the user can write their own code to initialize α and fill out P given observed data
## 3. The algorithm works by construcing a posterior over s, which jointly indexes both states and observed discrete outcomes. This allows one to handle unobserved outcomes in particular periods.
## 4. see: baum_welch_example.jl for an example of how to implement this code.
## 5. it also provides generic routines for intializing and transition probabilities


using SparseArrays, LinearAlgebra

# this object is mutable container for the objects required to construct the α-β algorithm and construct posteriors over states and transitions
mutable struct EM_data
    α::SparseMatrixCSC{Float64,Int64} # α is an S x T sparse array that holds p(s_{t},y_{1}^{t})
    β::SparseMatrixCSC{Float64,Int64} # holds p(y_{t+1}^{T}|s_{t})
    q_s::SparseMatrixCSC{Float64,Int64} # holds the posterior given observed data
    q_ss::Vector{SparseMatrixCSC{Float64,Int64}} # a T-vector of posteriors over transitions
    P::Vector{SparseMatrixCSC{Float64,Int64}} #<- stores p(s'|s) given observed data which must be updated each time
end

# this function implements the α-β routine assuming that α, β, and P:
# (1) have been initialized given the current parameter guess
# (2) contain only entries in states that are "possible" given observed outcomes and possible transitions (this greatly speeds up iteration)
function forward_back!(α,β,P::Vector{SparseMatrixCSC{Float64,Int64}},q_s,q_ss)
    K,T = size(α)
    @inbounds for t in 2:T
        @views fill!(α[:,t],0.)
        for s in nzrange(α,t-1)
            s_idx = α.rowval[s]
            for sn in nzrange(P[t-1],s_idx)
                sn_idx = P[t-1].rowval[sn]
                α[sn_idx,t] += P[t-1][sn_idx,s_idx]*α[s_idx,t-1]
            end
        end
    end
    @inbounds for t in reverse(1:T-1)
        for s in nzrange(α,t)
            s_idx = α.rowval[s]
            β[s_idx,t] = 0.
            for sn in nzrange(P[t],s_idx)
                sn_idx = P[t].rowval[sn]
                β[s_idx,t] += P[t][sn_idx,s_idx]*β[sn_idx,t+1]
            end
        end
    end
    # create the posterior over states
    @inbounds for t in 1:T
        @views q_s[:,t] .= α[:,t].*β[:,t]
        @views q_s[:,t] ./= sum(q_s[:,t])
    end
    # create the posterior over transitions
    @inbounds for t in 1:T-1
        norm = 0.
        for s in nzrange(α,t)
            k = α.rowval[s]
            for j in nzrange(P[t],k)
                j_idx = P[t].rowval[j]
                q_ss[t][j_idx,k] = β[j_idx,t+1]*P[t][j_idx,k]*α[k,t]
                norm += q_ss[t][j_idx,k]
            end
        end
        #@views q_ss[t][:,:] ./= norm
        # now normalize:
        q_ss[t].nzval ./= norm
    end
end

forward_back!(em::EM_data) = forward_back!(em.α,em.β,em.P,em.q_s,em.q_ss)
function forward_back!(EM::Vector{EM_data})
    for em in EM
        forward_back!(em) 
    end
end

# this function calculates the full log-likelihood:
function log_likelihood(EM::Vector{EM_data})
    ll = 0.
    for em in EM
        ll += log(sum(em.α[:,1] .* em.β[:,1]))
    end
    return ll
end

# the three functions below provide a generic interface for setting up the problem given an array of conditional outcome probabilities (p_y) and a sparse transition matrix (F)
# because of the sparse containers, this is efficient even if there is no missing data, but is robust to that case also.

# first a simply way to index over combinations of state and outcomes:
# s = (k-1)*J + j where k indexes the state and j indexes the outcome
s_inv(s,J) = mod(s-1,J)+1,fld(s-1,J)+1


# this function sets up a sparse container for the α-β algorithm in a discrete hidden markov model, where:
# (1) each y∈{-1,1,...J} are either missing (-1) or take one of J values.  
# (2) F is an array of sparse transition matrices, with F[j,t] a sparse K x K transition matrix for outcome j at time t
# (3) π0 is an initial distribution over states
# calling this function sets up the state-observation combinations with non-zero entries and initalizes β properly
function get_EM_data(y,F,π0)
    J,T = size(F)
    T = length(y)#<= the panel is usually shorter than the time horizon in the model
    K = length(π0)
    S = K*J
    α = spzeros(S,T)
    β = spzeros(S,T)
    q_s = spzeros(S,T)
    q_ss = [spzeros(S,S) for t in 1:T-1]
    P = [spzeros(S,S) for t in 1:T-1]
    # initialize α: (which provides an index for which states have positive probability)
    # start with period 1:
    for k in 1:K
        if π0[k]>0
            if y[1]!=-1
                j = y[1]
                s = (k-1)*J + j
                α[s] = π0[k]
            else
                for j in 1:K
                    s = (k-1)*J + j
                    α[s] = π0[k]
                end
            end
        end
    end
    for t in 1:T-1
        # for each s in α: add non-zero entries for P and α?
        for s in nzrange(α,t)
            s_idx = α.rowval[s]
            j,k = s_inv(s_idx,J)
            for kn in nzrange(F[j,t],k)
                k_idx = F[j,t].rowval[kn]
                if y[t+1]!=-1
                    jn = y[t+1]
                    sn = (k_idx-1)*J+jn
                    P[t][sn,s_idx] = 1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                    α[sn,t+1] = 1. #p_y[jn,k_idx,t+1]
                else
                    for jn in 1:J
                        sn = (k_idx-1)*J+jn
                        P[t][sn,s_idx] =  1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                        α[sn,t+1] = 1. #p_y[jn,k_idx,t+1]
                    end
                end
            end
        end
    end
    # initialize β in last period
    for s in nzrange(α,T)
        s_idx = α.rowval[s]
        β[s_idx,T] = 1.
    end
    
    return EM_data(α,β,q_s,q_ss,P)
end

# this function initializes α[:,1] and updates P for the forward-back algo, assuming:
# (1) p_y is a J x K x T array of outcome probabilities given state k and time t
# (2) is the J x T array of sparse transition matrices
# (3) π0 is the initial distribution over states
# (4) some data is missing
function update!(EM::EM_data,p_y,F,π0)
    J,K,T = size(p_y)
    # start with the initial conditions:
    for s in nzrange(EM.α,1)
        s_idx = EM.α.rowval[s]
        j,k = s_inv(s_idx,J)
        EM.α.nzval[s] = π0[k]*p_y[j,k,1]
    end
    for t in eachindex(EM.P)
        for s in nzrange(EM.α,t)
            s_idx = EM.α.rowval[s]
            j,k = s_inv(s_idx,J)
            for sn in nzrange(EM.P[t],s_idx)
                sn_idx = EM.P[t].rowval[sn]
                jn,kn = s_inv(sn_idx,J)
                EM.P[t][sn_idx,s_idx] = F[j,t][kn,k]*p_y[jn,kn,t+1]
            end
        end
    end
end
