# this function can't be called until after BaumWelch.jl defines s_inv
# it gives a distribution over choices and states jointly
function get_choice_state_distribution!(Π::SparseMatrixCSC{Float64,Int64},logP,model::ddc_model) #<- need choice probs? T?
    fill!(Π,0.)
    J = model.J
    K = size(model.V,1)
    T = size(Π,2)
    # initialize the first period:
    for k in 1:K
        if model.π0[k]>0
            for j in 1:J
                s = (k-1)*J + j
                Π[s,1] = model.π0[k] * exp(logP[j,k,1])
            end
        end
    end
    # iterate forward until the last period
    for t in 1:T-1 #
        for s in nzrange(Π,t)
            s_idx = Π.rowval[s]
            qs = Π.nzval[s]
            j,k_idx = s_inv(s_idx,J)
            for kn in nzrange(model.F[j,t],k_idx)
                kn_idx = model.F[j,t].rowval[kn]
                fkk = model.F[j,t][kn_idx,k_idx]
                for jn in 1:J
                    if model.choice_set[jn,kn_idx,t+1]
                        sn = (kn_idx - 1)*J + jn
                        Π[sn,t+1] += fkk * exp(logP[jn,kn_idx,t+1]) * qs
                    end
                end
            end
        end
    end
end

# this function solves for a distribution over states only (not choices)
# Π is not sparse (should it be?)
function get_state_distribution!(Π::Array{Float64,2},P,model::ddc_model) #<- need choice probs? T?
    J,K,T = size(model.u)
    fill!(Π,0.)
    @views Π[:,1] .= model.π0[:]
    for t in 1:T-1
        for k in 1:K #<- especially here
            for j in 1:J
                if model.choice_set[j,k,t]
                    for kn in nzrange(model.F[j,t],k)
                        kn_idx = model.F[j,t].rowval[kn]
                        Π[kn_idx,t+1] += P[j,k,t] * model.F[j,t][kn_idx,k]
                    end
                end
            end
        end 
    end
end

function em_mean(Π,t::Int64,model::ddc_model,pars,data,f::Function,condition::Function = (x->true))
    m = 0.
    d = 0.
    if issparse(Π)
        for s in nzrange(Π,t)
            s_idx = Π.rowval[s]
            if condition(s_idx)
                wght = Π[s_idx,t]
                m += wght * f(s_idx,t,model,pars,data)
                d += wght
            end
        end
    else
        for s in axes(Π,1)
            if condition(s)
                wght = Π[s,t]
                m += wght * f(s,t,model,pars,data)
                d += wght
            end
        end
    end

    return m / d
end