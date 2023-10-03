# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare

function F_j(j,Kη,Kω,Kτ,TL) #<-
    # TL indicates whether a time limit applies
    S,A,P,H,F = j_inv(j)
    col_idx = []
    row_idx = []
    val = []
    k_idx = LinearIndices((2,Kη,Kω,Kτ))
    for kτ in 1:Kτ, kω in 1:Kω, kη in 1:Kη, kA in 1:2
        kω_next = min(kω+A*TL,Kω)
        kA_next = 1+A
        k = k_idx[kA,kη,kω,kτ]

        if kη==1
            # remain "unemployed"
            k2 = k_idx[kA_next,kη,kω_next,kτ] #
            push!(col_idx,k)
            push!(row_idx,k2)
            push!(val,0.) #1-λ[kτ])
            # move:
            for kη2 in 2:Kη
                k2 = k_idx[kA_next,kη2,kω_next,kτ] #
                push!(col_idx,k)
                push!(row_idx,k2)
                push!(val,0.)#λ[kτ] / (Kη - 1))
            end
            
        else
            # "unemployment":
            k2 = k_idx[kA_next,1,kω_next,kτ] #
            push!(col_idx,k)
            push!(row_idx,k2)
            push!(val,0.) #δ[kτ]*(1-λ[kτ]))
            # no shock:
            for kη2 in max(2,kη-1):min(Kη,kη+1)
                k2 = k_idx[kA_next,kη2,kω_next,kτ] #
                push!(col_idx,k)
                push!(row_idx,k2)
                push!(val,0. )#δ[kτ] * λ[kτ] / (Kη - 1))
            end
        end
    end
    return sparse(row_idx,col_idx,Vector{Float64}(val))
end

function F_j!(p::pars,m::ddc_model,Kω::Int64)
    Kη = p.Kη
    Kτ = p.Kτ
    k_inv = CartesianIndices((2,Kη,Kω,Kτ))

    for t in axes(m.F,2)
        for k in 1:m.K
            for j in 1:m.J
                if m.choice_set[j,k,t]
                    _,kη,_,kτ = Tuple(k_inv[k])
                    for k_next in nzrange(m.F[j,t],k)
                        k_idx = m.F[j,t].rowval[k_next]
                        _,kη2,_,kτ = Tuple(k_inv[k_idx])

                        if kη>1
                            if kη2>1
                                diag_bool = kη==kη2
                                updown_bool = kη2==min(kη+1,Kη) || kη2==max(kη-1,2)
                                m.F[j,t].nzval[k_next] = (1 - p.δ[kτ]) * (diag_bool * p.πW + updown_bool*(1 - p.πW)/2)
                            else
                                m.F[j,t].nzval[k_next] = p.δ[kτ] 
                            end
                        else
                            if kη2>1
                                m.F[j,t].nzval[k_next] = p.λ[kτ] / (Kη - 1)
                            else
                                m.F[j,t].nzval[k_next] = (1-p.λ[kτ])
                            end
                        end
                    end
                end
            end
        end
    end
end

# filling in the derivatives assumes a parameter ordering of λ for all types then δ for all types, then πW
function F_j!(p::pars,m::ddc_model,∂m::ddc_derivative,Kω::Int64)
    Kη = p.Kη
    Kτ = p.Kτ
    k_inv = CartesianIndices((2,Kη,Kω,Kτ))
    dπW = p.πW*(1-p.πW) #<- multiply derivatives by this because πW is a logit transformation of the parameter
    for t in axes(m.F,2)
        for k in 1:m.K
            for j in 1:m.J
                if m.choice_set[j,k,t]
                    _,kη,_,kτ = Tuple(k_inv[k])
                    for k_next in nzrange(m.F[j,t],k)
                        k_idx = m.F[j,t].rowval[k_next]
                        _,kη2,_,kτ = Tuple(k_inv[k_idx])
                        dδ = p.δ[kτ] * (1-p.δ[kτ]) #<- multiply derivatives by this because δ is a logit transformation of the parameter
                        dλ = p.λ[kτ] * (1-p.λ[kτ]) #<- multiply derivatives by this because λ is a logit transformation of the parameter
                        
                        if kη>1
                            if kη2>1
                                diag_bool = kη==kη2
                                updown_bool = kη2==min(kη+1,Kη) || kη2==max(kη-1,2)
                                m.F[j,t].nzval[k_next] = (1 - p.δ[kτ]) * (diag_bool * p.πW + updown_bool*(1 - p.πW)/2)
                                ∂m.F[Kτ+kτ,j,t].nzval[k_next] = - (diag_bool * p.πW + updown_bool*(1 - p.πW)/2) * dδ
                                ∂m.F[2Kτ+1,j,t].nzval[k_next] = (1 - p.δ[kτ]) * (diag_bool - updown_bool/2) * dπW
                            else
                                m.F[j,t].nzval[k_next] = p.δ[kτ]
                                ∂m.F[Kτ+kτ,j,t].nzval[k_next] = dδ
                            end
                        else
                            if kη2>1
                                m.F[j,t].nzval[k_next] = p.λ[kτ] / (Kη - 1)
                                ∂m.F[kτ,j,t].nzval[k_next] = 1 / (Kη - 1) * dλ
                            else
                                m.F[j,t].nzval[k_next] = (1-p.λ[kτ])
                                ∂m.F[kτ,j,t].nzval[k_next] = - 1 * dλ
                            end
                        end
                    end
                end
            end
        end
    end
end