
function solve!(m::ddc_model,md::model_data,p::pars)
    T = max((17-md.ageyng)*4,md.T)
    J = m.J
    K = m.K
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    fill!(m.V,0.)
    tnow = 1
    for t in reverse(1:T)
        tnext = 3-tnow #<- tnow = 1 => tnext =2, tnow=2 => tnext =1
        for k in 1:K
            fill!(m.v,0.) #<-?
            for j in 1:J
                if m.choice_set[j,k,t]
                    kA,kη,kω,kτ = Tuple(k_inv[k])
                    S,A,P,H,F = j_inv(j)
                    age_yng = md.ageyng + fld(t,4)
                    eligible = (kω<md.Kω || !md.TL) && (age_yng<18) #<- change what we mean by eligible

                    @inbounds m.v[j] = utility(S,A,H,F,p,md,kA,kη,kτ,t,eligible)

                    # calculate the continuation value for the choice
                    for k_ind in nzrange(m.F[j,t],k)#  
                        @inbounds k2 = m.F[j,t].rowval[k_ind]
                        @inbounds Fjt = m.F[j,t].nzval[k_ind]
                        @inbounds m.v[j] +=  p.β * Fjt *m.V[k2,tnext]
                    end
                end
            end
            # now normalize by the highest value:
            @views choice_probs!(m.logP[:,k,t],m.v,p.σ,m.choice_set[:,k,t],m.G)
            @inbounds m.V[k,tnow] = m.v[end]
        end
        # swap locations of next period and current period for V 
        # if tnow = 1, tnext
        tnow = tnext #<- if 1, moves to 2, if 2, moves to 1.
    end
    return nothing
end

function solve!(m::ddc_model,∂m::ddc_derivative,md::model_data,p::pars,t::Int64,tnow::Int64,tnext::Int64)
    J = m.J
    K = m.K
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    fill!(∂m.logP,0.)
    for k in 1:K
        fill!(∂m.v,0.)
        fill!(m.v,0.) #<-?
        for j in 1:J
            if m.choice_set[j,k,t]

                kA,kη,kω,kτ = Tuple(k_inv[k])
                S,A,P,H,F = j_inv(j)
                age_yng = md.ageyng + fld(t,4)
                eligible = (kω<md.Kω || !md.TL) && (age_yng<18) #<- change what we mean by eligible

                @inbounds @views m.v[j] = utility(∂m.v[:,j],S,A,H,F,p,md,kA,kη,kτ,t,eligible)

                # calculate the continuation value for the choice
                for k_ind in nzrange(m.F[j,t],k)#  
                    k2 = m.F[j,t].rowval[k_ind]
                    Fjt = m.F[j,t].nzval[k_ind]
                    βF = p.β*Fjt
                    @inbounds m.v[j] += βF*m.V[k2,tnext]
                    for p_idx in ∂m.u_idx
                        @inbounds ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                    end
                    for p_idx in ∂m.β_idx
                        @inbounds ∂m.v[p_idx,j] += Fjt * m.V[k2,tnext]
                        @inbounds ∂m.v[p_idx,j] += βF * ∂m.V[p_idx,k2,tnext]
                    end
                    for p_idx in ∂m.σ_idx
                        @inbounds ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                    end
                    l = 1
                    for p_idx in ∂m.F_idx
                        @inbounds ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                        @inbounds ∂m.v[p_idx,j] += p.β*∂m.F[l,j,t].nzval[k_ind]*m.V[k2,tnext]
                        l += 1
                    end
                end
            end
        end
        # now normalize by the highest value:
        @inbounds @views choice_probs!(m.logP[:,k,t],m.v,p.σ,m.choice_set[:,k,t],m.G)
        m.V[k,tnow] = m.v[end]
        
        # now finish the recursion for parameters that effect u, F, and σ
        @inbounds @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.u_idx,:,k],∂m.v[∂m.u_idx,:],m.choice_set[:,k,t],m.G)
        # F enters only through v
        @inbounds @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.F_idx,:,k],∂m.v[∂m.F_idx,:],m.choice_set[:,k,t],m.G)
        # β
        @inbounds @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.β_idx,:,k],∂m.v[∂m.β_idx,:],m.choice_set[:,k,t],m.G)
        # σ is enters into V both through v and through Emax
        @inbounds @views get_derivatives_dσ!(m.logP[:,k,t],m.v,p.σ,∂m.logP[∂m.σ_idx,:,k],∂m.v[∂m.σ_idx,:],m.choice_set[:,k,t],m.G,false)

        # copy results for the next iteration.
        @inbounds @views ∂m.V[:,k,tnow] .= ∂m.v[:,end]
    end
    return nothing
end

function solve!(m::ddc_model,∂m::ddc_derivative,md::model_data,p::pars)
    T = max((17-md.ageyng)*4,md.T)
    J = m.J
    K = m.K
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    fill!(m.V,0.)
    fill!(∂m.v,0.)
    fill!(∂m.V,0.)
    @views fill!(∂m.logP,0.)
    tnow = 1
    for t in reverse(1:T)
        tnext = 3-tnow #<- tnow = 1 => tnext =2, tnow=2 => tnext =1
        for k in 1:K
            fill!(∂m.v,0.)
            fill!(m.v,0.) #<-?
            for j in 1:J
                if m.choice_set[j,k,t]

                    kA,kη,kω,kτ = Tuple(k_inv[k])
                    S,A,P,H,F = j_inv(j)
                    age_yng = md.ageyng + fld(t,4)
                    eligible = (kω<md.Kω) && (age_yng<18)

                    @views m.v[j] = utility(∂m.v[:,j],S,A,H,F,p,md,kA,kη,kτ,t,eligible)

                    # calculate the continuation value for the choice
                    for k_ind in nzrange(m.F[j,t],k)#  
                        k2 = m.F[j,t].rowval[k_ind]
                        Fjt = m.F[j,t].nzval[k_ind]
                        βF = p.β*Fjt
                        m.v[j] += βF*m.V[k2,tnext]
                        for p_idx in ∂m.u_idx
                            ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                        end
                        for p_idx in ∂m.β_idx
                            ∂m.v[p_idx,j] += Fjt * m.V[k2,tnext]
                            ∂m.v[p_idx,j] += βF * ∂m.V[p_idx,k2,tnext]
                        end
                        for p_idx in ∂m.σ_idx
                            ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                        end
                        l = 1
                        for p_idx in ∂m.F_idx
                            ∂m.v[p_idx,j] += βF*∂m.V[p_idx,k2,tnext]
                            ∂m.v[p_idx,j] += p.β*∂m.F[l,j,t].nzval[k_ind]*m.V[k2,tnext]
                            l += 1
                        end
                    end
                end
            end
            # now normalize by the highest value:
            @views choice_probs!(m.logP[:,k,t],m.v,p.σ,m.choice_set[:,k,t],m.G)
            m.V[k,tnow] = m.v[end]
            
            # now finish the recursion for parameters that effect u, F, and σ
            @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.u_idx,:,k,t],∂m.v[∂m.u_idx,:],m.choice_set[:,k,t],m.G)
            # F enters only through v
            @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.F_idx,:,k,t],∂m.v[∂m.F_idx,:],m.choice_set[:,k,t],m.G)
            # β
            @views get_derivatives_du!(m.logP[:,k,t],p.σ,∂m.logP[∂m.β_idx,:,k,t],∂m.v[∂m.β_idx,:],m.choice_set[:,k,t],m.G)
            # σ is enters into V both through v and through Emax
            @views get_derivatives_dσ!(m.logP[:,k,t],m.v,p.σ,∂m.logP[∂m.σ_idx,:,k,t],∂m.v[∂m.σ_idx,:],m.choice_set[:,k,t],m.G,false)

            # copy results for the next iteration.
            @views ∂m.V[:,k,tnow] .= ∂m.v[:,end]
        end
        # swap locations of next period and current period for V 
        # if tnow = 1, tnext
        tnow = tnext #<- if 1, moves to 2, if 2, moves to 1.
    end
    return nothing
end

function reduce_choice_probabilities!(model)
    for t in axes(model.logP,3), k in axes(model.logP,2)
        @views reduce_choice_probs!(model.logP[:,k,t],model.choice_set[:,k,t],model.G)
    end
end
