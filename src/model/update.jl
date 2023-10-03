
# a version here that calculates derivatives also
function update!(x,p::pars)
    # -- u_idx -- #
    # α parameters
    pos = 1
    p.αθ = exp.(x[pos:pos+p.Kτ-1])
    pos += p.Kτ
    p.αH = x[pos:pos+p.Kτ-1]
    pos += p.Kτ
    p.αA = x[pos:pos+p.Kτ-1]
    pos += p.Kτ
    p.αS = x[pos:pos+p.Kτ-1]
    pos += p.Kτ
    p.αF = x[pos:pos+p.Kτ-1]
    pos += p.Kτ
    p.αR = x[pos:pos+1]
    p.αP = x[pos+2]
    pos += 3
    # wq
    p.wq = exp(x[pos])
    pos += 1
    # βΓ 
    p.βΓ = x[pos:pos+1]
    pos += 2
    # βw
    p.βw = x[pos:pos+p.Kτ+1]
    pos += p.Kτ + 2
    # βf
    p.βf = x[pos:pos+p.Kτ+9]
    pos += p.Kτ + 10
    # ση
    p.ση = x[pos]
    pos += 1

    # ----- F_idx ------ #
    p.λ[:] = logit.(x[pos:pos+p.Kτ-1])
    pos += p.Kτ
    p.δ[:] = logit.(x[pos:pos+p.Kτ-1])
    pos += p.Kτ
    p.πW = logit(x[pos])
    pos += 1

    # ----- σ_idx (nested logit dispersion) ---- #
    # σ (shock dispersion)
    p.σ[:] = exp.(x[pos:pos+2])
    pos += 3

    # ---- β_idx (discounting) ----- #
    p.β = logit(x[pos])
    #pos += 1
    # -- others not in this step -- #
    #σ_W <- dispersion of measurement error for wages
    #σ_PF #<- dispersion of measurement error for childcare prices
    # πη #<- initial distribution of η for experimental samples
    # βτ #<- type selection
    return nothing
end

function update_inv(p::pars)
    u = [log.(p.αθ);p.αH;p.αA;p.αS;p.αF;p.αR;p.αP;log(p.wq);p.βΓ;p.βw;p.βf;p.ση]
    F = [logit_inv.(p.λ);logit_inv.(p.δ);logit_inv(p.πW)]
    σ = log.(p.σ)
    β = logit_inv(p.β)
    return [u;F;σ;β]
end
# update the derivative g to account for the fact that many parameters are transformed.
function apply_chain_rule!(g,p::pars,σ_idx::UnitRange{Int64},β_idx::UnitRange{Int64})
    # note: transformation of parameters of utility are accounted for already in utility()
    # note: transformation of F_j parameterd (λ and δ) is dealt with already in F_j!
    # σ:
    @views g[σ_idx] .*= p.σ #<- exp transformation
    # β:
    @views g[β_idx] *= p.β*(1-p.β) #<- logit transformation
end


# this function updates the model with all parameters that don't depend on individual data
# -- i.e. transition matrices, dispersion parameters, and discounting
function update!(p::pars,M::Vector{ddc_model},Kω)
    for mi in eachindex(M)
        F_j!(p,M[mi],Kω[mi])
    end
    return nothing
end
function update!(p::pars,M::Vector{ddc_model})
    for mi in eachindex(M)
        F_j!(p,M[mi],1)
    end
    return nothing
end

function update!(p::pars,M::Matrix{ddc_model},Kω)
    for i in axes(M,1), j in axes(M,2)
        F_j!(p,M[i,j],Kω[i])
    end
    return nothing
end

function update!(p::pars,M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},Kω)
    for i in axes(M,1), j in axes(M,2)
        F_j!(p,M[i,j],∂M[i,j],Kω[i])
    end
    return nothing
end

function update(p::pars,Kω)
    M = [ddc_model(p,NL,kω) for kω in Kω]
    update!(p,M,Kω)
    return M
end
