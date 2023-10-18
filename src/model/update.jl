
# a version here that calculates derivatives also
function pars(x,p)
    (;Kτ) = p
    # -- u_idx -- #
    # α parameters
    pos = 1
    αθ = exp.(x[pos:pos+Kτ-1])
    pos += Kτ
    αH = x[pos:pos+Kτ-1]
    pos += Kτ
    αA = x[pos:pos+Kτ-1]
    pos += Kτ
    αS = x[pos:pos+Kτ-1]
    pos += Kτ
    αF = x[pos:pos+Kτ-1]
    pos += Kτ
    αR = x[pos:pos+1]
    αP = x[pos+2]
    pos += 3
    # wq
    wq = exp(x[pos])
    pos += 1
    # βΓ 
    βΓ = x[pos:pos+1]
    pos += 2
    # βw
    βw = x[pos:pos+Kτ+1]
    pos += Kτ + 2
    # βf
    βf = x[pos:pos+Kτ+9]
    pos += Kτ + 10
    # ση
    ση = x[pos]
    pos += 1

    # ----- F_idx ------ #
    λ₀ = logit.(x[pos:pos+Kτ-1])
    pos += Kτ
    δ = logit.(x[pos:pos+Kτ-1])
    pos += Kτ
    λ₁ = logit.(x[pos:pos+Kτ-1])
    pos += Kτ
    μₒ = x[pos] #?
    σₒ = exp(x[pos+1])
    pos += 2

    # ----- σ_idx (nested logit dispersion) ---- #
    # σ (shock dispersion)
    σ = exp.(x[pos:pos+2])
    pos += 3

    # ---- β_idx (discounting) ----- #
    β = logit(x[pos])
    #pos += 1
    # -- others not in this step -- #
    #σ_W <- dispersion of measurement error for wages
    #σ_PF #<- dispersion of measurement error for childcare prices
    # πη #<- initial distribution of η for experimental samples
    # βτ #<- type selection
    p = (;p...,αA,αH,αθ,αS,αF,αP,αR,βΓ,wq,βw,βf,ση,λ₀,δ,λ₁,μₒ,σₒ,σ,β)
    p = update_transitions(p)
    return p
end

function pars_full(x,p)
    np = 10p.Kτ + 25
    p = pars(x[1:np],p)
    K = prod(size(p.βτ))
    βτ = reshape(x[(np+1):(np+K)],27,p.Kτ-1)
    np += K
    K = prod(size(p.πη))
    πη = reshape(x[(np+1):(np+K)],2,p.Kη,p.Kτ,3)
    np += K
    σ_W = x[np+1]
    σ_PF = x[np+2]
    return (;p...,βτ,πη,σ_W,σ_PF)
end

function pars(x,p,f::Vector{Symbol},tf::Vector{Int64}) #<-?
    pos = 1
    R = eltype(x)
    Xnew = Union{R,Vector{R}}[]
    for kf in eachindex(f)
        v = getfield(p,f[kf])
        K = length(v)
        if tf[kf]==1
            if K==1
                vnew = x[pos]
            else
                vnew = x[pos:(pos+K-1)]
            end
        elseif tf[kf]==2
            if K==1
                vnew = exp(x[pos])
            else
                vnew = exp.(x[pos:(pos+K-1)])
            end  
        elseif tf[kf]==3
            if K==1
                vnew = logit(x[pos])
            else
                vnew = logit.(x[pos:(pos+K-1)])
            end
        end
        pos += K
        push!(Xnew,vnew)
    end
    pnew = NamedTuple(zip(f,Xnew))
    p = (;p...,pnew...)
    p = update_transitions(p)
    return p
end

function pars_inv(p,f::Vector{Symbol},ft::Vector{Int64})
    x = Float64[]
    for kf in eachindex(f)
        if ft[kf]==1
            push!(x,getfield(p,f[kf])...)
        elseif ft[kf]==2
            push!(x,log.(getfield(p,f[kf]))...)
        elseif ft[kf]==3
            push!(x,logit_inv.(getfield(p,f[kf]))...)
        end
    end
    return x
end

function pars_inv(p)
    u = [log.(p.αθ);p.αH;p.αA;p.αS;p.αF;p.αR;p.αP;log(p.wq);p.βΓ;p.βw;p.βf;p.ση]
    F = [logit_inv.(p.λ₀);logit_inv.(p.δ);logit_inv.(p.λ₁);p.μₒ;log(p.σₒ)]
    σ = log.(p.σ)
    β = logit_inv(p.β)
    return [u;F;σ;β]
end

function pars_inv_full(p)
    x = pars_inv(p)
    push!(x,p.βτ...)
    push!(x,p.πη...)
    push!(x,p.σ_W)
    push!(x,p.σ_PF)
end
