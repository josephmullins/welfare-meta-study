
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
    αR = x[pos]
    λR = x[pos+1]
    αP = x[pos+2]
    pos += 3
    # wq
    wq = exp.(x[pos:pos+Kτ-1])
    pos += Kτ    
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
    σ₁ = exp.(x[pos:pos+Kτ-1])
    pos += Kτ
    σ₂ = exp.(x[pos:pos+Kτ-1])
    pos += Kτ
    σ₃ = exp.(x[pos:pos+Kτ-1])
    pos += Kτ

    # ---- β_idx (discounting) ----- #
    β = logit(x[pos])
    #pos += 1
    # -- others not in this step -- #
    #σ_W <- dispersion of measurement error for wages
    #σ_PF #<- dispersion of measurement error for childcare prices
    # πη #<- initial distribution of η for experimental samples
    # βτ #<- type selection
    p = (;p...,αA,αH,αθ,αS,αF,αP,αR,λR,βΓ,wq,βw,βf,ση,λ₀,δ,λ₁,μₒ,σₒ,σ₁,σ₂,σ₃,β)
    p = update_transitions(p)
    return p
end

function pars_full(x,p)
    np = 14p.Kτ + 21
    p = pars(x[1:np],p)
    K = prod(size(p.βτ))
    βτ = reshape(x[(np+1):(np+K)],23,p.Kτ-1)
    np += K
    K = prod(size(p.πη))
    πη = reshape(x[(np+1):(np+K)],2,p.Kη,p.Kτ,3)
    np += K
    σ_W = x[np+1]
    σ_PF = x[np+2]
    #μ_PF = x[np+3]
    #σ_PF2 = exp(x[np+4])
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
    p = update_transitions(x,p)
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
    u = [log.(p.αθ);p.αH;p.αA;p.αS;p.αF;p.αR;p.λR;p.αP;log.(p.wq);p.βΓ;p.βw;p.βf;p.ση]
    F = [logit_inv.(p.λ₀);logit_inv.(p.δ);logit_inv.(p.λ₁);p.μₒ;log(p.σₒ)]
    #σ = log.(p.σ)
    β = logit_inv(p.β)
    return [u;F;log.(p.σ₁);log.(p.σ₂);log.(p.σ₃);β]
end

function pars_inv_full(p)
    x = pars_inv(p)
    push!(x,p.βτ...)
    push!(x,p.πη...)
    push!(x,p.σ_W)
    push!(x,p.σ_PF)
    #push!(x,p.μ_PF)
    #push!(x,log(p.σ_PF2))
end

# a key-based update routine for a specific type (k)
function pars(x,p,kτ::Int64,f::Vector{Symbol},tf::Vector{Int64}) #<-?
    pos = 1
    R = eltype(x)
    Xnew = Vector{R}[]
    for kf in eachindex(f)
        v = getfield(p,f[kf])
        K = length(v)
        if tf[kf]==1
            vnew = convert(Vector{R},v)
            vnew[kτ] = x[pos]
        elseif tf[kf]==2
            vnew = convert(Vector{R},v)
            vnew[kτ] = exp(x[pos])
        elseif tf[kf]==3
            vnew = convert(Vector{R},v)
            vnew[kτ] = logit(x[pos])
        end
        pos += 1
        push!(Xnew,vnew)
    end
    pnew = NamedTuple(zip(f,Xnew))
    p = (;p...,pnew...)
    p = update_transitions(x,p)
    return p
end

function pars_inv(p,kτ::Int64,f::Vector{Symbol},ft::Vector{Int64})
    x = Float64[]
    for kf in eachindex(f)
        if ft[kf]==1
            push!(x,getfield(p,f[kf])[kτ])
        elseif ft[kf]==2
            push!(x,log(getfield(p,f[kf])[kτ]))
        elseif ft[kf]==3
            push!(x,logit_inv(getfield(p,f[kf])[kτ]))
        end
    end
    return x
end
