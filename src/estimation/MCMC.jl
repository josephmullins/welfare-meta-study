# the basis of estimation is the quasi-likelihood:

# G(θ_0) ∼ N(0,V[G])
# and so: G(θ_0)V^{-1}G() is...?
# 
using Turing, LinearAlgebra

# would have to change this if adding type-specific parameters
function get_zxb!(zxb,pars,data)
    thB,thC = predict_skills(pars,data)
    for iz in eachindex(data.Z)
        zxb[pos] += thB * data.Z[iz]
        pos += 1
    end
    for iz in eachindex(data.Z)
        zxb[pos] += thC * data.Z[iz]
        pos += 1
    end
end


@model function quasilikelihood(zy,data,factor_scores)
    num_X = size(data[1].X,1)
    δI ~ [Uniform(0,3) for i in 1:2]
    δθ ~ [Uniform(0.5,1) for i in 1:2]
    g ~ [Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ [Flat() for i in 1:num_X]
    pars = (δI = δI,δθ = δθ,g = g, β = β)
    # get variance
    V = moment_variance(pars,data,factor_scores) / N
    zxb = zeros(eltype(δI))
    for d in data
        get_zxb!(zxb,pars,d)
    end
    N = length(data)
    zy ~ MvNormal(zxb / N,V)
end

@model function test_like(y,x)
    β ~ MultivariateNormal(zeros(3),I(3))
    for i in eachindex(y)
        y[i] ~ Normal(dot(x[:,i],β),1)
    end
end

@model function test_like(data)
    β ~ MultivariateNormal(zeros(3),I(3))
    for d in data
        d.y ~ Normal(dot(d.X,β),1)
    end
end

N = 1000
β0 = [1.,0.,0.5]
x = rand(3,N)
Fϵ = Normal()
y = x'*β0 .+ rand(Fϵ,N)
data = [(X = x[:,i],y = dot(β0,x[:,i])+rand(Fϵ)) for i in 1:N]

sample(test_like(data),NUTS(),1000)

@model function test_like(y,data)
    β ~ MultivariateNormal(zeros(3),I(3))
    #r = zeros(eltype(β),length(y))
    N = length(y)
    r = zero(eltype(β))
    for i in eachindex(data)
        r += y[i] - dot(data[i].X,β)
        #y[i] ~ Normal(dot(data[i].X,β),1)
    end
    r /= N
    r ~ Normal(0,1/N)
end

sample(test_like(y,data),NUTS(),1000)

# ---- test:
N = 1000
β0 = [1.,0.,0.5]
z = rand(N,3)
γ = rand(3,3)
x = z * 3 + 0.2*rand(Normal(),N,3)
Fϵ = Normal()
y = x*β0 .+ rand(Fϵ,N)

zy = (z' * y)[:] / N
zx = z' * x / N
V = z' * z / N^2

# let's test a quasilikelihood:
# Var[1 / N ∑[(y-xβ)x]] = (1 / N) σ² 𝔼[x'x]
#!?
# hmm. not sure about this? can we do thinks this way?
# CLT says: g(X,β) →_d N(0,V[g])

@model function moment_like(zy,zx,V)
    β ~ MultivariateNormal(zeros(3),I(3))
    zy ~ MultivariateNormal(zx*β,V)
end

sample(moment_like(zy,zx,V),NUTS(),1000)

# NEXT: do the same as above but with V "estimated" and updated each time.
# this is it. it works. job now is to figure out the domain stuff.
# NOTE: this also shows that Turing already does the bijection for you
@model function moment_like(zy,zx,y,z,x,::Type{T} = Float64) where {T}
    #β ~ MultivariateNormal(zeros(3),I(3))
    #β ~ fill(Uniform(-2,2),3)
    β = Vector{T}(undef,3)
    for i in 1:3
        β[i] ~ Uniform(-2,2)
    end
    e = y - x * β
    V = (z'z / N) * (e'e / N)
    zy ~ MultivariateNormal(zx*β,V / N)
end

chain = sample(moment_like(zy,zx,y,z,x),NUTS(),1000)
