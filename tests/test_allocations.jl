
function getp(α)
    return (;α)
end

function plain_logit!(logP,vj)
    norm = 0.
    vmax = -Inf
    for j ∈ eachindex(vj)
        vj[j]>vmax ? vmax=vj[j] : nothing
    end
    for j ∈ eachindex(vj)
        norm += exp( vj[j] - vmax )
    end
    norm = log(norm)
    for j ∈ eachindex(vj)
        logP[j] = (vj[j] - vmax) - norm
    end
    vj[1] = vmax + norm
end

function utility(j,x,α)
    return α[1] * (j-x)^2 + α[2] * j
end

function utility(R::DataType,j,x,α)::R
    return α[1] * (j-x)^2 + α[2] * j
end


function solve_model(α,K)
    J = 5
    v = zeros(eltype(α),J,K)
    logP = zeros(eltype(α),J,K)
    for x in 1:K
        for j in 1:J
            v[j,x] = utility(j,x,α)
        end
        #@views plain_logit!(logP[:,x],v[:,x])
    end
    return sum(logP)
end

function solve_model(R,α,K)
    J = 5
    v = zeros(R,J,K)
    logP = zeros(R,J,K)
    for x in 1:K
        for j in 1:J
            v[j,x] = utility(j,x,α)
        end
        #@views plain_logit!(logP[:,x],v[:,x])
    end
    return sum(logP)
end

function solve_model2(α,K)
    R = eltype(α)
    solve_model(R,α,K)
end


α = [2.,1.]
p = (;α)

solve_model(α,100)
solve_model(Float64,α,100)
solve_model2(α,100)

@time solve_model(α,100)
@time solve_model(Float64,α,100) # this is some fascinating shit.
@time solve_model2(α,100) #now this is some weird ass shit
# so how do we pass this?

# function fill_A!(A,x)
#     for k in eachindex(A)
#         A[k] = log(x * k)
#     end
# end

function test_f1(x,K)
    A = zeros(eltype(x),5,K,2)
    #fill_A!(A,x)
    return sum(A)
end

function inner_function(R,x,K)
    A = zeros(R,5,K,2)
    #fill_A!(A,x)
    return sum(A)
end

function test_f2(x,K)
    return inner_function(eltype(x),x,K)
end


test_f1(α,100)
test_f2(α,100)

@time test_f1(α,100)
@time test_f2(α,100)
