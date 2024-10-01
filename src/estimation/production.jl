# this function returns the reduced form parameters β given structural parameters
function get_β(;δI,g,δθ,β)
    β_ = zeros(eltype(δI),3*16,2)
    for t in 1:16
        for j in 1:2
            d = δθ[j] ^ (t-1)
            β_[(t-1)*3+1,j] = d*δI[j]
            β_[(t-1)*3+2,j] = d*g[j,1]
            β_[(t-1)*3+3,j] = d*g[j,2]
        end
    end
    return [β_;β]
end

function get_β_single(;δI,g₁,g₂,δθ,β)
    β_ = zeros(eltype(δI),3*16)
    for t in 1:16
        d = δθ ^ (t-1)
        β_[(t-1)*3+1] = d*δI
        β_[(t-1)*3+2] = d*g₁
        β_[(t-1)*3+3] = d*g₂
    end
    return [β_;β]
end

# this function returns the matrix X such that: 𝔼[θ|X] = X * get_β()
function get_X(est_data)
    N = length(est_data)
    X = zeros(N,3*16)
    for n in eachindex(est_data)
        T = est_data[n].T
        for t in 1:T
            X[n,((t-1)*3+1):t*3] .= [est_data[n].logY[T+1-t],est_data[n].H[T+1-t],est_data[n].F[T+1-t]]
        end
    end
    Xc = hcat((d.X for d in est_data)...)'
    return [X Xc]
end


# this function returns the variance of the moments given parameters and given the data arranged appropriately
# - Y is assumed to be N x 2
# - X is N x num_X
# - Z is num_Z x Z

function get_variance(Y,X,Z,pars)
    e = Y - X*get_β(;pars.δI,pars.δθ,pars.g,pars.β)
    num_Z,N = size(Z)
    G = zeros(N,2num_Z)
    for n in 1:N
        @views G[n,:] .= kron(e[n,:],Z[:,n])
    end
    return cov(G)
end

# get production parameters given nunmber of X variables in intercept
function production_pars(x,num_X)
    return (δI = x[1:2],
    δθ = x[3:4],
    g = reshape(x[5:8],2,2),
    β = reshape(x[9:(8+num_X*2)],num_X,2))
end

# get production parameters in one dimension given number of types
function prod_pars(x,K)
    return (δI = x[1], δθ = x[2], g₁ = x[3:(2+K)], g₂ = x[(3+K):(2+2K)])
end


function get_controls(p,md::model_data,em::EM_data,data::likelihood_data)
    # control for: location, type, in MFIP control for application status and county.
    # location * type
    # (MFIP*app_status==2, MFIP*app_status==3) * county==anoka,
    # control for age?
    sz = (2,p.Kη,md.Kω,p.Kτ)
    k_inv = CartesianIndices(sz)
    K = prod(sz)
    s_inv = CartesianIndices((9,K))
    kτ_assigned = k_assign(em,p.Kτ,s_inv,k_inv)
    X = Float64[]
    for s in ("FTP","CTJF","MFIP")
        for kτ in 1:p.Kτ
            push!(X,md.source==s && kτ_assigned==kτ)
        end
    end
    if md.source=="MFIP"
        push!(X,data.X_type[5:6]...)
        push!(X,data.X_type[7:8]...)
        push!(X,kron(data.X_type[5:6],data.X_type[7:8])...)
    else
        push!(X,fill(0.,8)...)
    end
    return X
end
function get_instruments(p,md::model_data,em::EM_data,data::likelihood_data)
    X = get_controls(p,md,em,data)
    Z = X[1:(3*p.Kτ)] .* (md.arm==1)
    push!(Z,(X[(2p.Kτ+1):(3p.Kτ)] .* (md.arm==2))...) #<- extra dummy for MFIP
    return [X;Z]
end
function get_controls_simple(p,md::model_data,em::EM_data,data::likelihood_data)
    # control for type only, see how it all looks
    sz = (2,p.Kη,md.Kω,p.Kτ)
    k_inv = CartesianIndices(sz)
    K = prod(sz)
    s_inv = CartesianIndices((9,K))
    kτ_assigned = k_assign(em,p.Kτ,s_inv,k_inv)
    X = zeros(p.Kτ) #<- what about age? number of kids?
    X[kτ_assigned] = 1.
    return X
end

function k_assign(em::EM_data,Kτ,s_inv,k_inv)
    pτ = zeros(Kτ)
    for s in axes(em.q_s,1)
        j,k = Tuple(s_inv[s])
        _,_,_,kτ = Tuple(k_inv[k])
        pτ[kτ] += em.q_s[s,1]
    end
    return argmax(pτ)
end

function production_data(p,em::EM_data,md::model_data,data::likelihood_data)
    X = get_controls(p,md,em,data)
    #X = get_controls_simple(p,md,em,data) #<- do this instead? what happens?
    Z = get_instruments(p,md,em,data)
    if md.source=="CTJF"
        T = 4*4
    else
        T = 3*4
    end
    logY = zeros(T)
    H = zeros(T)
    F = zeros(T)

    sz = (2,p.Kη,md.Kω,p.Kτ)
    k_inv = CartesianIndices(sz)
    K = prod(sz)
    s_inv = CartesianIndices((9,K))

    t_care = findfirst(data.pay_care_valid)

    # adding some alternative instruments to see what happens
    # d = 0.95
    # Z = zeros(4)
    for t in 1:T

        logY[t] = em_mean(em.q_s,t,s->log_full(s,t,s_inv,k_inv,p,md))
        if !isnothing(t_care)
            F[t] = data.pay_care[t_care]
        else
            F[t] = em_mean(em.q_s,t,s->paid_care(s,s_inv))
        end
        if !data.choice_missing[t]
            H[t] = data.EMP[t]
        else
            H[t] = em_mean(em.q_s,t,s->unpaid_care(s,s_inv))
        end
            
        #H[t] = em_mean(em.q_s,t,s->unpaid_care(s,s_inv))
        #F[t] = em_mean(em.q_s,t,s->paid_care(s,s_inv))
        # decay = d^(T-t)
        # Z[1] += decay * logY[t]
        # Z[2] += decay * H[t]
        # Z[3] += decay * F[t]
        # Z[4] += (T-t) * decay / d * logY[t]
    end
    # Z = [X;Z]
    # M = [scores.BPIE,scores.BPIN,scores.PBS,scores.ENGAGE,scores.REPEAT
    #     ,scores.SUSPEND,scores.ACHIEVE,scores.TCH_AVG] #<- don't need this, store it elsewhere.
    return (ageyng = md.ageyng, X = X, Z = Z, logY = logY, H = H, F = F, T = T)
end

function predict_skills(pars,data) #<- data is a named tuple?
    thB = 0.
    thC = 0.
    for t in 1:data.T
        thB = pars.δI[1]*data.logY[t] + pars.g[1,1]*data.H[t] + pars.g[1,2]*data.F[t] + pars.δθ[1]*thB
        thC = pars.δI[2]*data.logY[t] + pars.g[2,1]*data.H[t] + pars.g[2,2]*data.F[t] + pars.δθ[2]*thC
    end
    # now add controls:
    @views thB += dot(data.X,pars.β[:,1])
    @views thC += dot(data.X,pars.β[:,2])
    return thB,thC
end

function production_data(panel::DataFrame,p)
    (;vj,V,logP) = get_model(p)
    logπτ = zeros(p.Kτ)
    est_data = []
    for d in groupby(panel,[:source,:id])
        md = model_data(d)
        dat = likelihood_data(d)
        em = get_EM_data(p,md,dat)
        solve!(logP,V,vj,p,md)
        update!(logπτ,em,logP,p,md,dat)
        forward_back!(em)
        ed = production_data(p,em,md,dat)
        push!(est_data,ed)
    end
    return est_data
end

function gfunc!(g,pars,data,factor_scores)
    thB,thC = predict_skills(pars,data)
    pos = 1
    rB = factor_scores[1] - thB
    rC = factor_scores[2] - thC
    for iz in eachindex(data.Z)
        g[pos] += rB * data.Z[iz]
        pos += 1
    end
    for iz in eachindex(data.Z)
        g[pos] += rC * data.Z[iz]
        pos += 1
    end
end


function moment_variance(pars,data,factor_scores)
    nmom = 2length(data[1].Z)
    N = length(data)
    G = zeros(eltype(pars.δI),N,nmom)
    for n in eachindex(data)
        @views gfunc!(G[n,:],pars,data[n],factor_scores[n,:])
    end
    return cov(G)
end