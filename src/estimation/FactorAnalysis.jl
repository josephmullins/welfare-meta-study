using DataFrames, DataFramesMeta, CSV, LinearAlgebra, Random, Optim

function prep_scores(data::DataFrame,columns)
    # demean:
    for v in columns
        m = mean(skipmissing(data[!,v]))
        data[!,v] .-= m
    end
    return Matrix(coalesce.(data[:,[columns...]],0.))
end

function get_covariances(data::DataFrame,columns)
    nmoms = 2 * (7 * 8) ÷ 2 + 8 * 9 ÷ 2 
    v = zeros(nmoms)
    pos = 1
    for site in keys(columns)
        cols = getfield(columns,site)
        d = @subset(data,:source.==string(site))
        for i in eachindex(cols)
            v[pos] = var(skipmissing(d[!,cols[i]]))
            pos += 1
            for j in 1:(i-1)
                v[pos] = cov(Matrix(dropmissing(d[!,[cols[i],cols[j]]])))[2,1]
                pos += 1
            end
        end
    end
    return v
end

function measurement_system(x)
    Λ = zeros(eltype(x),8,2)
    Σ = zeros(eltype(x),2,2,3) #<- do we have to do this? I don't think so?
    # update:
    # site ordering: MFIP,FTP,CTJF

    # assumed ordering: BPIE,BPN,PBS,ENGAGE,REPEAT,SUSPEND,ACHIEVE,TCH_AVG (potential issues with age in the last case!)
    # normalizations:
    @views Λ[1:6,1] = x[1:6]
    @views Λ[4:8,2] = x[7:11]

    # set covariance in Minnesota
    Σ[1,1,1] = 1. 
    Σ[2,2,1] = 1.
    c = logit(x[12]) #<- this sign restriction is also a normalization with the sign of the factor loadings
    Σ[1,2,1] = c 
    Σ[2,1,1] = c

    dσ = zeros(eltype(x),2,2)
    for l in 1:2
        dσ[[1,2,4]] .= x[(12+(l-1)*3+1):(12+(l*3))]
        @views Σ[:,:,l+1] = dσ * dσ'
    end
    D = diagm(x[19:26].^2)
    return Λ,Σ,D
end

# other arguments here are the weighting matrix and the data
function FA_objective(x,v_data,wght)
    Λ,Σ,D = measurement_system(x)
    N = (7,7,8)
    nmoms = 2 * (7 * 8) ÷ 2 + 8 * 9 ÷ 2 
    v_model = zeros(eltype(x),nmoms)
    pos = 1
    for l in eachindex(N)
        V = Λ * Σ[:,:,l] * Λ' .+ D
        # then flatten into a thing
        for i in 1:N[l]
            v_model[pos] = V[i,i]
            pos += 1
            for j in 1:(i-1)
                v_model[pos] = V[i,j]
                pos += 1
            end
        end
    end
    r = v_model .- v_data
    return r' * wght * r
end

function boot_moments(data,columns,bootseed,bootsamples)
    nmoms = 2 * (7 * 8) ÷ 2 + 8 * 9 ÷ 2 
    Vb = zeros(nmoms,bootsamples)
    Random.seed!(bootseed)
    N = nrow(data)
    for b in 1:bootsamples
        Ib = rand(1:N,N)
        Vb[:,b] = get_covariances(data[Ib,:],columns)
    end
    return Vb
end

# -- Now should we do the bootstrap
function factor_analysis(data::DataFrame,columns;bootseed = 1823,bootsamples = 50)
    # get the estimates:
    v_data = get_covariances(data,columns)
    wght = get_weighting_matrix(data,columns,bootseed,bootsamples)

    x0 = ones(26)
    #x0[1:3] .= -1. 
    res = Optim.optimize(x->FA_objective(x,v_data,wght),x0,LBFGS(),autodiff=:forward)
    est = res.minimizer
    Xb = zeros(26,bootsamples)
    Vb = boot_moments(data,columns,bootseed,bootsamples)
    for b in 1:bootsamples
        println(b)
        res = Optim.optimize(x->FA_objective(x,Vb[:,b],wght),est,LBFGS(),autodiff=:forward)
        Xb[:,b] = res.minimizer
    end
    sd = std(Xb,dims=2)[:]
    return est,sd
end

function get_weighting_matrix(data::DataFrame,columns,bootseed = 1823, bootsamples = 50)
    Vb = boot_moments(data,columns,bootseed,bootsamples)
    return diagm(1 ./ var(Vb,dims=2)[:])
end

function estimate_system(data::DataFrame,wght::Matrix{Float64},columns,x0 = ones(26))
    v_data = get_covariances(data,columns)
    res = Optim.optimize(x->FA_objective(x,v_data,wght),x0,LBFGS(),autodiff=:forward)
    return measurement_system(res.minimizer)
end

function factor_scores(data::DataFrame,M,Λ,Σ,D)
    D = D[1:7,1:7]
    Λ = Λ[1:7,:]
    th = zeros(nrow(data),2)
    scs = ("CTJF","FTP","MFIP")
    for s in 1:3
        Is = data.source.==scs[s]
        @views th[Is,:] .= M[Is,1:7] * inv(Λ * Σ[:,:,s] * Λ' + D) * Λ * Σ[:,:,s]
    end
    return th
end

function write_measurement_table!(Λ,Λ_se,D,Dse)
    form(x) = x!=0 ? @sprintf("%0.2f",x) : "-"
    formse(x) = x!=0 ? string("(",@sprintf("%0.2f",x),")") : ""

    file = open("output/tables/factor_analysis.tex", "w")
    write(file,"Measure & \$\\lambda^{m}_{B}\$ & \$\\lambda^{m}_{C}\$ & \$\\sigma^2_{m}\$ \\\\ \\cmidrule(r){1-4} \n")
    measures = ("BPI-Externalizing","BPI-Internalizing","Positive Behavior Scale","School Engagement","Ever Repeat Grade","Ever Suspended","School Achievement - Parent","School Achievement - Teacher")
    for m in eachindex(measures)
        write(file,measures[m]," &",form(Λ[m,1])," & ",form(Λ[m,2])," & ",form(D[m,m]),"\\\\ \n")
        write(file," &",formse(Λ_se[m,1])," & ",formse(Λ_se[m,2])," & ",formse(2sqrt(Dse[m,m])*D[m,m]),"\\\\ \n")
    end
    close(file)
end