using DataFrames, DataFramesMeta, CSV, Turing, LinearAlgebra
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/production.jl")
include("../src/estimation/FactorAnalysis.jl")


Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"est_childsample_K5")
p = loadpars_vec(p,"current_est")


scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
scores = @orderby(scores,:source,:id)

cols = (:BPIE,:BPIN,:PBS,:ENGAGE,:REPEAT,:SUSPEND,:ACHIEVE,:TCH_AVG)
columns = (MFIP = cols[1:end-1],FTP = cols[1:end-1],CTJF = cols)
# convert factor scores to an array for estimation
M = prep_scores(scores,cols)
# calculate the optimal weighting matrix
wght = get_weighting_matrix(scores,columns)
Λ,Σ,D = estimate_system(scores,wght,columns)
# calculate the factor scores:

th = factor_scores(scores,M,Λ,Σ,D)

#th ./= std(th,dims=1) #<- normalize the scale of the factors (? no need for this?)

# est,sd = factor_analysis(scores,columns)
# est_y,sd_y = factor_analysis(@subset(scores,:AGEKID.<=8),columns)
# est_o,sd_o = factor_analysis(@subset(scores,:AGEKID.>8),columns)


panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data
panel[!,:case_idx] .= 0 #<- don't need this here

panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
end

est_data = production_data(panel,p)
N = length(est_data)
X = get_X(est_data) #<- get the X variables
#Z = hcat((d.Z for d in est_data)...) #<- get the instruments

num_X = size(est_data[1].X,1)

# y = X*β(δ) + η
# can we write a true likelihood conditional on z?
# X = Π * Z + ϵ #<- approximate as normal?
# p(y|Z) = ∑_x (x*)
# should we be including a control for age in here?
# also want to eventually include some controls for η

@model function model_likelihood(th,X)
    num_X = size(X,2)
    δI ~ Uniform(0,3) #[Uniform(0,3) for i in 1:2]
    δθ ~ Uniform(0.,1.)
    g₁ ~ Uniform(-2,2)
    g₂ ~ Uniform(-2,2) #[Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ filldist(Turing.Flat(),num_X - 48) #[Flat() for i in 1:num_X]
    m = X * get_β_single(;δI,g₁,g₂,δθ,β)
    σ ~ FlatPos(0) #?
    th ~ MvNormal(m, I*σ)
end

chain_1 = sample(model_likelihood(th[:,1],X),NUTS(),1000)

X2 = X[:,1:(3*16+15)]

# TYPO HERE!! so the results are just different
chain_2 = sample(model_likelihood(th[:,1],X2),NUTS(),1000)

# now work on the case with types:
function get_β_hetero(;δI,g₁,g₂,δθ,β)
    β_ = zeros(eltype(δI),5*16)
    for t in 1:16
        d = δθ ^ (t-1)
        β_[(t-1)*5+1] = d*δI
        β_[((t-1)*5+2):((t-1)*5+3)] .= d*g₁
        β_[((t-1)*5+4):((t-1)*5+5)] .= d*g₂
    end
    return [β_;β]
end

function get_X_spec2(est_data)
    N = length(est_data)
    X = zeros(N,5*16)
    for n in eachindex(est_data)
        # extract a dummy for xk
        xk = [sum(est_data[n].X[k:5:15]) for k in 1:5]
        # then simplify the dummy because this is too slow.
        xk = [sum(xk[1:3]),sum(xk[4:5])]

        T = est_data[n].T
        for t in 1:T
            #X[n,((t-1)*11+1):t*11] .= [est_data[n].logY[T+1-t]; xk .* est_data[n].H[T+1-t]; xk .* est_data[n].F[T+1-t]]
            X[n,((t-1)*5+1):t*5] .= [est_data[n].logY[T+1-t]; xk .* est_data[n].H[T+1-t]; xk .* est_data[n].F[T+1-t]]
        end
    end
    Xc = hcat((d.X for d in est_data)...)'
    return [X Xc[:,1:15]]
end

X = get_X_spec2(est_data) #<- get the X variables


@model function model_hetero(th,X)
    num_X = size(X,2)
    δI ~ Uniform(0,3) #[Uniform(0,3) for i in 1:2]
    δθ ~ Uniform(0.,1.)
    g₁ ~ filldist(Uniform(-2,2),2)
    g₂ ~ filldist(Uniform(-2,2),2) #[Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ filldist(Turing.Flat(),num_X - 5*16) #[Flat() for i in 1:num_X]
    m = X * get_β_hetero(;δI,g₁,g₂,δθ,β)
    σ ~ FlatPos(0) #?
    th ~ MvNormal(m, I*σ)
end

chain_3 = sample(model_hetero(th[:,1],X),NUTS(),2000)
