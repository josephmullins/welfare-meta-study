using DataFrames, DataFramesMeta, CSV, Turing, LinearAlgebra
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/production.jl")
include("../src/estimation/FactorAnalysis.jl")


Kτ = 2 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K2")


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
Z = hcat((d.Z for d in est_data)...) #<- get the instruments
Z = Z'

# project X onto Z
Xhat = Z * inv(Z' * Z) * Z' * X

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

chain_iv = sample(model_likelihood(th[:,1],Xhat),NUTS(),2000)

