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

p = loadpars_vec(p,"est_childsample_K4")


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

num_X = size(est_data[1].X,1)
x0 = zeros(8+2num_X)
x0[1:2] .= 0.5
x0[3:8] .= 0.8
pd = production_pars(x0,num_X)

nz = size(Z,1)
zy = reshape(Z*th / N,2nz)
zx = Z * X / N

# get initial variance
V = get_variance(th,X,Z,pd) / N

# y = X*β(δ) + η
# can we write a true likelihood conditional on z?
# X = Π * Z + ϵ #<- approximate as normal?
# p(y|Z) = ∑_x (x*)
@model function gmm_likelihood(zy,zx,V)
    num_Z,num_X = size(zx)
    δI ~ filldist(Uniform(0,3), 2) #[Uniform(0,3) for i in 1:2]
    δθ ~ filldist(Uniform(0.8,1.),2)
    g ~ filldist(Uniform(-10,10), 2, 2) #[Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ filldist(Turing.Flat(),num_X - 48,2) #[Flat() for i in 1:num_X]
    zxb = zx * get_β(;δI,g,δθ,β)
    zy ~ MvNormal(reshape(zxb,2num_Z), V)
end

chain = sample(gmm_likelihood(zy,zx,V),NUTS(),200)
# get first stage estimates:
est1 = (δI = mean(group(chain,:δI)).nt.mean,
    g = reshape(mean(group(chain,:g)).nt.mean,2,2),
    δθ = mean(group(chain,:δθ)).nt.mean, #<- fix this
    β = reshape(mean(group(chain,:β)).nt.mean,num_X,2)
)
V = get_variance(th,X,Z,est1) / N

chain = sample(gmm_likelihood(zy,zx,V),NUTS(),1000)

chain_data = chain.value.data[:,1:8,1]
names = [:Ib,:Ic,:Db,:Dc,:gNb,:gNc,:gFb,:gFc]
d = DataFrame(Dict((names[i],chain_data[:,i]) for i in 1:8))
CSV.write("../output/prod_ests_chain.csv",d)

# try something else here

@model function gmm_likelihood(zy,zx,V)
    num_Z,num_X = size(zx)
    δI ~ filldist(Uniform(0,3), 2) #[Uniform(0,3) for i in 1:2]
    δθ ~ filldist(Uniform(0.8,1.),2)
    g ~ filldist(Uniform(-2,2), 2, 2) #[Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ filldist(Turing.Flat(),num_X - 48,2) #[Flat() for i in 1:num_X]
    zxb = zx * get_β(;δI,g,δθ,β)
    zy ~ MvNormal(reshape(zxb,2num_Z), V)
end

chain_1 = sample(gmm_likelihood(zy,zx,V),NUTS(),200)
# get first stage estimates:
est2 = (δI = mean(group(chain_1,:δI)).nt.mean,
    g = reshape(mean(group(chain_1,:g)).nt.mean,2,2),
    δθ = mean(group(chain_1,:δθ)).nt.mean, #<- fix this
    β = reshape(mean(group(chain_1,:β)).nt.mean,num_X,2)
)
V = get_variance(th,X,Z,est2) / N

chain_2 = sample(gmm_likelihood(zy,zx,V),NUTS(),1000)

chain_data = chain_2.value.data[:,1:8,1]
names = [:Ib,:Ic,:Db,:Dc,:gNb,:gNc,:gFb,:gFc]
d = DataFrame(Dict((names[i],chain_data[:,i]) for i in 1:8))
CSV.write("../output/prod_ests_chain.csv",d)
