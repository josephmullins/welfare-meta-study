using DataFrames, DataFramesMeta, CSV, Turing, LinearAlgebra
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/production.jl")
include("../src/estimation/FactorAnalysis.jl")


Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")


scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
scores = @orderby(scores,:source,:id)

cols = (:BPIE,:BPIN,:PBS,:ENGAGE,:REPEAT,:SUSPEND,:ACHIEVE,:TCH_AVG)
columns = (MFIP = cols[1:end-1],FTP = cols[1:end-1],CTJF = cols)
# convert factor scores to an array for estimation
M = prep_scores(scores,cols)

# run the factor analysis along with the bootstrap
est,sd = factor_analysis(scores,columns)
Λ,Σ,D = measurement_system(est) #<- estimated system
Λ_se,_,Dse = measurement_system(sd) #<- standard errors
## write results to table
write_measurement_table!(Λ,Λ_se,D,Dse)

# Calculate factor scores using the measurement system
## var = λ^2 + D, so one sd is sqrt of this.
th = factor_scores(scores,M,Λ,Σ,D) 

panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
# we still need a case ID when we make the model data objects, but we don't need to
#  distinguish across cases when the likelihood isn't involved, hence...
panel[!,:case_idx] .= 0 

# merge in the data on scores
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
end

est_data = production_data(panel,p)
N = length(est_data)
X = get_X(est_data) #<- get the X variables
Z = hcat((d.Z for d in est_data)...) #<- get the instruments
Z = Z'

# project X onto Z for the IV model
Xhat = Z * inv(Z' * Z) * Z' * X

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
# set the length of each chain
length_chain = 1_000

chain_B_mle = sample(model_likelihood(th[:,1],X),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_B_mle = DataFrame(chain_B_mle)[:,[:δI,:δθ,:g₁,:g₂]]
data_B_mle[!,:version] .= "mle"
data_B_mle[!,:skill] .= "Behavioral"

chain_C_mle = sample(model_likelihood(th[:,2],X),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_C_mle = DataFrame(chain_C_mle)[:,[:δI,:δθ,:g₁,:g₂]]
data_C_mle[!,:version] .= "mle"
data_C_mle[!,:skill] .= "Cognitive"

chain_B_iv = sample(model_likelihood(th[:,1],Xhat),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_B_iv = DataFrame(chain_B_iv)[:,[:δI,:δθ,:g₁,:g₂]]
data_B_iv[!,:version] .= "iv"
data_B_iv[!,:skill] .= "Behavioral"


chain_C_iv = sample(model_likelihood(th[:,2],Xhat),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_C_iv = DataFrame(chain_C_iv)[:,[:δI,:δθ,:g₁,:g₂]]
data_C_iv[!,:version] .= "iv"
data_C_iv[!,:skill] .= "Cognitive"

break
# save these results:
CSV.write("output/production_ests.csv",[data_B_mle;data_C_mle;data_B_iv;data_C_iv])

function get_β_hetero(;δI,g₁,g₂,δθ,β,Kτ)
    num_inputs = 1 + 2Kτ
    β_ = zeros(eltype(δI),num_inputs*16)
    for t in 1:16
        d = δθ ^ (t-1)
        β_[(t-1)*num_inputs+1] = d*δI
        β_[((t-1)*num_inputs+2):((t-1)*num_inputs+1+Kτ)] .= d*g₁
        β_[((t-1)*num_inputs+2+Kτ):((t-1)*num_inputs+num_inputs)] .= d*g₂
    end
    return [β_;β]
end

function get_X_hetero(est_data,Kτ)
    N = length(est_data)
    X = zeros(N,(2Kτ+1)*16)
    for n in eachindex(est_data)
        # extract a dummy for xk
        xk = [sum(est_data[n].X[k:Kτ:(3Kτ)]) for k in 1:Kτ]

        T = est_data[n].T
        num_inputs = 2Kτ + 1
        for t in 1:T
            X[n,((t-1)*num_inputs+1):t*num_inputs] .= [est_data[n].logY[T+1-t]; xk .* est_data[n].H[T+1-t]; xk .* est_data[n].F[T+1-t]]
        end
    end
    Xc = hcat((d.X for d in est_data)...)'
    return [X Xc[:,1:12]]
end

X = get_X_hetero(est_data,Kτ) 


@model function model_hetero(th,X,Kτ)
    num_inputs = 2Kτ + 1
    num_X = size(X,2)
    δI ~ Uniform(0,3) #[Uniform(0,3) for i in 1:2]
    δθ ~ Uniform(0.,1.)
    g₁ ~ filldist(Uniform(-2,2),Kτ)
    g₂ ~ filldist(Uniform(-2,2),Kτ) #[Uniform(-2,2) for i in 1:2, j in 1:2]
    β ~ filldist(Turing.Flat(),num_X - num_inputs*16) #[Flat() for i in 1:num_X]
    m = X * get_β_hetero(;δI,g₁,g₂,δθ,β,Kτ)
    σ ~ FlatPos(0) #?
    th ~ MvNormal(m, I*σ)
end

vnames = [["g₁[" * string(k) * "]" for k in 1:Kτ];["g₂[" * string(k) * "]" for k in 1:Kτ]]
vnames = [["δI","δθ"];vnames]

chain_B_hetero = sample(model_hetero(th[:,1],X,Kτ),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_B_hetero = DataFrame(chain_B_hetero)[:,vnames]
data_B_hetero[!,:skill] .= "Behavioral"

chain_C_hetero = sample(model_hetero(th[:,2],X,Kτ),NUTS(),MCMCThreads(),length_chain,Threads.nthreads())
data_C_hetero = DataFrame(chain_C_hetero)[:,vnames]
data_C_hetero[!,:skill] .= "Cognitive"

CSV.write("output/production_ests_hetero.csv",[data_B_hetero;data_C_hetero])

