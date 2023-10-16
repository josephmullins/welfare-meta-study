using .Threads, Random
using DataFrames, CSV
using DataFramesMeta

struct likelihood_data
    case_idx::Int64 #<- this indicates which version of the model to access (based on location, treatment arm, age, age of youngest, and number of kids)
    t0::Int64 #<- this indicates the quarter the first observation relative to the first quarter in the model to map t in this data to t in the model.
    # wages and childcare
    T::Int64 #<- length of the panel (useful)
    chcare_valid::Vector{Bool}
    chcare::Vector{Float64}
    log_chcare::Vector{Float64}
    wage_valid::Vector{Bool}
    logW::Vector{Float64}

    # choices
    choice_missing::Vector{Bool}
    AFDC::Vector{Int64}
    EMP::Vector{Int64}
    FS::Vector{Int64}
    FC::Vector{Int64} #<- mostly not visible
 
    # education dummies
    less_hs::Int64
    hs::Int64
    some_coll::Int64
    coll::Int64

    # type demographics
    X_type::Vector{Float64}

    # indicator for whether participated last period in welfare
    kA::Int64

    # indicator for whether this obseration is in the leave-in or holdout sample:
    use::Bool
end

include("tools/BaumWelch.jl")
include("estimation/initialize.jl")
include("estimation/likelihood.jl")
include("estimation/type_probabilities.jl")
include("estimation/expectation.jl")
include("estimation/maximization.jl")
#include("estimation/EM.jl")
#include("estimation/statistics.jl")
#include("estimation/bootstrap.jl")