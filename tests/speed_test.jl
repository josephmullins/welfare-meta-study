include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 4 #?
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

x0 = pars_inv(p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp)
end

MD,EM,data,n_idx = estimation_setup(panel);


Random.seed!(2020)
shuffle!(MD)

forward_back_threaded!(p,EM,MD,data,n_idx)



# function solve_model_chunk(x,p,MD,EM::Vector{EM_data},data::Vector{likelihood_data},n_idx)
#     # setup data:
#     T = 18*4
#     K = 2 * p.Kη * p.Kτ * 9 #<- maximal state size
#     R = eltype(x)
#     logP = zeros(R,9,K,T)
#     V = zeros(R,K,2)
#     vj = zeros(R,9)
#     ll = 0.
#     for md in MD
#         solve!(logP,V,vj,p,md)
#     end
#     return nothing
# end

# function solve_threaded(x,p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
#     chunks = Iterators.partition(MD,length(MD) ÷ Threads.nthreads())
#     tasks = map(chunks) do chunk
#         Threads.@spawn solve_model_chunk(x,p,chunk,EM,data,n_idx)
#     end
#     ll = fetch.(tasks)
#     return nothing
# end

# solve_threaded(x0,p,EM,MD,data,n_idx)
# @time solve_threaded(x0,p,EM,MD,data,n_idx)


using ForwardDiff
function get_G(x,p,block,ft,EM,MD,data,n_idx)
    ForwardDiff.gradient(y->log_likelihood_threaded(y,p,block,ft,EM,MD,data,n_idx),x)
end
block = [:αH,:αA,:βw]
ft = [1,1,1]
x0 = pars_inv(p,block,ft)

log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
get_G(x0,p,block,ft,EM,MD,data,n_idx)

@time log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
@time get_G(x0,p,block,ft,EM,MD,data,n_idx)

block = [:μₒ,:σₒ]
ft = [1,2]
x0 = pars_inv(p,block,ft)

l0 = log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
g0 = get_G(x0,p,block,ft,EM,MD,data,n_idx)


@time log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
@time get_G(x0,p,block,ft,EM,MD,data,n_idx)

# p = mstep_major_block(p,block,ft,EM,MD,n_idx,2)

# block = [:λ₀,:λ₁,:λᵤ] 
# ft = [1,1,1]
# x0 = pars_inv(p,block,ft)
# p = mstep_major_block(p,block,ft,EM,MD,n_idx,2)


function update_transitions(p)
    πₒ = zeros(eltype(p.μₒ),p.Kη-1,p.Kη)
    for k in axes(πₒ,2), k2 in axes(πₒ,1)
        πₒ[k2,k] = p_offer(k2+1,k,p)
    end
    return (;p...,πₒ)
end

l1 = log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
g1 = get_G(x0,p,block,ft,EM,MD,data,n_idx)


@time log_likelihood_threaded(x0,p,block,ft,EM,MD,data,n_idx)
@time get_G(x0,p,block,ft,EM,MD,data,n_idx)

