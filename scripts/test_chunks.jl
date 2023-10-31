include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K4")

x_est = pars_inv_full(p)

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

function chunked_likelihood(p,nchunks,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx)
    chunks = Iterators.partition(MD,length(MD) ÷ nchunks)
    tasks = map(chunks) do chunk
        Threads.@spawn log_likelihood_chunk(p,eltype(p.αH),chunk,EM,data,n_idx)
    end
    ll = fetch.(tasks)
    return sum(ll)
end

chunked_likelihood(p,2,EM,MD,data,n_idx)
chunked_likelihood(p,4,EM,MD,data,n_idx)
chunked_likelihood(p,8,EM,MD,data,n_idx)

