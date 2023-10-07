include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 3 #
Kη = 6
p = pars(Kτ,Kη)
p.βw[1:Kτ] .= LinRange(6,7.5,Kτ)
p.βw[Kτ+1] = -0.2
p.βw[Kτ+2] = 0.003
p.βf[1:Kτ] .= 3.
p.σ[:] .= [2.,2.,2.]
p.ση = 2.
p.σ_W = 2.
p.σ_PF = 2.

x0 = update_inv(p)
update!(x0,p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again :-)
sipp = @subset panel :source.=="SIPP"

panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp)
end

# @chain panel begin
#     @select :id :source
#     unique()
#     groupby(:source)
#     @combine(:N = length(:id))
# end

M,∂M,MD,EM,data,n_idx = estimation_setup(panel);

G = zeros(length(x0),nthreads())
LL = zeros(nthreads())

Random.seed!(2020)
shuffle!(MD)
solvetime = zeros(length(MD))
forward_back_threaded!(p,EM,M,MD,data,n_idx)


function log_likelihood_bench!(x,G::Matrix{Float64},solvetime::Vector{Float64},M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},p::pars,data::Vector{likelihood_data},n_idx::Vector{Vector{Int64}})
    fill!(G,0.)
    fill!(LL,0.)
    update!(x,p)
    update!(p,M,∂M,(1,8,9))
    @threads :static for i in eachindex(MD) # in MD
        t0 = time()
        md = MD[i]
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))

        # ------- Step 1: get the log-likelihood of choices --------
        # to save on memory allocations we to do this for each t over all observations that fit this case, then we update and discard data from t.
 
        @views ll = log_likelihood_choices(p,G[:,threadid()],M[m_idx,threadid()],∂M[m_idx,threadid()],EM,data,md,n_idx)

        # ------- Step 2: get the log-likelihood of prices and transitions ----- #
        for n ∈ n_idx[md.case_idx]
            if data[n].use
                @views ll += log_likelihood(G[:,threadid()],EM[n],md,p,M[m_idx,threadid()],∂M[m_idx,threadid()],data[n])
            end
        end
        #LL[threadid()] += ll
        solvetime[i] = time() - t0
    end
    for it in axes(G,2)
        @views apply_chain_rule!(G[:,it],p,∂M[1,1].σ_idx,∂M[1,1].β_idx)
    end
    #return sum(LL)
end

#ll = log_likelihood_threaded(x0,G,LL,M,∂M,EM,MD,p,data,n_idx)
#println(" -- Evaluation time with $(nthreads()) cores:")
log_likelihood_bench!(x0,G,solvetime,M,∂M,EM,MD,p,data,n_idx)

chunks = Iterators.partition(solvetime, length(solvetime) ÷ nthreads())
a = [sum(c) for c in chunks]

println(" -- Eval time should be close to: $(maximum(a))")
println(" -- Maximum solve time is: $(maximum(solvetime))")

chunks = Iterators.partition(solvetime, length(solvetime) ÷ 20)
a = [sum(c) for c in chunks]

# compare the evaluation time to the one implied by solvetime.

# ll = log_likelihood_threaded(x0,G,LL,M,∂M,EM,MD,p,data,n_idx)
# println(" -- Evaluation time with $(nthreads()) cores:")
# @time log_likelihood_threaded(x0,G,LL,M,∂M,EM,MD,p,data,n_idx)

