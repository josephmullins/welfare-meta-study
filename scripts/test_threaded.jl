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
MD = MD[1:400]

forward_back_threaded!(p,EM,M,MD,data,n_idx)

ll = log_likelihood_threaded(x0,G,LL,M,∂M,EM,MD,p,data,n_idx)
println(" -- Evaluation time with $(nthreads()) cores:")
@time log_likelihood_threaded(x0,G,LL,M,∂M,EM,MD,p,data,n_idx)
