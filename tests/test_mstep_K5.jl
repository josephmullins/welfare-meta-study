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

# block = [:λ₀,:λ₁,:δ,:λR] #,:μₒ,:σₒ] #<- this is still slow, so we know it's not just about lambda_u
# ft = [1,1,3,1] #,1,2]
# p = mstep_major_block(p,block,ft,EM,MD,n_idx,40)

block = [:μₒ,:σₒ]
ft = [1,2]
p = mstep_major_block(p,block,ft,EM,MD,n_idx,3)

function update_transitions(p)
    πₒ = zeros(eltype(p.μₒ),p.Kη-1,p.Kη)
    for k in axes(πₒ,2), k2 in axes(πₒ,1)
        πₒ[k2,k] = p_offer(k2+1,k,p)
    end
    return (;p...,πₒ)
end

p = mstep_major_block(p,block,ft,EM,MD,n_idx,3)

# block = [:λ₀,:λ₁,:λᵤ] 
# ft = [1,1,1]
# p = mstep_major_block(p,block,ft,EM,MD,n_idx,40)
