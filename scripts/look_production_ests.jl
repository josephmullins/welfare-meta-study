include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"current_est")
p = loadpars_vec(p,"est_childsample_K5")

chain0 = CSV.read("output/production_ests.csv",DataFrame)
mle = @subset(chain0,:version.=="mle",:skill.=="Behavioral")
e1 = mean.(eachcol(mle[:,1:4]))
iv = @subset(chain0,:version.=="iv",:skill.=="Behavioral")
e2 = std.(eachcol(iv[:,1:4]))

q25 = quantile.(eachcol(mle[:,1:4]),0.025)
q75 = quantile.(eachcol(mle[:,1:4]),0.975)

chain = CSV.read("output/production_ests_hetero.csv",DataFrame)
chain_B = chain[1:10000,:]
chain_C = chain[10001:20000,:]

pB = prod_pars([mean(x) for x in eachcol(chain_B)[1:12]],Kτ)
pC = prod_pars([mean(x) for x in eachcol(chain_C)[1:12]],Kτ)
