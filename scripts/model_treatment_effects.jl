# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/counterfactuals.jl")

Kτ = 5 #
Kη = 4
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"current_est")
p = loadpars_vec(p,"est_childsample_K5")
#p = loadpars_vec(p,"est_noSIPP_K3")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    #vcat(sipp)
end

MD,EM,data,n_idx = estimation_setup(panel);

function prod_pars(x,K)
    return (δI = x[1], δθ = x[2], g₁ = x[3:(2+K)], g₂ = x[(3+K):(2+2K)])
end

chain = CSV.read("output/production_ests_hetero.csv",DataFrame)
chain_B = chain[1:10000,:]
chain_C = chain[10001:20000,:]

pB = prod_pars([mean(x) for x in eachcol(chain_B)[1:12]],Kτ)
pC = prod_pars([mean(x) for x in eachcol(chain_C)[1:12]],Kτ)


MD1 = copy(MD)
MD2 = copy(MD)
MD3 = copy(MD)
MD4 = copy(MD)
for m in eachindex(MD)
    MD1[m] = full_treatment(MD[m])
    MD2[m] = incentives_only(MD[m])
    MD3[m] = work_requirements_only(MD[m])
    MD4[m] = time_limits_only(MD[m])
    MD[m] = control(MD[m])
end

function decomposition_counterfactual(p,pB,pC,MD,MD1,MD2,MD3,MD4,data,n_idx)
    d1 = counterfactual(p,pB,pC,MD,MD1,data,n_idx)
    d1[!,:case] .= "Treatment"

    d2 = counterfactual(p,pB,pC,MD,MD2,data,n_idx)
    d2[!,:case] .= "Incentives Only"

    d3 = counterfactual(p,pB,pC,MD,MD3,data,n_idx)
    d3[!,:case] .= "Work Requirements Only"

    d4 = counterfactual(p,pB,pC,MD,MD4,data,n_idx)
    d4[!,:case] .= "Time Limits Only"
    return [d1;d2;d3;d4]
end

d = decomposition_counterfactual(p,pB,pC,MD,MD1,MD2,MD3,MD4,data,n_idx)

break


# TODO: add calculation of child skills here. do we account for heterogeneity or do we not?
n_boot = 3
x_est = pars_inv_full(p) #<- here's an issue. The probabilities are not full rank. Surely won't invert?
#V = readdlm("output/var_est_K5")
V = 0.001 * I(length(x_est)) # we shuld also be polishing estimates maybe
p_bootstrap = rand(MvNormal(x_est,V),n_boot)
cols = names(d)

# put this in a function?
Db = d[1:0,:]
for b in axes(p_bootstrap,2)
    pb = pars_full(p_bootstrap[:,b],p)
    db = decomposition_counterfactual(p,MD,MD1,MD2,MD3,MD4,data,n_idx)
    db[!,:boot] .= b
    Db = [Db; db]
end
Db = combine(groupby(Db,[:source,:variable,:year,:Q,:case]),:value => Statistics.std => :sd, :value => x->quantile(x,0.025) => :q25, :value => x->quantile(x,0.975), :q75)


#d = calculate_treatment_effects(p,EM,MD,data,n_idx)

# QUESTION: DO WE WANT TO MAKE OUTCOMES A FUNCTION OF TYPE. IT WILL EFFECT WHAT WE DO.

#write everything to file here.
