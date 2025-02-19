# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/production.jl")
include("../src/counterfactuals.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
end

MD,EM,data,n_idx = estimation_setup(panel);

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

n_boot = 100
x_est = pars_inv_full(p) 
V = readdlm("output/var_est_K5")
V = Hermitian(V)
p_bootstrap = rand(MvNormal(x_est,V),n_boot)
cols = names(d)

Random.seed!(2233)
pd_boot = rand(1:10000,n_boot)

# put this in a function?
function boot_cf(d)
    Db = d[1:0,:]
    Db[!,:boot] .= []
    for b in 1:n_boot
        @show "working on trial $b"
        pb = pars_full(p_bootstrap[:,b],p)
        pB = prod_pars(chain_B[pd_boot[b],1:12],Kτ)
        pC = prod_pars(chain_C[pd_boot[b],1:12],Kτ)
        db = decomposition_counterfactual(pb,pB,pC,MD,MD1,MD2,MD3,MD4,data,n_idx)
        db[!,:boot] .= b
        Db = [Db; db]
    end
    return Db
end

Db = boot_cf(d)

Db = @combine(groupby(Db,[:source,:variable,:year,:Q,:case]),:sd = std(:value),:q25 = quantile(:value,0.05),:q75 = quantile(:value,0.95))

# write results to file for creating figures
@chain d begin
    @subset(:variable.=="Emp" .|| :variable.=="AFDC")
    innerjoin(_,Db,on=[:source,:variable,:year,:Q,:case])
    CSV.write("output/decomp_counterfactual.csv",_)
end

# write a table with the other stuff that matters:
d2 = @chain d begin
    @subset :variable.!="Emp" :variable.!="AFDC" :variable.!="Earn"
    innerjoin(_,Db,on=[:source,:variable,:year,:Q,:case])
end
# write to file
CSV.write("output/decomp_counterfactual2.csv",d2)

# Write results to a  table for the paper
file = open("output/tables/decomp_counterfactual.tex", "w")
cases = unique(d2.case)
vars = unique(d2.variable)
for s in ("FTP","CTJF","MFIP")
    write(file,"& \\multicolumn{4}{c}{",s,"}\\\\ \n")
    write(file,"&",tex_delimit(cases),"\\\\ \\cmidrule(r){2-5} \n")
    for v in vars
        d3 = @subset d2 :source.==s :variable.==v
        write(file,String(v)," & ",tex_delimit(form.(d3.value)),"\\\\ \n")
        write(file," & ",tex_delimit(formci.(d3.q25,d3.q75)),"\\\\ \n")
    end
end
close(file)
