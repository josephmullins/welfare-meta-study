# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/counterfactuals.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")

panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")

# pull out the sipp data
sipp = @subset panel :source.=="SIPP"

MD,EM,data,n_idx = estimation_setup(sipp);

chain = CSV.read("output/production_ests_hetero.csv",DataFrame)
chain_B = chain[1:10000,:]
chain_C = chain[10001:20000,:]

pB = prod_pars([mean(x) for x in eachcol(chain_B)[1:12]],Kτ)
pC = prod_pars([mean(x) for x in eachcol(chain_C)[1:12]],Kτ)


MD1 = Array{model_data,2}(undef,length(MD),2)
MD2 = Array{model_data,2}(undef,length(MD),2)
MD3 = Array{model_data,2}(undef,length(MD),2)
for m in eachindex(MD)
    md = MD[m]
    MD1[m,2] = convert_sipp(md,"FTP")
    md = @set md.SOI = 10
    MD1[m,1] = @set md.y0 = 1994

    MD2[m,2] = convert_sipp(md,"CTJF")
    md = @set md.SOI = 7
    MD2[m,1] = @set md.y0 = 1996

    MD3[m,2] = convert_sipp(md,"MFIP")
    md = @set md.SOI = 24
    MD3[m,1] = @set md.y0 = 1994
end

# extent out the horizon for simulation
for n in eachindex(data)
    d = data[n]
    data[n] = @set d.T = 24
end

function non_selected_counterfactual(p,pB,pC,MD1,MD2,MD3,data,n_idx)
    @views d1 = counterfactual(p,pB,pC,MD1[:,1],MD1[:,2],data,n_idx)
    d1[!,:case] .= "FTP"

    @views d2 = counterfactual(p,pB,pC,MD2[:,1],MD2[:,2],data,n_idx)
    d2[!,:case] .= "CTJF"

    @views d3 = counterfactual(p,pB,pC,MD3[:,1],MD3[:,2],data,n_idx)
    d3[!,:case] .= "MFIP"

    return [d1;d2;d3]
end

d = non_selected_counterfactual(p,pB,pC,MD1,MD2,MD3,data,n_idx)

n_boot = 70
x_est = pars_inv_full(p)
V = readdlm("output/var_est_K5")
V = Hermitian(V)
p_bootstrap = rand(MvNormal(x_est,V),n_boot)
cols = names(d)

Random.seed!(2233)
pd_boot = rand(1:10000,n_boot)

function boot_cf(d)
    Db = d[1:0,:]
    Db[!,:boot] .= []
    for b in 1:n_boot
        @show "working on trial $b"
        pb = pars_full(p_bootstrap[:,b],p)
        pB = prod_pars(chain_B[pd_boot[b],1:12],Kτ)
        pC = prod_pars(chain_C[pd_boot[b],1:12],Kτ)
        db = non_selected_counterfactual(pb,pB,pC,MD1,MD2,MD3,data,n_idx)
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
    CSV.write("output/non_selected_counterfactual.csv",_)
end

# what is different when comparing SIPP to any site?
## (1) unemployment rate.
## (2) the type selection probabilities.
## (3) the location index for childcare.
## (4) the cpi/years
## (5) the coefficients and covariates used for type probabilities.

# write a table with the other stuff that matters:
d2 = @chain d begin
    @subset :variable.!="Emp" :variable.!="AFDC" :variable.!="Earn"
    innerjoin(_,Db,on=[:source,:variable,:year,:Q,:case])
end
# save other statistics to file
CSV.write("output/non_selected_counterfactual2.csv",d2)

# --- Write results to a table

file = open("output/tables/non_selected_counterfactual.tex", "w")
cases = unique(d2.case)
vars = unique(d2.variable)
write(file,"&",tex_delimit(cases),"\\\\ \\cmidrule(r){2-4} \n")
for v in vars
    d3 = @subset d2 :variable.==v
    write(file,String(v), " & ",tex_delimit(form.(d3.value)),"\\\\ \n")
    write(file," & ",tex_delimit(formci.(d3.q25,d3.q75)),"\\\\ \n")
end
close(file)

