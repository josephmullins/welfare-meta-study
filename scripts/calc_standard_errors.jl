include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")

x_est = pars_inv_full(p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
sipp = @subset panel :source.=="SIPP"
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    vcat(sipp) #<- add this back in eventually
end

MD,EM,data,n_idx = estimation_setup(panel);

Random.seed!(2020)
shuffle!(MD)

forward_back_threaded!(p,EM,MD,data,n_idx)
LL = log_likelihood_n(x_est,p,EM,MD,data,n_idx)

ll(x) = log_likelihood_n(x,p,EM,MD,data,n_idx)
using ForwardDiff: jacobian, Chunk, JacobianConfig

cfg = JacobianConfig(ll, x_est, Chunk{4}());

scores = jacobian(ll,x_est,cfg)
ss = sum(scores,dims=1)[:]
i_drop = abs.(ss).<1e-10
i_keep = .~i_drop

N = sum(length(n_idx[md.case_idx]) for md in MD) #<- this ~slightly~ overstates the sample size?

V = inv(cov(scores[:,i_keep])) / N
se = sqrt.(diag(V))

se_full = zeros(length(x_est))
se_full[i_keep] .= se
V_full = diagm(fill(1e-8,length(x_est))) 
V_full[i_keep,i_keep] .= V

writedlm("output/var_est_K5",V_full)

# -- now let's make some tables?
p2 = pars_full(se_full,p)

# -- functions for writing output:
using Printf
form(x) = @sprintf("%0.2f",x)
formse(x) = string("(",@sprintf("%0.2f",x),")")
# a helper function to write a collection of strings into separate columns
function tex_delimit(x)
    str = x[1]
    num_col = length(x)
    for i in 2:num_col
        str *=  "&" * x[i]
    end
    return str
end



# ------ make a table for preferences --------------- #
file = open("output/tables/preference_ests.tex", "w")
write(file," & \\multicolumn{6}{c}{Type-Specific Parameters} \\\\ \n")
write(file,"Type & \$\\alpha_{H}\$ & \$\\alpha_{A}\$ & \$\\alpha_{S}\$ & \$\\alpha_{F}\$ & \$\\alpha_{\\theta}\$ & \$y\$ \\\\ \\cmidrule(r){1-1} \\cmidrule(r){2-7} \n")

for k in 1:Kτ
    str = "\$k_{\\tau}=$k\$ &" * form(p.αH[k]) * "&" * form(p.αA[k]) * "&" * form(p.αS[k]) * " & " * form(p.αF[k]) * " & " * form(p.αθ[k]) * "&" * form(p.wq[k]) * "\\\\ \n"
    write(file,str)
    str = "&" * formse(p2.αH[k]) * "&" * formse(p2.αA[k]) * "&" * formse(p2.αS[k]) * " & " * formse(p2.αF[k]) * " & " * formse(p.αθ[k] * log(p2.αθ[k])) * "&" * formse(p.wq[k] * log(p2.wq[k])) * "\\\\ \n"
    write(file,str)
end

write(file,"& \\multicolumn{6}{c}{Global Parameters} \\\\ \n")
write(file,"& \$\\beta\$ & \$\\sigma_{F}\$ & \$\\sigma_{H}\$ & \$\\sigma_{P}\$ & \$\\alpha_{R}\$ & \$\\alpha_{P}\$ \\\\ \\cmidrule(r){2-7}")
write(file, "&" * form(p.β) * "&" * form(p.σ[1]) * "&" * form(p.σ[2]) * "&" * form(p.σ[3]) * "&" * form(p.αR) * "&" * form(p.αP) * "\\\\ \n")
write(file, "&" * formse(p.β * (1-p.β) * logit_inv(p2.β)) * "&" * formse(p.σ[1] * log(p2.σ[1])) * "&" * formse(p.σ[2] * log(p2.σ[2])) * "&" * formse(p.σ[3] * log(p2.σ[3])) * "&" * formse(p2.αR) * "&" * formse(p2.αP) * "\\\\ \n")
close(file)

# --------- make a table for wage shock parameters ------- #

file = open("output/tables/transition_ests.tex", "w")
write(file," & \\multicolumn{3}{c}{Type-Specific Parameters} \\\\ \n")
write(file,tex_delimit(["Type","\$\\lambda_0\$","\$\\lambda_1\$","\$\\delta\$"]),"\\\\ \\cmidrule(r){1-1}\\cmidrule(r){2-4} \n")
for k in 1:Kτ
    write(file,"\$k_{\\tau}=$k\$ &",tex_delimit(form.([p.λ₀[k],p.λ₁[k],p.δ[k]])),"\\\\ \n")
    write(file,"&",tex_delimit(formse.([logit_inv(p2.λ₀[k]) * (1-p.λ₀[k]) * p.λ₀[k],logit_inv(p2.λ₁[k]) * (1-p.λ₁[k]) * p.λ₁[k],logit_inv(p2.δ[k]) * (1-p.δ[k]) * p.δ[k]])),"\\\\ \n")
end
write(file,"& \\multicolumn{3}{c}{Global Parameters} \\\\ \n")
write(file,tex_delimit(["","\$\\mu_{offer}\$","\$\\sigma_{offer}\$","\$\\lambda_R\$"]),"\\\\ \\cmidrule(r){2-4} \n")
write(file,"&",tex_delimit(form.([p.μₒ,p.σₒ,p.λR])),"\\\\ \n")
write(file,"&",tex_delimit(formse.([p2.μₒ,p.σₒ * log(p2.σₒ),p2.λR])),"\\\\ \n")

close(file)

# ------ make a table for prices ------- #
# wages: k1,k2,k3,k4,k5,unemp,age
# childcare: k1,k2,k3,k4,k5,unemp,num kids, age youngest <=5, FTP - control, FTP - treat, CTJF - control, CTJF - treat, MFIP - control, MFIP - treat, MFIP - treat 2 
file = open("output/tables/price_ests.tex", "w")
write(file," & Wages & Childcare \\\\ \\cmidrule(r){2-3} \n")
for k in 1:Kτ
    write(file,"Type $k &",form(p.βw[k]),"&",form(p.βf[k]),"\\\\ \n")
    write(file,"& ",formse(p2.βw[k]),"&",formse(p2.βf[k]),"\\\\ \n")
end
# next two covariates are wages only
write(file,"Unemployment Rate &",form(p.βw[Kτ+1]),"&",form(p.βf[Kτ+1]),  "\\\\ \n ")
write(file," & ",formse(p2.βw[Kτ+1]),formse(p2.βf[Kτ+1])," \\\\ \n")
write(file,"Age &",form(p.βw[Kτ+2]),"& - \\\\ \n ")
write(file," & ",formse(p2.βw[Kτ+2]),"& \\\\ \n")
# the remaining covariates are for childcare only:
vnames = ["Num. Kids","Youngest \$\\leq 5\$","FTP Control","FTP Treat","CTJF Control","CTJF Treat","MFIP Control","MFIP Treat","MFIP Incentives"]
for k in eachindex(vnames)
    write(file,vnames[k],"& - &",form(p.βf[Kτ+1+k]),"\\\\ \n")
    write(file,"& - &",formse(p2.βf[Kτ+1+k]),"\\\\ \n")
end
# finally, standard errors
write(file," Measurement error (std. dev) & ",form(p.σ_W)," &",form(p.σ_PF),"\\\\ \n")
write(file, " & ",formse(p.σ_W*p2.σ_W)," & ",formse(p.σ_PF * p2.σ_PF),"\\\\ \n")
close(file)

# 


# ------ make a some data to illustrate type selection ------- #
dist_η = zeros(Kη,2)
denom_η = zeros(2)
dist_τ = zeros(Kτ,2)
denom_τ = zeros(2)
for md in MD
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv2 = CartesianIndices((9,K))
    l = 1 + 1*(md.source=="SIPP")
    for n in n_idx[md.case_idx]
        for s in nzrange(EM[n].q_s,1)
            s_idx = EM[n].q_s.rowval[s]
            wght = EM[n].q_s.nzval[s]
            _,k = Tuple(s_inv2[s_idx])
            _,kη,kω,kτ = Tuple(k_inv[k])
            dist_η[kη,l] += wght
            denom_η[l] += wght
            dist_τ[kτ,l] += wght
            denom_τ[l] += wght
        end
    end
end
dist_η ./= denom_η'
dist_τ ./= denom_τ'
d = [DataFrame(var = "Wage Shock",value = 1:Kη,dist = dist_η[:,1],source="Experiments");
DataFrame(var = "Wage Shock",value = 1:Kη,dist = dist_η[:,2],source="SIPP");
DataFrame(var = "Type",value = 1:Kτ,dist = dist_τ[:,1],source="Experiments");
DataFrame(var = "Type",value = 1:Kτ,dist = dist_τ[:,2],source="SIPP")]
CSV.write("output/initial_dists.csv",d)