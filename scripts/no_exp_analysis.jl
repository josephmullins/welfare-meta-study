# this script loads estimates from the case without experimental data and compares to the baselin
include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5_noexp")

x_est = pars_inv_full(p)

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

# we have to split the panel and then put it back together again
panel = @chain scores begin
    @select :id :source
    innerjoin(panel,on=[:id,:source])
    @subset :arm.==0
end

MD,EM,data,n_idx = estimation_setup(panel);

Random.seed!(2020)
shuffle!(MD)


forward_back_threaded!(p,EM,MD,data,n_idx)
LL = log_likelihood_n(x_est,p,EM,MD,data,n_idx)

scores = get_score(x_est,p,EM,MD,data,n_idx)

# we want to look only at σ and β:
sc = scores[:,(11p.Kτ+21):(11p.Kτ+24)]
xe = x_est[(11p.Kτ+21):(11p.Kτ+24)]

N = sum(length(n_idx[md.case_idx]) for md in MD) #<- this ~slightly~ overstates the sample size?

V = inv(cov(sc)) / N
se = sqrt.(diag(V))

p2 = pars(se,p,[:σ;:β],[2,3])

# -- now let's make some tables?
#p2 = pars_full(se_full,p)

pb = loadpars_vec(p,"est_childsample_K5")
V = readdlm("output/var_est_K5")
se = sqrt.(diag(V))
pb2 = pars_full(se,pb)


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

# ------ make the comparison table
file = open("output/tables/no_exp_ests.tex", "w")
write(file,"& \$\\beta\$ & \$\\sigma_{3}\$ & \$\\sigma_{2}\$ & \$\\sigma_{1}\$ \\\\ \\cmidrule(r){2-5}")
write(file, "Control Group Only &" * form(p.β) * "&" * form(p.σ[1]) * "&" * form(p.σ[2]) * "&" * form(p.σ[3]) *  "\\\\ \n")
write(file, "&" * formse(p.β * (1-p.β) * logit_inv(p2.β)) * "&" * formse(p.σ[1] * log(p2.σ[1])) * "&" * formse(p.σ[2] * log(p2.σ[2])) * "&" * formse(p.σ[3] * log(p2.σ[3])) * "\\\\ \n")
write(file, "Full Sample &" * form(pb.β) * "&" * form(pb.σ[1]) * "&" * form(pb.σ[2]) * "&" * form(pb.σ[3]) *  "\\\\ \n")
write(file, "&" * formse(pb.β * (1-pb.β) * logit_inv(pb2.β)) * "&" * formse(pb.σ[1] * log(pb2.σ[1])) * "&" * formse(pb.σ[2] * log(pb2.σ[2])) * "&" * formse(pb.σ[3] * log(p2.σ[3])) * "\\\\ \n")
close(file)

