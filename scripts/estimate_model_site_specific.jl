include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 3 #
Kη = 5 #?
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

#p = expectation_maximization(p,EM,MD,n_idx; max_iter = 4, mstep_iter = 20,save = true)

# -- Do FTP first
mdFTP = MD[[md.source=="FTP" || md.source=="SIPP" for md in MD]]
pF = expectation_maximization(p,EM,mdFTP,n_idx;max_iter = 50,mstep_iter = 120,save = true)
#basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample_K2.csv")
savepars_vec(pF,"est_FTP_K3")
d = exante_model_fit(pF,EM,mdFTP,data,n_idx,"modelfit_exante_FTP.csv")

# -- Now CTJF
mdCTJF = MD[[md.source=="CTJF" || md.source=="SIPP" for md in MD]]
pC = expectation_maximization(p,EM,mdCTJF,n_idx;max_iter = 50,mstep_iter = 120,save = true)
#basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample_K2.csv")
savepars_vec(pC,"est_CTJF_K3")
d = exante_model_fit(pC,EM,mdCTJF,data,n_idx,"modelfit_exante_CTJF.csv")

# -- Now MFIP
mdMFIP = MD[[md.source=="MFIP" || md.source=="SIPP" for md in MD]]
pM = expectation_maximization(p,EM,mdMFIP,n_idx;max_iter = 50,mstep_iter = 120,save = true)
#basic_model_fit(p,EM,MD,data,n_idx,"model_stats_childsample_K2.csv")
savepars_vec(pM,"est_MFIP_K3")
d = exante_model_fit(pM,EM,mdMFIP,data,n_idx,"modelfit_exante_MFIP.csv")
