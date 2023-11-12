# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/counterfactuals.jl")

Kτ = 3 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"current_est")
#p = loadpars_vec(p,"est_childsample_K4")
p = loadpars_vec(p,"est_noSIPP_K3")

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
MD1 = copy(MD)
for m in eachindex(MD)
    MD1[m] = full_treatment(MD[m])
    MD[m] = control(MD[m])
end

d = counterfactual(p,MD,MD1,data,n_idx)

#d = calculate_treatment_effects(p,EM,MD,data,n_idx)

#write everything to file here.
