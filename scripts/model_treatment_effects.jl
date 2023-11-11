# a script to check the model's unconditional fit of the data
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/counterfactuals.jl")

Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
nests = get_nests()
p = (;p...,nests)

#p = loadpars_vec(p,"current_est")
p = loadpars_vec(p,"est_childsample_K4")

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


d = calculate_treatment_effects(p,EM,MD,data,n_idx)

#write everything to file here.
