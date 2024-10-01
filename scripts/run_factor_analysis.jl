using DataFrames, DataFramesMeta, CSV, Turing, LinearAlgebra
include("../src/model.jl")
include("../src/estimation.jl")
include("../src/estimation/production.jl")
include("../src/estimation/FactorAnalysis.jl")

scores = CSV.read("../Data/Data_child_prepped.csv",DataFrame,missingstring = "NA")
scores = @orderby(scores,:source,:id)

cols = (:BPIE,:BPIN,:PBS,:ENGAGE,:REPEAT,:SUSPEND,:ACHIEVE,:TCH_AVG)
columns = (MFIP = cols[1:end-1],FTP = cols[1:end-1],CTJF = cols)
# convert factor scores to an array for estimation
M = prep_scores(scores,cols)
# calculate the optimal weighting matrix
wght = get_weighting_matrix(scores,columns)
Λ,Σ,D = estimate_system(scores,wght,columns)

# calculate the factor scores:
est,sd = factor_analysis(scores,columns)
[est sd]
# here we should run the whole analysis.
Λ,Σ,D = measurement_system(est)
Λ_se,_,Dse = measurement_system(sd)




# ------ now let's make a table of estimates ------- #
using Printf
form(x) = x!=0 ? @sprintf("%0.2f",x) : "-"
formse(x) = x!=0 ? string("(",@sprintf("%0.2f",x),")") : ""


file = open("output/tables/factor_analysis.tex", "w")

write(file,"Measure & \$\\lambda^{m}_{B}\$ & \$\\lambda^{m}_{B}\$ & \$\\sigma^2_{m}\$ \\\\ \\cmidrule(r){1-4} \n")
measures = ("BPI-Externalizing","BPI-Internalizing","Positive Behavior Scale","School Engagement","Ever Repeat Grade","Ever Suspended","School Achievement - Parent","School Achievement - Teacher")
for m in eachindex(measures)
    write(file,measures[m]," &",form(Λ[m,1])," & ",form(Λ[m,2])," & ",form(D[m,m]),"\\\\ \n")
    write(file," &",formse(Λ_se[m,1])," & ",formse(Λ_se[m,2])," & ",formse(2sqrt(Dse[m,m])*D[m,m]),"\\\\ \n")
end
close(file)



# var = λ^2 + D, so one sd is sqrt of this.
th = factor_scores(scores,M,Λ,Σ,D) #<- add Standard Errors?

break
# -- look at treatment effects in terms of the factor score:
panel = CSV.read("../Data/Data_prepped.csv",DataFrame,missingstring = "NA")
#select!(panel,Not(:case_idx)) #<- have to add this to other script or re-clean data

scores = @chain panel begin
    @select :id :source :arm :app_status :county
    unique()
    innerjoin(scores,on=[:id,:source])
    @transform(:thB = th[:,1],:thC = th[:,2])
end

using FixedEffectModels
reg(scores, @formula(thB ~ source*app_status*county + source*arm))
# very few effects here? compare this with what's in the reports?
@chain scores begin
    @subset :source.=="FTP"
    reg(@formula(thB ~ arm + fe(app_status)))
end

@chain scores begin
    @subset :source.=="CTJF"
    reg(@formula(thC ~ arm + fe(county)*fe(app_status)))
end

@chain scores begin
    @subset :source.=="MFIP" :arm.!=2
    reg(@formula(thC ~ arm + fe(county)*fe(app_status)))
end

@chain scores begin
    @subset :source.=="MFIP" :arm.!=1
    @transform :arm2 = :arm.==2
    reg(@formula(thC ~ arm2 + fe(county)*fe(app_status)))
end
