include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 5 #
Kη = 4 #?
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

# temporarily add this utility function, I think it goes better than the other
function utility(S,A,H,F,p,md::model_data,kA::Int64,kη::Int64,kτ::Int64,t::Int64,eligible::Bool = true)
    if kη>1
        W = exp(logwage(p,md,kτ,kη,t))
    else
        W = 0.
    end
    age_qtrs = md.ageyng*4 + t
    kid_developing = (md.ageyng+fld(t,4))<17
    Γt = kid_developing * exp(p.βΓ[1]*age_qtrs + p.βΓ[2]*age_qtrs^2)
    prF = kid_developing * exp(logpriceF(p,md,kτ,t))
    year = min(2010,md.y0 + fld(md.q0 + t-1,4)) #<- assume expected policy environment is fixed beyond 2010
    Y,_ = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + eligible *  A)
    full_income = p.wq[kτ] + max(Y - prF*F,0.)
    αC = 1. + p.αθ[kτ] * Γt
    #make participation more/less expensive while working *and/or* if newly applying:
    αA = p.αA[kτ] + md.R*(1-H)*p.αR[1]  + p.αP * (2-kA)
    return αC * log(full_income) - αA*A - p.αF[kτ]*F - p.αH[kτ] * H - p.αS[kτ] * S
end


p = expectation_maximization(p,EM,MD,n_idx;max_iter = 250,mstep_iter = 120,save = true)

basic_model_fit(p,EM,MD,data,n_idx,"model_stats_K5.csv")
d = exante_model_fit(p,EM,MD,data,n_idx,"modelfit_exante_K5.csv")
savepars_vec(p,"est_childsample_K5")