
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
    if A==1
        Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],eligible * (S + A))
    else
        Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S)
    end
    full_income = p.wq[kτ] + max(Y - prF*F,0.)
    αC = 1. + p.αθ[kτ] * Γt
    #make participation more/less expensive while working *and/or* if newly applying:
    αA = p.αA[kτ] + md.R*(1-H)*p.αR[1]  + p.αP * (2-kA)
    return αC * log(full_income) - αA*A - p.αF[kτ]*F - p.αH[kτ] * H - p.αS[kτ] * S
end
