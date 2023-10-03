
function utility(S,A,H,F,p::pars,md::model_data,kA::Int64,kη::Int64,kτ::Int64,t::Int64,eligible::Bool = true)
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
    Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + eligible * A)
    full_income = p.wq*(112-30H) + max(Y - prF*F,0.)
    αC = 1. + p.αθ[kτ] * Γt
    #make work more or less expensive while participating:
    αH = p.αH[kτ] + md.R*A*p.αR[2] 
    #make participation more/less expensive while working *and* if newly applying:
    αA = p.αA[kτ] + md.R*(1-H)*p.αR[1]  + p.αP * (2-kA)
    return αC * log(full_income) - αA*A - p.αF[kτ]*F - αH * H - p.αS[kτ] * S
end

function utility(du,S,A,H,F,p::pars,md::model_data,kA::Int64,kη::Int64,kτ::Int64,t::Int64,eligible::Bool = true)
    if kη>1
        W = exp(logwage(p,md,kτ,kη,t))
    else
        W = 0.
    end
    Kτ = p.Kτ
    age_qtrs = md.ageyng*4 + t
    kid_developing = (md.ageyng+fld(t,4))<17
    Γt = kid_developing * exp(p.βΓ[1]*age_qtrs + p.βΓ[2]*age_qtrs^2)
    prF = kid_developing * exp(logpriceF(p,md,kτ,t))
    year = min(2010,md.y0 + fld(md.q0 + t-1,4))
    Y,dY = budget(W*H,0.,md.SOI,md.source,md.arm,year,md.numkids,md.cpi[min(end,t)],S + eligible * A)
    full_income = p.wq*(112-30H) + max(Y - prF*F,0.)
    αC = 1. + p.αθ[kτ] * Γt
    # make work more or less expensive while participating
    αH = p.αH[kτ] + md.R*A*p.αR[2]
    # make participation more/less expensive while working *and* if new applicant
    αA = p.αA[kτ] + md.R*(1-H)*p.αR[1] + p.αP * (2-kA) 

    # derivatives:
    #αθ,αH,αS,αF,αR
    du[kτ] = p.αθ[kτ] * Γt * log(full_income)
    du[Kτ+kτ] = - H
    du[2Kτ+kτ] = - A
    du[3Kτ+kτ] = - S
    du[4Kτ+kτ] = - F
    du[5Kτ+1] = - md.R*(1-H)*A
    du[5Kτ+2] = - md.R*H*A
    du[5Kτ+3] = - A * (2-kA)
    # calc marginal utility of income:
    mu = αC / full_income
    pos = 5Kτ+4
    # wq
    du[pos] = (112-30H) * mu * p.wq
    pos += 1
    # βΓ
    du[pos] = age_qtrs * Γt * log(full_income) * p.αθ[kτ]
    du[pos+1] = age_qtrs^2 * Γt * log(full_income) * p.αθ[kτ]
    pos += 2
    # βw
    if (kη>1) && (H==1) && (Y>F*prF)
        # p.βw[kτ] + p.βw[p.Kτ+1]*md.unemp[min(end,t)] + p.βw[p.Kτ+2]*((md.age0-18)*4+t) + p.ση*p.ηgrid[kη-1]
        du[pos+kτ-1] = W * dY * mu
        pos += Kτ
        du[pos] = W * dY * mu * md.unemp[min(t,end)]
        du[pos+1] = W * dY * mu * ((md.age0-18)*4+t)
        pos += 2
    else
        pos += Kτ+2
    end
    # βf
    mu_p = - mu * (Y>prF) * prF * F
    du[pos+kτ-1] = mu_p #<- 
    pos += Kτ
    du[pos] = md.unemp[min(t,end)] * mu_p
    du[pos+1] = md.numkids * mu_p
    du[pos+2] = (md.ageyng+fld(t-1,4)<=5) * mu_p
    du[pos+3+md.loc_ind-1] = (md.loc_ind>0) * mu_p
    pos += 10
    # ση
    if (kη>1) && (H==1) && (Y>F*prF)
        du[pos] = p.ηgrid[kη-1] * W * dY * mu
    end
    return αC * log(full_income) - αA*A - p.αF[kτ]*F - αH * H - p.αS[kτ] * S
end

