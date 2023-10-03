
logwage(p::pars,md::model_data,kτ::Int64,kη::Int64,t::Int64) = p.βw[kτ] + p.βw[p.Kτ+1]*md.unemp[min(end,t)] + p.βw[p.Kτ+2]*((md.age0-18)*4+t) + p.ση*p.ηgrid[kη-1]

# add staff to child ratios?
#NOTE: we need to add potential experimental variation here
# do we want to estimate it or simply plug it in??
function logpriceF(p::pars,md::model_data,kτ::Int64,t::Int64) 
    # index is FTP 0,1 CTJF 0,1 MFIP 0,1,2 (7 total relative to SIPP)
    return p.βf[kτ] + p.βf[p.Kτ+1]*md.unemp[min(end,t)] + p.βf[p.Kτ+2]*md.numkids + p.βf[p.Kτ+3]*((md.ageyng+fld(t-1,4)<=5)) + p.βf[p.Kτ+3+md.loc_ind]*(md.loc_ind>0)
end

