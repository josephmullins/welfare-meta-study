# set everything up
include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K5")
#p = loadpars_vec(p,"current_est")

x_est = pars_inv_full(p)

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

forward_back_threaded!(p,EM,MD,data,n_idx)


# a couple of functions for getting the childcare fit.
function childcare_fit(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    (;V,vj,logP) = get_model(p)
    D = DataFrame(source = [],arm = [],F = [], est_sample = [],app_status = [])
    for md in MD
        solve!(logP,V,vj,p,md)
        #println(md.case_idx)
        for n in n_idx[md.case_idx]
            d = childcare_fit(p,logP,EM[n],md,data[n])
            #d[!,:n_idx] .= n
            if !isnothing(d)
                D = [D;d]
            end
        end
    end
    D = D[.!isnan.(D.F),:]
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample]),:F => Statistics.mean => :F)

    return d
end
function f_work(s_idx,s_inv)
    j,k = Tuple(s_inv[s_idx])
    _,_,_,H,_ = j_inv(j)
    return H>0
end
function paid_care(s_idx,t,logP,s_inv,k_inv)
    j,k = Tuple(s_inv[s_idx])
    _,kη,_,_ = Tuple(k_inv[k])
    EF = 0.
    norm = 0.
    for j in choice_set(kη>1)
        _,_,_,H,F = j_inv(j)
        norm += H * exp(logP[j,k,t])
        EF += F * exp(logP[j,k,t])
    end
    return EF / norm
end
# next: update paid care to be an expected value:

function childcare_fit(p,logP,em::EM_data,md::model_data,data::likelihood_data)
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((9,K))

    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end

    t = findfirst(data.chcare_valid)
    if !isnothing(t)
        #F = em_mean(em.q_s,t,s->paid_care(s,s_inv),s->f_work(s,s_inv))
        F = em_mean(em.q_s,t,s->paid_care(s,t,logP,s_inv,k_inv),s->f_work(s,s_inv))
        return DataFrame(source = md.source, arm = md.arm, F = F,est_sample = data.use,app_status = app_status)
    else
        return nothing
    end
    
end

d = childcare_fit(p,EM,MD,data,n_idx)

CSV.write("output/childcare_fit.csv",d)