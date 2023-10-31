# set everything up
include("../src/model.jl")
include("../src/estimation.jl")

Kτ = 4 #
Kη = 5
p = pars(Kτ,Kη)
p = update_transitions(p)
nests = get_nests()
p = (;p...,nests)

p = loadpars_vec(p,"est_childsample_K4")

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

break

block = [:αH,:αF]
ft = [1,1]
p2 = mstep_major_block(p,block,ft,EM,MD,n_idx,100)

block = [:αH,:αF,:σ]
ft = [1,1,2]
p2 = mstep_major_block(p,block,ft,EM,MD,n_idx,100)

block = [:αH,:αF,:βw,:βf]
ft = [1,1,1,1]
p3 = mstep_major_block(p2,block,ft,EM,MD,n_idx,100)


break

# a couple of functions for getting the childcare fit.
function childcare_fit(p,EM::Vector{EM_data},MD,data::Vector{likelihood_data},n_idx)
    #(;V,vj,logP) = get_model(p)
    D = DataFrame(source = [],arm = [],year=[],Q=[],F = [], est_sample = [],app_status = [])
    for md in MD
        #solve!(logP,V,vj,p,md)
        #println(md.case_idx)
        for n in n_idx[md.case_idx]
            d = childcare_fit(p,[],EM[n],md,data[n])
            #d[!,:n_idx] .= n
            D = [D;d]
        end
    end
    D = D[.!isnan.(D.F),:]
    d = combine(groupby(D,[:source,:arm,:app_status,:est_sample,:year,:Q]),:F => Statistics.mean => :F)

    return d
end
function f_work(s_idx,s_inv)
    j,k = Tuple(s_inv[s_idx])
    _,_,_,H,_ = j_inv(j)
    return H>0
end

function childcare_fit(p,logP,em::EM_data,md::model_data,data::likelihood_data)
    T = size(em.q_s,2)
    Q = 0:T-1
    N = zeros(T)
    F = zeros(T)
    #H = zeros(T)
    
    k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
    K = prod(size(k_inv))
    s_inv = CartesianIndices((9,K))

    # year = md.y0 .+ fld.(md.q0-1 .+ (0:T-1),4)
    # Q = mod.(md.q0-1 .+ (0:T-1),4) .+ 1
    year = md.y0 .+ fld.(md.q0 .+ (0:T-1),4) #<- start with one quarter delay in this model.
    Q = mod.(md.q0 .+ (0:T-1),4) #<- as above
    for t in 1:T
        F[t] = em_mean(em.q_s,t,s->paid_care(s,s_inv),s->f_work(s,s_inv))
        #H[t] = em_mean(em.q_s,t,s->f_work(s,s_inv))
    end
    keep = .!data.choice_missing
    if md.source=="MFIP"
        app_status = 3 - 2*data.X_type[5] - data.X_type[6]
    elseif md.source=="FTP" || md.source=="CTJF"
        app_status = data.X_type[5]
    else
        app_status = 0
    end
    return DataFrame(source = md.source, arm = md.arm, year = year[keep], Q = Q[keep], F = F[keep],est_sample = data.use,app_status = app_status)
end

d = childcare_fit(p,EM,MD,data,n_idx)

CSV.write("output/childcare_fit.csv",d)