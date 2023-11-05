
function estimation_setup(panel::DataFrame)
    case_idx = @chain panel begin
        @select(:source,:panel,:SOI,:arm,:age,:ageyng,:numkids)
        unique()
        @transform(:case_idx = eachindex(:source))
        @orderby(:case_idx)
    end

    cases = @chain panel begin
        #@subset(:t0.==0) #<- collect these from the first available period for each dataset
        # first collect the state-specific panel data:
        @select(:source,:panel,:SOI,:year,:Q,:cpi,:unemp)
        unique()
        # then merge with the case variables to bring in that data for each case.
        innerjoin(case_idx,on=[:source,:panel,:SOI])
        @orderby(:case_idx,:year,:Q)
    end

    # collect the different versions of the model (indexed by case_idx)
    MD = model_data[]
    for df in groupby(cases,:case_idx)
        md = model_data(df)
        push!(MD,md)
    end

    # create the index of observations for each case. this tells us which observations pertain to each case.
    id = @chain panel begin
        @select(:id,:source,:panel,:SOI,:arm,:age,:ageyng,:numkids)
        unique()
        innerjoin(case_idx,on=[:source,:panel,:SOI,:arm,:age,:ageyng,:numkids])
        @orderby(:case_idx,:source,:id)
        @transform(:n = eachindex(:id)) #<- turn the id number into an ordered integer from 1.
    end
    n_idx = Vector{Int64}[]
    for df in groupby(id,:case_idx)
        push!(n_idx,df.n)
    end

    # finally, merge the case index to the full panel:
    panel = @chain id begin
        @select(:id,:n,:source,:case_idx)
        innerjoin(panel,on=[:id,:source])
        @orderby(:case_idx,:n)
    end

    # put the data together for evaluating the likelihood
    data = likelihood_data[]
    EM = EM_data[]

    for df in groupby(panel,[:case_idx,:id])
        d = likelihood_data(df)
        case_idx = d.case_idx
        md = MD[case_idx]
        m_idx = 1 + (md.arm==1)*((md.source=="FTP") + 2*(md.source=="CTJF"))
        em = get_EM_data(p,md,d)
        push!(data,d)
        push!(EM,em)
    end
    
    return MD,EM,data,n_idx
end

# function to get model data from the dataframe df, used in estimation_setup()
function model_data(df)
    source = df.source[1]
    if source=="SIPP"
        T = size(df,1) - 1 #<- subtract one because we drop the first period for SIPP
    else
        T = size(df,1)
    end
    arm = df.arm[1]
    # locations: FTP0,FTP1,CT0,CT1,MF0,MF1,MF2 (SIPP excluded)
    # these are used to create dummy variables for the price of childcare
    loc_ind = 1*(source=="FTP") + (source=="CTJF")*3 + (source=="MFIP")*5 + arm
    if source=="FTP" && arm==1
        Kω = 1 + 7 #<- 21 month time limit
        TL = true
        R = 1.
        # TL = fill(true,T) 
        # R = ones(T)
    elseif source=="CTJF" && arm==1
        Kω = 1 + 8 #<- 24 month time limit
        TL = true
        R = 1.
        # TL = fill(true,T)
        # R = ones(T)
    elseif source=="MFIP" && arm==1
        Kω = 1
        TL = false
        R = 1
        # TL = fill(false,T)
        # R = ones(T)
    else
        TL = false
        R = 0
        # TL = fill(false,T)
        # R = zeros(T)
        Kω = 1
    end
    # get the type block:
    if source=="SIPP"
        type_block = 1:4
    elseif source=="FTP"
        type_block = 5:9
    elseif source=="CTJF"
        type_block = 10:15
    elseif source=="MFIP"
        type_block = 16:23
    end

    return model_data(df.case_idx[1],df.year[1],df.Q[1],T,df.age[1],df.ageyng[1],source,arm,loc_ind,df.SOI[1],df.numkids[1],
    df.unemp,df.cpi,R,Kω,TL,type_block)
end

# function to get likelihood data from the data frame, used in estimation_setup()
function likelihood_data(df)
    case_idx = df.case_idx[1]
    t0 = df.t0[1] 
    T = size(df,1)
    pay_care_valid = .!ismissing.(df.pay_care)
    pay_care = coalesce.(df.pay_care,false)
    chcare_valid = .!ismissing.(df.chcare)
    chcare = coalesce.(df.chcare,-1.)
    log_chcare = log.(coalesce.(max.(df.chcare,0.),0.))
    wage_valid = .!ismissing.(df.EARN) .& (df.EARN.>0)
    logW = log.(coalesce.(df.EARN,0.))
    choice_missing = ismissing.(df.AFDC) .| ismissing.(df.FS) .| ismissing.(df.EMP)
    AFDC = coalesce.(df.AFDC,-1)
    EMP = coalesce.(df.EMP,-1)
    FS = coalesce.(df.FS,-1)
    FC = coalesce.(df.pay_care,-1)
    X_type = []
    source = df.source[1]
    Kx = 4 + (source=="FTP") + 2*(source=="CTJF") + 4*(source=="MFIP")
    X_type = zeros(Kx)
    ed_dum = 2*df.hs[1] + 3*df.some_coll[1] + 3*df.coll[1] #<- grouping some_coll and coll
    X_type[1] = 1.
    if ed_dum>1
        X_type[ed_dum] = 1.
    end
    X_type[4] = df.numkids[1]
    if source=="FTP"
        X_type[5] = df.app_status[1] #<- new applicant (recipient excluded)
    elseif source=="CTJF"
        X_type[5] = df.app_status[1] #<- new applicant (recipient excluded)
        X_type[6] = df.county[1]==2 #<- New Haven county (Manchester excluded)
    elseif source=="MFIP"
        X_type[5] = df.app_status[1]==1 #<- new applicant (recipient excluded)
        X_type[6] = df.app_status[1]==2 #<- re-applicant
        X_type[7] = df.county[1]==2 #<- Anoka county (Hennepin excluded)
        X_type[8] = df.county[1]==3 #<- Dakota county (Hennepin excluded)
    end
    # leave-out sample is non-control group new applicants in Anoka county, dakota county, and Manchester county
    leaveout = (source=="MFIP" && df.arm[1]>0 && df.county[1]>1 && df.app_status[1]==1) || (source=="CTJF" && df.arm[1]==1 && df.county[1]==1 && df.app_status[1]==1)
    use = !leaveout
    if source=="SIPP"
        # for the SIPP we drop the first observation in order to get lagged AFDC
        # - add 1 t0 because we won't use the first observation in the data
        # - similarly subtract one from T
        kA = AFDC[1]
        return likelihood_data(
        case_idx,t0+1,T-1,pay_care_valid[2:end],pay_care[2:end],chcare_valid[2:end],chcare[2:end],log_chcare[2:end],wage_valid[2:end],logW[2:end],
        choice_missing[2:end],AFDC[2:end],EMP[2:end],FS[2:end],FC[2:end],
        df.less_hs[1],df.hs[1],df.some_coll[1],df.coll[1],X_type,kA,true
        )
    else
        if source=="MFIP"
            kA = df.app_status[1]==3 
        else
            kA = df.app_status[1]==2
        end
        return likelihood_data(
        case_idx,t0,T,pay_care_valid,pay_care,chcare_valid,chcare,log_chcare,wage_valid,logW,
        choice_missing,AFDC,EMP,FS,FC,
        df.less_hs[1],df.hs[1],df.some_coll[1],df.coll[1],X_type,AFDC[1],use
        )
    end
end

# function to initialize EM data given model and data
# this function is important! It determines the sparsity structure of each EM_data object, which all calls to the likelihood function will use
function get_EM_data(p,md::model_data,data::likelihood_data)
    J = 9
    T = data.T #
    t0 = data.t0
    Kτ = p.Kτ
    Kη = p.Kη
    Kω = md.Kω
    K = 2*Kτ*Kη*Kω 
    S = K*J
    k_idx = LinearIndices((2,Kη,Kω,Kτ))
    k_inv = CartesianIndices((2,Kη,Kω,Kτ))
    s_inv = CartesianIndices((J,K))

    α = spzeros(S,T)
    β = spzeros(S,T)
    q_s = spzeros(S,T)
    q_ss = [spzeros(S,S) for t in 1:T-1]
    P = [spzeros(S,S) for t in 1:T-1]
    # initialize α: (which provides an index for which states have positive probability)
    # start with period 1:
    for kτ in 1:Kτ, kη in 1:Kη
        kA = 1 + data.kA #<- lagged AFDC
        k = k_idx[kA,kη,1,kτ]
        if !data.wage_valid[1] || kη>1 #<- rule out states when kη=1 and EARN>0
            if !data.choice_missing[1]
                AFDC = data.AFDC[1]
                FS = max(AFDC,data.FS[1]) #<- AFDC=1 implies FS=1 in the model
                EMP = data.EMP[1]
                if EMP==0
                    j = j_idx(FS,AFDC,EMP,0)
                    s = (k-1)*J + j
                    α[s,1] = 1 / (Kη*Kτ)
                else
                    j1 = j_idx(FS,AFDC,EMP,0)
                    j2 = j_idx(FS,AFDC,EMP,1)
                    s1 = (k-1)*J + j1
                    s2 = (k-1)*J + j2
                    α[s1,1] = 0.5 / (Kη*Kτ)
                    α[s2,1] = 0.5 / (Kη*Kτ)
                end
            else
                for j in 1:J
                    s = (k-1)*J + j
                    α[s,1] = 1 / (Kη*Kτ)
                end
            end
        end
    end
    for t in 1:T-1
        # for each s in α: add non-zero entries for P and α?
        for s in nzrange(α,t)
            s_idx = α.rowval[s]
            j,k = Tuple(s_inv[s_idx])
            _,A,_,_,_ = j_inv(j)
            _,kη,kω,kτ = Tuple(k_inv[k])
            kω_next = min(kω + A,md.Kω)
            kA_next = 1 + A
            for kη_next in 1:Kη
                kn_idx = k_idx[kA_next,kη_next,kω_next,kτ]
                fkk = p.Fη[kη_next,kη,kτ]
                if fkk>0
                    if !data.wage_valid[t+1] || kη_next>1 #<- rule out states when kη=1 and EARN>0
                        if !data.choice_missing[t+1] #y[t+1]!=-1
                            AFDC = data.AFDC[t+1]
                            FS = max(AFDC,data.FS[t+1]) #<- AFDC=1 implies FS=1 in the model
                            EMP = data.EMP[t+1]
                            if EMP==0
                                jn = j_idx(FS,AFDC,EMP,0)
                                sn = (kn_idx-1)*J+jn
                                P[t][sn,s_idx] = 1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                                α[sn,t+1] = 1. #p_y[jn,k_idx,t+1]
                            else
                                if data.pay_care_valid[t+1]
                                    FC = data.FC[t+1]
                                    jn = j_idx(FS,AFDC,EMP,FC)
                                    sn = (kn_idx-1)*J+jn
                                    P[t][sn,s_idx] = 1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                                    α[sn,t+1] = 1. #p_y[jn,k_idx,t+1]
                                else
                                    jn1 = j_idx(FS,AFDC,EMP,0)
                                    jn2 = j_idx(FS,AFDC,EMP,1)
                                    sn1 = (kn_idx-1)*J+jn1
                                    P[t][sn1,s_idx] = 1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                                    α[sn1,t+1] = 1. #p_y[jn,k_idx,t+1]
                                    sn2 = (kn_idx-1)*J+jn2
                                    P[t][sn2,s_idx] = 1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                                    α[sn2,t+1] = 1. #p_y[jn,k_idx,t+1]
                                end
                            end
                        else
                            for jn in 1:J
                                sn = (kn_idx-1)*J+jn
                                P[t][sn,s_idx] =  1. #F[j,t][k_idx,k]*p_y[jn,k_idx,t+1]
                                α[sn,t+1] = 1. #p_y[jn,k_idx,t+1]
                            end
                        end
                    end
                end
            end
        end
    end
    # initialize β in last period
    for s in nzrange(α,T)
        s_idx = α.rowval[s]
        β[s_idx,T] = 1.
    end
    
    return EM_data(α,β,q_s,q_ss,P)
end
