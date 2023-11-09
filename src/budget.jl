include("tools/Transfers.jl")

#include("MDRC_Transfers.jl")

function budget_MDRC(E,state,site,arm,year,num_kids,cpi)
    E *= cpi #<- re-inflate income variables
    #N *= cpi
    N = 0.
    tax = 0.
    dtax = 0.
    if E>0
        tax = Transfers.TAX(E*12,state,year,num_kids)
        tax /= 12
        # in case you see this later and think there should be a dtax /= 12...
        # remember: d[f(α*x)/α]/dx = f'(α*x)
    end

    if site=="MFIP"
       tanf = MFIP(E,N,year,num_kids,arm) 
    elseif site=="CTJF"
        tanf = CTJF(E,N,year,num_kids,arm) 
    elseif site=="FTP"
        tanf = FTP(E,N,year,num_kids,arm) 
    elseif site=="LAGAIN"
        tanf = LAGAIN(E,N,year,num_kids,arm) 
    end
    y = (E + N + tanf - tax) / cpi
    # once again, when reading below remember that d[f(cpi * x)/cpi] / dx = f'(cpi * x)
    return y
end

# budget(E,N,state,year,num_kids,cpi,p)
function budget(E,N,state,source,arm,year,num_kids,cpi,p)
    if source=="SIPP" || p<=1
        return Transfers.budget(E,N,state,year,num_kids,cpi,p)
    else
        return budget_MDRC(E,state,source,arm,year,num_kids,cpi)
    end
end

# Transfers.TANF(E,N::Float64,SOI::Int64,year::Int64,numchild::Int64)
# Transfers.TANF(E,N,NS,rg,rn,dd0,rd0,dd1,rd1,PS,M)


# HAVE TO adjust how this is described in the paper (snap benefit has benefit standard deduction)
function MFIP(E,N,year,numchild,j)
    ii = (year-1975)*51*4 + (24-1)*4 + numchild
    ii2 = (year-1979)*5 + numchild+1
    MB = Transfers.BS[ii]
    if j==0
        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,120,0.33,MB,MB)
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    else
        snap = Transfers.SNAP(0.,N,year,numchild,MB)
        #G = Transfers.MA[ii2]
        MB += snap
        BS = 1.2*MB

        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,0.,0.38,BS,MB)
    end
    return tanf
end

function CTJF(E,N,year,numchild,j)
    ii = (year-1975)*51*4 + (7-1)*4 + numchild
    ii2 = (year-1979)*5 + numchild+1
    BS = Transfers.BS[ii]
    if j==0
        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,120,0.33,BS,BS)
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    else
        snap = Transfers.SNAP(0.,N,year,numchild,BS)
        BS += snap
        NS = Transfers.PG[ii2] #<- once we fix the welfare database we can update this.
        tanf = Transfers.TANF(E,N,NS,1.,0.,0.,0.,0.,1.,BS,BS)
    end
    return tanf
end

function FTP(E,N,year,numchild,j)
    ii = (year-1975)*51*4 + (10-1)*4 + numchild
    #ii2 = (year-1979)*5 + numchild+1
    BS = Transfers.BS[ii]
    if j==0
        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,120,0.33,BS,BS)
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    else
        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,200.,0.5,BS,BS)
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    end    
    return tanf
end

# NEXT: write the LAGAIN function
# THEN: write a budget function that takes site and treatment arm as arguments
function LAGAIN(E,N,year,numchild,j)
    ii = (year-1975)*51*4 + (5-1)*4 + numchild
    #ii2 = (year-1979)*5 + numchild+1
    BS = Transfers.BS[ii]

    if j==0
        tanf = Transfers.TANF(E,N,0.,0.,0.,0.,0.,120,0.33,BS,BS)
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    else
        tanf = Transfers.TANF(E,N,5,year,numchild) #<- this is the standard already? difference is in control?
        snap = Transfers.SNAP(E,N,year,numchild,tanf)
        tanf += snap
    end
    return tanf
end

