# this script defines the states and their transition
# "states" here are those that are endogenous to the model or unobserved
# kτ: unobserved type
# kω: cumulative time use
# kη: wage shock
# kA: indicator for whether you participated last year in welfare

# extend the initial wage offer here, would be a good idea for model fit.
function Fη(kη_next,kη,λ,δ,πW,Kη)
    if kη==1
        if kη_next==1
            return 1-λ
        else
            return λ / (Kη - 1)
        end
    else
        if kη_next==1
            return δ
        else
            p = (kη_next==kη)*πW + (kη_next==max(kη-1,2)) * (1-πW)/2 + (kη_next==min(kη+1,Kη)) * (1-πW)/2
            return (1-δ) * p
        end
    end
end

function next(A,kA,kω;Kω)
    kA_next = 1+A
    kω_next = min(kω + A,Kω)
    return kA_next,kω_next
end