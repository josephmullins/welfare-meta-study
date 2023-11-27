# - here we define choices and their mapping to a single index
# - we also define the structure of the nested logit

# choices are: 
# S ∈(0,1) food stamps
# A ∈(0,1) AFDC/TANF (A=1 only if S=1)
# P = S + A indexes program participation (sometimes useful)
# H ∈(0,1) labor supply (restricted 30 hours)
# F ∈(0,1) formal care (F=1 only if H=1)

j_idx(S,A,H,F) = (S + A)*3 + H + F + 1

J = 9 #<- 9 choices, total
function j_inv(j)
    p = fld(j-1,3)
    hf = mod(j-1,3)
    S::Int64 = p>0
    A::Int64 = p==2
    H::Int64 = hf>0
    F::Int64 = hf==2
    return S,A,p,H,F
end

# don't need this!
function choice_set(job_offer::Bool)
    if job_offer
        return (1,2,3,4,5,6,7,8,9)
    else
        return (1,4,7)
    end
end

# --- this is the nested logit structure when having a job offer:
function get_nests()
    B₁ = [[1,],[2,3],[4,],[5,6],[7,],[8,9]]
    C₁ = [[1,],[2,],[3,],[4,],[5,],[6,],[7,],[8,],[9,]]
    B₂ = [[1,2],[3,4],[5,6]]
    C₂ = [[1,],[2,3],[4,],[5,6],[7,],[8,9]] #<- same as B₁
    B₃ = [[1,2,3]]
    C₃ = [[1,2,3],[4,5,6],[7,8,9]]

    B = (B₁,B₂,B₃)
    C = (C₁,C₂,C₃)
    return (;B,C)
end