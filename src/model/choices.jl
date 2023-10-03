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


# the nesting is:
# program participation choice -> work choice -> care choice (if working)
layers = 1:3 #<- three layers,  program choice, work choice, then care choice
parents = [10:15,16:18,19:19]
children = [fill(nothing,J);[1:1];[2:3];[4:4];[5:6];[7:7];[8:9];[10:11];[12:13];[14:15];[16:18]]
n_nodes = 19

NL = NestedLogitTree(J,n_nodes,layers,parents,children)

# we remove working from the choice set when there is no wage offer (kη==1)
function choice_set!(m::ddc_model,Kη::Int64,Kω::Int64,Kτ::Int64)
    k_inv = CartesianIndices((2,Kη,Kω,Kτ))
    for t in axes(m.choice_set,3), j in 1:m.G.nchoices
        S,A,P,H,F = j_inv(j)
        for k in 1:m.K
            _,kη,_,_ = Tuple(k_inv[k])
            if (kη==1) && (H==1)
                m.choice_set[j,k,t] = false
            end
        end
    end
    for t in axes(m.choice_set,3), k in 1:m.K
        @views extend_choice_set!(m.choice_set[:,k,t],m.G)
    end
end
