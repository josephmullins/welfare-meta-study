# code to compute nested logit choice probabilities and derivatives efficiently
using Distributions
# parameters: number of layers, number of partitions in each layer

# layers are indexed from 1, the lowest layer (i.e. nodes in layer 1 have no children)
struct NestedLogitTree
    nchoices::Int64
    nnodes::Int64
    layers::UnitRange{Int64} #<- indexes the number of layers
    parents::Vector{UnitRange{Int64}} #<- one unit range for each layer, indicating who the parents are in that layer
    children::Vector{Union{Nothing,UnitRange{Int64}}} #<- if a parent, store the slice of nodes of your children
    parentmap::Vector{Int64} #<-indicates the parent of each node
end

NestedLogitTree(nchoices,nnodes,layers,parents,children) = NestedLogitTree(nchoices,nnodes,layers,parents,children,get_parentmap(layers,parents,children))


# this function calculates choice probabilities by:
# (1) iterating through the parents in each layer and calculating conditional choice probabilities
# (2) saving inclusive values for use in the layer above
# it assumes that: logP is a vector for storing these (log) conditional choice probabilities
# V is a vector for storing inclusive values.
# IMPORTANT: the first entries in V from 1 to nchoices must contain the utilities from each choice.
function choice_probs!(logP,V,σ,NL::NestedLogitTree) #
    for l in NL.layers
        for p in NL.parents[l]
            eV = 0.
            vmax = -Inf
            @inbounds for j in NL.children[p]
                if V[j]>vmax
                    vmax = V[j]
                end
                logP[j] = V[j] / σ[l] #<- or eV += exp(logP[j]) quicker or slower?
            end
            @inbounds for j in NL.children[p]
                if V[j]>-Inf
                    eV += exp( (V[j] - vmax) / σ[l])
                end
            end
            log_eV = vmax / σ[l] +  log(eV)
            V[p] = σ[l]*log_eV
            # if all the choices result in -Infty, then we'll get an error here
            @inbounds for j in NL.children[p]
                if logP[j]>-Inf
                    logP[j] -= log_eV #<- 
                end
            end
        end
    end
end

function choice_probs!(logP,V,σ,choice_set,NL::NestedLogitTree) #
    for l in NL.layers
        for p in NL.parents[l]
            eV = 0.
            vmax = -Inf
            @inbounds for j in NL.children[p]
                if choice_set[j]
                    if V[j]>vmax
                        vmax = V[j]
                    end
                    logP[j] = V[j] / σ[l] #<- or eV += exp(logP[j]) quicker or slower?
                end
            end
            @inbounds for j in NL.children[p]
                if choice_set[j]
                    eV += exp( (V[j] - vmax) / σ[l])
                end
            end
            log_eV = vmax / σ[l] + log(eV)
            V[p] = σ[l]*log_eV
            # if all the choices result in -Infty, then we'll get an error here
            @inbounds for j in NL.children[p]
                if choice_set[j] && logP[j]>-Inf
                    logP[j] -= log_eV #<- this creates a NaN if any V==0 (->log(V)=-Inf)
                end
            end
        end
    end
end

# this function calculates derivatives with respect to utilities and σ
# logP and V are vectors of choice probabilities and inclusive values which have already been solved
# dV is a matrix that stores the derivative of inclusive values wrt utilities
# dlogP is a matrix that stores the derivative of condition log choice probabilities wrt utilities
# dVdσ and dlogPdσ are similar objects for derivatives wrt dispersion of taste shocks
# as above, the first block of V must contain the identify matrix, as it is du/du' by definition
# NOTE: this function is deprecated because the indexation is transposed relative to get_derivatives_du/dσ (which are preferred)
function get_derivatives!(logP,V,σ,dlogP,dV,dlPdσ,dVdσ,du,NL::NestedLogitTree)
    fill!(dlogP,0.) #<- we might want to think about improving on this
    @views dV[1:NL.nchoices,1:NL.nchoices] .= du #<- 
    fill!(view(dV,NL.nchoices+1:NL.nnodes,:),0.) #<- same goes for here
    fill!(dlPdσ,0.)
    fill!(dVdσ,0.)

    for l in NL.layers
        for p in NL.parents[l]
            # update dV with respect to σ
            dVdσ[p,l] = V[p] / σ[l]

            for j in NL.children[p]

                Pj = exp(logP[j])
                # update dV with respect to σ[l] again
                dVdσ[p,l] += - V[j] * Pj / σ[l]

                # update dlogP and dV (with respect to utilities)
                for i in 1:NL.nchoices #<- in theory we don't need to iterate over all of these.
                    dlogP[j,i] += dV[j,i] / σ[l]
                    for k in NL.children[p]
                        dlogP[k,i] += - Pj * dV[j,i] * 1 / σ[l]
                    end
                    dV[p,i] += Pj * dV[j,i]
                end

                # update dlogP with respect to σ[l] and σ[l2] for l2<l
                dlPdσ[j,l] += - V[j] / σ[l]^2
                for i in NL.children[p]
                    dlPdσ[i,l] += V[j] * Pj / σ[l]^2
                end
                for l2 = 1:(l-1)
                    dlPdσ[j,l2] +=  dVdσ[j,l2] / σ[l]
                    for i in NL.children[p]
                        dlPdσ[i,l2] += - dVdσ[j,l2] * Pj / σ[l]
                    end
                end

                # update dV with respect to σ[l2] for l2<l
                for l2=1:(l-1)
                    dVdσ[p,l2] += Pj * dVdσ[j,l2]
                end
            end
        end
    end
end

# NOTE: this function assumes that dV already contains du' / dθ in the first J columns of dV.
# - it then recursively computes the derivative of the inclusive values up to the final layer, V
function get_derivatives_du!(logP,σ,dlogP,dV,choice_set,NL::NestedLogitTree)
    fill!(dlogP,0.) #<- we might want to think about improving on this
    fill!(view(dV,:,NL.nchoices+1:NL.nnodes),0.) #<- same goes for here

    for l in NL.layers
        for p in NL.parents[l]

            for j in NL.children[p]
                if choice_set[j]
                    Pj = exp(logP[j])
                    # update dlogP and dV (with respect to dθ)
                    for i in axes(dV,1)
                        dlogP[i,j] += dV[i,j] / σ[l]
                        for k in NL.children[p]
                            if choice_set[k]
                                dlogP[i,k] += - Pj * dV[i,j] * 1 / σ[l]
                            end
                        end
                        dV[i,p] += Pj * dV[i,j]
                    end
                end
            end
        end
    end
end

function get_derivatives_dσ!(logP,V,σ,dlPdσ,dVdσ,choice_set,NL::NestedLogitTree,reset=true)
    if reset
        fill!(dlPdσ,0.) #<-?? always reset, no?
        fill!(dVdσ,0.)
    end

    for l in NL.layers
        for p in NL.parents[l]
            # update dV with respect to σ
            if choice_set[p]
                dVdσ[l,p] = V[p] / σ[l]

                for j in NL.children[p]
                    if choice_set[j]

                        Pj = exp(logP[j])
                        # update dV with respect to σ[l] again
                        dVdσ[l,p] += - V[j] * Pj / σ[l]

                        # update dlogP with respect to σ[l] and σ[l2] for l2<l
                        dlPdσ[l,j] += - V[j] / σ[l]^2
                        for i in NL.children[p]
                            if choice_set[i]
                                dlPdσ[l,i] += V[j] * Pj / σ[l]^2
                            end
                        end
                        for l2 in NL.layers
                            dlPdσ[l2,j] +=  dVdσ[l2,j] / σ[l]
                            for i in NL.children[p]
                                if choice_set[i]
                                    dlPdσ[l2,i] += - dVdσ[l2,j] * Pj / σ[l]
                                end
                            end
                        end

                        # update dV with respect to σ[l2] for l2<l (?)
                        for l2 in NL.layers #1:(l-1)
                            dVdσ[l2,p] += Pj * dVdσ[l2,j]
                        end
                    end
                end
            end
        end
    end
end




# function to get total choice probabilities from the logP
# for evaluating choice probabilities it seems like this would be useful!
function reduce_choice_probs!(P,logP,choice_set,NL::NestedLogitTree)
    for p in reverse(eachindex(NL.children))
        if choice_set[p]
            if !isnothing(NL.children[p]) #<- if the node has children, add the conditional log choice probability of this parent to the child
                for c in NL.children[p]
                    if choice_set[c]
                        #println(p," ",c)
                        P[c] = logP[c] + P[p]
                    end
                end
            else #<- if the node is has no children, it is final, and we exponentiate to get the overall probability
                P[p] = exp(P[p]) #<-
            end
        end
    end 
    return nothing
end
function reduce_choice_probs(logP,choice_set,NL::NestedLogitTree)
    P = zeros(NL.nnodes)
    reduce_choice_probs!(P,logP,choice_set,NL)
end

# reduce the choice probabilities in-place, in log form
function reduce_choice_probs!(logP,choice_set,NL::NestedLogitTree)
    for p in reverse(eachindex(NL.children))
        if choice_set[p]
            if !isnothing(NL.children[p])
                for c in NL.children[p]
                    if choice_set[c]
                        #println(p," ",c)
                        logP[c] += logP[p]
                    end
                end
            end
        end
    end 
end

function draw(P,NL::NestedLogitTree)
    choice = zeros(Int64,NL.layers[end])
    node = NL.nnodes #<- the beginning node is always the last entry

    for l in reverse(NL.layers)
        j = rand(Categorical(P[NL.children[node]]))
        c = NL.children[node][j]
        choice[l] = c
        node = c
    end
    return choice
end


# this function assumes choice is a vector with the node choice in each layer has the node for each layer corresponding to the choice
function log_likelihood(choice,logP,NL::NestedLogitTree,weight = 1.)
    ll = 0.
    for l in NL.layers
        ll += weight*logP[choice[l]]
    end
    return ll
end

function log_likelihood(j::Int64,logP,NL::NestedLogitTree,weight = 1.)
    ll = 0.
    for l in NL.layers
        ll += weight*logP[j]
        j = NL.parentmap[j] #<- move up to the node in the next layer
    end
    return ll
end


# this function does the same as above, but also saves a derivative which it adds to g in place
function log_likelihood(choice,g,logP,dlogP,NL::NestedLogitTree,weight = 1.)
    ll = 0.
    for l in NL.layers
        ll += weight*logP[choice[l]]
        for p in eachindex(g)
            g[p] += weight*dlogP[p,choice[l]]
        end
    end
    return ll
end

function log_likelihood(j::Int64,g,logP,dlogP,NL::NestedLogitTree,weight = 1.)
    ll = 0.
    for l in NL.layers
        ll += weight*logP[j]
        for p in eachindex(g)
            g[p] += weight*dlogP[p,j]
        end
        j = NL.parentmap[j]
    end
    return ll
end


function get_parentmap(layers,parents,children)
    parentmap = zeros(Int64,length(children))
    for l in layers
        for p in parents[l]
            parentmap[children[p]] .= p
        end
    end
    return parentmap
end

get_parentmap(NL::NestedLogitTree) = get_parentmap(NL.layers,NL.parents,NL.children)


function get_parent_choices(choice,NL)
    choices = zeros(Int64,length(NL.layers),length(choice))
    parentmap = get_parentmap(NL)
    # calculate the 
    for n in eachindex(choice)
        node = choice[n]
        choices[1,n] = node
        for l in NL.layers[2:end]
            node = parentmap[node]
            choices[l,n] = node
        end
    end
    return choices
end
# log-likelihood of all choices
function loglike(choices,V,σ,logP,NL::NestedLogitTree) #!?
    ll = 0.
    for n in axes(V,2)
        @views choice_probs!(logP[:,n],V[:,n],σ,NL)
        @views ll += log_likelihood(choices[:,n],logP[:,n],NL)
    end
    return ll
end

# recursively define a node as being in the choice set if one of its children are in the choice set
function extend_choice_set!(choice_set,NL::NestedLogitTree)
    for l in NL.layers
        for p in NL.parents[l]
            choice_set[p] = false
            for j in NL.children[p]
                choice_set[p] |= choice_set[j]
            end
        end
    end
end