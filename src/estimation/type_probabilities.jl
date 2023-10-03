function log_type_prob!(logπτ, g, β, X)
    fill!(g,0.)
    norm = 0.
    Kτ = length(logπτ)
    Kx = length(X)
    for kτ in eachindex(logπτ)
        if kτ==1
            logπτ[kτ] = 0.
            norm += 1.
        else
            @views logπτ[kτ] = dot(β[:,kτ-1],X)
            norm += exp(logπτ[kτ])
        end
    end
    @views logπτ[:] .-= log(norm)
    for kτ in 1:Kτ, kτ2 in 2:Kτ
        pos = ((kτ2-2)*Kx+1):((kτ2-1)*Kx)
        coef = (kτ2==kτ) - exp(logπτ[kτ2])
        g[pos,kτ] .= coef * X
    end
    return nothing
end
function log_type_prob!(logπτ, β, X)
    norm = 0.
    for kτ in eachindex(logπτ)
        if kτ==1
            logπτ[kτ] = 0.
            norm += 1.
        else
            @views logπτ[kτ] = dot(β[:,kτ-1],X)
            norm += exp(logπτ[kτ])
        end
    end
    @views logπτ[:] .-= log(norm)
    return nothing
end
#this has already been written properly.
function log_likelihood_type(G,x,β,source::String,EM::Vector{EM_data},MD,p::pars,data::Vector{likelihood_data},n_idx::Vector{Vector{Int64}},J::Int64)
    KX = size(β,1)
    @views β[:] .= x
    fill!(G,0.)
    g = zeros(KX*(p.Kτ-1),p.Kτ)
    logπτ = zeros(p.Kτ)
    ll = 0.
    for md in MD
        if md.source==source
            K = 2 * p.Kη * md.Kω * p.Kτ
            s_inv = CartesianIndices((J,K))
            k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
            for n in n_idx[md.case_idx]
                if data[n].use
                    ll += log_likelihood_type(G,logπτ,β,g,EM[n],data[n],s_inv,k_inv)
                end
            end
        end
    end
    return -ll
end
function log_likelihood_type(G,logπτ,β,g,em::EM_data,data::likelihood_data,s_inv,k_inv)
    ll = 0
    log_type_prob!(logπτ,g,β,data.X_type)
    for s in nzrange(em.q_s,1)
        s_idx = em.q_s.rowval[s]
        _,k = Tuple(s_inv[s_idx])
        _,_,_,kτ = Tuple(k_inv[k])
        wght = em.q_s.nzval[s]
        ll += logπτ[kτ]*wght
        @views G[:] .-= g[:,kτ]*wght # a minus because we minimizer the negative log likelihood
    end
    return ll
end

function log_type_prob!(logπτ,p::pars,md::model_data,data::likelihood_data)
    β = view(p.βτ,md.type_block,:)
    log_type_prob!(logπτ, β, data.X_type)
end
