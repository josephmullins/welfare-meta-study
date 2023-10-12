using Optim

function mstep_major!(p::pars,Gstore::Matrix{Float64},LL::Vector{Float64},M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    N_ = sum(length(n_idx[md.case_idx]) for md in MD)
    function fg!(F,G,x)
        if G !== nothing
            # code to compute gradient here
            # writing the result to the vector G
            ll = log_likelihood_threaded(x,Gstore,LL,M,∂M,EM,MD,p,data,n_idx) / N_
            @views G[:] .= -sum(Gstore,dims=2) / N_
            return -ll
        else
            return -log_likelihood_threaded(x,LL,M,EM,MD,p,data,n_idx) / N_
        end
    end
    x0 = update_inv(p)
    res = Optim.optimize(Optim.only_fg!(fg!),x0,LBFGS(),Optim.Options(show_trace = true,iterations=iterations))
    update!(res.minimizer,p)
end

function mstep_major_block!(p::pars,Gstore::Matrix{Float64},LL::Vector{Float64},block,M::Matrix{ddc_model},∂M::Matrix{ddc_derivative},EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    N_ = sum(length(n_idx[md.case_idx]) for md in MD)
    x0 = update_inv(p)
    function fg!(F,G,x)
        x0[block] .= x
        if G !== nothing
            # code to compute gradient here
            # writing the result to the vector G
            ll = log_likelihood_threaded(x0,Gstore,LL,M,∂M,EM,MD,p,data,n_idx) / N_
            @views G[:] .= -sum(Gstore[block,:],dims=2) / N_
            return -ll
        else
            return -log_likelihood_threaded(x0,LL,M,EM,MD,p,data,n_idx) / N_
        end
    end
    xstart = x0[block]
    res = Optim.optimize(Optim.only_fg!(fg!),xstart,LBFGS(),Optim.Options(show_trace = true,iterations=iterations))
    x0[block] .= res.minimizer
    update!(x0,p)
end


function mstep_types!(p::pars,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
    Kx = (5,6,7,9) #
    sources = ("SIPP","FTP","CTJF","MFIP")
    blocks = (1:5,6:11,12:18,19:27)
    pos = 1
    for s in eachindex(sources)
        source = sources[s]
        block = blocks[s]
        kx = Kx[s]
        nβ = kx * (p.Kτ-1)
        G = zeros(nβ)
        x0 = reshape(p.βτ[block,:],nβ)
        β = view(p.βτ,block,:)
        type_obj(F,G,x) = log_likelihood_type(G,x,β,source,EM,MD,p,data,n_idx,J)
        res = Optim.optimize(Optim.only_fg!(type_obj),x0,LBFGS(),Optim.Options(show_trace=false,iterations=100))

        β[:] .= res.minimizer
        pos += nβ
    end
end

function mstep_πη!(p::pars,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
    denom = zeros(2,1,p.Kτ,3)
    fill!(p.πη,0.)
    for md in MD
        K = 2 * p.Kη * md.Kω * p.Kτ
        s_inv = CartesianIndices((J,K))
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))
        if md.source!="SIPP"
            loc = (md.source=="FTP") + 2(md.source=="CTJF") + 3(md.source=="MFIP")
            for n in n_idx[md.case_idx]
                if data[n].use
                    for s in nzrange(EM[n].q_s,1)
                        s_idx = EM[n].q_s.rowval[s]
                        _,k = Tuple(s_inv[s_idx])
                        kA,kη,_,kτ = Tuple(k_inv[k])
                        wght = EM[n].q_s.nzval[s]
                        p.πη[kA, kη, kτ, loc] += wght
                        denom[kA, 1, kτ, loc] += wght
                    end
                end
            end
        end
    end
    p.πη ./= denom
end

function mstep_σ!(p::pars,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
    denom_w = 0.
    denom_pf = 0.
    numer_w = 0.
    numer_pf = 0.
    for md in MD
        K = 2 * p.Kη * md.Kω * p.Kτ
        s_inv = CartesianIndices((J,K))
        k_inv = CartesianIndices((2,p.Kη,md.Kω,p.Kτ))

        for n in n_idx[md.case_idx]
            if data[n].use
                for t in axes(EM[n].q_s,2)
                    if data[n].wage_valid[t]
                        for s in nzrange(EM[n].q_s,t)
                            s_idx = EM[n].q_s.rowval[s]
                            _,k = Tuple(s_inv[s_idx])
                            _,kη,_,kτ = Tuple(k_inv[k])
                            if kη>1
                                wght = EM[n].q_s.nzval[s]
                                numer_w += wght * (data[n].logW[t] - logwage(p,md,kτ,kη,t))^2
                                denom_w += wght
                            end
                        end
                    end
                    if data[n].log_chcare[t]>0
                        for s in nzrange(EM[n].q_s,t)
                            s_idx = EM[n].q_s.rowval[s]
                            _,k = Tuple(s_inv[s_idx])
                            _,_,_,kτ = Tuple(k_inv[k])
                            wght = EM[n].q_s.nzval[s]
                            numer_pf += wght * (data[n].log_chcare[t] - logpriceF(p,md,kτ,t))^2
                            denom_pf += wght
                        end
                    end
                end
            end
        end
    end
    p.σ_W = sqrt( numer_w / denom_w )
    p.σ_PF = sqrt( numer_pf / denom_pf )
end

function mstep_prices!(p::pars,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx::Vector{Vector{Int64}},J::Int64)
    x0 = update_inv(p)
    Gstore = zeros(length(x0))
    price_block = (5p.Kτ+6):(7p.Kτ+18)
    function fg!(F,G,x)
        x0[price_block] .= x
        ll = prices_log_likelihood(x0,Gstore,EM,MD,p,data,n_idx,J)
        G[:] .= -Gstore[price_block]
        return -ll
    end
    xstart = x0[price_block]
    res = Optim.optimize(Optim.only_fg!(fg!),xstart,LBFGS(),Optim.Options(show_trace=true,iterations=20))
    x0[price_block] .= res.minimizer
    update!(x0,p)
    return nothing
end
