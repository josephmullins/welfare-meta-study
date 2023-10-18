using Optim

function mstep_major(p,EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    N_ = sum(length(n_idx[md.case_idx]) for md in MD)
    x0 = pars_inv(p)
    objective(x) = -log_likelihood_threaded(x,p,EM,MD,data,n_idx) / N_
    res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=iterations))
    return pars(res.minimizer,p)
end

function mstep_major_block(p,fnames::Vector{Symbol},ft::Vector{Int64},EM::Vector{EM_data},MD::Vector{model_data},n_idx,iterations = 40)
    N_ = sum(length(n_idx[md.case_idx]) for md in MD)
    x0 = pars_inv(p,fnames,ft)
    objective(x) = -log_likelihood_threaded(x,p,fnames,ft,EM,MD,data,n_idx) / N_
    res = Optim.optimize(objective,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace = true,iterations=iterations))
    return pars(res.minimizer,p,fnames,ft)
end

function mstep_types!(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
    Kx = (5,6,7,9) #
    sources = ("SIPP","FTP","CTJF","MFIP")
    blocks = (1:5,6:11,12:18,19:27)
    pos = 1
    for s in eachindex(sources)
        source = sources[s]
        block = blocks[s]
        kx = Kx[s]
        nβ = kx * (p.Kτ-1)
        x0 = reshape(p.βτ[block,:],nβ)
        type_obj(x) = -log_likelihood_type(x,kx,source,EM,MD,p,data,n_idx,J)
        res = Optim.optimize(type_obj,x0,LBFGS(),autodiff = :forward,Optim.Options(show_trace=false,iterations=100))
        @views p.βτ[block,:][:] .= res.minimizer
        pos += nβ
    end
end

function mstep_πη!(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
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

function mstep_σ(p,EM::Vector{EM_data},MD::Vector{model_data},data::Vector{likelihood_data},n_idx,J::Int64)
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
    σ_W = sqrt( numer_w / denom_w )
    σ_PF = sqrt( numer_pf / denom_pf )
    return (;p...,σ_W,σ_PF)
end
