using DelimitedFiles
using Printf

function savepars_vec(p,f::String)
    x = pars_inv_full(p)
    writedlm("output/" * f,x)
end
function loadpars_vec(p,f::String)
    x = readdlm("output/" * f)[:]
    p = pars_full(x,p)
end

# functions for writing numbers to tables
form(x) = @sprintf("%0.2f",x)
formse(x) = string("(",@sprintf("%0.2f",x),")")

# a helper function to write a collection of strings into separate columns
function tex_delimit(x)
    str = x[1]
    num_col = length(x)
    for i in 2:num_col
        str *=  "&" * x[i]
    end
    return str
end

function write_estimates_table(p,p2,Kτ)
    # ------ make a table for preferences --------------- #
    file = open("output/tables/preference_ests.tex", "w")
    write(file," & \\multicolumn{6}{c}{Type-Specific Parameters} \\\\ \n")
    write(file,"Type & \$\\alpha_{H}\$ & \$\\alpha_{A}\$ & \$\\alpha_{S}\$ & \$\\alpha_{F}\$ & \$\\alpha_{\\theta}\$ & \$y\$ \\\\ \\cmidrule(r){1-1} \\cmidrule(r){2-7} \n")

    for k in 1:Kτ
        str = "\$k=$k\$ &" * form(p.αH[k]) * "&" * form(p.αA[k]) * "&" * form(p.αS[k]) * " & " * form(p.αF[k]) * " & " * form(p.αθ[k]) * "&" * form(p.wq[k]) * "\\\\ \n"
        write(file,str)
        str = "&" * formse(p2.αH[k]) * "&" * formse(p2.αA[k]) * "&" * formse(p2.αS[k]) * " & " * formse(p2.αF[k]) * " & " * formse(p.αθ[k] * log(p2.αθ[k])) * "&" * formse(p.wq[k] * log(p2.wq[k])) * "\\\\ \n"
        write(file,str)
    end

    write(file,"& \\multicolumn{6}{c}{Global Parameters} \\\\ \n")
    write(file,"& \$\\beta\$ & \$\\sigma_{3}\$ & \$\\sigma_{2}\$ & \$\\sigma_{1}\$ & \$\\alpha_{R}\$ & \$\\alpha_{P}\$ \\\\ \\cmidrule(r){2-7}")
    write(file, "&" * form(p.β) * "&" * form(p.σ[1]) * "&" * form(p.σ[2]) * "&" * form(p.σ[3]) * "&" * form(p.αR) * "&" * form(p.αP) * "\\\\ \n")
    write(file, "&" * formse(p.β * (1-p.β) * logit_inv(p2.β)) * "&" * formse(p.σ[1] * log(p2.σ[1])) * "&" * formse(p.σ[2] * log(p2.σ[2])) * "&" * formse(p.σ[3] * log(p2.σ[3])) * "&" * formse(p2.αR) * "&" * formse(p2.αP) * "\\\\ \n")
    close(file)

    # --------- make a table for wage shock parameters ------- #

    file = open("output/tables/transition_ests.tex", "w")
    write(file," & \\multicolumn{3}{c}{Type-Specific Parameters} \\\\ \n")
    write(file,tex_delimit(["Type","\$\\lambda_0\$","\$\\lambda_1\$","\$\\delta\$"]),"\\\\ \\cmidrule(r){1-1}\\cmidrule(r){2-4} \n")
    for k in 1:Kτ
        write(file,"\$k=$k\$ &",tex_delimit(form.([p.λ₀[k],p.λ₁[k],p.δ[k]])),"\\\\ \n")
        write(file,"&",tex_delimit(formse.([logit_inv(p2.λ₀[k]) * (1-p.λ₀[k]) * p.λ₀[k],logit_inv(p2.λ₁[k]) * (1-p.λ₁[k]) * p.λ₁[k],logit_inv(p2.δ[k]) * (1-p.δ[k]) * p.δ[k]])),"\\\\ \n")
    end
    write(file,"& \\multicolumn{3}{c}{Global Parameters} \\\\ \n")
    write(file,tex_delimit(["","\$\\mu_{o}\$","\$\\sigma_{o}\$","\$\\lambda_R\$"]),"\\\\ \\cmidrule(r){2-4} \n")
    write(file,"&",tex_delimit(form.([p.μₒ,p.σₒ,p.λR])),"\\\\ \n")
    write(file,"&",tex_delimit(formse.([p2.μₒ,p.σₒ * log(p2.σₒ),p2.λR])),"\\\\ \n")

    close(file)

    # ------ make a table for prices ------- #
    # wages: k1,k2,k3,k4,k5,unemp,age
    # childcare: k1,k2,k3,k4,k5,unemp,num kids, age youngest <=5, FTP - control, FTP - treat, CTJF - control, CTJF - treat, MFIP - control, MFIP - treat, MFIP - treat 2 
    file = open("output/tables/price_ests.tex", "w")
    write(file," & Wages & Childcare \\\\ \\cmidrule(r){2-3} \n")
    for k in 1:Kτ
        write(file,"Type $k &",form(p.βw[k]),"&",form(p.βf[k]),"\\\\ \n")
        write(file,"& ",formse(p2.βw[k]),"&",formse(p2.βf[k]),"\\\\ \n")
    end
    # next two covariates are wages only
    write(file,"Unemployment Rate &",form(p.βw[Kτ+1]),"&",form(p.βf[Kτ+1]),  "\\\\ \n ")
    write(file," & ",formse(p2.βw[Kτ+1]),"&",formse(p2.βf[Kτ+1])," \\\\ \n")
    write(file,"Age &",form(p.βw[Kτ+2]),"& - \\\\ \n ")
    write(file," & ",formse(p2.βw[Kτ+2]),"& \\\\ \n")
    # the remaining covariates are for childcare only:
    vnames = ["Num. Kids","Youngest \$\\leq 5\$","FTP Control","FTP Treat","CTJF Control","CTJF Treat","MFIP Control","MFIP Treat","MFIP Incentives"]
    for k in eachindex(vnames)
        write(file,vnames[k],"& - &",form(p.βf[Kτ+1+k]),"\\\\ \n")
        write(file,"& - &",formse(p2.βf[Kτ+1+k]),"\\\\ \n")
    end
    # finally, standard errors
    write(file," Measurement error (std. dev) & ",form(p.σ_W)," &",form(p.σ_PF),"\\\\ \n")
    write(file, " & ",formse(p.σ_W*p2.σ_W)," & ",formse(p.σ_PF * p2.σ_PF),"\\\\ \n")
    close(file)

    # -------- make a table for type selection -------- #
    vnames = ["Const.","High School","Some College","Num Kids"]
    pos = 1
    file = open("output/tables/type_ests.tex","w")
    write(file,["& \$k = $x\$" for x in 2:p.Kτ]...,"\\\\ \n")
    write(file," & \\multicolumn{$(p.Kτ-1)}{c}{SIPP} \\\\ \\cmidrule(r){2-$(p.Kτ)} \n")
    for v in vnames
        write(file,v)
        for k in 1:p.Kτ-1
            write(file," & ",form(p.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        for k in 1:p.Kτ-1
            write(file," & ",formse(p2.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        pos += 1
    end
    vnames = [vnames;"New Applicant"]
    write(file," & \\multicolumn{$(p.Kτ-1)}{c}{FTP} \\\\ \\cmidrule(r){2-$(p.Kτ)} \n")
    for v in vnames
        write(file,v)
        for k in 1:p.Kτ-1
            write(file," & ",form(p.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        for k in 1:p.Kτ-1
            write(file," & ",formse(p2.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        pos += 1
    end
    vnames = [vnames;"New Haven"]
    write(file," & \\multicolumn{$(p.Kτ-1)}{c}{CTJF} \\\\ \\cmidrule(r){2-$(p.Kτ)} \n")
    for v in vnames
        write(file,v)
        for k in 1:p.Kτ-1
            write(file," & ",form(p.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        for k in 1:p.Kτ-1
            write(file," & ",formse(p2.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        pos += 1
    end
    vnames[end] = "Re-Applicant"
    vnames = [vnames;"Anoka";"Dakota"]
    write(file," & \\multicolumn{$(p.Kτ-1)}{c}{MFIP} \\\\ \\cmidrule(r){2-$(p.Kτ)} \n")
    for v in vnames
        write(file,v)
        for k in 1:p.Kτ-1
            write(file," & ",form(p.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        for k in 1:p.Kτ-1
            write(file," & ",formse(p2.βτ[pos,k]))
        end
        write(file,"\\\\ \n")
        pos += 1
    end
    close(file)

end