using DataFrames
a = DataFrame()
for i in 1:3
    global a
    b = DataFrame(x=rand(2),y=rand(2))
    a = vcat(a,b)
end

@show a 