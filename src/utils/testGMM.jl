include("GMM.jl")
using Distributions

d1 = Normal(0,1.0)
d2 = Normal(4,1.0)
d3 = Normal(7,2.0)

mm = MixtureModel([d1,d2,d3],[0.25,0.5,0.25])

x = rand(mm, 1000)
x = collect([xi] for xi in x)
w = ones(1000)

gmm = GMM(3,1,μ=[[1.],[2.],[3.]])

fit_em!(gmm,x,w,verbose=true,threshold=0.1)


y = linspace(-5,15,1000)
w = collect( pdf(mm, yi) for yi in y )
y = collect([yi] for yi in y)

gmm = GMM(3,1,μ=[[1.],[2.],[3.]])

fit_em!(gmm,x,w,verbose=true,threshold=0.1)