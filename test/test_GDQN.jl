push!(LOAD_PATH, ".")
include(joinpath("..", "src", "DRL.jl"))
using DRL
importall POMDPModels
using MXNet
using DiscreteValueIteration
using POMDPToolbox
using GenerativeModels
using Distributions
using PDMats
using DataFrames

# stuff to make things work
importall POMDPs

gw = GridWorld(); #GridWorld(sx=20,sy=1,rs=[GridWorldState(3,1)],rv=[5.],penalty=-10.0, tp=1.0)
ip = InvertedPendulum()
iterator(ipa::POMDPModels.InvertedPendulumActions) = ipa.actions

sim = RolloutSimulator(max_steps=100)

# define initial_state sampling with a distribution
function GenerativeModels.initial_state(gw::GridWorld, s0_dist::Sampleable, rng::AbstractRNG)
    s_vec = rand(s0_dist)
    x = Int(min(max(round(s_vec[1]),1), 1)*gw.size_x)
    y = Int(min(max(round(s_vec[2]),1), 1)*gw.size_y)
    GridWorldState(x,y)
end

# define initial_state sampling with a distribution
function GenerativeModels.initial_state(ip::InvertedPendulum, s0_dist::Sampleable, rng::AbstractRNG)
    s_vec = rand(s0_dist)
    if abs(s_vec[1]) > pi/2
        s_vec[1] = pi*rand(rng) - pi/2
    end
    (s_vec[1], s_vec[2])
end

#GDQN
println("Testing with GDQN")
gw_dist = Nullable(rl.GMM(1,2,[1.],[[4.,1.]],[PDMat([3. 0.; 0. 0.001])]))
ip_dist = Nullable(rl.GMM(1,2,[1.],[[0.,0.]],[PDMat([0.5 0.; 0. 0.5])]))

gdqn = rl.GDQN(max_steps=50, checkpoint_interval=25, num_epochs=750, target_refresh_interval=100)
gdqnpol = rl.solve(gdqn, gw, s0_dist=Nullable{rl.GMM}(gw_dist) )


for s in iterator(states(gw))
    a = action(gdqnpol, s)
    println("state $(s.x), action $(a)")
end
r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, gw, gdqnpol, initial_state(gw, RandomDevice()))
end
println("Avg total reward $(r_total/N_sim)")

df = DataFrame(gdqn.stats);
writetable("testGDQN.csv", df);

println("Testing with DVI")
dvi = ValueIterationSolver()
dvipol = solve(dvi,gw,verbose=true)
# for s in iterator(states(gw))
#     a = action(pol, s)
#     println("state $(s.x), action $(a)")
# end

r_total = 0
N_sim = 50
for i in 1:N_sim
    r_total += simulate(sim, gw, dvipol, initial_state(gw, RandomDevice()))
end
println("Avg total reward $(r_total/N_sim)")


for s in iterator(states(gw))
    a = action(dvipol, s)
    a1 = action(gdqnpol, s)

    if isterminal(gw, s)
        continue
    end

    if a1 != a
        println("actions differ at state $(s), dqn $(a1) vs dvi $(a)")
    end
end
