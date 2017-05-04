using Distributions
using PDMats

typealias PDMatVector{T<:AbstractPDMat{Float64}} Vector{T}


type GMM
	n::Int # number of gaussians
    d::Int # dimension of data
    α::Vector{Float64} # weights
    μ::Vector{Array{Float64,1}} # (n,d) array of mean vectors
    Σ::PDMatVector # n (d x d) covariance matrices
end

function GMM(n::Int, d::Int;
            α::Vector{Float64}=ones(Float64, n)/n,
            μ::Vector{Array{Float64,1}}=repeat([zeros(Float64, d)]; outer=[n]),
            Σ::PDMatVector=repeat([PDMat(eye(d))]; outer=[n])
            )
    GMM(n,d,α,μ,Σ)
end


function Distributions.MixtureModel(gmm::GMM)
    Distributions.MixtureModel(FullNormal, collect(zip(gmm.μ, gmm.Σ)), gmm.α)
end

function fit_em!(gmm::GMM, x::Vector{Vector{Float64}}, w::Vector{Float64}; verbose::Bool=false, threshold::Float64=0.01)
    m = length(x)
    n = gmm.n
    if verbose
        println("GMM: Fitting mixture of $(n) gaussians using EM")
    end

    sum_w = sum(w)

    loglikelihood = 0
    iter = 1

    while true # maybe switch to fixed max_iterations?
        α = gmm.α
        μ = gmm.μ
        Σ = gmm.Σ
        # E step
        # find p(x_i|z_i = j ; µ, Σ, α). TODO: vectorize this
        pij = zeros(m, n)
        for i in 1:m
            for j in 1:n
                z = x[i] - μ[j]
                pij[i,j] = (α[j]*exp(-0.5*invquad(Σ[j],z) - log(2*π) - logdet(Σ[j])))[1]
            end
        end
        p_i = pij*ones(n,1)

        # check for convergence
        new_loglikelihood = sum(w.*log(p_i))

        # print for debug
        if verbose
            println(" → iter $(iter): log likelihood: $(new_loglikelihood)")
        end

        Δ = abs(new_loglikelihood - loglikelihood)
        if Δ < threshold
            break
        end
        loglikelihood = new_loglikelihood



        # M step
        # normalize pij over z
        Qij = pij./p_i

        wQij = w.*Qij
        α_ = zeros(α)
        μ_ = repeat([zeros(Float64, gmm.d)]; outer=[n])
        Σ_ = repeat([0.01*eye(gmm.d,gmm.d)]; outer=[n])
        for j in 1:n
            sum_wQ = sum(wQij[:,j])

            α_[j] = sum_wQ/sum_w

            μ_[j] = sum(wQij[:,j].*x)/sum_wQ

            for i in 1:m
                Σ_[j] += ( wQij[i,j]*(x[i] - μ_[j])*(x[i] - μ_[j])' )/sum_wQ
            end
        end

        gmm.α = α_
        gmm.μ = μ_
        try
            gmm.Σ = collect( PDMat( full(Symmetric(mat)) ) for mat in Σ_ )
        catch ex
            println(x)
            println(w)
            println(Σ_)
            rethrow(ex)
        end
        iter += 1
    end

    if verbose
        println(" ⇨ Fitting complete")
        for j in 1:n
            println("   $(j):\tα: $(gmm.α[j])")
            println("    \tμ: $(gmm.μ[j])")
            println("    \tΣ: $(gmm.Σ[j].mat)")
        end
    end
end