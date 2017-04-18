type GMM
	n::Int # number of gaussians
    d::Int # dimension of data
    α::Vector{Float64} # weights
    µ::Vector{Array{Float64,1}} # (n,d) array of mean vectors
    Σ::Vector{Array{Float64,2}} # n (d x d) covariance matrices
end

function GMM(n::Int, d::Int;
            α::Vector{Float64}=ones(Float64, n)/n,
            µ::Vector{Array{Float64,1}}=repeat(zeros(Float64, d); outer=[n]),
            Σ::Vector{Array{Float64,2}}=repeat(eye(d); outer=[n])
            )
    GMM(n,d,α,µ,Σ)
end


function Distributions.MixtureModel(gmm::GMM)
    Distributions.MixtureModel(FullNormal, zip(gmm.µ,gmm.Σ), gmm.α)
end

function fit_em!(gmm:GMM, x::Vector{Vector{Float64}}, w::Vector{Float64})
    loglikelihood = 0
    while true # maybe switch to fixed max_iterations?
        # TODO write EM algorithm to update gmm α, μ, Σ

        if abs(new_loglikelihood - loglikelihood) < THRESHOLD
            break
        end

        loglikelihood = new_loglikelihood
    end
end