###########################################################
# This file contains many helper functions that main.jl   #
# uses along its computations. These functions are simple #
# but would clutter the presentation, and so are defined  #
# outside the main.jl file.                               #
###########################################################

using RadiiPolynomial, DifferentialEquations

# Implementation of the logistic map
function logistic(x::Float64, ρ::Float64)
    return (1+ρ)*x - ρ*x^2;
end

# Implementation of the derivative of the logistic map
function logistic_prime(x::Float64, ρ::Float64)
    return 1+ρ - 2*ρ*x;
end

# Implementation of Df
function Df(equilibrium::Vector{Float64}, σ::Float64, ρ::Float64)
    return [
        0       1       0       0
        σ^2    0       -σ^2*logistic_prime(equilibrium[3], ρ)     0
        0       0       0       1
        -σ^2*logistic_prime(equilibrium[1], ρ)     0       σ^2        0
    ]
end

# Generating the manifold from the parameterization
function generate_manifold_data(a::Sequence, range::Matrix{Float64}, num_points::Vector{Int})
    data = zeros(4, (num_points[1]*num_points[2]))

    for i in 0:num_points[1]-1
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            data[:,1 + i*num_points[2] + j] = a(θ₁, θ₂)
        end
    end

    return data
end

# Computing the Minkowski distance between a data set and Π_R
function distance_from_fixed_points(data)
    min_distance = Inf
    min_distance_location = (data[:,1])

    reflection = [
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 -1 0 0
    ]

    for i in 1:length(data[1,:])
        x = data[:,i]

        reflected_x = reflection * x

        if 0.5*norm(x-reflected_x) < min_distance
            min_distance = 0.5*norm(x-reflected_x)
            min_distance_location = x
        end
        
    end

    return min_distance, min_distance_location
end

# Integrating a single point along a vector field similar to f
function integrate_point(x₀::Vector{Float64}, σ::Float64, ρ::Float64, Tspan::Tuple{Float64, Float64})
    f(x,p,t) = [
        x[2]
        σ*(x[1] - logistic(x[3], ρ))
        x[4]
        σ*(x[3] - logistic(x[1], ρ))
    ]

    prob = ODEProblem(f, x₀, Tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-14, saveat = 0.001)

    return mapreduce(permutedims, vcat, sol.u)', sol.t
end

# Applying the relfection R to a set of points
function reflection_data(data)
    new_data = copy(data)

    reflection = [
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 -1 0 0
    ]

    for i in 1:length(data[1,:])
        new_data[:,i] = reflection*data[:,i]
    end

    return new_data
end

# Getting the parameters θ that result in the closest approximation to a point by P(θ)
function get_theta(a::Sequence, range::Matrix{Float64}, num_points::Vector{Int}, x::Vector{Float64})
    θ = [0,0]
    min_distance = Inf

    for i in [0,num_points[1]-1]
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            if norm(x₀-x) < min_distance
                min_distance = norm(x₀-x)
                θ = [θ₁, θ₂]
            end
        end
    end

    for i in 0:num_points[1]-1
        for j in [0,num_points[2]-1]
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            if norm(x₀-x) < min_distance
                min_distance = norm(x₀-x)
                θ = [θ₁, θ₂]
            end
        end
    end    

    return θ
end

# Numerical integration of a point along the map f
function g(X, a::Sequence, σ::Float64, ρ::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    f(x,p,t) = [
        x[2]
        σ*(x[1] - logistic(x[3], ρ))
        x[4]
        σ*(x[3] - logistic(x[1], ρ))
    ]

    prob = ODEProblem(f, a(θ₁, θ₂), (L,0))
    sol = solve(prob, Tsit5(), reltol = 1e-14, save_everystep = false)

    return mapreduce(permutedims, vcat, sol.u)'[:,end]
end

# Numerical approximation of connection operator
function w(X, a::Sequence, σ::Float64, ρ::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    return [
        θ₁^2 + θ₂^2 - 0.95
        g(X, a, σ, ρ)[1] - g(X, a, σ, ρ)[3]
        g(X, a, σ, ρ)[2] + g(X, a, σ, ρ)[4]
    ]
end

# Derivative of numerical approximation of connection operator
function Dw(X, a::Sequence, σ::Float64, ρ::Float64)
    h = 1e-6
    A = zeros(3,3)
    Id = [
        1 0 0
        0 1 0
        0 0 1
    ]

    for i in 1:3
        A[:,i] = 1/(2*h) * (w(X+h*Id[:,i], a, σ, ρ)-w(X-h*Id[:,i], a, σ, ρ))
    end

    return A
end

# Finding a numerical candidate for the connection between the stable manifold and Π_R
function candidate_finder(X, a::Sequence, σ::Float64, ρ::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    X, = newton(X -> (w(X, a, σ, ρ), Dw(X, a, σ, ρ)), X)

    return X[1:2],X[3]
end

# Implementation of a basic linear interpolation
function interpolation(t::Float64, space_data::Vector{Float64}, time_data::Vector{Float64})
    in_range = false
    x = 0
    for i in 1:length(time_data)-1
        if time_data[i] <= t && t <= time_data[i+1]
            # Linear Interpolation
            x = space_data[i] + (t - time_data[i])*(space_data[i+1] - space_data[i])/(time_data[i+1] - time_data[i])
            in_range = true
        end
    end

    if in_range
        return x
    else
        return C_NULL
    end
end

# Implementation of Ψ
function psi(u::Sequence, k::Int, ν::Float64, m::Int)
    N = order(u)
    max = 0
    for j in m+1:k+m
        current = 0
        if abs(k-j) <= N
            current += u[abs(k-j)]
        end

        if abs(k+j) <= N
            current += u[abs(k+j)]
        end

        current = abs(current)/(2*ν^j)

        if current > max
            max = current
        end
    end

    return max
end

# Using (a_α) and u to generate the connecting orbit for a a single input of time
function connection(t::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    u = component(X,3)

    if abs(t) < 2*L
        # In the connecting orbit
        if t >= 0
            return Evaluation(-1/L * (abs(t) - L)) * u
        else
            return reflection_data(Evaluation(-1/L * (abs(t) - L)) * u)
        end
    else
        # In the manifolds
        Λ = [
            λ₁ 0
            0 λ₂
        ]
        θ_end = exp(Λ*(abs(t) - 2*L))*θ

        if t >= 0
            return a(θ_end[1], θ_end[2])
        else
            return reflection_data(a(θ_end[1], θ_end[2]))
        end
    end
end

# Using (a_α) and u to generate the connecting orbit for a certain set of inputs of time
function connection_data(time_data::Vector{Float64},λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    num_points = length(time_data)
    space_data = zeros(4, num_points)

    for i in 1:num_points
        space_data[:,i] = connection(time_data[i], λ₁, λ₂, a, X)
    end

    return space_data
end

function N(x::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    return connection(x, λ₁, λ₂, a, X)[1]
end

function M(x::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    return connection(x, λ₁, λ₂, a, X)[3]
end