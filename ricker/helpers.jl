using RadiiPolynomial, DifferentialEquations, TaylorSeries

function ricker(x::Float64, r::Float64)
    return x*exp(r*(1-x));
end

function ricker_prime(x, r::Float64)    # Note, removed type declaration because it causes problems for autodiff.
    return exp(r*(1-x))*(1-r*x);
end

function two_cycle_finder(r)
    X = [0.5, 1.5]

    g(X) = [
        ricker(X[1],r) - X[2]
        ricker(X[2],r) - X[1]
    ]

    Dg(X) = [
        ricker_prime(X[1],r)   -1
        -1                  ricker_prime(X[2],r)
    ]

    X, = newton(X -> (g(X), Dg(X)), X)

    return X
end

function Df(equilibrium::Vector{Float64}, σ::Float64, r::Float64)
    return [
        0       1       0       0
        σ^2    0       -σ^2*ricker_prime(equilibrium[3], r)     0
        0       0       0       1
        -σ^2*ricker_prime(equilibrium[1], r)     0       σ^2        0
    ]
end

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

function generic_project(x::Sequence, N::Int64)
    s = space(x)
    M = order(x)

    if length(M) == 1

        if s == Taylor(order(x))
            return project(x, Taylor(N))
        elseif s == Chebyshev(order(x))
            return project(x, Chebyshev(N))
        end
    
    elseif length(M) == 2

        if s == Taylor(M[1]) ⊗ Taylor(M[2])
            return project(x, Taylor(N) ⊗ Taylor(N))
        elseif s == Chebyshev(M[1]) ⊗ Chebyshev(M[2])
            return project(x, Chebyshev(N) ⊗ Chebyshev(N))
        end

    end

    return C_NULL
end

function exp2dTaylor(a::Sequence)
    N = order(a)

    x, y = set_variables("x y", order = N[1]*N[2])

    t = 0*TaylorN(1)

    for α = 0:N[1]
        for β = 0:N[2]
            t += a[(α, β)] * x^α * y^β
        end
    end

    t = exp(t)

    b = zeros(space(a))

    for α = 0:N[1]
        for β = 0:N[2]
            b[(α, β)] = getcoeff(t, (α, β))
        end
    end

    return b
end

function N(x::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    return connection(x, λ₁, λ₂, a, X)[1]
end

function M(x::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    return connection(x, λ₁, λ₂, a, X)[3]
end