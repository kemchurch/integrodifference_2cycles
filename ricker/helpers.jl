using RadiiPolynomial, DifferentialEquations, TaylorSeries

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

function ricker_prime(x, r::Float64)    # Note, removed type declaration because it causes problems for autodiff.
    return exp(r*(1-x))*(1-r*x);
end

function ricker(x::Float64, r::Float64)
    return x*exp(r*(1-x));
end

function Df(equilibrium::Vector{Float64}, σ::Float64, r::Float64)
    return [
        0       1       0       0
        σ^2    0       -σ^2*ricker_prime(equilibrium[3], r)     0
        0       0       0       1
        -σ^2*ricker_prime(equilibrium[1], r)     0       σ^2        0
    ]
end

function integrate_boundary(a::Sequence, σ::Float64, r::Float64, range::Matrix{Float64}, num_points::Vector{Int}, T::Float64)
    points = zeros(4,1)

    tspan = (0.0,-T)

    f(x,p,t) = [
        x[2]
        σ*(x[1] - ricker(x[3], r))
        x[4]
        σ*(x[3] - ricker(x[1], r))
    ]

    
    for i in [0,num_points[1]-1]
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            prob = ODEProblem(f, x₀, tspan)
            sol = solve(prob, Tsit5(), reltol = 1e-6, saveat = 0.001)

            points = [points;;mapreduce(permutedims, vcat, sol.u)']
        end
    end

    for i in 0:num_points[1]-1
        for j in [0,num_points[2]-1]
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            prob = ODEProblem(f, x₀, tspan)
            sol = solve(prob, Tsit5(), reltol = 1e-6, saveat = 0.001)

            points = [points;;mapreduce(permutedims, vcat, sol.u)']
        end
    end    

    return points[:,2:end]
end

function integrate_point(x₀::Vector{Float64}, σ::Float64, r::Float64, Tspan::Tuple{Float64, Float64})
    f(x,p,t) = [
        x[2]
        σ*(x[1] - ricker(x[3], r))
        x[4]
        σ*(x[3] - ricker(x[1], r))
    ]

    prob = ODEProblem(f, x₀, Tspan)
    sol = solve(prob, Tsit5()#=, reltol = 1e-10=#, saveat = 0.001)

    return mapreduce(permutedims, vcat, sol.u)', sol.t
end

function integrate_conjugacy_point(θ₀::Vector{Float64}, a::Sequence, λ₁::Float64, λ₂::Float64, Tspan::Tuple{Float64, Float64})
    f(θ,p,t) = [
        λ₁*θ[1]
        λ₂*θ[2]
    ]

    prob = ODEProblem(f, θ₀, Tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-14, saveat = 0.001)

    θ_sol = mapreduce(permutedims, vcat, sol.u)'

    x_sol = zeros(4, length(θ_sol[1,:]))

    for i in 1:length(θ_sol[1,:])
        x_sol[:,i] = a(θ_sol[1,i], θ_sol[2,i])
    end

    return x_sol, sol.t
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

function g(X, a::Sequence, σ::Float64, r::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    f(x,p,t) = [
        x[2]
        σ*(x[1] - ricker(x[3], r))
        x[4]
        σ*(x[3] - ricker(x[1], r))
    ]

    prob = ODEProblem(f, a(θ₁, θ₂), (L,0))
    sol = solve(prob, Tsit5(), reltol = 1e-14, save_everystep = false)

    return mapreduce(permutedims, vcat, sol.u)'[:,end]
end

function w(X, a::Sequence, σ::Float64, r::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    return [
        θ₁^2 + θ₂^2 - 0.95
        g(X, a, σ, r)[1] - g(X, a, σ, r)[3]
        g(X, a, σ, r)[2] + g(X, a, σ, r)[4]
    ]
end

function Dw(X, a::Sequence, σ::Float64, r::Float64)
    h = 0.000001
    A = zeros(3,3)
    Id = [
        1 0 0
        0 1 0
        0 0 1
    ]

    for i in 1:3
        A[:,i] = 1/(2*h) * (w(X+h*Id[:,i], a, σ, r)-w(X-h*Id[:,i], a, σ, r))
    end

    return A
end

function candidate_finder(X, a::Sequence, σ::Float64, r::Float64)
    L = X[1]
    θ₁ = X[2]
    θ₂ = X[3]

    X, = newton(X -> (w(X, a, σ, r), Dw(X, a, σ, r)), X)

    return X[1:2],X[3]
end

function orbit_sol_to_data(X::Sequence, num_points::Int)
    L = component(X,1)[1]
    u = component(X,3)

    space_data = zeros(4, num_points)
    time_data = LinRange(0,2*L,num_points)

    for i in 1:num_points
        chebyshev_time = 1/L * (time_data[i] - L)
        space_data[:,num_points + 1 - i] = Evaluation(chebyshev_time) * u
    end

    return space_data, time_data
end

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
    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    uₙ = component(component(X,3),1)
    uₘ = component(component(X,3),3)

    if abs(x) < 2*L
        # In the connecting orbit
        if x >= 0
            return Evaluation(-1/L * (x - L)) * uₙ
        else
            return Evaluation(-1/L * (-x - L)) * uₘ
        end
    else
        # In the manifolds
        f(θ,p,t) = [
            λ₁*θ[1]
            λ₂*θ[2]
        ]

        prob = ODEProblem(f, θ, (0.0, abs(x)-2*L))
        sol = solve(prob, Tsit5(), reltol = 1e-14,)

        θ_end = sol.u[end]

        if x >= 0
            return a(θ_end[1], θ_end[2])[1]
        else
            return a(θ_end[1], θ_end[2])[3]
        end
    end
end

function M(x::Float64, λ₁::Float64, λ₂::Float64, a::Sequence, X::Sequence)
    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    uₙ = component(component(X,3),1)
    uₘ = component(component(X,3),3)

    if abs(x) < 2*L
        # In the connecting orbit
        if x >= 0
            return Evaluation(-1/L * (x - L)) * uₘ
        else
            return Evaluation(-1/L * (-x - L)) * uₙ
        end
    else
        # In the manifolds
        f(θ,p,t) = [
            λ₁*θ[1]
            λ₂*θ[2]
        ]

        prob = ODEProblem(f, θ, (0.0, abs(x)-2*L))
        sol = solve(prob, Tsit5(), reltol = 1e-14,)

        θ_end = sol.u[end]

        if x >= 0
            return a(θ_end[1], θ_end[2])[3]
        else
            return a(θ_end[1], θ_end[2])[1]
        end
    end
end