include("./float_functions.jl")
include("./helpers.jl")
include("./interval_functions.jl")
include("./proof_functions.jl")

using RadiiPolynomial, GLMakie, JLD2

import ApproxFun.Fun

# Setting up the equation parameters
σ::Float64 = 10;
r::Float64 = 2.2;

# Setting up the numerics parameters
N_manif_compute = [18,11];  # Number of Taylor coefficients for the computation of the manifold parameterization
N_cheb_compute = 20; # Number of Chebyshev coefficients for the computation of the connecting orbit

N_manif_proof = [30,20];  # Number of Taylor coefficients for the computation of the manifold parameterization
N_cheb_proof = 500; # Number of Chebyshev coefficients for the proof of the connecting orbit

ν::Float64 = 1.05
weights::Vector{Float64} = [σ,1,1,σ,1,σ,1]
r_star::Float64 = 1e-5

manifold_range::Matrix{Float64} = [-1 1; -1 1];  # How long in time we grow the manifolds
manifold_num_points::Vector{Int64} = [1000,1000]; # How many points in time are we using

vector_length::Vector{Float64} = [1, 1];    # Length of eigenvectors for parameterization method

if r <= 2 || r >= sqrt(5)
    error("Chosen r does not satisfy the requirements")
end

# Two cycle of the logistic map where n₋ < n₊
n₋ = (r+2 - sqrt(r^2 -4))/(2*r);
n₊ = (r+2 + sqrt(r^2 -4))/(2*r);

println("n₋ = " * string(n₋)  * "\nn₊ = " * string(n₊));

# Evaluating the derivative of the logistic map at n₋ and n₊
n₋_prime = logistic_prime(n₋, r);
n₊_prime = logistic_prime(n₊, r);    

println("n₋_prime = " * string(n₋_prime) * "\nn₊_prime = " * string(n₊_prime));
println();

#Unstable manifold
equilibrium₁ = [n₋; 0; n₊; 0];

A₁ = Df(equilibrium₁, σ, r);

λ₁ = sqrt(σ^2 * (1 - sqrt(n₋_prime*n₊_prime)));
ξ₁ = [(-sqrt(Complex(n₊_prime))/(sqrt(Complex(n₋_prime))*λ₁)).re; -sqrt(n₊_prime/n₋_prime); 1/λ₁; 1];
ξ₁ = (vector_length[1] / norm(ξ₁)) * ξ₁;

println("λ₁ = " * string(λ₁) * "\nξ₁ = " * string(ξ₁));
println("||A₁*ξ₁ - λ₁*ξ₁|| = " * string(norm(A₁*ξ₁ - λ₁*ξ₁)));
println();

λ₂ = sqrt(σ^2 * (1 + sqrt(n₋_prime*n₊_prime)));
ξ₂ = [(sqrt(Complex(n₊_prime))/(sqrt(Complex(n₋_prime))*λ₂)).re; sqrt(n₊_prime/n₋_prime); 1/λ₂; 1];
ξ₂ = (vector_length[2] / norm(ξ₂)) * ξ₂;

println("λ₂ = " * string(λ₂) * "\nξ₂ = " * string(ξ₂));
println("||A₁*ξ₂ - λ₂*ξ₂|| = " * string(norm(A₁*ξ₂ - λ₂*ξ₂)));
println();

# Stable manifold
equilibrium₂ = [n₊; 0; n₋; 0];

A₂ = Df(equilibrium₂, σ, r);

λ₃ = -sqrt(σ^2*(1-sqrt(n₋_prime*n₊_prime)));
ξ₃ = [(-sqrt(Complex(n₋_prime))/(sqrt(complex(n₊_prime))*λ₃)).re; -sqrt(n₋_prime/n₊_prime); 1/λ₃; 1];
ξ₃ = (vector_length[1] / norm(ξ₃)) * ξ₃;

println("λ₃ = " * string(λ₃) * "\nξ₃ = " * string(ξ₃));
println("||A₂*ξ₃ - λ₃*ξ₃|| = " * string(norm(A₂*ξ₃ - λ₃*ξ₃)));
println();

λ₄ = -sqrt(σ^2*(1+sqrt(n₋_prime*n₊_prime)));
ξ₄ = [(sqrt(Complex(n₋_prime))/(sqrt(Complex(n₊_prime))*λ₄)).re; sqrt(n₋_prime/n₊_prime); 1/λ₄; 1];
ξ₄ = (vector_length[2] / norm(ξ₄)) * ξ₄;

println("λ₄ = " * string(λ₄) * "\nξ₄ = " * string(ξ₄));
println("||A₂*ξ₄ - λ₄*ξ₄|| = " * string(norm(A₂*ξ₄ - λ₄*ξ₄)));
println();



S_manif_compute = (Taylor(N_manif_compute[1]) ⊗ Taylor(N_manif_compute[2]))^4; # 2-index Taylor sequence space
S_manif_proof = (Taylor(N_manif_proof[1]) ⊗ Taylor(N_manif_proof[2]))^4; # 2-index Taylor sequence space

a = zeros(S_manif_compute);

a, = newton!((F, DF, a) -> (F_manif!(F, a, N_manif_compute, σ, r, equilibrium₂, λ₃, λ₄, ξ₃, ξ₄), DF_manif!(DF, a, N_manif_compute, σ, r, equilibrium₂, λ₃, λ₄, ξ₃, ξ₄)), a, tol = 1e-15)

save_object("approximate_logistic.jld2", a.coefficients[:])

a₁ = component(a,1)
a₂ = component(a,2)
a₃ = component(a,3)
a₄ = component(a,4)

println("a₁(N₁,0) = " * string(a₁[(N_manif_compute[1],0)]) * ", a₁(0,N₂) = " * string(a₁[(0,N_manif_compute[2])]))
println("a₂(N₁,0) = " * string(a₂[(N_manif_compute[1],0)]) * ", a₂(0,N₂) = " * string(a₂[(0,N_manif_compute[2])]))
println("a₃(N₁,0) = " * string(a₃[(N_manif_compute[1],0)]) * ", a₃(0,N₂) = " * string(a₃[(0,N_manif_compute[2])]))
println("a₄(N₁,0) = " * string(a₄[(N_manif_compute[1],0)]) * ", a₄(0,N₂) = " * string(a₄[(0,N_manif_compute[2])]))
println()


manifold_data = generate_manifold_data(a, manifold_range, manifold_num_points)
reflected_manifold_data = reflection_data(manifold_data)

distance, location = distance_from_fixed_points(manifold_data)
println("Minkowksi distance between manifold and fixed points of R = " * string(distance) * " attained at point x = " * string(location))
println()


θ = get_theta(a, manifold_range, manifold_num_points, location)

θ,L = candidate_finder([θ[1],θ[2],0], a, σ, r)

println("L = " * string(L))
println("θ = " * string(θ))
println()

#Numerically integrating the candidate solution
numerical_orbit_data, numerical_orbit_time = integrate_point(a(θ[1],θ[2]), σ, r, (L,0.0))

S_orbit_compute = ParameterSpace() × ParameterSpace()^2 × Chebyshev(N_cheb_compute)^4
S_orbit_proof = ParameterSpace() × ParameterSpace()^2 × Chebyshev(N_cheb_proof)^4
X = zeros(S_orbit_compute)

chebyshev_multiplier = [1;1/2 * ones(N_cheb_compute,1)]

X.coefficients[:] = [L/2;θ;
    chebyshev_multiplier .* Fun(t -> interpolation(t, numerical_orbit_data[1,:], collect(LinRange(-1,1, length(numerical_orbit_data[1,:])))), N_cheb_compute+1).coefficients;
    chebyshev_multiplier .* Fun(t -> interpolation(t, numerical_orbit_data[2,:], collect(LinRange(-1,1, length(numerical_orbit_data[1,:])))), N_cheb_compute+1).coefficients;
    chebyshev_multiplier .* Fun(t -> interpolation(t, numerical_orbit_data[3,:], collect(LinRange(-1,1, length(numerical_orbit_data[1,:])))), N_cheb_compute+1).coefficients;
    chebyshev_multiplier .* Fun(t -> interpolation(t, numerical_orbit_data[4,:], collect(LinRange(-1,1, length(numerical_orbit_data[1,:])))), N_cheb_compute+1).coefficients;
    ]

X, = newton!((F, DF, X) -> (F_orbit!(F, X, a, N_cheb_compute, σ, r), DF_orbit!(DF, X, a, N_cheb_compute, σ, r)), X, tol = 1e-15)

#save_object("orbit.jld2", X.coefficients[:])

L = component(X,1)[1]
θ = component(X,2)[1:2]

u₁ = component(component(X,3),1)
u₂ = component(component(X,3),2)
u₃ = component(component(X,3),3)
u₄ = component(component(X,3),4)

println("L = " * string(L))
println("θ = " * string(θ))
println("u₁(N_cheb) = " * string(u₁[N_cheb_compute]))
println("u₂(N_cheb) = " * string(u₂[N_cheb_compute]))
println("u₃(N_cheb) = " * string(u₃[N_cheb_compute]))
println("u₄(N_cheb) = " * string(u₄[N_cheb_compute]))
println()


connecting_orbit_data, connecting_orbit_time = orbit_sol_to_data(X, 5)

# Integrating possible intersection point
#connecting_orbit_data, connecting_orbit_time = integrate_point(a(θ[1],θ[2]), σ, r, (L,0.0))
reflected_connecting_orbit_data = reflection_data(connecting_orbit_data)

# Applying the conjugacy to get values inside the manifold
manifold_orbit_data, manifold_orbit_time = integrate_conjugacy_point(θ, a, λ₃, λ₄, (2*L,1.5))
reflected_manifold_orbit_data = reflection_data(manifold_orbit_data)

#=
a = project(a, S_manif_proof)
X = project(X, S_orbit_proof)

println("Beginning Manifold Proof")
r_min, r_max = manif_proof(a, N_manif_proof);

println("Beginning Connecting Orbit Proof")
orbit_proof(X, a, weights, N_cheb_proof, ν, r_min, r_star)
=#

println(N(0.5, λ₃, λ₄, a, X))

# Plotting

# Using GLMakie
figLogistic = Figure()

ODEax = Axis3(figLogistic[1,1],
    title = L"Manifolds and conntecting orbit for $σ = 10$ and $r = 2.2$",
    titlesize = 20,
    xlabel = L"$x_1$",
    xlabelsize = 20,
    ylabel = L"$x_3$",
    ylabelsize = 20,
    zlabel = L"$x_2$",
    zlabelsize = 20   
)


# Stable manifold
stable = GLMakie.surface!(ODEax,
    reshape(manifold_data[1,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(manifold_data[3,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(manifold_data[2,:], manifold_num_points[2], manifold_num_points[1]),
    color = reshape(manifold_data[4,:], manifold_num_points[2], manifold_num_points[1]),
    colorrange = (-4,4),
    transparency = true,
)

# Unstable manifold
unstable = GLMakie.surface!(ODEax,
    reshape(reflected_manifold_data[1,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(reflected_manifold_data[3,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(reflected_manifold_data[2,:], manifold_num_points[2], manifold_num_points[1]),
    color = reshape(reflected_manifold_data[4,:], manifold_num_points[2], manifold_num_points[1]),
    colorrange = (-4,4),
    transparency = true,
)


# Connecting orbit

orbit = GLMakie.lines!(ODEax,
    [reverse(reflected_manifold_orbit_data[1,:]); reverse(reflected_connecting_orbit_data[1,:]);connecting_orbit_data[1,:]; manifold_orbit_data[1,:]],
    [reverse(reflected_manifold_orbit_data[3,:]); reverse(reflected_connecting_orbit_data[3,:]);connecting_orbit_data[3,:]; manifold_orbit_data[3,:]],
    [reverse(reflected_manifold_orbit_data[2,:]); reverse(reflected_connecting_orbit_data[2,:]);connecting_orbit_data[2,:]; manifold_orbit_data[2,:]],
    color = [reverse(reflected_manifold_orbit_data[4,:]); reverse(reflected_connecting_orbit_data[4,:]);connecting_orbit_data[4,:]; manifold_orbit_data[4,:]],
    colorrange = (-4,4),
    label = "Connecting Orbit"
)


fp1 = GLMakie.scatter!(ODEax,
    n₊,
    n₋,
    0,
    color = :red,
    markersize = 15,
    label = "Fixed Points"
)

fp2 = GLMakie.scatter!(ODEax,
    n₋,
    n₊,
    0,
    color = :red,
    markersize = 15
)

Colorbar(figLogistic[1,2], limits = (-4,4), label = L"x_4", labelsize = 20)
axislegend()


IDEax = Axis(figLogistic[1,3],
    title = L"Two-cycle of IDE for $σ = 10$ and $r = 2.2$",
    titlesize = 20,
    xlabel = L"$t$",
    xlabelsize = 20,
    ylabel = L"$y$",
    ylabelsize = 20,
    limits = (nothing, (0.6, 1.3))  
)

GLMakie.lines!(IDEax,
    [reverse(-manifold_orbit_time); reverse(-connecting_orbit_time); connecting_orbit_time; manifold_orbit_time],
    [reverse(reflected_manifold_orbit_data[1,:]); reverse(reflected_connecting_orbit_data[1,:]);connecting_orbit_data[1,:]; manifold_orbit_data[1,:]],
    color = :red,
    label = L"N"
)

GLMakie.lines!(IDEax,
    [reverse(-manifold_orbit_time); reverse(-connecting_orbit_time); connecting_orbit_time; manifold_orbit_time],
    [reverse(reflected_manifold_orbit_data[3,:]); reverse(reflected_connecting_orbit_data[3,:]);connecting_orbit_data[3,:]; manifold_orbit_data[3,:]],
    color = :blue,
    label = L"M"
)


GLMakie.lines!(IDEax,
    collect(LinRange(-1.5,1.5,100)),
    n₋*ones(100,1)[:],
    color = :black,
    label = L"n_-"
)

GLMakie.lines!(IDEax,
    collect(LinRange(-1.5,1.5,100)),
    n₊*ones(100,1)[:],
    color = :black,
    label = L"n_+"
)




axislegend()




display(GLMakie.Screen(),figLogistic)




