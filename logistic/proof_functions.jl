###########################################################
# This file contains the implementations all computable   #
# bounds for the radii polynomial theorem                 #
###########################################################

using IntervalArithmetic

# Implementing the proof for the local stable manifold
function manif_proof(a::Sequence, weights::Vector{Float64}, N::Vector{Int})  
    # Setting some spaces
    ℓ¹ = Ell1(IdentityWeight())    
    higher_space = (Taylor(2*N[1]) ⊗ Taylor(2*N[2]))^4

    # Turning everything into an interval
    a = interval.(a)
    weights = interval.(weights)

    # Defining interval variables
    i0 = interval(0)
    i1 = interval(1)
    i2 = interval(2)
    i4 = interval(4)

    σ = interval(10)
    ρ = @interval(2.2)

    # Rigorously computing eigenvalues and eigenvectors

    # Two cycle of the logistic map where n₋ < n₊
    n₋ = (ρ+i2 - sqrt(ρ^2 -i4))/(i2*ρ);
    n₊ = (ρ+i2 + sqrt(ρ^2 -i4))/(i2*ρ);

    # Evaluating the derivative of the logistic map at n₋ and n₊
    n₋_prime = i1+ρ - i2*ρ*n₋
    n₊_prime = i1+ρ - i2*ρ*n₊
    
    equilibrium = [n₊; i0; n₋; i0];

    λ₁ = -sqrt(σ^2*(i1-sqrt(n₋_prime*n₊_prime)));
    ξ₁ = [(-sqrt(Complex(n₋_prime))/(sqrt(complex(n₊_prime))*λ₁)).re; -sqrt(n₋_prime/n₊_prime); i1/λ₁; i1];
    ξ₁ = i1/i2norm(ξ₁) * ξ₁

    λ₂ = -sqrt(σ^2*(i1+sqrt(n₋_prime*n₊_prime)));
    ξ₂ = [(sqrt(Complex(n₋_prime))/(sqrt(Complex(n₊_prime))*λ₂)).re; sqrt(n₋_prime/n₊_prime); i1/λ₂; i1];
    ξ₂ = i1/i2norm(ξ₂) * ξ₂

    A_dagger = zeros(Interval{Float64}, space(a), space(a))
    DF_manif!(A_dagger, a, N, σ, ρ, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

    A = inv(mid.(A_dagger))
    A = interval.(A)

    F = zeros(Interval{Float64}, higher_space)
    F_manif!(F, project(a, higher_space), 2*N, σ, ρ, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

    F_N = project(F, space(a))

    DΦ = zeros(Interval{Float64}, higher_space, higher_space)
    DΦ!(DΦ, project(a, higher_space), σ, ρ)

    # Y₀ Bound
    V = A * F_N

    Y_0_finite_part = interval.([0,0,0,0])
    for i in 1:4
        Y_0_finite_part[i] = norm(component(V,i), ℓ¹)
    end

    Y_0_tail = interval.([0,0,0,0])
    for i in 1:4
        for α₁ in N[1]+1:2*N[1]
            for α₂ in N[2]+1:2*N[2]
                Y_0_tail[i] += interval(1)/(interval(α₁) * λ₁ + interval(α₂) * λ₂) * abs(component(F,i)[(α₁, α₂)])
            end
        end
    end

    Y_0_bound = Y_0_finite_part + Y_0_tail
    Y_0_bound = weights .* Y_0_bound

    Y₀ = maximum(Y_0_bound)

    if !isguaranteed(Y₀)
        println("Y₀ is not guaranteed")
    end

    Y₀ = interval(sup(Y₀))
    println("Y₀ = " * string(sup(Y₀)))
    
    
    # Z₀ Bound
    B = interval.(project(I, space(a), space(a))) - A*A_dagger

    Z_0_bound = interval.([0,0,0,0])

    for i in 1:4
        for j in 1:4
            Z_0_bound[i] += i1 / weights[j] * opnorm(component(B,i,j), ℓ¹)
        end
    end

    Z_0_bound = weights .* Z_0_bound
    Z₀ = maximum( weights .* Z_0_bound)

    if !isguaranteed(Z₀)
        println("Z₀ is not guaranteed")
    end

    Z₀ = interval(sup(Z₀))
    println("Z₀ = " * string(sup(Z₀)))

    # Z₁ Bound

    Z_1_bound = interval.([0,0,0,0])

    for i in 1:4
        for j in 1:4
            Z_1_bound[i] += i1 / weights[j] * opnorm(component(DΦ,i,j), ℓ¹)
        end
    end

    Z_1_bound = weights .* Z_1_bound
    Z₁ = (i1/(min(abs(interval(N[1]+1)*λ₁), abs(interval(N[2]+1)*λ₂)))) * maximum(Z_1_bound)

    if !isguaranteed(Z₁)
        println("Z₁ is not guaranteed")
    end

    Z₁ = interval(sup(Z₁))
    println("Z₁ = " * string(sup(Z₁)))

    # Z₂ Bound

    Z_2_bound = interval.([0,0,0,0])

    for i in 1:4
        t1 = i1 / weights[3]^2 * opnorm(component(A,i,2), ℓ¹)
        if i == 2
            t1 = max(t1, (i1/(min(abs(interval(N[1]+1)*λ₁), abs(interval(N[2]+1)*λ₂)))))
        end

        t2 = i1 / weights[1]^2 * opnorm(component(A,i,4), ℓ¹)
        if i == 4
            t2 = max(t1, (i1/(min(abs(interval(N[1]+1)*λ₁), abs(interval(N[2]+1)*λ₂)))))
        end

        Z_2_bound[i] = i2*σ^2*ρ*(t1 + t2)
    end

    Z_2_bound = weights .* Z_2_bound
    Z₂ = maximum(Z_2_bound)

    if !isguaranteed(Z₂)
        println("Z₂ is not guaranteed")
    end

    Z₂ = interval(sup(Z₂))
    println("Z₂ = " * string(sup(Z₂)))

    delta = sqrt((i1-Z₀-Z₁)^2 - i4*Z₂*Y₀)
    println("Δ = " * string(delta))

    r_min = sup(interval(0.5) * (i1-Z₀-Z₁ - delta) / Z₂)
    r_max = inf(interval(0.5) * (i1-Z₀-Z₁ + delta) / Z₂)

    println("(r_min, r_max) = " *string((r_min, r_max)))
    println()

    return (r_min, r_max)
end

# Implementing the proof for the projected BVP problem
function orbit_proof(X::Sequence, a::Sequence, weights::Vector{Float64}, N_cheb::Int, ν::Float64, δ::Float64, r_star::Float64)
    #Turning everything into an interval
    X = interval.(X)
    a = interval.(a)
    weights = interval.(weights)
    ν = @interval(1.05)
    δ = interval.(δ)
    r_star = interval.(r_star)
    σ = interval(10)
    ρ = @interval(2.2)

    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    u = component(X,3)

    u₁ = component(u,1)
    u₂ = component(u,2)
    u₃ = component(u,3)
    u₄ = component(u,4)

    a₁ = component(a,1)
    a₂ = component(a,2)
    a₃ = component(a,3)
    a₄ = component(a,4)

    # Defining interval variables
    i0 = interval(0)
    i1 = interval(1)
    i2 = interval(2)
    i4 = interval(4)

    #Setting some spaces and norms
    higher_space = ParameterSpace() × ParameterSpace()^2 × (Chebyshev(2*N_cheb))^4

    ℓᵥ = Ell1(GeometricWeight(ν))

    #Defining Operators

    A_dagger = zeros(Interval{Float64}, space(X), space(X))
    DF_orbit!(A_dagger, X, a, N_cheb, σ, ρ, δ)

    A = inv(mid.(A_dagger))
    A = interval.(A)

    F = zeros(Interval{Float64}, higher_space)
    F_orbit!(F, project(X, higher_space), a, N_cheb, σ, ρ, δ)

    F_N = project(F, space(X))

    # Y_0 Bound
    V = A * F_N

    Y_0_manif_error = interval.([0,0,0,0,0,0,0])

    for i in 1:4
        Y_0_manif_error[1] += δ * (component(component(A,1,3),i)[1,1])
        Y_0_manif_error[2] += δ * (component(component(A,2,3),1,i)[1,1])
        Y_0_manif_error[3] += δ * (component(component(A,2,3),2,i)[1,1])

        Y_0_manif_error[3+1] += δ * norm(Sequence(Chebyshev(N_cheb), component(component(A,3,3),1,i).coefficients[:,1] ), ℓᵥ)
        Y_0_manif_error[3+2] += δ * norm(Sequence(Chebyshev(N_cheb), component(component(A,3,3),2,i).coefficients[:,1] ), ℓᵥ)
        Y_0_manif_error[3+3] += δ * norm(Sequence(Chebyshev(N_cheb), component(component(A,3,3),3,i).coefficients[:,1] ), ℓᵥ)
        Y_0_manif_error[3+4] += δ * norm(Sequence(Chebyshev(N_cheb), component(component(A,3,3),4,i).coefficients[:,1] ), ℓᵥ)
    end

    Y_0_finite_part = interval.([0,0,0,0,0,0,0])
    
    Y_0_finite_part[1] = norm(component(V,1))
    Y_0_finite_part[2] = norm(component(component(V,2),1))
    Y_0_finite_part[3] = norm(component(component(V,2),2))

    for i in 1:4
        Y_0_finite_part[3+i] = norm(component(component(V,3),i), ℓᵥ)
    end

    Y_0_tail = interval.([0,0,0,0,0,0,0])

    for i in 1:4
        for k = N_cheb+1:2*N_cheb
            Y_0_tail[3+i] += interval(1)/interval(k) * abs(component(component(F,3),i)[k]) * ν^interval(k)
        end
    end

    Y_0_bound = Y_0_manif_error + Y_0_finite_part + Y_0_tail
    Y_0_bound = weights .* Y_0_bound

    Y₀ = maximum(Y_0_bound)

    if !isguaranteed(Y₀)
        println("Y₀ is not guaranteed")
    end

    Y₀ = interval(sup(Y₀))
    println("Y₀ = " * string(sup(Y₀)))

    # Z_0 Bound
    B = interval.(project(I, space(X), space(X))) - A*A_dagger

    new_space = ParameterSpace() × ParameterSpace() × ParameterSpace() × Chebyshev(N_cheb) × Chebyshev(N_cheb) × Chebyshev(N_cheb) × Chebyshev(N_cheb)
    B_new = LinearOperator(new_space, new_space, B[:,:])

    spaces = [ℓ∞(), ℓ∞(), ℓ∞(), ℓᵥ, ℓᵥ, ℓᵥ, ℓᵥ]

    Z_0_bound = interval.([0,0,0,0,0,0,0])

    for i = 1:7
        for j = 1:7
            Z_0_bound[i] += interval(1)/weights[j] * opnorm(component(B_new,i,j), spaces[j], spaces[i])
        end
    end

    Z_0_bound = weights .* Z_0_bound

    Z₀ = maximum(Z_0_bound)

    if !isguaranteed(Z₀)
        println("Z₀ is not guaranteed")
    end

    Z₀ = interval(sup(Z₀))
    println("Z₀ = " * string(sup(Z₀)))

    # Z_1 Bound

    # Build ẑ

    ẑ = zeros(Interval{Float64}, space(X))

    # L
    component(ẑ,1)[1] = interval(0)

    # θ
    component(ẑ,2)[1] = i1/(ν^interval(N_cheb+1)) * (i1/weights[3+1] + i1/weights[3+3])
    component(ẑ,2)[2] = i1/(ν^interval(N_cheb+1)) * (i1/weights[3+2] + i1/weights[3+4])

    # u₁
    component(component(ẑ,3),1)[0] = i1/(ν^interval(N_cheb+1)) * i1/(weights[3+1]) + δ*(θ[1]/(i1 - θ[1])^i2)*i1/(i1-θ[2])* i1/(weights[1+1]) + δ*(θ[2]/(i1 - θ[2])^i2)*i1/(i1-θ[1])* i1/(weights[1+2])
    component(component(ẑ,3),1)[N_cheb] = L*i1/(i2*ν^interval(N_cheb+1))* i1/(weights[3+2])

    # u₂
    component(component(ẑ,3),2)[0] = i1/(ν^interval(N_cheb+1)) * i1/(weights[3+2]) + δ*(θ[1]/(i1 - θ[1])^i2)*i1/(i1-θ[2])* i1/(weights[1+1]) + δ*(θ[2]/(i1 - θ[2])^i2)*i1/(i1-θ[1])* i1/(weights[1+2])
    for k in 1:N_cheb
        component(component(ẑ,3),2)[k] = L*i2*σ^2*ρ*(psi(u₃, k+1, ν, N_cheb) + psi(u₃, k-1, ν, N_cheb)) / weights[3+3]
    end
    component(component(ẑ,3),2)[N_cheb] += L*σ^2*((i1/weights[3+1] + (i1+ρ)/weights[3+3]))/(i2*ν^interval(N_cheb+1))

    # u₃
    component(component(ẑ,3),3)[0] = i1/(ν^interval(N_cheb+1)) * i1/(weights[3+3]) + δ*(θ[1]/(i1 - θ[1])^i2)*i1/(i1-θ[2])* i1/(weights[1+1]) + δ*(θ[2]/(i1 - θ[2])^i2)*i1/(i1-θ[1])* i1/(weights[1+2])
    component(component(ẑ,3),3)[N_cheb] = L*i1/(i2*ν^interval(N_cheb+1))* i1/(weights[3+4])

    # u₄
    component(component(ẑ,3),4)[0] = i1/(ν^interval(N_cheb+1)) * i1/(weights[3+4]) + δ*(θ[1]/(i1 - θ[1])^i2)*i1/(i1-θ[2])* i1/(weights[1+1]) + δ*(θ[2]/(i1 - θ[2])^i2)*i1/(i1-θ[1])* i1/(weights[1+2])
    for k in 1:N_cheb
        component(component(ẑ,3),4)[k] = L*i2*σ^2*ρ*(psi(u₁, k+1, ν, N_cheb) + psi(u₁, k-1, ν, N_cheb)) / weights[3+1]
    end
    component(component(ẑ,3),4)[N_cheb] += L*σ^2*((i1/weights[3+3] + (i1+ρ)/weights[3+1]))/(i2*ν^interval(N_cheb+1))

    V = abs.(A) * ẑ

    Z_1_finite_part = interval.([0,0,0,0,0,0,0])

    Z_1_finite_part[1] = component(V,1)[1]
    Z_1_finite_part[2] = component(V,2)[1]
    Z_1_finite_part[3] = component(V,2)[2]

    for i in 1:4
        Z_1_finite_part[3+i] = norm(component(component(V,3),i), ℓᵥ)
    end

    Z_1_tail = interval.([0,0,0,0,0,0,0])

    Z_1_tail[3+1] = norm(u₂, ℓᵥ)/weights[1] + L/weights[3+2]
    Z_1_tail[3+2] = σ^2*((norm(u₁, ℓᵥ) + (interval(1) + ρ) * norm(u₃, ℓᵥ) + ρ*norm(u₃ * u₃, ℓᵥ))/weights[1] + L*(interval(1)/weights[3+1] + (interval(1)+ρ)/weights[3+3] + interval(2)*ρ*norm(u₃, ℓᵥ)/weights[3+3]))
    Z_1_tail[3+3] = norm(u₄, ℓᵥ)/weights[1] + L/weights[3+4]
    Z_1_tail[3+4] = σ^2*((norm(u₃, ℓᵥ) + (interval(1) + ρ) * norm(u₁, ℓᵥ) + ρ*norm(u₁ * u₁, ℓᵥ))/weights[1] + L*(interval(1)/weights[3+3] + (interval(1)+ρ)/weights[3+1] + interval(2)*ρ*norm(u₁, ℓᵥ)/weights[3+1]))

    Z_1_tail = Z_1_tail * ν/(interval(N_cheb)+interval(1))

    Z_1_bound = Z_1_finite_part + Z_1_tail

    Z_1_bound = weights .* Z_1_bound

    Z₁ = maximum(Z_1_bound)

    if !isguaranteed(Z₁)
        println("Z₁ is not guaranteed")
    end

    Z₁ = interval(sup(Z₁))
    println("Z₁ = " * string(sup(Z₁)))

    # Z_2 Bound

    Z_2_finite_part = interval.([0,0,0,0,0,0,0])
    ζ₁ = interval.([0,0,0,0,0,0,0])
    ζ₂ = interval.([0,0,0,0,0,0,0])

    Z_2_finite_part[1] = i2 * (i1/weights[2]^2 + i1/weights[3]^2)
    Z_2_finite_part[2] = i0
    Z_2_finite_part[3] = i0

    c₁ = abs(θ[1] + r_star*interval(-1,1)/weights[1+1])
    c₂ = abs(θ[2] + r_star*interval(-1,1)/weights[1+2])

    for j in 1:4
        ζ₁[3+j] += Evaluation(c₁, c₂) * (Derivative(2,0) * component(a,j))
        ζ₁[3+j] += i2 * (Evaluation(c₁, c₂) * (Derivative(1,1) * component(a,j)))
        ζ₁[3+j] += Evaluation(c₁, c₂) * (Derivative(0,2) * component(a,j))

        ζ₁[3+j] = abs(ζ₁[3+j])

        ζ₂[3+j] += δ*(i2/(interval(1)-c₁)^3)*(interval(1)/(interval(1)-c₂))
        ζ₂[3+j] += i2*δ*(interval(1)/(interval(1)-c₁)^2)*(interval(1)/(interval(1)-c₂)^2)  
        ζ₂[3+j] += δ*(interval(1)/(interval(1)-c₁))*(i2/(interval(1)-c₂)^3)          
    end

    Z_2_finite_part = i1/min(weights[1+1], weights[1+2]) * (ζ₁ + ζ₂)

    Z_2_tail = interval.([0,0,0,0,0,0,0])
    Φ_hat = interval.([0,0,0,0,0,0,0])
    Ψ_hat = interval.([0,0,0,0,0,0,0])

    Φ_hat[3+1] = i1/weights[3+2]
    Φ_hat[3+2] = σ^2 * ((i1/weights[3+1]+ (i1+ρ)/weights[3+3] + i2*norm(u₃, ℓᵥ)/weights[3+3] + r_star/weights[3+3]^2))
    Φ_hat[3+3] = i1/weights[3+4]
    Φ_hat[3+4] = σ^2 * ((i1/weights[3+3]+ (i1+ρ)/weights[3+1] + i2*norm(u₁, ℓᵥ)/weights[3+1] + r_star/weights[3+1]^2))

    Ψ_hat[3+1] = i0
    Ψ_hat[3+2] = i2*σ^2*ρ/weights[3+3]^2
    Ψ_hat[3+3] = i0
    Ψ_hat[3+4] = i2*σ^2*ρ/weights[3+1]^2

    Z_2_tail = i2 * ν * (Φ_hat/weights[1] + L*Ψ_hat)

    Z_2_bound = Z_2_finite_part + Z_2_tail

    A_new = LinearOperator(new_space, new_space, A[:,:])
    A_norms = interval.(zeros(7,7))

    for i = 1:7
        for j = 1:7
            if i == j && i > 3
                A_norms[i,j] = maximum([opnorm(component(A_new, i,j), spaces[j], spaces[i]), i1 / (i2*(interval(N_cheb)+i1))])
            else
                A_norms[i,j] = opnorm(component(A_new, i,j), spaces[j], spaces[i])
            end
        end
    end
    
    norm_A = maximum( (A_norms *( i1 ./ weights)) .* (weights))

    Z_2_bound = norm_A * Z_2_bound

    Z_2_bound = weights .* Z_2_bound

    Z₂ = maximum(Z_2_bound)

    if !isguaranteed(Z₂)
        println("Z₂ is not guaranteed")
    end

    Z₂ = interval(sup(Z₂))
    println("Z₂ = " * string(sup(Z₂)))

    delta = sqrt((i1-Z₀-Z₁)^2 - i4*Z₂*Y₀)
    println("Δ = " * string(delta))

    r_min = sup(interval(0.5) * (i1-Z₀-Z₁ - delta) / Z₂)
    r_max = inf(interval(0.5) * (i1-Z₀-Z₁ + delta) / Z₂)
    println("(r_min, r_max) = " *string((r_min, r_max)))
    println()

    return (r_min, r_max)
end