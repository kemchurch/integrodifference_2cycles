function manif_proof(a::Sequence, N::Vector{Int})  
    # Setting some spaces
    ℓ¹ = Ell1(IdentityWeight())    
    higher_space = (Taylor(2*N[1]) ⊗ Taylor(2*N[2]))^4

    # Turning everything into an interval
    a = interval.(a)

    # Defining interval variables
    i0 = interval(0)
    i1 = interval(1)
    i2 = interval(2)
    i4 = interval(4)

    σ = interval(10)
    r = @interval(2.2)

    # Rigorously computing eigenvalues and eigenvectors

    # Two cycle of the logistic map where n₋ < n₊
    n₋ = (r+i2 - sqrt(r^2 -i4))/(i2*r);
    n₊ = (r+i2 + sqrt(r^2 -i4))/(i2*r);

    # Evaluating the derivative of the logistic map at n₋ and n₊
    n₋_prime = i1+r - i2*r*n₋
    n₊_prime = i1+r - i2*r*n₊
    
    equilibrium = [n₊; i0; n₋; i0];

    λ₁ = -sqrt(σ^2*(i1-sqrt(n₋_prime*n₊_prime)));
    ξ₁ = [(-sqrt(Complex(n₋_prime))/(sqrt(complex(n₊_prime))*λ₁)).re; -sqrt(n₋_prime/n₊_prime); i1/λ₁; i1];
    ξ₁ = i1/i2norm(ξ₁) * ξ₁

    λ₂ = -sqrt(σ^2*(i1+sqrt(n₋_prime*n₊_prime)));
    ξ₂ = [(sqrt(Complex(n₋_prime))/(sqrt(Complex(n₊_prime))*λ₂)).re; sqrt(n₋_prime/n₊_prime); i1/λ₂; i1];
    ξ₂ = i1/i2norm(ξ₂) * ξ₂

    A_dagger = zeros(Interval{Float64}, space(a), space(a))
    DF_manif!(A_dagger, a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

    A = inv(mid.(A_dagger))
    A = interval.(A)

    F = zeros(Interval{Float64}, higher_space)
    F_manif!(F, project(a, higher_space), 2*N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

    F_N = project(F, space(a))

    DΦ = zeros(Interval{Float64}, higher_space, higher_space)
    DΦ!(DΦ, project(a, higher_space), σ, r)

    # Y₀ Bound
    V = A * F_N

    sum1 = interval.([0,0,0,0])
    for i in 1:4
        sum1[i] = norm(component(V,i), ℓ¹)
    end

    sum2 = interval.([0,0,0,0])
    for i in 1:4
        for α₁ in N[1]+1:2*N[1]
            for α₂ in N[2]+1:2*N[2]
                sum2[i] += interval(1)/(interval(α₁) * λ₁ + interval(α₂) * λ₂) * abs(component(F,i)[(α₁, α₂)])
            end
        end
    end

    sum = sum1 + sum2

    Y₀ = maximum(sum)

    if !isguaranteed(Y₀)
        println("Y₀ is not guaranteed")
    end

    Y₀ = interval(sup(Y₀))
    println("Y₀ = " * string(Y₀))
    
    
    # Z₀ Bound
    B = interval.(project(I, space(a), space(a))) - A*A_dagger

    sum = interval.([0,0,0,0])

    for i in 1:4
        for j in 1:4
            sum[i] += opnorm(component(B,i,j), ℓ¹)
        end
    end

    Z₀ = maximum(sum)

    if !isguaranteed(Z₀)
        println("Z₀ is not guaranteed")
    end

    Z₀ = interval(sup(Z₀))
    println("Z₀ = " * string(Z₀))

    # Z₁ Bound

    sum = interval.([0,0,0,0])

    for i in 1:4
        for j in 1:4
            sum[i] += opnorm(component(DΦ,i,j), ℓ¹)
        end
    end

    Z₁ = (interval(1)/(abs(interval(N[1])*λ₁ + interval(N[2])*λ₂))) * maximum(sum)

    if !isguaranteed(Z₁)
        println("Z₁ is not guaranteed")
    end

    Z₁ = interval(sup(Z₁))

    println("Z₁ = " * string(Z₁))

    # Z₂ Bound

    sum = interval.([0,0,0,0])

    for i in 1:4
        sum[i] = interval(2)*opnorm(component(A,i,2), ℓ¹) + interval(2)*opnorm(component(A,i,4), ℓ¹)
    end

    Z₂ = maximum(sum)

    if !isguaranteed(Z₂)
        println("Z₂ is not guaranteed")
    end

    Z₂ = interval(sup(Z₂))
    println("Z₂ = " * string(Z₂))

    delta = sqrt((interval(1)-Z₀-Z₁)^2 - interval(4)*Z₂*Y₀)
    println("Δ = " * string(delta))

    r_min = sup(interval(0.5) * (interval(1)-Z₀-Z₁ - delta))
    r_max = inf(interval(0.5) * (interval(1)-Z₀-Z₁ + delta))

    println("(r_min, r_max) = " *string((r_min, r_max)))
    println()

    return (r_min, r_max)
end

function orbit_proof(X::Sequence, a::Sequence, weights::Vector{Float64}, N_cheb::Int, ν::Float64, δ::Float64, r_star::Float64)
    #Turning everything into an interval
    X = interval.(X)
    a = interval.(a)
    weights = interval.(weights)
    ν = @interval(1.05)
    δ = interval.(δ)
    r_star = interval.(r_star)
    σ = interval(10)
    r = @interval(2.2)

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

    #Setting some spaces and norms
    higher_space = ParameterSpace() × ParameterSpace()^2 × (Chebyshev(2*N_cheb))^4

    ℓᵥ = Ell1(GeometricWeight(ν))

    #Defining Operators

    A_dagger = zeros(Interval{Float64}, space(X), space(X))
    DF_orbit!(A_dagger, X, a, N_cheb, σ, r, δ)

    A = inv(mid.(A_dagger))
    A = interval.(A)

    F = zeros(Interval{Float64}, higher_space)
    F_orbit!(F, project(X, higher_space), a, N_cheb, σ, r, δ)

    F_N = project(F, space(X))

    # Y_0 Bound
    V = A * F_N

    sum1 = interval.([0,0,0,0,0,0,0])
    
    sum1[1] = norm(component(V,1))
    sum1[2] = norm(component(component(V,2),1))
    sum1[3] = norm(component(component(V,2),2))

    for i in 1:4
        sum1[3+i] = norm(component(component(V,3),i), ℓᵥ)
    end

    sum2 = interval.([0,0,0,0,0,0,0])

    for i in 1:4
        for k = N_cheb+1:2*N_cheb
            sum2[3+i] += interval(1)/interval(k) * abs(component(component(F,3),i)[k]) * ν^interval(k)
        end
    end

    sum = sum1 + sum2
    sum = weights .* sum

    Y₀ = maximum(sum)

    if !isguaranteed(Y₀)
        println("Y₀ is not guaranteed")
    end

    Y₀ = interval(sup(Y₀))
    println("Y₀ = " * string(Y₀))

    # Z_0 Bound
    B = interval.(project(I, space(X), space(X))) - A*A_dagger

    new_space = ParameterSpace() × ParameterSpace() × ParameterSpace() × Chebyshev(N_cheb) × Chebyshev(N_cheb) × Chebyshev(N_cheb) × Chebyshev(N_cheb)
    B_new = LinearOperator(new_space, new_space, B[:,:])

    spaces = [ℓ∞(), ℓ∞(), ℓ∞(), ℓᵥ, ℓᵥ, ℓᵥ, ℓᵥ]

    sum = interval.([0,0,0,0,0,0,0])

    for i = 1:7
        for j = 1:7
            sum[i] += interval(1)/weights[j] * opnorm(component(B_new,i,j), spaces[j], spaces[i])
        end
    end

    sum = weights .* sum

    Z₀ = maximum(sum)

    if !isguaranteed(Z₀)
        println("Z₀ is not guaranteed")
    end

    Z₀ = interval(sup(Z₀))
    println("Z₀ = " * string(Z₀))

    # Z_1 Bound

    # Build ẑ

    ẑ = zeros(Interval{Float64}, space(X))

    # L
    component(ẑ,1)[1] = interval(0)

    # θ
    component(ẑ,2)[1] = interval(1)/(ν^interval(N_cheb+1)) * (interval(1)/weights[3+1] + interval(1)/weights[3+3])
    component(ẑ,2)[2] = interval(1)/(ν^interval(N_cheb+1)) * (interval(1)/weights[3+2] + interval(1)/weights[3+4])

    # u₁
    component(component(ẑ,3),1)[0] = interval(1)/(ν^interval(N_cheb+1)) * interval(1)/(weights[3+1])
    component(component(ẑ,3),1)[N_cheb] = L*interval(1)/(interval(2)*ν^interval(N_cheb+1))* interval(1)/(weights[3+2])

    # u₂
    component(component(ẑ,3),2)[0] = interval(1)/(ν^interval(N_cheb+1)) * interval(1)/(weights[3+2])
    for k in 1:N_cheb
        component(component(ẑ,3),2)[k] = L*interval(2)*σ^2*r*(psi(u₃, k+1, ν, N_cheb) + psi(u₃, k-1, ν, N_cheb)) / weights[3+3]
    end
    component(component(ẑ,3),2)[N_cheb] += L*σ^2*((interval(1)/weights[3+1] + (interval(1)+r)/weights[3+3]))/(interval(2)*ν^interval(N_cheb+1))

    # u₃
    component(component(ẑ,3),3)[0] = interval(1)/(ν^interval(N_cheb+1)) * interval(1)/(weights[3+3])
    component(component(ẑ,3),3)[N_cheb] = L*interval(1)/(interval(2)*ν^interval(N_cheb+1))* interval(1)/(weights[3+4])

    # u₄
    component(component(ẑ,3),4)[0] = interval(1)/(ν^interval(N_cheb+1)) * interval(1)/(weights[3+4])
    for k in 1:N_cheb
        component(component(ẑ,3),4)[k] = L*interval(2)*σ^2*r*(psi(u₁, k+1, ν, N_cheb) + psi(u₁, k-1, ν, N_cheb)) / weights[3+1]
    end
    component(component(ẑ,3),4)[N_cheb] += L*σ^2*((interval(1)/weights[3+3] + (interval(1)+r)/weights[3+1]))/(interval(2)*ν^interval(N_cheb+1))

    V = abs.(A) * ẑ

    sum1 = interval.([0,0,0,0,0,0,0])

    sum1[1] = component(V,1)[1]
    sum1[2] = component(V,2)[1]
    sum1[3] = component(V,2)[2]

    for i in 1:4
        sum1[3+i] = norm(component(component(V,3),i), ℓᵥ)
    end

    sum2 = interval.([0,0,0,0,0,0,0])

    sum2[3+1] = norm(u₂, ℓᵥ)/weights[1] + L/weights[3+2]
    sum2[3+2] = σ^2*((norm(u₁, ℓᵥ) + (interval(1) + r) * norm(u₃, ℓᵥ) + r*norm(u₃ * u₃, ℓᵥ))/weights[1] + L*(interval(1)/weights[3+1] + (interval(1)+r)/weights[3+3] + interval(2)*r*norm(u₃, ℓᵥ)/weights[3+3]))
    sum2[3+3] = norm(u₄, ℓᵥ)/weights[1] + L/weights[3+4]
    sum2[3+4] = σ^2*((norm(u₃, ℓᵥ) + (interval(1) + r) * norm(u₂, ℓᵥ) + r*norm(u₂ * u₁, ℓᵥ))/weights[1] + L*(interval(1)/weights[3+3] + (interval(1)+r)/weights[3+1] + interval(2)*r*norm(u₁, ℓᵥ)/weights[3+1]))

    sum2 = sum2 * ν/(interval(N_cheb)+interval(1))

    sum = sum1 + sum2

    sum = weights .* sum

    Z₁ = maximum(sum)

    if !isguaranteed(Z₁)
        println("Z₁ is not guaranteed")
    end

    Z₁ = interval(sup(Z₁))
    println("Z₁ = " * string(Z₁))

    # Z_2 Bound

    sum1 = interval.([0,0,0,0,0,0,0])

    sum1[1] = interval(4)

    c₁ = θ[1] + r_star*interval(-1,1)
    c₂ = θ[2] + r_star*interval(-1,1)

    for j in 1:4
        temp_1 = Evaluation(θ[1] + r_star, θ[2] + r_star) * (Derivative(2,0) * component(a,j))
        temp_1 += interval(2) * (Evaluation(θ[1] + r_star, θ[2] + r_star) * (Derivative(1,1) * component(a,j)))
        temp_1 += Evaluation(θ[1] + r_star, θ[2] + r_star) * (Derivative(0,2) * component(a,j))

        sum1[3+j] += abs(temp_1)

        sum1[3+j] += δ*interval(-1,1)*(interval(2)/(interval(1)-c₁)^3)*(interval(1)/(interval(1)-c₂))
        sum1[3+j] += δ*interval(-1,1)*(interval(1)/(interval(1)-c₁)^2)*(interval(1)/(interval(1)-c₂)^2)  
        sum1[3+j] += δ*interval(-1,1)*(interval(1)/(interval(1)-c₁))*(interval(2)/(interval(1)-c₂)^3)          
    end

    sum2 = interval.([0,0,0,0,0,0,0])

    sum2[3+1] += interval(1)
    sum2[3+2] += σ^2 * ((interval(1)/weights[3+1]+ (interval(1)+r)/weights[3+3] + interval(2)*norm(u₃, ℓᵥ)/weights[3+3]^2 + r_star/weights[3+3]^2)/weights[1] + interval(2)*r/weights[3+3]^2)
    sum2[3+3] += interval(1)
    sum2[3+4] += σ^2 * ((interval(1)/weights[3+3]+ (interval(1)+r)/weights[3+1] + interval(2)*norm(u₁, ℓᵥ)/weights[3+1]^2 + r_star/weights[3+1]^2)/weights[1] + interval(2)*r/weights[3+1]^2)

    sum2[3:7] = interval(2) * ν * sum2[3:7]

    sum = sum1 + sum2

    A_new = LinearOperator(new_space, new_space, A[:,:])
    A_norms = interval.(zeros(7,7))

    for i = 1:7
        for j = 1:7
            if i == j && i > 3
                A_norms[i,j] = maximum([opnorm(component(A_new, i,j), spaces[j], spaces[i]), interval(1) / (interval(2)*(interval(N_cheb)+interval(1)))])
            else
                A_norms[i,j] = opnorm(component(A_new, i,j), spaces[j], spaces[i])
            end
        end
    end
    
    norm_A = maximum( (A_norms *( interval(1) ./ weights)) .* (weights))

    sum = norm_A * sum

    sum = weights .* sum

    Z₂ = maximum(sum)

    if !isguaranteed(Z₂)
        println("Z₂ is not guaranteed")
    end

    Z₂ = interval(sup(Z₂))
    println("Z₂ = " * string(Z₂))

    delta = sqrt((interval(1)-Z₀-Z₁)^2 - interval(4)*Z₂*Y₀)
    println("Δ = " * string(delta))

    r_min = sup(interval(0.5) * (interval(1)-Z₀-Z₁ - delta))
    r_max = inf(interval(0.5) * (interval(1)-Z₀-Z₁ + delta))

    println("(r_min, r_max) = " *string((r_min, r_max)))
    println()

    return (r_min, r_max)
end