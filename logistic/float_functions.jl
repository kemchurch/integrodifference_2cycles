using RadiiPolynomial, DifferentialEquations, IntervalArithmetic

function F_manif!(F_manif::Sequence, a::Sequence, N::Vector{Int}, σ::Float64, r::Float64, equilibrium::Vector{Float64}, λ₁::Float64, λ₂::Float64, ξ₁::Vector{Float64}, ξ₂::Vector{Float64})
    #Extracting what space we are in and initializaing Φ
    s = space(component(a,1));
    Φ = zeros(s^4);

    #Setting all components of F to be 0
    F_manif .= 0;

    #Extracting the components
    F_manif₁ = component(F_manif, 1);
    F_manif₂ = component(F_manif, 2);
    F_manif₃ = component(F_manif, 3);
    F_manif₄ = component(F_manif, 4);

    a₁ = component(a, 1);
    a₂ = component(a, 2);
    a₃ = component(a, 3);
    a₄ = component(a, 4);

    Φ₁ = component(Φ, 1);
    Φ₂ = component(Φ, 2);
    Φ₃ = component(Φ, 3);
    Φ₄ = component(Φ, 4);

    #Computing Φ for higher order terms
    Φ!(Φ, a, σ, r);

    #Computing the sequence operator D = α₁λ₁ + α₂λ₂
    D₁ = zeros(s,s);
    D₂ = zeros(s,s);

    for i in 0:N[1]
        for j in 0:N[2]
            D₁[(i,j),(i,j)] = i;
            D₂[(i,j),(i,j)] = j;
        end
    end

    D = λ₁*D₁ + λ₂*D₂;

    #Setting the appropriate values for the higher order terms
    F_manif₁[:] = (D * a₁ - Φ₁)[:];
    F_manif₂[:] = (D * a₂ - Φ₂)[:];
    F_manif₃[:] = (D * a₃ - Φ₃)[:];
    F_manif₄[:] = (D * a₄ - Φ₄)[:];

    #Setting the appropriate values for the intial conditions
    F_manif₁[(0,0)] = a₁[(0,0)] - equilibrium[1];
    F_manif₂[(0,0)] = a₂[(0,0)] - equilibrium[2];
    F_manif₃[(0,0)] = a₃[(0,0)] - equilibrium[3];
    F_manif₄[(0,0)] = a₄[(0,0)] - equilibrium[4];

    F_manif₁[(1,0)] = a₁[(1,0)] - ξ₁[1];
    F_manif₂[(1,0)] = a₂[(1,0)] - ξ₁[2];
    F_manif₃[(1,0)] = a₃[(1,0)] - ξ₁[3];
    F_manif₄[(1,0)] = a₄[(1,0)] - ξ₁[4];

    F_manif₁[(0,1)] = a₁[(0,1)] - ξ₂[1];
    F_manif₂[(0,1)] = a₂[(0,1)] - ξ₂[2];
    F_manif₃[(0,1)] = a₃[(0,1)] - ξ₂[3];
    F_manif₄[(0,1)] = a₄[(0,1)] - ξ₂[4];
end

function DF_manif!(DF_manif::LinearOperator, a::Sequence, N::Vector{Int}, σ::Float64, r::Float64, equilibrium::Vector{Float64}, λ₁::Float64, λ₂::Float64, ξ₁::Vector{Float64}, ξ₂::Vector{Float64})
    DF_manif .= 0

    s = space(component(a,1))

    D₁ = zeros(s,s)
    D₂ = zeros(s,s)

    for i in 0:N[1]
        for j in 0:N[2]
            D₁[(i,j),(i,j)] = i 
            D₂[(i,j),(i,j)] = j
        end
    end

    D = λ₁*D₁ + λ₂*D₂

    DΦ = zeros(s^4, s^4)
    DΦ!(DΦ, a, σ, r)

    # Setting the higher order terms
    component(DF_manif,1,1).coefficients[:] = -component(DΦ,1,1).coefficients[:] + D.coefficients[:]
    component(DF_manif,2,1).coefficients[:] = -component(DΦ,2,1).coefficients[:]
    component(DF_manif,3,1).coefficients[:] = -component(DΦ,3,1).coefficients[:]
    component(DF_manif,4,1).coefficients[:] = -component(DΦ,4,1).coefficients[:]

    component(DF_manif,1,2).coefficients[:] = -component(DΦ,1,2).coefficients[:]
    component(DF_manif,2,2).coefficients[:] = -component(DΦ,2,2).coefficients[:] + D.coefficients[:]
    component(DF_manif,3,2).coefficients[:] = -component(DΦ,3,2).coefficients[:]
    component(DF_manif,4,2).coefficients[:] = -component(DΦ,4,2).coefficients[:]

    component(DF_manif,1,3).coefficients[:] = -component(DΦ,1,3).coefficients[:]
    component(DF_manif,2,3).coefficients[:] = -component(DΦ,2,3).coefficients[:]
    component(DF_manif,3,3).coefficients[:] = -component(DΦ,3,3).coefficients[:] + D.coefficients[:]
    component(DF_manif,4,3).coefficients[:] = -component(DΦ,4,3).coefficients[:]

    component(DF_manif,1,4).coefficients[:] = -component(DΦ,1,4).coefficients[:]
    component(DF_manif,2,4).coefficients[:] = -component(DΦ,2,4).coefficients[:]
    component(DF_manif,3,4).coefficients[:] = -component(DΦ,3,4).coefficients[:]
    component(DF_manif,4,4).coefficients[:] = -component(DΦ,4,4).coefficients[:] + D.coefficients[:]

    # Setting the lower order terms to zero
    for i in 0:1
        for j in 0:1-i
            component(DF_manif,1,1)[(i,j),:] .= 0
            component(DF_manif,2,1)[(i,j),:] .= 0
            component(DF_manif,3,1)[(i,j),:] .= 0
            component(DF_manif,4,1)[(i,j),:] .= 0

            component(DF_manif,1,2)[(i,j),:] .= 0
            component(DF_manif,2,2)[(i,j),:] .= 0
            component(DF_manif,3,2)[(i,j),:] .= 0
            component(DF_manif,4,2)[(i,j),:] .= 0

            component(DF_manif,1,3)[(i,j),:] .= 0
            component(DF_manif,2,3)[(i,j),:] .= 0
            component(DF_manif,3,3)[(i,j),:] .= 0
            component(DF_manif,4,3)[(i,j),:] .= 0

            component(DF_manif,1,4)[(i,j),:] .= 0
            component(DF_manif,2,4)[(i,j),:] .= 0
            component(DF_manif,3,4)[(i,j),:] .= 0
            component(DF_manif,4,4)[(i,j),:] .= 0
        end
    end

    # Putting the correct lower order terms
    component(DF_manif,1,1)[(0,0),(0,0)] = 1;
    component(DF_manif,2,2)[(0,0),(0,0)] = 1;
    component(DF_manif,3,3)[(0,0),(0,0)] = 1;
    component(DF_manif,4,4)[(0,0),(0,0)] = 1;

    component(DF_manif,1,1)[(1,0),(1,0)] = 1;
    component(DF_manif,2,2)[(1,0),(1,0)] = 1;
    component(DF_manif,3,3)[(1,0),(1,0)] = 1;
    component(DF_manif,4,4)[(1,0),(1,0)] = 1;

    component(DF_manif,1,1)[(0,1),(0,1)] = 1;
    component(DF_manif,2,2)[(0,1),(0,1)] = 1;
    component(DF_manif,3,3)[(0,1),(0,1)] = 1;
    component(DF_manif,4,4)[(0,1),(0,1)] = 1;
end

function Φ!(Φ::Sequence, a::Sequence, σ::Float64, r::Float64)
    #Setting all componends of Φ to be 0
    Φ .= 0;

    #Extracting the components
    Φ₁ = component(Φ, 1);
    Φ₂ = component(Φ, 2);
    Φ₃ = component(Φ, 3);
    Φ₄ = component(Φ, 4);

    a₁ = component(a, 1);
    a₂ = component(a, 2);
    a₃ = component(a, 3);
    a₄ = component(a, 4);

    #Setting the appropriate values for the higher order terms
    Φ₁[:] = project(a₂, space(Φ₁))[:];
    Φ₂[:] = project(σ^2*(a₁ - (1+r)*a₃ + r*(a₃*a₃)), space(Φ₂))[:];
    Φ₃[:] = project(a₄, space(Φ₃))[:];
    Φ₄[:] = project(σ^2*(a₃ - (1+r)*a₁ + r*(a₁*a₁)), space(Φ₄))[:];
end

function DΦ!(DΦ::LinearOperator, a::Sequence, σ::Float64, r::Float64)
    # Initialize DΦ to be zero, then fill in the correct blocks
    DΦ .= 0

    # Extract the space of our sequence
    s = space(component(a,1))

    a₁ = component(a, 1);
    a₃ = component(a, 3);

    component(DΦ, 1, 2).coefficients[:] = project(I, s, s).coefficients[:]

    component(DΦ, 2, 1).coefficients[:] = project(σ^2*I, s, s).coefficients[:]
    component(DΦ, 2, 3).coefficients[:] = project(-σ^2*(1+r)*I, s, s).coefficients[:] + project(σ^2* 2*r*Multiplication(a₃), s, s).coefficients[:]

    component(DΦ, 3, 4).coefficients[:] = project(I, s, s).coefficients[:]

    component(DΦ, 4, 1).coefficients[:] = project(-σ^2*(1+r)*I, s, s).coefficients[:] + project(σ^2* 2*r*Multiplication(a₁), s, s).coefficients[:]
    component(DΦ, 4, 3).coefficients[:] = project(σ^2*I, s, s).coefficients[:]
end

function F_orbit!(F_orbit::Sequence, X::Sequence, a::Sequence, N_cheb::Int, σ::Float64, r::Float64)
    F_orbit .= 0

    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    u = component(X,3)

    u₁ = component(u,1)
    u₂ = component(u,2)
    u₃ = component(u,3)
    u₄ = component(u,4)

    Φ = zeros(Chebyshev(N_cheb+1)^4)
    Φ!(Φ, u, σ, r)

    Φ₁ = component(Φ,1)
    Φ₂ = component(Φ,2)
    Φ₃ = component(Φ,3)
    Φ₄ = component(Φ,4)

    # How deep in the manifold are we
    component(F_orbit,1)[1] = θ[1]^2 + θ[2]^2 - 0.95

    # We end up in the plane of fixed point of R
    component(F_orbit,2)[1] = Evaluation(1)*(u₁ - u₃)
    component(F_orbit,2)[2] = Evaluation(1)*(u₂ + u₄)

    F_orbit_cheb = component(F_orbit,3)

    F_orbit_cheb₁ = component(F_orbit_cheb,1)
    F_orbit_cheb₂ = component(F_orbit_cheb,2)
    F_orbit_cheb₃ = component(F_orbit_cheb,3)
    F_orbit_cheb₄ = component(F_orbit_cheb,4)


    # Solution to the ODE

    # Constructing operator (Du)ₖ = 2k uₖ
    D = zeros(Chebyshev(N_cheb), Chebyshev(N_cheb))

    for k in 0:N_cheb
        D[k,k] = 2*k
    end

    # Constructing operator (TΦ)ₖ = Φₖ₋₁ + Φₖ₊₁
    T = zeros(Chebyshev(N_cheb+1), Chebyshev(N_cheb))

    for k in 1:N_cheb
        T[k,k-1] = 1
        T[k,k+1] = -1
    end

    # Computing higher order terms
    F_orbit_cheb₁[:] = (D * u₁ + L* T * Φ₁)[:]
    F_orbit_cheb₂[:] = (D * u₂ + L* T * Φ₂)[:]
    F_orbit_cheb₃[:] = (D * u₃ + L* T * Φ₃)[:]
    F_orbit_cheb₄[:] = (D * u₄ + L* T * Φ₄)[:]

    # Computing lower order terms
    F_orbit_cheb₁[0] = Evaluation(-1)*u₁ - Evaluation(θ[1], θ[2])*component(a,1)
    F_orbit_cheb₂[0] = Evaluation(-1)*u₂ - Evaluation(θ[1], θ[2])*component(a,2)
    F_orbit_cheb₃[0] = Evaluation(-1)*u₃ - Evaluation(θ[1], θ[2])*component(a,3)
    F_orbit_cheb₄[0] = Evaluation(-1)*u₄ - Evaluation(θ[1], θ[2])*component(a,4)
end

function DF_orbit!(DF_orbit::LinearOperator, X::Sequence, a::Sequence, N_cheb::Int, σ::Float64, r::Float64)
    DF_orbit .= 0

    L = component(X,1)[1]
    θ = component(X,2)[1:2]
    u = component(X,3)

    Φ = zeros(Chebyshev(N_cheb+1)^4)
    Φ!(Φ, project(u, Chebyshev(N_cheb+1)^4), σ, r)

    DΦ = zeros(Chebyshev(N_cheb+1)^4,Chebyshev(N_cheb+1)^4)
    DΦ!(DΦ, project(u, Chebyshev(N_cheb+1)^4), σ, r)

    # Constructing operator (Du)ₖ = 2k uₖ
    D = zeros(Chebyshev(N_cheb), Chebyshev(N_cheb))

    for k in 0:N_cheb
        D[k,k] = 2*k
    end

    # Constructing operator (TΦ)ₖ = Φₖ₋₁ + Φₖ₊₁
    T = zeros(Chebyshev(N_cheb+1), Chebyshev(N_cheb))

    for k in 1:N_cheb
        T[k,k-1] = 1
        T[k,k+1] = -1
    end

    # D_L F_L
    # No dependence

    # D_θ F_L
    component(DF_orbit,1,2)[1,1] = 2*θ[1]
    component(DF_orbit,1,2)[1,2] = 2*θ[2]

    # D_u F_L
    # No dependence

    # D_L F_θ
    # No dependence

    # D_θ F_θ
    # No dependence

    # D_u F_θ
    component(component(DF_orbit,2,3),1,1).coefficients[:] = project(Evaluation(1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]
    component(component(DF_orbit,2,3),1,3).coefficients[:] = -project(Evaluation(1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]

    component(component(DF_orbit,2,3),2,2).coefficients[:] = project(Evaluation(1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]
    component(component(DF_orbit,2,3),2,4).coefficients[:] = project(Evaluation(1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]

    # D_L F_u
    # Higher order terms
    component(component(DF_orbit,3,1),1).coefficients[2:end] = (T * component(Φ,1)).coefficients[2:end]
    component(component(DF_orbit,3,1),2).coefficients[2:end] = (T * component(Φ,2)).coefficients[2:end]
    component(component(DF_orbit,3,1),3).coefficients[2:end] = (T * component(Φ,3)).coefficients[2:end]
    component(component(DF_orbit,3,1),4).coefficients[2:end] = (T * component(Φ,4)).coefficients[2:end]

    # D_θ F_u
    # Initial terms
    component(component(DF_orbit,3,2),1,1).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(1,0) * component(a,1)))
    component(component(DF_orbit,3,2),2,1).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(1,0) * component(a,2)))
    component(component(DF_orbit,3,2),3,1).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(1,0) * component(a,3)))
    component(component(DF_orbit,3,2),4,1).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(1,0) * component(a,4)))

    component(component(DF_orbit,3,2),1,2).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(0,1) * component(a,1)))
    component(component(DF_orbit,3,2),2,2).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(0,1) * component(a,2)))
    component(component(DF_orbit,3,2),3,2).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(0,1) * component(a,3)))
    component(component(DF_orbit,3,2),4,2).coefficients[1] = -(Evaluation(θ[1], θ[2]) * (Derivative(0,1) * component(a,4)))
    
    # D_u F_u
    # Higher order terms
    component(component(DF_orbit,3,3),1,1).coefficients[:] = (L * T * project(component(DΦ,1,1), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:] + D.coefficients[:]
    component(component(DF_orbit,3,3),2,1).coefficients[:] = (L * T * project(component(DΦ,2,1), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),3,1).coefficients[:] = (L * T * project(component(DΦ,3,1), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),4,1).coefficients[:] = (L * T * project(component(DΦ,4,1), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]

    component(component(DF_orbit,3,3),1,2).coefficients[:] = (L * T * project(component(DΦ,1,2), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),2,2).coefficients[:] = (L * T * project(component(DΦ,2,2), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:] + D.coefficients[:]
    component(component(DF_orbit,3,3),3,2).coefficients[:] = (L * T * project(component(DΦ,3,2), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),4,2).coefficients[:] = (L * T * project(component(DΦ,4,2), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]

    component(component(DF_orbit,3,3),1,3).coefficients[:] = (L * T * project(component(DΦ,1,3), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),2,3).coefficients[:] = (L * T * project(component(DΦ,2,3), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),3,3).coefficients[:] = (L * T * project(component(DΦ,3,3), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:] + D.coefficients[:]
    component(component(DF_orbit,3,3),4,3).coefficients[:] = (L * T * project(component(DΦ,4,3), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]

    component(component(DF_orbit,3,3),1,4).coefficients[:] = (L * T * project(component(DΦ,1,4), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),2,4).coefficients[:] = (L * T * project(component(DΦ,2,4), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),3,4).coefficients[:] = (L * T * project(component(DΦ,3,4), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:]
    component(component(DF_orbit,3,3),4,4).coefficients[:] = (L * T * project(component(DΦ,4,4), Chebyshev(N_cheb), Chebyshev(N_cheb+1))).coefficients[:] + D.coefficients[:]

    # Initial terms
    component(component(DF_orbit,3,3),1,1).coefficients[1,:] = project(Evaluation(-1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]
    component(component(DF_orbit,3,3),2,1).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),3,1).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),4,1).coefficients[1,:] .= 0

    component(component(DF_orbit,3,3),1,2).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),2,2).coefficients[1,:] = project(Evaluation(-1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]
    component(component(DF_orbit,3,3),3,2).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),4,2).coefficients[1,:] .= 0

    component(component(DF_orbit,3,3),1,3).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),2,3).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),3,3).coefficients[1,:] = project(Evaluation(-1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:]
    component(component(DF_orbit,3,3),4,3).coefficients[1,:] .= 0

    component(component(DF_orbit,3,3),1,4).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),2,4).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),3,4).coefficients[1,:] .= 0
    component(component(DF_orbit,3,3),4,4).coefficients[1,:] = project(Evaluation(-1), Chebyshev(N_cheb), ParameterSpace()).coefficients[:] 
end
