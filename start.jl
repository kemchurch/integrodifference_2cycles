using LinearAlgebra, Grassmann, ForwardDiff

basis"4"

function Angle(z;branch_cut=pi)
   angle_shift = pi - branch_cut
   rotation = exp(im*angle_shift)
   return angle(rotation*z) - angle_shift 
end

function Log(z;branch_cut=pi)
    return log(abs(z)) + im*Angle(z;branch_cut=branch_cut)
end

function Sqrt(z;branch_cut=pi)
    return exp((1/2)*Log(z;branch_cut=branch_cut))
end

function A(λ,df,N₀,N₁)
    # N₀ = N, N₁ = M
    return [
        0 1 0 0;
        1 0 -df(N₁)/λ 0
        0 0 0 1
        -df(N₀) 0 1 0
    ]
end

function A_exterior(λ,df,N₀,N₁)
    a = A(λ,df,N₀,N₁)
    return [
        a[1,1]+a[2,2] a[2,3] a[2,4] -a[1,3] -a[1,4] 0
        a[3,2] a[1,1]+a[3,3] a[3,4] a[1,2] 0 -a[1,4]
        a[4,2] a[4,3] a[1,1]+a[4,4] 0 a[1,2] a[1,3]
        -a[3,1] a[2,1] 0 a[2,2]+a[3,3] a[3,4] -a[2,4]
        -a[4,1] 0 a[2,1] a[4,3] a[2,2]+a[4,4] a[2,3]
        0 -a[4,1] a[3,1] -a[4,2] a[3,2] a[3,3]+a[4,4]
    ]
end

function growth_modes(λ,df,n₊,n₋)
    # Output the two unstable modes at -∞, and the two stable modes at +∞, in order.
    inner_sqrt_factor = Sqrt(df(n₊)*df(n₋)/λ; branch_cut=pi/2)
    outer_sqrt_factor_0 = Sqrt(1 + inner_sqrt_factor)
    outer_sqrt_factor_1 = Sqrt(1 - inner_sqrt_factor)
    return [
    +outer_sqrt_factor_0;
    +outer_sqrt_factor_1;
    -outer_sqrt_factor_0;
    -outer_sqrt_factor_1
    ]
end

function analytic_basis(λ,df,n₊,n₋)
    # Output a matrix whose first two columns are a basis for V₋(λ), and last two columns are a basis for V₊(λ)
    ξ₋₀,ξ₋₁,ξ₊₀,ξ₊₁ = growth_modes(λ,df,n₊,n₋)
    V₋₀ = [1 - ξ₋₀^2; (1 - ξ₋₀^2)*ξ₋₀; df(n₋); df(n₋)*ξ₋₀]
    V₋₁ = [1 - ξ₋₁^2; (1 - ξ₋₁^2)*ξ₋₁; df(n₋); df(n₋)*ξ₋₁]
    V₊₀ = [1 - ξ₊₀^2; (1 - ξ₊₀^2)*ξ₊₀; df(n₊); df(n₊)*ξ₊₀]
    V₊₁ = [1 - ξ₊₁^2; (1 - ξ₊₁^2)*ξ₊₁; df(n₊); df(n₊)*ξ₊₁]
    return (V₋₀, V₋₁, V₊₀, V₊₁)
end

function Evans_system_exterior(u,p,t)
    λ = p[1]
    df = p[2]
    mode = p[3]
    σ = p[4]
    return A_exterior(λ,df,N(t/σ,λ₃, λ₄, a, X),M(t/σ,λ₃, λ₄, a, X))*u - mode*u
end

function Evans_evaluate(λ,L,df,n₊,n₋;σ=10)
    V₁,V₂,V₃,V₄ = analytic_basis(λ,df,n₊,n₋)
    ξ₋₀,ξ₋₁,ξ₊₀,ξ₊₁ = growth_modes(λ,df,n₊,n₋)
    exterior_W₋_init = ext4_isomorphism(ext4_embedding(V₁)∧ext4_embedding(V₂))
    exterior_W₊_init = ext4_isomorphism(ext4_embedding(V₃)∧ext4_embedding(V₄))
    p₋ = (λ,df,ξ₋₀+ξ₋₁,σ)
    p₊ = (λ,df,ξ₊₀+ξ₊₁,σ)
    W₋ = solve(ODEProblem(Evans_system_exterior,exterior_W₋_init,(-L,0.0),p₋))(0)
    W₊ = solve(ODEProblem(Evans_system_exterior,exterior_W₊_init,(L,0.0),p₊))(0)
    return (ext4_embedding(W₋)∧ext4_embedding(W₊)).v[1]
end

function ext4_embedding(vect;b=basis"4")
    # Embed a vector in \C^4 or \C^6 into ∧^1(\C^4) or ∧^2(\C^4)
    _,_,v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄, v₁₂₃, v₁₂₄, v₁₃₄, v₂₃₄, v₁₂₃₄ = b
    if !( (length(vect)==4) | (length(vect)==6) )
        error("Input must be four or six-dimensional.")
    elseif length(vect)==4
        return vect[1]*v₁ + vect[2]*v₂ + vect[3]*v₃ + vect[4]*v₄
    elseif length(vect)==6
        return vect[1]*v₁₂ + vect[2]*v₁₃ + vect[3]*v₁₄ + vect[4]*v₂₃ + vect[5]*v₂₄ + vect[6]*v₃₄
    end
end

function ext4_isomorphism(V)
    # Isomorphism of exterior algebra with concrete power of \C.
    return Vector(V.v)
end

function lazy_μ(df,n₊,n₋)
    # Assuming the 2-cycle is monotonic and df is monotonic, this is exact (up to error from the proof)
    return max(abs(df(n₊)),abs(df(n₋)))^2
end

function lazy_μ_non_monotone(df,n₊,n₋)
    # Find mu assuming df is possibly non-monotone, but that the 2-cycle is monotone.
    # This function is unsafe! For specific use only.
    x0 = (n₊+n₋)/2
    d2f(x) = ForwardDiff.derivative(df,x)
    while abs(d2f(x0))>1E-12
        x0 = x0 - 0.01*d2f(x0)
    end
    return df(x0)^2
end

function curve_Λ(μ,ϵ₁,ϵ₂;granularity=0.05)
    #upper circle curve
    upper_arclength = (μ+ϵ₂)*(pi+2*ϵ₁)
    upper_arc_points = ceil(Int,upper_arclength/granularity)
    upper_curve = (μ+ϵ₂)*exp.(im*LinRange(-ϵ₁,π+ϵ₁,upper_arc_points))
    #negative real line segment
    line_segment_points = ceil(Int,(μ+2*ϵ₂-1)/granularity)
    negative_real_line_segment = exp(im*(pi+ϵ₁))*LinRange(μ+ϵ₂,1-ϵ₂,line_segment_points)
    #lower circle curve
    lower_arglength = (1-ϵ₂)*(pi+2*ϵ₁)
    lower_arc_points = ceil(Int,lower_arglength/granularity)
    lower_curve = (1-ϵ₂)*exp.(im*LinRange(π+ϵ₁,-ϵ₁,lower_arc_points))
    #positive real line segment
    positive_real_line_segment = exp(im*(-ϵ₁))*LinRange(1-ϵ₂,μ+ϵ₂,line_segment_points)
    return vcat(upper_curve,negative_real_line_segment,lower_curve,positive_real_line_segment)
end

function map_curve_to_Evans(γ,L,df,n₊,n₋;σ=10)
    Evans_curve = Vector{ComplexF64}(undef,length(γ)) 
    for n in axes(γ,1)
        Evans_curve[n] = Evans_evaluate(γ[n],L,df,n₊,n₋;σ=σ)
    end
    return Evans_curve
end

printstyled("::: Starting LOGISTIC growth function proofs :::\n ",color=:red)
include(raw"logistic\main.jl")
printstyled("::: Starting Evans function calculations for LOGISITC growth function :::\n ",color=:red)
df = x->logistic_prime(x,r)
μ = lazy_μ(df,n₊,n₋)
println("μ = "*string(μ)*" \n ")
ϵ₁ = 0.1
ϵ₂ = 0.1
L = 8
γ = curve_Λ(μ,ϵ₁,ϵ₂)
evans_curve = map_curve_to_Evans(γ,L,df,n₊,n₋)

figEvans = Figure()
axEvans_logistic = Axis(figEvans[1,1],xlabel=L"\Re(z)", ylabel=L"\Im(z)",title="Logistic")
evansplot = GLMakie.lines!(axEvans_logistic,real(evans_curve),imag(evans_curve),color=LinRange(1,length(evans_curve),length(evans_curve)))

printstyled("::: Starting RICKER growth function proofs :::\n ",color=:red)
include(raw"ricker\main.jl")
printstyled("::: Starting Evans function calculations for RICKER growth function :::\n ",color=:red)
df = x->ricker_prime(x,r)
μ = lazy_μ_non_monotone(df,n₊,n₋)
println("μ = "*string(μ)*" \n ")
ϵ₁ = 0.3
ϵ₂ = 0.3
L = 8
γ = curve_Λ(μ,ϵ₁,ϵ₂)
evans_curve = map_curve_to_Evans(γ,L,df,n₊,n₋)

axEvans_ricker = Axis(figEvans[1,2],xlabel=L"\Re(z)", ylabel=L"\Im(z)",title="Ricker")
evansplot = GLMakie.lines!(axEvans_ricker,real(evans_curve),imag(evans_curve),color=LinRange(1,length(evans_curve),length(evans_curve)))
display(GLMakie.Screen(),figEvans)