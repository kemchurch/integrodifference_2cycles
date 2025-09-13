# Code to accompany "Spatially inhomogeneous two-cycles in an integrodifference equation"
This code implements the computer-assisted proofs of the paper "Spatially inhomogeneous two-cycles in an integrodifference equation" by Church, Constantineau and Lessard (2025).

To start, the following commands can be run from the REPL assuming the current directory is the parent folder which contains "start.jl". This will replicate all computer-assisted proofs and produce most of the figures.
```julia
import Pkg
Pkg.activate("2cycles")
Pkg.instantiate()
include("start.jl")
```
