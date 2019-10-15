using SparseArrays

export SCSMatrix, SCSData, SCSSettings, SCSSolution, SCSInfo, SCSCone, SCSVecOrMatOrSparse

SCSVecOrMatOrSparse = Union{VecOrMat, SparseMatrixCSC{Float64,Int}}

SCSInt = Union{Int64, Int32}

abstract type LinearSolver end
struct DirectSolver <: LinearSolver end
struct IndirectSolver <: LinearSolver end
struct GpuSolver <: LinearSolver end

scsint_t(::Type{<:LinearSolver}) = Int64
scsint_t(::Type{GpuSolver}) = Int32

clib(::Type{DirectSolver}) = direct
clib(::Type{IndirectSolver}) = indirect
clib(::Type{GpuSolver}) = gpu

struct SCSMatrix{T <: SCSInt}
    values::Ptr{Cdouble}
    rowval::Ptr{T}
    colptr::Ptr{T}
    m::T
    n::T
end

# Version where Julia manages the memory for the vectors.
struct ManagedSCSMatrix{T <: SCSInt}
    values::Vector{Cdouble}
    rowval::Vector{T}
    colptr::Vector{T}
    m::T
    n::T
end

function ManagedSCSMatrix(m::T, n::T, A::SCSVecOrMatOrSparse) where T
    A_sparse = sparse(A)

    values = copy(A_sparse.nzval)
    rowval = A_sparse.rowval .- 1
    colptr = A_sparse.colptr .- 1

    return ManagedSCSMatrix{T}(values, rowval, colptr, m, n)
end

# Returns an SCSMatrix. The vectors are *not* GC tracked in the struct.
# Use this only when you know that the managed matrix will outlive the SCSMatrix.
SCSMatrix(m::ManagedSCSMatrix) =
    SCSMatrix(pointer(m.values), pointer(m.rowval), pointer(m.colptr), m.m, m.n)


struct SCSSettings{T <: SCSInt}
    normalize::T # boolean, heuristic data rescaling
    scale::Cdouble # if normalized, rescales by this factor
    rho_x::Cdouble # x equality constraint scaling
    max_iters::T # maximum iterations to take
    eps::Cdouble # convergence tolerance
    alpha::Cdouble # relaxation parameter
    cg_rate::Cdouble # for indirect, tolerance goes down like (1/iter)^cg_rate
    verbose::T # boolean, write out progress
    warm_start::T # boolean, warm start (put initial guess in Sol struct)
    acceleration_lookback::T # acceleration memory parameter
    write_data_filename::Cstring

    SCSSettings{T}() where T = new{T}()
    SCSSettings{T}(normalize, scale, rho_x, max_iters, eps, alpha, cg_rate, verbose, warm_start, acceleration_lookback, write_data_filename) where T =
    new{T}(normalize, scale, rho_x, max_iters, eps, alpha, cg_rate, verbose, warm_start, acceleration_lookback, write_data_filename)
end

function _SCS_user_settings(default_settings::SCSSettings{T};
        normalize=default_settings.normalize,
        scale=default_settings.scale,
        rho_x=default_settings.rho_x,
        max_iters=default_settings.max_iters,
        eps=default_settings.eps,
        alpha=default_settings.alpha,
        cg_rate=default_settings.cg_rate,
        verbose=default_settings.verbose,
        warm_start=default_settings.warm_start,
        acceleration_lookback=default_settings.acceleration_lookback,
        write_data_filename=default_settings.write_data_filename) where T
    return SCSSettings{T}(normalize, scale, rho_x, max_iters, eps, alpha, cg_rate, verbose,warm_start, acceleration_lookback, write_data_filename)
end

function SCSSettings(linear_solver::Type{<:LinearSolver}; options...)

    scsint = scsint_t(linear_solver)
    mmatrix = ManagedSCSMatrix(scsint(0), scsint(0), spzeros(1,1))
    matrix = Ref(SCSMatrix(mmatrix))
    default_settings = Ref(SCSSettings{scsint}())
    dummy_data = Ref(SCSData{scsint}(0, 0,
        Base.unsafe_convert(Ptr{SCSMatrix{scsint}}, matrix),
        pointer([0.0]), pointer([0.0]),
        Base.unsafe_convert(Ptr{SCSSettings{scsint}}, default_settings)))
    SCS_set_default_settings(linear_solver, dummy_data)
    return _SCS_user_settings(default_settings[]; options...)
end

struct SCSData{T <: SCSInt}
    # A has m rows, n cols
    m::T
    n::T
    A::Ptr{SCSMatrix{T}}
    # b is of size m, c is of size n
    b::Ptr{Cdouble}
    c::Ptr{Cdouble}
    stgs::Ptr{SCSSettings{T}}
end

struct SCSSolution
    x::Ptr{Nothing}
    y::Ptr{Nothing}
    s::Ptr{Nothing}
end


struct SCSInfo{T <: SCSInt}
    iter::T
    status::NTuple{32, Cchar} # char status[32]
    statusVal::T
    pobj::Cdouble
    dobj::Cdouble
    resPri::Cdouble
    resDual::Cdouble
    resInfeas::Cdouble
    resUnbdd::Cdouble
    relGap::Cdouble
    setupTime::Cdouble
    solveTime::Cdouble
end

SCSInfo(::Type{T}) where T = SCSInfo{T}(0, ntuple(_ -> zero(Cchar), 32), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function raw_status(info::SCSInfo)
    s = collect(info.status)
    len = findfirst(iszero, s) - 1
    # There is no method String(::Vector{Cchar}) so we convert to `UInt8`.
    return String(UInt8[s[i] for i in 1:len])
end

# SCS solves a problem of the form
# minimize        c' * x
# subject to      A * x + s = b
#                 s in K
# where K is a product cone of
# zero cones,
# linear cones { x | x >= 0 },
# second-order cones { (t,x) | ||x||_2 <= t },
# semi-definite cones { X | X psd }, and
# exponential cones {(x,y,z) | y e^(x/y) <= z, y>0 }.
# dual exponential cones {(u,v,w) | −u e^(v/u) <= e w, u<0}
# power cones {(x,y,z) | x^a * y^(1-a) >= |z|, x>=0, y>=0}
# dual power cones {(u,v,w) | (u/a)^a * (v/(1-a))^(1-a) >= |w|, u>=0, v>=0}
struct SCSCone{T <: SCSInt}
    f::T # number of linear equality constraints
    l::T # length of LP cone
    q::Ptr{T} # array of second-order cone constraints
    qsize::T # length of SOC array
    s::Ptr{T} # array of SD constraints
    ssize::T # length of SD array
    ep::T # number of primal exponential cone triples
    ed::T # number of dual exponential cone triples
    p::Ptr{Cdouble} # array of power cone params, must be \in [-1, 1], negative values are interpreted as specifying the dual cone
    psize::T # length of power cone array
end

# Returns an SCSCone. The q, s, and p arrays are *not* GC tracked in the
# struct. Use this only when you know that q, s, and p will outlive the struct.
function SCSCone(f::T, l::T, q::Vector{T}, s::Vector{T},
                 ep::T, ed::T, p::Vector{Cdouble}) where T
    return SCSCone{T}(f, l, pointer(q), length(q), pointer(s), length(s), ep, ed, pointer(p), length(p))
end


mutable struct Solution
    x::Array{Float64, 1}
    y::Array{Float64, 1}
    s::Array{Float64, 1}
    info::SCSInfo
    ret_val::Int
end

function sanitize_SCS_options(options)
    options = Dict(options)
    if haskey(options, :linear_solver)
        linear_solver = options[:linear_solver]
        if linear_solver in available_solvers
            nothing
        else
            throw(ArgumentError("Unrecognized linear_solver passed to SCS: $linear_solver;\nRecognized options are: $(join(available_solvers, ", ", " and "))."))
        end
        delete!(options, :linear_solver)
    else
        linear_solver = IndirectSolver # the default linear_solver
    end

    SCS_options = append!([:linear_solver], fieldnames(SCSSettings))
    unrecognized = setdiff(keys(options), SCS_options)
    if length(unrecognized) > 0
        plur = length(unrecognized) > 1 ? "s" : ""
        throw(ArgumentError("Unrecognized option$plur passed to SCS: $(join(unrecognized, ", "));\nRecognized options are: $(join(SCS_options, ", ", " and "))."))
    end
    return linear_solver, options
end
