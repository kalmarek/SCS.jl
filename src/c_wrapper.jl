export SCS_init, SCS_solve, SCS_finish, SCS_version

# SCS solves a problem of the form
# minimize        c' * x
# subject to      A * x + s = b
#                 s in K
# where K is a product cone of
# zero cones,
# linear cones { x | x >= 0 },
# second-order cones (SOC) { (t,x) | ||x||_2 <= t },
# semi-definite cones (SDC) { X | X psd }, and
# exponential cones {(x,y,z) | y e^(x/y) <= z, y>0 }.
#
#
# Description of input argments:
# A is the matrix with m rows and n cols
# b is of length m x 1
# c is of length n x 1
#
# f (num primal zero / dual free cones, i.e. primal equality constraints)
# l (num linear cones)
# q (array of SOCs sizes)
# s (array of SDCs sizes)
# ep (num primal exponential cones)
# ed (num dual exponential cones).
#
# Returns a Solution object.
function SCS_solve(linear_solver::Type{<:LinearSolver},
        m::Int, n::Int, A::SCSVecOrMatOrSparse, b::Array{Float64},
        c::Array{Float64}, f::Int, l::Int, q::Array{Int}, s::Array{Int},
        ep::Int, ed::Int, p::Array{Float64},
        primal_sol::Vector{Float64}=Float64[],
        dual_sol::Vector{Float64}=Float64[],
        slack::Vector{Float64}=Float64[];
        options...)

    n > 0 || throw(ArgumentError("The number of variables in SCSModel must be greater than 0"))
    m > 0 || throw(ArgumentError("The number of constraints in SCSModel must be greater than 0"))

    scsint = scsint_t(linear_solver)

    managed_matrix = ManagedSCSMatrix(scsint(m), scsint(n), A)
    matrix = Ref(SCSMatrix(managed_matrix))
    settings = Ref(SCSSettings(linear_solver; options...))
    data = Ref(SCSData{scsint}(m, n,
        Base.unsafe_convert(Ptr{SCSMatrix{scsint}}, matrix),
        pointer(b),
        pointer(c),
        Base.unsafe_convert(Ptr{SCSSettings{scsint}}, settings)))

    # we can't rely on automatic conversion in SCSCone{IntT} constructor, since these need to be GC.@preserved
    q_scsint = convert(Array{scsint}, q)
    s_scsint = convert(Array{scsint}, s)
    cone = Ref(SCSCone(scsint(f), scsint(l), q_scsint, s_scsint, scsint(ep), scsint(ed), p))
    info = Ref(SCSInfo(scsint))

    ws = (:warm_start=>true) in options

    if ws && length(primal_sol) == n && length(dual_sol) == length(slack) == m
        x = primal_sol
        y = dual_sol
        s = slack
    else
        x = zeros(n)
        y = zeros(m)
        s = zeros(m)
    end
    solution = SCSSolution(pointer(x), pointer(y), pointer(s))

    Base.GC.@preserve managed_matrix matrix settings b c q_scsint s_scsint p begin
        p_work = SCS_init(linear_solver, data, cone, info)
        status = SCS_solve(linear_solver, p_work, data, cone, solution, info)
        SCS_finish(linear_solver, p_work)
    end

    return Solution(x, y, s, info[], status)

end

# Wrappers for the direct C API.
# Do not call these wrapper methods directly unless you understand the
# use of Base.GC.@preserve in the SCS_solve helper above.

# Take Ref{}s because SCS might modify the structs
const available_solvers = Base.@isdefined(gpu) ? [DirectSolver, IndirectSolver, GpuSolver] : [DirectSolver, IndirectSolver]

for T in available_solvers
    lib = clib(T)
    scsint = scsint_t(T)

    @eval begin
        function SCS_set_default_settings(::Type{$T}, data::Ref{SCSData{$scsint}})
            ccall((:scs_set_default_settings, $lib), Nothing,
                (Ref{SCSData{$scsint}},), data)
        end

        function SCS_init(::Type{$T}, data::Ref{SCSData{$scsint}},
            cone::Ref{SCSCone{$scsint}}, info::Ref{SCSInfo{$scsint}})
            p_work = ccall((:scs_init, $lib), Ptr{Nothing},
                (Ptr{SCSData{$scsint}}, Ptr{SCSCone{$scsint}}, Ptr{SCSInfo{$scsint}}),
                data, cone, info)

            return p_work
        end

        # solution struct is simple enough, we know it won't be modified by SCS so take by value
        function SCS_solve(::Type{$T}, p_work::Ptr{Nothing}, data::Ref{SCSData{$scsint}},
            cone::Ref{SCSCone{$scsint}}, solution::SCSSolution, info::Ref{SCSInfo{$scsint}})

            status = ccall((:scs_solve, $lib), $scsint,
                (Ptr{Nothing}, Ptr{SCSData{$scsint}}, Ptr{SCSCone{$scsint}}, Ref{SCSSolution}, Ptr{SCSInfo{$scsint}}),
                p_work, data, cone, solution, info)

            return status
        end

        function SCS_finish(::Type{$T}, p_work::Ptr{Nothing})
            ccall((:scs_finish, $lib), Nothing,
                (Ptr{Nothing}, ),
                p_work)
        end
    end
end

# This one is safe to call
function SCS_version()
    return unsafe_string(ccall((:scs_version, SCS.direct), Cstring, ()))
end
