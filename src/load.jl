using Umlaut
using Umlaut: Tape, Input, Constant, mkcall, Variable, V


struct ONNXCtx
    name2var::Dict{String, Variable}
    backends::Vector{Symbol}
    exec::Bool
end

ONNXCtx(backends; exec=true) = ONNXCtx(Dict(), backends, exec)
ONNXCtx(;exec=true) = ONNXCtx(Dict(), [:ONNX], exec)

# TODO: implement rebind_context!()

"""
    getindex(tape::Tape{ONNXCtx}, onnx_name::String)

Get operation on the tape using the name in ONNX graph
"""
Base.getindex(tape::Tape{ONNXCtx}, onnx_name::String) =
    tape[tape.c.name2var[onnx_name]]

###############################################################################
#                               Operations                                    #
###############################################################################

"""
    push_call!(tape::Tape{ONNXCtx}, fn, args...; kwargs)

Shortcut for `push!(tape, mkcall(fn, args..))` also handling
keyword arguments and respecting `ONNXCtx.exec` setting.
"""
function push_call!(tape::Tape{ONNXCtx}, fn, args...; kwargs...)
    kwargs = NamedTuple(kwargs)
    if !isempty(kwargs)
        args = (kwargs, fn, args...)
        fn = Core.kwfunc(fn)
    end
    op = tape.c.exec ? mkcall(fn, args...) : mkcall(fn, args...; val=nothing)
    return push!(tape, op)
end


# A few constants to keep function signatures concise
struct OpConfig{BE, Op} end
const VarVec = Vector{Umlaut.Variable}
const AttrDict = Dict{Symbol, Any}


function load_node!(tape::Tape, nd::NodeProto, backend::Symbol)
    args = [tape.c.name2var[name] for name in nd.input]
    attrs = convert(Dict{Symbol, Any}, Dict(nd.attribute))
    conf = OpConfig{backend, Symbol(nd.op_type)}()
    try
        out = load_node!(tape, conf, args, attrs)
        # out = push_call!(tape, function(x)
        #     println(nd.output, "::", size(x), " = ", nd.name, "(", nd.input, ")")
        #     x
        # end, out)
        ismissing(out) && return out
        if out isa Tuple
            for i=1:length(nd.output)
                tape.c.name2var[nd.output[i]] = out[i]
            end
        else
            # out = push_call!(tape, function(x)
            #     @info "$(nd.name)[$(nd.op_type)]" s = size(x)
            #     x
            # end, out)
            tape.c.name2var[nd.output[1]] = out
        end
    catch
        @error "Error while loading node $nd"
        rethrow()
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Gemm}, args::VarVec, attrs::AttrDict)
    if (length(args) == 2 && get(attrs, :alpha, 1) == 1 &&
        get(attrs, :transA, 0) == 0 && get(attrs, :transB, 0) == 0)
        # simplified version: just matrix multiplication
        # note: arguments are swapped to account for row-major arrays
        return push_call!(tape, *, args[2], args[1])
    else
        # complete GEMM version
        kw = rename_keys(attrs, Dict(
            :transA => :tA,
            :transB => :tB,
            :alpha => :α,
            :beta => :β
        ))
        return push_call!(tape, onnx_gemm, args...; kw...)
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Conv}, args::VarVec, attrs::AttrDict)
    kw = from_onnx_conv(attrs) |> NamedTuple
    return push_call!(tape, conv, args...; kw...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :MaxPool}, args::VarVec, attrs::AttrDict)
    kw = from_onnx_conv(attrs; pooling=true) |> NamedTuple
    return push_call!(tape, maxpool, args[1]; kw...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :GlobalAveragePool}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, global_average_pool, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Flatten}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_flatten, args...; attrs...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Add}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, add, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sub}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, sub, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Mul}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, mul, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Div}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, div, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Relu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, relu, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Elu}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, elu, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Tanh}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, tanh, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Erf}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, erf, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Identity}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, identity, args[1])
end

function onnx_reshape(a, s)
    new_size = map(((i, dim),) -> dim == 0 ? size(a, ndims(a) - i + 1) : dim == -1 ? (:) : dim, enumerate(s))
    @show s new_size size(a)
    return reshape(a, Iterators.reverse(new_size)...)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Reshape}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_reshape, args[1], args[2])
end

function onnx_transpose(x, perm)
    perm = reverse(ndims(x) .- perm)
    return permutedims(x, perm)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Transpose}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_transpose, args[1], attrs[:perm])
end

function instance_normalize(x::AbstractArray{T}, scale, bias; kwargs...) where T
    μ = zeros(T, size(x, ndims(x) - 1))
    σ² = zeros(T, size(x, ndims(x) - 1))
    return instance_norm(x, scale, bias, μ, σ²; ϵ=get(kwargs, :epsilon, T(1f-5)), training=false)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :InstanceNormalization}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, instance_normalize, args[1], args[2], args[3]; attrs...)
end

function batched_mul4(A, B)
    @show ndims(A) ndims(B) size(A) size(B)
    if ndims(A) == 0 || ndims(B) == 0
        return A .* B
    end

    nA,mA,sA... = size(A)
    nB,mB,sB... = size(B)

    @assert sA == sB "$sA != $sB"
    @assert mA == nB "$mA != $nB"

    A = reshape(A, nA, mA, :)
    B = reshape(B, nB, mB, :)

    C = NNlib.batched_mul(A, B)
    reshape(C, nA, mB, sA...)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :MatMul}, args::VarVec, attrs::AttrDict)
    A_ndims = ndims(args[1]._op.val)
    B_ndims = ndims(args[2]._op.val)
    if A_ndims == 2 && B_ndims == 2
        return push_call!(tape, *, args[2], args[1])
    elseif A_ndims in (2, 3) && B_ndims in (2, 3)
        return push_call!(tape, NNlib.batched_mul, args[2], args[1])
    else
        return push_call!(tape, batched_mul4, args[2], args[1])
    end
end

function onnx_softmax(x; axis)
    dims = axis == -1 ? ndims(x) - axis : size(x)
    return NNlib.softmax(x; dims)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Softmax}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_softmax, args[1]; axis=attrs[:axis])
end

sigmoid(x) = NNlib.sigmoid.(x)
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sigmoid}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, sigmoid, args...)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :BatchNormalization},
        args::VarVec, attrs::AttrDict)
    kw = from_onnx_norm(attrs)
    bn = push_call!(tape, batch_norm, args...; kw...)
    if bn._op.val isa Tuple
        # usual in training mode
        # unpack tuples into calls to getfield
        y = push_call!(tape, getfield, bn, 1)
        μnext = push_call!(tape, getfield, bn, 2)
        σ²next = push_call!(tape, getfield, bn, 3)
        return y, μnext, σ²next
    else
        return bn
    end
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Shape}, args::VarVec, attrs::AttrDict)
    # TODO: handle start and end attributes
    return push_call!(tape, size_vector, args[1])
end

onnx_fill(val, s) = fill(val, reverse(s)...)
function load_node!(tape::Tape, ::OpConfig{:ONNX, :ConstantOfShape}, args::VarVec, attrs::AttrDict)
    val = array(attrs[:value]) |> only
    return push_call!(tape, onnx_fill, val, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Constant}, args::VarVec, attrs::AttrDict)
    val_attr = first(keys(attrs))
    val = if val_attr == :value
        array(attrs[val_attr])
    else
        error("Don't know how to load constant value from attribute $val_attr")
    end
    return push!(tape, Constant(val))
end

function onnx_where(cond, x, y)
    out = similar(x)
    for i in eachindex(out)
        out[i] = cond[i] ? x[i] : y[i]
    end
    out
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Where}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_where, args...)
end

function onnx_expand(x, s)
    x .* ones(reverse(s)...)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Expand}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_expand, args...)
end

function onnx_eq(a, b) 
    # if a == [64] && b == [-1]
        # return [true] # Workaround for the expand
    #end
    return a .== b
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Equal}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_eq, args...)
end

onnx_sin(x) = sin.(x)
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sin}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_sin, args[1])
end

onnx_cos(x) = cos.(x)
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Cos}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_cos, args[1])
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Gather}, args::VarVec, attrs::AttrDict)
    axis = get(attrs, :axis, 0)
    return push_call!(tape, onnx_gather, args...; axis)
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Unsqueeze}, args::VarVec, attrs::AttrDict)
    if length(args) == 2
        # ONNX >= v13
        return push_call!(tape, onnx_unsqueeze, args...)
    elseif length(args) == 1
        # ONNX < v13
        axes = attrs[:axes]
        v_axes = push!(tape, Constant(axes))
        return push_call!(tape, onnx_unsqueeze, args[1], v_axes)
    else
        throw(ArgumentError("Cannot load node from Unsqueeze with $(length(args)) arguments"))
    end
end

function onnx_reduce_mean(x; axes=nothing, keepdims=1)
    old_size = size(x)
    out = if isnothing(axes)
        mean(x)
    else
        dims = [axis >= 0 ? ndims(x) - axis : -axis for axis in axes]
        mean(x; dims)
    end

    out = if keepdims == 1
        out
    elseif isnothing(axes)
        reshape(out, (1 for _ in old_size)...)
    else
        dims = [axis >= 0 ? ndims(x) - axis : -axis for axis in axes]
        reshape(out, [d for (i, d) in enumerate(old_size) if i ∉ dims]...)
    end

    return out
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :ReduceMean}, args::VarVec, attrs::AttrDict)
    axes = get(attrs, :axes, nothing)
    keepdims = get(attrs, :keepdims, 1)
    return push_call!(tape, onnx_reduce_mean, args[1]; axes, keepdims)
end

onnx_pow(x, y) = x .^ y
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Pow}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_pow, args...)
end

onnx_sqrt(x) = sqrt.(x)
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Sqrt}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_sqrt, args[begin])
end


function load_node!(tape::Tape, ::OpConfig{:ONNX, :Slice}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_slice, args...)
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Split}, inputs::VarVec, attrs::AttrDict)
    axis = get(attrs, :axis, 0)
    split = if haskey(attrs, :split) # Version 1, 2, 11
        attrs[:split]
    elseif length(args) == 2
        inputs[2]
    else
        # the results cannot be split in multiple outputs on the tape
        # if the output size is not known during tracing.
        error("Unhandled case where split is not provided")
    end
    out = push_call!(tape, onnx_split, first(inputs), split; axis)
    return Tuple(
        push_call!(tape, getfield, out, i)
        for i in eachindex(split)
    )
end

function load_node!(tape::Tape, ::OpConfig{:ONNX, :Concat}, args::VarVec, attrs::AttrDict)
    axis = attrs[:axis]
    return push_call!(tape, onnx_concat, args...; axis)
end

function onnx_resize(x, output_size, scales, sizes=nothing;
    antialias=0, mode="nearest", nearest_mode="floor", coordinate_transformation_mode="half_pixel",
    cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0., keep_aspect_ratio_policy="stretch")

    @assert isnothing(output_size) "Resize currently does not support providing the output size (output_size = $(output_size)"
    @assert mode == "nearest" "Only mode == \"nearest\" is currently supported (got \"$mode\")"
    @assert antialias == 0 "Only antialias == 0 is currently supported (got $antialias)"

    scales = Tuple(Int.(Iterators.reverse(scales)))
    return NNlib.upsample_nearest(x, scales)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Resize}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_resize, args...; attrs...)
end

function onnx_cast(x; to)
    T = if to == 1
        Float64
    elseif to == 2
        UInt8
    elseif to == 3
        Int8
    elseif to == 4
        UInt16
    elseif to == 5
        Int16
    elseif to == 6
        Int32
    elseif to == 7
        Int64
    elseif to == 8
        String
    elseif to == 9
        Bool
    elseif to == 10
        Float16
    elseif to == 11
        Double
    elseif to == 12
        UInt32
    elseif to == 13
        UInt64
    elseif to ∈ (0, 14, 15, 16)
        onnx_type = var"TensorProto.DataType".T(to)
        error("Unsupported cast to type $onnx_type")
    else
        error("Invalid ONNX DataType $to")
    end

    return T.(x)
end
function load_node!(tape::Tape, ::OpConfig{:ONNX, :Cast}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, onnx_cast, args[1]; to=attrs[:to])
end

###############################################################################
#                                    API                                      #
###############################################################################


"""
    load(io::IO, model_args...; backends=[:ONNX], exec::Bool=true)
    load(filename::String, model_args...; backends=[:ONNX], exec::Bool=true)

Load an ONNX model as a Umlaut.Tape. The way a particular ONNX node is deserialized is
controlled by methods of [load_node!](@ref) dispatched by backend and node's op_type.

`backends` parameter can be used to customize the loading process.

`exec` parameter instructs the loader to execute every added operation just after
the addition, making the debugging easier. Default is `true`.

See also: [`save!`](@ref)
"""
function load(io::IO, args...; backends=[:ONNX], exec::Bool=true)
    onnx_model = decode(ProtoDecoder(io), ModelProto);
    g = onnx_model.graph;
    tape = Tape(ONNXCtx(backends; exec=exec))
    # create map of initializers
    init_vals = Dict{String, Any}(init.name => array(init)
        for init in g.initializer)

    # load inputs; if input has init value, take it
    # otherwise take the next available argument value
    arg_idx = 1
    used_init_names = Set([])
    for inp in g.input
        val = get(init_vals, inp.name, missing)
        v = V(0)   # will be overwritten
        if val === missing && exec == true
            @assert(
                arg_idx <= length(args),
                "Neither initializer, nor argument is provided for input $(inp.name)"
            )
            val = args[arg_idx]
            arg_idx += 1
            v = push!(tape, Input(val))
        else
            # convert inputs that also have initializers to constants
            # these are usually model parameters, but may
            v = push!(tape, Constant(val))
        end
        tape.c.name2var[inp.name] = v
        push!(used_init_names, inp.name)
    end
    # load the rest of initilizers as constants
    for init in g.initializer
        name = init.name
        if !in(name, used_init_names)
            val = init_vals[name]
            v = push!(tape, Umlaut.Constant(val))
            tape.c.name2var[name] = v
        end
    end

    # https://github.com/onnx/onnx/blob/main/docs/IR.md#optional-inputs-and-outputs
    optional = push!(tape, Constant(nothing))
    tape.c.name2var[""] = optional

    # load nodes
    for nd in g.node
        success = false
        for backend in tape.c.backends
          if !ismissing(load_node!(tape, nd, backend))
            success = true
            @debug "Loaded $(nd.op_type) using backend $(backend)"
            break
          end
        end
        success || error("Couldn't load node for $(nd.op_type), " *
                         "tried the following backends: $(tape.c.backends)")
    end
    if length(g.output) == 1
        tape.result = Umlaut.bound(tape, V(length(tape)))
    else
        # tuple output: we expect tape to contain these outputs as vars  destructured
        # from a multi-ouput op using a sequence of `getfield()` calls
        vars = [tape.c.name2var[o.name] for o in g.output]
        @assert(all(tape[v] isa Call && tape[v].fn == getfield for v in vars),
            "Don't understand this multi-output result of the graph")
        tape.result = tape[vars[1]].args[1]
    end
    return tape
end

function load(filename::String, args...; backends=[:ONNX], exec::Bool=true)
    return open(filename) do io
        load(io, args...; backends=backends, exec=exec)
    end
end
