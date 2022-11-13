import Pkg

modelproto(graph;kwargs...) = ModelProto(;
    ir_version=7,
    opset_import=[OperatorSetIdProto(version=14)],
    producer_name="ONNX.jl",
    producer_version=string(Pkg.Types.Context().env.project.version), # TODO: Ugh....
    graph=graph,
    kwargs...)


"""
    graphproto()
Return an [`ONNX.GraphProto`](@ref) with all fields initialized to empty arrays.
"""
graphproto(name; kwargs...) = GraphProto(;
    node = NodeProto[],
    initializer = TensorProto[],
    input = ValueInfoProto[],
    output = ValueInfoProto[],
    value_info = ValueInfoProto[],
    name = name,
    kwargs...
)


add!(gp::GraphProto, np::NodeProto) = push!(gp.node, np)

add!(gp::GraphProto, tp::TensorProto) = push!(gp.initializer, tp)


##############################################################################
#                                 Utils                                    #
##############################################################################

# can we make it more robust?
iskwfunc(f) = endswith(string(f), "##kw")

function kwargs2dict(op::Umlaut.Call)
    kw = iskwfunc(op.fn) ? op.args[1] : (;)
    return Dict(zip(keys(kw), values(kw)))
end

macro opconfig_kw(backend, fn)
    return quote
        $OpConfig{$backend, <:Union{typeof($fn), typeof(Core.kwfunc($fn))}}
    end
end

function NodeProto(op_type::String, op::Umlaut.Call, attrs::Dict=Dict())
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    return NodeProto(
        input=[onnx_name(v) for v in args],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[AttributeProto(k, v) for (k, v) in attrs],
        op_type=op_type
    )
end

ValueInfoProto(op::Umlaut.AbstractOp) = ValueInfoProto(
    onnx_name(op),
    # utils in write.jl reverse the shape, so we don't do it here
    # try the following for example:
    #     TypeProto_Tensor((4, 3), Float64).shape.dim[1].dim_value
    # which gives 3 instead of 4
    size(op.val),
    eltype(op.val) ∈ (Int, Float64, Float32, Bool) ?
        eltype(op.val) : Float32,
)

##############################################################################
#                                 Methods                                    #
##############################################################################

onnx_name(v::Variable) = isnothing(v._op.val) ? "" : "x$(v.id)"
onnx_name(op::Umlaut.AbstractOp) = "x$(op.id)"


"""
    save_node!(g::GraphProto, op::Umlaut.Call)
    save_node!(g::GraphProto, ::OpConfig{:Backend, Fn}, op::Umlaut.Call)

Serialize a single operation from a tape to graph.
"""
function save_node!(g::GraphProto, op::Umlaut.Call)
    # @info "Saving $(op)" fn = op.fn name = nameof(op.fn) m = methods(op.fn)
    save_node!(g, OpConfig{:ONNX, typeof(op.fn)}(), op)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(getfield)}, op::Umlaut.Call)
    # Do nothing: getfield is only used to destructure multi-ouput nodes
    # and doesn't need to be written to ONNX graph.
    # Using getfield() for anything other then destructuring is thus a mistake.
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(*)}, op::Umlaut.Call)
    nd = NodeProto(
        input=[onnx_name(v) for v in reverse(op.args)],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[],
        op_type="Gemm"
    )
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_gemm), op::Umlaut.Call)
    kw_dict = kwargs2dict(op)
    attrs = rename_keys(kw_dict, Dict(
        :tA => :transA,
        :tB => :transB,
        :α => :alpha,
        :β => :beta
    ))
    nd = NodeProto("Gemm", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, conv), op::Umlaut.Call)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    w = args[2]._op.val
    # ONNXRuntime gives the following error for Float64:
    # NOT_IMPLEMENTED : Could not find an implementation for the node x3:Conv(11)')
    eltype(w) == Float64 && @warn "Not all ONNX runtimes support input & weights as Float64"
    attrs = from_nnlib_conv(kwargs2dict(op), ndims(w) - 2)
    nd = NodeProto("Conv", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, maxpool), op::Umlaut.Call)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    x = args[1]._op.val
    attrs = from_nnlib_conv(kwargs2dict(op), ndims(x) - 2)
    nd = NodeProto("MaxPool", op, attrs)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(global_average_pool)}, op::Umlaut.Call)
    nd = NodeProto("GlobalAveragePool", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_flatten), op::Umlaut.Call)
    nd = NodeProto("Flatten", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(add)}, op::Umlaut.Call)
    nd = NodeProto("Add", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(sub)}, op::Umlaut.Call)
    nd = NodeProto("Sub", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(mul)}, op::Umlaut.Call)
    nd = NodeProto("Mul", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(div)}, op::Umlaut.Call)
    nd = NodeProto("Div", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_pow)}, op::Umlaut.Call)
    nd = NodeProto("Pow", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_eq)}, op::Umlaut.Call)
    nd = NodeProto("Equal", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_where)}, op::Umlaut.Call)
    nd = NodeProto("Where", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_expand)}, op::Umlaut.Call)
    nd = NodeProto("Expand", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(relu)}, op::Umlaut.Call)
    nd = NodeProto("Relu", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(elu)}, op::Umlaut.Call)
    nd = NodeProto("Elu", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_sqrt)}, op::Umlaut.Call)
    nd = NodeProto("Sqrt", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(tanh)}, op::Umlaut.Call)
    nd = NodeProto("Tanh", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_cos)}, op::Umlaut.Call)
    nd = NodeProto("Cos", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_sin)}, op::Umlaut.Call)
    nd = NodeProto("Sin", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(erf)}, op::Umlaut.Call)
    nd = NodeProto("Erf", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(NNlib.batched_mul)}, op::Umlaut.Call)
    nd = NodeProto(
        input=[onnx_name(v) for v in reverse(op.args)],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[],
        op_type="MatMul"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(batched_mul4)}, op::Umlaut.Call)
    nd = NodeProto(
        input=[onnx_name(v) for v in reverse(op.args)],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[],
        op_type="MatMul"
    )
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, batch_norm), op::Umlaut.Call)
    kw_dict = kwargs2dict(op)
    attrs = from_nnlib_norm(kw_dict)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    output = if Bool(get(attrs, :training_mode, 0))
        vars = unpacked_vars(op)
        @assert(all([v isa V for v in vars]),
            "Not all output vars of batch_norm are unpacked to the tape")
        [onnx_name(v) for v in vars]
    else
        [onnx_name(op)]
    end
    nd = NodeProto(
        input=[onnx_name(v) for v in args],
        output=output,
        name=onnx_name(op),
        attribute=AttributeProto[AttributeProto(k, v) for (k, v) in attrs],
        op_type="BatchNormalization"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, instance_normalize), op::Umlaut.Call)
    kw_dict = kwargs2dict(op)
    attrs = from_nnlib_norm(kw_dict)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    output = if Bool(get(attrs, :training_mode, 0))
        vars = unpacked_vars(op)
        @assert(all([v isa V for v in vars]),
            "Not all output vars of batch_norm are unpacked to the tape")
        [onnx_name(v) for v in vars]
    else
        [onnx_name(op)]
    end
    nd = NodeProto(
        input=[onnx_name(v) for v in args],
        output=output,
        name=onnx_name(op),
        attribute=AttributeProto[AttributeProto(k, v) for (k, v) in attrs],
        op_type="InstanceNormalization"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_reduce_mean), op::Umlaut.Call)
    attrs = kwargs2dict(op)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    nd = NodeProto(
        input=[onnx_name(v) for v in args],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[AttributeProto(k, v) for (k, v) in attrs],
        op_type="ReduceMean"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(size)}, op::Umlaut.Call)
    nd = NodeProto("Shape", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, <:Any}, op::Umlaut.Constant)
    @assert(
        op.val isa AbstractArray,
        "ONNX.jl currently doesn't support saving constants of type $(typeof(op.val))"
    )
    attr_name = :value
    attr_value = from_nnlib(op.val)
    nd = NodeProto(
        input=[],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.([attr_name], [attr_value]),
        op_type="Constant"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_fill)}, op::Umlaut.Call)
    attr_name = :value
    attr_value = TensorProto([op.args[1]], onnx_name(op) * "_value")
    nd = NodeProto(
        input=[onnx_name(op.args[2])],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.([attr_name], [attr_value]),
        op_type="ConstantOfShape",
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_gather), op::Umlaut.Call)
    data = iskwfunc(op.fn) ? op.args[3]._op.val : op.args[1]._op.val
    kw_dict = kwargs2dict(op)
    dim = get(kw_dict, :dim, ndims(data))
    axis = ndims(data) - dim
    nd = NodeProto("Gather", op, Dict(:axis => axis))
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_unsqueeze), op::Umlaut.Call)
    nd = NodeProto("Unsqueeze", op)
    push!(g.node, nd)
end


function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(onnx_slice)}, op::Umlaut.Call)
    nd = NodeProto("Slice", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_split), op::Umlaut.Call)
    attrs = kwargs2dict(op)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    vars = unpacked_vars(op)
    @assert(all([v isa V for v in vars]),
        "Not all output vars of split are unpacked to the tape")
    output = [onnx_name(v) for v in vars]
    nd = NodeProto(
        input=[onnx_name(v) for v in args],
        output=output,
        name=onnx_name(op),
        attribute=AttributeProto.(keys(attrs), values(attrs)),
        op_type="Split"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_concat), op::Umlaut.Call)
    nd = NodeProto("Concat", op, kwargs2dict(op))
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(identity)}, op::Umlaut.Call)
    nd = NodeProto("Identity", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_reshape), op::Umlaut.Call)
    nd = NodeProto("Reshape", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_transpose), op::Umlaut.Call)
    nd = NodeProto(
        input=[onnx_name(op.args[1])],
        output = [onnx_name(op)],
        attribute=[AttributeProto(:perm, op.args[2])],
        op_type="Transpose",
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(size_vector)}, op::Umlaut.Call)
    INLINE_SHAPES = true
    if INLINE_SHAPES
        push!(g.initializer, TensorProto(op.val, onnx_name(op)))
    else
        nd = NodeProto("Shape", op)
        push!(g.node, nd)
    end
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(sigmoid)}, op::Umlaut.Call)
    nd = NodeProto("Sigmoid", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_softmax), op::Umlaut.Call)
    nd = NodeProto("Softmax", op)
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_cast), op::Umlaut.Call)
    attrs = kwargs2dict(op)
    nd = NodeProto(
        input=[onnx_name(op.args[3])],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.(keys(attrs), values(attrs)),
        op_type="Cast"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::OpConfig{:ONNX, typeof(permutedims)}, op::Umlaut.Call)
    @assert length(op.args) == 2
    x, size = op.args
    nd = NodeProto(
        input=[onnx_name(x)],
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto[AttributeProto("perm", reverse(size .- 1))],
        op_type="Transpose"
    )
    push!(g.node, nd)
end

function save_node!(g::GraphProto, ::@opconfig_kw(:ONNX, onnx_resize), op::Umlaut.Call)
    attrs = kwargs2dict(op)
    args = iskwfunc(op.fn) ? op.args[3:end] : op.args
    input = [onnx_name(v) for v in args]
    nd = NodeProto(;
        input,
        output=[onnx_name(op)],
        name=onnx_name(op),
        attribute=AttributeProto.(keys(attrs), values(attrs)),
        op_type="Resize"
    )
    push!(g.node, nd)
end

##############################################################################
#                                    API                                     #
##############################################################################

"""
    save(io::IO, tape::Umlaut.Tape{ONNXCtx})
    save(filename::String, tape::Umlaut.Tape{ONNXCtx})

Save tape as an ONNX model. The way a particular operation is serialized is
controlled by methods of [save_node!](@ref).

See also: [`load!`](@ref)
"""
function save(io::IO, tape::Tape{ONNXCtx}; external=false)
    g = graphproto("generated_model")


    if external
        file = open("model.data", "w")
    end

    offset = 0

    for (i, op) in enumerate(tape)
        if op isa Umlaut.Input
            # add input to g.input, but not to g.initializer
            push!(g.input, ValueInfoProto(op))
        elseif op isa Umlaut.Constant
            # add constant to g.initializer, but not to g.input
            # some models out there also put constants & parameters
            # to g.init, but it seems to be an outdated practise
            if !isnothing(op.val)
                if external && prod(size(op.val)) > 64
                    bytes_length = write(file, reshape(op.val, :))
                    push!(g.initializer,
                          TensorProto(op.val, onnx_name(op),
                                      "model.data", offset, bytes_length))
                    offset += bytes_length
                else
                    push!(g.initializer, TensorProto(op.val, onnx_name(op)))
                end
            end
        elseif op isa Umlaut.Call
            save_node!(g, op)
        else
            error("$(typeof(op)) is not yet supported in model export")
        end
    end
    res = tape[tape.result]
    if res.val isa Tuple
        # if the last operation in the graph is multi-output, there must be
        # unpacked elements of that var
        vars = unpacked_vars(res)
        @assert(all(v isa V for v in vars), "Cannot save the tape because the result " *
            "is multi-output, but its elements aren't destructured to the tape")
        for v in vars
            push!(g.output, ValueInfoProto(tape[v]))
        end
    else
        push!(g.output, ValueInfoProto(tape[tape.result]))
    end
    m = modelproto(g);
    PB.encode(ProtoEncoder(io), m)
end


function save(filename::String, tape::Tape{ONNXCtx}; external=false)
    open(filename, "w") do io
        save(io, tape; external)
    end
end
