


function remove_unused!(graph::GraphProto)
    valids = Set((output.name for output in graph.output))
    valid_nodes = Set()

    for node in Iterators.reverse(graph.node)
        if any(∈(valids), node.output)
            union!(valids, node.input)
            push!(valid_nodes, node.name)
        end
    end

    nodes_before = length(graph.node)
    filter!(
        node -> node.name ∈ valid_nodes,
        graph.node,
    )

    initializer_before = length(graph.initializer)
    filter!(
        initializer -> any(node -> any(input -> input == initializer.name, node.input),
                           graph.node),
        graph.initializer,
    )

    @info "Removed $(nodes_before - length(graph.node)) nodes"
    @info "Removed $(initializer_before - length(graph.initializer)) initializers"

    #graph
    nothing
end
