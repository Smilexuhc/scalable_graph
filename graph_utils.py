import dgl


def build_graph(data, node_attr=None, edge_attr=None):
    """
    Initialize DGLGraph.
    Args:
        data: graph data to initialize graph
              (1) list of edge pairs (e.g. [(0, 2), (3, 1), ...])
              (2) pair of vertex IDs representing end nodes (e.g. ([0, 3, ...],  [2, 1, ...]))
              (3) scipy sparse matrix
              (4) networkx graph
        node_attr: dict, {node_feat_name:tensor}
        edge_attr: dict, {edge_feat_name:tensor}

    Returns:
        DGLGraph
    """
    graph = dgl.DGLGraph(graph_data=data)

    if node_attr is not None:
        for key, value in node_attr:
            graph.ndata[key] = value

    if edge_attr is not None:
        for key, value in edge_attr:
            graph.edata[key] = value

    return graph





