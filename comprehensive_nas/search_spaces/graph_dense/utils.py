import networkx as nx

MAX_EDGES_201 = None
VERTICES_201 = None
OPS_201 = ["nor_conv_3x3", "nor_conv_1x1", "avg_pool_3x3", "skip_connect", "none"]


def create_nasbench201_graph(op_node_labelling, edge_attr=True):
    assert len(op_node_labelling) == 6
    # the graph has 8 nodes (6 operation nodes + input + output)
    G = nx.DiGraph()
    if edge_attr:
        edge_list = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        G.add_edges_from(edge_list)
        edge_attribute = {}
        remove_edge_list = []
        for i, edge in enumerate(edge_list):
            edge_attribute[edge] = {"op_name": op_node_labelling[i]}
            if op_node_labelling[i] == "none":
                remove_edge_list.append(edge)
        nx.set_edge_attributes(G, edge_attribute)
        G.remove_edges_from(remove_edge_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 3 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        # Assign dummy variables as node attributes:
        for i in G.nodes:
            G.nodes[i]["op_name"] = "1"
        G.graph_type = "edge_attr"
    else:
        edge_list = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 6),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
        G.add_edges_from(edge_list)

        # assign node attributes and collate the information for nodes to be removed
        # (i.e. nodes with 'skip_connect' or 'none' label)
        node_labelling = ["input"] + op_node_labelling + ["output"]
        nodes_to_remove_list = []
        remove_nodes_list = []
        edges_to_add_list = []
        for i, n in enumerate(node_labelling):
            G.nodes[i]["op_name"] = n
            if n == "none" or n == "skip_connect":
                input_nodes = [edge[0] for edge in G.in_edges(i)]
                output_nodes = [edge[1] for edge in G.out_edges(i)]
                nodes_to_remove_info = {
                    "id": i,
                    "input_nodes": input_nodes,
                    "output_nodes": output_nodes,
                }
                nodes_to_remove_list.append(nodes_to_remove_info)
                remove_nodes_list.append(i)

                if n == "skip_connect":
                    for n_i in input_nodes:
                        edges_to_add = [(n_i, n_o) for n_o in output_nodes]
                        edges_to_add_list += edges_to_add

        # reconnect edges for removed nodes with 'skip_connect'
        G.add_edges_from(edges_to_add_list)

        # remove nodes with 'skip_connect' or 'none' label
        G.remove_nodes_from(remove_nodes_list)

        # after removal, some op nodes have no input nodes and some have no output nodes
        # --> remove these redundant nodes
        nodes_to_be_further_removed = []
        for n_id in G.nodes():
            in_edges = G.in_edges(n_id)
            out_edges = G.out_edges(n_id)
            if n_id != 0 and len(in_edges) == 0:
                nodes_to_be_further_removed.append(n_id)
            elif n_id != 7 and len(out_edges) == 0:
                nodes_to_be_further_removed.append(n_id)

        G.remove_nodes_from(nodes_to_be_further_removed)
        G.graph_type = "node_attr"

    # create the arch string for querying nasbench dataset
    arch_query_string = (
        f"|{op_node_labelling[0]}~0|+"
        f"|{op_node_labelling[1]}~0|{op_node_labelling[2]}~1|+"
        f"|{op_node_labelling[3]}~0|{op_node_labelling[4]}~1|{op_node_labelling[5]}~2|"
    )

    G.name = arch_query_string
    return G


MAX_EDGES_301 = 13
VERTICES_301 = 6
HPS_301 = 2
OPS_301 = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

edge_to_coord_mapping = {
    0: (0, 2),
    1: (1, 2),
    2: (0, 3),
    3: (1, 3),
    4: (2, 3),
    5: (0, 4),
    6: (1, 4),
    7: (2, 4),
    8: (3, 4),
    9: (0, 5),
    10: (1, 5),
    11: (2, 5),
    12: (3, 5),
    13: (4, 5),
}
coord_to_edge_mapping = {
    (0, 2): 0,
    (1, 2): 1,
    (0, 3): 2,
    (1, 3): 3,
    (2, 3): 4,
    (0, 4): 5,
    (1, 4): 6,
    (2, 4): 7,
    (3, 4): 8,
    (0, 5): 9,
    (1, 5): 10,
    (2, 5): 11,
    (3, 5): 12,
    (4, 5): 13,
}


def create_nasbench301_graph(edge_attr=True):
    adjacency_matrix = None

    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    nx.set_edge_attributes(G, edge_attr)
    for i in G.nodes:
        G.nodes[i]["op_name"] = "1"
    G.graph_type = "edge_attr"
    return G
