from comprehensive_nas.search_spaces.graph_grammar.topologies import AbstractTopology


class DAG2Nodes(AbstractTopology):
    edge_list = [[(1, 2)]]

    def __init__(self, *edge_vals, variant: int):
        super().__init__()
        assert variant == 1

        self.name = f"dag1_{variant}"
        edge_dict = {
            self.edge_list[0][0]: edge_vals[0],  # 1
        }
        self.create_graph(edge_dict)

        # Assign dummy variables as node attributes:
        for i in self.nodes:
            self.nodes[i]["op_name"] = "1"
        self.graph_type = "edge_attr"
        self.set_scope(self.name, recursively=False)


class DAG3Nodes(AbstractTopology):
    edge_list = [
        [
            (1, 2),
            (2, 3),
        ],  # 1
        [
            (1, 2),
            (2, 3),
            (1, 3),
        ],  # 2
    ]

    def __init__(self, *edge_vals, variant: int):
        super().__init__()
        assert 1 <= variant <= 2

        self.name = f"dag3_{variant}"
        edge_dict = dict(zip(self.edge_list[variant - 1], edge_vals))
        self.create_graph(edge_dict)

        # Assign dummy variables as node attributes:
        for i in self.nodes:
            self.nodes[i]["op_name"] = "1"
        self.graph_type = "edge_attr"
        self.set_scope(self.name, recursively=False)


class DAG4Nodes(AbstractTopology):
    edge_list = [
        [
            (1, 2),
            (2, 3),
            (3, 4),
        ],  # 1
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
        ],  # 2
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 4),
        ],  # 3
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (2, 4),
        ],  # 4
        [
            (1, 2),
            (3, 4),
            (1, 3),
            (2, 4),
        ],  # 5
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (1, 4),
        ],  # 6
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
        ],  # 7
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 4),
            (2, 4),
        ],  # 8
        [
            (1, 2),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ],  # 9
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (2, 4),
            (3, 4),
        ],  # 10
    ]

    def __init__(self, *edge_vals, variant: int):
        super().__init__()
        assert 1 <= variant <= 10

        self.name = f"dag4_{variant}"
        edge_dict = dict(zip(self.edge_list[variant - 1], edge_vals))
        self.create_graph(edge_dict)

        # Assign dummy variables as node attributes:
        for i in self.nodes:
            self.nodes[i]["op_name"] = "1"
        self.graph_type = "edge_attr"
        self.set_scope(self.name, recursively=False)
