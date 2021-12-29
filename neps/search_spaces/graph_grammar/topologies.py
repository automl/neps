import inspect
from abc import ABCMeta

from .graph import Graph


class AbstractTopology(Graph, metaclass=ABCMeta):
    edge_list: list = []

    def __init__(self, name: str = None, scope: str = None):
        super().__init__(name=name, scope=scope)

    def mutate(self):
        pass

    def sample(self):
        pass

    def create_graph(self, vals: dict):
        def get_args_and_defaults(func):
            signature = inspect.signature(func)
            return list(signature.parameters.keys()), {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }

        def get_op_name_from_dict(val: dict):
            # currently assumes that missing args are ints!
            op = val["op"]
            args: dict = {}
            arg_names, default_args = get_args_and_defaults(op)
            for arg_name in arg_names:
                if arg_name == "self" or arg_name == "kwargs" or arg_name in args.keys():
                    continue
                if arg_name in val.keys():
                    args[arg_name] = val[arg_name]
                elif arg_name in default_args.keys():
                    args[arg_name] = default_args[arg_name]
                else:
                    args[arg_name] = 42

            if "groups" in args and args["groups"] != 1:
                args["C_in"] = args["groups"]
                args["C_out"] = args["groups"]

            return op(**args).get_op_name

        assert isinstance(vals, dict)
        for (u, v), val in vals.items():
            self.add_edge(u, v)
            if isinstance(val, dict):
                _val = val
                _val["op_name"] = get_op_name_from_dict(val)
            else:
                if isinstance(val, int):  # for synthetic benchmarks
                    _val = {"op": val, "op_name": val}
                else:
                    _val = {"op": val, "op_name": val.get_op_name}
            self.edges[u, v].update(_val)

    @property
    def get_op_name(self):
        return type(self).__name__


class Linear(AbstractTopology):
    edge_list = [
        (1, 2),
        (2, 3),
    ]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "linear"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Residual(AbstractTopology):
    edge_list = [
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "residual"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Diamond(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "diamond"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class DiamondMid(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "diamond_mid"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class DenseNNodeDAG(AbstractTopology):
    edge_list = []

    def __init__(self, *edge_vals, number_of_nodes: int):
        super().__init__()

        self.edge_list = [
            (i + 1, j + 1) for j in range(number_of_nodes) for i in range(j)
        ]

        self.name = f"dense_{number_of_nodes}_node_dag"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class DownsampleBlock(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3)]

    def __init__(self, *edge_vals) -> None:
        super().__init__()
        self.name = f"{self.__class__.__name__}"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)
        self.graph_type = "edge_attr"
